import gc
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from math import gcd

import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy.signal import resample_poly

from ..asr import TimedWord, get_timed_transcript
from text_anon import AnonSpan, AnonymizationConfig, pipeline_with_spans
from .._conda_env import get_conda_env_python
from .._paths import CENSORED_WAV, F5_SPEECH_EDIT_HELPER, KNN_REF_DIR


_CONDA_ENV = os.environ.get("F5_TTS_CONDA_ENV", "f5-tts")
_BEEP_SENTINEL = "XXXXXX"
_CENSORED_WAV = CENSORED_WAV
_HELPER = str(F5_SPEECH_EDIT_HELPER)

_CONTEXT_BEFORE = 10.0  # must match CONTEXT_BEFORE in f5_speech_edit_helper.py
_CONTEXT_AFTER  = 5.0   # must match CONTEXT_AFTER  in f5_speech_edit_helper.py


@dataclass
class AudioInterval:
    t_start: float
    t_end: float
    replacement_text: str
    suppressed: bool = False


# ── Text to segment mapping ──────────────────────────────────────────────────

def _build_word_char_map(words: list[TimedWord]) -> list[tuple[int, int, TimedWord]]:
    entries: list[tuple[int, int, TimedWord]] = []
    pos = 0
    for i, w in enumerate(words):
        end = pos + len(w.word)
        entries.append((pos, end, w))
        pos = end + (1 if i < len(words) - 1 else 0)
    return entries


def char_spans_to_time(
    spans: list[AnonSpan],
    word_map: list[tuple[int, int, TimedWord]],
) -> list[AudioInterval]:
    intervals: list[AudioInterval] = []
    for span in spans:
        overlapping = [
            w for cs, ce, w in word_map
            if cs < span.end and ce > span.start
        ]
        if not overlapping:
            continue
        prev_end = intervals[-1].t_end if intervals else 0.0
        t_start = max(prev_end, overlapping[0].start)
        intervals.append(AudioInterval(
            t_start=t_start,
            t_end=overlapping[-1].end,
            replacement_text=span.replacement,
            suppressed=span.replacement == _BEEP_SENTINEL,
        ))
    return intervals


# ── Audio helpers ────────────────────────────────────────────────────────────

def _resample_np(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    g = gcd(orig_sr, target_sr)
    return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)


def _fit_beep(beep: np.ndarray, target_len: int, sr: int) -> np.ndarray:
    if target_len <= 0:
        return np.empty(0, dtype=np.float32)
    reps = (target_len // len(beep)) + 1
    fitted = np.tile(beep, reps)[:target_len].copy()
    fade_n = min(int(sr * 0.005), target_len // 2)
    if fade_n > 0:
        fitted[:fade_n] *= np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
        fitted[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
    return fitted


def _match_amplitude(src: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    rms_ref = np.sqrt(np.mean(ref ** 2))
    rms_src = np.sqrt(np.mean(src ** 2))
    if rms_src < eps:
        return src
    return src * (rms_ref / (rms_src + eps))


# ── Voice conversion ─────────────────────────────────────────────────────────

def _load_mono_16k(path: str, target_sr: int = 16000) -> torch.Tensor:
    x, sr = torchaudio.load(path, normalize=True)
    if x.shape[0] > 1:
        x = x.mean(dim=0, keepdim=True)
    if sr != target_sr:
        x = torchaudio.functional.resample(x, orig_freq=sr, new_freq=target_sr)
    return x


def apply_knn_vc(
    src_path: str,
    out_path: str,
    ref_wav_paths: list[str],
    device: str = "cuda",
    topk: int = 4,
) -> None:
    """Apply kNN-VC voice conversion to the full audio, changing speaker identity."""
    knn_vc = torch.hub.load(
        "bshall/knn-vc", "knn_vc",
        prematched=True, trust_repo=True, pretrained=True, device=device,
    )
    try:
        src_tensor = _load_mono_16k(src_path)
        query_seq = knn_vc.get_features(src_tensor)
        ref_tensors = [_load_mono_16k(p) for p in ref_wav_paths]
        matching_set = knn_vc.get_matching_set(ref_tensors)
        out_wav = knn_vc.match(query_seq, matching_set, topk=topk)
        torchaudio.save(out_path, out_wav.unsqueeze(0), 16000)
    finally:
        del knn_vc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── F5-TTS speech editing ────────────────────────────────────────────────────

def _window_text(
    words: list[TimedWord],
    interval: AudioInterval,
    prior_intervals: list[AudioInterval] | None = None,
) -> str:
    """Text covering the context window around a single edit interval.

    prior_intervals: already-applied edits whose replacement text should appear
    in the before-context instead of the original words, matching the mutated audio.
    """
    win_start = max(0.0, interval.t_start - _CONTEXT_BEFORE)
    win_end   = interval.t_end + _CONTEXT_AFTER
    before_words = [w for w in words if w.start >= win_start and w.end <= interval.t_start + 0.01]

    if prior_intervals:
        before: list[str] = []
        w_idx = 0
        while w_idx < len(before_words):
            w = before_words[w_idx]
            matched = next(
                (p for p in prior_intervals if w.start < p.t_end and w.end > p.t_start),
                None,
            )
            if matched:
                before.append(matched.replacement_text)
                while w_idx < len(before_words) and before_words[w_idx].start < matched.t_end:
                    w_idx += 1
            else:
                before.append(w.word)
                w_idx += 1
    else:
        before = [w.word for w in before_words]

    after = [w.word for w in words if w.start >= interval.t_end - 0.01 and w.end <= win_end]
    parts = (
        ([" ".join(before)] if before else [])
        + [interval.replacement_text]
        + ([" ".join(after)] if after else [])
    )
    return " ".join(parts)


def _estimate_durations(
    audio_path: str,
    transcript: str,
    intervals: list[AudioInterval],
) -> list[float]:
    """
    Estimate replacement duration per interval using the speaker's own rate.
    Uses iso-8859-1 for Portuguese characters to count as one character
    """
    info = sf.info(audio_path)
    secs_per_char = info.duration / max(1, len(transcript.encode("iso-8859-1", "replace")))
    return [
        max(0.3, len(iv.replacement_text.encode("iso-8859-1", "replace")) * secs_per_char)
        for iv in intervals
    ]


def _speech_edit_f5(
    audio_path: str,
    words: list[TimedWord],
    transcript: str,
    intervals: list[AudioInterval],
    output_path: str,
) -> list[float]:
    """Call f5_speech_edit_helper.py via conda subprocess. Returns actual decoded durations."""
    parts_to_edit = [[iv.t_start, iv.t_end] for iv in intervals]
    fix_duration  = _estimate_durations(audio_path, transcript, intervals)
    target_texts  = [_window_text(words, iv, prior_intervals=intervals[:i]) for i, iv in enumerate(intervals)]
    print(fix_duration)

    abs_output = os.path.abspath(output_path)
    subprocess.run(
        [
            get_conda_env_python(_CONDA_ENV, os.environ.get("F5_TTS_PYTHON")),
            _HELPER,
            "--audio",         os.path.abspath(audio_path),
            "--target_texts",  json.dumps(target_texts),
            "--parts_to_edit", json.dumps(parts_to_edit),
            "--fix_duration",  json.dumps(fix_duration),
            "--output",        abs_output,
        ],
        check=True,
    )

    dur_path = abs_output + ".dur.json"
    try:
        with open(dur_path) as _f:
            return json.load(_f)
    finally:
        if os.path.exists(dur_path):
            os.unlink(dur_path)


# ── Beep position adjustment and application ─────────────────────────────────

def _adjust_beep_times(
    suppressed: list[AudioInterval],
    tts_intervals: list[AudioInterval],
    fix_durations: list[float],
) -> list[AudioInterval]:
    """Shift beep interval times to account for variable-duration F5-TTS replacements."""
    adjusted = []
    for sup in suppressed:
        shift = sum(
            fix_dur - (iv.t_end - iv.t_start)
            for iv, fix_dur in zip(tts_intervals, fix_durations)
            if iv.t_end <= sup.t_start
        )
        adjusted.append(AudioInterval(
            t_start=sup.t_start + shift,
            t_end=sup.t_end + shift,
            replacement_text=sup.replacement_text,
            suppressed=True,
        ))
    return adjusted

def _apply_beeps(
    audio_path: str,
    intervals: list[AudioInterval],
    output_path: str,
) -> None:
    """Splice fixed-length beep tone at each suppressed interval."""
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]

    beep_base, beep_sr = sf.read(_CENSORED_WAV, dtype="float32")
    if beep_base.ndim > 1:
        beep_base = beep_base[:, 0]
    beep_base = _resample_np(beep_base, beep_sr, sr)
    beep_fixed = _fit_beep(beep_base, len(beep_base), sr)

    for iv in reversed(intervals):
        s_start = max(0, min(int(iv.t_start * sr), len(audio)))
        s_end = max(s_start, min(int(iv.t_end * sr), len(audio)))
        beep_seg = _match_amplitude(beep_fixed, audio[s_start:s_end]) if s_end > s_start else beep_fixed
        audio = np.concatenate([audio[:s_start], beep_seg, audio[s_end:]])

    sf.write(output_path, audio, sr)



# ── Core synthesis logic ─────────────────────────────────────────────────────

def _run_synthesis(
    audio_path: str,
    output_path: str,
    words: list[TimedWord],
    anonymized_text: str,
    transcript: str,
    intervals: list[AudioInterval],
) -> tuple[list[AudioInterval], list[dict]]:
    """Apply F5-TTS edits. Returns (adjusted_suppressed, segment_map)."""
    tts_intervals = [iv for iv in intervals if not iv.suppressed]
    suppressed_intervals = [iv for iv in intervals if iv.suppressed]

    if not tts_intervals and not suppressed_intervals:
        shutil.copy2(audio_path, output_path)
        return [], []

    if tts_intervals:
        print("  Running F5-TTS speech editing...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f5_tmp = f.name
        try:
            fix_durations = _speech_edit_f5(audio_path, words, transcript, tts_intervals, f5_tmp)
            shutil.move(f5_tmp, output_path)
            f5_tmp = ""
        finally:
            if f5_tmp and os.path.exists(f5_tmp):
                os.unlink(f5_tmp)

        cumulative_shift = 0.0
        segment_map: list[dict] = []
        for iv, fix_dur in zip(tts_intervals, fix_durations):
            out_start = iv.t_start + cumulative_shift
            segment_map.append({
                "orig_start": iv.t_start,
                "orig_end":   iv.t_end,
                "out_start":  out_start,
                "out_end":    out_start + fix_dur,
            })
            cumulative_shift += fix_dur - (iv.t_end - iv.t_start)

        return _adjust_beep_times(suppressed_intervals, tts_intervals, fix_durations), segment_map
    else:
        shutil.copy2(audio_path, output_path)
        return suppressed_intervals, []


# ── Main pipeline ─────────────────────────────────────────────────────────────

def anonymize_audio(
    audio_path: str,
    output_path: str,
    config: AnonymizationConfig | None = None,
    device: str = "cuda",
    ref_wav_paths: list[str] | None = None,
    knn_topk: int = 4,
) -> tuple[str, str, str]:
    """Returns (transcript, anonymized_text, output_path)."""
    if config is None:
        config = AnonymizationConfig(regex="gen", per="gen", loc="gen", org="gen", misc="gen")

    print("Step 1: Transcribing with word timestamps")
    words = get_timed_transcript(audio_path, device=device)
    transcript = " ".join(w.word for w in words)
    print(f"  {transcript[:120]}{'…' if len(transcript) > 120 else ''}")

    print("Step 2: Identifying sensitive spans")
    _, spans = pipeline_with_spans(transcript, config=config)
    for s in spans:
        print(f"  [{s.start}:{s.end}] '{s.original}' -> '{s.replacement}'")

    entities = [{"start": s.start, "end": s.end, "original": s.original, "replacement": s.replacement} for s in spans]
    anonymized_text, _, _segment_map = synthesize_anonymized(
        audio_path, output_path, words, entities,
        ref_wav_paths=ref_wav_paths, device=device, knn_topk=knn_topk,
    )

    return transcript, anonymized_text, output_path


# ── Synthesis from pre-computed entities (web review flow) ───────────────────

def synthesize_anonymized(
    audio_path: str,
    output_path: str,
    words: list[TimedWord],
    entities: list[dict],
    ref_wav_paths: list[str] | None = None,
    device: str = "cuda",
    knn_topk: int = 4,
) -> tuple[str, str, list[dict]]:
    """
    Synthesize audio given pre-computed words and user-reviewed entity dicts.
    Returns (anonymized_text, output_path, segment_map).
    segment_map: [{orig_start, orig_end, out_start, out_end}] in seconds for each replaced segment.
    If ref_wav_paths is provided, applies kNN-VC after F5-TTS to convert speaker identity.
    """
    from text_anon import apply_entities

    transcript = " ".join(w.word for w in words)
    anonymized_text = apply_entities(transcript, entities)

    spans = [
        AnonSpan(
            start=e["start"],
            end=e["end"],
            original=e.get("original", transcript[e["start"]:e["end"]]),
            replacement=e["replacement"],
        )
        for e in entities
        if e.get("replacement")
    ]

    word_map = _build_word_char_map(words)
    intervals = char_spans_to_time(spans, word_map)

    print(f"Mapping {len(spans)} span(s) to {len(intervals)} audio interval(s)")
    for iv in intervals:
        print(f"  {iv.t_start:.2f}s–{iv.t_end:.2f}s → '{iv.replacement_text}'")

    suppressed, segment_map = _run_synthesis(audio_path, output_path, words, anonymized_text, transcript, intervals)

    if ref_wav_paths:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Applying kNN-VC (speaker identity conversion)")
        vc_tmp = output_path + ".vc_tmp.wav"
        try:
            apply_knn_vc(output_path, vc_tmp, ref_wav_paths, device=device, topk=knn_topk)
            os.replace(vc_tmp, output_path)
        except Exception:
            if os.path.exists(vc_tmp):
                os.unlink(vc_tmp)
            raise

    if suppressed:
        _apply_beeps(output_path, suppressed, output_path)

    print(f"Saved to {output_path}")
    return anonymized_text, output_path, segment_map


# ── CLI entry point ───────────────────────────────────────────────────────────
# Run as: python -m audio.pipelines.infilling_f5_tts [input] [output] [ref_wavs...]

if __name__ == "__main__":
    import argparse

    _DEFAULT_REF_WAVS = [
        str(p)
        for p in sorted(KNN_REF_DIR.glob("in_aca*.wav"))[:3]
    ]

    parser = argparse.ArgumentParser(description="Infilling anonymization pipeline (F5-TTS).")
    parser.add_argument("input", nargs="?", default="data/cati_exemplo.wav")
    parser.add_argument("output", nargs="?", default="data/output_pipeline_infilling_f5.wav")
    parser.add_argument("ref_wavs", nargs="*", help="Reference wavs for kNN-VC speaker conversion")
    args = parser.parse_args()
    ref_paths = args.ref_wavs if args.ref_wavs else None
    anonymize_audio(args.input, args.output, ref_wav_paths=ref_paths)
