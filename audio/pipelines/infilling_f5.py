import gc
import os
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
from ..tts.f5 import text_to_speech as f5_tts
from .._paths import CENSORED_WAV, KNN_REF_DIR, SYNTH_TMP_DIR


CROSSFADE_MS = 40
_BEEP_SENTINEL = "XXXXXX"
_CENSORED_WAV = CENSORED_WAV


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

def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    g = gcd(orig_sr, target_sr)
    return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)


def _match_amplitude(src: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    rms_ref = np.sqrt(np.mean(ref ** 2))
    rms_src = np.sqrt(np.mean(src ** 2))
    if rms_src < eps:
        return src
    return src * (rms_ref / (rms_src + eps))


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


def _adjust_sample(orig: int, deltas: list[tuple[int, int]]) -> int:
    return orig + sum(d for thresh, d in deltas if orig >= thresh)


def _crossfade(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    n = min(n, len(a), len(b))
    if n == 0:
        return np.concatenate([a, b])
    t = np.linspace(0.0, np.pi / 2, n, dtype=np.float32)
    fade_out = np.cos(t)
    fade_in = np.sin(t)
    blended = a[-n:] * fade_out + b[:n] * fade_in
    return np.concatenate([a[:-n], blended, b[n:]])


def _trim_silence(audio: np.ndarray, sr: int, threshold_db: float = -42.0, frame_ms: int = 20, pad_ms: int = 80) -> np.ndarray:
    """Remove leading and trailing silence below threshold_db using 20ms frames, keeping pad_ms of silence at each end."""
    frame = max(1, int(sr * frame_ms / 1000))
    pad_frames = int(pad_ms / frame_ms)
    n = len(audio) // frame
    if n == 0:
        return audio
    energies = np.array([np.mean(audio[i * frame:(i + 1) * frame] ** 2) for i in range(n)])
    power_thresh = 10 ** (threshold_db / 10)
    active = np.where(energies > power_thresh)[0]
    if len(active) == 0:
        return audio
    start = max(0, active[0] - pad_frames) * frame
    end = min(len(audio), (active[-1] + 1 + pad_frames) * frame)
    return audio[start:end]


def _get_local_ref(
    audio: np.ndarray,
    sr: int,
    words: list[TimedWord],
    transcript: str,
    t_start: float,
    t_end: float,
    max_ref_s: float = 12.0,
    min_ref_s: float = 5.0,
) -> tuple[np.ndarray, str]:
    """Return (ref_audio, ref_text) using context before [t_start, t_end], excluding it.
    Falls back to after-context only when before is too short (< min_ref_s)."""
    audio_duration = len(audio) / sr

    before_s = min(max_ref_s, t_start)
    after_s = min(max_ref_s, max(0.0, audio_duration - t_end))

    ref_t_start = t_start - before_s
    ref_t_end = t_end + after_s

    before_audio = audio[int(ref_t_start * sr):int(t_start * sr)]
    after_audio = audio[int(t_end * sr):int(ref_t_end * sr)]

    before_words = [w.word for w in words if w.start >= ref_t_start and w.end <= t_start + 0.01]
    after_words = [w.word for w in words if w.start >= t_end - 0.01 and w.end <= ref_t_end]

    if before_s >= min_ref_s and before_words:
        return before_audio, " ".join(before_words).strip()
    if after_s >= min_ref_s and after_words:
        return after_audio, " ".join(after_words).strip()

    return audio, transcript


_SYNTH_TMP_DIR = str(SYNTH_TMP_DIR)
os.makedirs(_SYNTH_TMP_DIR, exist_ok=True)


def _synthesize(text: str, ref_audio: str, ref_text: str, voice: str = "original") -> tuple[np.ndarray, int]:
    """Synthesize text with F5-TTS using ref_audio/ref_text for voice cloning."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=_SYNTH_TMP_DIR) as f:
        tmp = f.name
    try:
        f5_tts(text, output_path=tmp, ref_audio=ref_audio, ref_text=ref_text, voice=voice)
        if not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
            raise RuntimeError(f"F5-TTS produced no output for: {text!r:.80}")
        audio, sr = sf.read(tmp, dtype="float32")
    finally:
        os.unlink(tmp)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio, sr


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



# ── Main pipeline ─────────────────────────────────────────────────────────────

def anonymize_audio(
    audio_path: str,
    output_path: str,
    config: AnonymizationConfig | None = None,
    device: str = "cuda",
    ref_wav_paths: list[str] | None = None,
    knn_topk: int = 4,
    voice: str = "original",
) -> tuple[str, str, str]:
    """Returns (transcript, anonymized_text, output_path)."""
    if config is None:
        config = AnonymizationConfig(regex="gen", per="gen", loc="gen", org="gen", misc="gen")

    working_audio = audio_path

    print("Step 1: Transcribing with word timestamps")
    words = get_timed_transcript(working_audio, device=device)
    transcript = " ".join(w.word for w in words)
    print(f"  {transcript[:120]}{'…' if len(transcript) > 120 else ''}")

    print("Step 2: Identifying sensitive spans")
    _, spans = pipeline_with_spans(transcript, config=config)
    for s in spans:
        print(f"  [{s.start}:{s.end}] '{s.original}' -> '{s.replacement}'")

    entities = [{"start": s.start, "end": s.end, "original": s.original, "replacement": s.replacement} for s in spans]
    anonymized_text, _, _segment_map = synthesize_anonymized(
        working_audio, output_path, words, entities,
        ref_wav_paths=ref_wav_paths, device=device, knn_topk=knn_topk,
        voice=voice,
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
    voice: str = "original",
) -> tuple[str, str, list[dict]]:
    """
    Synthesize audio given pre-computed words and user-reviewed entity dicts.
    Returns (anonymized_text, output_path, segment_map) where segment_map is a list of
    {orig_start, orig_end, out_start, out_end} dicts (seconds) for each replaced segment.
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

    working_audio = audio_path

    intervals = char_spans_to_time(spans, word_map)

    print(f"Mapping {len(spans)} span(s) to {len(intervals)} audio interval(s)")
    for iv in intervals:
        print(f"  {iv.t_start:.2f}s–{iv.t_end:.2f}s → '{iv.replacement_text}'")

    audio, sr = sf.read(working_audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]

    original_audio = audio.copy()

    crossfade_n = int(sr * CROSSFADE_MS / 1000)
    orig_sr = sr
    suppressed_intervals = [iv for iv in intervals if iv.suppressed]
    tts_deltas: list[tuple[int, int]] = []
    splice_info: list[tuple[int, int, int]] = []  # (orig_s_start, orig_s_end, tts_samples)

    for iv in reversed(intervals):
        if iv.suppressed:
            continue
        s_start = max(0, min(int(iv.t_start * sr), len(audio)))
        s_end = max(s_start, min(int(iv.t_end * sr), len(audio)))
        before = audio[:s_start]
        after = audio[s_end:]

        ref_audio_arr, ref_text = _get_local_ref(
            original_audio, sr, words, transcript, iv.t_start, iv.t_end
        )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=_SYNTH_TMP_DIR) as f:
            ref_tmp = f.name
        try:
            sf.write(ref_tmp, ref_audio_arr, sr)
            tts_audio, tts_sr = _synthesize(iv.replacement_text, ref_tmp, ref_text, voice)
        finally:
            if os.path.exists(ref_tmp):
                os.unlink(ref_tmp)

        tts_audio = _resample(tts_audio, tts_sr, sr)
        tts_audio = _trim_silence(tts_audio, sr)
        tts_audio = _match_amplitude(tts_audio, audio[s_start:s_end])
        splice_info.append((s_start, s_end, len(tts_audio)))
        delta = len(tts_audio) - (s_end - s_start)
        if delta != 0:
            tts_deltas.append((s_end, delta))
        joined = _crossfade(before, tts_audio, crossfade_n)
        audio = _crossfade(joined, after, crossfade_n)

    sf.write(output_path, audio, sr)

    if ref_wav_paths:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Applying kNN-VC")
        vc_tmp = output_path + ".vc_tmp.wav"
        try:
            apply_knn_vc(output_path, vc_tmp, ref_wav_paths, device=device, topk=knn_topk)
            os.replace(vc_tmp, output_path)
        except Exception:
            if os.path.exists(vc_tmp):
                os.unlink(vc_tmp)
            raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        audio, sr = sf.read(output_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]

    if suppressed_intervals:
        beep_base, beep_sr = sf.read(_CENSORED_WAV, dtype="float32")
        if beep_base.ndim > 1:
            beep_base = beep_base[:, 0]
        beep_base = _resample(beep_base, beep_sr, sr)
        beep_fixed = _fit_beep(beep_base, len(beep_base), sr)
        for iv in reversed(suppressed_intervals):
            t_start_adj = _adjust_sample(int(iv.t_start * orig_sr), tts_deltas) / orig_sr
            t_end_adj = _adjust_sample(int(iv.t_end * orig_sr), tts_deltas) / orig_sr
            s_start = max(0, min(int(t_start_adj * sr), len(audio)))
            s_end = max(s_start, min(int(t_end_adj * sr), len(audio)))
            beep_seg = _match_amplitude(beep_fixed, audio[s_start:s_end]) if s_end > s_start else beep_fixed
            audio = np.concatenate([audio[:s_start], beep_seg, audio[s_end:]])
        sf.write(output_path, audio, sr)

    splice_info.sort(key=lambda x: x[0])
    cumulative_delta = 0
    segment_map: list[dict] = []
    for orig_start, orig_end, tts_len in splice_info:
        out_start = orig_start + cumulative_delta
        segment_map.append({
            "orig_start": orig_start / orig_sr,
            "orig_end":   orig_end   / orig_sr,
            "out_start":  out_start  / orig_sr,
            "out_end":    (out_start + tts_len) / orig_sr,
        })
        cumulative_delta += tts_len - (orig_end - orig_start)

    print(f"Saved to {output_path}")
    return anonymized_text, output_path, segment_map


# ── CLI entry point ───────────────────────────────────────────────────────────
# Run as: python -m audio.pipelines.infilling_f5 [input] [output] [ref_wavs...]

if __name__ == "__main__":
    import argparse

    _DEFAULT_REF_WAVS = [
        str(p)
        for p in sorted(KNN_REF_DIR.glob("in_aca*.wav"))[:3]
    ]

    parser = argparse.ArgumentParser(description="Infilling pipeline with F5-TTS voice cloning.")
    parser.add_argument("input", nargs="?", default="data/cati_exemplo.wav")
    parser.add_argument("output", nargs="?", default="data/output_pipeline_infilling_f5.wav")
    parser.add_argument("ref_wavs", nargs="*", help="Reference wavs for kNN-VC speaker conversion")
    parser.add_argument("--voice", default="original", help="F5-TTS voice name (default: original = clone from input)")
    args = parser.parse_args()
    ref_paths = args.ref_wavs if args.ref_wavs else None
    anonymize_audio(
        args.input, args.output,
        ref_wav_paths=ref_paths,
        voice=args.voice,
    )
