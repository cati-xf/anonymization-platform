import os
import sys
import tempfile
from dataclasses import dataclass
from math import gcd
import gc

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import torch
import torchaudio

from ..asr import TimedWord, get_timed_transcript
from text_anon import AnonSpan, pipeline_with_spans
from text_anon import AnonymizationConfig
from ..tts.google_tts import text_to_speech as google_tts
from .._paths import CENSORED_WAV, KNN_REF_DIR



CROSSFADE_MS = 30

BEEP_SENTINEL = "XXXXXX"

@dataclass
class AudioInterval:
    # start, end of segment and what to replace with
    t_start: float
    t_end:   float
    replacement_text: str
    suppressed: bool = False


# ── Text to segment mapping ───────────────────────────────────────────────────

def _build_word_char_map(words: list[TimedWord]) -> list[tuple[int, int, TimedWord]]:
    """ 
    for each timed word associate start and end char 
    words = [TimedWord("dia",   0.4, 0.7), TimedWord("Ana",   0.8, 1.1), TimedWord("Silva", 1.2, 1.7),]
    (cs=4,  ce=7,  TimedWord("dia",   0.4, 0.7)),  (cs=8,  ce=11, TimedWord("Ana",   0.8, 1.1)), (cs=12, ce=17, TimedWord("Silva", 1.2, 1.7))
    """
    entries: list[tuple[int, int, TimedWord]] = []
    pos = 0
    for i, w in enumerate(words):
        end = pos + len(w.word)
        entries.append((pos, end, w))
        pos = end + (1 if i < len(words) - 1 else 0)
    return entries


def char_spans_to_time(spans: list[AnonSpan], word_map: list[tuple[int, int, TimedWord]],) -> list[AudioInterval]:
    """ 
    Find every word that overlaps with each sensitive span, and returns its start and end timestamps 
    AnonSpan(start=8, end=17, original="Ana Silva", replacement="pessoa anónima")
    Dia: no overlap, Ana: yes, Silva: yes
    AudioInterval(t_start=0.8, t_end=1.7, replacement_text="pessoa anónima")
    """
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
            t_start          = t_start,
            t_end            = overlapping[-1].end,
            replacement_text = span.replacement,
            suppressed       = span.replacement == BEEP_SENTINEL,
        ))
    return intervals



# ── Audio helpers ─────────────────────────────────────────────────────────────

def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """ Converts to another sample rate to match before joining """
    if orig_sr == target_sr:
        return audio
    # simplify to smallest form to be cheaper
    g = gcd(orig_sr, target_sr)
    # uses polyphase filter to resample by up/down
    return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)


def _match_amplitude(src: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale src so its RMS matches ref's RMS."""
    rms_ref = np.sqrt(np.mean(ref ** 2))
    rms_src = np.sqrt(np.mean(src ** 2))
    if rms_src < eps:
        return src
    return src * (rms_ref / (rms_src + eps))


def _fit_beep(beep: np.ndarray, target_len: int, sr: int) -> np.ndarray:
    """Loop or trim beep to be same length as sensitive snippet with 5 ms fades."""
    if target_len <= 0:
        return np.empty(0, dtype=np.float32)
    reps = (target_len // len(beep)) + 1
    fitted = np.tile(beep, reps)[:target_len].copy()
    fade_n = min(int(sr * 0.005), target_len // 2)
    if fade_n > 0:
        fitted[:fade_n]  *= np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
        fitted[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
    return fitted


def _adjust_sample(orig: int, deltas: list[tuple[int, int]]) -> int:
    """Shift a sample index by accumulated TTS length deltas at earlier positions."""
    return orig + sum(d for thresh, d in deltas if orig >= thresh)


def _crossfade(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """ Blend the end of a into the start of b, overlap of n samples """
    n = min(n, len(a), len(b))
    if n == 0:
        return np.concatenate([a, b])
    fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
    fade_in  = np.linspace(0.0, 1.0, n, dtype=np.float32)
    blended  = a[-n:] * fade_out + b[:n] * fade_in
    return np.concatenate([a[:-n], blended, b[n:]])


def _synthesize(text: str) -> tuple[np.ndarray, int]:
    """Return (audio, sample_rate) for the synthesized text span"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        google_tts(text, output_path=tmp)
        audio, tts_sr = sf.read(tmp, dtype="float32")
    finally:
        os.unlink(tmp) #always delete temp
    return audio, tts_sr

_CENSORED_WAV = CENSORED_WAV


def _load_mono_16k(path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load a wav as a mono (1, T) tensor at target_sr, regardless of source format."""
    x, sr = torchaudio.load(path, normalize=True)
    if x.shape[0] > 1:          # stereo/multichannel → mono
        x = x.mean(dim=0, keepdim=True)
    if sr != target_sr:
        x = torchaudio.functional.resample(x, orig_freq=sr, new_freq=target_sr)
    return x


def apply_knn_vc(src_path: str, out_path: str, ref_wav_paths: list[str], device: str = "cuda", topk: int = 4,) -> None:
    knn_vc = torch.hub.load(
        "bshall/knn-vc", "knn_vc",
        prematched=True, trust_repo=True, pretrained=True, device=device,
    )
    try:
        src_tensor   = _load_mono_16k(src_path)
        query_seq    = knn_vc.get_features(src_tensor)
        ref_tensors  = [_load_mono_16k(p) for p in ref_wav_paths]
        matching_set = knn_vc.get_matching_set(ref_tensors)
        out_wav      = knn_vc.match(query_seq, matching_set, topk=topk)
        torchaudio.save(out_path, out_wav.unsqueeze(0), 16000)
    finally:
        del knn_vc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _apply_knn_vc_to_segment(
    audio: np.ndarray,
    src_sr: int,
    knn_vc,
    matching_set,
    target_sr: int,
    topk: int = 4,
) -> np.ndarray:
    """Apply pre-loaded kNN-VC to a single TTS segment array, return at target_sr."""
    audio_16k = _resample(audio, src_sr, 16000)
    tensor = torch.from_numpy(audio_16k).unsqueeze(0)
    query_seq = knn_vc.get_features(tensor)
    out_wav = knn_vc.match(query_seq, matching_set, topk=topk)
    return _resample(out_wav.squeeze(0).cpu().numpy(), 16000, target_sr)



# ── Main pipeline ─────────────────────────────────────────────────────────────

def anonymize_audio(
    audio_path: str,
    output_path: str,
    config: AnonymizationConfig | None = None,
    device: str = "cuda",
    ref_wav_paths: list[str] | None = None,
    knn_topk: int = 4,
    span_filter=None,
) -> tuple[str, str, str]:
    """ returns (transcript, anonymized_text, output_path) """
    if config is None:
        config = AnonymizationConfig(regex="gen", per="gen", loc="gen", org="gen", misc="gen")

    working_audio = audio_path

    print("Step 1: Transcribing with word timestamps")
    words = get_timed_transcript(working_audio, device=device)
    transcript = " ".join(w.word for w in words)
    print(f"  {transcript[:120]}{'…' if len(transcript) > 120 else ''}")

    print("Step 2: Identifying sensitive spans")
    _, spans = pipeline_with_spans(transcript, config=config)
    if span_filter is not None:
        spans = [s for s in spans if not span_filter(s)]
    for s in spans:
        print(f"  [{s.start}:{s.end}] '{s.original}' -> '{s.replacement}'")

    entities = [{"start": s.start, "end": s.end, "original": s.original, "replacement": s.replacement} for s in spans]
    anonymized_text, _, _segment_map = synthesize_anonymized(
        working_audio, output_path, words, entities,
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
    Returns (anonymized_text, output_path).
    """
    from text_anon import AnonSpan, apply_entities

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

    crossfade_n = int(sr * CROSSFADE_MS / 1000)
    orig_sr = sr
    suppressed_intervals = [iv for iv in intervals if iv.suppressed]
    tts_deltas: list[tuple[int, int]] = []
    splice_info: list[tuple[int, int, int]] = []  # (orig_s_start, orig_s_end, tts_samples)

    tts_intervals = [iv for iv in intervals if not iv.suppressed]
    _knn_model = None
    _knn_matching = None
    if not ref_wav_paths and tts_intervals:
        print("  Loading kNN-VC for per-segment voice conversion")
        _knn_model = torch.hub.load(
            "bshall/knn-vc", "knn_vc",
            prematched=True, trust_repo=True, pretrained=True, device=device,
        )
        _knn_matching = _knn_model.get_matching_set([_load_mono_16k(working_audio)])

    try:
        for iv in reversed(intervals):
            if iv.suppressed:
                continue
            s_start = max(0, min(int(iv.t_start * sr), len(audio)))
            s_end   = max(s_start, min(int(iv.t_end * sr), len(audio)))
            before = audio[:s_start]
            after  = audio[s_end:]
            tts_audio, tts_sr = _synthesize(iv.replacement_text)
            tts_audio = _resample(tts_audio, tts_sr, sr)
            if _knn_model is not None:
                tts_audio = _apply_knn_vc_to_segment(tts_audio, sr, _knn_model, _knn_matching, sr, topk=knn_topk)
            tts_audio = _match_amplitude(tts_audio, audio[s_start:s_end])
            splice_info.append((s_start, s_end, len(tts_audio)))
            delta = len(tts_audio) - (s_end - s_start)
            if delta != 0:
                tts_deltas.append((s_end, delta))
            joined = _crossfade(before, tts_audio, crossfade_n)
            audio  = _crossfade(joined, after, crossfade_n)
    finally:
        if _knn_model is not None:
            del _knn_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
            t_end_adj   = _adjust_sample(int(iv.t_end * orig_sr), tts_deltas) / orig_sr
            s_start = max(0, min(int(t_start_adj * sr), len(audio)))
            s_end   = max(s_start, min(int(t_end_adj * sr), len(audio)))
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
# Run as: python -m audio.pipelines.infilling [input] [output] [ref_wavs...]

_DEFAULT_REF_WAVS = [
    str(p)
    for p in sorted(KNN_REF_DIR.glob("in_aca*.wav"))[:3]
]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Infilling anonymization pipeline.")
    parser.add_argument("input",  nargs="?", default="data/cati_exemplo.wav")
    parser.add_argument("output", nargs="?", default="data/output_pipeline_infilling_knn.wav")
    parser.add_argument("ref_wavs", nargs="*")
    args = parser.parse_args()
    ref_paths = args.ref_wavs or _DEFAULT_REF_WAVS
    anonymize_audio(args.input, args.output, ref_wav_paths=ref_paths)
