import os
import re
import sys
import tempfile
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .. import asr as asr_module
import text_anon
from text_anon import AnonymizationConfig, AnonSpan
from ..tts import f5 as tts_f5_module
from .._paths import CENSORED_WAV


_BEEP_SENTINEL = "XXXXXX"
_TAG_RE = re.compile(r'^\[[A-Z]+_\d+\]$')
_CENSORED_WAV = CENSORED_WAV


def _is_beep(replacement: str) -> bool:
    return replacement == _BEEP_SENTINEL or bool(_TAG_RE.match(replacement))


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    g = gcd(orig_sr, target_sr)
    return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)


def _fit_beep(beep: np.ndarray, target_len: int, sr: int) -> np.ndarray:
    """Loop or trim beep to target_len with 5 ms fades."""
    if target_len <= 0:
        return np.empty(0, dtype=np.float32)
    reps = (target_len // len(beep)) + 1
    fitted = np.tile(beep, reps)[:target_len].copy()
    fade_n = min(int(sr * 0.005), target_len // 2)
    if fade_n > 0:
        fitted[:fade_n]  *= np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
        fitted[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
    return fitted


def _build_synthesis_text_and_beep_ranges(
    transcript: str, spans: list[AnonSpan]
) -> tuple[str, list[tuple[int, int]]]:
    """
    Build synthesis text and beep character ranges.
    Non-beep entities use their replacement text; beep entities keep the original word
    so TTS produces natural prosody for the whole sentence.
    Returns (synthesis_text, [(char_start, char_end), ...]) for beep positions.
    """
    parts: list[str] = []
    beep_ranges: list[tuple[int, int]] = []
    pos = 0
    cur = 0
    for span in sorted(spans, key=lambda s: s.start):
        gap = transcript[pos:span.start]
        parts.append(gap)
        cur += len(gap)
        if _is_beep(span.replacement):
            beep_ranges.append((cur, cur + len(span.original)))
            parts.append(span.original)
            cur += len(span.original)
        else:
            parts.append(span.replacement)
            cur += len(span.replacement)
        pos = span.end
    tail = transcript[pos:]
    parts.append(tail)
    return "".join(parts), beep_ranges


def _find_beep_intervals(
    synthesis_text: str,
    beep_char_ranges: list[tuple[int, int]],
    timed_words: list,
) -> list[tuple[float, float]]:
    """Map beep character ranges in synthesis_text to (start_s, end_s) using aligned word timestamps."""
    word_char_positions: list[tuple[int, int, object]] = []
    search_from = 0
    for tw in timed_words:
        needle = tw.word.strip()
        if not needle:
            continue
        idx = synthesis_text.lower().find(needle.lower(), search_from)
        if idx == -1:
            continue
        word_char_positions.append((idx, idx + len(needle), tw))
        search_from = idx + len(needle)

    intervals: list[tuple[float, float]] = []
    for bstart, bend in beep_char_ranges:
        start_t = end_t = None
        for wc_start, wc_end, tw in word_char_positions:
            if wc_start < bend and wc_end > bstart:
                if start_t is None:
                    start_t = tw.start
                end_t = tw.end
        if start_t is not None:
            intervals.append((start_t, end_t))
    return intervals


def _apply_beeps_inplace(audio_path: str, intervals: list[tuple[float, float]], output_path: str) -> None:
    """Splice fixed-length beep tone at each interval (processed in reverse to preserve sample offsets)."""
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    beep_base, beep_sr = sf.read(_CENSORED_WAV, dtype="float32")
    if beep_base.ndim > 1:
        beep_base = beep_base[:, 0]
    beep_base = _resample(beep_base, beep_sr, sr)
    beep_seg = _fit_beep(beep_base, len(beep_base), sr)
    for start_s, end_s in sorted(intervals, reverse=True):
        s = max(0, min(int(start_s * sr), len(audio)))
        e = max(s, min(int(end_s * sr), len(audio)))
        if e > s:
            audio = np.concatenate([audio[:s], beep_seg, audio[e:]])
    sf.write(output_path, audio, sr)


def _noop() -> None:
    pass


def _bounded_f5_ref(audio_path: str, transcript: str, timed_words: list | None = None):
    """Leading <=12s reference clip + the transcript of exactly that clip.

    F5's inf.py sizes its internal sentence-chunking (max_chars) and per-chunk
    duration estimate from len(ref_text) / ref_audio_duration. audio.tts.f5
    trims ref_audio to _MAX_REF_SECS but leaves ref_text full length, so a long
    utterance hands F5 ~12s of audio paired with a 20-25s ref_text -- a mismatch
    that breaks the in-context alignment and mis-sizes the batching. Reuse
    infilling_f5._get_local_ref (leading window: t_start=t_end=0 -> first <=12s
    of audio + the words spoken in it) to build a matched pair instead.

    Returns (ref_audio_path, ref_text, cleanup). cleanup unlinks the temp clip
    (a noop when the original path is returned unchanged -- i.e. audio already
    <=12s, or anything went wrong: this is a quality nudge, never a hard dep)."""
    from ..tts.f5 import _MAX_REF_SECS
    try:
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        if len(audio) <= int(_MAX_REF_SECS * sr):
            return audio_path, transcript, _noop
        if timed_words is None:
            timed_words = asr_module.get_timed_transcript(audio_path)
        from .infilling_f5 import _get_local_ref
        ref_arr, ref_text = _get_local_ref(audio, sr, timed_words, transcript, 0.0, 0.0)
        if not ref_text.strip() or len(ref_arr) == 0:
            return audio_path, transcript, _noop
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        sf.write(tmp, ref_arr, sr)

        def _cleanup() -> None:
            if os.path.exists(tmp):
                os.unlink(tmp)

        return tmp, ref_text, _cleanup
    except Exception as exc:
        print(f"  _bounded_f5_ref: using full reference ({type(exc).__name__}: {exc})")
        return audio_path, transcript, _noop


def anonymize_audio(
    audio_path: str,
    output_path: str,
    config: AnonymizationConfig | None = None,
    voice: str = "original",
) -> tuple[str, str, str]:
    """Transcribe, anonymize text, re-synthesize everything. Returns (transcript, anonymized_text, output_path)."""
    if config is None:
        config = AnonymizationConfig(regex="gen", per="gen", loc="gen", org="gen", misc="gen")

    print("Step 1: Transcribing audio")
    transcript = asr_module.get_transcript(audio_path)
    print(f"  {transcript[:120]}{'…' if len(transcript) > 120 else ''}")

    print("Step 2: Anonymizing text")
    anonymized, spans = text_anon.pipeline_with_spans(transcript, config)
    print(f"  {anonymized[:120]}{'…' if len(anonymized) > 120 else ''}")

    beep_spans = [s for s in spans if _is_beep(s.replacement)]
    ref_audio, ref_text, _cleanup = _bounded_f5_ref(audio_path, transcript)
    try:
        if not beep_spans:
            print("Step 3: Synthesizing speech")
            tts_f5_module.text_to_speech(anonymized, output_path, ref_audio, ref_text, voice)
            print(f"  Saved to {output_path}")
            return transcript, anonymized, output_path

        print(f"Step 3: Synthesizing speech with {len(beep_spans)} beep segment(s)")
        synthesis_text, beep_ranges = _build_synthesis_text_and_beep_ranges(transcript, spans)
        tts_f5_module.text_to_speech(synthesis_text, output_path, ref_audio, ref_text, voice)
    finally:
        _cleanup()

    print("Step 4: Aligning synthesized audio to locate beep positions")
    timed_words = asr_module.align_transcript(output_path, synthesis_text)
    beep_intervals = _find_beep_intervals(synthesis_text, beep_ranges, timed_words)

    if beep_intervals:
        print(f"Step 5: Applying {len(beep_intervals)} beep(s)")
        _apply_beeps_inplace(output_path, beep_intervals, output_path)
    else:
        print("  Warning: could not locate beep positions in aligned output")

    print(f"  Saved to {output_path}")
    return transcript, anonymized, output_path


def synthesize_from_entities(
    transcript: str,
    entities: list[dict],
    output_path: str,
    audio_path: str = "",
    voice: str = "original",
    timed_words: list | None = None,
) -> tuple[str, str]:
    """
    Synthesize audio from pre-computed entities. Returns (anonymized_text, output_path).

    timed_words: optional pre-computed TimedWord list for audio_path -- used only
    to build F5's bounded reference clip (see _bounded_f5_ref); re-derived via
    ASR when not supplied.
    """
    from text_anon import AnonSpan, apply_entities

    anonymized = apply_entities(transcript, entities)

    spans = [
        AnonSpan(
            start=e["start"],
            end=e["end"],
            original=e.get("original", transcript[e["start"]:e["end"]]),
            replacement=e.get("replacement", ""),
        )
        for e in entities
        if e.get("replacement")
    ]

    beep_spans = [s for s in spans if _is_beep(s.replacement)]
    ref_audio, ref_text, _cleanup = _bounded_f5_ref(audio_path, transcript, timed_words)
    try:
        if not beep_spans:
            print("Synthesizing full anonymized speech")
            tts_f5_module.text_to_speech(anonymized, output_path, ref_audio, ref_text, voice)
            print(f"  Saved to {output_path}")
            return anonymized, output_path

        print(f"Synthesizing with {len(beep_spans)} beep segment(s)")
        synthesis_text, beep_ranges = _build_synthesis_text_and_beep_ranges(transcript, spans)
        tts_f5_module.text_to_speech(synthesis_text, output_path, ref_audio, ref_text, voice)
    finally:
        _cleanup()

    print("Aligning synthesized audio to locate beep positions")
    out_timed_words = asr_module.align_transcript(output_path, synthesis_text)
    beep_intervals = _find_beep_intervals(synthesis_text, beep_ranges, out_timed_words)

    if beep_intervals:
        print(f"Applying {len(beep_intervals)} beep(s)")
        _apply_beeps_inplace(output_path, beep_intervals, output_path)
    else:
        print("  Warning: could not locate beep positions in aligned output")

    print(f"  Saved to {output_path}")
    return anonymized, output_path


# Run as: python -m audio.pipelines.asr_tts [input] [output]
if __name__ == "__main__":
    print("Script started", flush=True)
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "data/cati_exemplo.wav"
    out_file   = sys.argv[2] if len(sys.argv) > 2 else "data/output_pipeline_asr_tts_gen.wav"
    anonymize_audio(audio_file, out_file)
