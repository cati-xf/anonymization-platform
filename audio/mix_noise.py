#!/usr/bin/env python3
"""
mix_noise.py — Overlay original background noise onto anonymized speech.

For segments in the segment_map the noise is time-stretched/compressed to match
the new (anonymized) duration. Outside those segments the noise is copied 1:1.

Usage:
    python -m audio.mix_noise single <speech.wav> <noise.wav> <segment_map.json> <output.wav>

    segment_map.json: list of {"orig_start", "orig_end", "out_start", "out_end"}
                      (seconds, as returned by synthesize_anonymized in each pipeline)

Can also be imported and called directly:
    from audio.mix_noise import mix_noise
    mix_noise(speech_path, noise_path, segment_map, output_path)
"""

import argparse
import json
import os
import subprocess
import tempfile
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


ORIGINAL_EXTS = {".wav", ".mp3", ".aac", ".m4a"}

# Below this duration, a chunk's local RMS is too noisy to trust directly (entity
# spans from segment_map have no minimum duration and can be well under 0.5s, e.g.
# a short name/number) -- the measurement window is widened symmetrically around
# the chunk until it reaches this floor (see _compute_scale_array).
_MIN_CHUNK_S = 0.4
# Hanning crossfade between adjacent chunks' scale factors, so level corrections
# don't step discontinuously at chunk boundaries. Matches the 20ms fade convention
# already used in noise_separation.py's _VAD_FADE_MS.
_SCALE_FADE_MS = 20


# ── Audio helpers ─────────────────────────────────────────────────────────────

def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or len(audio) == 0:
        return audio
    g = gcd(orig_sr, target_sr)
    return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)


def _stretch_to_len(audio: np.ndarray, target_len: int) -> np.ndarray:
    """Resample audio array to exactly target_len samples (pitch-shifts, fine for noise)."""
    if len(audio) == 0 or target_len <= 0:
        return np.zeros(max(0, target_len), dtype=np.float32)
    if len(audio) == target_len:
        return audio
    g = gcd(target_len, len(audio))
    return resample_poly(audio, target_len // g, len(audio) // g).astype(np.float32)


def _extract(noise: np.ndarray, noise_sr: int, t_start: float, t_end: float) -> np.ndarray:
    """Extract noise[t_start:t_end] (seconds), zero-padding if out of bounds."""
    s = round(t_start * noise_sr)
    e = round(t_end * noise_sr)
    s = max(0, s)
    e = max(s, e)
    chunk = noise[s:min(e, len(noise))]
    pad = e - s - len(chunk)
    if pad > 0:
        chunk = np.pad(chunk, (0, pad))
    return chunk.astype(np.float32)


# ── Core builder ──────────────────────────────────────────────────────────────

def build_noise_track(
    noise: np.ndarray,
    noise_sr: int,
    speech_len: int,
    speech_sr: int,
    segment_map: list[dict],
) -> tuple[np.ndarray, list[dict]]:
    """
    Build a noise track of length speech_len at speech_sr.

    Segments in segment_map are time-stretched; gaps between them are 1:1.

    Returns (noise_track, chunks): chunks is the gap/segment/trailing-gap breakdown
    used to build the track, each entry carrying the exact noise audio written for
    that span -- reused by _compute_scale_array() for local SNR matching.
    """
    out = np.zeros(speech_len, dtype=np.float32)
    chunks: list[dict] = []
    segs = sorted(segment_map, key=lambda s: s["out_start"])

    out_pos = 0          # samples written into output so far
    orig_t = 0.0         # corresponding position in original noise (seconds)

    for seg in segs:
        # Clamp to [0, speech_len]: segment_map's out_start/out_end are the
        # synthesizing backend's own prediction of where each edit landed in
        # the final audio, and can overrun the actual saved output by a few
        # samples (or more, on long/entity-heavy files) when the backend's
        # duration bookkeeping drifts from what actually got written. Without
        # this, out[out_seg_start:out_seg_end] silently truncates via normal
        # slicing while the chunk built from the untruncated seg_out_len
        # doesn't, raising a broadcast ValueError on the += below.
        out_seg_start = min(max(0, round(seg["out_start"] * speech_sr)), speech_len)
        out_seg_end   = min(max(out_seg_start, round(seg["out_end"] * speech_sr)), speech_len)
        orig_seg_start = seg["orig_start"]
        orig_seg_end   = seg["orig_end"]

        # ── Gap before this segment (1:1 copy) ────────────────────────────────
        gap_out_len = out_seg_start - out_pos
        gap_orig_dur = orig_seg_start - orig_t   # should match gap_out_len/speech_sr
        if gap_out_len > 0:
            if gap_orig_dur > 0:
                chunk = _extract(noise, noise_sr, orig_t, orig_t + gap_orig_dur)
                chunk = _resample(chunk, noise_sr, speech_sr)
                chunk = _stretch_to_len(chunk, gap_out_len)   # absorb rounding slop
            else:
                chunk = np.zeros(gap_out_len, dtype=np.float32)
            out[out_pos:out_pos + gap_out_len] += chunk
            chunks.append({
                "out_start": out_pos, "out_end": out_pos + gap_out_len,
                "orig_start": orig_t, "orig_end": orig_t + max(gap_orig_dur, 0.0),
                "noise_chunk": chunk,
            })

        out_pos = out_seg_start
        orig_t  = orig_seg_start

        # ── Replaced segment (time-stretched) ─────────────────────────────────
        seg_out_len  = out_seg_end - out_seg_start
        seg_orig_dur = orig_seg_end - orig_seg_start
        if seg_out_len > 0:
            if seg_orig_dur > 0:
                chunk = _extract(noise, noise_sr, orig_seg_start, orig_seg_end)
                chunk = _resample(chunk, noise_sr, speech_sr)
                chunk = _stretch_to_len(chunk, seg_out_len)
            else:
                chunk = np.zeros(seg_out_len, dtype=np.float32)
            out[out_seg_start:out_seg_end] += chunk
            chunks.append({
                "out_start": out_seg_start, "out_end": out_seg_end,
                "orig_start": orig_seg_start, "orig_end": orig_seg_end,
                "noise_chunk": chunk,
            })

        out_pos = out_seg_end
        orig_t  = orig_seg_end

    # ── Trailing gap after last segment (1:1 copy) ────────────────────────────
    trail_out_len = speech_len - out_pos
    if trail_out_len > 0:
        trail_orig_dur = trail_out_len / speech_sr
        chunk = _extract(noise, noise_sr, orig_t, orig_t + trail_orig_dur)
        chunk = _resample(chunk, noise_sr, speech_sr)
        chunk = _stretch_to_len(chunk, trail_out_len)
        out[out_pos:] += chunk
        chunks.append({
            "out_start": out_pos, "out_end": speech_len,
            "orig_start": orig_t, "orig_end": orig_t + trail_orig_dur,
            "noise_chunk": chunk,
        })

    return out, chunks


# ── Audio loading ─────────────────────────────────────────────────────────────

def _load_audio(path: str) -> tuple[np.ndarray, int]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        audio, sr = sf.read(path, dtype="float32")
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", path, "-ac", "1", tmp],
                check=True, capture_output=True,
            )
            audio, sr = sf.read(tmp, dtype="float32")
        finally:
            os.unlink(tmp)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2)))


def _speech_scale(speech: np.ndarray, noise: np.ndarray, original: np.ndarray) -> float:
    """
    Compute scale so that RMS(scale*speech + noise) ≈ RMS(original).
    Derived from: rms_orig² ≈ scale²*rms_speech² + rms_noise²  (assuming independence).
    """
    rms_orig    = _rms(original)
    rms_noise   = _rms(noise)
    rms_speech  = _rms(speech)
    if rms_speech < 1e-8:
        return 1.0
    target_speech_power = rms_orig ** 2 - rms_noise ** 2
    if target_speech_power <= 0:
        # noise already louder than original — just match overall RMS
        return rms_orig / (rms_speech + 1e-8)
    return float(np.sqrt(target_speech_power) / rms_speech)


def _compute_scale_array(
    chunks: list[dict],
    speech: np.ndarray,
    original: np.ndarray,
    original_sr: int,
    noise: np.ndarray,
    noise_sr: int,
    speech_sr: int,
    speech_len: int,
) -> np.ndarray:
    """
    Per-sample gain envelope so that scale*speech (+ noise) locally matches
    original's RMS, computed chunk by chunk instead of once for the whole file --
    see build_noise_track's gap/segment breakdown.

    Chunks shorter than _MIN_CHUNK_S have too little audio for a reliable local RMS
    estimate. Rather than borrowing a neighboring chunk's scale outright (which, for
    a short entity span, is usually an untouched gap whose "no correction needed"
    scale says nothing about the span's actual level), the measurement window is
    widened symmetrically around the chunk -- in both the original and output
    timelines -- until it reaches _MIN_CHUNK_S, so every chunk gets a real,
    directly-measured scale. The result is smoothed with a short Hanning crossfade
    so adjacent chunks don't produce audible level steps.
    """
    if not chunks:
        return np.ones(speech_len, dtype=np.float32)

    orig_len_s = len(original) / original_sr
    scale_arr = np.ones(speech_len, dtype=np.float32)

    for c in chunks:
        out_s, out_e = c["out_start"], c["out_end"]
        orig_s, orig_e = c["orig_start"], c["orig_end"]
        dur = (out_e - out_s) / speech_sr

        if dur < _MIN_CHUNK_S:
            pad_s = (_MIN_CHUNK_S - dur) / 2.0

            pad_out = round(pad_s * speech_sr)
            out_s = out_s - pad_out
            out_e = out_e + pad_out
            if out_s < 0:
                out_e += -out_s
                out_s = 0
            if out_e > speech_len:
                out_s -= out_e - speech_len
                out_e = speech_len
            out_s = max(out_s, 0)

            orig_s = orig_s - pad_s
            orig_e = orig_e + pad_s
            if orig_s < 0:
                orig_e += -orig_s
                orig_s = 0.0
            if orig_e > orig_len_s:
                orig_s -= orig_e - orig_len_s
                orig_e = orig_len_s
            orig_s = max(orig_s, 0.0)

        speech_chunk = speech[out_s:out_e]
        original_chunk = _extract(original, original_sr, orig_s, orig_e)
        noise_chunk = _extract(noise, noise_sr, orig_s, orig_e)
        noise_chunk = _resample(noise_chunk, noise_sr, speech_sr)
        noise_chunk = _stretch_to_len(noise_chunk, out_e - out_s)

        scale = _speech_scale(speech_chunk, noise_chunk, original_chunk)
        scale_arr[c["out_start"]:c["out_end"]] = scale

    fade_samples = max(1, int(speech_sr * _SCALE_FADE_MS / 1000))
    window = np.hanning(fade_samples * 2 + 1)
    window /= window.sum()
    return np.convolve(scale_arr, window, mode="same").astype(np.float32)


# ── Public API ────────────────────────────────────────────────────────────────

def mix_noise(
    speech_path: str,
    noise_path: str,
    segment_map: list[dict],
    output_path: str,
    original_path: str | None = None,
) -> None:
    """
    Overlay noise_path onto speech_path with segment-aware time stretching.
    If original_path is given, scales speech so the SNR matches the original.
    Writes result to output_path at speech_path's sample rate.
    """
    speech, speech_sr = sf.read(speech_path, dtype="float32")
    if speech.ndim > 1:
        speech = speech.mean(axis=1)

    noise, noise_sr = sf.read(noise_path, dtype="float32")
    if noise.ndim > 1:
        noise = noise.mean(axis=1)

    noise_track, chunks = build_noise_track(noise, noise_sr, len(speech), speech_sr, segment_map)

    if original_path is not None:
        original, original_sr = _load_audio(original_path)
        scale_arr = _compute_scale_array(
            chunks, speech, original, original_sr, noise, noise_sr, speech_sr, len(speech)
        )
        print(f"  Speech scale: min={scale_arr.min():.4f} max={scale_arr.max():.4f} mean={scale_arr.mean():.4f}")
        speech = speech * scale_arr

    mixed = speech + noise_track
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed /= peak

    sf.write(output_path, mixed, speech_sr)
    print(f"Saved mixed audio → {output_path}")


# ── Batch processing ─────────────────────────────────────────────────────────

def batch_mix_noise(
    input_dir: str,
    speech_name: str = "anonymized.wav",
    noise_name: str = "sfxbg.wav",
    output_name: str = "mixed_normalized.wav",
    match_snr: bool = False,
) -> None:
    """Process every subdirectory of input_dir that contains the expected files."""
    root = os.path.abspath(input_dir)
    subdirs = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )
    if not subdirs:
        print(f"No subdirectories found in {root}")
        return

    ok = skipped = errors = 0
    for name in subdirs:
        folder = os.path.join(root, name)
        speech_path = os.path.join(folder, speech_name)
        noise_path  = os.path.join(folder, noise_name)
        output_path = os.path.join(folder, output_name)

        json_files = [f for f in os.listdir(folder) if f.endswith(".json")]
        if len(json_files) > 1:
            print(f"[SKIP] {name} — multiple JSON files: {', '.join(json_files)}")
            skipped += 1
            continue
        seg_map_path = os.path.join(folder, json_files[0]) if json_files else None

        missing = [f for f in (speech_path, noise_path) if not os.path.exists(f)]
        if missing:
            print(f"[SKIP] {name} — missing: {', '.join(os.path.basename(m) for m in missing)}")
            skipped += 1
            continue

        original_path = None
        if match_snr:
            originals = [
                f for f in os.listdir(folder)
                if os.path.splitext(f)[1].lower() in ORIGINAL_EXTS
                and f not in (speech_name, noise_name, output_name)
            ]
            if len(originals) == 1:
                original_path = os.path.join(folder, originals[0])
            elif len(originals) == 0:
                print(f"[WARN] {name} — --match-snr set but no original found, skipping SNR match")
            else:
                print(f"[WARN] {name} — --match-snr set but multiple originals found ({', '.join(originals)}), skipping SNR match")

        try:
            if seg_map_path:
                with open(seg_map_path) as f:
                    seg_map = json.load(f)
            else:
                seg_map = []
                print(f"[INFO] {name} — no segment map, copying noise 1:1")
            mix_noise(speech_path, noise_path, seg_map, output_path, original_path=original_path)
            ok += 1
        except Exception as exc:
            print(f"[ERROR] {name} — {exc}")
            errors += 1

    print(f"\nDone: {ok} mixed, {skipped} skipped, {errors} errors")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overlay noise onto anonymized speech with segment-aware time stretching."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    sp_single = subparsers.add_parser("single", help="Process one file")
    sp_single.add_argument("speech",        help="Anonymized speech WAV")
    sp_single.add_argument("noise",         help="Original background noise WAV")
    sp_single.add_argument("segment_map",   help="JSON file with segment map list")
    sp_single.add_argument("output",        help="Output WAV path")
    sp_single.add_argument("--original",    default=None, help="Original recording for SNR matching")

    sp_batch = subparsers.add_parser("batch", help="Process all subdirs in a folder")
    sp_batch.add_argument("input_dir",   help="Folder containing job subdirectories")
    sp_batch.add_argument("--speech",    default="anonymized.wav")
    sp_batch.add_argument("--noise",     default="sfxbg.wav")
    sp_batch.add_argument("--output",    default="mixed.wav")
    sp_batch.add_argument("--match-snr", action="store_true",
                          help="Scale speech to restore original speech-to-noise ratio")

    args = parser.parse_args()

    if args.mode == "single":
        with open(args.segment_map) as f:
            seg_map = json.load(f)
        mix_noise(args.speech, args.noise, seg_map, args.output, original_path=args.original)
    else:
        batch_mix_noise(
            args.input_dir,
            speech_name=args.speech,
            noise_name=args.noise,
            output_name=args.output,
            match_snr=args.match_snr,
        )
