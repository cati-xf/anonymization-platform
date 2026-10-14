"""
F5-TTS in-place speech editing helper. Run in the f5-tts conda environment via subprocess.

Usage:
    python f5_speech_edit_helper.py \
        --audio input.wav \
        --target_texts '["before replacement after", "before replacement2 after2"]' \
        --parts_to_edit "[[1.2, 2.4], [5.0, 6.3]]" \
        [--fix_duration "[1.2, 1.3]"] \
        --output output.wav

Processes each edit independently in a windowed context (CONTEXT_BEFORE s before,
CONTEXT_AFTER s after). Actual durations written to <output>.dur.json as a sidecar.
"""
import argparse
import json
import os
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torchaudio
from hydra.utils import get_class
from importlib.resources import files
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import load_model, load_vocoder
from f5_tts.model.utils import convert_char_to_pinyin


_F5_DIR = os.environ.get("F5_TTS_DIR", str(Path.home() / "F5-TTS" / "F5-TTS"))

CONTEXT_BEFORE = 10.0  # seconds of original audio kept before each edit
CONTEXT_AFTER  = 5.0   # seconds of original audio kept after each edit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--target_texts", required=True, help="JSON list of strings, one per interval")
    parser.add_argument("--parts_to_edit", required=True, help="JSON list of [start_sec, end_sec] pairs")
    parser.add_argument("--fix_duration", default=None, help="JSON list of durations (seconds) per edit")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    parts_to_edit: list[list[float]] = json.loads(args.parts_to_edit)
    fix_duration: list[float] | None = json.loads(args.fix_duration) if args.fix_duration else None
    target_texts: list[str] = json.loads(args.target_texts)

    device = (
        "cuda" if torch.cuda.is_available()
        else "xpu" if torch.xpu.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}", flush=True)

    # Portuguese model paths
    ckpt_path  = os.path.join(_F5_DIR, "ckpts/f5ptpt-tts/model_1259500.pt")
    vocab_file = os.path.join(_F5_DIR, "data/f5ptpt-tts_char/vocab.txt")
    cfg_path   = os.path.join(_F5_DIR, "src/f5_tts/configs/F5TTS_v1_Base.yaml")

    model_cfg = OmegaConf.load(cfg_path)
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")

    mel_spec_type      = model_cfg.model.mel_spec.mel_spec_type
    target_sample_rate = model_cfg.model.mel_spec.target_sample_rate
    n_mel_channels     = model_cfg.model.mel_spec.n_mel_channels
    hop_length         = model_cfg.model.mel_spec.hop_length
    nfe_step           = 32
    cfg_strength       = 2.0
    sway_sampling_coef = -1.0
    target_rms         = 0.1

    vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=False, local_path="")
    model   = load_model(
        model_cls=model_cls,
        model_cfg=model_cfg.model.arch,
        ckpt_path=ckpt_path,
        mel_spec_type=mel_spec_type,
        vocab_file=vocab_file,
        device=device,
    )

    # Load audio — kept on CPU, no global RMS normalization
    audio, sr = torchaudio.load(args.audio)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if sr != target_sample_rate:
        audio = torchaudio.functional.resample(audio, sr, target_sample_rate)

    current_audio = audio  # (1, T_total), CPU; mutated by each edit
    time_offset   = 0.0    # cumulative time drift from previous edits
    actual_durations: list[float] = []
    fix_dur_list = fix_duration.copy() if fix_duration is not None else None

    print(f"Editing {len(parts_to_edit)} segment(s) with windowed context", flush=True)

    for i, (orig_start, orig_end) in enumerate(parts_to_edit):
        fix_dur = fix_dur_list[i] if fix_dur_list is not None else (orig_end - orig_start)

        # (A) Adjusted times in current_audio's frame of reference
        start     = orig_start + time_offset
        end       = orig_end   + time_offset
        audio_dur = current_audio.shape[-1] / target_sample_rate

        # (B) Window bounds clamped to audio extent
        context_before = min(CONTEXT_BEFORE, start)
        context_after  = min(CONTEXT_AFTER, audio_dur - end)
        win_start_s    = start - context_before
        win_end_s      = end   + context_after

        # (C) Sample-level indices in current_audio
        ws = round(win_start_s * target_sample_rate)
        es = round(start       * target_sample_rate)
        ee = round(end         * target_sample_rate)
        we = min(current_audio.shape[-1], round(win_end_s * target_sample_rate))

        # (D) Extract window and per-window RMS normalization
        window     = current_audio[:, ws:we]
        rms_window = torch.sqrt(torch.mean(torch.square(window)))
        if rms_window > 1e-8 and rms_window < target_rms:
            window_norm  = window * (target_rms / rms_window)
            denorm_scale = rms_window / target_rms
        else:
            window_norm  = window
            denorm_scale = 1.0

        # (E) Mel spectrogram of window on device
        with torch.inference_mode():
            window_mel = model.mel_spec(window_norm.to(device)).permute(0, 2, 1)  # (1, T_w, n_mel)

        # (F) Frame-level offsets within the window
        local_sf         = round(context_before * target_sample_rate / hop_length)
        orig_span_frames = round((end - start)  * target_sample_rate / hop_length)
        fix_dur_frames   = round(fix_dur        * target_sample_rate / hop_length)
        local_ef         = min(local_sf + orig_span_frames, window_mel.shape[1])
        after_frames     = window_mel.shape[1] - local_ef

        # (G) Windowed mel_cond: keep_before + zeros_for_edit + keep_after
        mel_cond = torch.cat([
            window_mel[:, :local_sf, :],
            torch.zeros(1, fix_dur_frames, n_mel_channels, device=device),
            window_mel[:, local_ef:, :],
        ], dim=1)
        edit_mask = torch.cat([
            torch.ones( 1, local_sf,       dtype=torch.bool, device=device),
            torch.zeros(1, fix_dur_frames,  dtype=torch.bool, device=device),
            torch.ones( 1, after_frames,    dtype=torch.bool, device=device),
        ], dim=-1)

        # (H) Inference on this window only
        with torch.inference_mode():
            generated, _ = model.sample(
                cond=mel_cond,
                text=convert_char_to_pinyin([target_texts[i]]),
                duration=mel_cond.shape[1],
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                seed=None,
                edit_mask=edit_mask,
            )

        # (I) Decode ONLY the generated edit frames — keeps unedited regions artifact-free
        generated = generated.to(torch.float32)
        edit_mel  = generated[:, local_sf : local_sf + fix_dur_frames, :].permute(0, 2, 1)
        with torch.inference_mode():
            if mel_spec_type == "vocos":
                edit_audio = vocoder.decode(edit_mel)
            elif mel_spec_type == "bigvgan":
                edit_audio = vocoder(edit_mel).squeeze(0)
            else:
                raise ValueError(f"Unknown mel_spec_type: {mel_spec_type}")

        if edit_audio.ndim == 1:
            edit_audio = edit_audio.unsqueeze(0)
        edit_audio = edit_audio.cpu() * denorm_scale

        # (J) Splice into current_audio and update cumulative time offset
        actual_dur = edit_audio.shape[-1] / target_sample_rate
        actual_durations.append(actual_dur)
        current_audio = torch.cat([current_audio[:, :es], edit_audio, current_audio[:, ee:]], dim=-1)
        time_offset += actual_dur - (orig_end - orig_start)
        print(
            f"  Edit {i}: [{orig_start:.2f}s, {orig_end:.2f}s] -> {actual_dur:.2f}s "
            f"(offset {time_offset:+.3f}s)",
            flush=True,
        )

    # Write actual durations as sidecar so the pipeline can use true values for beep timing
    dur_path = args.output + ".dur.json"
    with open(dur_path, "w") as _f:
        json.dump(actual_durations, _f)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torchaudio.save(args.output, current_audio, target_sample_rate)
    print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
