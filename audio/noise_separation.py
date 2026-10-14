import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from ._conda_env import get_conda_env_python

# TUSS standalone inference package (+ its pretrained_models/ checkpoint dir).
_SI_DIR = Path(os.environ.get(
    "TUSS_DIR", str(Path(__file__).resolve().parent.parent / "third_party" / "standalone_inference")
)).expanduser()
_CKPT = _SI_DIR / "pretrained_models"

for _p in (str(_SI_DIR.parent), str(_SI_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_DIARIZEN_CONDA_ENV = os.environ.get("DIARIZEN_CONDA_ENV", "diarizen")
_DIARIZEN_MODEL_ID = os.environ.get("DIARIZEN_MODEL_ID", "BUT-FIT/diarizen-wavlm-large-s80-md-v2")
_DIARIZEN_HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diarizen_vad_helper.py")
_VAD_FADE_MS = 20

def _get_diarizen_python() -> str:
    return get_conda_env_python(_DIARIZEN_CONDA_ENV, os.environ.get("DIARIZEN_PYTHON"))


def _remove_leftover_voice(noise: np.ndarray, sr: int) -> np.ndarray:
    """Mute high-confidence residual speech in the separated noise track.

    Speech is detected via DiariZen (run out-of-process in its own conda env,
    since it needs a different dependency set than the rest of this codebase).
    """
    mono = noise if noise.ndim == 1 else noise.mean(axis=1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path = os.path.join(tmp_dir, "noise.wav")
        segments_path = os.path.join(tmp_dir, "segments.json")
        sf.write(audio_path, mono, sr)

        subprocess.run(
            [
                _get_diarizen_python(), _DIARIZEN_HELPER,
                "--audio", audio_path,
                "--output", segments_path,
                "--model", _DIARIZEN_MODEL_ID,
            ],
            check=True,
        )

        with open(segments_path) as f:
            segments = json.load(f)

    mask = np.ones(len(mono), dtype=np.float32)
    for start_s, end_s in segments:
        start = max(0, int(start_s * sr))
        end = min(len(mask), int(end_s * sr))
        if end > start:
            mask[start:end] = 0.0

    # Smooth the mask edges so muting doesn't introduce audible clicks.
    fade_samples = max(1, int(sr * _VAD_FADE_MS / 1000))
    window = np.hanning(fade_samples * 2 + 1)
    window /= window.sum()
    mask = np.convolve(mask, window, mode="same")

    return noise * (mask if noise.ndim == 1 else mask[:, None])


def separate_speech_noise(
    audio_path: str,
    speech_out: str,
    noise_out: str,
    device: str = "cpu",
    clean_leftover_voice: bool = True,
) -> None:
    """Separate speech from background using TUSS; write both components to WAV.

    When `clean_leftover_voice` is set, a DiariZen VAD pass mutes any
    high-confidence residual speech (typically the main speaker bleeding
    through) left in the noise track.
    """
    from standalone_inference.inference import PretrainedSeparator

    sep = PretrainedSeparator(str(_CKPT), device=device)
    result = sep.separate(audio_path, ["speech", "sfxbg"], css_segment_size=8, css_shift_size=4)

    speech = result.estimates[0]
    noise = result.estimates[1]
    sr = result.sample_rate

    if hasattr(speech, "numpy"):
        speech = speech.numpy()
    if hasattr(noise, "numpy"):
        noise = noise.numpy()

    sf.write(speech_out, speech, sr)

    if clean_leftover_voice:
        noise = _remove_leftover_voice(noise, sr)

    sf.write(noise_out, noise, sr)