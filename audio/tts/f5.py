import json
import os
import subprocess
import tempfile
from pathlib import Path

import soundfile as sf

from .._conda_env import get_conda_env_python

_F5_DIR = os.environ.get("F5_TTS_DIR", str(Path.home() / "F5-TTS" / "F5-TTS"))
_CONDA_ENV = os.environ.get("F5_TTS_CONDA_ENV", "f5-tts")
F5_REFS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f5_refs")
_MAX_REF_SECS = 12.0


def _load_voice_refs() -> dict[str, tuple[str, str]]:
    p = os.path.join(F5_REFS_DIR, "voice_refs.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return {k: (v["file"], v["transcript"]) for k, v in data.items()}


_VOICE_REFS: dict[str, tuple[str, str]] = _load_voice_refs()


def _prepare_ref_audio(ref_audio: str) -> tuple[str, bool]:
    """Return (path, needs_cleanup). Trims to ≤12s in a tempfile if needed."""
    ref_audio = os.path.abspath(ref_audio)
    audio, sr = sf.read(ref_audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    max_samples = int(_MAX_REF_SECS * sr)
    if len(audio) <= max_samples:
        return ref_audio, False
    tmp = tempfile.mktemp(suffix=".wav")
    sf.write(tmp, audio[:max_samples], sr)
    return tmp, True


def text_to_speech(
    text: str,
    output_path: str,
    ref_audio: str,
    ref_text: str,
    voice: str = "original",
) -> str:
    if voice != "original" and voice in _VOICE_REFS:
        fname, ref_text = _VOICE_REFS[voice]
        ref_audio = os.path.join(F5_REFS_DIR, fname)

    output_path = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_path)
    output_file = os.path.basename(output_path)

    ref_path, cleanup = _prepare_ref_audio(ref_audio)
    try:
        subprocess.run(
            [
                get_conda_env_python(_CONDA_ENV, os.environ.get("F5_TTS_PYTHON")),
                os.path.join(_F5_DIR, "inf.py"),
                "--ref_audio", ref_path,
                "--ref_text", ref_text,
                "--gen_text", text,
                "--output_file", output_file,
                "--output_dir", output_dir,
            ],
            check=True,
            cwd=_F5_DIR,
        )
    finally:
        if cleanup:
            os.unlink(ref_path)

    return output_path
