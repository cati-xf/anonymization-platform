import os
import shutil
from pathlib import Path

AUDIO_PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIO_PKG_DIR.parent

CENSORED_WAV = AUDIO_PKG_DIR / "censored.wav"
F5_SPEECH_EDIT_HELPER = AUDIO_PKG_DIR / "tts" / "f5_speech_edit_helper.py"
KNN_REF_DIR = REPO_ROOT / "ref" / "tmp"
SYNTH_TMP_DIR = REPO_ROOT / "data" / "synth_tmp"


def ensure_ffmpeg_on_path() -> None:
    """Make sure an `ffmpeg` executable is on PATH. Libraries such as whisperx
    shell out to plain `ffmpeg`, so it is not enough to resolve a path for our
    own calls. If none is installed, use $FFMPEG_BIN or the binary bundled with
    imageio-ffmpeg (for machines without root access to install one), exposed
    as data/bin/ffmpeg. PATH is process-wide, so subprocesses see it too."""
    if shutil.which("ffmpeg"):
        return
    exe = os.environ.get("FFMPEG_BIN")
    if not exe:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return  # nothing to fall back to; ffmpeg calls fail with "not found"
    bin_dir = REPO_ROOT / "data" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    link = bin_dir / "ffmpeg"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(exe)
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
