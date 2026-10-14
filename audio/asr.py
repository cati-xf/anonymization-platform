import difflib
import gc
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
import whisperx

from ._conda_env import get_conda_env_bin
from ._paths import REPO_ROOT

# Fine-tuned CAMÕES Whisper (ctranslate2). Overridable so a machine that isn't
# the original dev box can point at its own converted model without editing source.
_WHISPER_MODEL_DIR = os.environ.get(
    "WHISPER_MODEL_DIR", str(REPO_ROOT / "models" / "whisper_lv3_ft_ep_cpp_ct2")
)
# When truthy, keep the loaded whisperx model resident and reuse it across calls
# instead of reloading (and freeing) it every transcription. Off by default so
# the web app's memory profile is unchanged; batch evaluation jobs set it.
_CACHE_MODEL = os.environ.get("ASR_CACHE_MODEL", "").strip().lower() in ("1", "true", "yes", "on")
_MODEL_CACHE: dict = {}

# MFA lives in its own conda env (Kaldi deps conflict with the main venv); by
# default its binary is resolved from that env, MFA_BIN overrides the path.
_MFA_CONDA_ENV = os.environ.get("MFA_CONDA_ENV", "aligner")
_MFA_BIN      = os.environ.get("MFA_BIN") or get_conda_env_bin(_MFA_CONDA_ENV, "mfa")
_MFA_DICT     = os.environ.get("MFA_DICT",     "portuguese_mfa")
_MFA_ACOUSTIC = os.environ.get("MFA_ACOUSTIC", "portuguese_mfa")
# The whole file is aligned as a single utterance (see _mfa_align), so for
# longer recordings MFA's own defaults (10/40) can fail to find any alignment
# path at all (NoAlignmentsError) rather than just aligning poorly.
_MFA_BEAM       = os.environ.get("MFA_BEAM", "100")
_MFA_RETRY_BEAM = os.environ.get("MFA_RETRY_BEAM", "400")

_MFA_SILENCE = {"", "sp", "sil", "<eps>", "spn"}


@dataclass
class TimedWord:
    """ Single word from transcript with audio timestamps """
    word: str
    start: float
    end: float


def _normalize_for_mfa(word: str) -> str:
    return re.sub(r"[^\w]", "", word, flags=re.UNICODE).lower()


def _restore_original_words(original_words: list[str], aligned: list[TimedWord]) -> list[TimedWord]:
    """Map MFA-aligned tokens back to original Whisper word forms.

    MFA may drop words it can't align (they appear as 'spn' and are filtered).
    Sequence-align the normalized originals against MFA output to find which
    words were dropped, then interpolate timestamps for the gaps.
    """
    # Happy path: counts match, nothing was dropped.
    if len(aligned) == len(original_words):
        return [TimedWord(word=orig, start=tw.start, end=tw.end)
                for orig, tw in zip(original_words, aligned)]

    norm_orig = [_normalize_for_mfa(w) for w in original_words]
    mfa_labels = [tw.word for tw in aligned]

    # Place MFA-confirmed timestamps at their matched original indices.
    result: list[TimedWord | None] = [None] * len(original_words)
    matcher = difflib.SequenceMatcher(None, norm_orig, mfa_labels, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            for orig_i, mfa_j in zip(range(i1, i2), range(j1, j2)):
                result[orig_i] = TimedWord(
                    word=original_words[orig_i],
                    start=aligned[mfa_j].start,
                    end=aligned[mfa_j].end,
                )

    # Interpolate contiguous runs of dropped words.
    i = 0
    while i < len(original_words):
        if result[i] is not None:
            i += 1
            continue
        j = i
        while j < len(original_words) and result[j] is None:
            j += 1
        prev_end   = result[i - 1].end  if i > 0                    else 0.0
        next_start = result[j].start    if j < len(original_words)  else prev_end
        n    = j - i
        step = (next_start - prev_end) / n
        for k in range(n):
            result[i + k] = TimedWord(
                word=original_words[i + k],
                start=prev_end + k * step,
                end=prev_end + (k + 1) * step,
            )
        i = j

    return result  # type: ignore[return-value]  # all slots filled above


def _mfa_align(audio_path: str, transcript: str) -> list[TimedWord]:
    """Forced alignment via MFA. Writes a temp corpus, runs mfa align, parses TextGrid."""
    from praatio import textgrid as tgio

    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_dir = Path(tmpdir) / "corpus"
        corpus_dir.mkdir()
        out_dir = Path(tmpdir) / "output"

        shutil.copy2(audio_path, str(corpus_dir / "utt.wav"))
        (corpus_dir / "utt.lab").write_text(transcript, encoding="utf-8")

        aligner_env_bin = str(Path(_MFA_BIN).parent)
        env = os.environ.copy()
        env["PATH"] = aligner_env_bin + os.pathsep + env.get("PATH", "")

        # MFA's unpack_archive fails with FileExistsError if the extracted model
        # directory already exists (missing exist_ok in makedirs). Delete it first
        # so MFA can re-extract from the cached zip file.
        mfa_root = Path(env.get("MFA_ROOT_DIR", Path.home() / "Documents" / "MFA"))
        extracted_model = mfa_root / "extracted_models" / "acoustic" / f"{_MFA_ACOUSTIC}_acoustic"
        if extracted_model.exists():
            shutil.rmtree(extracted_model)

        proc = subprocess.run(
            [_MFA_BIN, "align", "--clean",
             str(corpus_dir), _MFA_DICT, _MFA_ACOUSTIC, str(out_dir),
             "--beam", _MFA_BEAM, "--retry_beam", _MFA_RETRY_BEAM],
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"mfa align failed (exit {proc.returncode}) for {audio_path!r}:\n"
                f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
            )

        # MFA may place TextGrid directly in out_dir or in a speaker subdirectory
        tg_files = list(out_dir.rglob("utt.TextGrid"))
        if not tg_files:
            raise FileNotFoundError(f"MFA produced no TextGrid under {out_dir}")

        tg = tgio.openTextgrid(str(tg_files[0]), includeEmptyIntervals=False)
        word_tier = tg.getTier("words")

        aligned: list[TimedWord] = []
        for e in word_tier.entries:
            label = e.label.strip()
            if label and label not in _MFA_SILENCE:
                aligned.append(TimedWord(word=label, start=e.start, end=e.end))

        original_words = transcript.split()
        return _restore_original_words(original_words, aligned)


def _get_model(device: str, compute_type: str):
    key = (device, compute_type)
    if _CACHE_MODEL and key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    model = whisperx.load_model(
        _WHISPER_MODEL_DIR,
        device=device,
        compute_type=compute_type,
        language="pt",
        task="transcribe",
        asr_options={"beam_size": 1},
    )
    model.model.feat_kwargs["feature_size"] = 128
    if _CACHE_MODEL:
        _MODEL_CACHE[key] = model
    return model


def _release_model(model) -> None:
    """Free the whisperx model unless it's being kept resident (ASR_CACHE_MODEL)."""
    if _CACHE_MODEL:
        return
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_and_transcribe(audio_path: str, device: str, compute_type: str):
    model = _get_model(device, compute_type)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, language="pt", task="transcribe")
    return model, audio, result


def get_transcript(audio_path: str, device="cuda", compute_type="float16") -> str:
    """ Get transcript from fine tuned (punctuation, etc) CAMÕES Whisper model """
    model, _, result = _load_and_transcribe(audio_path, device, compute_type)
    transcript = " ".join(seg["text"] for seg in result["segments"])
    _release_model(model)
    return transcript


def align_transcript(audio_path: str, transcript: str, device="cuda") -> list[TimedWord]:
    """Forced alignment of a known transcript to audio via MFA. Returns word-level timestamps."""
    return _mfa_align(audio_path, transcript)


def get_timed_transcript(audio_path: str, device="cuda", compute_type="float16") -> list[TimedWord]:
    """ Transcribe with CAMÕES Whisper, then force-align with MFA for word timestamps. """
    model, _, result = _load_and_transcribe(audio_path, device, compute_type)
    transcript = " ".join(seg["text"].strip() for seg in result["segments"])

    _release_model(model)

    return _mfa_align(audio_path, transcript)
