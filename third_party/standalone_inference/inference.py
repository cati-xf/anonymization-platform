from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torchaudio


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .nets.model_wrapper import SeparationModel
from .utils.audio_utils import resample
from .utils.average_model_params import average_model_params
from .utils.config import yaml_to_parser


RESAMPLE_RATE = 48000
ALLOWED_CKPT_SUFFIXES = (".ckpt", ".pth")


@dataclass
class SeparationResult:
    estimates: torch.Tensor | np.ndarray
    mixture: torch.Tensor | np.ndarray
    sample_rate: int
    prompts: list[str]


def _resolve_checkpoint_paths(ckpt_path: Path) -> tuple[Path, list[Path]]:
    if ckpt_path.is_dir():
        assert (ckpt_path / "checkpoints").exists(), f"{ckpt_path} must contain checkpoints directory."
        config_path = ckpt_path / "hparams.yaml"
        ckpt_paths = [
            path
            for path in (ckpt_path / "checkpoints").iterdir()
            if path.suffix in ALLOWED_CKPT_SUFFIXES and path.name != "last.ckpt"
        ]
        assert ckpt_paths, f"No checkpoint files found under {ckpt_path / 'checkpoints'}"
        assert all(
            ckpt_paths[0].suffix == path.suffix for path in ckpt_paths
        ), "all the suffix of the pre-trained weights files must be the same"
    else:
        assert ckpt_path.suffix in ALLOWED_CKPT_SUFFIXES, f"Only {ALLOWED_CKPT_SUFFIXES} files are supported."
        config_path = ckpt_path.parent.parent / "hparams.yaml"
        ckpt_paths = [ckpt_path]

    return config_path, ckpt_paths


def _build_model_from_checkpoint(ckpt_path: Path) -> SeparationModel:
    config_path, ckpt_paths = _resolve_checkpoint_paths(ckpt_path)

    hparams = yaml_to_parser(config_path)
    hparams = hparams.parse_args([])
    model = SeparationModel(
        hparams.encoder_name,
        hparams.encoder_conf,
        hparams.decoder_name,
        hparams.decoder_conf,
        hparams.model_name,
        hparams.model_conf,
        hparams.css_conf,
        hparams.variance_normalization,
    )

    state_dict = average_model_params(ckpt_paths)
    cleaned_state_dict = {key.replace("model.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(cleaned_state_dict)
    return model


class PretrainedSeparator:
    """Run a pretrained source separation model from Python."""

    def __init__(self, ckpt_path: str | Path, device: str | torch.device | None = None):
        self.ckpt_path = Path(ckpt_path)
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = _build_model_from_checkpoint(self.ckpt_path).to(self.device)
        self.model.eval()

    def separate(
        self,
        audio: torch.Tensor | np.ndarray | str | Path,
        prompts: Sequence[str],
        *,
        sample_rate: int | None = None,
        start_time: int = 0,
        end_time: int = -1,
        ref_channel: int = 0,
        css_segment_size: int | None = None,
        css_shift_size: int | None = None,
        return_numpy: bool = False,
    ) -> SeparationResult:
        mix, fs = self._load_audio(audio, sample_rate)

        if mix.ndim == 1:
            mix = mix.unsqueeze(0)
        if mix.ndim != 2:
            raise ValueError(f"Expected audio with shape (channels, samples) or (samples,); got {tuple(mix.shape)}")
        if ref_channel < 0 or ref_channel >= mix.shape[0]:
            raise IndexError(f"ref_channel {ref_channel} is out of range for {mix.shape[0]} channels")

        start_sample = max(0, start_time) * fs
        if end_time < 0:
            end_sample = mix.shape[-1]
        else:
            end_sample = max(start_time + 1, end_time) * fs

        mix = mix[[ref_channel], start_sample:end_sample].clone()
        mix_return = mix.clone()
        mix = mix.to(self.device)

        if fs != RESAMPLE_RATE:
            mix = resample(mix, fs, RESAMPLE_RATE)

        prompts_list = list(prompts)
        with torch.inference_mode():
            if css_segment_size is not None:
                self.model.css_segment_size = css_segment_size
                self.model.css_shift_size = css_segment_size // 2 if css_shift_size is None else css_shift_size
                estimates = self.model.css(mix, [prompts_list])
            else:
                estimates = self.model(mix, [prompts_list])

        if fs != RESAMPLE_RATE:
            estimates = resample(estimates, RESAMPLE_RATE, fs)

        if estimates.ndim == 3 and estimates.shape[0] == 1:
            estimates = estimates.squeeze(0)
        if mix_return.ndim == 2 and mix_return.shape[0] == 1:
            mix_return = mix_return.squeeze(0)

        estimates = estimates.cpu()
        mix_return = mix_return.cpu()

        if return_numpy:
            return SeparationResult(
                estimates=estimates.numpy(),
                mixture=mix_return.numpy(),
                sample_rate=fs,
                prompts=prompts_list,
            )

        return SeparationResult(
            estimates=estimates,
            mixture=mix_return,
            sample_rate=fs,
            prompts=prompts_list,
        )

    def _load_audio(
        self,
        audio: torch.Tensor | np.ndarray | str | Path,
        sample_rate: int | None,
    ) -> tuple[torch.Tensor, int]:
        if isinstance(audio, (str, Path)):
            return torchaudio.load(Path(audio))

        if sample_rate is None:
            raise ValueError("sample_rate must be provided when audio is a tensor or NumPy array")

        mix = torch.as_tensor(audio, dtype=torch.float32)
        return mix, sample_rate