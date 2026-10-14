# Standalone Inference

This folder contains the minimal Python wrapper needed to run the pretrained separation model from code.

Usage from the repository root:

```python
from standalone_inference.inference import PretrainedSeparator

separator = PretrainedSeparator("pretrained_models/tuss.medium.2-4src/checkpoints/model.pth")
result = separator.separate(audio_waveform, ["speech", "sfxbg", "musicbg"], sample_rate=48000)
```

The wrapper accepts `torch.Tensor`, `numpy.ndarray`, or an audio file path as input.