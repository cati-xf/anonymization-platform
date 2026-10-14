# TUSS standalone inference

Minimal inference-only copy of **TUSS** (Task-aware Unified Source Separation),
from <https://github.com/merlresearch/unified-source-separation>
(commit `f7711c1120e00669d905a4fef9c8494a0f8c9662`).

- `nets/`, `utils/`, `loss_functions/` and `pretrained_models/` (the
  `tuss.medium.2-4src` checkpoint) are unmodified copies from that repository.
- `inference.py` is a thin wrapper (`PretrainedSeparator`) added on top of it.

Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL).
Released under AGPL-3.0-or-later (see `LICENSE.md`); some files adapted from
ESPnet under Apache-2.0 (see `LICENSES/Apache-2.0.md`).
