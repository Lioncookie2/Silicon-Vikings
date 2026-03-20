"""
PyTorch 2.6+ defaults torch.load(weights_only=True). Ultralytics 8.1 .pt checkpoints
require weights_only=False for trusted official weights. Safe for competition weights.
"""
from __future__ import annotations

import torch

_orig_load = torch.load


def _patched_load(*args: object, **kwargs: object) -> object:
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False  # type: ignore[assignment]
    return _orig_load(*args, **kwargs)


torch.load = _patched_load  # type: ignore[assignment]
