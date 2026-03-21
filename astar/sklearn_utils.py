"""Delt sklearn-hjelp for 6-klasse sannsynligheter."""
from __future__ import annotations

from typing import Any

import numpy as np

NUM_CLASSES = 6


def predict_proba_fixed6(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Map model.predict_proba til fast layout [0..5].
    Tåler at treningssett manglet noen klasser (subset av classes_).
    """
    n = len(X)
    out = np.zeros((n, NUM_CLASSES), dtype=np.float64)
    if not hasattr(model, "predict_proba"):
        pred_cls = model.predict(X)
        out[np.arange(n), pred_cls.astype(np.int64)] = 1.0
        return out

    raw = model.predict_proba(X)
    cls = getattr(model, "classes_", None)
    if cls is None:
        out[:, : raw.shape[1]] = raw
    else:
        for j, c in enumerate(cls):
            ci = int(c)
            if 0 <= ci < NUM_CLASSES:
                out[:, ci] = raw[:, j]

    row_sum = np.sum(out, axis=1, keepdims=True)
    zero = row_sum[:, 0] <= 0
    if np.any(zero):
        out[zero] = 1.0 / NUM_CLASSES
        row_sum = np.sum(out, axis=1, keepdims=True)
    out /= row_sum
    return out
