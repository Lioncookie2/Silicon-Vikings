"""
Post-prosess: klasse-vekter og regime-baserte vekter pa sannsynlighetsrader (N,6).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .competition_metrics import CLASS_INDEX_NAMES, NUM_CLASSES

# Indekser i FEATURE_NAMES / forste del av settlement_port_radius
_IDX_COASTAL = 1
_IDX_DIST = 2
_IDX_HAS_SETT = 3
_IDX_N_MT = 5
_IDX_N_ST = 7


def apply_class_weights(probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    probs: (N, 6) eller (H,W,6), weights: (6,) positive.
    q'_k propto p_k * w_k, radvis renormalisert.
    """
    w = np.asarray(weights, dtype=np.float64).reshape(1, NUM_CLASSES)
    p = np.asarray(probs, dtype=np.float64)
    flat = p.ndim == 2
    if not flat:
        h, w_, k = p.shape
        p = p.reshape(-1, k)
    x = p * w
    s = np.sum(x, axis=1, keepdims=True)
    s = np.where(s > 0, s, 1.0)
    out = x / s
    if not flat:
        out = out.reshape(h, w_, NUM_CLASSES)
    return out.astype(np.float64)


def regime_from_feature_row(row: np.ndarray) -> str:
    """Grovt regime fra forste ~8 features (samme semantikk som FEATURE_NAMES)."""
    r = np.asarray(row, dtype=np.float64).ravel()
    if r.shape[0] < 8:
        return "default"
    coastal = r[_IDX_COASTAL] > 0.5
    has_s = r[_IDX_HAS_SETT] > 0.5
    dist = float(r[_IDX_DIST])
    n_mt = float(r[_IDX_N_MT])
    n_st = float(r[_IDX_N_ST])
    if n_mt >= 2.0:
        return "mountain_near"
    if coastal and (has_s or dist < 5.0):
        return "coastal_near_settlement"
    if has_s or dist < 8.0:
        return "inland_near_settlement"
    if n_st >= 2.0:
        return "high_density_settlement"
    if dist > 15.0:
        return "inland_far"
    return "default"


def apply_regime_class_weights(
    probs: np.ndarray,
    feature_rows: np.ndarray,
    weights_by_regime: dict[str, list[float]],
    default_weights: np.ndarray,
    *,
    min_samples_learned: int = 500,
    sample_counts: dict[str, int] | None = None,
) -> np.ndarray:
    """
    probs (N,6), feature_rows (N, F). Bruk regime-spesifikke vekter hvis sample_counts[regime] >= min.
    """
    p = np.asarray(probs, dtype=np.float64)
    X = np.asarray(feature_rows, dtype=np.float64)
    n = len(p)
    out = np.empty_like(p)
    dw = np.asarray(default_weights, dtype=np.float64)
    for i in range(n):
        reg = regime_from_feature_row(X[i])
        w = weights_by_regime.get(reg)
        if w is not None and sample_counts is not None:
            if sample_counts.get(reg, 0) >= min_samples_learned:
                out[i] = apply_class_weights(p[i : i + 1], np.asarray(w))[0]
                continue
        out[i] = apply_class_weights(p[i : i + 1], dw)[0]
    return out


def load_calibration_json(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data


def weights_vector_from_json(obj: dict[str, Any]) -> np.ndarray:
    wdict = obj.get("weights") or obj.get("class_weights")
    if not wdict:
        raise ValueError("calibration JSON mangler 'weights'")
    return np.asarray([float(wdict[k]) for k in CLASS_INDEX_NAMES], dtype=np.float64)


def apply_saved_calibration_json(pred_flat: np.ndarray, feature_rows: np.ndarray, path: Path) -> np.ndarray:
    """Bruk utdata fra calibrate_probs_historical (global eller regime)."""
    obj = load_calibration_json(path)
    mode = obj.get("mode", "global")
    if mode == "global":
        w = weights_vector_from_json(obj)
        return apply_class_weights(pred_flat, w)
    default_w = np.asarray([obj["default_weights"][k] for k in CLASS_INDEX_NAMES], dtype=np.float64)
    wbr = obj["weights_by_regime"]
    counts = {k: int(obj.get("regime_train_counts", {}).get(k, 0)) for k in wbr}
    wb = {r: np.asarray([wbr[r][k] for k in CLASS_INDEX_NAMES], dtype=np.float64) for r in wbr}
    min_n = int(obj.get("min_regime_samples", 500))
    out = np.empty_like(pred_flat)
    for i in range(len(pred_flat)):
        reg = regime_from_feature_row(feature_rows[i])
        wvec = wb.get(reg)
        if wvec is not None and counts.get(reg, 0) >= min_n:
            w = wvec
        else:
            w = default_w
        out[i] = apply_class_weights(pred_flat[i : i + 1], w)[0]
    return out
