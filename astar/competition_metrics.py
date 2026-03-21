"""
Konkurranse-lignende metrikker for sannsynlighets-tensorer (N,6).

Entropy-vektet KL: KL(p||q) per celle, vektet med Shannon-entropi H(p) for sann p.
Estimert score: 100 * exp(-3 * weighted_kl), klippet til [0, 100].

p = ground truth (historical y_prob), q = modell / blend / kalibrert prediksjon.
"""
from __future__ import annotations

from typing import Any

import numpy as np

NUM_CLASSES = 6
CLASS_INDEX_NAMES = ["empty", "settlement", "port", "ruin", "forest", "mountain"]
DEFAULT_SCORE_EXP_SCALE = 3.0


def kl_forward_per_row(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """KL(p || q) per rad; p,q shape (N, K)."""
    p = np.clip(p.astype(np.float64), eps, 1.0)
    q = np.clip(q.astype(np.float64), eps, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)), axis=1)


def entropy_per_row(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Shannon H(p) per rad."""
    p = np.clip(p.astype(np.float64), eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def mean_cross_entropy(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    q = np.clip(q.astype(np.float64), eps, 1.0)
    p = p.astype(np.float64)
    return float(np.mean(-np.sum(p * np.log(q), axis=1)))


def mean_kl_forward(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.mean(kl_forward_per_row(p, q)))


def entropy_weighted_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    sum_i H(p_i) * KL(p_i || q_i) / sum_i H(p_i).
    Unngår deling på null: hvis total entropi ~0, faller vi tilbake til mean KL.
    """
    h = entropy_per_row(p, eps=eps)
    kl = kl_forward_per_row(p, q, eps=eps)
    wsum = float(np.sum(h))
    if wsum < 1e-15:
        return float(np.mean(kl))
    return float(np.sum(h * kl) / wsum)


def competition_score(weighted_kl: float, *, scale: float = DEFAULT_SCORE_EXP_SCALE) -> float:
    s = 100.0 * float(np.exp(-scale * float(weighted_kl)))
    return float(np.clip(s, 0.0, 100.0))


def summarize_tensor_pair(
    p_flat: np.ndarray,
    q_flat: np.ndarray,
    *,
    score_scale: float = DEFAULT_SCORE_EXP_SCALE,
) -> dict[str, Any]:
    """p,q (N,6) normaliserte sannsynligheter."""
    wkl = entropy_weighted_kl(p_flat, q_flat)
    return {
        "n_cells": int(len(p_flat)),
        "mean_cross_entropy": mean_cross_entropy(p_flat, q_flat),
        "mean_kl_forward": mean_kl_forward(p_flat, q_flat),
        "entropy_weighted_kl": wkl,
        "estimated_competition_score": competition_score(wkl, scale=score_scale),
        "score_formula": f"100 * exp(-{score_scale} * entropy_weighted_kl), clipped [0,100]",
        "mean_entropy_of_truth": float(np.mean(entropy_per_row(p_flat))),
    }


def per_cell_metrics(p_flat: np.ndarray, q_flat: np.ndarray, eps: float = 1e-12) -> dict[str, np.ndarray]:
    """For debugging / analyse."""
    return {
        "kl": kl_forward_per_row(p_flat, q_flat, eps=eps),
        "entropy_p": entropy_per_row(p_flat, eps=eps),
    }
