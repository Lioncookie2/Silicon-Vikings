"""
Valgfri vektet ensemble av lagrede globale sklearn-modeller (GB/LR/RF).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .sklearn_utils import predict_proba_fixed6

# Må matche filnavn fra train_model --compare-all
ENSEMBLE_FILES: list[tuple[str, str, float]] = [
    ("gradient_boosting", "gradient_boosting.joblib", 0.6),
    ("logistic_regression", "logistic_regression.joblib", 0.25),
    ("random_forest", "random_forest.joblib", 0.15),
]


def load_global_ensemble(models_dir: Path) -> tuple[list[tuple[str, Any, float]], int, str]:
    """
    Returnerer (liste (navn, modell, normalisert_vekt), neighbor_radius, logg-streng).
    Manglende filer hoppes over; vekter renormaliseres.
    """
    models_dir = Path(models_dir)
    loaded: list[tuple[str, Any, float]] = []
    for name, fname, w in ENSEMBLE_FILES:
        p = models_dir / fname
        if not p.is_file():
            continue
        loaded.append((name, joblib.load(p), w))
    if not loaded:
        raise FileNotFoundError(
            f"Ingen av ensemble-modellene finnes under {models_dir}. "
            "Kjor: python -m astar.train_model --compare-all"
        )
    wsum = sum(t[2] for t in loaded)
    norm: list[tuple[str, Any, float]] = [(n, m, wt / wsum) for n, m, wt in loaded]

    nr = 1
    meta_p = models_dir / "gradient_boosting_meta.json"
    if meta_p.is_file():
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            nr = int(meta.get("neighbor_radius", 1))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    nf_by_name: dict[str, int] = {}
    for n, m, _ in norm:
        nf = getattr(m, "n_features_in_", None)
        if nf is not None:
            nf_by_name[n] = int(nf)
    if len(set(nf_by_name.values())) > 1:
        raise ValueError(f"Ensemble-modeller har ulikt antall features: {nf_by_name}")

    desc = ", ".join(f"{n}:{w:.4f}" for n, _, w in norm)
    return norm, nr, desc


def ensemble_predict_proba_fixed6(
    models_weighted: list[tuple[str, Any, float]],
    X: np.ndarray,
) -> np.ndarray:
    """Vektet snitt av predict_proba_fixed6 per rad; renormaliser til sum 1."""
    n = len(X)
    acc = np.zeros((n, 6), dtype=np.float64)
    for _name, model, wt in models_weighted:
        acc += wt * predict_proba_fixed6(model, X)
    rs = np.sum(acc, axis=1, keepdims=True)
    rs = np.where(rs > 0, rs, 1.0)
    acc = acc / rs
    return acc
