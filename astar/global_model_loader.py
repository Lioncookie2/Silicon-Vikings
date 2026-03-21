"""Last global sklearn-modell + metadata for predict."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from .global_model_paths import (
    meta_path_for_joblib,
    resolve_default_model_artifacts,
)


def load_global_model_bundle(
    artifact_path: Path | None = None,
    meta_path: Path | None = None,
) -> tuple[Any, dict[str, Any], Path, Path, str]:
    """
    Returnerer (model, meta_dict, artifact_path, meta_path, load_source_label).

    Uten artifact_path: best_global_model.joblib hvis finnes, ellers global_model.joblib.
    """
    if artifact_path is not None:
        ap = Path(artifact_path)
        mp = Path(meta_path) if meta_path is not None else meta_path_for_joblib(ap)
        label = str(ap)
    else:
        ap, mp, label = resolve_default_model_artifacts()

    if not ap.is_file():
        raise FileNotFoundError(str(ap))
    model = joblib.load(ap)
    meta: dict[str, Any] = {}
    if mp.is_file():
        meta = json.loads(mp.read_text(encoding="utf-8"))
    return model, meta, ap, mp, label


def load_global_model_or_exit(
    artifact_path: Path | None = None,
    meta_path: Path | None = None,
) -> tuple[Any, dict[str, Any], Path, Path, str]:
    try:
        return load_global_model_bundle(artifact_path, meta_path)
    except FileNotFoundError as e:
        raise SystemExit(
            f"Mangler global modell: {e}\n"
            "Kjor (fra repo-roten, med PYTHONPATH=.):\n"
            "  python -m astar.fetch_historical\n"
            "  python -m astar.build_dataset\n"
            "  python -m astar.train_model --compare-all\n"
            "eller enkeltmodell:\n"
            "  python -m astar.train_model --model random_forest\n"
            "Forventet (auto): astar/data/models/best_global_model.joblib "
            "eller astar/data/models/global_model.joblib"
        ) from e
