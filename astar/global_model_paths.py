"""Standard stier for lagret global modell."""
from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def models_dir() -> Path:
    return repo_root() / "astar" / "data" / "models"


def best_global_model_joblib() -> Path:
    return models_dir() / "best_global_model.joblib"


def best_global_model_meta() -> Path:
    return models_dir() / "best_global_model_meta.json"


def legacy_global_model_joblib() -> Path:
    """Eldre standard etter enkelt train_model-kjøring."""
    return models_dir() / "global_model.joblib"


def legacy_global_model_meta() -> Path:
    return models_dir() / "global_model_meta.json"


def default_model_joblib() -> Path:
    """Automatisk valg: best_global hvis finnes, ellers legacy global_model."""
    b = best_global_model_joblib()
    if b.is_file():
        return b
    return legacy_global_model_joblib()


def default_model_meta() -> Path:
    """Meta som hører til default_model_joblib()."""
    bj = best_global_model_joblib()
    if bj.is_file():
        return best_global_model_meta()
    return legacy_global_model_meta()


def resolve_default_model_artifacts() -> tuple[Path, Path, str]:
    """
    (joblib_path, meta_path, label_for_logging).
    Meta-fil brukes hvis den finnes; ellers tom dict ved lasting.
    """
    bj, bm = best_global_model_joblib(), best_global_model_meta()
    if bj.is_file():
        return bj, bm, "best_global_model.joblib"
    lj, lm = legacy_global_model_joblib(), legacy_global_model_meta()
    if lj.is_file():
        return lj, lm, "global_model.joblib (legacy)"
    raise FileNotFoundError(
        f"Mangler både {bj} og {lj}. Kjør train_model eller train_model --compare-all."
    )


def meta_path_for_joblib(artifact: Path) -> Path:
    """Forutsigbar meta-fil for et gitt .joblib-navn."""
    p = Path(artifact)
    stem = p.stem
    if stem == "best_global_model":
        return best_global_model_meta()
    if stem == "global_model":
        return legacy_global_model_meta()
    return p.with_name(f"{stem}_meta.json")
