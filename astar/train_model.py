"""
Tren global modell pa historical Astar-celledata (completed rounds).

Modeller: random_forest, logistic_regression, gradient_boosting

Kjor:
  PYTHONPATH=. python -m astar.train_model
  PYTHONPATH=. python -m astar.train_model --compare-all
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from .cell_features import (
    FEATURE_SET_FULL,
    FEATURE_SET_SETTLEMENT_PORT,
    FEATURE_SET_SETTLEMENT_PORT_RADIUS,
    all_feature_names,
    infer_feature_set_from_feature_names,
)
from .global_model_paths import (
    best_global_model_joblib,
    best_global_model_meta,
    legacy_global_model_joblib,
    legacy_global_model_meta,
    models_dir,
)
from .historical_cells_loader import load_historical_cells, parse_round_ids_csv
from .sklearn_utils import predict_proba_fixed6

COMPARE_MODELS = ("random_forest", "logistic_regression", "gradient_boosting")
CE_TIE_EPS = 1e-4


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_training_arrays(
    *,
    root: Path,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], str, Path | None, dict | None]:
    """
    Returnerer X, y_arg, y_prob, feature_names, feature_set, dataset_ref_path, historical_stats.
    dataset_ref_path er npz-sti eller None ved inline historical.
    historical_stats er load-stats dict eller None.
    """
    first_n = args.first_n_rounds if args.first_n_rounds is not None else args.max_rounds
    use_hist = (
        args.round_ids is not None
        or args.latest_n_rounds is not None
        or first_n is not None
    )
    if use_hist:
        hist = args.historical_root or (root / "astar" / "data" / "historical")
        X, y_arg, y_prob, _meta, stats = load_historical_cells(
            hist,
            neighbor_radius=args.neighbor_radius,
            feature_set=args.feature_set,
            round_ids=parse_round_ids_csv(args.round_ids),
            latest_n_rounds=args.latest_n_rounds,
            first_n_rounds=first_n,
        )
        fnames = all_feature_names(args.feature_set)
        print("=== train_model (data fra historical, ikke npz) ===")
        print(f"filter: {stats['filter_mode']}")
        print(
            f"runder: {len(stats['round_ids_used'])}  seeds: {stats['seeds_loaded']}  "
            f"celler: {stats['cells_loaded']}"
        )
        print(f"round_ids: {stats['round_ids_used']}")
        print(f"feature_set: {args.feature_set}  n_features: {X.shape[1]}  neighbor_radius: {args.neighbor_radius}")
        return X, y_arg, y_prob, fnames, args.feature_set, None, stats

    dataset = args.dataset or (root / "astar" / "data" / "datasets" / "historical_cells.npz")
    data = np.load(dataset, allow_pickle=True)
    X = data["X"]
    y_arg = data["y_argmax"]
    y_prob = data["y_prob"]
    feature_names = [str(x) for x in data["feature_names"].tolist()]
    if "feature_set" in data.files:
        feature_set = str(np.asarray(data["feature_set"]).reshape(-1)[0])
    else:
        feature_set = infer_feature_set_from_feature_names(feature_names)
    print("=== train_model (npz) ===")
    print(f"dataset: {dataset}")
    print(f"celler: {len(X)}  n_features: {X.shape[1]}  feature_set: {feature_set}")
    return X, y_arg, y_prob, feature_names, feature_set, dataset, None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train global Astar model from historical dataset.")
    p.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Default: astar/data/datasets/historical_cells.npz. Ignoreres hvis --round-ids / --latest-n-rounds / --first-n-rounds / --max-rounds er satt (da lastes fra historical).",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="Default: astar/data/models")
    p.add_argument(
        "--model",
        choices=COMPARE_MODELS,
        default="random_forest",
        help="Enkeltmodell (ignoreres med --compare-all).",
    )
    p.add_argument(
        "--compare-all",
        action="store_true",
        help="Tren random_forest, logistic_regression, gradient_boosting; velg beste etter lavest val CE; skriv model_comparison.json og best_global_model*.joblib.",
    )
    p.add_argument(
        "--promote-best",
        action="store_true",
        help="Etter enkeltmodell-trening: kopier til best_global_model.joblib (valgfritt; --compare-all gjør dette automatisk for vinner).",
    )
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--historical-root",
        type=Path,
        default=None,
        help="Brukes ved runde-filter (sammen med --latest-n-rounds / --round-ids / ...).",
    )
    p.add_argument(
        "--round-ids",
        type=str,
        default=None,
        help="Kommaseparerte round UUID-er: last direkte fra historical (hopper over npz).",
    )
    p.add_argument(
        "--latest-n-rounds",
        type=int,
        default=None,
        help="Siste N fullforte runder fra historical (hopper over npz).",
    )
    p.add_argument("--first-n-rounds", type=int, default=None, help="Eldste N runder (historical inline).")
    p.add_argument("--max-rounds", type=int, default=None, help="Alias for --first-n-rounds.")
    p.add_argument(
        "--neighbor-radius",
        type=int,
        default=1,
        help="Nabosirkel for features ved historical inline (skal matche trening).",
    )
    p.add_argument(
        "--feature-set",
        choices=(
            FEATURE_SET_FULL,
            FEATURE_SET_SETTLEMENT_PORT,
            FEATURE_SET_SETTLEMENT_PORT_RADIUS,
        ),
        default=FEATURE_SET_FULL,
        help="Ved historical inline; ved npz utledes normalt fra lagrede feature_names.",
    )
    return p.parse_args()


def _make_model(name: str):
    if name == "logistic_regression":
        # sklearn>=1.8: multi_class fjernet (multinomial er standard for lbfgs ved >2 klasser)
        return LogisticRegression(max_iter=400, solver="lbfgs")
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=42)
    return RandomForestClassifier(
        n_estimators=250,
        max_depth=20,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )


def _cross_entropy(y_prob_true: np.ndarray, y_prob_pred: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(y_prob_pred, eps, 1.0)
    ce = -np.sum(y_prob_true * np.log(p), axis=1)
    return float(np.mean(ce))


def _val_class_counts(y_val: np.ndarray, n_classes: int = 6) -> dict[str, int]:
    return {str(i): int(np.sum(y_val == i)) for i in range(n_classes)}


def _build_meta(
    *,
    sklearn_model_kind: str,
    dataset: str | Path | None,
    n_cells: int,
    feature_names: list[str],
    feature_set: str,
    neighbor_radius: int,
    artifact_path: Path,
    extra: dict | None = None,
) -> dict:
    m = {
        "sklearn_model_kind": sklearn_model_kind,
        "dataset": str(dataset) if dataset is not None else "historical:inline",
        "n_cells_trained": int(n_cells),
        "feature_names": feature_names,
        "feature_set": feature_set,
        "neighbor_radius": int(neighbor_radius),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "artifact_path": str(artifact_path),
    }
    if extra:
        m.update(extra)
    return m


def _save_trained_model(
    model,
    model_name: str,
    *,
    out_dir: Path,
    dataset: Path | None,
    X: np.ndarray,
    y_arg: np.ndarray,
    feature_names: list[str],
    feature_set: str,
    neighbor_radius: int,
) -> tuple[Path, Path]:
    """Full fit pa hele X, y_arg; skriv joblib + {name}_meta.json."""
    model.fit(X, y_arg)
    artifact = out_dir / f"{model_name}.joblib"
    meta_path = out_dir / f"{model_name}_meta.json"
    joblib.dump(model, artifact)
    meta = _build_meta(
        sklearn_model_kind=model_name,
        dataset=dataset,
        n_cells=len(X),
        feature_names=feature_names,
        feature_set=feature_set,
        neighbor_radius=neighbor_radius,
        artifact_path=artifact,
    )
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return artifact, meta_path


def _copy_to_best(source_joblib: Path, source_meta: Path, out_dir: Path, *, winner_name: str, comparison_path: Path) -> None:
    dst_j = out_dir / "best_global_model.joblib"
    dst_m = out_dir / "best_global_model_meta.json"
    shutil.copyfile(source_joblib, dst_j)
    base = json.loads(source_meta.read_text(encoding="utf-8"))
    base["promoted_as_best"] = True
    base["best_source_model"] = winner_name
    base["model_comparison_json"] = str(comparison_path)
    base["artifact_path"] = str(dst_j)
    dst_m.write_text(json.dumps(base, indent=2, ensure_ascii=False), encoding="utf-8")


def _copy_to_legacy(source_joblib: Path, source_meta: Path, out_dir: Path) -> None:
    """Bakoverkompatibilitet med eldre dokumentasjon (global_model.joblib)."""
    shutil.copyfile(source_joblib, out_dir / "global_model.joblib")
    shutil.copyfile(source_meta, out_dir / "global_model_meta.json")


def run_compare_all(
    *,
    dataset: Path | None,
    dataset_label: str,
    out_dir: Path,
    X: np.ndarray,
    y_arg: np.ndarray,
    y_prob: np.ndarray,
    feature_names: list[str],
    feature_set: str,
    test_size: float,
    random_state: int,
    historical_stats: dict | None = None,
    neighbor_radius: int = 1,
) -> None:
    X_train, X_val, y_train, y_val, _yp_train, yp_val = train_test_split(
        X,
        y_arg,
        y_prob,
        test_size=test_size,
        random_state=random_state,
        stratify=y_arg,
    )

    rows: list[dict] = []
    saved_paths: dict[str, tuple[Path, Path]] = {}

    for name in COMPARE_MODELS:
        model = _make_model(name)
        model.fit(X_train, y_train)
        pred_cls = model.predict(X_val)
        acc = float(accuracy_score(y_val, pred_cls))
        pred_prob = predict_proba_fixed6(model, X_val)
        ce = _cross_entropy(yp_val, pred_prob)
        rep = classification_report(y_val, pred_cls, output_dict=True, zero_division=0)
        macro_f1 = float(rep.get("macro avg", {}).get("f1-score", 0.0))

        artifact, meta_path = _save_trained_model(
            model,
            name,
            out_dir=out_dir,
            dataset=dataset,
            X=X,
            y_arg=y_arg,
            feature_names=feature_names,
            feature_set=feature_set,
            neighbor_radius=neighbor_radius,
        )
        saved_paths[name] = (artifact, meta_path)

        rows.append(
            {
                "model": name,
                "val_cross_entropy_against_ground_truth_prob": ce,
                "val_accuracy_argmax": acc,
                "val_macro_f1": macro_f1,
                "val_class_counts_argmax": _val_class_counts(y_val),
                "classification_report": rep,
                "artifact_joblib": str(artifact),
                "artifact_meta": str(meta_path),
            }
        )

    rows_sorted = sorted(rows, key=lambda r: r["val_cross_entropy_against_ground_truth_prob"])
    winner = rows_sorted[0]["model"]
    wce = rows_sorted[0]["val_cross_entropy_against_ground_truth_prob"]
    second_ce = rows_sorted[1]["val_cross_entropy_against_ground_truth_prob"] if len(rows_sorted) > 1 else None

    tie_or_close_note: str | None = None
    if second_ce is not None and abs(second_ce - wce) <= CE_TIE_EPS:
        tie_or_close_note = (
            f"Topp-modellene er svært like på CE (diff {abs(second_ce - wce):.6g} ≤ {CE_TIE_EPS}); "
            f"primærvalg {winner}, men vurder også {rows_sorted[1]['model']}."
        )
    elif second_ce is not None and abs(second_ce - wce) < 0.02:
        tie_or_close_note = (
            f"Liten CE-forskjell mellom topp to ({abs(second_ce - wce):.6g}); "
            "accuracy er sekundær — behold fokus på CE."
        )

    comparison_path = out_dir / "model_comparison.json"
    comparison = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_label,
        "feature_set": feature_set,
        "historical_load_stats": historical_stats,
        "test_size": test_size,
        "random_state": random_state,
        "n_train_cells": int(len(X_train)),
        "n_val_cells": int(len(X_val)),
        "selection_metric": "val_cross_entropy_against_ground_truth_prob (lavere er bedre)",
        "ranked_by_cross_entropy": [r["model"] for r in rows_sorted],
        "winner": winner,
        "winner_val_cross_entropy": wce,
        "winner_val_accuracy_argmax": rows_sorted[0]["val_accuracy_argmax"],
        "models": rows,
        "note_if_similar_ce": tie_or_close_note,
        "best_global_model_joblib": str(out_dir / "best_global_model.joblib"),
        "ensemble_weighted_average": None,
        "ensemble_todo": (
            "Valgfri vektet ensemble (f.eks. 0.5 RF + 0.3 GB + 0.2 LR) er ikke implementert; "
            "kan legges inn i predict senere med lav risiko."
        ),
    }
    comparison_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")

    w_art, w_meta = saved_paths[winner]
    _copy_to_best(w_art, w_meta, out_dir, winner_name=winner, comparison_path=comparison_path)
    _copy_to_legacy(w_art, w_meta, out_dir)

    print(json.dumps({"model_comparison": str(comparison_path), "winner": winner, "val_ce": wce}, indent=2))
    if tie_or_close_note:
        print(tie_or_close_note)
    print(f"Promoted best -> {out_dir / 'best_global_model.joblib'}")
    print(f"Legacy copy -> {out_dir / 'global_model.joblib'}")


def main() -> None:
    args = parse_args()
    root = _repo_root()
    out_dir = args.out_dir or models_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y_arg, y_prob, feature_names, feature_set, dataset_path, hist_stats = _load_training_arrays(root=root, args=args)
    dataset_label = str(dataset_path) if dataset_path is not None else f"historical:inline ({hist_stats['filter_mode'] if hist_stats else '?'})"

    nr = int(hist_stats["neighbor_radius"]) if hist_stats else int(args.neighbor_radius)

    if args.compare_all:
        run_compare_all(
            dataset=dataset_path,
            dataset_label=dataset_label,
            out_dir=out_dir,
            X=X,
            y_arg=y_arg,
            y_prob=y_prob,
            feature_names=feature_names,
            feature_set=feature_set,
            test_size=args.test_size,
            random_state=args.random_state,
            historical_stats=hist_stats,
            neighbor_radius=nr,
        )
        return

    X_train, X_val, y_train, y_val, _yp_train, yp_val = train_test_split(
        X,
        y_arg,
        y_prob,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_arg,
    )

    model = _make_model(args.model)
    model.fit(X_train, y_train)

    pred_cls = model.predict(X_val)
    acc = float(accuracy_score(y_val, pred_cls))

    pred_prob = predict_proba_fixed6(model, X_val)

    ce = _cross_entropy(yp_val, pred_prob)

    artifact, meta_path = _save_trained_model(
        model,
        args.model,
        out_dir=out_dir,
        dataset=dataset_path,
        X=X,
        y_arg=y_arg,
        feature_names=feature_names,
        feature_set=feature_set,
        neighbor_radius=nr,
    )

    _copy_to_legacy(artifact, meta_path, out_dir)

    if args.promote_best:
        comp_stub = out_dir / "model_comparison.json"
        if not comp_stub.is_file():
            comp_stub.write_text(
                json.dumps(
                    {
                        "note": "Opprettet av train_model --promote-best uten full --compare-all",
                        "promoted_model": args.model,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        _copy_to_best(artifact, meta_path, out_dir, winner_name=args.model, comparison_path=comp_stub)
        print(f"Promoted -> {best_global_model_joblib()}")

    report = {
        "model": args.model,
        "dataset": dataset_label,
        "feature_set": feature_set,
        "historical_load_stats": hist_stats,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_features": int(X.shape[1]),
        "feature_names": feature_names,
        "val_accuracy_argmax": acc,
        "val_cross_entropy_against_ground_truth_prob": ce,
        "classification_report": classification_report(y_val, pred_cls, output_dict=True, zero_division=0),
        "artifact_joblib": str(artifact),
        "artifact_meta": str(meta_path),
        "legacy_global_model_joblib": str(legacy_global_model_joblib()),
        "notes": {
            "global_model": "Trent pa completed historical rounds.",
            "round_specific": "Aktiv runde: bruk explore kun til kalibrering, ikke som supervised labels.",
            "best_model": "Kjor train_model --compare-all for a sette best_global_model.joblib etter CE.",
        },
    }

    report_path = out_dir / f"training_report_{args.model}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {report_path}")
    print(f"Saved {artifact}")
    print(
        json.dumps(
            {
                "val_accuracy_argmax": acc,
                "val_cross_entropy_against_ground_truth_prob": ce,
                "n_train": int(len(X_train)),
                "n_val": int(len(X_val)),
            },
            indent=2,
        )
    )
    print(f"Legacy mirror: {legacy_global_model_joblib()}")
    if not args.promote_best:
        print("Tips: python -m astar.train_model --compare-all  (eller --promote-best) for predict auto-best.")


if __name__ == "__main__":
    main()
