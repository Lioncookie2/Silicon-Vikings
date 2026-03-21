"""
Leave-one-round-out (og valgfri holdout) evaluering mot historiske fullforte runder.

Ligner konkurranse bedre enn tilfeldig cellesplit: for hver test-runde trenes modellen
kun pa andre runders celler, og det evalueres pa hele test-runden.

Metrikker per fold: mean cross-entropy mot y_prob, mean KL(p_true||p_pred),
argmax-accuracy, Brier (distribusjon), enkle confidence- vs sannhets-statistikker.

Kjor:
  PYTHONPATH=. python -m astar.evaluate_historical_rounds
  PYTHONPATH=. python -m astar.evaluate_historical_rounds --ablation terrain_settlement
  PYTHONPATH=. python -m astar.evaluate_historical_rounds --mode holdout-last-n --holdout-n 2
  PYTHONPATH=. python -m astar.evaluate_historical_rounds --latest-n-rounds 3 --feature-set settlement_port
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .cell_features import (
    FEATURE_NAMES,
    FEATURE_SET_FULL,
    FEATURE_SET_SETTLEMENT_PORT,
    FEATURE_SET_SETTLEMENT_PORT_RADIUS,
    all_feature_names,
)
from .historical_cells_loader import (
    audit_settlement_keys_in_historical,
    load_historical_cells,
    parse_round_ids_csv,
)
from .sklearn_utils import predict_proba_fixed6
from .train_model import _make_model

NUM_CLASSES = 6

# Samme vekter som predict global_ensemble (trent pa fold, ikke joblib)
ENSEMBLE_SPEC: list[tuple[str, float]] = [
    ("gradient_boosting", 0.6),
    ("logistic_regression", 0.25),
    ("random_forest", 0.15),
]

BASE_ABLATION_SLICES: dict[str, list[int]] = {
    "full": list(range(len(FEATURE_NAMES))),
    "terrain_only": [0],
    "terrain_coastal": [0, 1],
    "terrain_settlement": [0, 1, 2, 3],
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ablation_column_indices(ablation: str, feature_set: str) -> list[int]:
    """Kolonner i X etter load_historical_cells (full bredde for valgt feature_set)."""
    names = all_feature_names(feature_set)
    if ablation == "full":
        return list(range(len(names)))
    if feature_set != FEATURE_SET_FULL:
        raise SystemExit(
            f"--ablation {ablation} støttes kun med --feature-set {FEATURE_SET_FULL}; "
            f"bruk --ablation full for {FEATURE_SET_SETTLEMENT_PORT}."
        )
    cols = BASE_ABLATION_SLICES.get(ablation)
    if cols is None:
        raise SystemExit(f"ukjent ablation: {ablation}")
    if max(cols, default=-1) >= len(FEATURE_NAMES):
        raise SystemExit("intern feil: ablation-indekser utenfor basis-features")
    return cols


def mean_cross_entropy(y_true_prob: np.ndarray, y_pred_prob: np.ndarray) -> float:
    eps = 1e-12
    q = np.clip(y_pred_prob, eps, 1.0)
    return float(np.mean(-np.sum(y_true_prob * np.log(q), axis=1)))


def mean_kl_forward(y_true_prob: np.ndarray, y_pred_prob: np.ndarray) -> float:
    """Mean KL(p || q) per celle; p = ground truth, q = pred."""
    eps = 1e-12
    p = np.clip(y_true_prob, eps, 1.0)
    q = np.clip(y_pred_prob, eps, 1.0)
    kl = np.sum(p * (np.log(p) - np.log(q)), axis=1)
    return float(np.mean(kl))


def mean_brier_prob(y_true_prob: np.ndarray, y_pred_prob: np.ndarray) -> float:
    return float(np.mean(np.sum((y_true_prob - y_pred_prob) ** 2, axis=1)))


def argmax_accuracy(y_true_prob: np.ndarray, y_pred_prob: np.ndarray) -> float:
    ta = np.argmax(y_true_prob, axis=1)
    pa = np.argmax(y_pred_prob, axis=1)
    return float(np.mean(ta == pa))


def confidence_diagnostics(y_true_prob: np.ndarray, y_pred_prob: np.ndarray) -> dict[str, float]:
    """Enkle signaler pa overkonfidens (hoy max(q) vs riktig klasse-masse)."""
    max_q = np.max(y_pred_prob, axis=1)
    max_p = np.max(y_true_prob, axis=1)
    true_cls = np.argmax(y_true_prob, axis=1)
    prob_on_true = y_pred_prob[np.arange(len(y_pred_prob)), true_cls]
    return {
        "mean_max_predicted_prob": float(np.mean(max_q)),
        "mean_max_true_prob": float(np.mean(max_p)),
        "mean_pred_prob_on_true_class": float(np.mean(prob_on_true)),
        "mean_gap_maxq_minus_prob_on_true": float(np.mean(max_q - prob_on_true)),
    }


def train_predict_single(model_name: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    m = _make_model(model_name)
    m.fit(X_train, y_train)
    return predict_proba_fixed6(m, X_test)


def train_predict_ensemble(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    preds: list[np.ndarray] = []
    weights: list[float] = []
    for name, w in ENSEMBLE_SPEC:
        m = _make_model(name)
        m.fit(X_train, y_train)
        preds.append(predict_proba_fixed6(m, X_test))
        weights.append(w)
    wsum = sum(weights)
    wn = [wi / wsum for wi in weights]
    acc = np.zeros((len(X_test), NUM_CLASSES), dtype=np.float64)
    for p, wi in zip(preds, wn):
        acc += wi * p
    rs = np.sum(acc, axis=1, keepdims=True)
    rs = np.where(rs > 0, rs, 1.0)
    return acc / rs


def evaluate_one_split(
    y_prob_test: np.ndarray,
    pred_prob: np.ndarray,
) -> dict[str, Any]:
    ce = mean_cross_entropy(y_prob_test, pred_prob)
    kl = mean_kl_forward(y_prob_test, pred_prob)
    br = mean_brier_prob(y_prob_test, pred_prob)
    acc = argmax_accuracy(y_prob_test, pred_prob)
    diag = confidence_diagnostics(y_prob_test, pred_prob)
    return {
        "mean_cross_entropy": ce,
        "mean_kl_forward": kl,
        "mean_brier": br,
        "argmax_accuracy": acc,
        "n_cells": int(len(y_prob_test)),
        **diag,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Round-aware LOO eval pa historical completed rounds.")
    p.add_argument("--historical-root", type=Path, default=None, help="Default: astar/data/historical")
    p.add_argument("--out-dir", type=Path, default=None, help="Default: astar/data/evals")
    p.add_argument("--neighbor-radius", type=int, default=1)
    p.add_argument(
        "--round-ids",
        type=str,
        default=None,
        help="Kommaseparerte round UUID-er (subset av historical).",
    )
    p.add_argument("--latest-n-rounds", type=int, default=None, help="Siste N runder (round_number).")
    p.add_argument("--first-n-rounds", type=int, default=None, help="Eldste N runder.")
    p.add_argument("--max-rounds", type=int, default=None, help="Alias for --first-n-rounds.")
    p.add_argument(
        "--feature-set",
        choices=(
            FEATURE_SET_FULL,
            FEATURE_SET_SETTLEMENT_PORT,
            FEATURE_SET_SETTLEMENT_PORT_RADIUS,
        ),
        default=FEATURE_SET_FULL,
    )
    p.add_argument(
        "--mode",
        choices=("loo", "holdout-last-n"),
        default="loo",
        help="loo = leave-one-round-out; holdout-last-n = test siste N runder (sortert pa round_number).",
    )
    p.add_argument("--holdout-n", type=int, default=2, help="Antall siste runder som test ved holdout-last-n.")
    p.add_argument(
        "--ablation",
        choices=tuple(BASE_ABLATION_SLICES.keys()),
        default="full",
        help="Feature-deling for ablation (kun med --feature-set full, unntatt ablation=full).",
    )
    p.add_argument(
        "--models",
        type=str,
        default="random_forest,logistic_regression,gradient_boosting",
        help="Kommaseparert. Legg til global_ensemble for fersk 0.6GB+0.25LR+0.15RF pa hver fold.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = _repo_root()
    hist = args.historical_root or (root / "astar" / "data" / "historical")
    out_dir = args.out_dir or (root / "astar" / "data" / "evals")
    out_dir.mkdir(parents=True, exist_ok=True)

    first_n = args.first_n_rounds if args.first_n_rounds is not None else args.max_rounds

    feature_cols = ablation_column_indices(args.ablation, args.feature_set)
    all_names = all_feature_names(args.feature_set)
    feat_names_used = [all_names[i] for i in feature_cols]

    settlement_audit = audit_settlement_keys_in_historical(hist)

    X_full, y_arg, y_prob, meta_rows, stats = load_historical_cells(
        hist,
        neighbor_radius=args.neighbor_radius,
        feature_set=args.feature_set,
        round_ids=parse_round_ids_csv(args.round_ids),
        latest_n_rounds=args.latest_n_rounds,
        first_n_rounds=first_n,
    )
    X = np.ascontiguousarray(X_full[:, feature_cols], dtype=np.float32)

    rvec = np.asarray([m["round_id"] for m in meta_rows], dtype=object)
    svec = np.asarray([m["seed_index"] for m in meta_rows], dtype=np.int64)
    round_order = [s["round_id"] for s in stats["round_summaries"]]
    round_metas = list(stats["round_summaries"])

    print("=== evaluate_historical_rounds ===")
    print(f"mode={args.mode}  feature_set={args.feature_set}  ablation={args.ablation}")
    print(f"filter: {stats['filter_mode']}")
    print(
        f"runder: {len(stats['round_ids_used'])}  seeds: {stats['seeds_loaded']}  "
        f"celler: {stats['cells_loaded']}  n_features_eval: {X.shape[1]}"
    )
    print(f"round_ids: {stats['round_ids_used']}")
    print(f"settlement audit rich_stats={settlement_audit.get('rich_economic_stats_present')}")

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in model_list:
        if m not in ("random_forest", "logistic_regression", "gradient_boosting", "global_ensemble"):
            raise SystemExit(f"Ukjent modell: {m}")

    if len(round_order) < 2 and args.mode == "loo":
        raise SystemExit("LOO krever minst 2 runder med analyse-data.")

    csv_rows: list[dict[str, Any]] = []
    per_fold_records: list[dict[str, Any]] = []
    ce_by_model: dict[str, list[float]] = defaultdict(list)
    kl_by_model: dict[str, list[float]] = defaultdict(list)

    if args.mode == "loo":
        for test_rid in round_order:
            train_mask = rvec != test_rid
            test_mask = rvec == test_rid
            if np.sum(train_mask) == 0 or np.sum(test_mask) == 0:
                continue
            X_tr, y_tr = X[train_mask], y_arg[train_mask]
            X_te = X[test_mask]
            y_prob_te = y_prob[test_mask]
            s_te = svec[test_mask]

            fold_models: dict[str, dict[str, Any]] = {}
            for mname in model_list:
                if mname == "global_ensemble":
                    pred = train_predict_ensemble(X_tr, y_tr, X_te)
                else:
                    pred = train_predict_single(mname, X_tr, y_tr, X_te)
                metrics = evaluate_one_split(y_prob_te, pred)
                per_seed: dict[str, dict[str, Any]] = {}
                for sid in np.unique(s_te):
                    sm = s_te == sid
                    if np.sum(sm) == 0:
                        continue
                    per_seed[str(int(sid))] = evaluate_one_split(y_prob_te[sm], pred[sm])
                fold_models[mname] = {**metrics, "per_seed": per_seed}
                ce_by_model[mname].append(metrics["mean_cross_entropy"])
                kl_by_model[mname].append(metrics["mean_kl_forward"])
                csv_rows.append(
                    {
                        "mode": "loo",
                        "test_round_id": test_rid,
                        "seed_scope": "all_seeds_in_round",
                        "model": mname,
                        **metrics,
                    }
                )
                for sid_str, smetrics in per_seed.items():
                    csv_rows.append(
                        {
                            "mode": "loo",
                            "test_round_id": test_rid,
                            "seed_scope": f"seed_{sid_str}",
                            "model": mname,
                            **smetrics,
                        }
                    )

            per_fold_records.append(
                {
                    "test_round_id": test_rid,
                    "n_train_cells": int(np.sum(train_mask)),
                    "n_test_cells": int(np.sum(test_mask)),
                    "models": fold_models,
                }
            )
            print(
                f"LOO test_round={test_rid[:8]}... n_test={int(np.sum(test_mask))} "
                f"CE_gb={fold_models.get('gradient_boosting', {}).get('mean_cross_entropy', 'n/a')}"
            )

    else:
        sorted_meta = sorted(round_metas, key=lambda d: (d.get("round_number", 0), d["round_id"]))
        if args.holdout_n >= len(sorted_meta):
            raise SystemExit("--holdout-n ma være mindre enn antall runder.")
        hold_ids = {m["round_id"] for m in sorted_meta[-args.holdout_n :]}
        train_mask = np.array([rid not in hold_ids for rid in rvec])
        test_mask = ~train_mask
        X_tr, y_tr = X[train_mask], y_arg[train_mask]
        X_te = X[test_mask]
        y_prob_te = y_prob[test_mask]
        s_te = svec[test_mask]

        fold_models = {}
        for mname in model_list:
            if mname == "global_ensemble":
                pred = train_predict_ensemble(X_tr, y_tr, X_te)
            else:
                pred = train_predict_single(mname, X_tr, y_tr, X_te)
            metrics = evaluate_one_split(y_prob_te, pred)
            per_seed: dict[str, dict[str, Any]] = {}
            for sid in np.unique(s_te):
                sm = s_te == sid
                if np.sum(sm) == 0:
                    continue
                per_seed[str(int(sid))] = evaluate_one_split(y_prob_te[sm], pred[sm])
            fold_models[mname] = {**metrics, "per_seed": per_seed}
            ce_by_model[mname].append(metrics["mean_cross_entropy"])
            kl_by_model[mname].append(metrics["mean_kl_forward"])
            csv_rows.append(
                {
                    "mode": "holdout_last_n",
                    "test_round_id": ",".join(sorted(hold_ids))[:200],
                    "holdout_round_ids": list(hold_ids),
                    "seed_scope": "all_seeds_in_holdout",
                    "model": mname,
                    **metrics,
                }
            )
            for sid_str, smetrics in per_seed.items():
                csv_rows.append(
                    {
                        "mode": "holdout_last_n",
                        "test_round_id": ",".join(sorted(hold_ids))[:200],
                        "holdout_round_ids": list(hold_ids),
                        "seed_scope": f"seed_{sid_str}",
                        "model": mname,
                        **smetrics,
                    }
                )
        per_fold_records.append(
            {
                "test_round_id": "holdout:" + ",".join(sorted(hold_ids)),
                "holdout_round_ids": list(hold_ids),
                "n_train_cells": int(np.sum(train_mask)),
                "n_test_cells": int(np.sum(test_mask)),
                "models": fold_models,
            }
        )

    per_model_summary: dict[str, Any] = {}
    for mname in model_list:
        ces = ce_by_model[mname]
        kls = kl_by_model[mname]
        if not ces:
            continue
        per_model_summary[mname] = {
            "mean_cross_entropy_across_folds": float(np.mean(ces)),
            "std_cross_entropy_across_folds": float(np.std(ces)),
            "mean_kl_forward_across_folds": float(np.mean(kls)),
            "std_kl_forward_across_folds": float(np.std(kls)),
            "n_folds": len(ces),
        }

    winner = min(per_model_summary.keys(), key=lambda k: per_model_summary[k]["mean_cross_entropy_across_folds"])
    rk = sorted(
        per_model_summary.keys(),
        key=lambda k: per_model_summary[k]["mean_cross_entropy_across_folds"],
    )
    runner = rk[1] if len(rk) > 1 else None
    margin = None
    if runner:
        margin = (
            per_model_summary[runner]["mean_cross_entropy_across_folds"]
            - per_model_summary[winner]["mean_cross_entropy_across_folds"]
        )

    ce_std_winner = per_model_summary[winner]["std_cross_entropy_across_folds"]
    explore_hint = (
        "Vurder litt mer konservativ explore-blend (f.eks. 0.55/0.45) nar CE-varians over runder er hoy."
        if ce_std_winner > 0.15
        else "Moderat varians mellom runder — standard 0.6/0.4 blend er rimelig utgangspunkt."
    )

    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "holdout_n": args.holdout_n if args.mode == "holdout-last-n" else None,
        "historical_root": str(hist.resolve()),
        "load_stats": stats,
        "settlement_fields_audit": settlement_audit,
        "n_rounds": len(round_order),
        "round_ids": round_order,
        "feature_set": args.feature_set,
        "feature_ablation": args.ablation,
        "feature_columns_indices": feature_cols,
        "feature_names_used": feat_names_used,
        "models_evaluated": model_list,
        "per_fold": per_fold_records,
        "per_model_summary": per_model_summary,
        "winner_by_mean_cross_entropy": winner,
        "second_place": runner,
        "ce_margin_vs_second": margin,
        "recommendations": {
            "default_global_model": (
                winner
                if winner != "global_ensemble"
                else "bruk --baseline global_ensemble i predict, eller behold best_global_model (GB) som enkel default"
            ),
            "explore_blend_note": explore_hint,
            "feature_ablation": (
                "Sammenlign --ablation full vs terrain_settlement; hvis full klart bedre, behold alle nabo-features."
                if args.ablation == "full"
                else "Kjor ogsa --ablation full for a se om nabo-features lonner seg."
            ),
            "probability_quality": "Sjekk mean_gap_maxq_minus_prob_on_true: stor positiv verdi tyder pa overkonfidens i forhold til sann marginal.",
            "calibration_extension_todo": "Platt scaling / isotonic per klasse kan legges til senere; na rapporteres Brier og KL.",
            "settlement_stats_conclusion": settlement_audit.get("conclusion"),
        },
        "worst_rounds_by_model": {},
        "best_rounds_by_model": {},
    }

    if args.mode == "loo" and winner in model_list:
        fold_ces: list[tuple[str, float]] = []
        for rec in per_fold_records:
            rid = rec["test_round_id"]
            m = rec["models"].get(winner, {})
            if m:
                fold_ces.append((rid, m["mean_cross_entropy"]))
        if fold_ces:
            fold_ces.sort(key=lambda t: t[1])
            report["best_rounds_by_model"][winner] = {"lowest_ce_round": fold_ces[0][0], "ce": fold_ces[0][1]}
            report["worst_rounds_by_model"][winner] = {"highest_ce_round": fold_ces[-1][0], "ce": fold_ces[-1][1]}

    json_path = out_dir / "historical_round_eval.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = out_dir / "historical_round_eval_rows.csv"
    if csv_rows:
        all_keys: set[str] = set()
        for row in csv_rows:
            all_keys.update(row.keys())
        fields = sorted(all_keys)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(csv_rows)

    print(json.dumps({"wrote_json": str(json_path), "wrote_csv": str(csv_path), "winner": winner}, indent=2))
    print(per_model_summary)


if __name__ == "__main__":
    main()
