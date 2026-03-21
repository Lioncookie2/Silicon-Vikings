"""
Lær klasse-vekter (og valgfritt regime-vekter) pa historical predictions (etter GB+blend+floor).

Tren pa runder utenom holdout; evaluer pa holdout. Skriver JSON for bruk i predict --calibration-json.

Valg av vekter styres av --objective (ce | weighted_kl | score). Anbefaling pa holdout
er primært entropy_weighted_kl (lavere bedre) og estimated_competition_score (hoyere bedre);
CE-forbedring alene flagges som «ikke bruk i produksjon» nar score/wKL forverres.

Kjor fra Silicon-Vikings:
  export PYTHONPATH=.
  python -m astar.calibrate_probs_historical --holdout-last-n 3
  python -m astar.calibrate_probs_historical --objective ce --holdout-last-n 3 --out-json astar/data/models/prob_calibration_ce.json
  python -m astar.calibrate_probs_historical --objective weighted_kl --holdout-last-n 3
  python -m astar.calibrate_probs_historical --objective score --holdout-last-n 3 --compare-objectives
  python -m astar.calibrate_probs_historical --mode regime --holdout-last-n 3
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from scipy.optimize import minimize

from .baseline import DEFAULT_EPS, apply_floor_and_renorm, build_terrain_prior, load_detail_json
from .cell_features import feature_matrix_for_seed, infer_feature_set_from_feature_names
from .competition_metrics import (
    CLASS_INDEX_NAMES,
    competition_score,
    entropy_weighted_kl,
    mean_cross_entropy,
    summarize_tensor_pair,
)
from .global_model_loader import load_global_model_bundle
from .historical_cells_loader import discover_historical_round_dirs, filter_round_dirs, parse_round_ids_csv
from .prob_calibration import apply_class_weights, regime_from_feature_row
from .sklearn_utils import predict_proba_fixed6

NUM_CLASSES = 6


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_calibration_explore(explore_dir: Path | None) -> dict[str, float]:
    if explore_dir is None:
        return {}
    p = Path(explore_dir) / "analysis_summary.json"
    if not p.is_file():
        return {}
    s = _load_json(p)
    cal = s.get("calibration_suggestion") or {}
    if not isinstance(cal, dict):
        return {}
    out: dict[str, float] = {}
    for k in ("coast_boost", "near_settlement_boost", "coast_near_settle_port_boost", "dynamic_ruin_weight"):
        if k in cal:
            try:
                out[k] = float(cal[k])
            except (TypeError, ValueError):
                pass
    return out


@dataclass
class SeedRec:
    round_id: str
    round_number: int
    y_true_flat: np.ndarray
    pred_flat: np.ndarray
    X_flat: np.ndarray


def iter_seed_recs(
    *,
    round_dirs: list[Path],
    model: object,
    feature_set: str,
    neighbor_radius: int,
    calibration: dict[str, float],
    global_w: float,
    explore_w: float,
    prob_floor: float,
) -> Iterator[SeedRec]:
    for rd in round_dirs:
        detail = load_detail_json(rd / "round_detail.json")
        rid = str(detail.get("id", rd.name))
        rnum = int(detail.get("round_number", 0))
        analysis_dir = rd / "analysis"
        if not analysis_dir.is_dir():
            continue
        for seed_path in sorted(analysis_dir.glob("seed_*.json")):
            item = _load_json(seed_path)
            gt = item.get("ground_truth")
            initial_grid = item.get("initial_grid")
            seed_index = int(item.get("seed_index", -1))
            if gt is None or initial_grid is None or seed_index < 0:
                continue
            initial_states = detail.get("initial_states", [])
            if seed_index >= len(initial_states):
                continue
            gt_np = np.asarray(gt, dtype=np.float64)
            grid_np = np.asarray(initial_grid, dtype=np.int32)
            if gt_np.ndim != 3 or gt_np.shape[-1] != NUM_CLASSES:
                continue
            h, w, _ = gt_np.shape
            if grid_np.shape != (h, w):
                continue
            row_sums = np.sum(gt_np, axis=-1, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1.0)
            y_true = (gt_np / row_sums).reshape(-1, NUM_CLASSES)
            st = initial_states[seed_index]
            settlements = st.get("settlements", [])
            X = feature_matrix_for_seed(
                grid_np,
                settlements,
                neighbor_radius=neighbor_radius,
                feature_set=feature_set,
            )
            global_flat = predict_proba_fixed6(model, X)
            global_hw = global_flat.reshape(h, w, NUM_CLASSES)
            explore_hw = build_terrain_prior(
                grid_np.tolist(),
                settlements,
                eps=DEFAULT_EPS,
                coast_boost=float(calibration.get("coast_boost", 0.35)),
                near_settlement_boost=float(calibration.get("near_settlement_boost", 0.5)),
                coast_near_settle_port_boost=float(calibration.get("coast_near_settle_port_boost", 0.28)),
                dynamic_ruin_weight=float(calibration.get("dynamic_ruin_weight", 1.0)),
            )
            s = global_w + explore_w
            wg, we = global_w / s, explore_w / s
            blended = wg * global_hw + we * explore_hw
            rs = np.sum(blended, axis=-1, keepdims=True)
            rs = np.where(rs > 0, rs, 1.0)
            blended = blended / rs
            final_hw = apply_floor_and_renorm(blended, eps=float(prob_floor))
            pred_flat = final_hw.reshape(-1, NUM_CLASSES)
            yield SeedRec(rid, rnum, y_true, pred_flat, X)


def _scalar_training_loss(y_true: np.ndarray, q: np.ndarray, objective: str) -> float:
    objective = objective.strip().lower()
    if objective == "ce":
        return float(mean_cross_entropy(y_true, q))
    if objective == "weighted_kl":
        return float(entropy_weighted_kl(y_true, q))
    if objective == "score":
        wkl = float(entropy_weighted_kl(y_true, q))
        return float(-competition_score(wkl))
    raise ValueError(f"ukjent objective: {objective}")


def fit_global_weights(
    pred: np.ndarray,
    y: np.ndarray,
    *,
    objective: str = "ce",
    x0_log: np.ndarray | None = None,
) -> np.ndarray:
    """Finn w>0 som minimerer valgt treningsmal (ce / weighted_kl / score-proxy)."""
    p = np.asarray(pred, dtype=np.float64)
    y_true = np.asarray(y, dtype=np.float64)
    if x0_log is None:
        x0_log = np.zeros(NUM_CLASSES)

    def obj(logw: np.ndarray) -> float:
        w = np.exp(np.clip(logw, -8, 8))
        q = apply_class_weights(p, w)
        eps = 1e-12
        q = np.clip(q, eps, 1.0)
        return _scalar_training_loss(y_true, q, objective)

    res = minimize(obj, x0_log, method="L-BFGS-B", options={"maxiter": 200})
    return np.exp(np.clip(res.x, -8, 8))


def build_calibration_recommendation(before_va: dict[str, Any], after_va: dict[str, Any]) -> dict[str, Any]:
    """Primært wKL og estimert score; CE-forbedring uten score-gevinst = ikke produksjon."""
    ce_b = float(before_va["mean_cross_entropy"])
    ce_a = float(after_va["mean_cross_entropy"])
    wkl_b = float(before_va["entropy_weighted_kl"])
    wkl_a = float(after_va["entropy_weighted_kl"])
    sc_b = float(before_va["estimated_competition_score"])
    sc_a = float(after_va["estimated_competition_score"])
    d_ce = ce_a - ce_b
    d_wkl = wkl_a - wkl_b
    d_sc = sc_a - sc_b
    ce_improved = d_ce < -1e-12
    wkl_worse = d_wkl > 1e-12
    score_worse = d_sc < -1e-12
    wkl_better = d_wkl < -1e-12
    score_better = d_sc > 1e-12

    flags: dict[str, bool] = {
        "ce_improved": ce_improved,
        "entropy_weighted_kl_worse": wkl_worse,
        "estimated_score_worse": score_worse,
        "ce_improved_but_competition_worse": ce_improved and (wkl_worse or score_worse),
    }
    if flags["ce_improved_but_competition_worse"]:
        recommendation = "do_not_use_production"
        reason = (
            "CE falt pa holdout, men entropy_weighted_kl og/eller estimated_competition_score ble verre; "
            "ikke bruk denne kalibreringen som beslutningsgrunnlag for konkurranse-score."
        )
    elif wkl_better and score_better:
        recommendation = "use_in_predict"
        reason = "Holdout: lavere wKL og hoyere estimert score — foretrukket for produksjon."
    elif (not wkl_worse) and (not score_worse):
        recommendation = "marginal_ok"
        reason = "Holdout: ingen klar forverring pa wKL/score; kan brukes med forsiktighet."
    else:
        recommendation = "marginal_or_skip"
        reason = "Holdout: wKL og/eller estimert score ble ikke bedre; vurder a hoppe over eller tren pa mer data."

    return {
        "recommendation": recommendation,
        "reason": reason,
        "primary_metrics": "entropy_weighted_kl (lavere bedre), estimated_competition_score (hoyere bedre)",
        "deltas_val": {
            "mean_cross_entropy": d_ce,
            "entropy_weighted_kl": d_wkl,
            "estimated_competition_score": d_sc,
        },
        "flags": flags,
    }


def aggregate_metrics(y: np.ndarray, q: np.ndarray) -> dict[str, Any]:
    return summarize_tensor_pair(y, q)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Class-wise probability calibration on historical data.")
    p.add_argument("--historical-root", type=Path, default=None)
    p.add_argument("--latest-n-rounds", type=int, default=None)
    p.add_argument("--holdout-last-n", type=int, default=3, help="Siste N runder (etter round_number) = val.")
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--model-meta", type=Path, default=None)
    p.add_argument("--explore-dir", type=Path, default=None)
    p.add_argument("--global-weight", type=float, default=0.7)
    p.add_argument("--explore-weight", type=float, default=0.3)
    p.add_argument("--prob-floor", type=float, default=DEFAULT_EPS)
    p.add_argument("--mode", choices=("global", "regime"), default="global")
    p.add_argument("--min-regime-samples", type=int, default=800, help="Min treningsceller for regime-vekter.")
    p.add_argument(
        "--objective",
        choices=("ce", "weighted_kl", "score"),
        default="weighted_kl",
        help="Treningsmal for klasse-vekter: ce, entropy-weighted KL, eller negativ estimert score (score ~ samme optimum som wKL).",
    )
    p.add_argument(
        "--compare-objectives",
        action="store_true",
        help="Etter hovedkjoring: fitt ogsa ce og score/weighted_kl varianter og rapporter holdout-sammenligning i JSON.",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Default: astar/data/models/prob_calibration.json eller prob_calibration_regime.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = _repo_root()
    hist = args.historical_root or (root / "astar" / "data" / "historical")
    out_json = args.out_json
    if out_json is None:
        out_json = (
            root / "astar" / "data" / "models" / "prob_calibration_regime.json"
            if args.mode == "regime"
            else root / "astar" / "data" / "models" / "prob_calibration.json"
        )

    model, meta, ap, _mp, _ls = load_global_model_bundle(args.model_path, args.model_meta)
    fn_meta = meta.get("feature_names")
    if not fn_meta:
        raise SystemExit("Meta mangler feature_names")
    feature_names = [str(x) for x in fn_meta]
    feature_set = str(meta.get("feature_set") or infer_feature_set_from_feature_names(feature_names))
    neighbor_radius = int(meta.get("neighbor_radius", 1))
    cal_explore = _load_calibration_explore(Path(args.explore_dir) if args.explore_dir else None)

    all_dirs = discover_historical_round_dirs(hist)
    round_dirs, filter_desc = filter_round_dirs(
        all_dirs,
        round_ids=parse_round_ids_csv(None),
        latest_n_rounds=args.latest_n_rounds,
        first_n_rounds=None,
    )
    meta_rounds: list[tuple[Path, str, int]] = []
    for p in round_dirs:
        d = _load_json(p / "round_detail.json")
        meta_rounds.append((p, str(d.get("id", p.name)), int(d.get("round_number", 0))))
    meta_rounds.sort(key=lambda t: (t[2], t[1]))
    sorted_paths = [t[0] for t in meta_rounds]

    if args.holdout_last_n >= len(sorted_paths):
        raise SystemExit("--holdout-last-n ma være mindre enn antall runder")
    train_paths = sorted_paths[: -args.holdout_last_n]
    val_paths = sorted_paths[-args.holdout_last_n :]
    holdout_ids = {t[1] for t in meta_rounds[-args.holdout_last_n :]}

    train_recs = list(
        iter_seed_recs(
            round_dirs=train_paths,
            model=model,
            feature_set=feature_set,
            neighbor_radius=neighbor_radius,
            calibration=cal_explore,
            global_w=args.global_weight,
            explore_w=args.explore_weight,
            prob_floor=args.prob_floor,
        )
    )
    val_recs = list(
        iter_seed_recs(
            round_dirs=val_paths,
            model=model,
            feature_set=feature_set,
            neighbor_radius=neighbor_radius,
            calibration=cal_explore,
            global_w=args.global_weight,
            explore_w=args.explore_weight,
            prob_floor=args.prob_floor,
        )
    )
    if not train_recs or not val_recs:
        raise SystemExit("Mangler seed-data for train eller val")

    pred_tr = np.concatenate([r.pred_flat for r in train_recs], axis=0)
    y_tr = np.concatenate([r.y_true_flat for r in train_recs], axis=0)
    X_tr = np.concatenate([r.X_flat for r in train_recs], axis=0)
    pred_va = np.concatenate([r.pred_flat for r in val_recs], axis=0)
    y_va = np.concatenate([r.y_true_flat for r in val_recs], axis=0)
    X_va = np.concatenate([r.X_flat for r in val_recs], axis=0)

    before_tr = aggregate_metrics(y_tr, pred_tr)
    before_va = aggregate_metrics(y_va, pred_va)

    REGIMES = [
        "default",
        "coastal_near_settlement",
        "inland_near_settlement",
        "inland_far",
        "mountain_near",
        "high_density_settlement",
    ]

    obj_primary = str(args.objective)

    def _fit_and_metrics_global(objective: str) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
        wloc = fit_global_weights(pred_tr, y_tr, objective=objective)
        qtr = apply_class_weights(pred_tr, wloc)
        qva = apply_class_weights(pred_va, wloc)
        return wloc, aggregate_metrics(y_tr, qtr), aggregate_metrics(y_va, qva)

    if args.mode == "global":
        w, after_tr, after_va = _fit_and_metrics_global(obj_primary)
        w_dict = {CLASS_INDEX_NAMES[i]: float(w[i]) for i in range(NUM_CLASSES)}
        rec_block = build_calibration_recommendation(before_va, after_va)
        payload = {
            "version": 1,
            "mode": "global",
            "training_objective": obj_primary,
            "model_path": str(ap),
            "feature_set": feature_set,
            "blend": {"global_weight": args.global_weight, "explore_weight": args.explore_weight},
            "prob_floor": args.prob_floor,
            "filter_mode": filter_desc,
            "holdout_round_ids": sorted(holdout_ids),
            "train_rounds": len(train_paths),
            "val_rounds": len(val_paths),
            "weights": w_dict,
            "metrics_before": {"train": before_tr, "val": before_va},
            "metrics_after": {"train": after_tr, "val": after_va},
            **rec_block,
        }
    else:
        # Global base
        w_global = fit_global_weights(pred_tr, y_tr, objective=obj_primary)
        by_reg: dict[str, list[int]] = defaultdict(list)
        for i in range(len(pred_tr)):
            by_reg[regime_from_feature_row(X_tr[i])].append(i)

        weights_by_regime: dict[str, dict[str, float]] = {}
        sample_counts: dict[str, int] = {r: len(by_reg[r]) for r in REGIMES}
        for reg in REGIMES:
            idx = by_reg[reg]
            if len(idx) < args.min_regime_samples:
                weights_by_regime[reg] = {CLASS_INDEX_NAMES[j]: float(w_global[j]) for j in range(NUM_CLASSES)}
                continue
            pr = pred_tr[idx]
            yr = y_tr[idx]
            wr = fit_global_weights(
                pr,
                yr,
                objective=obj_primary,
                x0_log=np.log(np.maximum(w_global, 1e-6)),
            )
            weights_by_regime[reg] = {CLASS_INDEX_NAMES[j]: float(wr[j]) for j in range(NUM_CLASSES)}

        def apply_reg(pred: np.ndarray, Xrows: np.ndarray) -> np.ndarray:
            out = np.empty_like(pred)
            for i in range(len(pred)):
                reg = regime_from_feature_row(Xrows[i])
                wr = np.asarray(
                    [weights_by_regime[reg][k] for k in CLASS_INDEX_NAMES],
                    dtype=np.float64,
                )
                if sample_counts.get(reg, 0) < args.min_regime_samples:
                    wr = w_global
                out[i] = apply_class_weights(pred[i : i + 1], wr)[0]
            return out

        q_tr = apply_reg(pred_tr, X_tr)
        q_va = apply_reg(pred_va, X_va)
        after_tr = aggregate_metrics(y_tr, q_tr)
        after_va = aggregate_metrics(y_va, q_va)
        w_global_dict = {CLASS_INDEX_NAMES[i]: float(w_global[i]) for i in range(NUM_CLASSES)}
        rec_block = build_calibration_recommendation(before_va, after_va)
        payload = {
            "version": 1,
            "mode": "regime",
            "training_objective": obj_primary,
            "model_path": str(ap),
            "feature_set": feature_set,
            "blend": {"global_weight": args.global_weight, "explore_weight": args.explore_weight},
            "prob_floor": args.prob_floor,
            "min_regime_samples": args.min_regime_samples,
            "filter_mode": filter_desc,
            "holdout_round_ids": sorted(holdout_ids),
            "default_weights": w_global_dict,
            "weights_by_regime": weights_by_regime,
            "regime_train_counts": sample_counts,
            "metrics_before": {"train": before_tr, "val": before_va},
            "metrics_after": {"train": after_tr, "val": after_va},
            **rec_block,
        }

    if args.compare_objectives:
        cmp_rows: list[dict[str, Any]] = []
        for o in ("ce", "weighted_kl", "score"):
            if args.mode != "global":
                cmp_rows.append(
                    {
                        "objective": o,
                        "skipped": True,
                        "reason": "compare_objectives stottes kun for --mode global (enkle vekter).",
                    }
                )
                continue
            _w, _atr, av = _fit_and_metrics_global(o)
            cmp_rows.append(
                {
                    "objective": o,
                    "val_metrics_after": av,
                    "recommendation_detail": build_calibration_recommendation(before_va, av),
                }
            )
        best = None
        for row in cmp_rows:
            if row.get("skipped"):
                continue
            av = row.get("val_metrics_after") or {}
            sc = float(av.get("estimated_competition_score", -1e9))
            wkl = float(av.get("entropy_weighted_kl", 1e9))
            key = (sc, -wkl)
            if best is None or key > best[0]:
                best = (key, row["objective"])
        payload["objective_holdout_comparison"] = cmp_rows
        if best:
            payload["best_holdout_objective_by_score"] = best[1]

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "wrote": str(out_json),
                "recommendation": payload.get("recommendation"),
                "reason": payload.get("reason"),
                "flags": payload.get("flags"),
            },
            indent=2,
        )
    )
    print("val before:", before_va)
    print("val after:", payload["metrics_after"]["val"])


if __name__ == "__main__":
    main()
