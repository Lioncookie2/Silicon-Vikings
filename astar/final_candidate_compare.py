"""
Round-aware LOO: GB + global/explore-blend mot historical y_prob.

Sammenligner et lite sett faste kandidater (feature_set x blend) uten stor grid search.

Kjor fra repo-roten **Silicon-Vikings** (venv valgfritt — sett PYTHONPATH):

  cd /sti/til/Silicon-Vikings
  export PYTHONPATH=.

  # Standard (historical: astar/data/historical, implisitt)
  python -m astar.final_candidate_compare --latest-n-rounds 6

  # Lengre grunnlag (tar mer tid)
  python -m astar.final_candidate_compare --latest-n-rounds 10 \\
    --out-json astar/data/evals/final_candidate_compare_n10.json

  # Samme explore-kalibrering som predict (velg nyeste mappe under astar/analysis/explore/)
  python -m astar.final_candidate_compare --latest-n-rounds 6 \\
    --explore-dir astar/analysis/explore/20260321_145122_cc5442dd

Rapport: astar/data/evals/final_candidate_compare.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from .baseline import DEFAULT_EPS, apply_floor_and_renorm, build_terrain_prior, load_detail_json
from .cell_features import (
    FEATURE_SET_FULL,
    FEATURE_SET_SETTLEMENT_PORT,
    FEATURE_SET_SETTLEMENT_PORT_RADIUS,
    compute_settlement_maps,
    feature_row_at,
)
from .historical_cells_loader import discover_historical_round_dirs, filter_round_dirs
from .sklearn_utils import predict_proba_fixed6

NUM_CLASSES = 6

FEATURE_SET_BY_NAME = {
    "full": FEATURE_SET_FULL,
    "settlement_port": FEATURE_SET_SETTLEMENT_PORT,
    "settlement_port_radius": FEATURE_SET_SETTLEMENT_PORT_RADIUS,
}


@dataclass
class Candidate:
    name: str
    feature_set: str
    global_w: float
    explore_w: float


DEFAULT_CANDIDATES: list[Candidate] = [
    Candidate("A", FEATURE_SET_FULL, 0.6, 0.4),
    Candidate("B", FEATURE_SET_SETTLEMENT_PORT, 0.6, 0.4),
    Candidate("C", FEATURE_SET_SETTLEMENT_PORT_RADIUS, 0.6, 0.4),
    Candidate("D", FEATURE_SET_SETTLEMENT_PORT_RADIUS, 0.7, 0.3),
    Candidate("E", FEATURE_SET_SETTLEMENT_PORT_RADIUS, 0.65, 0.35),
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_calibration(explore_dir: Path | None) -> dict[str, float]:
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


def mean_cross_entropy(y_true_prob: np.ndarray, y_pred_prob: np.ndarray) -> float:
    eps = 1e-12
    q = np.clip(y_pred_prob, eps, 1.0)
    return float(np.mean(-np.sum(y_true_prob * np.log(q), axis=1)))


def mean_kl_forward(y_true_prob: np.ndarray, y_pred_prob: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(y_true_prob, eps, 1.0)
    q = np.clip(y_pred_prob, eps, 1.0)
    kl = np.sum(p * (np.log(p) - np.log(q)), axis=1)
    return float(np.mean(kl))


def blend_then_floor(
    global_hw: np.ndarray,
    explore_hw: np.ndarray,
    wg: float,
    we: float,
    prob_floor: float,
) -> np.ndarray:
    s = wg + we
    if s <= 0:
        raise ValueError("wg+we must be > 0")
    wg, we = wg / s, we / s
    blended = wg * global_hw + we * explore_hw
    rs = np.sum(blended, axis=-1, keepdims=True)
    rs = np.where(rs > 0, rs, 1.0)
    blended = blended / rs
    return apply_floor_and_renorm(blended, eps=prob_floor)


@dataclass
class SeedPack:
    round_id: str
    seed_index: int
    h: int
    w: int
    X: dict[str, np.ndarray]
    y_arg: np.ndarray
    y_prob: np.ndarray
    explore_hw: np.ndarray


def load_seed_packs(
    historical_root: Path,
    *,
    latest_n_rounds: int,
    calibration: dict[str, float],
    neighbor_radius: int,
) -> tuple[list[SeedPack], str, list[str]]:
    all_dirs = discover_historical_round_dirs(historical_root)
    round_dirs, filter_desc = filter_round_dirs(
        all_dirs,
        round_ids=None,
        latest_n_rounds=latest_n_rounds,
        first_n_rounds=None,
    )
    feature_sets = list(FEATURE_SET_BY_NAME.values())
    packs: list[SeedPack] = []
    round_order: list[str] = []

    for rd in round_dirs:
        detail = _load_json(rd / "round_detail.json")
        rid = str(detail.get("id", rd.name))
        round_order.append(rid)
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

            init_setts = initial_states[seed_index].get("settlements", [])
            has_s, dist_s = compute_settlement_maps(init_setts, h, w)

            rows_by_fs: dict[str, list[list[float]]] = {fs: [] for fs in feature_sets}
            y_arg_list: list[int] = []
            y_prob_list: list[list[float]] = []

            for y in range(h):
                for x in range(w):
                    prob = gt_np[y, x, :]
                    if not np.all(np.isfinite(prob)):
                        continue
                    s = float(np.sum(prob))
                    if s <= 0:
                        continue
                    prob = prob / s
                    y_arg_list.append(int(np.argmax(prob)))
                    y_prob_list.append(prob.tolist())
                    for fs in feature_sets:
                        feat = feature_row_at(
                            grid_np,
                            dist_s,
                            has_s,
                            x,
                            y,
                            neighbor_radius=neighbor_radius,
                            settlements=init_setts,
                            feature_set=fs,
                        )
                        rows_by_fs[fs].append(feat)

            if not y_arg_list:
                continue

            explore_hw = build_terrain_prior(
                grid_np.tolist(),
                init_setts,
                eps=DEFAULT_EPS,
                coast_boost=float(calibration.get("coast_boost", 0.35)),
                near_settlement_boost=float(calibration.get("near_settlement_boost", 0.5)),
                coast_near_settle_port_boost=float(calibration.get("coast_near_settle_port_boost", 0.28)),
                dynamic_ruin_weight=float(calibration.get("dynamic_ruin_weight", 1.0)),
            )

            Xdict = {fs: np.asarray(rows_by_fs[fs], dtype=np.float32) for fs in feature_sets}
            packs.append(
                SeedPack(
                    round_id=rid,
                    seed_index=seed_index,
                    h=h,
                    w=w,
                    X=Xdict,
                    y_arg=np.asarray(y_arg_list, dtype=np.int64),
                    y_prob=np.asarray(y_prob_list, dtype=np.float64),
                    explore_hw=explore_hw,
                )
            )

    if not packs:
        raise SystemExit(f"Ingen seed-pakker etter filter ({filter_desc}).")
    return packs, filter_desc, round_order


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Final candidate comparison (LOO + blend).")
    p.add_argument("--historical-root", type=Path, default=None)
    p.add_argument("--latest-n-rounds", type=int, default=6)
    p.add_argument("--explore-dir", type=Path, default=None, help="Valgfri calibration_suggestion.")
    p.add_argument("--prob-floor", type=float, default=DEFAULT_EPS)
    p.add_argument("--neighbor-radius", type=int, default=1)
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Default: astar/data/evals/final_candidate_compare.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = _repo_root()
    hist = args.historical_root or (root / "astar" / "data" / "historical")
    out_path = args.out_json or (root / "astar" / "data" / "evals" / "final_candidate_compare.json")
    calibration = _load_calibration(Path(args.explore_dir) if args.explore_dir else None)

    packs, filter_desc, round_order = load_seed_packs(
        hist,
        latest_n_rounds=args.latest_n_rounds,
        calibration=calibration,
        neighbor_radius=args.neighbor_radius,
    )
    unique_rounds = list(dict.fromkeys(round_order))
    if len(unique_rounds) < 2:
        raise SystemExit("LOO krever minst 2 runder i utvalget.")

    # Per kandidat: fold-CE, fold-KL for std; total cellevektet snitt
    fold_ce: dict[str, list[float]] = {c.name: [] for c in DEFAULT_CANDIDATES}
    fold_kl: dict[str, list[float]] = {c.name: [] for c in DEFAULT_CANDIDATES}
    total_cells: dict[str, int] = {c.name: 0 for c in DEFAULT_CANDIDATES}
    sum_ce: dict[str, float] = {c.name: 0.0 for c in DEFAULT_CANDIDATES}
    sum_kl: dict[str, float] = {c.name: 0.0 for c in DEFAULT_CANDIDATES}

    model_cache: dict[tuple[str, str], GradientBoostingClassifier] = {}
    # key (test_rid, feature_set_name) -> trained model

    for test_rid in unique_rounds:
        train_packs = [pk for pk in packs if pk.round_id != test_rid]
        test_packs = [pk for pk in packs if pk.round_id == test_rid]

        for fs in FEATURE_SET_BY_NAME.values():
            X_tr = np.concatenate([pk.X[fs] for pk in train_packs], axis=0)
            y_tr = np.concatenate([pk.y_arg for pk in train_packs], axis=0)
            m = GradientBoostingClassifier(random_state=42)
            m.fit(X_tr, y_tr)
            model_cache[(test_rid, fs)] = m

        for cand in DEFAULT_CANDIDATES:
            fs = cand.feature_set
            model = model_cache[(test_rid, fs)]
            fold_ce_sum = 0.0
            fold_kl_sum = 0.0
            fold_n = 0
            for pk in test_packs:
                X_te = pk.X[fs]
                y_prob_te = pk.y_prob
                global_flat = predict_proba_fixed6(model, X_te)
                global_hw = global_flat.reshape(pk.h, pk.w, NUM_CLASSES)
                final_hw = blend_then_floor(
                    global_hw,
                    pk.explore_hw,
                    cand.global_w,
                    cand.explore_w,
                    float(args.prob_floor),
                )
                pred_flat = final_hw.reshape(-1, NUM_CLASSES)
                ce = mean_cross_entropy(y_prob_te, pred_flat)
                kl = mean_kl_forward(y_prob_te, pred_flat)
                nloc = len(y_prob_te)
                fold_ce_sum += ce * nloc
                fold_kl_sum += kl * nloc
                fold_n += nloc
            if fold_n == 0:
                continue
            mce = fold_ce_sum / fold_n
            mkl = fold_kl_sum / fold_n
            fold_ce[cand.name].append(mce)
            fold_kl[cand.name].append(mkl)
            total_cells[cand.name] += fold_n
            sum_ce[cand.name] += fold_ce_sum
            sum_kl[cand.name] += fold_kl_sum

    rows: list[dict[str, Any]] = []
    for cand in DEFAULT_CANDIDATES:
        tc = total_cells[cand.name]
        if tc == 0:
            continue
        rows.append(
            {
                "candidate": cand.name,
                "model": "gradient_boosting",
                "feature_set": cand.feature_set,
                "global_weight": cand.global_w,
                "explore_weight": cand.explore_w,
                "mean_cross_entropy_cell_weighted": sum_ce[cand.name] / tc,
                "mean_kl_forward_cell_weighted": sum_kl[cand.name] / tc,
                "std_mean_cross_entropy_across_folds": float(np.std(fold_ce[cand.name])),
                "std_mean_kl_forward_across_folds": float(np.std(fold_kl[cand.name])),
                "n_loo_folds": len(fold_ce[cand.name]),
                "n_cells_evaluated": tc,
            }
        )

    rows.sort(key=lambda r: (r["mean_cross_entropy_cell_weighted"], r["mean_kl_forward_cell_weighted"]))
    winner = rows[0]
    baseline_a = next(r for r in rows if r["candidate"] == "A")

    d_ce = baseline_a["mean_cross_entropy_cell_weighted"] - winner["mean_cross_entropy_cell_weighted"]
    d_kl = baseline_a["mean_kl_forward_cell_weighted"] - winner["mean_kl_forward_cell_weighted"]
    rel_ce = d_ce / max(baseline_a["mean_cross_entropy_cell_weighted"], 1e-9)

    if winner["candidate"] == "A":
        improvement = "ingen (A er best)"
        resubmit = False
    elif d_ce >= 0.02 or rel_ce >= 0.02:
        improvement = "tydelig"
        resubmit = winner["candidate"] != "A"
    elif d_ce >= 0.008 or rel_ce >= 0.01:
        improvement = "moderat"
        resubmit = False
    else:
        improvement = "marginal / usikker"
        resubmit = False

    report = {
        "filter_mode": filter_desc,
        "latest_n_rounds": args.latest_n_rounds,
        "n_seed_packs": len(packs),
        "unique_rounds": unique_rounds,
        "explore_dir": str(args.explore_dir) if args.explore_dir else None,
        "calibration_used": calibration,
        "prob_floor": float(args.prob_floor),
        "neighbor_radius": args.neighbor_radius,
        "candidates_ranked": rows,
        "winner": winner,
        "baseline_A": baseline_a,
        "delta_vs_A": {
            "winner_ce_minus_A_ce": -d_ce,
            "A_ce_minus_winner_ce": d_ce,
            "relative_ce_improvement_vs_A": rel_ce,
            "kl_improvement_vs_A": d_kl,
        },
        "improvement_verdict": improvement,
        "recommend_resubmit_all_5_seeds": resubmit,
        "note": "recommend_resubmit er True kun ved tydelig forbedring vs A; moderat/marginal -> behold submissions.",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== final_candidate_compare (LOO + blend) ===")
    print(f"filter: {filter_desc}  seeds: {len(packs)}  LOO folds: {len(unique_rounds)}")
    print("cand  feature_set              blend    mean_CE     mean_KL   std_CE_folds")
    for r in sorted(rows, key=lambda x: x["mean_cross_entropy_cell_weighted"]):
        print(
            f" {r['candidate']:3}  {r['feature_set']:<24}  {r['global_weight']:.2f}/{r['explore_weight']:.2f}  "
            f"{r['mean_cross_entropy_cell_weighted']:.6f}  {r['mean_kl_forward_cell_weighted']:.6f}  "
            f"{r['std_mean_cross_entropy_across_folds']:.6f}"
        )
    print(f"Vinner: {winner['candidate']}  (CE={winner['mean_cross_entropy_cell_weighted']:.6f})")
    print(f"vs A (dagens referanse): ΔCE (A-winner)={d_ce:.6f}  relativ={rel_ce*100:.2f}%  vurdering: {improvement}")
    print(f"Anbefaling resubmit: {resubmit}")
    print(f"Skrev {out_path}")


if __name__ == "__main__":
    main()
