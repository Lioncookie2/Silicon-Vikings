"""
Offline query-budsjett-benk pa én historical seed: velg viewport-strategi, tilt mot GT-marginal.

Stotter budsjett f.eks. 10/20/40 og strategier: stride, random, uncertainty, dynamic_focus.
Metrikker: entropy_weighted_kl, estimated_competition_score (samme som competition_metrics).

  export PYTHONPATH=.
  python -m astar.historical_query_simulator \\
    --round-dir astar/data/historical/<uuid> --seed-index 0 --budget 20 --strategy uncertainty

  python -m astar.historical_query_simulator ... --compare-strategies --budget 20 --trials 3
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from .baseline import DEFAULT_EPS, apply_floor_and_renorm, build_terrain_prior, load_detail_json
from .cell_features import feature_matrix_for_seed, infer_feature_set_from_feature_names
from .competition_metrics import entropy_per_row, summarize_tensor_pair
from .explore_hierarchy import apply_regional_explore_scales, load_hierarchy_file, resolve_explore_scalar_boosts
from .global_model_loader import load_global_model_bundle
from .sklearn_utils import predict_proba_fixed6

NUM_CLASSES = 6


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _explore_cal(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    p = Path(path) / "analysis_summary.json"
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


def marginal_from_gt(gt_hw: np.ndarray, y0: int, x0: int, vh: int, vw: int) -> np.ndarray:
    patch = gt_hw[y0 : y0 + vh, x0 : x0 + vw, :]
    m = np.mean(patch, axis=(0, 1))
    s = float(np.sum(m))
    if s <= 0:
        return np.ones(NUM_CLASSES) / NUM_CLASSES
    return m / s


def random_viewports(h: int, w: int, vh: int, vw: int, budget: int, rng: random.Random) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if h < vh or w < vw:
        return [(0, 0)]
    for _ in range(budget * 5):
        if len(out) >= budget:
            break
        y0 = rng.randint(0, h - vh)
        x0 = rng.randint(0, w - vw)
        if (y0, x0) not in out:
            out.append((y0, x0))
    return out[:budget]


def stride_viewports(h: int, w: int, vh: int, vw: int, budget: int) -> list[tuple[int, int]]:
    if h < vh or w < vw:
        return [(0, 0)] * min(1, budget)
    step_y, step_x = max(1, vh), max(1, vw)
    out: list[tuple[int, int]] = []
    for y0 in range(0, h - vh + 1, step_y):
        for x0 in range(0, w - vw + 1, step_x):
            out.append((y0, x0))
            if len(out) >= budget:
                return out[:budget]
    if not out:
        out.append((0, 0))
    while len(out) < budget:
        out.append(out[-1])
    return out[:budget]


def _window_mean_entropy(entropy_map: np.ndarray, y0: int, x0: int, vh: int, vw: int) -> float:
    return float(np.mean(entropy_map[y0 : y0 + vh, x0 : x0 + vw]))


def uncertainty_viewports(
    h: int,
    w: int,
    vh: int,
    vw: int,
    budget: int,
    entropy_map: np.ndarray,
) -> list[tuple[int, int]]:
    if h < vh or w < vw:
        return [(0, 0)] * min(1, budget)
    candidates: list[tuple[float, int, int]] = []
    step = max(2, min(vh, vw) // 2)
    for y0 in range(0, h - vh + 1, step):
        for x0 in range(0, w - vw + 1, step):
            candidates.append((_window_mean_entropy(entropy_map, y0, x0, vh, vw), y0, x0))
    candidates.sort(key=lambda t: -t[0])
    chosen: list[tuple[int, int]] = []
    for _, y0, x0 in candidates:
        if len(chosen) >= budget:
            break
        overlap = False
        for cy, cx in chosen:
            if not (y0 + vh <= cy or cy + vh <= y0 or x0 + vw <= cx or cx + vw <= x0):
                overlap = True
                break
        if not overlap:
            chosen.append((y0, x0))
    if len(chosen) < budget:
        for _, y0, x0 in candidates:
            if (y0, x0) in chosen:
                continue
            chosen.append((y0, x0))
            if len(chosen) >= budget:
                break
    return chosen[:budget]


def apply_tilt_windows(
    tilted: np.ndarray,
    y_true: np.ndarray,
    windows: list[tuple[int, int]],
    vh: int,
    vw: int,
    tilt_weight: float,
) -> np.ndarray:
    out = tilted.copy()
    for y0, x0 in windows:
        marg = marginal_from_gt(y_true, y0, x0, vh, vw)
        patch = out[y0 : y0 + vh, x0 : x0 + vw, :].copy()
        tw = tilt_weight
        patch = (1.0 - tw) * patch + tw * marg.reshape(1, 1, NUM_CLASSES)
        rs3 = np.sum(patch, axis=-1, keepdims=True)
        rs3 = np.where(rs3 > 0, rs3, 1.0)
        out[y0 : y0 + vh, x0 : x0 + vw, :] = patch / rs3
    return out


def dynamic_focus_windows(
    tilted: np.ndarray,
    y_true: np.ndarray,
    h: int,
    w: int,
    vh: int,
    vw: int,
    budget: int,
    tilt_weight: float,
) -> np.ndarray:
    cur = tilted.copy()
    for _ in range(budget):
        flat = cur.reshape(-1, NUM_CLASSES)
        ent = entropy_per_row(flat).reshape(h, w)
        best = (-1.0, 0, 0)
        step = max(1, min(vh, vw) // 3)
        for y0 in range(0, max(1, h - vh + 1), step):
            for x0 in range(0, max(1, w - vw + 1), step):
                sc = _window_mean_entropy(ent, y0, x0, vh, vw)
                if sc > best[0]:
                    best = (sc, y0, x0)
        _, y0, x0 = best
        cur = apply_tilt_windows(cur, y_true, [(y0, x0)], vh, vw, tilt_weight)
    return cur


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Historical query budget benchmark (proxy).")
    p.add_argument("--round-dir", type=Path, required=True, help="Historical round med analysis/seed_*.json")
    p.add_argument("--seed-index", type=int, default=0)
    p.add_argument("--budget", type=int, default=20, help="Antall viewport-queries (10/20/40 anbefalt).")
    p.add_argument("--viewport", type=int, default=12, help="Kvadrat side (vh=vw)")
    p.add_argument("--trials", type=int, default=3, help="Gjenta (kun random-strategi varierer mellom trials).")
    p.add_argument("--tilt-weight", type=float, default=0.15, help="Hvor mye marginal-tilt blandes inn i vindu")
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--global-weight", type=float, default=0.7)
    p.add_argument("--explore-weight", type=float, default=0.3)
    p.add_argument("--prob-floor", type=float, default=DEFAULT_EPS)
    p.add_argument("--explore-dir", type=Path, default=None)
    p.add_argument(
        "--explore-hierarchy-mode",
        choices=("off", "global", "seed", "full"),
        default="seed",
    )
    p.add_argument("--min-regional-samples", type=int, default=200)
    p.add_argument(
        "--strategy",
        choices=("stride", "random", "uncertainty", "dynamic_focus"),
        default="random",
    )
    p.add_argument("--compare-strategies", action="store_true", help="Kjor alle strategier (trials kun for random).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    rd = args.round_dir
    detail = load_detail_json(rd / "round_detail.json")
    sp = rd / "analysis" / f"seed_{args.seed_index}.json"
    if not sp.is_file():
        raise SystemExit(f"Mangler {sp}")
    item = _load_json(sp)
    gt = np.asarray(item["ground_truth"], dtype=np.float64)
    ig = np.asarray(item["initial_grid"], dtype=np.int32)
    h, w, _ = gt.shape
    rs = np.sum(gt, axis=-1, keepdims=True)
    rs = np.where(rs > 0, rs, 1.0)
    y_true = gt / rs
    iss = detail["initial_states"][args.seed_index]
    settlements = iss.get("settlements", [])

    model, meta, _, _, _ = load_global_model_bundle(args.model_path, None)
    fn = meta.get("feature_names")
    if not fn:
        raise SystemExit("meta mangler feature_names")
    fs = str(meta.get("feature_set") or infer_feature_set_from_feature_names([str(x) for x in fn]))
    nr = int(meta.get("neighbor_radius", 1))
    X = feature_matrix_for_seed(ig, settlements, neighbor_radius=nr, feature_set=fs)
    gf = predict_proba_fixed6(model, X).reshape(h, w, NUM_CLASSES)
    cal_e = _explore_cal(Path(args.explore_dir) if args.explore_dir else None)
    hier = load_hierarchy_file(Path(args.explore_dir) if args.explore_dir else None)
    c = resolve_explore_scalar_boosts(cal_e, hier, args.seed_index, args.explore_hierarchy_mode)
    ex = build_terrain_prior(
        ig.tolist(),
        settlements,
        eps=DEFAULT_EPS,
        coast_boost=float(c.get("coast_boost", 0.35)),
        near_settlement_boost=float(c.get("near_settlement_boost", 0.5)),
        coast_near_settle_port_boost=float(c.get("coast_near_settle_port_boost", 0.28)),
        dynamic_ruin_weight=float(c.get("dynamic_ruin_weight", 1.0)),
    )
    ex = apply_regional_explore_scales(
        ex,
        hier,
        h,
        w,
        args.seed_index,
        mode=args.explore_hierarchy_mode,
        min_region_samples=int(args.min_regional_samples),
    )
    s = args.global_weight + args.explore_weight
    wg, we = args.global_weight / s, args.explore_weight / s
    bl = wg * gf + we * ex
    rs2 = np.sum(bl, axis=-1, keepdims=True)
    rs2 = np.where(rs2 > 0, rs2, 1.0)
    base_hw = apply_floor_and_renorm(bl / rs2, eps=float(args.prob_floor))

    y_flat = y_true.reshape(-1, NUM_CLASSES)
    base_flat = base_hw.reshape(-1, NUM_CLASSES)
    base_score = summarize_tensor_pair(y_flat, base_flat)

    vp = args.viewport
    strategies = ["stride", "random", "uncertainty", "dynamic_focus"] if args.compare_strategies else [args.strategy]

    def run_strategy(name: str) -> dict[str, Any]:
        trial_rows: list[dict[str, Any]] = []
        n_trials = args.trials if name == "random" else 1
        for t in range(n_trials):
            if name == "stride":
                wins = stride_viewports(h, w, vp, vp, args.budget)
            elif name == "random":
                wins = random_viewports(h, w, vp, vp, args.budget, rng)
            elif name == "uncertainty":
                ent_map = entropy_per_row(base_hw.reshape(-1, NUM_CLASSES)).reshape(h, w)
                wins = uncertainty_viewports(h, w, vp, vp, args.budget, ent_map)
            elif name == "dynamic_focus":
                tilted_df = dynamic_focus_windows(
                    base_hw, y_true, h, w, vp, vp, args.budget, args.tilt_weight
                )
                tilted_df = apply_floor_and_renorm(tilted_df, eps=float(args.prob_floor))
                m = summarize_tensor_pair(y_flat, tilted_df.reshape(-1, NUM_CLASSES))
                trial_rows.append({"trial": t, **m})
                continue
            else:
                raise SystemExit(f"ukjent strategi {name}")
            tilted = apply_tilt_windows(base_hw, y_true, wins, vp, vp, args.tilt_weight)
            tilted = apply_floor_and_renorm(tilted, eps=float(args.prob_floor))
            m = summarize_tensor_pair(y_flat, tilted.reshape(-1, NUM_CLASSES))
            trial_rows.append({"trial": t, **m})
        mean_sc = float(np.mean([r["estimated_competition_score"] for r in trial_rows])) if trial_rows else 0.0
        mean_wkl = float(np.mean([r["entropy_weighted_kl"] for r in trial_rows])) if trial_rows else 0.0
        return {
            "strategy": name,
            "mean_estimated_competition_score": mean_sc,
            "mean_entropy_weighted_kl": mean_wkl,
            "trials": trial_rows,
        }

    results = [run_strategy(s) for s in strategies]
    ranked = sorted(results, key=lambda r: (-r["mean_estimated_competition_score"], r["mean_entropy_weighted_kl"]))
    payload: dict[str, Any] = {
        "baseline": base_score,
        "budget": args.budget,
        "viewport": vp,
        "blend": {"global_weight": args.global_weight, "explore_weight": args.explore_weight},
        "explore_hierarchy_mode": args.explore_hierarchy_mode,
        "strategies": results,
        "ranking_by_mean_estimated_score": [r["strategy"] for r in ranked],
    }
    print(json.dumps(payload, indent=2))

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
