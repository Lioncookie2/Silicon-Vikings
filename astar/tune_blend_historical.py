"""
Grid-sok etter optimale global/explore-blend-vekter mot historical ground truth.

For hver seed i astar/data/historical/<round>/analysis/seed_*.json:
  - global: best_global_model (eller --model-path) proba pa initial_grid
  - explore: build_terrain_prior (samme som predict for blend), med valgfri
    calibration_suggestion fra --explore-dir/analysis_summary.json
  - bland: wg*global + we*prior_scores, rad-normaliser, deretter apply_floor_and_renorm
    (som predict med global_model + explore)

Kjor fra repo-roten Silicon-Vikings:
  cd /sti/til/Silicon-Vikings
  export PYTHONPATH=.
  python -m astar.tune_blend_historical --latest-n-rounds 6
  python -m astar.tune_blend_historical --latest-n-rounds 6 --explore-dir astar/analysis/explore/20260321_145122_cc5442dd
  python -m astar.tune_blend_historical --out-json astar/data/evals/blend_weight_tune.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .baseline import DEFAULT_EPS, apply_floor_and_renorm, build_terrain_prior, load_detail_json
from .cell_features import feature_matrix_for_seed, infer_feature_set_from_feature_names
from .global_model_loader import load_global_model_bundle
from .global_model_paths import models_dir as default_models_dir
from .historical_cells_loader import (
    discover_historical_round_dirs,
    filter_round_dirs,
    parse_round_ids_csv,
)
from .sklearn_utils import predict_proba_fixed6

NUM_CLASSES = 6

DEFAULT_BLEND_PAIRS: list[tuple[float, float]] = [
    (0.7, 0.3),
    (0.6, 0.4),
    (0.5, 0.5),
    (0.4, 0.6),
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_calibration(explore_dir: Path | None) -> tuple[dict[str, float], str | None]:
    if explore_dir is None:
        return {}, None
    summary_path = Path(explore_dir) / "analysis_summary.json"
    if not summary_path.is_file():
        return {}, f"mangler {summary_path}"
    summary = _load_json(summary_path)
    cal = summary.get("calibration_suggestion")
    if not cal or not isinstance(cal, dict):
        return {}, f"calibration_suggestion mangler i {summary_path}"
    out: dict[str, float] = {}
    for k in ("coast_boost", "near_settlement_boost", "coast_near_settle_port_boost", "dynamic_ruin_weight"):
        if k in cal:
            try:
                out[k] = float(cal[k])
            except (TypeError, ValueError):
                pass
    return out, str(summary_path)


def _parse_blends(s: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for part in s.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "/" not in part:
            raise SystemExit(f"Ugyldig blend (forventet wg/we med /): {part!r}")
        a, b = part.split("/", 1)
        pairs.append((float(a.strip()), float(b.strip())))
    return pairs


def mean_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-12
    q = np.clip(y_pred, eps, 1.0)
    return float(np.mean(-np.sum(y_true * np.log(q), axis=-1)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune global/explore blend mot historical y_prob.")
    p.add_argument("--historical-root", type=Path, default=None, help="Default: astar/data/historical")
    p.add_argument(
        "--explore-dir",
        type=Path,
        default=None,
        help="Valgfri explore-mappe med analysis_summary.json (calibration_suggestion).",
    )
    p.add_argument("--model-path", type=Path, default=None, help="Default: best_global_model.joblib")
    p.add_argument("--model-meta", type=Path, default=None)
    p.add_argument("--prob-floor", type=float, default=DEFAULT_EPS, help="Etter blend, som predict --prob-floor.")
    p.add_argument(
        "--blends",
        type=str,
        default="0.7/0.3,0.6/0.4,0.5/0.5,0.4/0.6",
        help="Kommaseparerte par global/explore, f.eks. 0.6/0.4,0.5/0.5",
    )
    p.add_argument("--round-ids", type=str, default=None)
    p.add_argument("--latest-n-rounds", type=int, default=None)
    p.add_argument("--first-n-rounds", type=int, default=None)
    p.add_argument("--max-rounds", type=int, default=None)
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Skriv full rapport (default: astar/data/evals/blend_weight_tune.json).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = _repo_root()
    hist = args.historical_root or (root / "astar" / "data" / "historical")
    out_json = args.out_json or (root / "astar" / "data" / "evals" / "blend_weight_tune.json")

    blend_pairs = _parse_blends(args.blends)
    if not blend_pairs:
        blend_pairs = list(DEFAULT_BLEND_PAIRS)

    calibration, cal_note = _load_calibration(Path(args.explore_dir) if args.explore_dir else None)

    model, meta, ap, _mp, load_src = load_global_model_bundle(args.model_path, args.model_meta)
    fn_meta = meta.get("feature_names")
    if not fn_meta:
        raise SystemExit("Modell-meta mangler feature_names.")
    feature_names = [str(x) for x in fn_meta]
    feature_set = str(meta.get("feature_set") or infer_feature_set_from_feature_names(feature_names))
    neighbor_radius = int(meta.get("neighbor_radius", 1))
    nf = getattr(model, "n_features_in_", None)
    if nf is not None and int(nf) != len(feature_names):
        raise SystemExit(f"Modell n_features_in_={nf} matcher ikke meta ({len(feature_names)}).")

    first_n = args.first_n_rounds if args.first_n_rounds is not None else args.max_rounds
    all_dirs = discover_historical_round_dirs(hist)
    round_dirs, filter_desc = filter_round_dirs(
        all_dirs,
        round_ids=parse_round_ids_csv(args.round_ids),
        latest_n_rounds=args.latest_n_rounds,
        first_n_rounds=first_n,
    )

    # Per blend: (sum ce*ncells, total cells) for cellevektet snitt; pluss per-seed for std
    weighted: dict[tuple[float, float], tuple[float, int]] = {
        pair: (0.0, 0) for pair in blend_pairs
    }
    ce_accum: dict[tuple[float, float], list[float]] = {pair: [] for pair in blend_pairs}
    n_cells_total = 0
    seeds_processed = 0
    per_seed_log: list[dict[str, Any]] = []

    for rd in round_dirs:
        detail_path = rd / "round_detail.json"
        if not detail_path.is_file():
            continue
        detail = load_detail_json(detail_path)
        rid = str(detail.get("id", rd.name))
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
            grid = initial_grid
            st = initial_states[seed_index]
            settlements = st.get("settlements", [])

            if gt_np.ndim != 3 or gt_np.shape[-1] != NUM_CLASSES:
                continue
            h, w, _ = gt_np.shape
            grid_np = np.asarray(grid, dtype=np.int32)
            if grid_np.shape != (h, w):
                continue

            row_sums = np.sum(gt_np, axis=-1, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1.0)
            y_true = gt_np / row_sums

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

            seeds_processed += 1
            n_cells_total += h * w

            seed_ces: dict[str, float] = {}
            n_cells_seed = h * w
            for wg, we in blend_pairs:
                s = wg + we
                if s <= 0:
                    raise SystemExit(f"Ugyldig blend: global+explore ma være > 0 (fikk {wg}, {we})")
                wg_n, we_n = wg / s, we / s
                blended = wg_n * global_hw + we_n * explore_hw
                rs = np.sum(blended, axis=-1, keepdims=True)
                rs = np.where(rs > 0, rs, 1.0)
                blended = blended / rs
                final_np = apply_floor_and_renorm(blended, eps=float(args.prob_floor))
                ce = mean_cross_entropy(y_true, final_np)
                ce_accum[(wg, we)].append(ce)
                wsum, wc = weighted[(wg, we)]
                weighted[(wg, we)] = (wsum + ce * n_cells_seed, wc + n_cells_seed)
                seed_ces[f"{wg}_{we}"] = ce

            per_seed_log.append(
                {
                    "round_id": rid,
                    "seed_index": seed_index,
                    "path": str(seed_path),
                    "mean_ce_by_blend_cli_pair": seed_ces,
                }
            )

    if seeds_processed == 0:
        raise SystemExit(f"Ingen seeds med ground_truth under {hist} (filter: {filter_desc}).")

    # Rapport: cellevektet mean CE (primær); std over seeds som spredningsindikator
    rows: list[dict[str, Any]] = []
    for wg, we in blend_pairs:
        vals = ce_accum[(wg, we)]
        wsum, wc = weighted[(wg, we)]
        rows.append(
            {
                "global_weight": wg,
                "explore_weight": we,
                "mean_cross_entropy_cell_weighted": float(wsum / wc) if wc else float("nan"),
                "mean_cross_entropy_over_seeds_unweighted": float(np.mean(vals)),
                "std_cross_entropy_over_seeds": float(np.std(vals)),
                "n_seeds": len(vals),
            }
        )
    rows.sort(key=lambda r: r["mean_cross_entropy_cell_weighted"])
    best = rows[0]

    report: dict[str, Any] = {
        "historical_root": str(hist.resolve()),
        "filter_mode": filter_desc,
        "n_round_dirs": len(round_dirs),
        "seeds_processed": seeds_processed,
        "n_cells": n_cells_total,
        "model_path": str(ap),
        "model_load_source": load_src,
        "feature_set": feature_set,
        "neighbor_radius": neighbor_radius,
        "explore_dir": str(args.explore_dir) if args.explore_dir else None,
        "calibration_used": calibration,
        "calibration_note": cal_note,
        "prob_floor": float(args.prob_floor),
        "blend_results_ranked": rows,
        "recommended_blend_cli": {
            "global_weight": best["global_weight"],
            "explore_weight": best["explore_weight"],
            "mean_cross_entropy_cell_weighted": best["mean_cross_entropy_cell_weighted"],
        },
        "per_seed_tail": per_seed_log[:50],
        "per_seed_tail_note": "Kun forste 50 seeds i rapport; full liste kan bli stor.",
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== tune_blend_historical ===")
    print(f"filter: {filter_desc}  seeds: {seeds_processed}  celler (tot): {n_cells_total}")
    print(f"modell: {ap}  feature_set: {feature_set}")
    print(f"explore-dir: {args.explore_dir or '—'}  kalibrering: {calibration or 'standard default'}")
    print("blend (global/explore)  mean_CE (cellevektet)  std_per_seed")
    for r in rows:
        print(
            f"  {r['global_weight']:.2f} / {r['explore_weight']:.2f}     "
            f"{r['mean_cross_entropy_cell_weighted']:.6f}  "
            f"(std {r['std_cross_entropy_over_seeds']:.6f})"
        )
    print(f"Anbefalt (lavest CE): {best['global_weight']}/{best['explore_weight']}")
    print(f"Skrev {out_json}")


if __name__ == "__main__":
    main()
