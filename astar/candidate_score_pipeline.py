"""
Sammenlign kandidat-pipeline-rader mot historical data (estimated score + CE + weighted KL).

Kandidater (A–F) er forhandsdefinerte; kjorer score_evaluate_historical logikk via import.

  export PYTHONPATH=.
  python -m astar.candidate_score_pipeline --latest-n-rounds 10

Krever ferdig prob_calibration*.json for B/C hvis du vil inkludere dem (ellers hoppes de over).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .baseline import DEFAULT_EPS, apply_floor_and_renorm, build_terrain_prior, load_detail_json
from .cell_features import feature_matrix_for_seed, infer_feature_set_from_feature_names
from .competition_metrics import summarize_tensor_pair
from .explore_hierarchy import apply_regional_explore_scales, load_hierarchy_file, resolve_explore_scalar_boosts
from .global_model_loader import load_global_model_bundle
from .historical_cells_loader import discover_historical_round_dirs, filter_round_dirs, parse_round_ids_csv
from .prob_calibration import apply_saved_calibration_json
from .sklearn_utils import predict_proba_fixed6

NUM_CLASSES = 6


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def run_candidate(
    name: str,
    *,
    round_dirs: list[Path],
    model: Any,
    feature_set: str,
    nr: int,
    cal_e: dict[str, float],
    explore_hier: dict[str, Any] | None,
    explore_hierarchy_mode: str,
    min_regional_samples: int,
    wg: float,
    we: float,
    prob_floor: float,
    calib_path: Path | None,
) -> dict[str, Any]:
    y_parts: list[np.ndarray] = []
    q_parts: list[np.ndarray] = []
    for rd in round_dirs:
        detail = load_detail_json(rd / "round_detail.json")
        ad = rd / "analysis"
        if not ad.is_dir():
            continue
        for sp in sorted(ad.glob("seed_*.json")):
            item = _load_json(sp)
            gt = item.get("ground_truth")
            ig = item.get("initial_grid")
            si = int(item.get("seed_index", -1))
            if gt is None or ig is None or si < 0:
                continue
            iss = detail.get("initial_states", [])
            if si >= len(iss):
                continue
            gt_np = np.asarray(gt, dtype=np.float64)
            gnp = np.asarray(ig, dtype=np.int32)
            if gt_np.ndim != 3 or gt_np.shape[-1] != NUM_CLASSES:
                continue
            h, w, _ = gt_np.shape
            if gnp.shape != (h, w):
                continue
            rs = np.sum(gt_np, axis=-1, keepdims=True)
            rs = np.where(rs > 0, rs, 1.0)
            y_flat = (gt_np / rs).reshape(-1, NUM_CLASSES)
            st = iss[si]
            settlements = st.get("settlements", [])
            X = feature_matrix_for_seed(gnp, settlements, neighbor_radius=nr, feature_set=feature_set)
            gf = predict_proba_fixed6(model, X).reshape(h, w, NUM_CLASSES)
            c = resolve_explore_scalar_boosts(cal_e, explore_hier, si, explore_hierarchy_mode)
            ex = build_terrain_prior(
                gnp.tolist(),
                settlements,
                eps=DEFAULT_EPS,
                coast_boost=float(c.get("coast_boost", 0.35)),
                near_settlement_boost=float(c.get("near_settlement_boost", 0.5)),
                coast_near_settle_port_boost=float(c.get("coast_near_settle_port_boost", 0.28)),
                dynamic_ruin_weight=float(c.get("dynamic_ruin_weight", 1.0)),
            )
            ex = apply_regional_explore_scales(
                ex,
                explore_hier,
                h,
                w,
                si,
                mode=explore_hierarchy_mode,
                min_region_samples=min_regional_samples,
            )
            s = wg + we
            bl = (wg / s) * gf + (we / s) * ex
            rs2 = np.sum(bl, axis=-1, keepdims=True)
            rs2 = np.where(rs2 > 0, rs2, 1.0)
            q_hw = apply_floor_and_renorm(bl / rs2, eps=float(prob_floor))
            q_flat = q_hw.reshape(-1, NUM_CLASSES)
            if calib_path and calib_path.is_file():
                q_flat = apply_saved_calibration_json(q_flat, X, calib_path)
                q_flat = apply_floor_and_renorm(q_flat.reshape(h, w, NUM_CLASSES), eps=float(prob_floor)).reshape(
                    -1, NUM_CLASSES
                )
            y_parts.append(y_flat)
            q_parts.append(q_flat)
    if not y_parts:
        return {"candidate": name, "error": "no_data"}
    y = np.concatenate(y_parts, axis=0)
    q = np.concatenate(q_parts, axis=0)
    m = summarize_tensor_pair(y, q)
    return {"candidate": name, **m}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare candidate pipelines on historical data.")
    p.add_argument("--historical-root", type=Path, default=None)
    p.add_argument("--latest-n-rounds", type=int, default=None)
    p.add_argument("--explore-dir", type=Path, default=None)
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--prob-floor", type=float, default=DEFAULT_EPS)
    p.add_argument("--calibration-global", type=Path, default=None, help="prob_calibration.json")
    p.add_argument("--calibration-regime", type=Path, default=None, help="prob_calibration_regime.json")
    p.add_argument(
        "--explore-modes",
        type=str,
        default="off,global,seed,full",
        help="Kommaseparert: off,global,seed,full — kjorer hver modus mot historical (sammen med prob-kalibrering nedenfor).",
    )
    p.add_argument("--min-regional-samples", type=int, default=200)
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = _repo_root()
    hist = args.historical_root or (root / "astar" / "data" / "historical")
    out_json = args.out_json or (root / "astar" / "data" / "evals" / "candidate_score_pipeline.json")

    model, meta, ap, _, _ = load_global_model_bundle(args.model_path, None)
    fn = meta.get("feature_names")
    if not fn:
        raise SystemExit("meta mangler feature_names")
    fs = str(meta.get("feature_set") or infer_feature_set_from_feature_names([str(x) for x in fn]))
    nr = int(meta.get("neighbor_radius", 1))
    cal_e = _explore_cal(Path(args.explore_dir) if args.explore_dir else None)
    explore_hier = load_hierarchy_file(Path(args.explore_dir) if args.explore_dir else None)

    all_dirs = discover_historical_round_dirs(hist)
    round_dirs, filter_desc = filter_round_dirs(
        all_dirs,
        round_ids=parse_round_ids_csv(None),
        latest_n_rounds=args.latest_n_rounds,
        first_n_rounds=None,
    )

    wg, we = 0.7, 0.3
    mode_list = [m.strip() for m in str(args.explore_modes).split(",") if m.strip()]
    valid_modes = ("off", "global", "seed", "full")
    for m in mode_list:
        if m not in valid_modes:
            raise SystemExit(f"Ugyldig explore-modus '{m}', forventet en av {valid_modes}")

    calib_specs: list[tuple[str, Path | None]] = [
        ("nocalib", None),
        ("global_calib", args.calibration_global),
        ("regime_calib", args.calibration_regime),
    ]
    rows: list[dict[str, Any]] = []
    for emode in mode_list:
        for cal_label, cp in calib_specs:
            if cal_label != "nocalib" and (cp is None or not cp.is_file()):
                rows.append(
                    {
                        "candidate": f"{emode}_{cal_label}",
                        "explore_hierarchy_mode": emode,
                        "prob_calibration": cal_label,
                        "skipped": True,
                        "reason": "mangler calibration JSON",
                    }
                )
                continue
            name = f"{emode}_{cal_label}"
            rows.append(
                run_candidate(
                    name,
                    round_dirs=round_dirs,
                    model=model,
                    feature_set=fs,
                    nr=nr,
                    cal_e=cal_e,
                    explore_hier=explore_hier,
                    explore_hierarchy_mode=emode,
                    min_regional_samples=int(args.min_regional_samples),
                    wg=wg,
                    we=we,
                    prob_floor=args.prob_floor,
                    calib_path=cp,
                )
            )
            rows[-1]["explore_hierarchy_mode"] = emode
            rows[-1]["prob_calibration"] = cal_label

    rows_scored = [r for r in rows if "estimated_competition_score" in r]
    rows_scored.sort(
        key=lambda r: (
            -float(r["estimated_competition_score"]),
            float(r.get("entropy_weighted_kl", 1e9)),
        )
    )

    report = {
        "filter_mode": filter_desc,
        "model": str(ap),
        "feature_set": fs,
        "blend": {"global_weight": wg, "explore_weight": we},
        "explore_modes_requested": mode_list,
        "ranking_key": "estimated_competition_score (desc), then entropy_weighted_kl (asc)",
        "candidates": rows,
        "ranked_by_estimated_score": [r["candidate"] for r in rows_scored],
        "best_candidate": rows_scored[0]["candidate"] if rows_scored else None,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"wrote": str(out_json), "ranking": report["ranked_by_estimated_score"]}, indent=2))
    for r in rows:
        if "estimated_competition_score" in r:
            print(
                f"  {r['candidate']}: score~{r['estimated_competition_score']:.3f}  "
                f"wKL={r['entropy_weighted_kl']:.5f}  CE={r['mean_cross_entropy']:.5f}"
            )


if __name__ == "__main__":
    main()
