"""
Evaluer historical seeds med competition-lignende score (entropy-weighted KL + estimat 0-100).

Stotter global+blend og valgfri --calibration-json (global eller regime).

  export PYTHONPATH=.
  python -m astar.score_evaluate_historical --latest-n-rounds 10
  python -m astar.score_evaluate_historical --calibration-json astar/data/models/prob_calibration.json
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
from .explore_hierarchy import (
    apply_regional_explore_scales,
    build_explore_hierarchy_diagnostic,
    load_hierarchy_file,
    resolve_explore_scalar_boosts,
)
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Competition-like score on historical predictions.")
    p.add_argument("--historical-root", type=Path, default=None)
    p.add_argument("--latest-n-rounds", type=int, default=None)
    p.add_argument("--round-ids", type=str, default=None)
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument(
        "--model-meta",
        type=Path,
        default=None,
        help="Meta-JSON (default: par til --model-path via meta_path_for_joblib, ellers best_global_model_meta.json).",
    )
    p.add_argument("--explore-dir", type=Path, default=None)
    p.add_argument("--global-weight", type=float, default=0.7)
    p.add_argument("--explore-weight", type=float, default=0.3)
    p.add_argument("--prob-floor", type=float, default=DEFAULT_EPS)
    p.add_argument("--calibration-json", type=Path, default=None)
    p.add_argument(
        "--explore-hierarchy-mode",
        choices=("off", "global", "seed", "full"),
        default="seed",
        help="Styring av explore_calibration_hierarchy.json (samme som predict).",
    )
    p.add_argument("--min-regional-samples", type=int, default=200)
    p.add_argument(
        "--compare-explore-modes",
        action="store_true",
        help="Evaluer off/global/seed/full og skriv per-modus metrics (samme runder/seeds).",
    )
    p.add_argument(
        "--debug-explore-hierarchy",
        action="store_true",
        help="Skriv utfyllende diagnose for explore-hierarchy (fil, boosts, regioner, fallbacks).",
    )
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args()


def _print_explore_hierarchy_diagnostic(diag: dict[str, Any]) -> None:
    print("\n=== diagnose: explore-hierarchy i historical score_evaluate ===")
    print(f"  explore_dir: {diag.get('explore_dir')}")
    print(f"  analysis_summary.json finnes: {diag.get('analysis_summary_exists')}  ({diag.get('analysis_summary_path')})")
    print(f"  explore_calibration_hierarchy.json finnes: {diag.get('hierarchy_json_exists')}  ({diag.get('hierarchy_json_path')})")
    print(f"  Nøkler fra summary (calibration_suggestion): {diag.get('base_calibration_keys_from_summary')}")
    print(f"  Hierarchy toppnivå-nøkler: {diag.get('hierarchy_top_level_keys')}")
    print(f"  Hierarchy global BOOST_KEYS: {diag.get('hierarchy_global_boost_keys')}")
    print(f"  Hierarchy per_seed indekser i fil: {diag.get('per_seed_keys_in_file')}")
    sfe = diag.get("scalar_branch_effective") or {}
    print(f"  Faktisk scalar-skill: off≠global={sfe.get('off_vs_global')}  global≠seed={sfe.get('global_vs_seed')}  regional(full)={sfe.get('seed_vs_full_regional_only')}")
    mm = diag.get("modes_meaningful_distinction") or {}
    print("  Forventet meningsfull modus-skill (utfall kan fortsatt være like pga global modell dominans):")
    for m in ("off", "global", "seed", "full"):
        print(f"    {m}: {'JA' if mm.get(m) else 'NEI — kollapser til samme explore-prior som enklere modus'}")
    if diag.get("compare_explore_modes_likely_identical_scores"):
        print("  >>> compare_explore_modes vil sannsynligvis gi IDENTISKE metrics (ingen hierarchy-effekt).")
    w = diag.get("warning")
    if w:
        print(f"  ADVARSEL: {w}")
    print(f"  Merknad: {diag.get('note_single_round_explore_dir')}")
    samples = diag.get("sample_seed_diagnostics") or {}
    if samples:
        print("  Eksempel (seed 0) — effective terrain-boosts etter resolve:")
        s0 = samples.get("0")
        if isinstance(s0, dict):
            eb = s0.get("effective_boosts") or {}
            print(f"    off:    {eb.get('off')}")
            print(f"    global: {eb.get('global')}")
            print(f"    seed:   {eb.get('seed')}")
            print(f"    full:   samme tensor som seed + regional class_scale i bins: {s0.get('regional_bins_eligible_full_mode')}")
            notes = s0.get("regional_notes_sample") or []
            if notes:
                print(f"    regional avvisninger (utdrag): {notes[:4]}")

    print("\n=== modus: brukes hierarchy? (kode-sti) ===")
    print("  off    — resolve: hierarchy ignoreres (mode==off). Regional: hopper over (mode!=full).")
    print("  global — resolve: base + hierarchy.global BOOST_KEYS hvis fil og dict finnes; ellers = off.")
    print("  seed   — resolve: som global + hierarchy.per_seed[seed] BOOST_KEYS; ellers = global.")
    print("  full   — resolve: som seed; deretter apply_regional_explore_scales hvis bins er eligible.")


def _evaluate_explore_mode(
    *,
    round_dirs: list[Path],
    model: object,
    feature_set: str,
    nr: int,
    cal_e: dict[str, float],
    explore_hier: dict[str, Any] | None,
    hierarchy_mode: str,
    min_regional_samples: int,
    global_weight: float,
    explore_weight: float,
    prob_floor: float,
    calibration_json: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_seed: list[dict[str, Any]] = []
    y_all: list[np.ndarray] = []
    q_all: list[np.ndarray] = []

    for rd in round_dirs:
        detail = load_detail_json(rd / "round_detail.json")
        rid = str(detail.get("id", rd.name))
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
            y_hw = gt_np / rs
            y_flat = y_hw.reshape(-1, NUM_CLASSES)
            st = iss[si]
            settlements = st.get("settlements", [])
            X = feature_matrix_for_seed(gnp, settlements, neighbor_radius=nr, feature_set=feature_set)
            gf = predict_proba_fixed6(model, X).reshape(h, w, NUM_CLASSES)
            c = resolve_explore_scalar_boosts(cal_e, explore_hier, si, hierarchy_mode)
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
                mode=hierarchy_mode,
                min_region_samples=min_regional_samples,
            )
            s = global_weight + explore_weight
            wg, we = global_weight / s, explore_weight / s
            bl = wg * gf + we * ex
            rs2 = np.sum(bl, axis=-1, keepdims=True)
            rs2 = np.where(rs2 > 0, rs2, 1.0)
            bl = bl / rs2
            final_hw = apply_floor_and_renorm(bl, eps=float(prob_floor))
            q_flat = final_hw.reshape(-1, NUM_CLASSES)
            if calibration_json and Path(calibration_json).is_file():
                q_flat = apply_saved_calibration_json(q_flat, X, Path(calibration_json))
                q_flat = apply_floor_and_renorm(q_flat.reshape(h, w, NUM_CLASSES), eps=float(prob_floor)).reshape(
                    -1, NUM_CLASSES
                )

            m = summarize_tensor_pair(y_flat, q_flat)
            per_seed.append({"round_id": rid, "seed_index": si, **m})
            y_all.append(y_flat)
            q_all.append(q_flat)

    if not per_seed:
        raise SystemExit("Ingen seeds evaluert")
    y_cat = np.concatenate(y_all, axis=0)
    q_cat = np.concatenate(q_all, axis=0)
    overall = summarize_tensor_pair(y_cat, q_cat)
    return overall, per_seed


def main() -> None:
    args = parse_args()
    root = _repo_root()
    hist = args.historical_root or (root / "astar" / "data" / "historical")
    out_json = args.out_json or (root / "astar" / "data" / "evals" / "score_evaluate_historical.json")

    model, meta, ap, _mp, _ = load_global_model_bundle(args.model_path, args.model_meta)
    fn = meta.get("feature_names")
    if not fn:
        raise SystemExit("Mangler feature_names i meta")
    feature_names = [str(x) for x in fn]
    feature_set = str(meta.get("feature_set") or infer_feature_set_from_feature_names(feature_names))
    nr = int(meta.get("neighbor_radius", 1))
    cal_e = _explore_cal(Path(args.explore_dir) if args.explore_dir else None)
    explore_hier = load_hierarchy_file(Path(args.explore_dir) if args.explore_dir else None)
    explore_diag = build_explore_hierarchy_diagnostic(
        Path(args.explore_dir) if args.explore_dir else None,
        cal_e,
        explore_hier,
        min_region_samples=int(args.min_regional_samples),
    )
    if args.compare_explore_modes or args.debug_explore_hierarchy:
        _print_explore_hierarchy_diagnostic(explore_diag)

    all_dirs = discover_historical_round_dirs(hist)
    round_dirs, filter_desc = filter_round_dirs(
        all_dirs,
        round_ids=parse_round_ids_csv(args.round_ids),
        latest_n_rounds=args.latest_n_rounds,
        first_n_rounds=None,
    )

    modes = ["off", "global", "seed", "full"] if args.compare_explore_modes else [args.explore_hierarchy_mode]
    per_mode: dict[str, Any] = {}
    primary_mode = args.explore_hierarchy_mode
    overall: dict[str, Any]
    per_seed: list[dict[str, Any]]

    for mode in modes:
        ov, ps = _evaluate_explore_mode(
            round_dirs=round_dirs,
            model=model,
            feature_set=feature_set,
            nr=nr,
            cal_e=cal_e,
            explore_hier=explore_hier,
            hierarchy_mode=mode,
            min_regional_samples=int(args.min_regional_samples),
            global_weight=args.global_weight,
            explore_weight=args.explore_weight,
            prob_floor=float(args.prob_floor),
            calibration_json=Path(args.calibration_json) if args.calibration_json else None,
        )
        per_mode[mode] = {
            "overall": ov,
            "mean_estimated_score_per_seed": float(np.mean([p["estimated_competition_score"] for p in ps])),
        }
        if mode == primary_mode:
            overall, per_seed = ov, ps

    if args.compare_explore_modes:
        ranked = sorted(
            per_mode.items(),
            key=lambda kv: (-kv[1]["overall"]["estimated_competition_score"], kv[1]["overall"]["entropy_weighted_kl"]),
        )
        explore_ranking = [m for m, _ in ranked]
        compare_explore_modes_ranked: list[dict[str, Any]] = []
        for m in explore_ranking:
            ov = per_mode[m]["overall"]
            compare_explore_modes_ranked.append(
                {
                    "mode": m,
                    "mean_cross_entropy": ov["mean_cross_entropy"],
                    "mean_kl_forward": ov["mean_kl_forward"],
                    "entropy_weighted_kl": ov["entropy_weighted_kl"],
                    "estimated_competition_score": ov["estimated_competition_score"],
                }
            )
        best_explore_mode = explore_ranking[0]
    else:
        explore_ranking = [primary_mode]
        compare_explore_modes_ranked = []
        best_explore_mode = primary_mode

    report = {
        "filter_mode": filter_desc,
        "model": str(ap),
        "feature_set": feature_set,
        "blend": {"global_weight": args.global_weight, "explore_weight": args.explore_weight},
        "prob_floor": args.prob_floor,
        "calibration_json": str(args.calibration_json) if args.calibration_json else None,
        "explore_hierarchy_mode": primary_mode,
        "compare_explore_modes": bool(args.compare_explore_modes),
        "per_explore_mode": per_mode if args.compare_explore_modes else None,
        "explore_mode_ranking_by_estimated_score": explore_ranking,
        "compare_explore_modes_ranked": compare_explore_modes_ranked if args.compare_explore_modes else None,
        "best_explore_mode": best_explore_mode,
        "explore_hierarchy_diagnostic": explore_diag
        if (args.compare_explore_modes or args.debug_explore_hierarchy)
        else None,
        "overall": overall,
        "per_seed": per_seed,
        "mean_estimated_score_per_seed": float(np.mean([p["estimated_competition_score"] for p in per_seed])),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.compare_explore_modes:
        print("\n=== explore-hierarchy-moduser (rangert: høyest score, deretter lavest wKL) ===")
        hdr = f"{'mode':<6} {'CE':>12} {'KL_fwd':>12} {'wKL':>12} {'est_score':>12}"
        print(hdr)
        print("-" * len(hdr))
        for row in compare_explore_modes_ranked:
            print(
                f"{row['mode']:<6} {row['mean_cross_entropy']:12.6f} {row['mean_kl_forward']:12.6f} "
                f"{row['entropy_weighted_kl']:12.6f} {row['estimated_competition_score']:12.4f}"
            )
        print(f"\nbest_explore_mode: {best_explore_mode}\n")

    print(json.dumps({"overall": overall, "wrote": str(out_json), "best_explore_mode": best_explore_mode}, indent=2))


if __name__ == "__main__":
    main()
