"""
Aktiv query-loop per seed: batcher med simulate(), oppdaterer working grid, reblender global+explore.

  export PYTHONPATH=.
  python3 -m astar.active_query_runner --round-id <uuid|round_number> --seed-index 0 --budget 50 \\
    --batch-sizes 10,10,30 --production-calibrated --explore-dir . \\
  (``--explore-dir .`` = repo root med analysis_summary.json; erstatt ikke med <run_id> fra mal.)

  --dry-run --round-dir ... : ingen API; plan + entropi + prediksjon pa initial grid.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .baseline import (
    DEFAULT_EPS,
    NUM_CLASSES,
    apply_floor_and_renorm,
    build_terrain_prior,
    load_detail_json,
    load_initial_state_for_seed,
    numpy_to_submission_list,
    validate_prediction,
)
from .cell_features import feature_matrix_for_seed, infer_feature_set_from_feature_names
from .client import AstarClient, get_active_round, resolve_round_identifier
from .competition_metrics import entropy_per_row
from .explore_hierarchy import apply_regional_explore_scales, load_hierarchy_file, resolve_explore_scalar_boosts
from .global_model_loader import load_global_model_or_exit
from .global_model_paths import models_dir as default_models_dir
from .prob_calibration import apply_saved_calibration_json
from .sklearn_utils import predict_proba_fixed6


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def compute_entropy_map(pred_hw: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(pred_hw, dtype=np.float64).reshape(-1, NUM_CLASSES)
    h = entropy_per_row(p, eps=eps)
    return h.reshape(pred_hw.shape[0], pred_hw.shape[1])


def disagreement_map(global_hw: np.ndarray, explore_hw: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(np.asarray(global_hw) - np.asarray(explore_hw)), axis=-1).astype(np.float64)


def viewport_around(h: int, w: int, cy: int, cx: int, vp: int) -> tuple[int, int, int, int]:
    vp = min(vp, h, w)
    half = vp // 2
    y0 = int(np.clip(cy - half, 0, max(0, h - vp)))
    x0 = int(np.clip(cx - half, 0, max(0, w - vp)))
    vw = min(vp, w - x0)
    vh = min(vp, h - y0)
    return (x0, y0, vw, vh)


def _viewport_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    union = float(aw * ah + bw * bh) - inter
    return inter / union if union > 0 else 0.0


def _viewport_center(v: tuple[int, int, int, int]) -> tuple[float, float]:
    vx, vy, vw, vh = v
    return (vx + vw / 2.0, vy + vh / 2.0)


def select_query_locations(
    *,
    entropy_map: np.ndarray,
    disagreement_map_: np.ndarray | None,
    dist_settlement: np.ndarray | None,
    n_queries: int,
    h: int,
    w: int,
    vp: int,
    phase: int,
    existing: list[tuple[int, int, int, int]],
    max_overlap: float = 0.35,
    min_center_dist: float | None = None,
) -> list[tuple[int, int, int, int]]:
    if n_queries <= 0:
        return []
    em = np.asarray(entropy_map, dtype=np.float64)
    if min_center_dist is None:
        min_center_dist = max(5.0, vp * 0.45)

    if phase == 1:
        score = em.copy()
    elif phase == 2 and disagreement_map_ is not None:
        dm = disagreement_map_.astype(np.float64)
        dmn = dm / (np.max(dm) + 1e-12)
        score = em * (1.0 + 2.0 * dmn)
    else:
        settle_boost = np.zeros_like(em)
        if dist_settlement is not None:
            ds = np.asarray(dist_settlement, dtype=np.float64)
            settle_boost = 1.0 / (1.0 + ds / 6.0)
        loc_vol = np.zeros_like(em)
        for yy in range(1, h - 1):
            for xx in range(1, w - 1):
                loc_vol[yy, xx] = float(np.max(em[yy - 1 : yy + 2, xx - 1 : xx + 2]))
        score = em * (1.0 + 0.35 * settle_boost) + 0.15 * loc_vol
        if disagreement_map_ is not None:
            dm = disagreement_map_.astype(np.float64)
            dmn = dm / (np.max(dm) + 1e-12)
            score = score * (1.0 + 0.5 * dmn)

    flat = score.ravel()
    order = np.argsort(-flat)
    ys, xs = np.divmod(order, w)
    chosen: list[tuple[int, int, int, int]] = list(existing)
    centers: list[tuple[float, float]] = [_viewport_center(v) for v in existing]

    for idx in range(len(order)):
        if len(chosen) - len(existing) >= n_queries:
            break
        cy, cx = int(ys[idx]), int(xs[idx])
        if flat[order[idx]] <= 0 and len(chosen) > len(existing):
            break
        v = viewport_around(h, w, cy, cx, vp)
        vx, vy, vw, vh = v
        if vw < 5 or vh < 5:
            continue
        if any(_viewport_iou(v, u) > max_overlap for u in chosen):
            continue
        c = _viewport_center(v)
        if centers and all(np.hypot(c[0] - oc[0], c[1] - oc[1]) < min_center_dist for oc in centers):
            if len(chosen) - len(existing) < n_queries // 2:
                continue
        chosen.append(v)
        centers.append(c)

    return chosen[len(existing) :]


def merge_simulate_into_grid(
    working_grid: np.ndarray,
    viewport_x: int,
    viewport_y: int,
    patch: Any,
) -> int:
    if patch is None:
        return 0
    g = np.asarray(patch, dtype=np.int32)
    if g.ndim != 2:
        return 0
    ph, pw = g.shape
    h, w = working_grid.shape
    n = 0
    for dy in range(ph):
        for dx in range(pw):
            gy, gx = viewport_y + dy, viewport_x + dx
            if 0 <= gy < h and 0 <= gx < w:
                v = int(g[dy, dx])
                if v < 0:
                    continue
                if working_grid[gy, gx] != v:
                    n += 1
                working_grid[gy, gx] = v
    return n


def run_query_batch(
    client: AstarClient | None,
    *,
    round_id: str,
    seed_index: int,
    viewports: list[tuple[int, int, int, int]],
    working_grid: np.ndarray,
    dry_run: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for vx, vy, vw, vh in viewports:
        if dry_run:
            out.append(
                {
                    "viewport": {"x": vx, "y": vy, "w": vw, "h": vh},
                    "dry_run": True,
                    "cells_updated": 0,
                }
            )
            continue
        assert client is not None
        res = client.simulate(
            round_id=round_id,
            seed_index=seed_index,
            viewport_x=vx,
            viewport_y=vy,
            viewport_w=vw,
            viewport_h=vh,
        )
        patch = res.get("grid")
        n_up = merge_simulate_into_grid(working_grid, vx, vy, patch)
        out.append(
            {
                "viewport": {"x": vx, "y": vy, "w": vw, "h": vh},
                "queries_used": res.get("queries_used"),
                "queries_max": res.get("queries_max"),
                "cells_updated": n_up,
                "grid_patch": patch,
            }
        )
    return out


def update_explore_from_queries(
    working_grid: np.ndarray,
    batch_result: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "queries_in_batch": len(batch_result),
        "total_cells_updated": sum(r.get("cells_updated", 0) for r in batch_result),
    }


@dataclass
class PredictionBundle:
    final_hw: np.ndarray
    global_hw: np.ndarray
    explore_hw: np.ndarray
    blended_pre_calib: np.ndarray
    mean_entropy: float
    X: np.ndarray


def recompute_prediction(
    *,
    working_grid: np.ndarray,
    settlements: list[dict[str, Any]],
    model: Any,
    feature_set: str,
    neighbor_radius: int,
    calibration: dict[str, Any] | None,
    explore_hier: dict[str, Any] | None,
    explore_hierarchy_mode: str,
    min_regional_samples: int,
    global_weight: float,
    explore_weight: float,
    prob_floor: float,
    calibration_json: Path | None,
    seed_index: int,
) -> PredictionBundle:
    grid_list = working_grid.tolist()
    X = feature_matrix_for_seed(
        grid_list,
        settlements,
        neighbor_radius=neighbor_radius,
        feature_set=feature_set,
    )
    h, w = working_grid.shape
    global_hw = predict_proba_fixed6(model, X).reshape(h, w, NUM_CLASSES)
    c = resolve_explore_scalar_boosts(
        dict(calibration or {}),
        explore_hier,
        seed_index,
        explore_hierarchy_mode,
    )
    explore_hw = build_terrain_prior(
        grid_list,
        settlements,
        eps=DEFAULT_EPS,
        coast_boost=float(c.get("coast_boost", 0.35)),
        near_settlement_boost=float(c.get("near_settlement_boost", 0.5)),
        coast_near_settle_port_boost=float(c.get("coast_near_settle_port_boost", 0.28)),
        dynamic_ruin_weight=float(c.get("dynamic_ruin_weight", 1.0)),
    )
    explore_hw = apply_regional_explore_scales(
        np.asarray(explore_hw, dtype=np.float64),
        explore_hier,
        h,
        w,
        seed_index,
        mode=explore_hierarchy_mode,
        min_region_samples=min_regional_samples,
    )
    s = global_weight + explore_weight
    wg, we = global_weight / s, explore_weight / s
    bl = wg * global_hw + we * explore_hw
    rs = np.sum(bl, axis=-1, keepdims=True)
    rs = np.where(rs > 0, rs, 1.0)
    blended = bl / rs
    flat = blended.reshape(-1, NUM_CLASSES)
    if calibration_json and Path(calibration_json).is_file():
        flat = apply_saved_calibration_json(flat, X, Path(calibration_json))
    tensor = flat.reshape(h, w, NUM_CLASSES)
    final_hw = apply_floor_and_renorm(tensor, eps=float(prob_floor))
    ent = compute_entropy_map(final_hw)
    return PredictionBundle(
        final_hw=final_hw,
        global_hw=global_hw,
        explore_hw=np.asarray(explore_hw, dtype=np.float64),
        blended_pre_calib=blended,
        mean_entropy=float(np.mean(ent)),
        X=X,
    )


def _load_explore_calibration(explore_dir: Path | None) -> dict[str, Any]:
    if explore_dir is None:
        return {}
    p = Path(explore_dir) / "analysis_summary.json"
    if not p.is_file():
        return {}
    s = json.loads(p.read_text(encoding="utf-8"))
    cal = s.get("calibration_suggestion") or {}
    return cal if isinstance(cal, dict) else {}


def _normalize_batch_sizes(parts: list[int], budget: int) -> list[int]:
    if not parts:
        return [budget]
    if sum(parts) == budget:
        return parts
    if sum(parts) < budget:
        return parts[:-1] + [parts[-1] + (budget - sum(parts))]
    out: list[int] = []
    rem = budget
    for z in parts:
        if rem <= 0:
            break
        take = min(z, rem)
        if take > 0:
            out.append(take)
            rem -= take
    return out if out else [budget]


def _min_dist_settlement_layer(h: int, w: int, settlements: list[dict[str, Any]]) -> np.ndarray:
    d = np.full((h, w), np.inf, dtype=np.float64)
    yy, xx = np.ogrid[0:h, 0:w]
    for s in settlements:
        if not s.get("alive", True):
            continue
        sx, sy = int(s["x"]), int(s["y"])
        dist = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2)
        d = np.minimum(d, dist)
    return d


def apply_production_flags(args: argparse.Namespace) -> None:
    if not args.production_calibrated:
        return
    mdir = default_models_dir()
    args.global_weight = 0.7
    args.explore_weight = 0.3
    args.prob_floor = 0.01
    gb = mdir / "gradient_boosting.joblib"
    if gb.is_file():
        if args.model_path is None:
            args.model_path = gb
        if args.model_meta is None:
            args.model_meta = mdir / "gradient_boosting_meta.json"
    pc = mdir / "prob_calibration.json"
    if args.calibration_json is None and pc.is_file():
        args.calibration_json = pc
    print(
        "[active_query] production-calibrated: blend 0.7/0.3 prob_floor=0.01 "
        f"model={args.model_path or 'auto'} calib={args.calibration_json or '—'}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Active query loop: batched simulate + reblend.")
    p.add_argument("--round-id", type=str, default=None)
    p.add_argument("--round-dir", type=Path, default=None, help="Til --dry-run uten API (round_detail.json).")
    p.add_argument("--seed-index", type=int, default=0)
    p.add_argument("--budget", type=int, default=50)
    p.add_argument("--batch-sizes", type=str, default="10,10,30")
    p.add_argument("--viewport-size", type=int, default=12)
    p.add_argument(
        "--explore-dir",
        type=Path,
        default=None,
        help="Mappe med analysis_summary.json (evt. explore_calibration_hierarchy.json). F.eks. . for repo root.",
    )
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--model-meta", type=Path, default=None)
    p.add_argument("--calibration-json", type=Path, default=None)
    p.add_argument("--global-weight", type=float, default=0.7)
    p.add_argument("--explore-weight", type=float, default=0.3)
    p.add_argument("--prob-floor", type=float, default=0.01)
    p.add_argument("--explore-hierarchy-mode", choices=("off", "global", "seed", "full"), default="seed")
    p.add_argument("--min-regional-samples", type=int, default=200)
    p.add_argument("--production-calibrated", action="store_true")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--working-grid-json",
        type=Path,
        default=None,
        help="H×W heltalls-grid (JSON list-of-lists). Fortsett etter tidligere batch; må matche seed fra --seed-index.",
    )
    p.add_argument(
        "--save-working-grid",
        type=Path,
        default=None,
        help="Skriv working_grid (etter alle batcher) som JSON list-of-lists.",
    )
    p.add_argument("--submit-final", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Ingen simulate/submit; krever --round-dir.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    apply_production_flags(args)
    root = _repo_root()

    batch_parts = [int(x.strip()) for x in str(args.batch_sizes).split(",") if x.strip()]
    batches = _normalize_batch_sizes(batch_parts, int(args.budget))
    if sum(batches) != int(args.budget):
        raise SystemExit(f"batch-sizes må summere til --budget: fikk {batches} sum={sum(batches)} budget={args.budget}")

    if args.dry_run:
        if not args.round_dir:
            raise SystemExit("--dry-run krever --round-dir med round_detail.json")
        detail = load_detail_json(Path(args.round_dir) / "round_detail.json")
        round_id = str(detail.get("id", Path(args.round_dir).name))
        client = None
    else:
        client = AstarClient()
        if args.round_id:
            raw = str(args.round_id).strip()
            round_id = resolve_round_identifier(client, raw)
            if raw != round_id:
                print(f"[active_query] round_id: {raw!r} (round_number) -> {round_id}")
            detail = client.get_round(round_id)
        else:
            round_id, detail = get_active_round(client)

    si = int(args.seed_index)
    seeds_count = int(detail.get("seeds_count", 1))
    if si < 0 or si >= seeds_count:
        raise SystemExit(f"seed-index {si} utenfor 0..{seeds_count - 1}")

    grid0, settlements = load_initial_state_for_seed(detail["initial_states"], si)
    working_grid = np.asarray(grid0, dtype=np.int32)
    h, w = working_grid.shape
    if args.working_grid_json:
        wg_path = Path(args.working_grid_json)
        raw = json.loads(wg_path.read_text(encoding="utf-8"))
        loaded = np.asarray(raw, dtype=np.int32)
        if loaded.shape != (h, w):
            raise SystemExit(f"--working-grid-json shape {loaded.shape} forventet {(h, w)}")
        working_grid = loaded
        print(f"[active_query] loaded working_grid from {wg_path}")

    calibration = _load_explore_calibration(Path(args.explore_dir) if args.explore_dir else None)
    explore_hier = load_hierarchy_file(Path(args.explore_dir) if args.explore_dir else None)

    model, meta, ap, _mp, load_src = load_global_model_or_exit(args.model_path, args.model_meta)
    fn = meta.get("feature_names")
    if not fn:
        raise SystemExit("meta mangler feature_names")
    feature_names = [str(x) for x in fn]
    feature_set = str(meta.get("feature_set") or infer_feature_set_from_feature_names(feature_names))
    nr = int(meta.get("neighbor_radius", 1))
    cal_path = Path(args.calibration_json) if args.calibration_json else None

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = root / "astar" / "analysis" / "active_query" / round_id
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "round_id": round_id,
        "seed_index": si,
        "budget": args.budget,
        "batch_sizes": batches,
        "dry_run": args.dry_run,
        "model_path": str(ap),
        "load_source": load_src,
        "feature_set": feature_set,
        "batches": [],
    }

    all_viewports: list[tuple[int, int, int, int]] = []
    bundle: PredictionBundle | None = None

    for bi, nq in enumerate(batches):
        phase = 1 if bi == 0 else (2 if bi == 1 else 3)
        if bundle is None:
            bundle = recompute_prediction(
                working_grid=working_grid,
                settlements=settlements,
                model=model,
                feature_set=feature_set,
                neighbor_radius=nr,
                calibration=calibration,
                explore_hier=explore_hier,
                explore_hierarchy_mode=args.explore_hierarchy_mode,
                min_regional_samples=int(args.min_regional_samples),
                global_weight=args.global_weight,
                explore_weight=args.explore_weight,
                prob_floor=float(args.prob_floor),
                calibration_json=cal_path,
                seed_index=si,
            )
        ent_before = bundle.mean_entropy
        dm = disagreement_map(bundle.global_hw, bundle.explore_hw)
        dist_s = _min_dist_settlement_layer(h, w, settlements)
        # Batch 1: velg ut fra global modells usikkerhet; senere batcher bruker blendet prediksjon.
        ent_map_for_select = (
            compute_entropy_map(bundle.global_hw) if phase == 1 else compute_entropy_map(bundle.final_hw)
        )

        vps = select_query_locations(
            entropy_map=ent_map_for_select,
            disagreement_map_=dm if phase >= 2 else None,
            dist_settlement=dist_s if phase >= 3 else None,
            n_queries=nq,
            h=h,
            w=w,
            vp=int(args.viewport_size),
            phase=phase,
            existing=all_viewports,
        )
        tries = 0
        while len(vps) < nq and tries < 800:
            tries += 1
            cy, cx = int(np.random.randint(0, h)), int(np.random.randint(0, w))
            v = viewport_around(h, w, cy, cx, int(args.viewport_size))
            if v in all_viewports or v in vps:
                continue
            if any(_viewport_iou(v, u) >= 0.5 for u in all_viewports + vps):
                continue
            vps.append(v)

        batch_runs = run_query_batch(
            client,
            round_id=round_id,
            seed_index=si,
            viewports=vps,
            working_grid=working_grid,
            dry_run=bool(args.dry_run),
        )
        all_viewports.extend(vps)
        up_meta = update_explore_from_queries(working_grid, batch_runs)

        bundle = recompute_prediction(
            working_grid=working_grid,
            settlements=settlements,
            model=model,
            feature_set=feature_set,
            neighbor_radius=nr,
            calibration=calibration,
            explore_hier=explore_hier,
            explore_hierarchy_mode=args.explore_hierarchy_mode,
            min_regional_samples=int(args.min_regional_samples),
            global_weight=args.global_weight,
            explore_weight=args.explore_weight,
            prob_floor=float(args.prob_floor),
            calibration_json=cal_path,
            seed_index=si,
        )
        ent_after = bundle.mean_entropy

        pred_list = numpy_to_submission_list(bundle.final_hw)
        errs = validate_prediction(pred_list, h, w, float(args.prob_floor))
        if errs:
            raise RuntimeError("validation:\n" + "\n".join(errs[:15]))

        batch_record = {
            "batch_index": bi,
            "phase": phase,
            "n_queries": nq,
            "mean_entropy_before": ent_before,
            "mean_entropy_after": ent_after,
            "viewports": [r.get("viewport") for r in batch_runs],
            "simulate_meta": batch_runs,
            "explore_update": up_meta,
        }
        report["batches"].append(batch_record)

        pred_path = out_dir / f"prediction_seed{si}_batch{bi}.json"
        pred_path.write_text(json.dumps(pred_list), encoding="utf-8")
        print(
            f"[batch {bi}] phase={phase} queries={nq} mean_entropy {ent_before:.5f} -> {ent_after:.5f} "
            f"wrote {pred_path.name}"
        )

    final_path = out_dir / f"prediction_seed{si}_final.json"
    assert bundle is not None
    final_list = numpy_to_submission_list(bundle.final_hw)
    final_path.write_text(json.dumps(final_list), encoding="utf-8")
    report["final_prediction_path"] = str(final_path)
    report["completed_at"] = datetime.now(timezone.utc).isoformat()

    report_path = out_dir / f"active_query_report_seed{si}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {report_path}")

    if args.save_working_grid:
        sgp = Path(args.save_working_grid)
        sgp.parent.mkdir(parents=True, exist_ok=True)
        sgp.write_text(json.dumps(working_grid.tolist()), encoding="utf-8")
        print(f"[active_query] saved working_grid -> {sgp}")

    if args.submit_final and not args.dry_run:
        assert client is not None
        r = client.submit_prediction(round_id=round_id, seed_index=si, prediction=final_list)
        print(f"submit-final: {r.status_code} {r.text[:300]}")


if __name__ == "__main__":
    main()
