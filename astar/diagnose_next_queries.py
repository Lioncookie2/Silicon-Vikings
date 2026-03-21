"""
Lettvekts-diagnose: foreslå neste simulate-viewports basert på global prediksjon + explore-dekning.

Heuristikker (enkle):
- høy prediksjonsentropi per celle
- høy sannsynlighet for settlement/port/ruin (klasser 1–3)
- initial-grid: «overgang» (varierte nabo-terrengkoder)
- lavere prioritet der explore allerede har dekket cellen; seeds med færre runs får litt boost

Kjor:
  PYTHONPATH=. python -m astar.diagnose_next_queries --round-dir ... [--explore-dir ...] -o next_queries.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .baseline import load_initial_state_for_seed
from .cell_features import feature_matrix_for_seed
from .global_ensemble import ensemble_predict_proba_fixed6, load_global_ensemble
from .global_model_loader import load_global_model_or_exit
from .global_model_paths import models_dir as default_models_dir
from .grid_codes import FOREST, MOUNTAIN, OCEAN, PLAINS, PORT, RUIN, SETTLEMENT
from .sklearn_utils import predict_proba_fixed6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Foreslå neste simulate-queries (heuristikk).")
    p.add_argument("--round-dir", type=Path, required=True, help="Mappe med round_detail.json")
    p.add_argument("--explore-dir", type=Path, default=None, help="Explore-mappe med runs/ (dekning)")
    p.add_argument("--models-dir", type=Path, default=None, help="Default: astar/data/models")
    p.add_argument(
        "--use-ensemble",
        action="store_true",
        help="Bruk GB+LR+RF-ensemble som i predict --baseline global_ensemble (ellers best enkeltmodell).",
    )
    p.add_argument("--viewport", type=int, default=15, help="Foreslått viewport-størrelse (kvadrat).")
    p.add_argument("--top", type=int, default=5, help="Antall topp-forslag.")
    p.add_argument("-o", "--out-json", type=Path, default=None, help="Skriv JSON hit.")
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _entropy_map(p: np.ndarray) -> np.ndarray:
    """p: (H,W,6)"""
    x = np.clip(p, 1e-12, 1.0)
    return -np.sum(x * np.log(x), axis=-1)


def _dynamic_map(p: np.ndarray) -> np.ndarray:
    return np.sum(p[..., 1:4], axis=-1)


def _transition_map(grid: np.ndarray) -> np.ndarray:
    """Høy score når naboer har ulike terrengkoder (enkel kant-detektor)."""
    h, w = grid.shape
    out = np.zeros((h, w), dtype=np.float64)
    for y in range(h):
        for x in range(w):
            codes = set()
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    codes.add(int(grid[ny, nx]))
            out[y, x] = float(len(codes))
    # Boost land/hav-kant + skog
    g = grid
    coast = np.zeros((h, w), dtype=np.float64)
    for y in range(h):
        for x in range(w):
            c = int(g[y, x])
            if c in (MOUNTAIN,):
                continue
            if c == OCEAN:
                continue
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and int(g[ny, nx]) == OCEAN:
                    coast[y, x] = 1.0
                    break
    forest_adj = np.zeros((h, w), dtype=np.float64)
    for y in range(h):
        for x in range(w):
            if int(g[y, x]) != FOREST:
                continue
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    nc = int(g[ny, nx])
                    if nc in (SETTLEMENT, PORT, RUIN, PLAINS, EMPTY):
                        forest_adj[y, x] = 1.0
                        break
    return out + 0.5 * coast + 0.3 * forest_adj


def _coverage_mask_and_runs_per_seed(
    explore_dir: Path | None, h: int, w: int, seeds: int
) -> tuple[list[np.ndarray], list[int]]:
    masks = [np.zeros((h, w), dtype=bool) for _ in range(seeds)]
    runs_count = [0] * seeds
    if explore_dir is None:
        return masks, runs_count
    rd = Path(explore_dir) / "runs"
    if not rd.is_dir():
        return masks, runs_count
    for fp in sorted(rd.glob("run_*.json")):
        try:
            r = _load_json(fp)
        except (json.JSONDecodeError, OSError):
            continue
        si = int(r.get("seed_index", -1))
        if not (0 <= si < seeds):
            continue
        runs_count[si] += 1
        vp = r.get("viewport") or {}
        vx, vy = int(vp.get("x", 0)), int(vp.get("y", 0))
        vw, vh = int(vp.get("w", 0)), int(vp.get("h", 0))
        for yy in range(vy, min(h, vy + vh)):
            for xx in range(vx, min(w, vx + vw)):
                masks[si][yy, xx] = True
    return masks, runs_count


def _viewport_from_center(cx: int, cy: int, w: int, h: int, vp: int) -> tuple[int, int, int, int]:
    half = vp // 2
    vx = max(0, min(w - vp, cx - half))
    vy = max(0, min(h - vp, cy - half))
    vw = min(vp, w - vx)
    vh = min(vp, h - vy)
    return vx, vy, vw, vh


def main() -> None:
    args = parse_args()
    md = Path(args.models_dir) if args.models_dir else default_models_dir()
    detail = _load_json(Path(args.round_dir) / "round_detail.json")
    h = int(detail["map_height"])
    w = int(detail["map_width"])
    seeds = int(detail.get("seeds_count", 5))
    round_id = str(detail.get("id", ""))

    if args.use_ensemble:
        ensemble, neighbor_radius, ens_desc = load_global_ensemble(md)
        pred_mode = f"ensemble[{ens_desc}]"
    else:
        model, _meta, _ap, _mp, load_src = load_global_model_or_exit(None, None)
        ensemble = None
        neighbor_radius = 1
        pred_mode = f"single({load_src})"

    cov_masks, runs_per = _coverage_mask_and_runs_per_seed(args.explore_dir, h, w, seeds)

    max_runs = max(runs_per) if runs_per else 0

    candidates: list[dict[str, Any]] = []

    for seed_idx in range(seeds):
        grid, settlements = load_initial_state_for_seed(detail["initial_states"], seed_idx)
        gnp = np.asarray(grid, dtype=np.int32)
        X = feature_matrix_for_seed(grid, settlements, neighbor_radius=neighbor_radius)
        if ensemble is not None:
            flat = ensemble_predict_proba_fixed6(ensemble, X)
        else:
            flat = predict_proba_fixed6(model, X)
        p = flat.reshape(h, w, 6)

        ent = _entropy_map(p)
        dyn = _dynamic_map(p)
        trans = _transition_map(gnp)
        covered = cov_masks[seed_idx]

        # Normaliser til omtrent [0,1] per kart for enkel addisjon
        def _norm01(a: np.ndarray) -> np.ndarray:
            lo, hi = float(np.min(a)), float(np.max(a))
            if hi <= lo:
                return np.zeros_like(a)
            return (a - lo) / (hi - lo + 1e-12)

        e_n = _norm01(ent)
        d_n = _norm01(dyn)
        t_n = _norm01(trans)
        uncov = np.where(covered, 0.35, 1.0)
        seed_boost = (1.0 + max_runs) / (1.0 + runs_per[seed_idx]) if max_runs > 0 else 1.0
        score = (1.0 * e_n + 0.85 * d_n + 0.55 * t_n) * uncov * float(seed_boost)

        flat_score = score.ravel()
        top_idx = int(np.argmax(flat_score))
        cy, cx = divmod(top_idx, w)
        s_max = float(flat_score[top_idx])
        vx, vy, vw, vh = _viewport_from_center(cx, cy, w, h, args.viewport)

        candidates.append(
            {
                "rank_key": s_max,
                "seed_index": seed_idx,
                "focus_cell": {"x": cx, "y": cy},
                "viewport": {"x": vx, "y": vy, "w": vw, "h": vh},
                "score": round(s_max, 6),
                "runs_so_far_this_seed": runs_per[seed_idx],
                "mean_entropy_map": round(float(np.mean(ent)), 6),
                "reason": "max kombinasjon (entropy+dynamikk+overgang) med lav dekning-penalty",
            }
        )

    # Global topp: sorter kandidater per seed, ta top N
    candidates.sort(key=lambda x: -x["rank_key"])
    topn = candidates[: args.top]

    out: dict[str, Any] = {
        "round_id": round_id,
        "prediction_source": pred_mode,
        "map_size": {"width": w, "height": h},
        "seeds_count": seeds,
        "explore_dir": str(args.explore_dir) if args.explore_dir else None,
        "runs_per_seed": {str(i): runs_per[i] for i in range(seeds)},
        "recommended_next_queries": topn,
        "note": "Heuristikk — ikke garantert optimal. Bruk som simulate viewport_x,y,w,h.",
    }

    text = json.dumps(out, indent=2, ensure_ascii=False)
    print(text)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text, encoding="utf-8")
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
