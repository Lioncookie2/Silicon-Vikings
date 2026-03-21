"""
Oppsummer ett Astar exploration-run (round_detail.json + runs/run_*.json).

  PYTHONPATH=. python -m astar.analyze_explore astar/analysis/explore/<mappe>
  PYTHONPATH=. python -m astar.analyze_explore astar/analysis/explore/<mappe> -o summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .grid_codes import count_viewport_cells


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Oppsummer exploration-data lokalt.")
    p.add_argument(
        "explore_dir",
        type=Path,
        help="Mappe med round_detail.json og runs/",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Skriv summary.json hit (default: <explore_dir>/analysis_summary.json).",
    )
    return p.parse_args()


def analyze(explore_dir: Path) -> dict[str, Any]:
    explore_dir = explore_dir.resolve()
    detail_path = explore_dir / "round_detail.json"
    if not detail_path.is_file():
        raise FileNotFoundError(detail_path)

    detail = _load_json(detail_path)
    runs_dir = explore_dir / "runs"
    run_files = sorted(runs_dir.glob("run_*.json"))
    manifest_path = explore_dir / "manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.is_file() else {}

    round_id = str(detail.get("id", ""))
    mw = int(detail.get("map_width", 0))
    mh = int(detail.get("map_height", 0))
    seeds_count = int(detail.get("seeds_count", len(detail.get("initial_states", []))))

    per_run: list[dict[str, Any]] = []
    agg_class_counts = {"empty_like": 0, "settlement": 0, "port": 0, "ruin": 0, "forest": 0, "mountain": 0}
    agg_coastal = {"coastal": 0, "non_coastal": 0}
    agg_near = {"near_initial_settlement": 0, "far_from_initial_settlement": 0}
    terrain_to_final: dict[str, dict[str, int]] = {}

    def _terrain_bucket(c: int) -> str:
        return "empty_like" if c in (0, 10, 11) else {1: "settlement", 2: "port", 3: "ruin", 4: "forest", 5: "mountain"}.get(c, "empty_like")

    initial_states = detail.get("initial_states", [])

    def _is_coastal(seed_idx: int, x: int, y: int) -> bool:
        grid = initial_states[seed_idx]["grid"]
        c = grid[y][x]
        if c in (10, 5):
            return False
        h = len(grid)
        w = len(grid[0]) if h else 0
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and grid[ny][nx] == 10:
                return True
        return False

    def _is_near_initial_settlement(seed_idx: int, x: int, y: int, dist: float = 6.0) -> bool:
        st = initial_states[seed_idx].get("settlements", [])
        for s in st:
            if not s.get("alive", True):
                continue
            dx = float(x - int(s["x"]))
            dy = float(y - int(s["y"]))
            if (dx * dx + dy * dy) ** 0.5 <= dist:
                return True
        return False

    for fp in run_files:
        r = _load_json(fp)
        grid = r.get("grid") or []
        st = r.get("settlements") or []
        counts = count_viewport_cells(grid)
        for k in agg_class_counts:
            agg_class_counts[k] += int(counts[k])

        seed_idx = int(r.get("seed_index", 0))
        vp = r.get("viewport") or {}
        vx, vy = int(vp.get("x", 0)), int(vp.get("y", 0))
        if seed_idx < len(initial_states):
            initial_grid = initial_states[seed_idx]["grid"]
            for yy, row in enumerate(grid):
                for xx, final_c in enumerate(row):
                    gx, gy = vx + xx, vy + yy
                    if gy >= len(initial_grid) or gx >= len(initial_grid[0]):
                        continue
                    init_c = int(initial_grid[gy][gx])
                    init_k = _terrain_bucket(init_c)
                    final_k = _terrain_bucket(int(final_c))
                    terrain_to_final.setdefault(init_k, {})
                    terrain_to_final[init_k][final_k] = terrain_to_final[init_k].get(final_k, 0) + 1
                    if _is_coastal(seed_idx, gx, gy):
                        agg_coastal["coastal"] += 1
                    else:
                        agg_coastal["non_coastal"] += 1
                    if _is_near_initial_settlement(seed_idx, gx, gy):
                        agg_near["near_initial_settlement"] += 1
                    else:
                        agg_near["far_from_initial_settlement"] += 1

        per_run.append(
            {
                "file": fp.name,
                "seed_index": r.get("seed_index"),
                "viewport": r.get("viewport"),
                "terrain_counts": counts,
                "n_settlements_in_viewport": len(st),
                "queries_used": r.get("queries_used"),
                "queries_max": r.get("queries_max"),
            }
        )

    seeds_observed = sorted(
        {int(p["seed_index"]) for p in per_run if p.get("seed_index") is not None}
    )
    viewports = [p["viewport"] for p in per_run]
    last_q_used = last_q_max = None
    if per_run:
        last_q_used = per_run[-1].get("queries_used")
        last_q_max = per_run[-1].get("queries_max")

    total_cells = max(1, sum(agg_class_counts.values()))
    dynamic = agg_class_counts["settlement"] + agg_class_counts["port"] + agg_class_counts["ruin"]
    dynamic_share = dynamic / float(total_cells)

    # Simple calibration proposal from observations
    coastal_total = max(1, agg_coastal["coastal"] + agg_coastal["non_coastal"])
    coastal_share = agg_coastal["coastal"] / float(coastal_total)
    port_share = agg_class_counts["port"] / float(total_cells)
    ruin_share = agg_class_counts["ruin"] / float(total_cells)
    settlement_share = agg_class_counts["settlement"] / float(total_cells)
    near_total = max(1, agg_near["near_initial_settlement"] + agg_near["far_from_initial_settlement"])
    near_share = agg_near["near_initial_settlement"] / float(near_total)

    calibration = {
        "coast_boost": round(0.20 + 0.80 * min(0.25, max(0.02, port_share)) / 0.25, 4),
        "coast_near_settle_port_boost": round(0.12 + 0.8 * min(0.20, max(0.01, port_share)) / 0.20, 4),
        "near_settlement_boost": round(0.30 + 1.2 * min(0.30, max(0.03, settlement_share)) / 0.30, 4),
        "dynamic_ruin_weight": round(0.20 + 1.2 * min(0.25, max(0.01, ruin_share)) / 0.25, 4),
        "coast_observed_share": round(coastal_share, 4),
        "near_initial_observed_share": round(near_share, 4),
        "dynamic_observed_share": round(dynamic_share, 4),
        "source_explore_dir": str(explore_dir),
    }

    return {
        "explore_dir": str(explore_dir),
        "round_id": round_id,
        "round_number": detail.get("round_number"),
        "status": detail.get("status"),
        "map_width": mw,
        "map_height": mh,
        "seeds_count": seeds_count,
        "manifest": manifest,
        "num_run_files_on_disk": len(run_files),
        "num_viewport_results": len(per_run),
        "seeds_observed": seeds_observed,
        "viewports_used": viewports,
        "queries_after_last_run": {"queries_used": last_q_used, "queries_max": last_q_max},
        "aggregate": {
            "class_counts": agg_class_counts,
            "share_settlement": round(settlement_share, 4),
            "share_port": round(port_share, 4),
            "share_ruin": round(ruin_share, 4),
            "coastal_vs_non_coastal": agg_coastal,
            "near_vs_far_initial_settlement": agg_near,
            "initial_terrain_to_observed_final": terrain_to_final,
        },
        "calibration_suggestion": calibration,
        "per_run": per_run,
    }


def main() -> None:
    args = parse_args()
    summary = analyze(args.explore_dir)
    out = args.output or (args.explore_dir.resolve() / "analysis_summary.json")
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== Astar exploration-oppsummering ===")
    print(
        f"[analyze_explore] round_id={summary['round_id']} — leser "
        f"alle {summary['num_run_files_on_disk']} run_*.json i runs/ "
        f"(samlet historikk for mappen, ikke bare siste batch)"
    )
    print(f"Mappe: {summary['explore_dir']}")
    print(f"round_id: {summary['round_id']}")
    print(f"Kart: {summary['map_width']}×{summary['map_height']}, seeds (runde): {summary['seeds_count']}")
    print(
        f"Antall run-filer: {summary['num_run_files_on_disk']} "
        f"→ aggregerte viewport-resultater: {summary['num_viewport_results']}"
    )
    print(f"Observert seed_index: {summary['seeds_observed']}")
    q = summary["queries_after_last_run"]
    print(f"Siste kall — queries_used/max: {q['queries_used']}/{q['queries_max']}")
    agg = summary["aggregate"]
    print(
        "Agg andeler: "
        f"settlement={agg['share_settlement']:.4f} "
        f"port={agg['share_port']:.4f} "
        f"ruin={agg['share_ruin']:.4f}"
    )
    print(f"Kyst/non-kyst: {agg['coastal_vs_non_coastal']}")
    print(f"Nær/fjern initial settlement: {agg['near_vs_far_initial_settlement']}")
    print(f"Kalibrering (forslag): {summary['calibration_suggestion']}")
    print()
    for pr in summary["per_run"]:
        print(f"  {pr['file']}  seed={pr['seed_index']}  vp={pr['viewport']}")
        tc = pr["terrain_counts"]
        print(
            f"    terreng i viewport: Empty={tc['empty_like']}  Settlement={tc['settlement']}  "
            f"Port={tc['port']}  Ruin={tc['ruin']}  Forest={tc['forest']}  Mountain={tc['mountain']}"
        )
        print(f"    settlements i liste: {pr['n_settlements_in_viewport']}")
    print()
    print(f"Skrev {out}")


if __name__ == "__main__":
    main()
