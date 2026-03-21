"""
Datainnsamling: aktiv runde, lagring av initial states, strukturert simulate med metadata.

Fra repo-roten:
  PYTHONPATH=. python -m astar.collect --max-queries 50
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .client import AstarClient, get_active_round
from .exploration import allocate_queries_per_seed, balanced_viewports, stride_cover_viewports


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dedupe_viewports(
    h: int,
    w: int,
    budget: int,
    vp: int = 15,
) -> list[tuple[int, int, int, int]]:
    """Combine balanced + stride cover until ``budget`` unique viewports."""
    if budget <= 0:
        return []
    seen: set[tuple[int, int, int, int]] = set()
    out: list[tuple[int, int, int, int]] = []

    for t in balanced_viewports(h, w, vp=vp, max_patches=min(10, budget)):
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= budget:
            return out

    for t in stride_cover_viewports(h, w, budget - len(out), vp=vp):
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= budget:
            break
    return out[:budget]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect Astar Island simulate observations.")
    p.add_argument("--max-queries", type=int, default=50, help="Total simulate calls (default 50).")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Base directory (default: astar/analysis/rounds/<round_id>).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Lagre round snapshot og skriv viewport-plan; ingen simulate-kall (0 queries).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = _repo_root()

    client = AstarClient()
    round_id, detail = get_active_round(client)

    h = int(detail["map_height"])
    w = int(detail["map_width"])
    seeds_count = int(detail.get("seeds_count", 5))

    out_base = args.out_dir or (root / "astar" / "analysis" / "rounds" / round_id)
    out_base.mkdir(parents=True, exist_ok=True)

    detail_path = out_base / "round_detail.json"
    detail_path.write_text(json.dumps(detail, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest: dict[str, Any] = {
        "round_id": round_id,
        "map_height": h,
        "map_width": w,
        "seeds_count": seeds_count,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "max_queries": args.max_queries,
        "strategy": "per_seed_balanced_stride",
    }
    (out_base / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    per_seed = allocate_queries_per_seed(args.max_queries, seeds_count)

    if args.dry_run:
        plan: dict[str, Any] = {"per_seed_budget": per_seed, "viewports": {}}
        for seed_idx in range(seeds_count):
            plan["viewports"][str(seed_idx)] = _dedupe_viewports(h, w, per_seed[seed_idx])
        plan_path = out_base / "exploration_plan.json"
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print(f"dry-run: wrote {detail_path}, {plan_path} (no simulate calls)")
        for seed_idx in range(seeds_count):
            print(f"  seed {seed_idx}: {per_seed[seed_idx]} viewports -> {plan['viewports'][str(seed_idx)]}")
        return

    runs: list[dict[str, Any]] = []
    query_index = 0

    for seed_idx in range(seeds_count):
        budget = per_seed[seed_idx]
        viewports = _dedupe_viewports(h, w, budget)
        for vx, vy, vw, vh in viewports:
            res = client.simulate(
                round_id=round_id,
                seed_index=seed_idx,
                viewport_x=vx,
                viewport_y=vy,
                viewport_w=vw,
                viewport_h=vh,
            )
            run: dict[str, Any] = {
                "query_index": query_index,
                "seed_index": seed_idx,
                "viewport": {
                    "x": vx,
                    "y": vy,
                    "w": vw,
                    "h": vh,
                },
                "grid": res.get("grid"),
                "settlements": res.get("settlements"),
                "queries_used": res.get("queries_used"),
                "queries_max": res.get("queries_max"),
                "width": res.get("width"),
                "height": res.get("height"),
                "viewport_response": res.get("viewport"),
            }
            runs.append(run)
            query_index += 1
            print(
                f"query {query_index}/{args.max_queries} seed={seed_idx} "
                f"vp=({vx},{vy},{vw},{vh}) used={res.get('queries_used')}"
            )

    collection = {
        **manifest,
        "runs": runs,
    }
    (out_base / "simulate_runs.json").write_text(
        json.dumps(collection, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {out_base} ({len(runs)} simulate calls)")


if __name__ == "__main__":
    main()
