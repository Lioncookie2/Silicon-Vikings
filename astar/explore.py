"""
Strategic use of the 50-query budget across seeds.

Run from repo root:
  PYTHONPATH=. python -m astar.explore
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .client import AstarClient, get_active_round_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Explore Astar Island with viewport queries.")
    p.add_argument("--max-queries", type=int, default=50, help="Stop after N simulate calls.")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: astar/analysis/last_explore.json under repo).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.out or (root / "astar" / "analysis" / "last_explore.json")

    client = AstarClient()
    rid, detail = get_active_round_id(client)
    h = int(detail["map_height"])
    w = int(detail["map_width"])
    seeds = int(detail.get("seeds_count", 5))

    results: list[dict] = []
    queries = 0
    vp = 15
    stride = 12

    for seed_idx in range(min(seeds, 5)):
        y = 0
        while y < h and queries < args.max_queries:
            x = 0
            while x < w and queries < args.max_queries:
                vw = min(vp, w - x)
                vh = min(vp, h - y)
                if vw < 5 or vh < 5:
                    break
                res = client.simulate(
                    round_id=rid,
                    seed_index=seed_idx,
                    viewport_x=x,
                    viewport_y=y,
                    viewport_w=vw,
                    viewport_h=vh,
                )
                results.append({"seed_index": seed_idx, "viewport": (x, y, vw, vh), "result": res})
                queries += 1
                x += stride
            y += stride

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"round_id": rid, "queries": queries, "data": results}, indent=2))
    print(f"Wrote {out} ({queries} queries)")


if __name__ == "__main__":
    main()
