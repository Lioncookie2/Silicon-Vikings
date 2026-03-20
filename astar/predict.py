"""
Build and submit W×H×6 terrain probability tensors for all seeds.

Run from repo root:
  PYTHONPATH=. python -m astar.predict

Uniform baseline with epsilon floor (never exact 0.0 on any class).
"""
from __future__ import annotations

import argparse

from .client import AstarClient, get_active_round_id

NUM_CLASSES = 6
EPS = 0.01


def uniform_tensor(height: int, width: int) -> list[list[list[float]]]:
    """height x width x NUM_CLASSES; inner list is class probabilities."""
    cell = [1.0 / NUM_CLASSES] * NUM_CLASSES
    cell = [max(EPS, v) for v in cell]
    s = sum(cell)
    cell = [c / s for c in cell]
    return [[list(cell) for _ in range(width)] for _ in range(height)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit Astar Island predictions.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print tensor shape only, do not call API (no ACCESS_TOKEN needed).",
    )
    p.add_argument("--dry-height", type=int, default=40)
    p.add_argument("--dry-width", type=int, default=40)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        pred = uniform_tensor(args.dry_height, args.dry_width)
        print(
            f"dry-run shape={len(pred)}x{len(pred[0])}x{len(pred[0][0])} "
            f"(sum first cell={sum(pred[0][0]):.6f})"
        )
        return

    client = AstarClient()
    rid, detail = get_active_round_id(client)
    h = int(detail["map_height"])
    w = int(detail["map_width"])
    seeds = int(detail.get("seeds_count", 5))

    pred = uniform_tensor(h, w)

    for seed_idx in range(seeds):
        r = client.submit_prediction(round_id=rid, seed_index=seed_idx, prediction=pred)
        print(f"seed {seed_idx}: {r.status_code} {r.text[:200]}")


if __name__ == "__main__":
    main()
