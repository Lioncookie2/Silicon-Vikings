"""
Hent historiske completed Astar-runder + post-round analysis per seed.

Kjor:
  PYTHONPATH=. python -m astar.fetch_historical
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

from .client import AstarClient


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hent historical Astar-data til disk.")
    p.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Default: astar/data/historical",
    )
    p.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Valgfri begrensning pa antall completed rounds.",
    )
    p.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Kun rounds + round_detail, ikke /analysis per seed.",
    )
    return p.parse_args()


def _safe_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = _repo_root()
    out_root = args.out_root or (root / "astar" / "data" / "historical")
    out_root.mkdir(parents=True, exist_ok=True)

    client = AstarClient()
    rounds = client.list_rounds()
    completed = [r for r in rounds if r.get("status") == "completed"]
    completed.sort(key=lambda r: int(r.get("round_number", 0)))

    if args.max_rounds is not None:
        completed = completed[: args.max_rounds]

    manifest = {
        "total_rounds_listed": len(rounds),
        "completed_selected": len(completed),
        "round_ids": [str(r["id"]) for r in completed],
        "errors": [],
    }

    for i, r in enumerate(completed, start=1):
        rid = str(r["id"])
        print(f"[{i}/{len(completed)}] round {r.get('round_number')} id={rid}")
        round_dir = out_root / rid
        round_dir.mkdir(parents=True, exist_ok=True)
        _safe_json_write(round_dir / "round_list_entry.json", r)

        try:
            detail = client.get_round(rid)
            _safe_json_write(round_dir / "round_detail.json", detail)
        except requests.HTTPError as e:
            msg = f"round_detail failed rid={rid}: {e}"
            print("  !", msg)
            manifest["errors"].append(msg)
            continue

        if args.skip_analysis:
            continue

        seeds_count = int(detail.get("seeds_count", 5))
        for seed_index in range(seeds_count):
            out_path = round_dir / "analysis" / f"seed_{seed_index}.json"
            try:
                analysis = client.get_analysis(rid, seed_index)
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else "n/a"
                msg = f"analysis unavailable rid={rid} seed={seed_index} status={status}"
                print("  -", msg)
                manifest["errors"].append(msg)
                continue

            payload = {
                "round_id": rid,
                "round_number": r.get("round_number"),
                "seed_index": seed_index,
                "width": analysis.get("width"),
                "height": analysis.get("height"),
                "score": analysis.get("score"),
                "initial_grid": analysis.get("initial_grid")
                or (detail.get("initial_states", [{}])[seed_index].get("grid") if detail.get("initial_states") else None),
                "ground_truth": analysis.get("ground_truth"),
                "prediction": analysis.get("prediction"),
                "raw_analysis": analysis,
            }
            _safe_json_write(out_path, payload)
            print(f"  saved analysis seed={seed_index}")

    _safe_json_write(out_root / "manifest.json", manifest)
    print(f"Done. Wrote historical data to {out_root}")


if __name__ == "__main__":
    main()
