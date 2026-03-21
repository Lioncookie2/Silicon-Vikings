"""
Bygg treningsdatasett fra historiske runder (celleniva).

Kjor:
  PYTHONPATH=. python -m astar.build_dataset
  PYTHONPATH=. python -m astar.build_dataset --latest-n-rounds 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .cell_features import (
    FEATURE_SET_FULL,
    FEATURE_SET_SETTLEMENT_PORT,
    FEATURE_SET_SETTLEMENT_PORT_RADIUS,
    all_feature_names,
)
from .historical_cells_loader import (
    audit_settlement_keys_in_historical,
    load_historical_cells,
    parse_round_ids_csv,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bygg historical treningsdatasett.")
    p.add_argument("--historical-root", type=Path, default=None, help="Default: astar/data/historical")
    p.add_argument("--out-dir", type=Path, default=None, help="Default: astar/data/datasets")
    p.add_argument(
        "--out-npz",
        type=Path,
        default=None,
        help="Output .npz (default: <out-dir>/historical_cells.npz).",
    )
    p.add_argument("--neighbor-radius", type=int, default=1)
    p.add_argument(
        "--round-ids",
        type=str,
        default=None,
        help="Kommaseparerte round UUID-er (høyeste prioritet).",
    )
    p.add_argument(
        "--latest-n-rounds",
        type=int,
        default=None,
        help="Ta med de N siste fullforte runder (sortert etter round_number).",
    )
    p.add_argument(
        "--first-n-rounds",
        type=int,
        default=None,
        help="Ta med de N eldste runder (sortert etter round_number).",
    )
    p.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Alias for --first-n-rounds (hurtig subset).",
    )
    p.add_argument(
        "--feature-set",
        choices=(
            FEATURE_SET_FULL,
            FEATURE_SET_SETTLEMENT_PORT,
            FEATURE_SET_SETTLEMENT_PORT_RADIUS,
        ),
        default=FEATURE_SET_FULL,
        help="full=8; settlement_port=+3 port; settlement_port_radius=+4 radius-tellinger (Chebyshev).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = _repo_root()
    historical_root = args.historical_root or (root / "astar" / "data" / "historical")
    out_dir = args.out_dir or (root / "astar" / "data" / "datasets")
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.out_npz or (out_dir / "historical_cells.npz")

    first_n = args.first_n_rounds if args.first_n_rounds is not None else args.max_rounds

    settlement_audit = audit_settlement_keys_in_historical(historical_root)

    X, y_argmax, y_prob, meta_rows, stats = load_historical_cells(
        historical_root,
        neighbor_radius=args.neighbor_radius,
        feature_set=args.feature_set,
        round_ids=parse_round_ids_csv(args.round_ids),
        latest_n_rounds=args.latest_n_rounds,
        first_n_rounds=first_n,
    )

    fnames = np.asarray(all_feature_names(args.feature_set))
    np.savez_compressed(
        npz_path,
        X=X,
        y_argmax=y_argmax,
        y_prob=y_prob,
        feature_names=fnames,
        feature_set=np.asarray(args.feature_set),
    )

    meta_path = out_dir / "historical_cells_meta.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for m in meta_rows:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    class_counts = np.zeros(6, dtype=np.int64)
    for i in range(len(y_argmax)):
        class_counts[int(y_argmax[i])] += 1

    report: dict[str, Any] = {
        "historical_root": str(historical_root),
        "out_npz": str(npz_path),
        "load_stats": stats,
        "settlement_fields_audit": settlement_audit,
        "feature_set": args.feature_set,
        "feature_names": fnames.tolist(),
        "n_cells": int(len(X)),
        "class_distribution_argmax": {str(i): int(class_counts[i]) for i in range(6)},
        "metadata_jsonl": str(meta_path),
    }
    report_path = out_dir / "dataset_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== build_dataset ===")
    print(f"filter: {stats['filter_mode']}")
    print(f"runder: {len(stats['round_ids_used'])}  seeds: {stats['seeds_loaded']}  celler: {stats['cells_loaded']}")
    print(f"feature_set: {args.feature_set}  n_features: {X.shape[1]}")
    print(f"settlement audit: rich_stats={settlement_audit.get('rich_economic_stats_present')}")
    print(f"Wrote {npz_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
