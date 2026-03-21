"""
Last celle-rader fra astar/data/historical med valgfri runde-filtrering.

Delt av build_dataset, train_model (hurtig subset) og evaluate_historical_rounds.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .cell_features import compute_settlement_maps, feature_row_at

NUM_CLASSES = 6


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_round_ids_csv(s: str | None) -> set[str] | None:
    """Kommaseparerte UUID-er; None hvis tom/ikke satt."""
    if not s or not str(s).strip():
        return None
    out = {p.strip() for p in str(s).split(",") if p.strip()}
    return out or None


def discover_historical_round_dirs(historical_root: Path) -> list[Path]:
    """Mapper med round_detail.json og analysis/."""
    out: list[Path] = []
    if not historical_root.is_dir():
        return out
    for p in sorted(historical_root.iterdir()):
        if not p.is_dir():
            continue
        if not (p / "round_detail.json").is_file():
            continue
        if not (p / "analysis").is_dir():
            continue
        out.append(p)
    return out


def round_meta_for_dir(rd: Path) -> tuple[str, int]:
    """(round_id, round_number)."""
    d = _load_json(rd / "round_detail.json")
    return str(d.get("id", rd.name)), int(d.get("round_number", 0))


def filter_round_dirs(
    dirs: list[Path],
    *,
    round_ids: set[str] | None,
    latest_n_rounds: int | None,
    first_n_rounds: int | None,
) -> tuple[list[Path], str]:
    """
    Returnerer (filtrerte stier, modusbeskrivelse).
    Prioritet: round_ids > latest_n_rounds > first_n_rounds > alle.
    """
    keyed = [(p, *round_meta_for_dir(p)) for p in dirs]
    if round_ids:
        chosen = [t[0] for t in keyed if t[1] in round_ids]
        return chosen, f"round_ids={sorted(round_ids)}"
    sorted_by_num = sorted(keyed, key=lambda t: (t[2], t[1]))
    if latest_n_rounds is not None and latest_n_rounds > 0:
        chosen = [t[0] for t in sorted_by_num[-latest_n_rounds:]]
        return chosen, f"latest_n_rounds={latest_n_rounds}"
    if first_n_rounds is not None and first_n_rounds > 0:
        chosen = [t[0] for t in sorted_by_num[:first_n_rounds]]
        return chosen, f"first_n_rounds={first_n_rounds}"
    return [t[0] for t in sorted_by_num], "all_rounds"


def audit_settlement_keys_in_historical(historical_root: Path, *, max_sample_rounds: int = 8) -> dict[str, Any]:
    """Skann settlement-objekter i round_detail initial_states."""
    dirs = discover_historical_round_dirs(historical_root)[:max_sample_rounds]
    keys: set[str] = set()
    n_settlements = 0
    for rd in dirs:
        detail = _load_json(rd / "round_detail.json")
        for st in detail.get("initial_states", []):
            for s in st.get("settlements", []):
                n_settlements += 1
                if isinstance(s, dict):
                    keys.update(s.keys())
    rich = {"population", "food", "wealth", "defense", "owner_id", "faction"} & keys
    return {
        "sampled_round_dirs": len(dirs),
        "settlements_sampled": n_settlements,
        "observed_settlement_keys": sorted(keys),
        "rich_economic_stats_present": bool(rich),
        "rich_keys_found": sorted(rich),
        "conclusion": (
            "Lagrede historical round_detail-settlements inneholder typisk kun x,y,has_port,alive. "
            "Population/food/wealth/defense finnes ikke i disse JSON-ene — bruk has_port-baserte features "
            "eller hent rik stats fra simulate (explore) til runde-spesifikk kalibrering, ikke global trening."
            if not rich
            else "Noen runder har ekstra settlement-felt; vurder leakage for hvert felt for global bruk."
        ),
    }


def load_historical_cells(
    historical_root: Path,
    *,
    neighbor_radius: int,
    feature_set: str,
    round_ids: set[str] | None = None,
    latest_n_rounds: int | None = None,
    first_n_rounds: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """
    Returnerer X, y_argmax, y_prob, meta_rows (per celle), stats (logging).
    """
    all_dirs = discover_historical_round_dirs(historical_root)
    round_dirs, filter_desc = filter_round_dirs(
        all_dirs,
        round_ids=round_ids,
        latest_n_rounds=latest_n_rounds,
        first_n_rounds=first_n_rounds,
    )

    X_rows: list[list[float]] = []
    y_arg_rows: list[int] = []
    y_prob_rows: list[list[float]] = []
    meta_rows: list[dict[str, Any]] = []
    round_ids_used: list[str] = []
    round_summaries: list[dict[str, Any]] = []
    seeds_used = 0

    for rd in round_dirs:
        detail = _load_json(rd / "round_detail.json")
        rid, rnum = str(detail.get("id", rd.name)), int(detail.get("round_number", 0))
        round_ids_used.append(rid)
        round_summaries.append(
            {"round_id": rid, "round_number": rnum, "path": str(rd.resolve())}
        )
        analysis_dir = rd / "analysis"

        for seed_file in sorted(analysis_dir.glob("seed_*.json")):
            item = _load_json(seed_file)
            gt = item.get("ground_truth")
            initial_grid = item.get("initial_grid")
            seed_index = int(item.get("seed_index", -1))
            if gt is None or initial_grid is None:
                continue

            gt_np = np.asarray(gt, dtype=np.float64)
            grid_np = np.asarray(initial_grid, dtype=np.int32)
            if gt_np.ndim != 3 or gt_np.shape[-1] != NUM_CLASSES:
                continue
            h, w, _ = gt_np.shape
            if grid_np.shape != (h, w):
                continue

            seeds_used += 1
            initial_states = detail.get("initial_states", [])
            init_setts = (
                initial_states[seed_index].get("settlements", [])
                if 0 <= seed_index < len(initial_states)
                else []
            )
            has_settlement, dist_to_settlement = compute_settlement_maps(init_setts, h, w)

            for y in range(h):
                for x in range(w):
                    prob = gt_np[y, x, :]
                    if not np.all(np.isfinite(prob)):
                        continue
                    s = float(np.sum(prob))
                    if s <= 0:
                        continue
                    prob = prob / s
                    y_arg = int(np.argmax(prob))

                    feat = feature_row_at(
                        grid_np,
                        dist_to_settlement,
                        has_settlement,
                        x,
                        y,
                        neighbor_radius=neighbor_radius,
                        settlements=init_setts,
                        feature_set=feature_set,
                    )
                    X_rows.append(feat)
                    y_arg_rows.append(y_arg)
                    y_prob_rows.append(prob.tolist())
                    meta_rows.append(
                        {
                            "round_id": rid,
                            "round_number": rnum,
                            "seed_index": seed_index,
                            "x": x,
                            "y": y,
                        }
                    )

    stats: dict[str, Any] = {
        "historical_root": str(historical_root.resolve()),
        "filter_mode": filter_desc,
        "round_dirs_count": len(round_dirs),
        "round_ids_used": round_ids_used,
        "round_summaries": round_summaries,
        "seeds_loaded": seeds_used,
        "cells_loaded": len(X_rows),
        "neighbor_radius": neighbor_radius,
        "feature_set": feature_set,
    }

    if not X_rows:
        raise ValueError(
            f"Ingen celler etter filter ({filter_desc}). Sjekk --round-ids / historical mappe."
        )

    X = np.asarray(X_rows, dtype=np.float32)
    y_arg = np.asarray(y_arg_rows, dtype=np.int64)
    y_prob = np.asarray(y_prob_rows, dtype=np.float32)
    return X, y_arg, y_prob, meta_rows, stats
