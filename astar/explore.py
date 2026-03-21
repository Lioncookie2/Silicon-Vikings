"""
Programmatisk Astar-datainnsamling via simulate-API (ingen manuell nettside).

Henter aktiv runde, lagrer initial states, kjører simulate og skriver én JSON per kall.

**Inkrementell utforskning (samme runde):**
- Første batch: ``python -m astar.explore --out-dir astar/analysis/explore/min_r16 --max-queries 10``
  (eller uten ``--out-dir`` for tidsstemplet standardmappe — noter stien).
- Neste batch: bruk **samme mappe** med ``--append-dir <sti>`` *eller* ``--out-dir <samme sti>``.
- Eksisterende ``runs/run_*.json`` overskrives ikke; nye filer får høyere ``run_#####``.
- Stride-planen fortsetter der du slapp (hopper over allerede utførte viewports).

Fra repo-roten (bruk venv: ``source .venv/bin/activate`` eller ``./.venv/bin/python -m ...``):

  export PYTHONPATH=.
  python -m astar.explore --test
  python -m astar.explore --max-queries 20
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .client import AstarClient, get_active_round
from .grid_codes import FOREST, MOUNTAIN, OCEAN


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _session_dir(root: Path, round_id: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short = round_id[:8] if len(round_id) >= 8 else round_id
    return root / "astar" / "analysis" / "explore" / f"{ts}_{short}"


def _next_run_index(runs_dir: Path) -> int:
    """Neste ledige run_NNNNN.json (1-basert); tåler hull i nummerering."""
    best = 0
    for p in runs_dir.glob("run_*.json"):
        stem = p.stem
        if not stem.startswith("run_"):
            continue
        try:
            n = int(stem[4:])
            best = max(best, n)
        except ValueError:
            continue
    return best + 1


def _test_plan(h: int, w: int, seeds_count: int) -> list[tuple[int, int, int, int, int]]:
    """
    Nøyaktig 5 simulate-kall: én per seed 0..4 med viewport (0,0,15×15) når mulig,
    ellers 5 vinduer på seed 0.
    """
    vp = 15
    out: list[tuple[int, int, int, int, int]] = []
    if seeds_count >= 5:
        for s in range(5):
            vw = min(vp, w)
            vh = min(vp, h)
            out.append((s, 0, 0, vw, vh))
    else:
        corners = [(0, 0), (max(0, w - vp), 0), (0, max(0, h - vp)), (max(0, w - vp), max(0, h - vp))]
        cx = max(0, (w - vp) // 2)
        cy = max(0, (h - vp) // 2)
        corners.append((cx, cy))
        for i, (x, y) in enumerate(corners[:5]):
            vw = min(vp, w - x)
            vh = min(vp, h - y)
            if vw >= 5 and vh >= 5:
                out.append((0, x, y, vw, vh))
    return out[:5]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate og lagre Astar-observasjoner (API).")
    p.add_argument(
        "--test",
        action="store_true",
        help="Kjør nøyaktig 5 simulate-kall (rask API-test + lokale JSON-filer).",
    )
    p.add_argument("--max-queries", type=int, default=50, help="Maks antall simulate-kall (ignoreres med --test).")
    p.add_argument(
        "--plan",
        choices=("stride", "diverse5"),
        default="stride",
        help="Plan for viewports. diverse5 = 5 kall med høy variasjon.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Utforskingsmappe (default: tidsstemplet under astar/analysis/explore/). "
        "Gjenbruk samme sti eller --append-dir for neste batch.",
    )
    p.add_argument(
        "--append-dir",
        type=Path,
        default=None,
        help="Fortsett i samme explore-mappe (anbefalt for batch 2+ sammen med --out-dir til samme sti).",
    )
    return p.parse_args()


def _stride_runs(
    h: int,
    w: int,
    seeds: int,
    max_queries: int,
    vp: int,
    stride: int,
) -> list[tuple[int, int, int, int, int]]:
    """(seed, vx, vy, vw, vh)."""
    out: list[tuple[int, int, int, int, int]] = []
    n = 0
    for seed_idx in range(min(seeds, 5)):
        y = 0
        while y < h and n < max_queries:
            x = 0
            while x < w and n < max_queries:
                vw = min(vp, w - x)
                vh = min(vp, h - y)
                if vw < 5 or vh < 5:
                    break
                out.append((seed_idx, x, y, vw, vh))
                n += 1
                x += stride
            y += stride
    return out


def _record_one(
    *,
    round_id: str,
    seed_index: int,
    vx: int,
    vy: int,
    vw: int,
    vh: int,
    res: dict,
) -> dict:
    vp_api = res.get("viewport") or {}
    return {
        "round_id": round_id,
        "seed_index": seed_index,
        "viewport": {"x": vx, "y": vy, "w": vw, "h": vh},
        "viewport_response": vp_api,
        "grid": res.get("grid"),
        "settlements": res.get("settlements"),
        "queries_used": res.get("queries_used"),
        "queries_max": res.get("queries_max"),
        "width": res.get("width"),
        "height": res.get("height"),
    }


def _find_first_cell(grid: list[list[int]], predicate: callable) -> tuple[int, int] | None:
    h = len(grid)
    w = len(grid[0]) if h else 0
    for y in range(h):
        for x in range(w):
            if predicate(grid[y][x], x, y):
                return x, y
    return None


def _coast_mask_from_grid(grid: list[list[int]]) -> list[list[bool]]:
    h = len(grid)
    w = len(grid[0]) if h else 0
    out = [[False for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            c = grid[y][x]
            if c in (OCEAN, MOUNTAIN):
                continue
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and grid[ny][nx] == OCEAN:
                    out[y][x] = True
                    break
    return out


def _vp_from_center(cx: int, cy: int, w: int, h: int, vp: int = 15) -> tuple[int, int, int, int]:
    vx = max(0, min(w - vp, cx - vp // 2))
    vy = max(0, min(h - vp, cy - vp // 2))
    vw = min(vp, w - vx)
    vh = min(vp, h - vy)
    return vx, vy, vw, vh


def _diverse5_plan(detail: dict, h: int, w: int, seeds: int) -> list[tuple[int, int, int, int, int]]:
    """
    5 kall med høy variasjon:
    1) samme region som før (0,0) på annen seed,
    2) sentrum,
    3) annet hjørne,
    4) kystnært område,
    5) skognært område.
    """
    vp = 15
    initial_states = detail.get("initial_states", [])
    plan: list[tuple[int, int, int, int, int]] = []

    # 1) same region on another seed
    seed_same_region = 1 if seeds > 1 else 0
    plan.append((seed_same_region, 0, 0, min(vp, w), min(vp, h)))

    # 2) center
    cx, cy = w // 2, h // 2
    vx, vy, vw, vh = _vp_from_center(cx, cy, w, h, vp=vp)
    plan.append((0, vx, vy, vw, vh))

    # 3) opposite corner
    plan.append((2 if seeds > 2 else 0, max(0, w - vp), max(0, h - vp), min(vp, w), min(vp, h)))

    # 4) coastal on a different seed
    seed_coast = 3 if seeds > 3 else 0
    coast_grid = initial_states[seed_coast]["grid"] if initial_states else []
    coast_mask = _coast_mask_from_grid(coast_grid) if coast_grid else []
    coast_xy = None
    if coast_mask:
        for y, row in enumerate(coast_mask):
            for x, is_coast in enumerate(row):
                if is_coast:
                    coast_xy = (x, y)
                    break
            if coast_xy:
                break
    if coast_xy is None:
        coast_xy = (w // 4, h // 4)
    vx, vy, vw, vh = _vp_from_center(coast_xy[0], coast_xy[1], w, h, vp=vp)
    plan.append((seed_coast, vx, vy, vw, vh))

    # 5) forest on another seed
    seed_forest = 4 if seeds > 4 else 0
    forest_grid = initial_states[seed_forest]["grid"] if initial_states else []
    forest_xy = _find_first_cell(forest_grid, lambda c, _x, _y: c == FOREST) if forest_grid else None
    if forest_xy is None:
        forest_xy = (3 * w // 4, 3 * h // 4)
    vx, vy, vw, vh = _vp_from_center(forest_xy[0], forest_xy[1], w, h, vp=vp)
    plan.append((seed_forest, vx, vy, vw, vh))

    return plan[:5]


def main() -> None:
    args = parse_args()
    root = _repo_root()
    client = AstarClient()
    round_id, detail = get_active_round(client)

    h = int(detail["map_height"])
    w = int(detail["map_width"])
    seeds = int(detail.get("seeds_count", 5))

    session = (args.append_dir or args.out_dir or _session_dir(root, round_id)).resolve()
    session.mkdir(parents=True, exist_ok=True)

    detail_path = session / "round_detail.json"
    detail_path.write_text(json.dumps(detail, indent=2, ensure_ascii=False), encoding="utf-8")

    runs_dir = session / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    n_on_disk = len(list(runs_dir.glob("run_*.json")))
    start_idx = _next_run_index(runs_dir)
    skip = start_idx - 1

    if args.test:
        full_plan = _test_plan(h, w, seeds)
        batch_limit = len(full_plan)
    elif args.plan == "diverse5":
        full_plan = _diverse5_plan(detail, h, w, seeds)
        batch_limit = args.max_queries
    else:
        batch_limit = args.max_queries
        full_plan = _stride_runs(h, w, seeds, skip + batch_limit, vp=15, stride=12)

    plan = full_plan[skip : skip + batch_limit]
    max_q = len(plan)

    now = datetime.now(timezone.utc).isoformat()
    print(
        f"[explore] round_id={round_id}\n"
        f"[explore] session={session}\n"
        f"[explore] runs allerede på disk: {n_on_disk} (neste run-index: {start_idx}, "
        f"viewport-hopp i plan: {skip})\n"
        f"[explore] nye queries i denne batchen: {max_q}\n"
        f"[explore] totalt runs etter batch (forventet): {n_on_disk + max_q}"
    )

    if max_q == 0:
        print("[explore] Ingen nye viewports (tom plan etter hopp). Skriver manifest/log; ingen simulate-kall.")
        manifest = {
            "round_id": round_id,
            "map_height": h,
            "map_width": w,
            "seeds_count": seeds,
            "mode": "test" if args.test else args.plan,
            "planned_queries_this_batch": 0,
            "cumulative_runs_on_disk": n_on_disk,
            "started_at": now,
            "initial_states_ref": "round_detail.json",
            "incremental": {
                "runs_on_disk_before_batch": n_on_disk,
                "runs_added_this_batch": 0,
                "runs_total_after_batch": n_on_disk,
                "next_run_index": start_idx,
                "viewport_skip": skip,
            },
        }
        (session / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        batch_line = {
            "utc": now,
            "round_id": round_id,
            "session_dir": str(session),
            "runs_on_disk_before_batch": n_on_disk,
            "runs_added_this_batch": 0,
            "runs_total_after_batch": n_on_disk,
            "mode": "test" if args.test else args.plan,
            "viewport_skip": skip,
        }
        with (session / "explore_batches.jsonl").open("a", encoding="utf-8") as bf:
            bf.write(json.dumps(batch_line, ensure_ascii=False) + "\n")
        (session / "summary.json").write_text(
            json.dumps({**manifest, "completed_at": now, "saved_queries_this_batch": 0}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[explore] Skrev {session} (0 nye filer i runs/)")
        return

    manifest = {
        "round_id": round_id,
        "map_height": h,
        "map_width": w,
        "seeds_count": seeds,
        "mode": "test" if args.test else args.plan,
        "planned_queries_this_batch": max_q,
        "cumulative_runs_on_disk_after_batch": n_on_disk + max_q,
        "started_at": now,
        "initial_states_ref": "round_detail.json",
        "incremental": {
            "runs_on_disk_before_batch": n_on_disk,
            "runs_added_this_batch": max_q,
            "runs_total_after_batch": n_on_disk + max_q,
            "next_run_index_start": start_idx,
            "viewport_skip": skip,
        },
    }
    (session / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    for i, (seed_idx, vx, vy, vw, vh) in enumerate(plan, start=start_idx):
        res = client.simulate(
            round_id=round_id,
            seed_index=seed_idx,
            viewport_x=vx,
            viewport_y=vy,
            viewport_w=vw,
            viewport_h=vh,
        )
        rec = _record_one(
            round_id=round_id,
            seed_index=seed_idx,
            vx=vx,
            vy=vy,
            vw=vw,
            vh=vh,
            res=res,
        )
        out_path = runs_dir / f"run_{i:05d}.json"
        out_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        relative = i - start_idx + 1
        print(f"[{relative}/{max_q}] seed={seed_idx} vp=({vx},{vy},{vw}x{vh}) queries_used={rec.get('queries_used')}/{rec.get('queries_max')}")

    done = datetime.now(timezone.utc).isoformat()
    summary = {
        **manifest,
        "completed_at": done,
        "saved_queries_this_batch": max_q,
        "cumulative_runs_on_disk_after_batch": n_on_disk + max_q,
    }
    (session / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    batch_line = {
        "utc": done,
        "round_id": round_id,
        "session_dir": str(session),
        "runs_on_disk_before_batch": n_on_disk,
        "runs_added_this_batch": max_q,
        "runs_total_after_batch": n_on_disk + max_q,
        "mode": "test" if args.test else args.plan,
        "viewport_skip": skip,
    }
    with (session / "explore_batches.jsonl").open("a", encoding="utf-8") as bf:
        bf.write(json.dumps(batch_line, ensure_ascii=False) + "\n")
    print(
        f"[explore] Ferdig: +{max_q} nye under runs/ (totalt på disk nå: {n_on_disk + max_q}). "
        f"Logg: {session / 'explore_batches.jsonl'}"
    )


if __name__ == "__main__":
    main()
