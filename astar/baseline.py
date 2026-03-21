"""
Terrain-aware prior for H×W×6 probability tensors (Astar Island).

Grid codes (API): 0 empty, 1 settlement, 2 port, 3 ruin, 4 forest, 5 mountain,
10 ocean, 11 plains.

Prediction classes: 0 empty, 1 settlement, 2 port, 3 ruin, 4 forest, 5 mountain.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

NUM_CLASSES = 6
DEFAULT_EPS = 0.01


def _coast_mask(grid: np.ndarray) -> np.ndarray:
    """Land celler (ikke hav 10) med minst én 4-nabo som er hav."""
    h, w = grid.shape
    ocean = grid == 10
    land = ~ocean & (grid != 5)
    coast = np.zeros((h, w), dtype=bool)
    for y in range(h):
        for x in range(w):
            if not land[y, x]:
                continue
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and ocean[ny, nx]:
                    coast[y, x] = True
                    break
    return coast


def _dist_layer(h: int, w: int, sx: int, sy: int) -> np.ndarray:
    yy, xx = np.ogrid[0:h, 0:w]
    return np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2)


def _min_dist_to_any_settlement(
    h: int,
    w: int,
    settlements: list[dict[str, Any]],
) -> np.ndarray:
    d = np.full((h, w), np.inf, dtype=np.float64)
    for s in settlements:
        if not s.get("alive", True):
            continue
        dist = _dist_layer(h, w, int(s["x"]), int(s["y"]))
        d = np.minimum(d, dist)
    return d


def build_terrain_prior(
    grid: list[list[int]],
    settlements: list[dict[str, Any]],
    *,
    eps: float = DEFAULT_EPS,
    sigma_settle: float = 6.0,
    coast_boost: float = 0.35,
    near_settlement_boost: float = 0.5,
    coast_near_settle_port_boost: float = 0.28,
    coast_near_settle_dist: float = 8.0,
    dynamic_ruin_weight: float = 1.0,
) -> np.ndarray:
    """
    Bygg (H, W, 6) ikke-negative scores; kall apply_floor_and_renorm etterpå.

    - Sterk masse på fjell/skog/tom der terrenget er statisk.
    - Boost settlement/port/ruin nær initial settlements og på kyst.
    """
    g = np.asarray(grid, dtype=np.int32)
    h, w = g.shape
    p = np.ones((h, w, NUM_CLASSES), dtype=np.float64)
    p[..., 0] = 0.25
    p[..., 1] = 0.05
    p[..., 2] = 0.05
    p[..., 3] = 0.05
    p[..., 4] = 0.15
    p[..., 5] = 0.05

    ocean = g == 10
    plains = g == 11
    empty = g == 0
    mountain = g == 5
    forest = g == 4
    ruin = g == 3
    settle = g == 1
    port = g == 2

    mask_ope = ocean | plains | empty
    p[mask_ope, :] = 0.01
    p[mask_ope, 0] = 0.88
    p[mask_ope, 4] = 0.04
    p[mask_ope, 1] = 0.02
    p[mask_ope, 2] = 0.015
    p[mask_ope, 3] = 0.015
    p[mask_ope, 5] = 0.015

    p[mountain, :] = 0.02
    p[mountain, 5] = 0.85
    p[mountain, 0] = 0.10

    p[forest, :] = 0.04
    p[forest, 4] = 0.78
    p[forest, 0] = 0.14

    p[ruin, :] = 0.06
    p[ruin, 3] = 0.70
    p[ruin, 0] = 0.18

    p[settle, :] = 0.05
    p[settle, 1] = 0.82
    p[settle, 0] = 0.10

    p[port, :] = 0.05
    p[port, 2] = 0.82
    p[port, 0] = 0.10

    coast = _coast_mask(g)
    p[coast, 1] *= 1.0 + coast_boost
    p[coast, 2] *= 1.0 + coast_boost * 0.85
    p[coast, 3] *= 1.0 + coast_boost * 0.55

    d = _min_dist_to_any_settlement(h, w, settlements)
    settle_w = np.exp(-(d**2) / (2.0 * sigma_settle**2))
    p[..., 1] *= 1.0 + near_settlement_boost * settle_w
    p[..., 2] *= 1.0 + coast_near_settle_port_boost * settle_w
    p[..., 3] *= dynamic_ruin_weight * (1.0 + 0.28 * settle_w)

    near_coast_settle = coast & (d <= coast_near_settle_dist)
    p[near_coast_settle, 1] *= 1.0 + coast_boost * 0.45
    p[near_coast_settle, 2] *= 1.0 + coast_near_settle_port_boost * 0.35

    return p


def apply_floor_and_renorm(tensor: np.ndarray, *, eps: float = DEFAULT_EPS) -> np.ndarray:
    t = np.asarray(tensor, dtype=np.float64)
    t = np.maximum(t, float(eps))
    s = np.sum(t, axis=-1, keepdims=True)
    s = np.where(s > 0, s, 1.0)
    return t / s


def load_detail_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_initial_state_for_seed(initial_states: list[dict[str, Any]], seed_index: int) -> tuple[list[list[int]], list[dict[str, Any]]]:
    st = initial_states[seed_index]
    grid = st["grid"]
    settlements = st.get("settlements", [])
    return grid, settlements


def uniform_tensor(height: int, width: int, eps: float = DEFAULT_EPS) -> list[list[list[float]]]:
    v = 1.0 / NUM_CLASSES
    return [[[v for _ in range(NUM_CLASSES)] for _ in range(width)] for _ in range(height)]


def numpy_to_submission_list(arr: np.ndarray) -> list[list[list[float]]]:
    return np.asarray(arr, dtype=np.float64).tolist()


def validate_prediction(
    pred: list[list[list[float]]],
    height: int,
    width: int,
    eps: float,
) -> list[str]:
    errs: list[str] = []
    if len(pred) != height:
        return [f"forventet høyde {height}, fikk {len(pred)}"]
    tol = 0.06
    for y in range(height):
        row = pred[y]
        if len(row) != width:
            errs.append(f"rad {y}: forventet bredde {width}, fikk {len(row)}")
            continue
        for x in range(width):
            cell = row[x]
            if len(cell) != NUM_CLASSES:
                errs.append(f"celle ({y},{x}): forventet {NUM_CLASSES} klasser, fikk {len(cell)}")
                continue
            if not all(math.isfinite(float(c)) for c in cell):
                errs.append(f"celle ({y},{x}): ikke-endelige verdier")
                continue
            s = float(sum(cell))
            if abs(s - 1.0) > tol:
                errs.append(f"celle ({y},{x}): sum={s:.4f} (forventet ~1)")
            if min(cell) < eps * 0.5 - 1e-12:
                errs.append(f"celle ({y},{x}): under gulv (min={min(cell):.6g}, eps={eps})")
    return errs[:80]


def build_prior_from_round_detail(
    detail: dict[str, Any],
    seed_index: int,
    *,
    eps: float = DEFAULT_EPS,
    calibration: dict[str, Any] | None = None,
) -> list[list[list[float]]]:
    grid, settlements = load_initial_state_for_seed(detail["initial_states"], seed_index)
    cal = calibration if isinstance(calibration, dict) else {}
    p = build_terrain_prior(
        grid,
        settlements,
        eps=eps,
        coast_boost=float(cal.get("coast_boost", 0.35)),
        near_settlement_boost=float(cal.get("near_settlement_boost", 0.5)),
        coast_near_settle_port_boost=float(cal.get("coast_near_settle_port_boost", 0.28)),
        dynamic_ruin_weight=float(cal.get("dynamic_ruin_weight", 1.0)),
    )
    return numpy_to_submission_list(apply_floor_and_renorm(p, eps=eps))
