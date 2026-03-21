"""
Konsistente celle-features for trening (build_dataset) og inferens (predict).

Samme rekkefølge og semantikk som i historical_cells.npz.

settlement_port_radius: som settlement_port pluss Chebyshev-radiuser (max(|dx|,|dy|) <= r):
  levende settlements i r3/r5, settlements med has_port i r5, fjell-celler (terrengkode) i r3.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .grid_codes import EMPTY, FOREST, MOUNTAIN, OCEAN, PLAINS, SETTLEMENT

FEATURE_NAMES = [
    "initial_terrain_code",
    "is_coastal",
    "dist_to_nearest_initial_settlement",
    "has_initial_settlement",
    "n_forest_r1",
    "n_mountain_r1",
    "n_empty_like_r1",
    "n_settlement_r1",
]

# Kun felt som finnes i historical round_detail initial_states (x,y,has_port,alive) — ingen leakage fra slutt-tilstand
SETTLEMENT_PORT_EXTRA_NAMES = [
    "dist_nearest_port_settlement",
    "nearest_alive_settlement_has_port",
    "count_port_settlements_chebyshev_r2",
]

# Ekstra til settlement_port: Chebyshev-radius på rutenettet (max(|dx|,|dy|) <= r)
SETTLEMENT_PORT_RADIUS_EXTRA_NAMES = [
    "count_alive_settlements_chebyshev_r3",
    "count_alive_settlements_chebyshev_r5",
    "count_port_settlements_chebyshev_r5",
    "count_mountain_cells_chebyshev_r3",
]

FEATURE_SET_FULL = "full"
FEATURE_SET_SETTLEMENT_PORT = "settlement_port"
FEATURE_SET_SETTLEMENT_PORT_RADIUS = "settlement_port_radius"


def all_feature_names(feature_set: str = FEATURE_SET_FULL) -> list[str]:
    if feature_set == FEATURE_SET_SETTLEMENT_PORT_RADIUS:
        return (
            list(FEATURE_NAMES)
            + list(SETTLEMENT_PORT_EXTRA_NAMES)
            + list(SETTLEMENT_PORT_RADIUS_EXTRA_NAMES)
        )
    if feature_set == FEATURE_SET_SETTLEMENT_PORT:
        return list(FEATURE_NAMES) + list(SETTLEMENT_PORT_EXTRA_NAMES)
    return list(FEATURE_NAMES)


def infer_feature_set_from_feature_names(names: list[str]) -> str:
    """Gjenkjenn feature_set fra lagret feature_names (npz/meta)."""
    n = [str(x) for x in names]
    if n == all_feature_names(FEATURE_SET_SETTLEMENT_PORT_RADIUS):
        return FEATURE_SET_SETTLEMENT_PORT_RADIUS
    if n == all_feature_names(FEATURE_SET_SETTLEMENT_PORT):
        return FEATURE_SET_SETTLEMENT_PORT
    if n == all_feature_names(FEATURE_SET_FULL):
        return FEATURE_SET_FULL
    raise ValueError(
        f"Ukjent feature_names-liste (len={len(n)}). Forventet {len(FEATURE_NAMES)} (full), "
        f"{len(all_feature_names(FEATURE_SET_SETTLEMENT_PORT))} (settlement_port), eller "
        f"{len(all_feature_names(FEATURE_SET_SETTLEMENT_PORT_RADIUS))} (settlement_port_radius)."
    )


def _alive_settlements(settlements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [s for s in settlements if isinstance(s, dict) and s.get("alive", True)]


def _port_aware_extra_features(
    settlements: list[dict[str, Any]],
    x: int,
    y: int,
    h: int,
    w: int,
    *,
    chebyshev_r: int = 2,
) -> list[float]:
    alive = _alive_settlements(settlements)
    cap_d = float(max(h, w, 1))

    best_d = cap_d
    nearest_hp = 0.0
    for s in alive:
        sx, sy = int(s["x"]), int(s["y"])
        d = float(np.sqrt((sx - x) ** 2 + (sy - y) ** 2))
        if d < best_d:
            best_d = d
            nearest_hp = 1.0 if s.get("has_port") else 0.0
    if not alive:
        best_d = cap_d

    port_ds: list[float] = []
    for s in alive:
        if not s.get("has_port"):
            continue
        sx, sy = int(s["x"]), int(s["y"])
        port_ds.append(float(np.sqrt((sx - x) ** 2 + (sy - y) ** 2)))
    dist_port = min(port_ds) if port_ds else cap_d

    cnt_port = 0
    for s in alive:
        if not s.get("has_port"):
            continue
        sx, sy = int(s["x"]), int(s["y"])
        if max(abs(sx - x), abs(sy - y)) <= chebyshev_r:
            cnt_port += 1

    return [dist_port, nearest_hp, float(cnt_port)]


def _count_alive_settlements_chebyshev(
    alive: list[dict[str, Any]],
    x: int,
    y: int,
    r: int,
) -> int:
    n = 0
    for s in alive:
        sx, sy = int(s["x"]), int(s["y"])
        if max(abs(sx - x), abs(sy - y)) <= r:
            n += 1
    return n


def _count_port_settlements_chebyshev(
    alive: list[dict[str, Any]],
    x: int,
    y: int,
    r: int,
) -> int:
    n = 0
    for s in alive:
        if not s.get("has_port"):
            continue
        sx, sy = int(s["x"]), int(s["y"])
        if max(abs(sx - x), abs(sy - y)) <= r:
            n += 1
    return n


def _count_mountain_cells_chebyshev_r(grid: np.ndarray, x: int, y: int, r: int) -> int:
    h, w = grid.shape
    cnt = 0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if max(abs(dx), abs(dy)) > r:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and int(grid[ny, nx]) == MOUNTAIN:
                cnt += 1
    return cnt


def _settlement_port_radius_extra_features(
    grid: np.ndarray,
    settlements: list[dict[str, Any]],
    x: int,
    y: int,
) -> list[float]:
    alive = _alive_settlements(settlements)
    return [
        float(_count_alive_settlements_chebyshev(alive, x, y, 3)),
        float(_count_alive_settlements_chebyshev(alive, x, y, 5)),
        float(_count_port_settlements_chebyshev(alive, x, y, 5)),
        float(_count_mountain_cells_chebyshev_r(grid, x, y, 3)),
    ]


def compute_settlement_maps(
    settlements: list[dict[str, Any]],
    h: int,
    w: int,
) -> tuple[np.ndarray, np.ndarray]:
    has = np.zeros((h, w), dtype=np.int32)
    dist = np.full((h, w), np.inf, dtype=np.float64)
    if not settlements:
        dist[:, :] = float(max(h, w))
        return has, dist

    for s in settlements:
        if not s.get("alive", True):
            continue
        sx, sy = int(s["x"]), int(s["y"])
        if 0 <= sy < h and 0 <= sx < w:
            has[sy, sx] = 1
            yy, xx = np.ogrid[0:h, 0:w]
            d = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2)
            dist = np.minimum(dist, d)

    finite = np.isfinite(dist)
    if not np.any(finite):
        dist[:, :] = float(max(h, w))
    else:
        dist[~finite] = float(np.max(dist[finite]))
    return has, dist


def is_coastal_cell(grid: np.ndarray, x: int, y: int) -> int:
    c = int(grid[y, x])
    if c in (OCEAN, MOUNTAIN):
        return 0
    h, w = grid.shape
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and int(grid[ny, nx]) == OCEAN:
            return 1
    return 0


def neighbor_counts(grid: np.ndarray, x: int, y: int, radius: int) -> tuple[int, int, int, int]:
    h, w = grid.shape
    forest = mountain = empty_like = settlement = 0
    for ny in range(max(0, y - radius), min(h, y + radius + 1)):
        for nx in range(max(0, x - radius), min(w, x + radius + 1)):
            if nx == x and ny == y:
                continue
            c = int(grid[ny, nx])
            if c == FOREST:
                forest += 1
            elif c == MOUNTAIN:
                mountain += 1
            elif c == SETTLEMENT:
                settlement += 1
            elif c in (EMPTY, OCEAN, PLAINS):
                empty_like += 1
            else:
                empty_like += 1
    return forest, mountain, empty_like, settlement


def feature_row_at(
    grid: np.ndarray,
    dist_to_settlement: np.ndarray,
    has_settlement: np.ndarray,
    x: int,
    y: int,
    *,
    neighbor_radius: int = 1,
    settlements: list[dict[str, Any]] | None = None,
    feature_set: str = FEATURE_SET_FULL,
) -> list[float]:
    coastal = is_coastal_cell(grid, x, y)
    fn, mn, en, sn = neighbor_counts(grid, x, y, neighbor_radius)
    h, w = grid.shape
    row = [
        float(grid[y, x]),
        float(coastal),
        float(dist_to_settlement[y, x]),
        float(has_settlement[y, x]),
        float(fn),
        float(mn),
        float(en),
        float(sn),
    ]
    if feature_set == FEATURE_SET_SETTLEMENT_PORT:
        row.extend(_port_aware_extra_features(settlements or [], x, y, h, w))
    elif feature_set == FEATURE_SET_SETTLEMENT_PORT_RADIUS:
        row.extend(_port_aware_extra_features(settlements or [], x, y, h, w))
        row.extend(_settlement_port_radius_extra_features(grid, settlements or [], x, y))
    return row


def feature_matrix_for_seed(
    grid: list[list[int]] | np.ndarray,
    settlements: list[dict[str, Any]],
    *,
    neighbor_radius: int = 1,
    feature_set: str = FEATURE_SET_FULL,
) -> np.ndarray:
    """
    Flatten (y-major): index i = y * w + x.
    Shape (h * w, n_features), float32.
    """
    g = np.asarray(grid, dtype=np.int32)
    h, w = g.shape
    has_s, dist_s = compute_settlement_maps(settlements, h, w)
    n_feat = len(all_feature_names(feature_set))
    out = np.zeros((h * w, n_feat), dtype=np.float32)
    i = 0
    for y in range(h):
        for x in range(w):
            out[i] = np.asarray(
                feature_row_at(
                    g,
                    dist_s,
                    has_s,
                    x,
                    y,
                    neighbor_radius=neighbor_radius,
                    settlements=settlements,
                    feature_set=feature_set,
                ),
                dtype=np.float32,
            )
            i += 1
    return out
