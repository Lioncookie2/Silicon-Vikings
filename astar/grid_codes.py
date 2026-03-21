"""API grid-koder og hjelpere for telling / mapping."""
from __future__ import annotations

from typing import Any

# API: 0 empty, 1 settlement, 2 port, 3 ruin, 4 forest, 5 mountain, 10 ocean, 11 plains
OCEAN = 10
PLAINS = 11
EMPTY = 0
SETTLEMENT = 1
PORT = 2
RUIN = 3
FOREST = 4
MOUNTAIN = 5


def count_viewport_cells(grid: list[list[int]] | Any) -> dict[str, int]:
    """Tell celler i et viewport-grid (2D liste)."""
    if not grid:
        return {
            "empty_like": 0,
            "settlement": 0,
            "port": 0,
            "ruin": 0,
            "forest": 0,
            "mountain": 0,
        }
    empty_like = settlement = port = ruin = forest = mountain = 0
    for row in grid:
        for c in row:
            if c in (OCEAN, PLAINS, EMPTY):
                empty_like += 1
            elif c == SETTLEMENT:
                settlement += 1
            elif c == PORT:
                port += 1
            elif c == RUIN:
                ruin += 1
            elif c == FOREST:
                forest += 1
            elif c == MOUNTAIN:
                mountain += 1
            else:
                empty_like += 1
    return {
        "empty_like": empty_like,
        "settlement": settlement,
        "port": port,
        "ruin": ruin,
        "forest": forest,
        "mountain": mountain,
    }
