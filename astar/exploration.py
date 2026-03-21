"""
Viewport layouts for structured exploration under a fixed query budget.
"""
from __future__ import annotations


def balanced_viewports(
    h: int,
    w: int,
    vp: int = 15,
    max_patches: int = 10,
) -> list[tuple[int, int, int, int]]:
    """
    Cover the map with up to `max_patches` viewports (default 10 for 40×40).

    Uses a 3×3 grid with stride ~(map−vp)/2 plus one centered patch when distinct.
    Returns (viewport_x, viewport_y, viewport_w, viewport_h).
    """
    if h < 5 or w < 5:
        return []

    stride_y = max(5, (h - vp) // 3) if h > vp else 0
    stride_x = max(5, (w - vp) // 3) if w > vp else 0
    ys = sorted({0, min(stride_y, max(0, h - vp)), max(0, h - vp)})
    xs = sorted({0, min(stride_x, max(0, w - vp)), max(0, w - vp)})
    if len(ys) < 3 and h > vp:
        ys = sorted({0, max(0, (h - vp) // 2), max(0, h - vp)})
    if len(xs) < 3 and w > vp:
        xs = sorted({0, max(0, (w - vp) // 2), max(0, w - vp)})

    seen: set[tuple[int, int, int, int]] = set()
    out: list[tuple[int, int, int, int]] = []
    for y in ys:
        for x in xs:
            vw = min(vp, w - x)
            vh = min(vp, h - y)
            if vw >= 5 and vh >= 5:
                t = (x, y, vw, vh)
                if t not in seen:
                    seen.add(t)
                    out.append(t)

    cx = max(0, (w - vp) // 2)
    cy = max(0, (h - vp) // 2)
    center = (cx, cy, min(vp, w - cx), min(vp, h - cy))
    if center[2] >= 5 and center[3] >= 5 and center not in seen:
        out.append(center)

    if len(out) <= max_patches:
        return out[:max_patches]

    # Evenly subsample if we over-generated
    step = max(1, len(out) // max_patches)
    return out[::step][:max_patches]


def allocate_queries_per_seed(total_queries: int, num_seeds: int) -> list[int]:
    """Split budget evenly; remainder goes to lower seed indices."""
    base = total_queries // num_seeds
    rem = total_queries % num_seeds
    return [base + (1 if i < rem else 0) for i in range(num_seeds)]


def stride_cover_viewports(
    h: int,
    w: int,
    max_viewports: int,
    vp: int = 15,
    stride: int | None = None,
) -> list[tuple[int, int, int, int]]:
    """Greedy grid sweep with fixed stride (default ~12 for 40×40)."""
    if max_viewports <= 0:
        return []
    st = stride if stride is not None else max(5, min(vp - 1, 12))
    out: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    y = 0
    while y < h and len(out) < max_viewports:
        x = 0
        while x < w and len(out) < max_viewports:
            vw = min(vp, w - x)
            vh = min(vp, h - y)
            if vw >= 5 and vh >= 5:
                t = (x, y, vw, vh)
                if t not in seen:
                    seen.add(t)
                    out.append(t)
            x += st
        y += st
    return out
