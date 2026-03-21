"""
Flerniva explore-kalibrering (filbasert).

Fil: explore_dir / explore_calibration_hierarchy.json

- global: overstyr/utvid analysis_summary.calibration_suggestion (BOOST_KEYS)
- per_seed: seed-spesifikke overstyr
- regions: 4x4 bins "by_bx" med class_scale[6] per celle i bin (krever _count >= min)

predict / score_evaluate: --explore-hierarchy-mode
  off     — ignor hele hierarchy-filen (kun analysis_summary)
  global  — base + hierarchy.global
  seed    — + per_seed (standard nar fil finnes)
  full    — + regional class_scale pa explore_hw etter build_terrain_prior
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

BOOST_KEYS = (
    "coast_boost",
    "near_settlement_boost",
    "coast_near_settle_port_boost",
    "dynamic_ruin_weight",
)

NUM_CLASSES = 6

# Samme defaults som build_terrain_prior i baseline nar nøkkel mangler i dict
TERRAIN_PRIOR_DEFAULTS: dict[str, float] = {
    "coast_boost": 0.35,
    "near_settlement_boost": 0.5,
    "coast_near_settle_port_boost": 0.28,
    "dynamic_ruin_weight": 1.0,
}


def load_hierarchy_file(explore_dir: Path | None) -> dict[str, Any] | None:
    if explore_dir is None:
        return None
    p = Path(explore_dir) / "explore_calibration_hierarchy.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def resolve_explore_scalar_boosts(
    base_calibration: dict[str, Any] | None,
    hierarchy: dict[str, Any] | None,
    seed_index: int,
    mode: str,
) -> dict[str, Any]:
    """
    mode: off | global | seed | full
    'seed' og 'full' inkluderer per_seed; 'global' kun hierarchy.global.
    """
    out = dict(base_calibration or {})
    if not hierarchy or mode == "off":
        return out
    g = hierarchy.get("global")
    if isinstance(g, dict):
        for k in BOOST_KEYS:
            if k in g:
                try:
                    out[k] = float(g[k])
                except (TypeError, ValueError):
                    pass
    if mode in ("seed", "full"):
        ps = hierarchy.get("per_seed") or {}
        row = ps.get(str(seed_index))
        if isinstance(row, dict):
            for k in BOOST_KEYS:
                if k in row:
                    try:
                        out[k] = float(row[k])
                    except (TypeError, ValueError):
                        pass
    return out


def _regions_map_for_seed(hierarchy: dict[str, Any], seed_index: int) -> dict[str, Any]:
    psr = hierarchy.get("per_seed_regions")
    if isinstance(psr, dict):
        row = psr.get(str(seed_index))
        if isinstance(row, dict) and row:
            return row
    return hierarchy.get("regions") or {}


def apply_regional_explore_scales(
    explore_hw: np.ndarray,
    hierarchy: dict[str, Any] | None,
    h: int,
    w: int,
    seed_index: int,
    *,
    mode: str,
    ny_bins: int = 4,
    nx_bins: int = 4,
    min_region_samples: int = 200,
) -> np.ndarray:
    """Multipliser explore-kanaler per celle i bin der class_scale + nok data."""
    if mode != "full" or not hierarchy:
        return explore_hw
    regions = _regions_map_for_seed(hierarchy, seed_index)
    if not regions:
        return explore_hw
    out = np.array(explore_hw, dtype=np.float64, copy=True)
    bh = max(1, h // ny_bins)
    bw = max(1, w // nx_bins)
    for yi in range(ny_bins):
        for xi in range(nx_bins):
            key = f"{yi}_{xi}"
            entry = regions.get(key)
            if not isinstance(entry, dict):
                continue
            cnt = int(entry.get("_count", entry.get("sample_count", 0)))
            if cnt < min_region_samples:
                continue
            scales = entry.get("class_scale")
            if scales is None or len(scales) != NUM_CLASSES:
                continue
            sc = np.asarray(scales, dtype=np.float64).reshape(1, 1, NUM_CLASSES)
            y0 = yi * bh
            y1 = h if yi == ny_bins - 1 else min(h, (yi + 1) * bh)
            x0 = xi * bw
            x1 = w if xi == nx_bins - 1 else min(w, (xi + 1) * bw)
            patch = out[y0:y1, x0:x1, :] * sc
            rs = np.sum(patch, axis=-1, keepdims=True)
            rs = np.where(rs > 0, rs, 1.0)
            out[y0:y1, x0:x1, :] = patch / rs
    return out


# Bakoverkompatibilitet
def merge_calibration_for_seed(
    base: dict[str, Any],
    hierarchy: dict[str, Any] | None,
    seed_index: int,
) -> dict[str, Any]:
    return resolve_explore_scalar_boosts(base, hierarchy, seed_index, "seed")


def effective_terrain_boosts(resolved: dict[str, Any]) -> dict[str, float]:
    """Faktiske tall sendt til build_terrain_prior (base + defaults)."""
    out: dict[str, float] = {}
    for k in BOOST_KEYS:
        d = TERRAIN_PRIOR_DEFAULTS[k]
        if k in resolved:
            try:
                out[k] = float(resolved[k])
            except (TypeError, ValueError):
                out[k] = d
        else:
            out[k] = d
    return out


def _global_boost_keys_present(hierarchy: dict[str, Any] | None) -> list[str]:
    if not hierarchy:
        return []
    g = hierarchy.get("global")
    if not isinstance(g, dict):
        return []
    return [k for k in BOOST_KEYS if k in g]


def _per_seed_boost_keys_present(hierarchy: dict[str, Any] | None, seed_index: int) -> list[str]:
    if not hierarchy:
        return []
    ps = hierarchy.get("per_seed") or {}
    row = ps.get(str(seed_index))
    if not isinstance(row, dict):
        return []
    return [k for k in BOOST_KEYS if k in row]


def regional_bins_eligible(
    hierarchy: dict[str, Any] | None,
    seed_index: int,
    *,
    min_region_samples: int = 200,
) -> tuple[list[str], list[str]]:
    """
    Returnerer (eligible_bin_keys, skipped_reasons_per_bin) korte notater.
    Kun for full-modus-relevant diagnostikk.
    """
    if not hierarchy:
        return [], ["ingen hierarchy"]
    regions = _regions_map_for_seed(hierarchy, seed_index)
    if not regions:
        return [], ["ingen regions/per_seed_regions for denne seeden"]
    eligible: list[str] = []
    notes: list[str] = []
    for key, entry in regions.items():
        if not isinstance(entry, dict):
            continue
        cnt = int(entry.get("_count", entry.get("sample_count", 0)))
        scales = entry.get("class_scale")
        if scales is None or len(scales) != NUM_CLASSES:
            notes.append(f"{key}: mangler class_scale eller feil lengde")
            continue
        if cnt < min_region_samples:
            notes.append(f"{key}: _count={cnt} < min={min_region_samples}")
            continue
        eligible.append(str(key))
    return eligible, notes


def build_explore_hierarchy_diagnostic(
    explore_dir: Path | None,
    base_calibration: dict[str, float],
    hierarchy: dict[str, Any] | None,
    *,
    sample_seed_indices: tuple[int, ...] = (0, 1, 2, 3, 4),
    min_region_samples: int = 200,
) -> dict[str, Any]:
    """
    Forklarer om off/global/seed/full faktisk kan skille seg — uten a kjore hele eval.
    """
    hier_path = Path(explore_dir) / "explore_calibration_hierarchy.json" if explore_dir else None
    summary_path = Path(explore_dir) / "analysis_summary.json" if explore_dir else None

    global_keys = _global_boost_keys_present(hierarchy)
    per_seed_summary: dict[str, Any] = {}
    for si in sample_seed_indices:
        pkeys = _per_seed_boost_keys_present(hierarchy, si)
        elig, reg_notes = regional_bins_eligible(hierarchy, si, min_region_samples=min_region_samples)
        eff_off = effective_terrain_boosts(resolve_explore_scalar_boosts(base_calibration, hierarchy, si, "off"))
        eff_glob = effective_terrain_boosts(resolve_explore_scalar_boosts(base_calibration, hierarchy, si, "global"))
        eff_seed = effective_terrain_boosts(resolve_explore_scalar_boosts(base_calibration, hierarchy, si, "seed"))
        per_seed_summary[str(si)] = {
            "hierarchy_per_seed_keys": pkeys,
            "regional_bins_eligible_full_mode": elig,
            "regional_notes_sample": reg_notes[:6],
            "effective_boosts": {"off": eff_off, "global": eff_glob, "seed": eff_seed},
            "scalar_identical_off_vs_global": eff_off == eff_glob,
            "scalar_identical_off_vs_seed": eff_off == eff_seed,
            "scalar_identical_global_vs_seed": eff_glob == eff_seed,
        }

    any_off_vs_global = any(
        not per_seed_summary[str(si)]["scalar_identical_off_vs_global"] for si in sample_seed_indices
    )
    any_global_vs_seed = any(
        not per_seed_summary[str(si)]["scalar_identical_global_vs_seed"] for si in sample_seed_indices
    )
    any_regional = any(
        len(per_seed_summary[str(si)]["regional_bins_eligible_full_mode"]) > 0 for si in sample_seed_indices
    )

    modes_meaningful = {
        "off": True,
        "global": bool(hierarchy) and any_off_vs_global,
        "seed": bool(hierarchy) and (any_off_vs_global or any_global_vs_seed),
        "full": bool(hierarchy) and (any_off_vs_global or any_global_vs_seed or any_regional),
    }

    warning: str | None = None
    compare_modes_superficial = not modes_meaningful["global"] and not any_global_vs_seed and not any_regional
    if not hierarchy:
        warning = (
            "explore_calibration_hierarchy.json finnes ikke (eller explore_dir mangler). "
            "Da ignoreres global/seed/full i praksis — alle moduser bruker kun analysis_summary + defaults."
        )
    elif compare_modes_superficial:
        warning = (
            "Sammenligning av off/global/seed/full er i praksis meningslos: ingen hierarchy-endring som påvirker "
            "scalar boosts eller regionale skalaer (pa utvalgte seeds). Alle moduser gir samme explore-prior."
        )
    elif global_keys and not any_off_vs_global and not any_global_vs_seed and not any_regional:
        warning = (
            "Hierarchy har global-nøkler men effective_boosts er like analysis_summary (samme tall) — "
            "ingen per_seed/regional effekt; alle moduser kollapser til samme prior."
        )

    return {
        "explore_dir": str(explore_dir) if explore_dir else None,
        "hierarchy_json_path": str(hier_path) if hier_path else None,
        "hierarchy_json_exists": bool(hier_path and hier_path.is_file()),
        "analysis_summary_path": str(summary_path) if summary_path else None,
        "analysis_summary_exists": bool(summary_path and summary_path.is_file()),
        "base_calibration_keys_from_summary": sorted(base_calibration.keys()),
        "hierarchy_top_level_keys": sorted(hierarchy.keys()) if hierarchy else [],
        "hierarchy_global_boost_keys": global_keys,
        "per_seed_keys_in_file": sorted((hierarchy.get("per_seed") or {}).keys()) if hierarchy else [],
        "sample_seed_diagnostics": per_seed_summary,
        "scalar_branch_effective": {
            "off_vs_global": any_off_vs_global,
            "global_vs_seed": any_global_vs_seed,
            "seed_vs_full_regional_only": any_regional,
        },
        "compare_explore_modes_likely_identical_scores": compare_modes_superficial
        or (bool(hierarchy) and not any_off_vs_global and not any_global_vs_seed and not any_regional),
        "modes_meaningful_distinction": modes_meaningful,
        "warning": warning,
        "note_single_round_explore_dir": (
            "Explore-mapper er ofte fra én runde. analysis_summary + hierarchy reflekterer den kjoringen — "
            "historiske runder har andre kart/seeds, så samme scalar kan være suboptimalt eller ufarlig; "
            "hvis hierarchy er tom eller mangler, er modus-sammenligning rent formell (identiske tall)."
        ),
    }
