"""
Etter active_query_runner batch 1: kopier prediksjoner, sammenlign før/etter, rapport + query-fordeling.

Working grid etter simulate lagres ikke på disk; ``prediction_seed{N}_final.json`` er allerede
kjørt med samme produksjonspipeline på merged grid — vi kopierer til ønsket filnavn og
reproduserer «før» fra ``get_round`` initial_states + ``recompute_prediction``.

  PYTHONPATH=. python3 -m astar.post_batch1_summarize \\
    --round-number 17 \\
    --explore-dir . \\
    --seed-dirs-glob 'astar/analysis/active_query/round_17_seed_*_batch1'
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .active_query_runner import compute_entropy_map, recompute_prediction, _load_explore_calibration
from .baseline import load_initial_state_for_seed, numpy_to_submission_list
from .cell_features import infer_feature_set_from_feature_names
from .client import AstarClient, resolve_round_identifier
from .explore_hierarchy import load_hierarchy_file
from .global_model_loader import load_global_model_or_exit
from .global_model_paths import models_dir as default_models_dir


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _rel_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _parse_seed_from_dir(p: Path) -> int | None:
    m = re.search(r"seed_(\d+)_batch", p.name)
    return int(m.group(1)) if m else None


def _mean_entropy_tensor(pred_hw: np.ndarray) -> float:
    ent = compute_entropy_map(pred_hw)
    return float(np.mean(ent))


def _production_bundle_for_seed(
    *,
    detail: dict[str, Any],
    seed_index: int,
    explore_dir: Path,
    model: Any,
    feature_set: str,
    neighbor_radius: int,
    cal_path: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    grid0, settlements = load_initial_state_for_seed(detail["initial_states"], seed_index)
    working = np.asarray(grid0, dtype=np.int32)
    calibration = _load_explore_calibration(explore_dir)
    explore_hier = load_hierarchy_file(explore_dir)
    bundle = recompute_prediction(
        working_grid=working,
        settlements=settlements,
        model=model,
        feature_set=feature_set,
        neighbor_radius=neighbor_radius,
        calibration=calibration,
        explore_hier=explore_hier,
        explore_hierarchy_mode="seed",
        min_regional_samples=200,
        global_weight=0.7,
        explore_weight=0.3,
        prob_floor=0.01,
        calibration_json=cal_path,
        seed_index=seed_index,
    )
    return bundle.final_hw, working


def _allocate_queries_largest_remainder(weights: list[float], total: int) -> list[int]:
    s = sum(weights)
    if s <= 0:
        n = len(weights)
        base = total // n
        out = [base] * n
        for i in range(total - base * n):
            out[i] += 1
        return out
    raw = [total * w / s for w in weights]
    floors = [int(np.floor(x)) for x in raw]
    rem = total - sum(floors)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - floors[i], reverse=True)
    out = list(floors)
    for k in range(rem):
        out[order[k % len(order)]] += 1
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post batch-1 summary for active_query outputs.")
    p.add_argument("--round-number", type=int, default=17)
    p.add_argument("--explore-dir", type=Path, default=Path("."))
    p.add_argument(
        "--seed-dirs-glob",
        type=str,
        default="astar/analysis/active_query/round_17_seed_*_batch1",
    )
    p.add_argument(
        "--out-summary",
        type=Path,
        default=None,
        help="Default: astar/analysis/active_query/round_17_summary_after_batch1.json",
    )
    p.add_argument("--large-l1-threshold", type=float, default=0.2)
    p.add_argument("--remaining-queries", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = _repo_root()
    pat = args.seed_dirs_glob
    gp = Path(pat)
    if gp.is_absolute():
        dirs = sorted(gp.parent.glob(gp.name))
    else:
        dirs = sorted(root.glob(pat))
    if not dirs:
        raise SystemExit(f"Ingen mapper matchet glob {pat!r} under {root}")

    client = AstarClient()
    round_id = resolve_round_identifier(client, str(args.round_number))
    detail = client.get_round(round_id)

    mdir = default_models_dir()
    model, meta, _, _, _ = load_global_model_or_exit(mdir / "gradient_boosting.joblib", mdir / "gradient_boosting_meta.json")
    fn = meta.get("feature_names")
    feature_set = str(meta.get("feature_set") or infer_feature_set_from_feature_names([str(x) for x in fn]))
    nr = int(meta.get("neighbor_radius", 1))
    cal_path = mdir / "prob_calibration.json"
    explore_dir = Path(args.explore_dir)
    if not explore_dir.is_absolute():
        explore_dir = (Path.cwd() / explore_dir).resolve()
    thr = float(args.large_l1_threshold)
    rem_q = int(args.remaining_queries)

    out_summary = args.out_summary
    if out_summary is None:
        out_summary = root / "astar" / "analysis" / "active_query" / "round_17_summary_after_batch1.json"
    out_summary = Path(out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    per_seed: list[dict[str, Any]] = []

    for d in dirs:
        si = _parse_seed_from_dir(d)
        if si is None:
            continue
        report_path = d / f"active_query_report_seed{si}.json"
        final_path = d / f"prediction_seed{si}_final.json"
        if not report_path.is_file() or not final_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        after = np.asarray(json.loads(final_path.read_text(encoding="utf-8")), dtype=np.float64)
        h, w, k = after.shape
        assert k == 6

        queries_used = sum(int(b.get("n_queries", 0)) for b in report.get("batches", []))
        cells_updated = sum(
            int(b.get("explore_update", {}).get("total_cells_updated", 0)) for b in report.get("batches", [])
        )

        before_hw, _ = _production_bundle_for_seed(
            detail=detail,
            seed_index=si,
            explore_dir=explore_dir,
            model=model,
            feature_set=feature_set,
            neighbor_radius=nr,
            cal_path=cal_path if cal_path.is_file() else None,
        )
        assert before_hw.shape == after.shape

        ent_before = _mean_entropy_tensor(before_hw)
        ent_after = _mean_entropy_tensor(after)
        l1 = np.sum(np.abs(before_hw - after), axis=-1)
        mean_l1 = float(np.mean(l1))
        cells_large = int(np.sum(l1 > thr))
        frac_large = float(cells_large / (h * w))

        out_name = d / f"prediction_final_seed{si}_after_batch1.json"
        out_name.write_text(json.dumps(numpy_to_submission_list(after)), encoding="utf-8")

        per_seed.append(
            {
                "seed": si,
                "report_dir": _rel_to_root(d, root),
                "queries_used": queries_used,
                "cells_updated_simulate": cells_updated,
                "mean_entropy_before": ent_before,
                "mean_entropy_after": ent_after,
                "delta_mean_entropy": ent_after - ent_before,
                "mean_l1_change": mean_l1,
                "cells_large_change": cells_large,
                "cells_large_change_l1_gt": cells_large,
                "frac_cells_large_change": frac_large,
                "large_l1_threshold": thr,
                "after_prediction_file": _rel_to_root(out_name, root),
                "after_prediction_note": (
                    "Identisk innhold som prediction_seed{N}_final.json fra active_query_runner "
                    "(merged grid ble ikke lagret; reproduksjon av etter-prediksjon krever simulate-replay)."
                ).replace("{N}", str(si)),
            }
        )

    per_seed.sort(key=lambda x: x["seed"])

    # Rangering: kombiner usikkerhet etter batch, observasjonseffekt og effekt per query
    def zscore(vals: list[float]) -> list[float]:
        a = np.asarray(vals, dtype=np.float64)
        m, s = float(np.mean(a)), float(np.std(a))
        if s < 1e-12:
            return [0.0] * len(vals)
        return [float((v - m) / s) for v in vals]

    ent_a = [r["mean_entropy_after"] for r in per_seed]
    l1_a = [r["mean_l1_change"] for r in per_seed]
    q_a = [max(1, r["queries_used"]) for r in per_seed]
    l1_per_q = [l1_a[i] / q_a[i] for i in range(len(per_seed))]

    ze, zl, zpq = zscore(ent_a), zscore(l1_a), zscore(l1_per_q)
    for i, r in enumerate(per_seed):
        # Større er bedre for «prioriter neste queries»
        r["ranking_score"] = float(0.45 * ze[i] + 0.35 * zl[i] + 0.20 * zpq[i])
    order = sorted(range(len(per_seed)), key=lambda i: per_seed[i]["ranking_score"], reverse=True)
    ranking = [{"rank": ri + 1, "seed": per_seed[i]["seed"], "ranking_score": per_seed[i]["ranking_score"]} for ri, i in enumerate(order)]

    n = len(per_seed)
    if rem_q < n:
        raise SystemExit(f"--remaining-queries ({rem_q}) må være minst antall seeds ({n}) for minst én query per seed.")
    # Minst 1 query per seed, rest proporsjonalt på (ranking_score + offset)
    base = [1] * n
    rest = rem_q - n
    wts = [max(0.05, float(s["ranking_score"]) + 1.2) for s in per_seed]
    if sum(wts) <= 0:
        wts = [1.0] * n
    extra = _allocate_queries_largest_remainder(wts, rest)
    alloc = [base[i] + extra[i] for i in range(n)]
    recommended = {per_seed[i]["seed"]: alloc[i] for i in range(n)}

    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "round_id": round_id,
        "round_number_arg": args.round_number,
        "seeds": per_seed,
        "ranking": ranking,
        "ranking_legend": (
            "ranking_score = 0.45*z(entropy_after) + 0.35*z(mean_l1_change) + 0.20*z(mean_l1_per_query); "
            "høyere = mer prioritert for neste simulate-budsjett."
        ),
        f"recommended_next_queries_{rem_q}": recommended,
        "recommended_allocation_method": (
            f"minst 1 query per seed; resterende {rem_q - n} fordelt proporsjonalt på max(0.05, ranking_score+1.2), "
            "største-rest-metode."
        ),
    }
    out_summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_summary}")
    for s in per_seed:
        print(
            f"seed {s['seed']}: queries={s['queries_used']} "
            f"H {s['mean_entropy_before']:.4f}->{s['mean_entropy_after']:.4f} "
            f"mean_L1={s['mean_l1_change']:.4f} large>{thr}={s['cells_large_change_l1_gt']} "
            f"score={s['ranking_score']:.3f}"
        )
    print(f"Anbefalt fordeling av neste {rem_q} queries: {recommended}")


if __name__ == "__main__":
    main()
