"""
Samlet «best pipeline»-rapport fra eksisterende eval-JSON (ingen nye modelltreningskjoringer).

Leser typisk:
  astar/data/evals/candidate_score_pipeline.json
  astar/data/evals/score_evaluate_historical.json

Skriver:
  astar/data/evals/best_pipeline_report.json

  export PYTHONPATH=.
  python -m astar.best_pipeline_report
  python -m astar.best_pipeline_report --candidate-json path --score-json path --out-json path
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _pick_best_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [r for r in rows if "estimated_competition_score" in r and "skipped" not in r]
    if not scored:
        return None
    scored.sort(
        key=lambda r: (
            -float(r["estimated_competition_score"]),
            float(r.get("entropy_weighted_kl", 1e9)),
        )
    )
    return scored[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate best pipeline recommendation from eval JSON files.")
    p.add_argument("--candidate-json", type=Path, default=None)
    p.add_argument("--score-json", type=Path, default=None)
    p.add_argument("--calibrate-json", type=Path, default=None, help="Valgfri prob_calibration*.json for anbefalingskontekst.")
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = _repo_root()
    evals = root / "astar" / "data" / "evals"
    cand_path = Path(args.candidate_json) if args.candidate_json else evals / "candidate_score_pipeline.json"
    score_path = Path(args.score_json) if args.score_json else evals / "score_evaluate_historical.json"
    out_json = Path(args.out_json) if args.out_json else evals / "best_pipeline_report.json"

    candidate_blob: dict[str, Any] | None = None
    if cand_path.is_file():
        candidate_blob = json.loads(cand_path.read_text(encoding="utf-8"))

    score_blob: dict[str, Any] | None = None
    if score_path.is_file():
        score_blob = json.loads(score_path.read_text(encoding="utf-8"))

    calibrate_meta: dict[str, Any] | None = None
    if args.calibrate_json and Path(args.calibrate_json).is_file():
        calibrate_meta = json.loads(Path(args.calibrate_json).read_text(encoding="utf-8"))

    best_row: dict[str, Any] | None = None
    if candidate_blob:
        best_row = _pick_best_candidate(list(candidate_blob.get("candidates") or []))

    reasons: list[str] = []
    current_best: dict[str, Any] = {}

    if best_row:
        current_best = {
            "candidate_name": best_row.get("candidate"),
            "explore_hierarchy_mode": best_row.get("explore_hierarchy_mode"),
            "prob_calibration": best_row.get("prob_calibration"),
            "blend": candidate_blob.get("blend") if candidate_blob else None,
            "model": candidate_blob.get("model") if candidate_blob else None,
            "feature_set": candidate_blob.get("feature_set") if candidate_blob else None,
            "historical_metrics": {
                "estimated_competition_score": best_row.get("estimated_competition_score"),
                "entropy_weighted_kl": best_row.get("entropy_weighted_kl"),
                "mean_cross_entropy": best_row.get("mean_cross_entropy"),
            },
        }
        reasons.append(
            f"Beste kandidat fra candidate_score_pipeline: {best_row.get('candidate')} "
            f"(score~{best_row.get('estimated_competition_score'):.4f}, "
            f"wKL={best_row.get('entropy_weighted_kl'):.6f})."
        )
    else:
        reasons.append("Fant ingen scored kandidater i candidate_score_pipeline.json — kjør candidate_score_pipeline først.")

    if score_blob and score_blob.get("overall"):
        ov = score_blob["overall"]
        reasons.append(
            f"Referanse score_evaluate_historical overall: score~{ov.get('estimated_competition_score'):.4f}, "
            f"wKL={ov.get('entropy_weighted_kl'):.6f} (sjekk at samme blend/kalibrering som produksjon)."
        )

    if calibrate_meta:
        reasons.append(
            f"Siste kalibrering JSON: recommendation={calibrate_meta.get('recommendation')} "
            f"(treningsmal={calibrate_meta.get('training_objective', 'n/a')})."
        )

    expected = None
    if best_row:
        expected = float(best_row["estimated_competition_score"])
    elif score_blob and score_blob.get("overall"):
        expected = float(score_blob["overall"]["estimated_competition_score"])

    report: dict[str, Any] = {
        "sources": {
            "candidate_score_pipeline": str(cand_path) if cand_path.is_file() else None,
            "score_evaluate_historical": str(score_path) if score_path.is_file() else None,
            "calibrate_json": str(args.calibrate_json) if args.calibrate_json else None,
        },
        "current_best_pipeline": current_best,
        "reasons": reasons,
        "expected_historical_estimated_score": expected,
        "note": "Dette er offline historisk estimat (ikke offisiell round-17 score).",
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"wrote": str(out_json), "expected_historical_estimated_score": expected}, indent=2))


if __name__ == "__main__":
    main()
