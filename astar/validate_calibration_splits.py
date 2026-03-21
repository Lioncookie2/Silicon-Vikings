"""
Sammenlign prob-kalibrering pa flere historiske vinduer og holdout-trening.

Del 1 — samme prob_calibration.json, varier latest-n-rounds (eval):
  for hver N: metrics uten / med calibration-json pa valgte runder.

Del 2 — varier holdout-last-n ved ny treningskjoring av kalibrering:
  kjor calibrate_probs_historical med holdout 2 og 3; les val metrics fra JSON.

  export PYTHONPATH=.
  python -m astar.validate_calibration_splits
  python -m astar.validate_calibration_splits --explore-dir astar/analysis/explore/<id>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .explore_hierarchy import load_hierarchy_file
from .global_model_loader import load_global_model_bundle
from .historical_cells_loader import discover_historical_round_dirs, filter_round_dirs, parse_round_ids_csv
from .cell_features import infer_feature_set_from_feature_names
from .score_evaluate_historical import _evaluate_explore_mode, _explore_cal


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _pick_metrics(ov: dict[str, Any]) -> dict[str, float]:
    return {
        "estimated_competition_score": float(ov["estimated_competition_score"]),
        "entropy_weighted_kl": float(ov["entropy_weighted_kl"]),
        "mean_cross_entropy": float(ov["mean_cross_entropy"]),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate prob calibration across splits.")
    p.add_argument("--historical-root", type=Path, default=None)
    p.add_argument("--explore-dir", type=Path, default=None)
    p.add_argument("--calibration-json", type=Path, default=None)
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--model-meta", type=Path, default=None)
    p.add_argument("--global-weight", type=float, default=0.7)
    p.add_argument("--explore-weight", type=float, default=0.3)
    p.add_argument("--prob-floor", type=float, default=0.01)
    p.add_argument("--min-regional-samples", type=int, default=200)
    p.add_argument("--skip-calibrate-reruns", action="store_true", help="Hopp over holdout 2/3 subprocess (kun eval-delen).")
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = _repo_root()
    hist = args.historical_root or (root / "astar" / "data" / "historical")
    cal_path = Path(args.calibration_json) if args.calibration_json else (root / "astar" / "data" / "models" / "prob_calibration.json")
    out_json = Path(args.out_json) if args.out_json else (root / "astar" / "data" / "evals" / "validate_calibration_splits.json")

    model, meta, ap, _mp, _ = load_global_model_bundle(args.model_path, args.model_meta)
    fn = meta.get("feature_names")
    if not fn:
        raise SystemExit("meta mangler feature_names")
    feature_names = [str(x) for x in fn]
    feature_set = str(meta.get("feature_set") or infer_feature_set_from_feature_names(feature_names))
    nr = int(meta.get("neighbor_radius", 1))
    cal_e = _explore_cal(Path(args.explore_dir) if args.explore_dir else None)
    explore_hier = load_hierarchy_file(Path(args.explore_dir) if args.explore_dir else None)

    all_dirs = discover_historical_round_dirs(hist)
    eval_rows: list[dict[str, Any]] = []

    for label, latest_n in (("latest_6", 6), ("latest_10", 10), ("all_rounds", None)):
        round_dirs, filter_desc = filter_round_dirs(
            all_dirs,
            round_ids=parse_round_ids_csv(None),
            latest_n_rounds=latest_n,
            first_n_rounds=None,
        )
        if not round_dirs:
            eval_rows.append({"label": label, "error": "ingen runder", "filter_mode": filter_desc})
            continue
        common_kw: dict[str, Any] = dict(
            round_dirs=round_dirs,
            model=model,
            feature_set=feature_set,
            nr=nr,
            cal_e=cal_e,
            explore_hier=explore_hier,
            hierarchy_mode="seed",
            min_regional_samples=int(args.min_regional_samples),
            global_weight=args.global_weight,
            explore_weight=args.explore_weight,
            prob_floor=float(args.prob_floor),
        )
        ov_before, _ = _evaluate_explore_mode(calibration_json=None, **common_kw)
        ov_after, _ = _evaluate_explore_mode(
            calibration_json=cal_path if cal_path.is_file() else None,
            **common_kw,
        )
        mb, ma = _pick_metrics(ov_before), _pick_metrics(ov_after)
        eval_rows.append(
            {
                "label": label,
                "latest_n_rounds": latest_n,
                "filter_mode": filter_desc,
                "n_rounds": len(round_dirs),
                "metrics_before": mb,
                "metrics_after": ma,
                "delta_estimated_score": ma["estimated_competition_score"] - mb["estimated_competition_score"],
                "delta_wkl": ma["entropy_weighted_kl"] - mb["entropy_weighted_kl"],
                "calibration_json_applied": cal_path.is_file(),
            }
        )

    calibrate_rows: list[dict[str, Any]] = []
    if not args.skip_calibrate_reruns:
        for ho in (2, 3):
            tmp_out = root / "astar" / "data" / "evals" / f"_validate_calibrate_holdout_{ho}.json"
            env = {**os.environ, "PYTHONPATH": str(root)}
            cmd = [
                sys.executable,
                "-m",
                "astar.calibrate_probs_historical",
                "--holdout-last-n",
                str(ho),
                "--out-json",
                str(tmp_out),
            ]
            if args.explore_dir:
                cmd.extend(["--explore-dir", str(args.explore_dir)])
            r = subprocess.run(cmd, cwd=str(root), env=env, capture_output=True, text=True)
            row: dict[str, Any] = {
                "holdout_last_n": ho,
                "exit_code": r.returncode,
                "stdout_tail": (r.stdout or "")[-800:],
            }
            if r.returncode == 0 and tmp_out.is_file():
                payload = json.loads(tmp_out.read_text(encoding="utf-8"))
                vb = _pick_metrics(payload["metrics_before"]["val"])
                va = _pick_metrics(payload["metrics_after"]["val"])
                row["val_metrics_before"] = vb
                row["val_metrics_after"] = va
                row["recommendation"] = payload.get("recommendation")
                row["flags"] = payload.get("flags")
                row["delta_val_score"] = va["estimated_competition_score"] - vb["estimated_competition_score"]
                row["delta_val_wkl"] = va["entropy_weighted_kl"] - vb["entropy_weighted_kl"]
            else:
                row["error"] = "calibrate_probs_historical feilet — se stdout_tail/stderr"
                row["stderr_tail"] = (r.stderr or "")[-800:]
            calibrate_rows.append(row)

    eval_ok_rows = [r for r in eval_rows if "delta_estimated_score" in r]
    eval_improves = [
        r
        for r in eval_ok_rows
        if r["delta_estimated_score"] > 1e-6 and r.get("delta_wkl", 0.0) < -1e-6
    ]
    rec_ok = {"use_in_predict", "marginal_ok"}
    cal_ok = [r for r in calibrate_rows if r.get("recommendation") in rec_ok]

    if eval_ok_rows and len(eval_improves) == len(eval_ok_rows):
        eval_verdict = "sterk: kalibrering forbedrer score og senker wKL pa alle testede latest-n vinduer"
    elif eval_improves:
        eval_verdict = "delvis: noen vinduer forbedres; sjekk tabell"
    else:
        eval_verdict = "svak: ikke konsekvent forbedring pa eval-vinduene"

    if args.skip_calibrate_reruns:
        cal_verdict = "ikke kjort (--skip-calibrate-reruns)"
    elif len(cal_ok) == len(calibrate_rows) and calibrate_rows:
        cal_verdict = "holdout 2 og 3: begge anbefalinger OK for trenet kalibrering"
    elif calibrate_rows:
        cal_verdict = "sjekk holdout-rader — ikke alle anbefalinger er like gode"
    else:
        cal_verdict = "ingen data"

    if eval_ok_rows and len(eval_improves) == len(eval_ok_rows):
        if args.skip_calibrate_reruns:
            overall = (
                "Eval-vinduene (6/10/alle) viser hoyere score og lavere wKL med prob_calibration.json; "
                "kjør uten --skip-calibrate-reruns for a bekrefte holdout 2/3 pa ny-fit for du låser som standard."
            )
        elif calibrate_rows and len(cal_ok) == len(calibrate_rows):
            overall = (
                "prob_calibration.json kan brukes som standard i neste runde "
                "(eval-vinduer + holdout 2/3 anbefalinger ser bra ut; gitt samme modell/blend/explore-kilde)."
            )
        elif calibrate_rows:
            overall = "Eval OK, men minst én holdout-fit har svak anbefaling — gjennomgå calibrate-rader før standardvalg."
        else:
            overall = "Eval OK; mangler holdout-kjoring (tom calibrate_rows)."
    elif eval_improves:
        overall = "Delvis støtte; gjennomgå eval-tabellen før standardvalg."
    else:
        overall = "Ikke sett som standard uten manuell gjennomgang av tabellene."

    report = {
        "model": str(ap),
        "feature_set": feature_set,
        "calibration_json": str(cal_path),
        "calibration_json_exists": cal_path.is_file(),
        "blend": {"global_weight": args.global_weight, "explore_weight": args.explore_weight},
        "prob_floor": args.prob_floor,
        "explore_dir": str(args.explore_dir) if args.explore_dir else None,
        "eval_across_latest_n": eval_rows,
        "calibrate_holdout_reruns": calibrate_rows,
        "verdict_eval_windows": eval_verdict,
        "verdict_calibrate_holdouts": cal_verdict,
        "recommendation_default_calibration_next_round": overall,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"wrote": str(out_json), "recommendation": overall}, indent=2, ensure_ascii=False))
    print("\n=== eval (fast prob_calibration.json) ===")
    for r in eval_rows:
        if "metrics_before" in r:
            print(
                f"  {r['label']}: score {r['metrics_before']['estimated_competition_score']:.4f} -> "
                f"{r['metrics_after']['estimated_competition_score']:.4f}  "
                f"wKL {r['metrics_before']['entropy_weighted_kl']:.5f} -> {r['metrics_after']['entropy_weighted_kl']:.5f}"
            )
        else:
            print(f"  {r.get('label')}: {r.get('error')}")
    if calibrate_rows:
        print("\n=== calibrate_probs_historical (ny fit, val holdout) ===")
        for r in calibrate_rows:
            if "val_metrics_before" in r:
                print(
                    f"  holdout_last_n={r['holdout_last_n']}: recommendation={r.get('recommendation')}  "
                    f"val score {r['val_metrics_before']['estimated_competition_score']:.4f} -> "
                    f"{r['val_metrics_after']['estimated_competition_score']:.4f}"
                )
            else:
                print(f"  holdout_last_n={r.get('holdout_last_n')}: {r.get('error', r)}")


if __name__ == "__main__":
    main()
