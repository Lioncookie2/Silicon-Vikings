#!/usr/bin/env python3
"""
End-to-end test: kjør norgesgruppen/submission/run.py som i konkurransen,
les predictions.json og regn ut detection/classification/hybrid mAP@0.5 mot val-GT.

Bruk: kopier best.pt (eller model.onnx) inn i submission/ først, eller bruk --copy-weights.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from metrics_ngd import (
    build_image_id_to_stem,
    compute_hybrid_maps,
    load_gt_for_split,
    load_predictions_json_to_preds,
    parse_data_yaml_nc_and_names,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluer submission run.py → JSON mot val-sett.")
    p.add_argument(
        "--submission-dir",
        type=Path,
        default=None,
        help="Mappe med run.py + best.pt eller model.onnx (standard: norgesgruppen/submission)",
    )
    p.add_argument(
        "--copy-weights",
        type=Path,
        default=None,
        help="Kopier denne vektfila inn i submission som best.pt før kjøring (.pt eller .onnx).",
    )
    p.add_argument("--split", choices=("val", "train"), default="val")
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Hvor predictions skrives (standard: runs/eval/predictions_run_py.json)",
    )
    p.add_argument("--detect-only", action="store_true", help="Videre til run.py --detect-only")
    p.add_argument("--tta", action="store_true", help="Videre til run.py --tta")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    sub = args.submission_dir or (root / "norgesgruppen" / "submission")
    run_py = sub / "run.py"
    if not run_py.is_file():
        raise FileNotFoundError(f"Mangler {run_py}")

    if args.copy_weights is not None:
        src = args.copy_weights.resolve()
        if not src.is_file():
            raise FileNotFoundError(src)
        if src.suffix.lower() == ".onnx":
            dst = sub / "model.onnx"
        else:
            dst = sub / "best.pt"
        shutil.copy2(src, dst)
        print(f"Kopierte vekter: {src} → {dst}")

    onnx = sub / "model.onnx"
    pt = sub / "best.pt"
    if not onnx.exists() and not pt.exists():
        raise FileNotFoundError(
            f"Legg best.pt eller model.onnx i {sub}, eller bruk --copy-weights"
        )

    split = args.split
    images_dir = root / "data" / "yolo" / "images" / split
    labels_dir = root / "data" / "yolo" / "labels" / split
    data_yaml = root / "data" / "yolo" / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError("Mangler data/yolo/data.yaml — kjør prepare_dataset.py")
    if not images_dir.is_dir():
        raise FileNotFoundError(images_dir)

    out_json = args.output_json or (root / "runs" / "eval" / "predictions_run_py.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(run_py),
        "--input",
        str(images_dir),
        "--output",
        str(out_json),
        "--conf",
        str(args.conf),
        "--imgsz",
        str(args.imgsz),
    ]
    if args.detect_only:
        cmd.append("--detect-only")
    if args.tta:
        cmd.append("--tta")

    print("Kjører:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"run.py feilet med kode {proc.returncode}")

    nc, _ = parse_data_yaml_nc_and_names(data_yaml)
    gt_by_stem, total_gt = load_gt_for_split(images_dir, labels_dir)
    id_to_stem = build_image_id_to_stem(images_dir)
    preds_by_stem = load_predictions_json_to_preds(out_json, id_to_stem)

    det, cls_map, hybrid, n_cls = compute_hybrid_maps(preds_by_stem, gt_by_stem, nc)

    print()
    print("=== End-to-end eval (submission run.py) ===")
    print(f"Predictions: {out_json}")
    print(f"Split: {split}  |  GT-bokser: {total_gt}")
    print(f"Antall prediksjonsrader (i JSON): {sum(len(v) for v in preds_by_stem.values())}")
    print()
    print(f"detection_mAP@0.5:        {det:.4f}")
    print(f"classification_mAP@0.5:   {cls_map:.4f}")
    print(f"hybrid (0.7 det + 0.3 cls): {hybrid:.4f}")
    print()
    print(f"Klasser med GT: {n_cls} (av nc={nc})")


if __name__ == "__main__":
    main()
