#!/usr/bin/env python3
"""
Direkte eval: Ultralytics YOLO på val/train uten å gå via run.py.
Samme metrikker som evaluate_submission_run.py (se metrics_ngd.py).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

_tc = Path(__file__).resolve().parent / "torch_compat.py"
_spec = importlib.util.spec_from_file_location("_torch_compat_eval", _tc)
assert _spec and _spec.loader
_spec.loader.exec_module(importlib.util.module_from_spec(_spec))

_nc = Path(__file__).resolve().parent / "numpy_compat.py"
_spec_nc = importlib.util.spec_from_file_location("_numpy_compat_eval", _nc)
assert _spec_nc and _spec_nc.loader
_nc_mod = importlib.util.module_from_spec(_spec_nc)
_spec_nc.loader.exec_module(_nc_mod)

import torch
from ultralytics import YOLO

from metrics_ngd import compute_hybrid_maps, load_gt_for_split, parse_data_yaml_nc_and_names


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluer mAP@0.5 (det + cls) + hybrid (direkte YOLO).")
    p.add_argument("--model", required=True, help="best.pt, last.pt eller model.onnx")
    p.add_argument("--data-yaml", type=Path, default=None)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--split", choices=("val", "train"), default="val")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    data_yaml = args.data_yaml or (root / "data" / "yolo" / "data.yaml")
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Mangler {data_yaml} — kjør prepare_dataset.py først.")

    split = args.split
    images_dir = root / "data" / "yolo" / "images" / split
    labels_dir = root / "data" / "yolo" / "labels" / split
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Mangler bilde-mappe: {images_dir}")

    nc, _names = parse_data_yaml_nc_and_names(data_yaml)
    gt_by_stem, total_gt = load_gt_for_split(images_dir, labels_dir)
    image_paths = sorted(
        p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not image_paths:
        raise RuntimeError(f"Ingen bilder i {images_dir}")

    device = pick_device()
    model = YOLO(str(Path(args.model).resolve()))

    results_list = model.predict(
        source=[str(p) for p in image_paths],
        conf=float(args.conf),
        imgsz=int(args.imgsz),
        device=device,
        verbose=False,
    )

    preds_by_stem: dict[str, list[tuple[float, tuple[float, float, float, float], int]]] = defaultdict(
        list
    )
    for img_path, res in zip(image_paths, results_list):
        stem = img_path.stem
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            s = float(boxes.conf[i].item())
            c = int(boxes.cls[i].item())
            preds_by_stem[stem].append((s, (x1, y1, x2, y2), c))

    det, cls_map, hybrid, n_cls = compute_hybrid_maps(preds_by_stem, gt_by_stem, nc)

    print("=== NorgesGruppen-lignende eval (direkte YOLO) ===")
    print(f"Modell: {args.model}")
    print(f"Device: {device}")
    print(f"Split: {split}  |  Bilder: {len(image_paths)}  |  GT-bokser: {total_gt}")
    print(f"Confidence >= {args.conf}  |  imgsz={args.imgsz}")
    print()
    print(f"detection_mAP@0.5 (klasse ignorert): {det:.4f}")
    print(f"classification_mAP@0.5 (mean over klasser m/ GT): {cls_map:.4f}")
    print(f"hybrid_score (0.7 det + 0.3 cls):     {hybrid:.4f}")
    print()
    print(f"Klasser med GT i {split}: {n_cls} (av nc={nc} i data.yaml)")


if __name__ == "__main__":
    main()
