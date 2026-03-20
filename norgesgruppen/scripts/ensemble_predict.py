"""
Combine predictions from multiple YOLO checkpoints using Weighted Boxes Fusion (WBF).

Requires: ensemble-boxes (pre-installed in competition sandbox).

Usage:
  python norgesgruppen/scripts/ensemble_predict.py \\
    --models runs/ngd_yolo/baseline/weights/best.pt runs/ngd_yolo/large/weights/best.pt \\
    --input /path/to/images --output predictions.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from PIL import Image

_tc = Path(__file__).resolve().parent / "torch_compat.py"
_spec = importlib.util.spec_from_file_location("_torch_compat_ens", _tc)
assert _spec and _spec.loader
_spec.loader.exec_module(importlib.util.module_from_spec(_spec))

import torch
from ensemble_boxes import weighted_boxes_fusion
from ultralytics import YOLO


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_image_id(image_path: Path) -> int:
    stem = image_path.stem
    tail = stem.split("_")[-1] if "_" in stem else stem
    return int(tail.lstrip("0") or "0")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ensemble YOLO models with WBF.")
    p.add_argument("--models", nargs="+", required=True, help="Paths to .pt weights.")
    p.add_argument("--input", required=True, help="Image folder.")
    p.add_argument("--output", required=True, help="Output JSON path.")
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument(
        "--iou-thr",
        type=float,
        default=0.55,
        help="WBF IoU threshold.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    device = "0" if torch.cuda.is_available() else "cpu"

    models = [YOLO(str(p)) for p in args.models]
    image_files = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    all_preds: list[dict] = []
    weights = [1.0] * len(models)

    for img_path in image_files:
        image_id = parse_image_id(img_path)
        boxes_list_per_model: list[list[list[float]]] = []
        scores_list_per_model: list[list[float]] = []
        labels_list_per_model: list[list[int]] = []

        with Image.open(img_path) as im:
            w, h = im.size

        for model in models:
            results = model.predict(
                source=str(img_path),
                conf=float(args.conf),
                imgsz=int(args.imgsz),
                device=device,
                verbose=False,
            )
            r = results[0]
            if r.boxes is None or len(r.boxes) == 0:
                boxes_list_per_model.append([])
                scores_list_per_model.append([])
                labels_list_per_model.append([])
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            bl: list[list[float]] = []
            sl: list[float] = []
            ll: list[int] = []
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i].tolist()
                bl.append([x1 / w, y1 / h, x2 / w, y2 / h])
                sl.append(float(confs[i]))
                ll.append(int(clss[i]))
            boxes_list_per_model.append(bl)
            scores_list_per_model.append(sl)
            labels_list_per_model.append(ll)

        if all(len(b) == 0 for b in boxes_list_per_model):
            continue

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list_per_model,
            scores_list_per_model,
            labels_list_per_model,
            weights=weights,
            iou_thr=float(args.iou_thr),
            skip_box_thr=0.0,
        )

        for i in range(len(fused_boxes)):
            nx1, ny1, nx2, ny2 = fused_boxes[i]
            x1, y1, x2, y2 = nx1 * w, ny1 * h, nx2 * w, ny2 * h
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            all_preds.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(fused_labels[i]),
                    "bbox": [float(x1), float(y1), float(bw), float(bh)],
                    "score": float(fused_scores[i]),
                }
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(all_preds, f, ensure_ascii=False)
    print(f"Wrote {len(all_preds)} predictions to {out}")


if __name__ == "__main__":
    main()
