from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

_tc = Path(__file__).resolve().parent / "torch_compat.py"
_spec = importlib.util.spec_from_file_location("_torch_compat_val", _tc)
assert _spec and _spec.loader
_spec.loader.exec_module(importlib.util.module_from_spec(_spec))

import torch
from ultralytics import YOLO


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_image_id(file_path: Path) -> int:
    stem = file_path.stem
    if "_" in stem:
        tail = stem.split("_")[-1]
    else:
        tail = stem
    return int(tail.lstrip("0") or "0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local YOLO validation inference.")
    parser.add_argument("--model", required=True, help="Path to model weights, e.g. best.pt")
    parser.add_argument(
        "--input",
        default=None,
        help="Input folder with images (default: data/yolo/images/val under repo).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: norgesgruppen/submission/local_predictions.json).",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    model_path = Path(args.model)
    input_dir = Path(args.input) if args.input else root / "data" / "yolo" / "images" / "val"
    output_path = Path(args.output) if args.output else root / "norgesgruppen" / "submission" / "local_predictions.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_files = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    )

    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))

    predictions: list[dict] = []
    for img_path in image_files:
        image_id = parse_image_id(img_path)
        results = model.predict(
            source=str(img_path),
            conf=float(args.conf),
            imgsz=int(args.imgsz),
            device=device,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            continue

        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].tolist()
            cls_id = int(boxes.cls[i].item())
            score = float(boxes.conf[i].item())

            x1, y1, x2, y2 = xyxy
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)

            predictions.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(cls_id),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score),
                }
            )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False)

    print(f"Images processed: {len(image_files)}")
    print(f"Predictions written: {len(predictions)}")
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()
