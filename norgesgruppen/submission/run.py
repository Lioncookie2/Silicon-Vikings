from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# PyTorch 2.6+ defaults weights_only=True; official YOLO .pt need False (trusted weights).
_orig_torch_load = torch.load


def _torch_load(*args: object, **kwargs: object) -> object:
    kwargs.setdefault("weights_only", False)  # type: ignore[assignment]
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load  # type: ignore[assignment]

from ultralytics import YOLO


def parse_image_id(image_path: Path) -> int:
    stem = image_path.stem
    if "_" in stem:
        candidate = stem.split("_")[-1]
    else:
        candidate = stem
    digits = "".join(ch for ch in candidate if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not parse image_id from filename: {image_path.name}")
    return int(digits.lstrip("0") or "0")


def iter_images(folder: Path) -> list[Path]:
    allowed = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in allowed])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input folder with images.")
    parser.add_argument("--output", required=True, help="Output JSON file path.")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size.")
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable test-time augmentation (slower, may improve mAP).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    model_path = Path(__file__).resolve().parent / "best.pt"

    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if not model_path.exists():
        raise ValueError(f"Model file missing: {model_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_files = iter_images(input_dir)

    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))

    output_rows: list[dict] = []
    for image_path in image_files:
        image_id = parse_image_id(image_path)
        results = model.predict(
            source=str(image_path),
            conf=float(args.conf),
            imgsz=int(args.imgsz),
            device=device,
            verbose=False,
            augment=bool(args.tta),
        )
        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            continue

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)

            output_rows.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(boxes.cls[i].item()),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(boxes.conf[i].item()),
                }
            )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_rows, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
