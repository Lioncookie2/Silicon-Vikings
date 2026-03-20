from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_yaml = project_root / "data" / "yolo" / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Could not find dataset config at {data_yaml}. Run scripts/prepare_dataset.py first."
        )

    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8m.pt")

    common_args = {
        "data": str(data_yaml),
        "imgsz": 1280,
        "epochs": 100,
        "patience": 20,
        "device": device,
        "project": str(project_root / "runs" / "ngd_yolo"),
        "name": "baseline",
        "exist_ok": True,
        "save": True,
    }

    try:
        print("Starting training with batch=-1 (auto batch).")
        model.train(batch=-1, **common_args)
    except Exception as exc:
        print(f"Auto batch failed ({exc}). Falling back to batch=8.")
        model.train(batch=8, **common_args)

    print("Training finished. Best weights should be under runs/ngd_yolo/baseline/weights/best.pt")


if __name__ == "__main__":
    main()
