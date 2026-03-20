from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

# PyTorch 2.6+ / ultralytics 8.1: load trusted weights (must run before ultralytics)
_tc = Path(__file__).resolve().parent / "torch_compat.py"
_spec = importlib.util.spec_from_file_location("_torch_compat_ng", _tc)
assert _spec and _spec.loader
_tc_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tc_mod)

import torch
from ultralytics import YOLO


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8 on NorgesGruppen shelf data.")
    p.add_argument(
        "--weights",
        type=str,
        default="yolov8m.pt",
        help="Starting weights file under repo root or absolute path (e.g. yolov8l.pt).",
    )
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch", type=int, default=-1, help="-1 = auto batch size.")
    p.add_argument("--name", type=str, default="baseline", help="Run name under runs/ngd_yolo/.")
    # Augmentation (upgrade path)
    p.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation probability (0-1).")
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--copy-paste", type=float, default=0.0, dest="copy_paste")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project_root = repo_root()
    data_yaml = project_root / "data" / "yolo" / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Could not find dataset config at {data_yaml}. Run norgesgruppen/scripts/prepare_dataset.py first."
        )

    weights_path = Path(args.weights)
    if not weights_path.is_file():
        alt = project_root / args.weights
        if alt.is_file():
            weights_path = alt
        else:
            raise FileNotFoundError(f"Weights not found: {args.weights} (also tried {alt})")

    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(weights_path))

    common_args: dict = {
        "data": str(data_yaml),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "patience": args.patience,
        "device": device,
        "project": str(project_root / "runs" / "ngd_yolo"),
        "name": args.name,
        "exist_ok": True,
        "save": True,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "copy_paste": args.copy_paste,
    }

    batch = int(args.batch)
    try:
        if batch == -1:
            print("Starting training with batch=-1 (auto batch).")
            model.train(batch=-1, **common_args)
        else:
            model.train(batch=batch, **common_args)
    except Exception as exc:
        print(f"Training failed ({exc}). Falling back to batch=8.")
        model.train(batch=8, **common_args)

    print(
        f"Training finished. Best weights: runs/ngd_yolo/{args.name}/weights/best.pt"
    )


if __name__ == "__main__":
    main()
