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

_nc = Path(__file__).resolve().parent / "numpy_compat.py"
_spec_nc = importlib.util.spec_from_file_location("_numpy_compat_ng", _nc)
assert _spec_nc and _spec_nc.loader
_nc_mod = importlib.util.module_from_spec(_spec_nc)
_spec_nc.loader.exec_module(_nc_mod)

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
    p.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="DataLoader workers (-1 = Ultralytics default, 0 = ofte mer stabilt på Mac).",
    )
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
            pass # let ultralytics auto-download

    if torch.cuda.is_available():
        device = "0"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon — ofte mye raskere enn CPU
    else:
        device = "cpu"
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
    w = int(args.workers)
    if w >= 0:
        common_args["workers"] = w

    batch = int(args.batch)
    try:
        if batch == -1:
            print("Starting training with batch=-1 (auto batch).")
            model.train(batch=-1, **common_args)
        else:
            model.train(batch=batch, **common_args)
    except Exception as exc:
        err = str(exc).lower()
        # Unngå «oom» som delstreng (falske treff); kun tydelige OOM-meldinger.
        is_oom = (
            "out of memory" in err
            or "mps backend out of memory" in err
            or "cuda out of memory" in err
        )
        if is_oom:
            print(f"Ser ut som minne-feil ({exc}). Prøver batch=4 …")
            model.train(batch=4, **common_args)
        else:
            raise

    run_dir = project_root / "runs" / "ngd_yolo" / args.name
    print(f"Training finished. Best weights: {run_dir / 'weights' / 'best.pt'}")
    print(f"Se også: norgesgruppen/TRAINING_OUTPUTS.md (metrics, plots, results.csv).")


if __name__ == "__main__":
    main()
