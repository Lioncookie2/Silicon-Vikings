#!/usr/bin/env python3
"""
Mål hvor lang én treningsepisode tar (én epoch), og estimer total tid for 50 / 100 / 200 epochs.

Bruker samme oppsett som train_yolo.py (data.yaml, vekter, batch, imgsz, device).
"""
from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

_tc = Path(__file__).resolve().parent / "torch_compat.py"
_spec = importlib.util.spec_from_file_location("_torch_compat_bench", _tc)
assert _spec and _spec.loader
_spec.loader.exec_module(importlib.util.module_from_spec(_spec))

_nc = Path(__file__).resolve().parent / "numpy_compat.py"
_spec_nc = importlib.util.spec_from_file_location("_numpy_compat_bench", _nc)
assert _spec_nc and _spec_nc.loader
_mod_nc = importlib.util.module_from_spec(_spec_nc)
_spec_nc.loader.exec_module(_mod_nc)

import torch
from ultralytics import YOLO


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark: tid for 1 YOLO-epoch + estimater.")
    p.add_argument("--weights", type=str, default="yolov8m.pt")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--workers", type=int, default=0, help="0 anbefales ofte på Mac.")
    p.add_argument(
        "--name",
        type=str,
        default="_epoch_benchmark",
        help="Run-navn under runs/ngd_yolo/ (unngå å overskrive 'baseline').",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    data_yaml = root / "data" / "yolo" / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Mangler {data_yaml} — kjør prepare_dataset.py først.")

    weights_path = Path(args.weights)
    if not weights_path.is_file():
        alt = root / args.weights
        if alt.is_file():
            weights_path = alt
        else:
            raise FileNotFoundError(f"Weights not found: {args.weights}")

    device = pick_device()
    model = YOLO(str(weights_path))

    common = {
        "data": str(data_yaml),
        "imgsz": int(args.imgsz),
        "epochs": 1,
        "patience": 999,
        "device": device,
        "project": str(root / "runs" / "ngd_yolo"),
        "name": args.name,
        "exist_ok": True,
        "plots": False,
        "val": True,
        "workers": int(args.workers),
        "verbose": True,
    }

    print(f"Device: {device}  |  batch={args.batch}  |  imgsz={args.imgsz}")
    print("Kjører nøyaktig 1 epoch (inkl. validering etter epoch) …")
    t0 = time.perf_counter()
    model.train(batch=int(args.batch), **common)
    elapsed = time.perf_counter() - t0

    print()
    print("=== Resultat ===")
    print(f"Tid for 1 epoch (train + val for denne epoch): {elapsed:.1f} s ({elapsed / 60:.2f} min)")
    print()
    print("Estimert total tid (lineært, omtrentlig):")
    for n in (50, 100, 200):
        total_s = elapsed * n
        print(f"  {n:3d} epochs: {total_s / 60:.1f} min  ({total_s / 3600:.2f} timer)")
    print()
    print("Merk: Senere epochs kan være litt raskere/tyngre (cache, mosaic off mot slutten).")
    print(f"Midlertidig run-mappe: runs/ngd_yolo/{args.name}/ (kan slettes om ønskelig).")


if __name__ == "__main__":
    main()
