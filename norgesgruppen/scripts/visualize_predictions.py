#!/usr/bin/env python3
"""
Kjør inferens på bilder og lagre overlay med bbox + produktnavn + confidence.

Standard utmappe: <repo>/runs/predictions/<tag>/
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

_tc = Path(__file__).resolve().parent / "torch_compat.py"
_spec = importlib.util.spec_from_file_location("_torch_compat_viz", _tc)
assert _spec and _spec.loader
_spec.loader.exec_module(importlib.util.module_from_spec(_spec))

import cv2
import numpy as np
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


def parse_data_yaml_names(data_yaml: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    in_names = False
    for raw in data_yaml.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if line.strip().startswith("names:"):
            in_names = True
            continue
        if in_names:
            if not line.startswith(" ") and line.strip() and not line.startswith("\t"):
                break
            s = line.strip()
            if ":" in s and s[0].isdigit():
                k, v = s.split(":", 1)
                names[int(k.strip())] = v.strip().strip("\"'")
    return names


def label_for_class(names: dict[int, str], cls_id: int, conf: float, max_len: int = 42) -> str:
    raw = names.get(cls_id, f"class_{cls_id}")
    if len(raw) > max_len:
        raw = raw[: max_len - 1] + "…"
    return f"{raw}  {conf:.2f}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tegn prediksjoner på bilder → runs/predictions/")
    p.add_argument("--model", required=True, help="best.pt, last.pt eller model.onnx")
    p.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Fil eller mappe (standard: data/yolo/images/val)",
    )
    p.add_argument(
        "--data-yaml",
        type=Path,
        default=None,
        help="For klassenavn (standard: data/yolo/data.yaml)",
    )
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument(
        "--tag",
        type=str,
        default="viz",
        help="Undermappe under runs/predictions/",
    )
    p.add_argument("--limit", type=int, default=0, help="Maks antall bilder (0 = alle).")
    return p.parse_args()


def collect_images(source: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if source.is_file():
        return [source] if source.suffix.lower() in exts else []
    return sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in exts)


def main() -> None:
    args = parse_args()
    root = repo_root()
    data_yaml = args.data_yaml or (root / "data" / "yolo" / "data.yaml")
    source = args.source or (root / "data" / "yolo" / "images" / "val")

    if not data_yaml.is_file():
        raise FileNotFoundError(f"Mangler {data_yaml}")
    if not source.exists():
        raise FileNotFoundError(f"Mangler kilde: {source}")

    names = parse_data_yaml_names(data_yaml)
    out_dir = root / "runs" / "predictions" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(source)
    if args.limit and args.limit > 0:
        images = images[: int(args.limit)]
    if not images:
        raise RuntimeError("Ingen bilder å prosessere.")

    device = pick_device()
    model = YOLO(str(Path(args.model).resolve()))
    print(f"Device: {device}")

    rng = np.random.default_rng(42)
    palette = {
        c: tuple(int(x) for x in rng.integers(48, 255, size=3)) for c in range(len(names))
    }

    for img_path in images:
        res = model.predict(
            source=str(img_path),
            conf=float(args.conf),
            imgsz=int(args.imgsz),
            device=device,
            verbose=False,
        )[0]
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        boxes = res.boxes
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                conf = float(boxes.conf[i].item())
                cls_id = int(boxes.cls[i].item())
                color = palette.get(cls_id, (0, 255, 0))
                cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
                text = label_for_class(names, cls_id, conf)
                (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                ty = max(y1 - 4, th + 4)
                cv2.rectangle(im, (x1, ty - th - 4), (x1 + tw + 4, ty + baseline), color, -1)
                cv2.putText(
                    im,
                    text,
                    (x1 + 2, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        out_path = out_dir / f"pred_{img_path.name}"
        cv2.imwrite(str(out_path), im)

    print(f"Lagret {len(images)} bilde(r) i: {out_dir}")


if __name__ == "__main__":
    main()
