from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


def repo_root() -> Path:
    """norgesgruppen/scripts/*.py -> repo root is parents[2]."""
    return Path(__file__).resolve().parents[2]


def resolve_images_dir(raw_dir: Path) -> Path:
    """Training images live under train_images/ or train_images/images/."""
    direct = raw_dir / "train_images"
    nested = direct / "images"
    if nested.is_dir() and any(nested.iterdir()):
        return nested
    return direct


def coco_bbox_to_yolo(
    bbox: list[float], image_width: int, image_height: int
) -> tuple[float, float, float, float] | None:
    x, y, w, h = bbox
    if w <= 0 or h <= 0 or image_width <= 0 or image_height <= 0:
        return None

    x1 = max(0.0, min(float(x), float(image_width)))
    y1 = max(0.0, min(float(y), float(image_height)))
    x2 = max(0.0, min(float(x + w), float(image_width)))
    y2 = max(0.0, min(float(y + h), float(image_height)))

    clipped_w = x2 - x1
    clipped_h = y2 - y1
    if clipped_w <= 0 or clipped_h <= 0:
        return None

    cx = (x1 + x2) / 2.0 / float(image_width)
    cy = (y1 + y2) / 2.0 / float(image_height)
    nw = clipped_w / float(image_width)
    nh = clipped_h / float(image_height)
    return cx, cy, nw, nh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from COCO annotations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split.")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.85,
        help="Train split ratio (rest goes to val).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = repo_root()
    raw_dir = project_root / "data" / "raw"
    images_dir = resolve_images_dir(raw_dir)
    annotations_path = raw_dir / "annotations.json"
    yolo_root = project_root / "data" / "yolo"

    yolo_images_train = yolo_root / "images" / "train"
    yolo_images_val = yolo_root / "images" / "val"
    yolo_labels_train = yolo_root / "labels" / "train"
    yolo_labels_val = yolo_root / "labels" / "val"

    for folder in [
        yolo_images_train,
        yolo_images_val,
        yolo_labels_train,
        yolo_labels_val,
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    print(f"Reading annotations: {annotations_path}")
    print(f"Source images directory: {images_dir}")
    with annotations_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    image_id_to_info = {img["id"]: img for img in images}
    annotations_by_image_id: dict[int, list[dict]] = defaultdict(list)
    for ann in annotations:
        annotations_by_image_id[ann["image_id"]].append(ann)

    random.seed(args.seed)
    shuffled_images = images[:]
    random.shuffle(shuffled_images)

    split_idx = int(len(shuffled_images) * args.train_ratio)
    train_images = shuffled_images[:split_idx]
    val_images = shuffled_images[split_idx:]
    train_ids = {img["id"] for img in train_images}

    invalid_boxes = 0
    copied_images = 0
    written_labels = 0

    for img in shuffled_images:
        img_id = img["id"]
        file_name = img["file_name"]
        width = int(img["width"])
        height = int(img["height"])

        src_img = images_dir / file_name
        if not src_img.exists():
            continue

        split = "train" if img_id in train_ids else "val"
        dst_img = yolo_images_train / file_name if split == "train" else yolo_images_val / file_name
        dst_label = (
            yolo_labels_train / f"{Path(file_name).stem}.txt"
            if split == "train"
            else yolo_labels_val / f"{Path(file_name).stem}.txt"
        )

        shutil.copy2(src_img, dst_img)
        copied_images += 1

        label_lines: list[str] = []
        for ann in annotations_by_image_id.get(img_id, []):
            category_id = int(ann["category_id"])
            yolo_box = coco_bbox_to_yolo(ann["bbox"], width, height)
            if yolo_box is None:
                invalid_boxes += 1
                continue
            cx, cy, bw, bh = yolo_box
            label_lines.append(f"{category_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        dst_label.write_text("\n".join(label_lines), encoding="utf-8")
        written_labels += 1

    max_category_id = max((int(c["id"]) for c in categories), default=-1)
    names = [""] * (max_category_id + 1)
    for c in categories:
        cid = int(c["id"])
        names[cid] = str(c["name"])

    expected_nc = 356
    if len(names) != expected_nc:
        print(
            f"WARNING: Expected {expected_nc} classes, found {len(names)} based on category ids 0..{max_category_id}."
        )

    data_yaml = yolo_root / "data.yaml"
    yaml_lines = [
        f"path: {yolo_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(names)}",
        "names:",
    ]
    yaml_lines.extend([f"  {idx}: {name}" for idx, name in enumerate(names)])
    data_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print("==== Dataset preparation complete ====")
    print(f"Total images in COCO: {len(images)}")
    print(f"Total annotations in COCO: {len(annotations)}")
    print(f"Train images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")
    print(f"Copied images: {copied_images}")
    print(f"Written label files: {written_labels}")
    print(f"Invalid/ignored boxes: {invalid_boxes}")
    print(f"data.yaml written to: {data_yaml}")


if __name__ == "__main__":
    main()
