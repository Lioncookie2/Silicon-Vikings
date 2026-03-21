"""
Delte metrikker for NorgesGruppen-lignende evaluering (detection + classification mAP@0.5, hybrid).

Brukes av eval_ngd.py og evaluate_submission_run.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def parse_data_yaml_nc_and_names(data_yaml: Path) -> tuple[int, dict[int, str]]:
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
    nc = max(names.keys(), default=-1) + 1 if names else 0
    return nc, names


def parse_image_id_from_stem(stem: str) -> int:
    if "_" in stem:
        candidate = stem.split("_")[-1]
    else:
        candidate = stem
    digits = "".join(ch for ch in candidate if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not parse image_id from stem: {stem}")
    return int(digits.lstrip("0") or "0")


def yolo_label_to_xyxy(
    parts: list[str], img_w: int, img_h: int
) -> tuple[int, tuple[float, float, float, float]] | None:
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))
    cx, cy, nw, nh = map(float, parts[1:5])
    x1 = (cx - nw / 2) * img_w
    y1 = (cy - nh / 2) * img_h
    x2 = (cx + nw / 2) * img_w
    y2 = (cy + nh / 2) * img_h
    return cls, (float(x1), float(y1), float(x2), float(y2))


def load_gt_for_split(
    images_dir: Path, labels_dir: Path
) -> tuple[dict[str, list[tuple[int, tuple[float, float, float, float]]]], int]:
    gt: dict[str, list[tuple[int, tuple[float, float, float, float]]]] = {}
    total = 0
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    from PIL import Image

    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in exts:
            continue
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"
        with Image.open(img_path) as im:
            w, h = im.size
        boxes: list[tuple[int, tuple[float, float, float, float]]] = []
        if label_path.is_file():
            text = label_path.read_text(encoding="utf-8").strip()
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                parsed = yolo_label_to_xyxy(line.split(), w, h)
                if parsed is not None:
                    boxes.append(parsed)
                    total += 1
        gt[stem] = boxes
    return gt, total


def build_image_id_to_stem(images_dir: Path) -> dict[int, str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out: dict[int, str] = {}
    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in exts:
            continue
        iid = parse_image_id_from_stem(img_path.stem)
        if iid in out and out[iid] != img_path.stem:
            raise ValueError(f"Flere bilder med samme image_id {iid}: {out[iid]} vs {img_path.stem}")
        out[iid] = img_path.stem
    return out


def load_predictions_json_to_preds(
    json_path: Path,
    id_to_stem: dict[int, str],
) -> dict[str, list[tuple[float, tuple[float, float, float, float], int]]]:
    """Konkurranse-JSON: image_id, bbox [x,y,w,h] xywh, category_id, score -> preds_by_stem med xyxy."""
    import json

    with json_path.open(encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("predictions.json må være en JSON-liste")

    preds: dict[str, list[tuple[float, tuple[float, float, float, float], int]]] = {}
    for row in rows:
        iid = int(row["image_id"])
        stem = id_to_stem.get(iid)
        if stem is None:
            continue
        x, y, w, h = (float(row["bbox"][i]) for i in range(4))
        x2, y2 = x + max(0.0, w), y + max(0.0, h)
        score = float(row["score"])
        cat = int(row["category_id"])
        preds.setdefault(stem, []).append((score, (x, y, x2, y2), cat))
    return preds


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def coco_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def ap_from_sorted_matches(is_tp: list[int], scores: list[float], total_gt: int) -> float:
    if total_gt == 0:
        return 0.0
    scores_arr = np.array(scores, dtype=np.float64)
    order = np.argsort(-scores_arr)
    tp_cum = np.cumsum(np.array([is_tp[i] for i in order], dtype=np.float64))
    fp_cum = np.cumsum(np.array([1 - is_tp[i] for i in order], dtype=np.float64))
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    recalls = tp_cum / total_gt
    return coco_ap(recalls, precisions)


def detection_map_global(
    preds_by_stem: dict[str, list[tuple[float, tuple[float, float, float, float], int]]],
    gt_by_stem: dict[str, list[tuple[int, tuple[float, float, float, float]]]],
    iou_thresh: float,
    total_gt: int,
) -> float:
    flat: list[tuple[float, str, tuple[float, float, float, float]]] = []
    for stem, plist in preds_by_stem.items():
        for score, box, _c in plist:
            flat.append((score, stem, box))
    flat.sort(key=lambda x: -x[0])

    unmatched: dict[str, set[int]] = {
        stem: set(range(len(gts))) for stem, gts in gt_by_stem.items()
    }
    det_scores: list[float] = []
    det_tp: list[int] = []
    for score, stem, box in flat:
        if stem not in unmatched:
            unmatched[stem] = set()
        um = unmatched[stem]
        gts = gt_by_stem.get(stem, [])
        best_iou = 0.0
        best_j = -1
        for j in um:
            iou = iou_xyxy(box, gts[j][1])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh and best_j >= 0:
            um.remove(best_j)
            det_tp.append(1)
        else:
            det_tp.append(0)
        det_scores.append(score)
    return ap_from_sorted_matches(det_tp, det_scores, total_gt)


def classification_map_global_per_class(
    preds_by_stem: dict[str, list[tuple[float, tuple[float, float, float, float], int]]],
    gt_by_stem: dict[str, list[tuple[int, tuple[float, float, float, float]]]],
    class_id: int,
    iou_thresh: float,
    gt_count: int,
) -> float:
    flat: list[tuple[float, str, tuple[float, float, float, float]]] = []
    for stem, plist in preds_by_stem.items():
        for score, box, c in plist:
            if c == class_id:
                flat.append((score, stem, box))
    flat.sort(key=lambda x: -x[0])

    unmatched: dict[str, set[int]] = {
        stem: {j for j, (gc, _) in enumerate(gts) if gc == class_id}
        for stem, gts in gt_by_stem.items()
    }

    cls_scores: list[float] = []
    cls_tp: list[int] = []
    for score, stem, box in flat:
        gts = gt_by_stem.get(stem, [])
        um = unmatched[stem]
        best_iou = 0.0
        best_j = -1
        for j in um:
            iou = iou_xyxy(box, gts[j][1])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh and best_j >= 0:
            um.remove(best_j)
            cls_tp.append(1)
        else:
            cls_tp.append(0)
        cls_scores.append(score)
    return ap_from_sorted_matches(cls_tp, cls_scores, gt_count)


def compute_hybrid_maps(
    preds_by_stem: dict[str, list[tuple[float, tuple[float, float, float, float], int]]],
    gt_by_stem: dict[str, list[tuple[int, tuple[float, float, float, float]]]],
    nc: int,
    iou_thresh: float = 0.5,
) -> tuple[float, float, float, int]:
    """Returnerer (detection_mAP, classification_mAP, hybrid, antall klasser brukt i snitt)."""
    total_gt = sum(len(v) for v in gt_by_stem.values())
    det = detection_map_global(preds_by_stem, gt_by_stem, iou_thresh, total_gt)

    aps: list[float] = []
    for c in range(nc):
        gt_count = sum(1 for stem in gt_by_stem for gc, _ in gt_by_stem[stem] if gc == c)
        if gt_count == 0:
            continue
        aps.append(
            classification_map_global_per_class(preds_by_stem, gt_by_stem, c, iou_thresh, gt_count)
        )
    cls_map = float(np.mean(aps)) if aps else 0.0
    hybrid = 0.7 * det + 0.3 * cls_map
    return det, cls_map, hybrid, len(aps)
