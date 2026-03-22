import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json, sys
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path("norgesgruppen/scripts").resolve()))
from metrics_ngd import *
from ultralytics import YOLO

try:
    from ensemble_boxes import weighted_boxes_fusion
except ImportError:
    weighted_boxes_fusion = None

m = YOLO("/Users/tonne/Desktop/Silicon-Vikings/norgesgruppen/submission/model.onnx")
img_dir = Path("data/yolo/images/val")

results_json = []

for p in sorted(img_dir.iterdir()):
    if p.suffix.lower() not in ['.jpg', '.png', '.jpeg', '.bmp']: continue
    
    stem = p.stem
    iid = int("".join(c for c in (stem.split("_")[-1] if "_" in stem else stem) if c.isdigit()).lstrip("0") or "0")
    
    res1 = m.predict(source=str(p), conf=0.001, imgsz=1280, verbose=False)[0]
    
    with Image.open(p) as img:
        w, h = img.size
        img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
        
    res2 = m.predict(source=img_flip, conf=0.001, imgsz=1280, verbose=False)[0]
    
    boxes1, scores1, labels1 = [], [], []
    if res1.boxes is not None and len(res1.boxes) > 0:
        xyxy1 = res1.boxes.xyxy.cpu().numpy()
        cf1 = res1.boxes.conf.cpu().numpy()
        cl1 = res1.boxes.cls.cpu().numpy()
        for i in range(len(xyxy1)):
            boxes1.append([xyxy1[i][0]/w, xyxy1[i][1]/h, xyxy1[i][2]/w, xyxy1[i][3]/h])
            scores1.append(float(cf1[i]))
            labels1.append(int(cl1[i]))
            
    boxes2, scores2, labels2 = [], [], []
    if res2.boxes is not None and len(res2.boxes) > 0:
        xyxy2 = res2.boxes.xyxy.cpu().numpy()
        cf2 = res2.boxes.conf.cpu().numpy()
        cl2 = res2.boxes.cls.cpu().numpy()
        for i in range(len(xyxy2)):
            orig_x1, orig_y1, orig_x2, orig_y2 = xyxy2[i]
            x1 = w - orig_x2
            x2 = w - orig_x1
            boxes2.append([x1/w, orig_y1/h, x2/w, orig_y2/h])
            scores2.append(float(cf2[i]))
            labels2.append(int(cl2[i]))
            
    b_list = [b for b in [boxes1, boxes2] if len(b)>0]
    s_list = [s for s in [scores1, scores2] if len(s)>0]
    l_list = [l for l in [labels1, labels2] if len(l)>0]
    
    if len(b_list) > 0:
        fuse_b, fuse_s, fuse_l = weighted_boxes_fusion(b_list, s_list, l_list, weights=[1.0, 1.0], iou_thr=0.6, skip_box_thr=0.0)
        for i in range(len(fuse_b)):
            bx1, by1, bx2, by2 = fuse_b[i]
            x1, y1, x2, y2 = bx1*w, by1*h, bx2*w, by2*h
            bw, bh = max(0.0, x2-x1), max(0.0, y2-y1)
            results_json.append({"image_id": iid, "category_id": int(fuse_l[i]), "bbox": [x1, y1, bw, bh], "score": float(fuse_s[i])})

out_path = Path("runs/eval/predictions_onnx_tta.json")
out_path.parent.mkdir(exist_ok=True, parents=True)
with out_path.open("w") as f:
    json.dump(results_json, f)

nc, _ = parse_data_yaml_nc_and_names(Path("data/yolo/data.yaml"))
gt_by_stem, total_gt = load_gt_for_split(img_dir, Path("data/yolo/labels/val"))
id_to_stem = build_image_id_to_stem(img_dir)
preds_by_stem = load_predictions_json_to_preds(out_path, id_to_stem)
det, cls_map, hybrid, n_cls = compute_hybrid_maps(preds_by_stem, gt_by_stem, nc)
print(f"det={det:.4f}, cls={cls_map:.4f}, hybrid={hybrid:.4f}")
