from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse, json
from pathlib import Path
from ultralytics import YOLO

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--tta", action="store_true", default=False)
    parser.add_argument("--detect-only", action="store_true")
    args = parser.parse_args()

    input_dir, output_path = Path(args.input), Path(args.output)
    model = YOLO(str(Path(__file__).resolve().parent / "model.onnx"))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_rows = []
    
    for p in sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]):
        stem = p.stem
        iid = int("".join(c for c in (stem.split("_")[-1] if "_" in stem else stem) if c.isdigit()).lstrip("0") or "0")
        
        # Test NMS IOU 0.6
        results = model.predict(source=str(p), conf=args.conf, imgsz=args.imgsz, iou=0.6, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0: continue
            
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            output_rows.append({"image_id": iid, "category_id": 0 if args.detect_only else int(cls[i]), "bbox": [float(x1), float(y1), float(w), float(h)], "score": float(conf[i])})

    with output_path.open("w", encoding="utf-8") as f: json.dump(output_rows, f, ensure_ascii=False)

if __name__ == "__main__": main()
