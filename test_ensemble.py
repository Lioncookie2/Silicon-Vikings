import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path
from PIL import Image
import torch

_orig = torch.load
torch.load = lambda *args, **kwargs: _orig(*args, **{**kwargs, "weights_only": False})

from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

m1 = YOLO("/Users/tonne/Desktop/Silicon-Vikings/norgesgruppen/submission/best.pt")
m2 = YOLO("/Users/tonne/Desktop/Silicon-Vikings/norgesgruppen/submission/model.onnx")

print("Modeller lastet. Prøver inferens på 1 bilde...")
img_path = "/Users/tonne/Desktop/Silicon-Vikings/data/yolo/images/val/img_00002.jpg"
r1 = m1.predict(img_path, verbose=False, device='cpu', imgsz=1280)[0]
r2 = m2.predict(img_path, verbose=False, device='cpu', imgsz=1280)[0]

print(f"Modell 1 fant {len(r1.boxes)} bokser")
print(f"Modell 2 fant {len(r2.boxes)} bokser")
