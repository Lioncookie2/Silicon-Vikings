import json, sys
from pathlib import Path
from ensemble_boxes import weighted_boxes_fusion
sys.path.insert(0, str(Path("norgesgruppen/scripts").resolve()))
from metrics_ngd import *

# Last predictions fra ONNX (som jeg allerede kjørte med iou=0.7 før. Vent, jeg overskrev det med iou=0.6)
