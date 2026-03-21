# Trenings- og innsendingslogg

Fyll inn etter hver kjøring. Bruk `eval_ngd.py` for **sammenlignbare tall** mellom modeller (samme val-split, samme `--conf` / `--imgsz` om mulig).

| Dato | Run-navn (`--name`) | Modell / vekter | epochs | det_mAP@0.5 | cls_mAP@0.5 | hybrid (0.7/0.3) | YOLO val mAP50 | Notat |
|------|---------------------|-----------------|--------|-------------|-------------|------------------|----------------|-------|
|      | baseline            | yolov8m → best  | 100    |             |             |                  |                |       |

**Kolonner:**

- **det_mAP@0.5 / cls_mAP@0.5 / hybrid:** fra `python norgesgruppen/scripts/eval_ngd.py --model ...`
- **YOLO val mAP50:** fra `runs/ngd_yolo/<name>/results.csv` siste rad (`metrics/mAP50(B)` e.l.) — COCO-lignende, ikke identisk med hybriden over

**Visual sjekk:** `visualize_predictions.py --tag <beskrivende_navn>` → `runs/predictions/<tag>/`
