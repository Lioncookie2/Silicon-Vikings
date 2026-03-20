# NorgesGruppen Data — Object Detection

## Mål

Maksimere `0.7 × detection_mAP + 0.3 × classification_mAP` på hyllebilder.

## Pipeline

1. `python norgesgruppen/scripts/prepare_dataset.py` — COCO → YOLO, genererer `data/yolo/data.yaml`
2. `python norgesgruppen/scripts/train_yolo.py` — trening til `runs/ngd_yolo/baseline/weights/best.pt`
3. Kopier `best.pt` til `norgesgruppen/submission/best.pt`
4. Zip **innholdet** av `norgesgruppen/submission/` (ikke mappen) → `submission.zip`  
   Kun tillatte filtyper i zip (se konkurranseregler) — typisk `run.py` + `best.pt`.

## Oppgraderinger

- Større modell: `python norgesgruppen/scripts/train_yolo.py --weights yolov8l.pt --name large`
- Augmentering: `--mosaic 0.8 --mixup 0.1 --copy-paste 0.1`
- Ensemble: `norgesgruppen/scripts/ensemble_predict.py` med flere `best.pt`

## Viktig

`run.py` i innsendingen må ikke bruke forbudte moduler (se konkurranse-dokumentasjon).

Se også [experiments/results.md](experiments/results.md) for kjørelogg.
