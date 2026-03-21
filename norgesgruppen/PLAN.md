# NorgesGruppen Data — Object Detection

## Mål

Maksimere **hybrid score**: `0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5`.

- **Detection:** IoU ≥ 0.5 mot nærmeste GT-boks; kategori ignoreres.
- **Classification:** IoU ≥ 0.5 **og** riktig `category_id` (356 produktklasser, id 0–355 i trenings-`annotations.json`).
- **Kun deteksjon:** Hvis alle prediksjoner har `category_id: 0`, teller kun detection-delen (ca. maks 70 %). I kode: `python .../run.py ... --detect-only`.

## Pipeline

1. `python norgesgruppen/scripts/prepare_dataset.py` — COCO → YOLO, genererer `data/yolo/data.yaml`  
   Legg `annotations.json` i `data/raw/`. Bilder kan ligge flere steder; skriptet søker bl.a. `data/raw/product_images/`, `data/raw/product_images/train_images/`, `data/raw/train_images/`, pluss stier som matcher COCO `file_name`.
2. `python norgesgruppen/scripts/train_yolo.py` — trening til `runs/ngd_yolo/baseline/weights/best.pt`
3. Kopier vekter til `norgesgruppen/submission/` — f.eks. `best.pt` og/eller eksportert `model.onnx` (`run.py` prioriterer ONNX om begge finnes)
4. Bygg zip: `bash norgesgruppen/scripts/package_submission.sh` → **`submission.zip` i repo-roten** (eller `python norgesgruppen/scripts/package_submission.py -o submission.zip`).  
   Skriptet pakker kun `run.py`, valgfri `utils.py`, og opptil 3 vektfiler — arkivnavn er flate (`run.py` i rot, ikke `submission/run.py`). Verifiser med `unzip -l submission.zip | head`.

## Oppgraderinger

- Større modell: `python norgesgruppen/scripts/train_yolo.py --weights yolov8l.pt --name large`
- Augmentering: `--mosaic 0.8 --mixup 0.1 --copy-paste 0.1`
- Ensemble: `norgesgruppen/scripts/ensemble_predict.py` med flere `best.pt`

## Viktig

`run.py` i innsendingen må ikke bruke forbudte moduler (se konkurranse-dokumentasjon).

Se også [experiments/results.md](experiments/results.md) for kjørelogg.

## Evaluering og visualisering

Se **[PIPELINE.md](PIPELINE.md)** for hele flyten (epoch-benchmark, `run.py`, end-to-end eval, visualisering).

- **YOLO-output (filer, plots):** [TRAINING_OUTPUTS.md](TRAINING_OUTPUTS.md)
- **Direkte mAP (YOLO):** `python norgesgruppen/scripts/eval_ngd.py --model <vekt>`
- **mAP via faktisk `run.py`:** `python norgesgruppen/scripts/evaluate_submission_run.py --copy-weights <vekt>`
- **Overlay-bilder:** `python norgesgruppen/scripts/visualize_predictions.py --model <vekt> --tag <navn>`
