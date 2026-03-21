# Hvor YOLO lagrer resultater etter trening

Treningskallet i `norgesgruppen/scripts/train_yolo.py` bruker:

- `project`: `<repo>/runs/ngd_yolo`
- `name`: standard `baseline` (eller det du setter med `--name`)

## Mapper og filer (typisk)

Alt under én kjøring ligger i:

```text
runs/ngd_yolo/<run_name>/
├── weights/
│   ├── best.pt      # beste valideringsmetrikk (lagre denne til innsending)
│   └── last.pt      # siste epoch
├── results.csv      # metrics per epoch (mAP50, mAP50-95, precision, recall, loss, …)
├── results.png      # kurver (loss + val metrics)
├── confusion_matrix.png
├── confusion_matrix_normalized.png
├── BoxPR_curve.png      # PR per klasse (bokser)
├── BoxP_curve.png
├── BoxR_curve.png
├── BoxF1_curve.png
├── val_batch0_pred.jpg  # eksempelbilder (val)
├── train_batch0.jpg
├── args.yaml            # alle hyperparametre for denne kjøringen
└── events.out.tfevents.*  # TensorBoard (hvis aktivert)
```

Nøyaktige plot-navn kan variere litt med Ultralytics 8.1.x, men `weights/`, `results.csv` og `results.png` er alltid sentrale.

## Viktige metrics i `results.csv`

Typiske kolonner inkluderer (navn kan avvike noe):

- `metrics/mAP50(B)` — mAP ved IoU 0.5 (bokser)
- `metrics/mAP50-95(B)` — mAP over IoU 0.50:0.95
- `metrics/precision(B)`, `metrics/recall(B)`

**Merk:** Dette er **YOLOs standard validering** (COCO-lignende, per klasse). Den er **nær**, men ikke nødvendigvis **identisk** med NorgesGruppens offisielle hybrid-score. Bruk `norgesgruppen/scripts/eval_ngd.py` for tall som matcher konkurransens definisjon (0.7 deteksjon + 0.3 klassifisering).

## Rask kopi til innsending

```bash
cp runs/ngd_yolo/baseline/weights/best.pt norgesgruppen/submission/best.pt
bash norgesgruppen/scripts/package_submission.sh
```
