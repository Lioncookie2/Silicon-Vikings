# NorgesGruppen — komplett pipeline (tren → run.py → tall → bilder)

Målet er én forutsigbar loop for å sammenligne modeller og oppsett.

## 0. Forutsetninger

```bash
cd <repo>
source .venv/bin/activate
python norgesgruppen/scripts/prepare_dataset.py   # → data/yolo/data.yaml
```

## 1. Hvor lang tid tar trening?

**Én epoch** (inkl. validering etter epoch) + **lineære estimater** for 50 / 100 / 200 epochs:

```bash
python norgesgruppen/scripts/benchmark_epoch_time.py --batch 4 --imgsz 640 --workers 0
```

Juster `--batch`, `--imgsz`, `--weights` som i vanlig trening.  
Output-mappe: `runs/ngd_yolo/_epoch_benchmark/` (kan slettes etterpå).

**Full trening:**

```bash
python norgesgruppen/scripts/train_yolo.py --imgsz 640 --batch 4 --workers 0 --epochs 100
```

Vekter: `runs/ngd_yolo/<name>/weights/best.pt` — se [TRAINING_OUTPUTS.md](TRAINING_OUTPUTS.md).

## 2. `run.py` (konkurranseformat)

Offisiell innsending ligger i **`norgesgruppen/submission/run.py`** (kopieres inn i zip).

- Laster **`model.onnx`** hvis den finnes, ellers **`best.pt`** (i samme mappe som `run.py`).
- `--input` = bildemappe, `--output` = `predictions.json` (liste med `image_id`, `category_id`, `bbox` [x,y,w,h], `score`).

**Lokal test (val-bilder):**

```bash
cp runs/ngd_yolo/baseline/weights/best.pt norgesgruppen/submission/best.pt
python norgesgruppen/submission/run.py \
  --input data/yolo/images/val \
  --output runs/eval/preds.json \
  --conf 0.001 --imgsz 1280
```

## 3. Evaluering — to måter (samme metrikker)

### A) Direkte mot YOLO (raskest å iterere)

```bash
python norgesgruppen/scripts/eval_ngd.py --model runs/ngd_yolo/baseline/weights/best.pt --conf 0.001 --imgsz 1280
```

### B) End-to-end via `run.py` (samme kode som innsending)

```bash
python norgesgruppen/scripts/evaluate_submission_run.py \
  --copy-weights runs/ngd_yolo/baseline/weights/best.pt \
  --conf 0.001 --imgsz 1280
```

Begge gir **detection mAP@0.5**, **classification mAP@0.5**, **hybrid = 0.7×det + 0.3×cls** (lokal tilnærming til leaderboard).

Implementasjon: `norgesgruppen/scripts/metrics_ngd.py`.

## 4. Visuell sjekk (bokser + navn + confidence)

```bash
python norgesgruppen/scripts/visualize_predictions.py \
  --model runs/ngd_yolo/baseline/weights/best.pt \
  --tag baseline_val --limit 30
```

Output: `runs/predictions/<tag>/` (gitignored).

## 5. Sammenligning av modeller

| Steg | Noter i `experiments/results.md` |
|------|-----------------------------------|
| Trening | run-navn, epochs, imgsz/batch |
| `eval_ngd.py` eller `evaluate_submission_run.py` | det / cls / hybrid (samme `--conf`/`--imgsz` for rettferdig sammenligning) |
| `visualize_predictions.py --tag` | kort notat om feiltyper du ser |

## Kort referanse

| Skript | Formål |
|--------|--------|
| `benchmark_epoch_time.py` | Tid per epoch + estimat 50/100/200 |
| `train_yolo.py` | Trening |
| `submission/run.py` | Inferens → JSON (konkurranse) |
| `eval_ngd.py` | mAP direkte fra YOLO |
| `evaluate_submission_run.py` | mAP etter faktisk `run.py` |
| `visualize_predictions.py` | Overlay-bilder |
