# YOLOv8 Baseline for NorgesGruppen Product Detection

Dette prosjektet setter opp en første fungerende YOLOv8-baseline for object detection på butikkhyllebilder.

## 1) Datastruktur

Plasser data slik:

- `data/raw/train_images/` - alle treningsbilder
- `data/raw/annotations.json` - COCO-annotasjoner
- `data/raw/product_images/` - referansebilder (brukes ikke i baseline v1)

## 2) Installer avhengigheter

Kjør fra prosjektroten (`Norgesgruppen Data`):

```bash
pip install -r requirements.txt
```

## 3) Klargjor YOLO-datasett

Konverter COCO til YOLO-format og splitt 85/15 train/val:

```bash
python scripts/prepare_dataset.py
```

Valgfrie argumenter:

```bash
python scripts/prepare_dataset.py --seed 42 --train-ratio 0.85
```

Etter kjoering lages:

- `data/yolo/images/train/`
- `data/yolo/images/val/`
- `data/yolo/labels/train/`
- `data/yolo/labels/val/`
- `data/yolo/data.yaml`

## 4) Tren YOLOv8-baseline

Trening starter fra `yolov8m.pt`:

```bash
python scripts/train_yolo.py
```

Standardinnstillinger:

- `imgsz=1280`
- `epochs=100`
- `patience=20`
- `batch=-1` (fallback til `8` ved behov)
- GPU (`device=0`) hvis tilgjengelig, ellers CPU
- output til `runs/ngd_yolo/baseline/`

Forventet modellsti:

- `runs/ngd_yolo/baseline/weights/best.pt`

## 5) Lokal validering i konkurranseformat

Kjor inferens lokalt og skriv JSON i konkurranseformat:

```bash
python scripts/validate_local.py --model runs/ngd_yolo/baseline/weights/best.pt --input data/yolo/images/val --output submission/local_predictions.json --conf 0.25 --imgsz 1280
```

## 6) Klargjor submission

Kopier beste modell til `submission/`:

```bash
copy runs\ngd_yolo\baseline\weights\best.pt submission\best.pt
```

Sorg for at `submission/` inneholder minst:

- `run.py`
- `best.pt`

Pakk innholdet i `submission/` slik at `run.py` ligger i roten av zip-filen.

Eksempel med PowerShell:

```powershell
Compress-Archive -Path submission\* -DestinationPath submission.zip -Force
```

## Konkurransegrensesnitt

Arrangoren kjører:

```bash
python run.py --input /data/images --output /output/predictions.json
```

`submission/run.py` skriver en JSON-liste med objekter:

```json
{
  "image_id": 42,
  "category_id": 17,
  "bbox": [x, y, w, h],
  "score": 0.91
}
```

der `category_id` matcher datasettet, og bbox er i format `[x, y, width, height]`.
