# Silicon Vikings — NM i AI 2026

Monorepo for alle tre oppgavene: **NorgesGruppen** (objektdeteksjon), **Tripletex** (regnskapsagent), **Astar Island** (verdensprediksjon). Se også [dokumentasjon](https://app.ainm.no/docs).

## Mappestruktur

| Mappe | Innhold |
|-------|---------|
| [norgesgruppen/](norgesgruppen/) | YOLOv8-pipeline, `submission/run.py`, trenings-skript |
| [tripletex/](tripletex/) | FastAPI `/solve`, LLM-agent, Docker/Cloud Run |
| [astar/](astar/) | API-klient, utforskning, prediksjonsinnsending |
| [gcp/](gcp/) | Google Cloud-oppsett (primært Tripletex) |
| `data/` | COCO-data og YOLO-eksport (`data/yolo/` etter `prepare_dataset`) |

## NorgesGruppen — rask start

**Scoring (hybrid):** `0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5`.  
Standard `run.py` sender ut modellens `category_id` per boks (full scorepotensial). Valgfritt: `--detect-only` setter `category_id=0` overalt (typisk maks ~70 % av hybrid score — ren deteksjon).

```bash
pip install -r requirements.txt
python norgesgruppen/scripts/prepare_dataset.py
python norgesgruppen/scripts/train_yolo.py
cp runs/ngd_yolo/baseline/weights/best.pt norgesgruppen/submission/best.pt
# Lager <repo>/submission.zip med run.py + vekter i roten (ikke nested mappe)
bash norgesgruppen/scripts/package_submission.sh
# valgfritt annet navn/sti: bash norgesgruppen/scripts/package_submission.sh /sti/submission.zip
unzip -l submission.zip | head   # skal vise run.py på rot
```

Eldre snarvei fra rot: `python scripts/prepare_dataset.py` (videresender til `norgesgruppen/scripts/`).

**PyTorch 2.6:** trenings-skript laster `torch_compat` for YOLO `.pt`-vekter. `submission/run.py` har innebygd samme patch (selvstendig zip til konkurransen).

**Vekter i zip:** `run.py` bruker `model.onnx` hvis den finnes, ellers `best.pt`. Du kan også pakke med `python norgesgruppen/scripts/package_submission.py -o submission.zip`.

**Etter trening — tall:** `python norgesgruppen/scripts/eval_ngd.py --model runs/ngd_yolo/baseline/weights/best.pt`  
(detection mAP@0.5, classification mAP@0.5, hybrid 0.7/0.3 — se docstring; avvik fra offisiell leaderboard er mulig).

**Etter trening — bilder:** `python norgesgruppen/scripts/visualize_predictions.py --model runs/ngd_yolo/baseline/weights/best.pt --tag baseline_v1`  
→ `runs/predictions/baseline_v1/`.

**Hvor YOLO lagrer filer:** [norgesgruppen/TRAINING_OUTPUTS.md](norgesgruppen/TRAINING_OUTPUTS.md)

**Hele løypa (benchmark → trening → run.py → eval → bilder):** [norgesgruppen/PIPELINE.md](norgesgruppen/PIPELINE.md)

## Tripletex — rask start

```bash
pip install -r tripletex/requirements.txt
export GEMINI_API_KEY=...   # eller OPENAI_API_KEY
export PYTHONPATH=.
uvicorn tripletex.main:app --host 0.0.0.0 --port 8000
```

Deploy: [gcp/setup.md](gcp/setup.md) og [tripletex/Dockerfile](tripletex/Dockerfile).

## Astar Island — rask start

På macOS finnes ofte bare `python3`; bruk prosjektets **venv** (eller full sti til interpreteren).

```bash
cd Silicon-Vikings
python3 -m venv .venv                    # første gang
source .venv/bin/activate              # hver terminaløkt (eller bruk .venv/bin/python direkte)
pip install -r astar/requirements.txt
cp .env.example .env                   # deretter: rediger .env og lim inn token (IKKE i .env.example)
export PYTHONPATH=.

python -m astar.collect --dry-run
python -m astar.collect --max-queries 50
python -m astar.predict --dry-run --round-dir astar/analysis/rounds/<uuid>
python -m astar.predict
python -m astar.explore --test
python -m astar.explore --max-queries 50
python -m astar.visualize --round-dir astar/analysis/rounds/<uuid> --seed 0 -o /tmp/v.png
```

Uten `activate` (samme effekt):

```bash
./.venv/bin/pip install -r astar/requirements.txt
export PYTHONPATH=.
./.venv/bin/python -m astar.explore --test
```

Se [astar/PLAN.md](astar/PLAN.md) for flyt og strategi.

## Konkurranse

- Totalscore: gjennomsnitt av de tre oppgavene (ca. 33 % hver).
- GCP (Cloud Run, Vertex/Gemini) er valgfritt men anbefalt for Tripletex HTTPS-endepunkt — se [gcp/PLAN.md](gcp/PLAN.md).
