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

```bash
pip install -r requirements.txt
python norgesgruppen/scripts/prepare_dataset.py
python norgesgruppen/scripts/train_yolo.py
cp runs/ngd_yolo/baseline/weights/best.pt norgesgruppen/submission/best.pt
bash norgesgruppen/scripts/package_submission.sh
```

Eldre snarvei fra rot: `python scripts/prepare_dataset.py` (videresender til `norgesgruppen/scripts/`).

**PyTorch 2.6:** trenings-skript laster `torch_compat` for YOLO `.pt`-vekter. `submission/run.py` har innebygd samme patch (selvstendig zip til konkurransen).

## Tripletex — rask start

```bash
pip install -r tripletex/requirements.txt
export GEMINI_API_KEY=...   # eller OPENAI_API_KEY
export PYTHONPATH=.
uvicorn tripletex.main:app --host 0.0.0.0 --port 8000
```

Deploy: [gcp/setup.md](gcp/setup.md) og [tripletex/Dockerfile](tripletex/Dockerfile).

## Astar Island — rask start

```bash
pip install -r astar/requirements.txt
export ACCESS_TOKEN=...   # JWT fra app.ainm.no
export PYTHONPATH=.
python -m astar.predict --dry-run          # uten API
python -m astar.predict                    # sender inn (krever aktiv runde)
python -m astar.explore --max-queries 50   # viewport-utforskning
```

## Konkurranse

- Totalscore: gjennomsnitt av de tre oppgavene (ca. 33 % hver).
- GCP (Cloud Run, Vertex/Gemini) er valgfritt men anbefalt for Tripletex HTTPS-endepunkt — se [gcp/PLAN.md](gcp/PLAN.md).
