# Astar Island

## Oppsett

1. Logg inn på [app.ainm.no](https://app.ainm.no)
2. Hent JWT fra nettleser (cookie `access_token` eller Authorization header)
3. Sett token: `export ACCESS_TOKEN='...'` **eller** `cp .env.example .env` og rediger **`.env`** (lim inn JWT på `ACCESS_TOKEN=`). **Ikke** legg token i `.env.example` — den filen kan committes; kun **`.env`** leses av klienten.
4. Bruk **venv** (mange systemer har ikke `python`, bare `python3`):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r astar/requirements.txt
   ```

   Alternativt uten `activate`: `./.venv/bin/pip install ...` og `./.venv/bin/python -m astar.explore --test`

## Flyt (dagens kode)

| Steg | Hvor | Hva |
|------|------|-----|
| Aktiv runde | `client.list_rounds()` | Finner første runde med `status == "active"`. |
| Rundedetaljer | `client.get_round(round_id)` | `round_id` er **UUID-streng** (ikke int). Gir bl.a. `map_width`, `map_height`, `initial_states` (ett objekt per seed med `grid` og `settlements`). |
| Observe | `client.simulate(...)` | Én **stokastisk** full simulering; svar inneholder kun **viewport**-utklipp av slutt-`grid` + `settlements` i vinduet, pluss `queries_used` / `queries_max`. |
| Submit | `client.submit_prediction(...)` | Sender `prediction[y][x][class]` med 6 sannsynligheter per celle (sum 1, ingen harde nuller). |

Hjelpefunksjon: `get_active_round(client)` → `(round_id: str, detail: dict)`.

## Datainnsamling — anbefalt strategi

- **Aktiv runde (anbefalt): bruk queries i batcher**, ikke nødvendigvis alle 50 med én gang. Kjør f.eks. 10 simulate-kall, kjør `analyze_explore`, kjør `predict --baseline global_model --explore-dir ...` (dry-run), og gjenta med nye 10 når du vil oppdatere kalibreringen. Slik sparer du budsjett til senere runder av hypotesetesting.
- **Inkrementell explore:** Nye kall skrives som nye `runs/run_#####.json`; eksisterende filer overskrives ikke. Bruk **samme mappe** på batch 2+ via `--append-dir <sti>` eller `--out-dir <samme sti>`. Stride-planen **fortsetter** etter antall allerede lagrede runs (unngår duplikat-viewports). Logg: `explore_batches.jsonl` + konsolloppsummering (før/nye/totalt).
- **`analyze_explore`** leser **alle** `run_*.json` i `runs/` og overskriver `analysis_summary.json` med **samlet** statistikk (ikke bare siste batch).
- **Predict:** `global_model` + `--explore-dir` gir gradvis sterkere runde-kalibrering etter hvert som flere runs finnes i mappen; uten gyldig `analysis_summary.json` brukes ren global modell.

### `collect.py` (alternativ bulk)

- **Fordeling på seeds:** `allocate_queries_per_seed(50, 5)` → **10 kall per seed** (jevnt; rest ved ulike budsjett går til lavere indekser).
- **Viewport-valg:** `astar/collect.py` bruker **balanced** (3×3 + senter) der mulig, supplert med **stride-cover** slik at du får nok unike vinduer innenfor hver seeds 10 kall.
- **Prioritet innen kart:** Fokuser vinduer som overlapper **kyst** (land ved hav), **tett på initial settlements**, og **varierte inland-soner** — der er dynamikken størst. Fjell og hav er mer forutsigbare, men litt dekning bekrefter statikk.
- **Skjulte parametre:** Du kan ikke estimere dem eksplisitt; bruk observasjoner til empiriske **sannsynlighetsfordelinger** (f.eks. andel ruin i utsnitt, spredning av bosetning) og la terrain-baseline (`astar/baseline.py`) kode inn struktur fra startkartet.

## Kommandoer (fra repo-roten)

```bash
cd Silicon-Vikings
source .venv/bin/activate   # eller: export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH=.

# Snapshot av aktiv runde + plan uten å bruke simulate-budsjett
python -m astar.collect --dry-run

# Full innsamling (50 simulate-kall) → astar/analysis/rounds/<round_id>/
python -m astar.collect --max-queries 50

# Simulate via API → én JSON per kall under astar/analysis/explore/<tid>_<round>/
python -m astar.explore --test
python -m astar.explore --max-queries 50

# Oppsummer lagret økt + PNG-plots (baseline-heatmap + viewports)
python -m astar.analyze_explore astar/analysis/explore/<session_mappe>
python -m astar.visualize --explore-dir astar/analysis/explore/<session_mappe> --seed 0
# → astar/analysis/plots/<session_mappe>/seed0_*.png

# Historical **completed** rounds = eneste supervised kilde (ground truth i datasettet)
python -m astar.fetch_historical
python -m astar.build_dataset

# Rask iterasjon (2–3 siste runder, lite datasett — ev. port-features)
python -m astar.build_dataset --latest-n-rounds 3 --out-npz astar/data/datasets/historical_cells_fast.npz
python -m astar.train_model --dataset astar/data/datasets/historical_cells_fast.npz --model gradient_boosting
python -m astar.evaluate_historical_rounds --latest-n-rounds 3 --mode holdout-last-n --holdout-n 1
# Alternativt: tren direkte fra disk uten npz (samme filter som over)
python -m astar.train_model --latest-n-rounds 3 --model gradient_boosting --compare-all
# Settlement-port features (kun felt fra initial_states: has_port m.m.; ingen rik økonomi i historical JSON)
python -m astar.build_dataset --latest-n-rounds 3 --feature-set settlement_port --out-npz astar/data/datasets/historical_cells_port.npz
python -m astar.train_model --dataset astar/data/datasets/historical_cells_port.npz --compare-all

# Sammenlign RF / GB / LR på samme val-split; vinner = lavest cross-entropy mot sann y_prob (ikke accuracy)
python -m astar.train_model --compare-all
# → astar/data/models/model_comparison.json, per-modell *.joblib, best_global_model.joblib

# Round-aware eval (leave-one-round-out): generalisering til nye runder, ikke tilfeldig cellesplit
python -m astar.evaluate_historical_rounds
python -m astar.evaluate_historical_rounds --models random_forest,logistic_regression,gradient_boosting,global_ensemble
python -m astar.evaluate_historical_rounds --ablation terrain_settlement
python -m astar.evaluate_historical_rounds --mode holdout-last-n --holdout-n 2
# → astar/data/evals/historical_round_eval.json, historical_round_eval_rows.csv

# Blend global vs terrain-prior mot historical GT (bruker best_global_model.joblib)
python -m astar.tune_blend_historical --latest-n-rounds 6
# Med calibration_suggestion fra analyze_explore (bytt til din nyeste økt):
python -m astar.tune_blend_historical --latest-n-rounds 6 --explore-dir astar/analysis/explore/20260321_145122_cc5442dd
# → astar/data/evals/blend_weight_tune.json

# Final kandidat-sammenligning (LOO: GB × feature_set × blend A–E; tar ofte 2–10+ min)
# Fra repo-roten med PYTHONPATH=. (venv etter behov)
export PYTHONPATH=.
python -m astar.final_candidate_compare --latest-n-rounds 6
python -m astar.final_candidate_compare --latest-n-rounds 10 --out-json astar/data/evals/final_candidate_compare_n10.json
python -m astar.final_candidate_compare --latest-n-rounds 6 --explore-dir astar/analysis/explore/20260321_145122_cc5442dd
# → astar/data/evals/final_candidate_compare.json

# Probability improvement stack (entropy-weighted KL + estimat score 0–100; krever scipy)
python -m astar.score_evaluate_historical --latest-n-rounds 10
python -m astar.score_evaluate_historical --latest-n-rounds 10 --calibration-json astar/data/models/prob_calibration.json
# Lær klasse-vekter (holdout); sjekk val metrics + recommendation i JSON
python -m astar.calibrate_probs_historical --holdout-last-n 3 --explore-dir astar/analysis/explore/20260321_145122_cc5442dd --global-weight 0.7 --explore-weight 0.3
python -m astar.calibrate_probs_historical --mode regime --holdout-last-n 3 --explore-dir astar/analysis/explore/20260321_145122_cc5442dd
# Predict med valgfri kalibrering: --calibration-json astar/data/models/prob_calibration.json
# Valgfri explore-hierarki: legg explore_calibration_hierarchy.json i samme mappe som analysis_summary.json
python -m astar.candidate_score_pipeline --latest-n-rounds 10 --explore-dir astar/analysis/explore/20260321_145122_cc5442dd --calibration-global astar/data/models/prob_calibration.json
python -m astar.historical_query_simulator --round-dir astar/data/historical/<uuid> --seed-index 0 --budget 20

# Enkeltmodell (speiler også til global_model.joblib for bakoverkompatibilitet)
python -m astar.train_model --model random_forest
# Valgfritt: marker som aktiv best for predict uten full sammenligning
# python -m astar.train_model --model gradient_boosting --promote-best

# Prediksjon: terrain (default), uniform, eller global historisk modell + valgfri explore-blend
python -m astar.predict --dry-run --round-dir astar/analysis/rounds/<uuid>
python -m astar.predict --dry-run --round-dir ... --save-predictions-dir /tmp/pred
python -m astar.predict --baseline uniform --dry-run
python -m astar.predict   # submit aktiv runde

# Kun én seed (f.eks. etter 429 på én): resubmit overskriver samme seed
python -m astar.predict --baseline global_model --explore-dir "$EXP" --seed-indices 2 \
  --global-weight 0.6 --explore-weight 0.4 --prob-floor 0.01

# Aktiv runde — batchvis (eksempel: 10 + 10 queries, samme explore-mappe)
EXP=astar/analysis/explore/min_aktiv_runde
python -m astar.explore --out-dir "$EXP" --max-queries 10
python -m astar.analyze_explore "$EXP"
python -m astar.predict --baseline global_model --round-dir astar/analysis/rounds/<uuid> \
  --explore-dir "$EXP" --global-weight 0.6 --explore-weight 0.4 --dry-run --no-submit

python -m astar.explore --append-dir "$EXP" --max-queries 10
python -m astar.analyze_explore "$EXP"
python -m astar.predict --baseline global_model --round-dir astar/analysis/rounds/<uuid> \
  --explore-dir "$EXP" --global-weight 0.6 --explore-weight 0.4 --dry-run --no-submit

# Submit nar du er fornoyd (API / aktiv runde):
# python -m astar.predict --baseline global_model --explore-dir "$EXP" --global-weight 0.6 --explore-weight 0.4

# (Alternativ) Alt i én økt:
python -m astar.explore --max-queries 50
python -m astar.analyze_explore astar/analysis/explore/<session_mappe>
python -m astar.predict --baseline global_model \
  --round-dir astar/analysis/rounds/<uuid> \
  --explore-dir astar/analysis/explore/<session_mappe> \
  --global-weight 0.6 --explore-weight 0.4 --prob-floor 0.01 \
  --dry-run --save-predictions-dir /tmp/pred --predict-summary-json /tmp/predict_summary.json
```

**Global supervisert modell** trenes **kun** på `historical_cells.npz` fra fullførte runder. Aktiv runde (f.eks. 16) med explore-simuleringer gir **ikke** supervised labels her — den dataen brukes bare som **kalibrering** oppå den globale modellen.

**Modellvalg:** Kjør `train_model --compare-all` og velg vinner etter **lavest validerings-cross-entropy** mot `y_prob` i datasettet; accuracy er sekundær. Resultat: `model_comparison.json` samt `best_global_model.joblib` + `best_global_model_meta.json`. `predict --baseline global_model` laster automatisk **best** hvis filen finnes, ellers `global_model.joblib` (legacy).

**Runde-spesifikk kalibrering** kommer fra explore: `analyze_explore` → `calibration_suggestion` i `analysis_summary.json`, blandet med global `predict_proba` (standard 0.6/0.4). Mangler analyse → ren global modell.

**Valgfri global ensemble i predict:** `--baseline global_ensemble` vekter `gradient_boosting.joblib` / `logistic_regression.joblib` / `random_forest.joblib` (0.6 / 0.25 / 0.15, renormalisert hvis noen mangler). Samme explore-blend og `prob_floor` som `global_model`. Sammenlign dry-runs med `--predict-summary-json` (ulik `baseline` + `ensemble_members`).

**Siste queries (heuristikk):** `python -m astar.diagnose_next_queries --round-dir ... --explore-dir "$EXP" -o next_q.json` — foreslår 1–3 viewport-kandidater (entropi, dynamiske klasser, kant/coast, explore-dekning).

Vektet multi-modell-ensemble **i train-rapporten** (`ensemble_todo` i `model_comparison.json`) er fortsatt fremtidig arbeid; runtime-ensemble er dekket av `global_ensemble` over.

```bash
# Visualisering (krever display eller -o fil.png)
python -m astar.visualize --round-dir astar/analysis/rounds/<uuid> --seed 0 \
  --runs astar/analysis/rounds/<uuid>/simulate_runs.json -o /tmp/astar_viz.png
```

## Terrengklasser (6)

Tom (inkl. hav/slette), bosetning, havn, ruin, skog, fjell — sannsynligheter per celle må summere til 1; unngå eksakt **0.0** (bruk gulv f.eks. 0.01).

## Filer

| Fil | Rolle |
|-----|--------|
| `client.py` | REST-klient, UUID `round_id` |
| `baseline.py` | Terrain-prior, validering, gulv + renormalisering |
| `exploration.py` | Viewport-fordeling |
| `collect.py` | Lagre `round_detail.json`, `manifest.json`, `simulate_runs.json`, `exploration_plan.json` (dry-run) |
| `predict.py` | Submit / dry-run / `--baseline global_model` + explore-blend |
| `train_model.py` | `--compare-all` → `model_comparison.json`, `best_global_model.joblib` |
| `data/models/model_comparison.json` | CE/accuracy per modell, rangering, vinner |
| `global_ensemble.py` | Laster GB/LR/RF + vektet `predict_proba` |
| `diagnose_next_queries.py` | Forslag til neste simulate-viewports |
| `evaluate_historical_rounds.py` | LOO / holdout eval per runde (CE, KL, Brier, per-seed) |
| `cell_features.py` | Felles celle-features for `build_dataset` og `predict` |
| `global_model_loader.py` | Laster `global_model.joblib` med tydelig feil ved manglende fil |
| `visualize.py` | Startkart + heatmaps P(S)/P(P)/P(R) |
| `explore.py` | Alternativ utforskning |
