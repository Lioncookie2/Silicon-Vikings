# Tripletex — terminal og Cloud Shell (kun denne oppgaven)

Kort oppskrift for livet i `gcloud`, logger og deploy. Resten av repoet (Astar, NorgesGruppen) ignorerer du hvis du bare jobber med Tripletex.

## 0. Én gang per økt

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

Bytt `YOUR_PROJECT_ID` med lab-prosjektet deres (samme som i Cloud Shell-profilen).

## 1. Miljøvariabler (lim inn og tilpass)

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export REGION=europe-north1
export TTX_SERVICE=tripletex-agent
```

## 2. Hent siste agent-logger (etter en submit på ainm)

Krever `roles/logging.viewer` (eller bredere) på prosjektet.

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="'"$TTX_SERVICE"'" AND jsonPayload.log_schema="v2-rich"' \
  --project "$PROJECT_ID" \
  --freshness=2h \
  --limit=80 \
  --format="table(timestamp,severity,jsonPayload.request_id,jsonPayload.agent_log)"
```

Hvis `agent_log` ser tom ut, prøv `jsonPayload.message` i stedet:

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="'"$TTX_SERVICE"'" AND jsonPayload.log_schema="v2-rich"' \
  --project "$PROJECT_ID" \
  --freshness=2h \
  --limit=40 \
  --format="value(jsonPayload.message)"
```

**Sjekk at du ser `v2-rich`:** Korte linjer uten `step=` / `path=` betyr ofte gammelt Cloud Run-image — bygg og deploy på nytt (pkt. 5).

## 3. Én konkret `/solve` (lim inn `request_id` fra loggen)

```bash
RID=PASTE-FULL-UUID-HERE
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="'"$TTX_SERVICE"'" AND jsonPayload.request_id="'"$RID"'"' \
  --project "$PROJECT_ID" \
  --freshness=48h \
  --limit=300 \
  --format="value(jsonPayload.agent_log)"
```

Lagre til fil for å dele eller lete i `less`:

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="'"$TTX_SERVICE"'" AND jsonPayload.request_id="'"$RID"'"' \
  --project "$PROJECT_ID" \
  --freshness=48h \
  --limit=300 \
  --format=json > "/tmp/ttx-${RID}.json"
```

## 4. Automatisk sammendrag (API-feil per path, per request)

Fra **repo-roten** `Silicon-Vikings/` (der `tripletex/scripts/` ligger):

```bash
cd /path/to/Silicon-Vikings

python3 tripletex/scripts/summarize_agent_logs.py \
  --project "$PROJECT_ID" \
  --freshness 24h \
  --limit 800 \
  --format md \
  -o /tmp/ttx-agent-report.md

cat /tmp/ttx-agent-report.md
```

Kun advarsler og opp:

```bash
python3 tripletex/scripts/summarize_agent_logs.py \
  --project "$PROJECT_ID" \
  --freshness 7d \
  --limit 1200 \
  --filter 'severity>=WARNING' \
  --format md \
  -o /tmp/ttx-warnings.md
```

CSV til Sheets:

```bash
python3 tripletex/scripts/summarize_agent_logs.py \
  --project "$PROJECT_ID" \
  --freshness 7d \
  --format csv \
  -o /tmp/ttx-runs.csv
```

## 5. Bygg og deploy (oppdatert kode ut til samme URL)

Fra repo-roten — se også [gcp/setup.md](../gcp/setup.md).

```bash
cd /path/to/Silicon-Vikings
export REGION=europe-north1

gcloud builds submit --config tripletex/cloudbuild.yaml .
```

Deretter Cloud Run (tilpass image-URL til deres Artifact Registry-navn):

```bash
gcloud run deploy tripletex-agent \
  --image "${REGION}-docker.pkg.dev/${PROJECT_ID}/nmiai/tripletex-agent:latest" \
  --region "$REGION" \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=...
```

Anbefalt på sikt: API-nøkkel i Secret Manager (`--set-secrets`), ikke klartekst i historikk.

## 6. Valgfritt: JSONL-kjøringer til GCS

Gir én fil per `request_id` for batch-analyse uten å parse Logging-UI. Se tabellen over miljøvariabler i [PLAN.md](PLAN.md) (`TRIPLETEX_RUN_LOG_GCS_BUCKET`, IAM `storage.objectCreator`).

Liste dagens filer:

```bash
gsutil ls "gs://DIN-BUCKET/tripletex-runs/$(date -u +%Y-%m-%d)/"
```

## 7. Hva du sender tilbake (mal)

Etter en kjøring med **lav score**, lim inn dette (fyll ut):

1. **request_id** (UUID fra logger).
2. **run_summary**-linje om du fant den, eller utdrag fra `summarize_agent_logs.py` for den requesten.
3. **Topp api_error-path** (f.eks. `/ledger/voucher`) og **første validationMessages** / status.
4. **task_preview** eller første setning av oppgaveteksten (ikke hemmeligheter).
5. Om du nettopp **deployet**: bekreft at logger har `log_schema=v2-rich`.

Da kan neste steg i koden være: justere `SYSTEM_PROMPT`, legge til `task_handlers`, eller fikse klient — basert på faktisk feilmønster.

## Hurtig: source helper-skript

Fra repo-roten:

```bash
source tripletex/scripts/tripletex_shell.sh
ttx_help
```

Sett `TTX_PROJECT_ID` (eller `PROJECT_ID`) før du kaller `ttx_logs_recent`, `ttx_logs_rid`, `ttx_summarize`.
