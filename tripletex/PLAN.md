# Tripletex — AI Accounting Agent

## Kjøre lokalt

Fra repo-roten (slik at `tripletex` er et Python-pakke):

```bash
export GEMINI_API_KEY=...   # eller GOOGLE_API_KEY, eller OPENAI_API_KEY
export PYTHONPATH=.
uvicorn tripletex.main:app --host 0.0.0.0 --port 8000
```

HTTPS for test: `npx cloudflared tunnel --url http://localhost:8000`

## Endepunkt

- `POST /solve` — body som i [konkurransedokumentasjon](https://app.ainm.no/docs/tripletex/overview)
- `GET /health` — healthcheck for Cloud Run

## Arkitektur

- `main.py` — FastAPI
- `agent.py` — LLM genererer JSON-plan med Tripletex API-kall, deretter kjøring
- `tripletex_client.py` — HTTP mot proxy med Basic auth
- `task_handlers/` — valgfrie deterministiske snarveier (utvid etter hvert)

## Strukturert logging

Alle hendelser skrives som én JSON-linje per event til **stdout** → automatisk sendt til **Cloud Logging** av Cloud Run.

### Modul

`structured_log.py` eksponerer:

- `log_event(severity, message, **fields)` — skriv én JSON-rad til stdout.
- `set_request_id(uuid)` / `get_request_id()` — lagrer request-ID i `contextvars` (én per HTTP-kall).
- `log_api_error(step, method, path, status, body)` — strukturert 4xx/5xx-logg med `validationMessages`.

Sensitive felt (`session_token`, `credentials`, `api_key` m.fl.) strippes automatisk.

### Logs Explorer-filtre

`jsonPayload.message` er **én sammenhengende tekstlinje** (type hendelse + step + path + status + ev. detail).

```
# Alle hendelser for én request:
jsonPayload.request_id="<uuid>"

# Alle feil og advarsler:
severity>=WARNING

# API-feil (teksten starter med api_error | …):
jsonPayload.message=~"^api_error"

# Spesifikk path (bruk jsonPayload.path ELLER søk i message):
jsonPayload.path="/invoice"

# Skjul tomme rader (Uvicorn uten jsonPayload.message):
jsonPayload.message!=""
```

### gcloud — hent logger (lange `jsonPayload.message`)

Bytt `PROJECT_ID` hvis ditt prosjekt er annerledes.

**Kun agent-linjer (filtrerer bort tomme Uvicorn-rader), siste time:**

```bash
export PROJECT_ID=ai-nm26osl-1867

gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="tripletex-agent" AND jsonPayload.log_schema="v2-rich"' \
  --project "$PROJECT_ID" \
  --freshness=1h \
  --limit 100 \
  --format="value(jsonPayload.agent_log)"
```

(`agent_log` er en kopi av den fulle tekstlinjen; bruk den hvis `message` ser kort ut i konsollen.)

**Med tidsstempel (lettere å se rekkefølge):**

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="tripletex-agent" AND jsonPayload.log_schema="v2-rich"' \
  --project "$PROJECT_ID" \
  --freshness=1h \
  --limit 100 \
  --format="table(timestamp,jsonPayload.agent_log)"
```

**Én konkret `/solve`-request (lim inn full `request_id` fra en logglinje):**

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="tripletex-agent" AND jsonPayload.request_id="PASTE-FULL-UUID-HERE"' \
  --project "$PROJECT_ID" \
  --freshness=24h \
  --limit 200 \
  --format="value(jsonPayload.message)"
```

**Full JSON per oppføring (hvis du vil se alle felter, ikke bare `message`):**

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="tripletex-agent" AND jsonPayload.message!=""' \
  --project "$PROJECT_ID" \
  --freshness=1h \
  --limit 20 \
  --format=json
```

**Alternativ (enkelt, men blander med Uvicorn):**

```bash
gcloud run services logs read tripletex-agent \
  --region europe-north1 \
  --project "$PROJECT_ID" \
  --limit 200
```

**«Nesten live» uten ekstra pakker (anbefalt i Cloud Shell):** kjør på nytt etter hver test:

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="tripletex-agent" AND jsonPayload.log_schema="v2-rich"' \
  --project "$PROJECT_ID" \
  --freshness=5m \
  --limit 30 \
  --format="value(jsonPayload.agent_log)"
```

**Live tail** (`gcloud beta run services logs tail …`) krever pakken `google-cloud-cli-log-streaming`.  
I **Cloud Shell** er ofte `gcloud components install` skrudd av — bruk da **apt** i stedet, deretter tail:

```bash
sudo apt-get update && sudo apt-get install -y google-cloud-cli-log-streaming

gcloud beta run services logs tail tripletex-agent \
  --region europe-north1 \
  --project "$PROJECT_ID"
```

Hvis du ikke vil installere noe: bruk kommandoen over med `--freshness=5m`, eller kjør `gcloud run services logs read` i en loop manuelt.

> **Merk:** Ser du fortsatt korte linjer som bare `api_call` uten `step=` / `path=` → du kjører **gammelt image**. Sjekk at logg har `jsonPayload.log_schema="v2-rich"` etter deploy. Prosess: `git pull` → `gcloud builds submit` → `gcloud run deploy`.

## Automatisert feedback (logger)

### `run_summary` per `/solve`

Etter hver agent-kjøring logges **`run_summary`** (JSON-felt) med bl.a. `outcome` (`agent_done` | `max_steps` | `deterministic`), `steps_used`, `api_error_count`, `had_max_steps`, `last_error_path`, `task_preview`. Filtrer i Logs Explorer:

```
jsonPayload.message=~"^run_summary"
```

### Script: sammendrag gruppert på `request_id`

Fra repo-roten (krever `gcloud` og `roles/logging.viewer`):

```bash
export PROJECT_ID=your-project-id
python3 tripletex/scripts/summarize_agent_logs.py \
  --project "$PROJECT_ID" \
  --freshness 24h \
  --limit 800 \
  --format md \
  -o /tmp/agent-report.md

# CSV for regneark:
python3 tripletex/scripts/summarize_agent_logs.py \
  --project "$PROJECT_ID" \
  --freshness 7d \
  --format csv \
  -o /tmp/agent-runs.csv
```

Valgfritt ekstra filter (AND):

```bash
python3 tripletex/scripts/summarize_agent_logs.py \
  --project "$PROJECT_ID" \
  --freshness 1h \
  --filter 'severity>=WARNING' \
  --format md
```

### Cloud Scheduler (valgfritt)

Opprett en **Cloud Run Job** eller bruk **Cloud Build** med trigger som kjører scriptet og laster rapport til **GCS** (`gsutil cp`) eller sender til **Slack** (webhook som Secret Manager). Service account for jobben trenger minst **`roles/logging.viewer`**.

### Log sink til BigQuery (trender)

1. I GCP Console: **Logging** → **Log router** → **Create sink**.
2. **Sink destination:** BigQuery dataset (f.eks. `tripletex_logs`), tabell la stå auto (partitioned).
3. **Inclusion filter** (samme som agent-filter):

```
resource.type="cloud_run_revision"
resource.labels.service_name="tripletex-agent"
jsonPayload.log_schema="v2-rich"
```

4. IAM: sinkens service account får **BigQuery Data Editor** på datasettet.

**Eksempel-SQL** (juster tabellnavn etter faktisk `_AllLogs` / partitioned table navn):

```sql
-- api_error per path siste 7 døgn
SELECT
  jsonPayload.path AS path,
  COUNT(*) AS errors
FROM `PROJECT_ID.tripletex_logs._AllLogs`
WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
  AND jsonPayload.message LIKE 'api_error%'
GROUP BY 1
ORDER BY errors DESC
LIMIT 50;
```

```sql
-- run_summary outcomes
SELECT
  jsonPayload.outcome AS outcome,
  COUNT(*) AS n,
  AVG(SAFE_CAST(jsonPayload.api_error_count AS FLOAT64)) AS avg_api_errors
FROM `PROJECT_ID.tripletex_logs._AllLogs`
WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
  AND jsonPayload.message LIKE 'run_summary%'
GROUP BY 1;
```

(BQ-skjema varierer litt med «native» vs «bucket» sink — inspiser kolonner i konsollen.)

### Log-based metrics og varsling

1. **Monitoring** → **Logging** → **Log-based metrics** → **Create metric**.
2. **Filter:** f.eks. `jsonPayload.message=~"hard_stop|max_steps_reached"` og `resource.labels.service_name="tripletex-agent"`.
3. **Metric type:** Counter.
4. **Alerting** → opprett policy på metrikken (f.eks. > N per time etter deploy).

Alternativt egen metrikk for `jsonPayload.message=~"^run_summary"` med `jsonPayload.had_max_steps="true"` (streng/bool avhenger av serialisering — test i Logs Explorer først).

### IAM (kort)

| Rolle | Bruk |
|------|------|
| `roles/logging.viewer` | `gcloud logging read`, summarize-script, Scheduler/Job |
| `roles/logging.configWriter` | Opprette log sinks |
| `roles/bigquery.dataEditor` | Sink skriver til BQ dataset |

## Deploy (GCP)

Se [gcp/setup.md](../gcp/setup.md).
