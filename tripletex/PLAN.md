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

## Cloud Shell / terminal — arbeidsflyt

Når du lever i **Cloud Shell** etter innsending på app.ainm.no: klon repo, sett prosjekt, `source` hjelpefilen, hent logger og rapporter.

```bash
cd ~
git clone <repo-url> Silicon-Vikings && cd Silicon-Vikings
gcloud config set project YOUR_PROJECT_ID
export PROJECT_ID=YOUR_PROJECT_ID
source tripletex/scripts/tripletex_shell.sh
ttx_help
```

Typisk etter en kjøring med lav score:

1. `ttx_logs_recent 2h` — finn `request_id` i tabellen.
2. `ttx_logs_rid <full-uuid>` — lim utdrag inn i feilsøking (eller til AI).
3. `ttx_summarize 24h` → `ttx_report` — oversikt med **API errors by path** og `run_summary` per request.

Valgfritt: `ttx_logs_api_errors 24h`, `ttx_logs_warnings 6h`, `ttx_summarize_csv 7d` (CSV i `/tmp/ttx-agent-runs.csv`).

Hjelpefunksjonene ligger i [tripletex/scripts/tripletex_shell.sh](scripts/tripletex_shell.sh). Service-navn og region kan overstyres: `TTX_SERVICE`, `TTX_REGION`.

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

### Kjør-artefakter (GCS JSONL)

Hver `/solve` kan lagre en **kopi av alle strukturerte logglinjer** (samme JSON som stdout) til **Google Cloud Storage** for analyse uten å parse Cloud Logging.

**Miljøvariabler (Cloud Run → Edit & deploy new revision → Variables):**

| Variabel | Beskrivelse |
|----------|-------------|
| `TRIPLETEX_RUN_LOG_GCS_BUCKET` | Bucket-navn (påkrevd for å aktivere). Tom = ingen opplasting. |
| `TRIPLETEX_RUN_LOG_GCS_PREFIX` | Valgfri mappe-prefix, f.eks. `tripletex-runs` (ingen ledende `/`). |

**Objekter per kjøring** (`prefix`/`YYYY-MM-DD`/`{request_id}.jsonl` og `…_meta.json` med `outcome`, `task_preview`, m.m.):

**IAM:** Cloud Run **service account** (eller den identiteten tjenesten kjører som) trenger f.eks. **`roles/storage.objectCreator`** på bucketen (eller smalere custom role med `storage.objects.create`). Ingen nøkler i repo — bruk Workload Identity.

**Liste filer:**

```bash
gsutil ls "gs://MY-BUCKET/tripletex-runs/$(date -u +%Y-%m-%d)/"
```

Opplasting feiler aldri HTTP-responsen; feil skrives til stderr.

**Agent-justeringer (miljø):** `TRIPLETEX_HARD_STOP_PATH_STREAK` (standard `8`) — antall påfølgende feil på *samme* `method+path` før hard stop tvinger `done` (identiske retries tvinger fortsatt `done` etter 5 gjentakelser).

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
| `roles/storage.objectCreator` | Agent laster opp kjør-jsonl til GCS-bucket |

## Deploy (GCP)

Se [gcp/setup.md](../gcp/setup.md).

## Daglig playbook (maksimere Tripletex-score)

Mål: **riktig sluttstate i sandkassen** og **færre unødvendige skrivekall** (POST/PUT/DELETE). Bruk én fast runde per dag du konkurrerer.

### 1) Start (10–15 min)

1. `cd ~/Silicon-Vikings && git pull`
2. Bekreft at du er på siste `main` (sjekk at teamets siste fiks er med — f.eks. `POST /employee` / userType).
3. Bygg og deploy Cloud Run etter [gcp/setup.md](../gcp/setup.md) (`gcloud builds submit` → `gcloud run deploy`).
4. Test: `curl -sS "$SERVICE_URL/health"` → `{"status":"ok"}`.
5. Lim inn **HTTPS-URL** på app.ainm.no (samme URL hver gang med mindre dere bytter tjeneste).

### 2) Cloud Shell — klargjør logger (2 min)

```bash
cd ~/Silicon-Vikings
export PROJECT_ID=<ditt-prosjekt>
gcloud config set project "$PROJECT_ID"
source tripletex/scripts/tripletex_shell.sh
```

### 3) For hver oppgave du sender inn (etter score)

1. `ttx_logs_recent 1h` → noter **`request_id`** for siste kjøring.
2. `ttx_logs_rid <uuid>` → lim inn i notat / del med team om dere feilsøker sammen.
3. Sjekk **`run_summary`**: `outcome` (`agent_done` vs `max_steps`), `api_error_count`, `last_error_path`, `had_max_steps`.
4. Klassifiser i én setning, f.eks.: «userType 422», «voucher PUT fortegn», «404 feil bilags-id», «bankkontonummer», «hard_stop».

### 4) Prioriter neste kodefiks (høyest effekt først)

| Signal | Typisk tiltak |
|--------|----------------|
| Samme `path` + 422 ofte | Utvid `SYSTEM_PROMPT` med én presis regel **eller** deterministisk handler i `task_handlers/` |
| `max_steps` / `hard_stop` | Kortere vei i prompten; færre «prøv igjen»-løkker; evt. øk `MAX_STEPS` bare hvis dere har tokens/steg-budsjett |
| `404` på `/ledger/voucher/{id}` | Prompt/hint: id **kun** fra `GET /ledger/voucher` i periode (allerede i agent) — vurder sterkere tvang i hint |
| `agent_done` men lav score | Ofte «falsk ferdig» — se siste `reasoning` og siste vellykkede API-kall; mangler steg (lønn, bilag, …) |
| Mange GET på samme path | OK for poeng; fokuser på å kutte unødvendige POST/PUT |

Etter fiks: **commit → push → deploy** før neste innsending.

### 5) Slutt på dagen (10 min)

```bash
ttx_summarize 24h
ttx_report
```

eller `ttx_summarize_csv 7d` → se hvilke **paths** som topper `api_error`. Noter 1–3 mønstre dere skal ta neste økt.

### 6) Husk

- Tilfeldige oppgaver → dere optimaliserer **porteføljen** (mange typer), ikke én oppgavetekst.
- Logger **øker ikke** score alene; de **peker** på hva som skal endres i `agent.py` / `task_handlers/`.
- Etter deploy: logger skal ha `jsonPayload.log_schema="v2-rich"` (se avsnitt om strukturert logging over).
