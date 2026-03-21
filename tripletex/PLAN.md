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

**Live tail (krever `beta`-sporet — installer med `gcloud components install beta` hvis den mangler):**

```bash
gcloud beta run services logs tail tripletex-agent \
  --region europe-north1 \
  --project "$PROJECT_ID"
```

> **Merk:** Ser du fortsatt korte linjer som bare `api_call` uten `step=` / `path=` → du kjører **gammelt image**. Sjekk at logg har `jsonPayload.log_schema="v2-rich"` etter deploy. Prosess: `git pull` → `gcloud builds submit` → `gcloud run deploy`.

Se [TASK_COVERAGE.md](TASK_COVERAGE.md) for sjekkliste over støttede oppgavetyper.

## Deploy (GCP)

Se [gcp/setup.md](../gcp/setup.md).
