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

```
# Alle hendelser for én request:
jsonPayload.request_id="<uuid>"

# Alle feil og advarsler:
severity>=WARNING

# Alle API-feil:
jsonPayload.message="api_error"

# Spesifikk path:
jsonPayload.path="/invoice"
```

Se [TASK_COVERAGE.md](TASK_COVERAGE.md) for sjekkliste over støttede oppgavetyper.

## Deploy (GCP)

Se [gcp/setup.md](../gcp/setup.md).
