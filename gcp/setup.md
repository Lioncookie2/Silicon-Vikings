# GCP-oppsett for Tripletex-agent

## 1. Søk om konto

- Gå til lagets side på [app.ainm.no](https://app.ainm.no)
- Alle medlemmer må være **Vipps-verifisert**
- Send søknad om GCP Lab (`@gcplab.me`) — begrenset antall plasser

## 2. Lokalt: `gcloud` CLI

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

## 3. Aktiver API-er

- Cloud Run, Artifact Registry, Cloud Build (og Vertex AI hvis dere bruker det)

## 4. Bygg og deploy Tripletex (manuelt)

Fra **repo-roten** `Silicon-Vikings/`:

```bash
export REGION=europe-north1
# Bygger image med tag :latest og :$BUILD_ID (SHORT_SHA brukes ikke ved manuell submit).
gcloud builds submit --config tripletex/cloudbuild.yaml .
```

Tilpass `tripletex/cloudbuild.yaml` (`_REGION`, `_REPO`, `_IMAGE`) ved behov.

Deretter Cloud Run (eksempel):

```bash
gcloud run deploy tripletex-agent \
  --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/nmiai/tripletex-agent:latest \
  --region $REGION \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=... 
```

**Anbefaling:** legg API-nøkkel i **Secret Manager** og referer med `--set-secrets` i stedet for klartekst.

## 5. Innsending

Lim inn Cloud Run HTTPS-URL i Tripletex-submit på app.ainm.no.

## Utvikling uten GCP

```bash
cd Silicon-Vikings
export GEMINI_API_KEY=...
export PYTHONPATH=.
uvicorn tripletex.main:app --host 0.0.0.0 --port 8000
npx cloudflared tunnel --url http://localhost:8000
```
