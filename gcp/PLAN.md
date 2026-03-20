# Google Cloud — NM i AI

## Hva vi bruker GCP til

| Tjeneste | Formål |
|----------|--------|
| **Cloud Run** | Offentlig HTTPS for Tripletex `POST /solve` |
| **Artifact Registry** | Docker-image for Tripletex-agenten |
| **Cloud Build** | Bygge og pushe image (valgfritt CI) |
| **Vertex AI / Gemini** | LLM-kall kan kjøres fra Cloud Run med workload identity eller API-nøkkel i Secret Manager |

NorgesGruppen (ZIP-innsending) og Astar Island (API fra egen maskin) **krever ikke** GCP.

## Manuelt steg

Søk om GCP Lab-konto på [app.ainm.no](https://app.ainm.no) når hele laget er Vipps-verifisert — se [setup.md](setup.md).
