#!/bin/bash
# Pass på å stoppe scriptet hvis en kommando feiler
set -e

PROJECT_ID="ai-nm26osl-1867"
PROJECT_NUMBER="806169416908"
REGION="europe-west1"
REPO="tripletex-repo"
IMAGE_NAME="solver"
SERVICE_NAME="tripletex-solver"
SECRET_NAME="llm-api-key"

IMAGE_URL="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:latest"

echo "🔐 Gir Cloud Run tilgang til API-nøkkelen..."
gcloud secrets add-iam-policy-binding $SECRET_NAME \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor" \
    --project=$PROJECT_ID || echo "Klarte ikke å sette IAM-policy. Kanskje den allerede er satt, eller du mangler rettigheter. Fortsetter uansett..."

echo "🚀 Bygger Docker-image med Cloud Build..."
gcloud builds submit --tag $IMAGE_URL .

echo "🚢 Deployer til Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_URL \
  --region $REGION \
  --allow-unauthenticated \
  --set-env-vars="GOOGLE_API_KEY=secrets_kommer_via_env" \
  --set-secrets="LLM_API_KEY=${SECRET_NAME}:latest" \
  --timeout=300 \
  --concurrency=10 \
  --memory=1024Mi

echo "✅ Ferdig! Gå til Cloud Run i Google-konsollen din for å finne Endpoint URL-en (som slutter på .run.app). Husk å legge til /solve på slutten!"
