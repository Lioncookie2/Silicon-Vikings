#!/bin/bash
# Log observer script for Tripletex Agent in Google Cloud Run
# This script tails logs from Cloud Run and saves a copy locally for debugging.

# Define variables
PROJECT="ai-nm26osl-1867"
REGION="europe-north1"
SERVICE="tripletex-agent"
LOG_DIR="$(dirname "$0")/logs"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Generate a timestamped filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/run_${TIMESTAMP}.log"

echo "============================================================"
echo "🎯 Starting Live Logging for Tripletex Agent"
echo "📂 Saving logs to: $LOG_FILE"
echo "🛑 Press Ctrl+C to stop logging"
echo "============================================================"

# Tail the logs and save a copy locally
gcloud beta run services logs tail "$SERVICE" \
  --project "$PROJECT" \
  --region "$REGION" | tee "$LOG_FILE"
