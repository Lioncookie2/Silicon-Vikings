#!/usr/bin/env bash
# Package NorgesGruppen submission zip (run from repo root).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SUB="$ROOT/norgesgruppen/submission"
if [[ ! -f "$SUB/run.py" ]]; then
  echo "Missing $SUB/run.py"
  exit 1
fi
if [[ ! -f "$SUB/best.pt" ]]; then
  echo "Missing $SUB/best.pt — after training run:"
  echo "  cp runs/ngd_yolo/baseline/weights/best.pt $SUB/best.pt"
  exit 1
fi
OUT="${1:-$ROOT/submission_ng.zip}"
cd "$SUB"
zip -r "$OUT" run.py best.pt -x "*.json" -x "test_*"
echo "Created $OUT"
