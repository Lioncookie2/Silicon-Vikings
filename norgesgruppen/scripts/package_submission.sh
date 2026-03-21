#!/usr/bin/env bash
# Pakk NorgesGruppen-innlevering (kjør fra hvor som helst).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec python3 "$ROOT/norgesgruppen/scripts/package_submission.py" -o "${1:-$ROOT/submission.zip}"
