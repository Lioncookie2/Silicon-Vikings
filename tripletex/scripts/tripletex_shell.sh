#!/usr/bin/env bash
# Tripletex — helper functions for Cloud Shell / terminal.
#
# Bootstrap (Cloud Shell Editor, from home or workspace):
#   git clone <your-fork-or-repo-url> Silicon-Vikings && cd Silicon-Vikings
#   gcloud config set project YOUR_PROJECT_ID
#   export PROJECT_ID=YOUR_PROJECT_ID
#   source tripletex/scripts/tripletex_shell.sh
#   ttx_help
#
# After a graded run on app.ainm.no, copy request_id from logs (or use ttx_logs_recent),
# then: ttx_logs_rid <uuid>  and paste the output when debugging with your team/AI.
#
# shellcheck disable=SC2034  # TTX_* used after source by user

_TTX_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_TTX_REPO_ROOT="$(cd "$_TTX_SCRIPT_DIR/../.." && pwd)"

TTX_PROJECT_ID="${TTX_PROJECT_ID:-${PROJECT_ID:-}}"
TTX_REGION="${TTX_REGION:-europe-north1}"
TTX_SERVICE="${TTX_SERVICE:-tripletex-agent}"

# Log filter base (must match tripletex/scripts/summarize_agent_logs.py DEFAULT_FILTER pieces)
TTX_LOG_FILTER_BASE="resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${TTX_SERVICE}\" AND jsonPayload.log_schema=\"v2-rich\""

ttx_help() {
  cat <<'EOF'
Tripletex shell helpers (set PROJECT_ID or TTX_PROJECT_ID first):

  ttx_bootstrap                 — print clone/source one-liner reminder
  ttx_logs_recent [freshness]   — last agent lines (default freshness 1h)
  ttx_logs_warnings [freshness] — severity>=WARNING (default 6h)
  ttx_logs_api_errors [fr]    — lines whose message starts with api_error (default 24h)
  ttx_logs_rid <uuid>         — all lines for one request_id (48h window)
  ttx_summarize [freshness]     — markdown report -> /tmp/ttx-agent-report.md
  ttx_summarize_csv [freshness] — CSV -> /tmp/ttx-agent-runs.csv
  ttx_report                    — open last markdown report with less
  ttx_run_logs                  — gcloud run services logs read (quick, mixes Uvicorn)

Deploy (from repo root): see gcp/setup.md and tripletex/cloudbuild.yaml
EOF
}

ttx_bootstrap() {
  cat <<'EOF'
# Paste and adjust YOUR_PROJECT_ID / git URL:
#   git clone <REPO_URL> Silicon-Vikings && cd Silicon-Vikings
#   gcloud config set project YOUR_PROJECT_ID
#   export PROJECT_ID=YOUR_PROJECT_ID
#   source tripletex/scripts/tripletex_shell.sh
#   ttx_logs_recent 2h
EOF
}

ttx__require_project() {
  if [[ -z "${TTX_PROJECT_ID}" ]]; then
    echo "Set PROJECT_ID or TTX_PROJECT_ID first." >&2
    return 1
  fi
}

ttx_logs_recent() {
  ttx__require_project || return 1
  local fr="${1:-1h}"
  gcloud logging read \
    "${TTX_LOG_FILTER_BASE}" \
    --project "${TTX_PROJECT_ID}" \
    --freshness="${fr}" \
    --limit=60 \
    --format="table(timestamp,severity,jsonPayload.request_id,jsonPayload.agent_log)"
}

ttx_logs_warnings() {
  ttx__require_project || return 1
  local fr="${1:-6h}"
  gcloud logging read \
    "(${TTX_LOG_FILTER_BASE}) AND severity>=WARNING" \
    --project "${TTX_PROJECT_ID}" \
    --freshness="${fr}" \
    --limit=80 \
    --format="table(timestamp,severity,jsonPayload.request_id,jsonPayload.agent_log)"
}

ttx_logs_api_errors() {
  ttx__require_project || return 1
  local fr="${1:-24h}"
  gcloud logging read \
    "(${TTX_LOG_FILTER_BASE}) AND jsonPayload.message=~\"^api_error\"" \
    --project "${TTX_PROJECT_ID}" \
    --freshness="${fr}" \
    --limit=100 \
    --format="table(timestamp,jsonPayload.request_id,jsonPayload.path,jsonPayload.agent_log)"
}

ttx_logs_rid() {
  ttx__require_project || return 1
  local rid="${1:?usage: ttx_logs_rid <request_uuid>}"
  gcloud logging read \
    "(${TTX_LOG_FILTER_BASE}) AND jsonPayload.request_id=\"${rid}\"" \
    --project "${TTX_PROJECT_ID}" \
    --freshness=48h \
    --limit=400 \
    --format="value(jsonPayload.agent_log)"
}

ttx_summarize() {
  ttx__require_project || return 1
  local fr="${1:-24h}"
  if [[ ! -f "${_TTX_REPO_ROOT}/tripletex/scripts/summarize_agent_logs.py" ]]; then
    echo "Repo root not found (expected ${_TTX_REPO_ROOT}). cd to Silicon-Vikings and re-source this script." >&2
    return 1
  fi
  (cd "${_TTX_REPO_ROOT}" && python3 tripletex/scripts/summarize_agent_logs.py \
    --project "${TTX_PROJECT_ID}" \
    --freshness "${fr}" \
    --limit 800 \
    --format md \
    -o /tmp/ttx-agent-report.md)
  echo "Wrote /tmp/ttx-agent-report.md — run: ttx_report   (or: less /tmp/ttx-agent-report.md)"
}

ttx_summarize_csv() {
  ttx__require_project || return 1
  local fr="${1:-7d}"
  if [[ ! -f "${_TTX_REPO_ROOT}/tripletex/scripts/summarize_agent_logs.py" ]]; then
    echo "Repo root not found (expected ${_TTX_REPO_ROOT}). cd to Silicon-Vikings and re-source this script." >&2
    return 1
  fi
  (cd "${_TTX_REPO_ROOT}" && python3 tripletex/scripts/summarize_agent_logs.py \
    --project "${TTX_PROJECT_ID}" \
    --freshness "${fr}" \
    --limit 2000 \
    --format csv \
    -o /tmp/ttx-agent-runs.csv)
  echo "Wrote /tmp/ttx-agent-runs.csv"
}

ttx_report() {
  if [[ ! -f /tmp/ttx-agent-report.md ]]; then
    echo "No /tmp/ttx-agent-report.md — run ttx_summarize first." >&2
    return 1
  fi
  less /tmp/ttx-agent-report.md
}

ttx_run_logs() {
  ttx__require_project || return 1
  gcloud run services logs read "${TTX_SERVICE}" \
    --region "${TTX_REGION}" \
    --project "${TTX_PROJECT_ID}" \
    --limit 120
}
