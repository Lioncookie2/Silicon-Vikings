#!/usr/bin/env python3
"""
Summarize Tripletex agent Cloud Logging entries grouped by request_id.

Uses `gcloud logging read` (JSON output) — run from Cloud Shell or any machine
with gcloud authenticated and roles/logging.viewer.

Example:
  export PROJECT_ID=ai-nm26osl-1867
  python3 tripletex/scripts/summarize_agent_logs.py --project "$PROJECT_ID" --freshness 24h --format md

See tripletex/PLAN.md section «Automatisert feedback».
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from collections import Counter, defaultdict
from typing import Any


DEFAULT_FILTER = (
    'resource.type="cloud_run_revision" '
    'AND resource.labels.service_name="tripletex-agent" '
    'AND jsonPayload.log_schema="v2-rich"'
)

# Primary event name: first segment of jsonPayload.message / agent_log (before " | ")
TRACKED_PREFIXES = (
    "api_error",
    "hard_stop",
    "path_fail_streak",
    "max_steps_reached",
    "json_parse_error",
    "solve_completed",
    "agent_done",
    "agent_finished",
    "solve_start",
    "solve_exception",
    "run_summary",
    "deterministic_handler",
)


def _event_from_payload(payload: dict[str, Any]) -> str:
    msg = payload.get("agent_log") or payload.get("message") or ""
    if isinstance(msg, str) and " | " in msg:
        return msg.split(" | ", 1)[0].strip()
    if isinstance(msg, str):
        return msg.strip()[:80]
    return "unknown"


def _rid(payload: dict[str, Any]) -> str | None:
    rid = payload.get("request_id")
    return rid if isinstance(rid, str) and rid else None


def fetch_logs(
    project: str,
    freshness: str,
    limit: int,
    extra_filter: str | None,
) -> list[dict[str, Any]]:
    filt = DEFAULT_FILTER
    if extra_filter:
        filt = f"({filt}) AND ({extra_filter})"
    cmd = [
        "gcloud",
        "logging",
        "read",
        filt,
        f"--project={project}",
        f"--freshness={freshness}",
        f"--limit={limit}",
        "--format=json",
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        print("ERROR: gcloud not found. Install Google Cloud SDK.", file=sys.stderr)
        sys.exit(127)
    except subprocess.CalledProcessError as e:
        print(e.stderr or str(e), file=sys.stderr)
        sys.exit(e.returncode)
    if not out.strip():
        return []
    data = json.loads(out)
    if not isinstance(data, list):
        return []
    return data


def parse_entries(entries: list[dict[str, Any]]) -> tuple[dict[str, dict], Counter]:
    """Return per-request_id aggregates and global path counter for api_error."""
    by_rid: dict[str, dict[str, Any]] = {}
    path_errors: Counter = Counter()

    for ent in entries:
        payload = ent.get("jsonPayload")
        if not isinstance(payload, dict):
            continue
        rid = _rid(payload)
        if not rid:
            continue
        ev = _event_from_payload(payload)
        ts = ent.get("timestamp") or ""

        if rid not in by_rid:
            by_rid[rid] = {
                "request_id": rid,
                "events": Counter(),
                "task_preview": "",
                "steps_used": None,
                "timestamps": [],
                "run_summary": None,
            }
        g = by_rid[rid]
        g["events"][ev] += 1
        g["timestamps"].append(ts)

        if ev == "agent_task" and payload.get("task_preview"):
            g["task_preview"] = str(payload["task_preview"])[:200]
        if ev == "agent_finished" and payload.get("steps_used") is not None:
            g["steps_used"] = payload["steps_used"]
        if ev == "run_summary":
            g["run_summary"] = {k: v for k, v in payload.items() if k not in ("message", "agent_log", "timestamp", "severity", "log_schema")}

        if ev == "api_error":
            p = payload.get("path")
            if isinstance(p, str) and p:
                path_errors[f"{p}"] += 1

    return by_rid, path_errors


def write_markdown(by_rid: dict[str, dict], path_errors: Counter, out: io.TextIOBase) -> None:
    out.write("# Tripletex agent log summary\n\n")
    out.write(f"Requests: **{len(by_rid)}**\n\n")

    out.write("## API errors by path (all requests)\n\n")
    if path_errors:
        out.write("| Path | Count |\n|------|-------|\n")
        for path, c in path_errors.most_common(30):
            out.write(f"| `{path}` | {c} |\n")
    else:
        out.write("_No api_error lines in sample._\n")
    out.write("\n")

    out.write("## Per request\n\n")
    for rid in sorted(by_rid.keys(), key=lambda r: max(by_rid[r]["timestamps"] or [""]), reverse=True):
        g = by_rid[rid]
        out.write(f"### `{rid[:8]}…`\n\n")
        if g["task_preview"]:
            out.write(f"- **Task:** {g['task_preview']}\n")
        if g["steps_used"] is not None:
            out.write(f"- **steps_used:** {g['steps_used']}\n")
        if g["run_summary"]:
            out.write(f"- **run_summary:** `{json.dumps(g['run_summary'], ensure_ascii=False)[:500]}`\n")
        out.write("- **Events:**\n")
        for ev, c in g["events"].most_common():
            if ev in TRACKED_PREFIXES or c > 1 or ev not in ("llm_call", "DEBUG"):
                out.write(f"  - `{ev}`: {c}\n")
        out.write("\n")


def write_csv(by_rid: dict[str, dict], out: io.TextIOBase) -> None:
    w = csv.writer(out)
    w.writerow(
        [
            "request_id",
            "task_preview",
            "steps_used",
            "api_error",
            "hard_stop",
            "max_steps_reached",
            "json_parse_error",
            "agent_done",
            "solve_completed",
            "run_outcome",
        ]
    )
    for rid, g in sorted(by_rid.items(), key=lambda x: x[0]):
        ev = g["events"]
        rs = g.get("run_summary") or {}
        outcome = rs.get("outcome", "")
        w.writerow(
            [
                rid,
                (g["task_preview"] or "")[:300],
                g["steps_used"] if g["steps_used"] is not None else "",
                ev.get("api_error", 0),
                ev.get("hard_stop", 0),
                ev.get("max_steps_reached", 0),
                ev.get("json_parse_error", 0),
                ev.get("agent_done", 0),
                ev.get("solve_completed", 0),
                outcome,
            ]
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize tripletex-agent Cloud Logging by request_id.")
    ap.add_argument("--project", required=True, help="GCP project ID")
    ap.add_argument("--freshness", default="24h", help="e.g. 1h, 24h, 7d")
    ap.add_argument("--limit", type=int, default=500, help="Max log entries to fetch")
    ap.add_argument("--filter", default="", help="Additional Cloud Logging filter AND-clause")
    ap.add_argument("--format", choices=("md", "csv"), default="md")
    ap.add_argument("-o", "--output", default="-", help="Output file (default stdout)")
    args = ap.parse_args()

    entries = fetch_logs(args.project, args.freshness, args.limit, args.filter or None)
    by_rid, path_errors = parse_entries(entries)

    if args.output == "-":
        stream: io.TextIOBase = sys.stdout
        close = False
    else:
        stream = open(args.output, "w", encoding="utf-8")
        close = True
    try:
        if args.format == "md":
            write_markdown(by_rid, path_errors, stream)
        else:
            write_csv(by_rid, stream)
    finally:
        if close:
            stream.close()

    if not by_rid:
        print("No matching log entries (check filter, freshness, limit).", file=sys.stderr)


if __name__ == "__main__":
    main()
