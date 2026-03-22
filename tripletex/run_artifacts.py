"""
Optional per-request JSONL artifacts uploaded to GCS after each /solve.

Enable with:
  TRIPLETEX_RUN_LOG_GCS_BUCKET=my-bucket
  TRIPLETEX_RUN_LOG_GCS_PREFIX=tripletex-runs   # optional, no leading/trailing slashes

Cloud Run: grant the service account storage.objectCreator on the bucket.
"""
from __future__ import annotations

import json
import os
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

# Max size for any string value stored in the artifact (per field)
_ARTIFACT_STRING_CAP = 12_000

_buffer: ContextVar[list[dict[str, Any]] | None] = ContextVar("run_artifact_buffer", default=None)


def gcs_run_logging_enabled() -> bool:
    return bool(os.environ.get("TRIPLETEX_RUN_LOG_GCS_BUCKET", "").strip())


def begin_run_artifact_recording() -> None:
    """Start buffering structured log payloads for the current request (no-op if GCS disabled)."""
    if not gcs_run_logging_enabled():
        return
    _buffer.set([])


def record_structured_payload(payload: dict[str, Any]) -> None:
    """Append one sanitized log line copy (same shape as stdout JSON) if recording is active."""
    buf = _buffer.get()
    if buf is None:
        return
    buf.append(_compact_for_artifact(dict(payload)))


def flush_run_artifacts(request_id: str) -> None:
    """
    Upload buffered lines to GCS as JSONL + a small meta JSON file.
    Clears the buffer. Safe to call when recording was never started or GCS disabled.
    """
    buf = _buffer.get()
    _buffer.set(None)
    if not gcs_run_logging_enabled() or not buf:
        return

    bucket_name = os.environ["TRIPLETEX_RUN_LOG_GCS_BUCKET"].strip()
    prefix = os.environ.get("TRIPLETEX_RUN_LOG_GCS_PREFIX", "tripletex-runs").strip().strip("/")
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    base = f"{prefix}/{day}/{request_id}" if prefix else f"{day}/{request_id}"

    jsonl = "\n".join(json.dumps(rec, default=str, ensure_ascii=False) for rec in buf) + "\n"

    meta: dict[str, Any] = {
        "request_id": request_id,
        "record_count": len(buf),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    for rec in reversed(buf):
        if isinstance(rec.get("outcome"), str) and "steps_used" in rec:
            for key in (
                "outcome", "steps_used", "api_error_count", "task_preview",
                "had_max_steps", "last_error_path", "last_error_status",
            ):
                if key in rec:
                    meta[key] = rec[key]
            break

    try:
        from google.cloud import storage  # type: ignore[import-untyped]

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        bucket.blob(f"{base}.jsonl").upload_from_string(
            jsonl, content_type="application/x-ndjson; charset=utf-8"
        )
        bucket.blob(f"{base}_meta.json").upload_from_string(
            json.dumps(meta, default=str, ensure_ascii=False, indent=2),
            content_type="application/json; charset=utf-8",
        )
    except Exception as exc:
        # Never fail the request because artifact upload failed
        print(f"[tripletex] run_artifacts: GCS upload failed: {exc}", flush=True)


def _compact_for_artifact(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _compact_for_artifact(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_compact_for_artifact(v) for v in obj]
    if isinstance(obj, str) and len(obj) > _ARTIFACT_STRING_CAP:
        return obj[:_ARTIFACT_STRING_CAP] + "…[truncated]"
    return obj
