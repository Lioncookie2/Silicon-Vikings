"""
Structured JSON logging for Cloud Run / Cloud Logging.

Each call to log_event() writes exactly one JSON line to stdout.
Cloud Logging automatically parses lines that are valid JSON and
promotes the `severity` field to the log severity level.

Usage:
    from .structured_log import log_event, set_request_id, get_request_id

    set_request_id("abc-123")           # once per HTTP request
    log_event("INFO", "api_call", step=0, method="GET", path="/customer", status=200)

Reference: https://cloud.google.com/logging/docs/structured-logging
"""
from __future__ import annotations

import json
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

# Per-request context variable — set once in main.py, read everywhere
_request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

# Fields that must never appear in log output (credentials, tokens)
_SENSITIVE_KEYS = frozenset(
    {
        "session_token", "password", "token", "secret",
        "credential", "credentials", "tripletex_credentials",
        "authorization", "api_key", "apikey",
    }
)

MAX_FIELD_BYTES = 3000  # truncate any single string field beyond this size


def set_request_id(request_id: str) -> None:
    """Set the current request ID for the duration of this async context."""
    _request_id_var.set(request_id)


def get_request_id() -> str | None:
    """Return the request ID set for the current context, or None."""
    return _request_id_var.get()


def _sanitize(value: Any) -> Any:
    """Recursively remove sensitive keys and truncate long strings."""
    if isinstance(value, dict):
        return {
            k: _sanitize(v)
            for k, v in value.items()
            if k.lower() not in _SENSITIVE_KEYS
        }
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, str) and len(value) > MAX_FIELD_BYTES:
        return value[:MAX_FIELD_BYTES] + "…[truncated]"
    return value


def log_event(
    severity: str,
    message: str,
    **fields: Any,
) -> None:
    """
    Write one structured JSON log line to stdout.

    Args:
        severity: Cloud Logging severity string — INFO, WARNING, ERROR, DEBUG.
        message:  Short human-readable description (also used as the log message).
        **fields: Arbitrary key-value pairs included in jsonPayload.
                  Sensitive keys are stripped automatically.
    """
    rid = get_request_id()
    payload: dict[str, Any] = {
        "severity": severity.upper(),
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if rid is not None:
        payload["request_id"] = rid

    for k, v in fields.items():
        if k.lower() not in _SENSITIVE_KEYS:
            payload[k] = _sanitize(v)

    print(json.dumps(payload, default=str, ensure_ascii=False), flush=True)


def log_api_error(step: int, method: str, path: str, status: int, body: Any) -> None:
    """Convenience wrapper: log a 4xx/5xx API error with extracted detail."""
    detail = _extract_error_detail(body)
    log_event(
        "WARNING",
        "api_error",
        step=step,
        method=method,
        path=path,
        status=status,
        detail=detail,
    )


def _extract_error_detail(body: Any) -> str:
    """Pull the most useful info from a Tripletex error response body."""
    if isinstance(body, dict):
        parts: list[str] = []
        if "message" in body:
            parts.append(str(body["message"])[:400])
        if "validationMessages" in body:
            msgs = body["validationMessages"]
            if isinstance(msgs, list):
                for vm in msgs[:5]:
                    if isinstance(vm, dict):
                        parts.append(f"  • {vm.get('field','?')}: {vm.get('message','?')}"[:200])
                    else:
                        parts.append(f"  • {str(vm)[:200]}")
        return "\n".join(parts) if parts else str(body)[:400]
    return str(body)[:400]
