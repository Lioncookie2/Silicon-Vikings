"""Deterministic activity creation + body validation for agent preflight."""
from __future__ import annotations

import re
from typing import Any

from ..structured_log import log_event
from ..tripletex_client import TripletexClient


def validate_activity_body(body: dict[str, Any] | None) -> str | None:
    """
    Return human-readable validation error, or None if OK for POST /activity.
    """
    if body is None:
        return "POST /activity requires a JSON body with name and activityType:{id}."
    if not isinstance(body, dict):
        return "POST /activity body must be a JSON object."
    if "project" in body:
        return "POST /activity must not include a 'project' field; activities are global."
    at = body.get("activityType")
    if at is None:
        return "POST /activity requires activityType:{id:X}. Call GET /activity/type first."
    if isinstance(at, dict):
        if at.get("id") is None:
            return "activityType must be activityType:{id:number} from GET /activity/type."
    elif isinstance(at, int):
        pass
    else:
        return "activityType must be an object {id:N} or use id from GET /activity/type."
    return None


def _pick_activity_type_id(client: TripletexClient, name_hint: str | None) -> int | None:
    r = client.get(
        "/activity/type",
        params={"fields": "id,name", "count": 50},
    )
    if r.status_code != 200:
        return None
    values = r.json().get("values") or []
    if not values:
        return None
    hint_l = (name_hint or "").strip().lower()
    if hint_l:
        for row in values:
            if not isinstance(row, dict):
                continue
            n = row.get("name")
            if isinstance(n, str) and hint_l in n.lower():
                try:
                    return int(row["id"])
                except (TypeError, ValueError, KeyError):
                    continue
    try:
        return int(values[0]["id"])
    except (TypeError, ValueError, KeyError, IndexError):
        return None


def _parse_activity_name(prompt: str) -> tuple[str | None, str | None, bool]:
    """Returns (name, number, is_general)."""
    m = re.search(
        r"(?:aktivitet|activity)\s*[\"«]([^\"»]+)[\"»]",
        prompt,
        re.IGNORECASE,
    )
    name = m.group(1).strip() if m else None
    if not name:
        m2 = re.search(
            r"(?:named|navn|name)\s*[:.]?\s*[\"«]([^\"»]+)[\"»]",
            prompt,
            re.IGNORECASE,
        )
        name = m2.group(1).strip() if m2 else None
    num_m = re.search(r"(?:number|nr\.?|nummer)\s*[:.]?\s*([A-Za-z0-9._-]+)", prompt, re.IGNORECASE)
    number = num_m.group(1).strip() if num_m else None
    is_general = bool(re.search(r"\bgeneral\b|\bgenerell\b|\bisGeneral\b", prompt, re.IGNORECASE))
    return name, number, is_general


def handle_create_activity(prompt: str, client: TripletexClient) -> bool:
    name, number, is_general = _parse_activity_name(prompt)
    if not name:
        log_event("INFO", "activity_handler_skip", reason="no_name")
        return False

    type_id = _pick_activity_type_id(client, name)
    if type_id is None:
        log_event("WARNING", "activity_handler_no_type")
        return False

    body: dict[str, Any] = {
        "name": name,
        "isGeneral": is_general,
        "activityType": {"id": type_id},
    }
    if number:
        body["number"] = number

    err = validate_activity_body(body)
    if err:
        log_event("WARNING", "activity_handler_precheck", detail=err)
        return False

    r = client.post("/activity", json=body)
    if r.status_code not in (200, 201):
        log_event("WARNING", "activity_handler_post_failed", status=r.status_code)
        return False
    log_event("INFO", "activity_handler_ok", name=name, activity_type_id=type_id)
    return True
