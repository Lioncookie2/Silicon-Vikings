"""Deterministic project creation (projectManager from GET /employee)."""
from __future__ import annotations

import json
import os
import re
from typing import Any

from pydantic import BaseModel, Field

from ..structured_log import log_event
from ..tripletex_client import TripletexClient


class ProjectExtractor(BaseModel):
    name: str | None = Field(default=None, description="The name of the project to create")
    customer_name_hint: str | None = Field(default=None, description="The name of the customer, if specified")
    start_date: str | None = Field(default=None, description="The start date in YYYY-MM-DD format, if specified")


def _parse_project_fields(
    prompt: str, today: str
) -> tuple[str | None, str | None, str | None]:
    """project_name, customer_name_hint, start_date ISO"""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=key)
            model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
            
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=f"Extract project details. Today is {today}.",
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=ProjectExtractor,
                ),
            )
            if resp.text:
                data = json.loads(resp.text)
                ex = ProjectExtractor.model_validate(data)
                return ex.name, ex.customer_name_hint, ex.start_date or today
        except Exception as e:
            log_event("WARNING", "project_llm_extract_failed", error=str(e))
            pass
            
    # Fallback to regex
    pname = None
    m = re.search(
        r"(?:prosjekt|project)\s*[\"«]([^\"»]+)[\"»]",
        prompt,
        re.IGNORECASE,
    )
    if m:
        pname = m.group(1).strip()
    if not pname:
        m2 = re.search(
            r"(?:named|navn)\s*[:.]?\s*[\"«]([^\"»]+)[\"»]",
            prompt,
            re.IGNORECASE,
        )
        pname = m2.group(1).strip() if m2 else None

    cust = None
    cm = re.search(
        r"(?:kunde|customer)\s*[\"«]([^\"»]+)[\"»]",
        prompt,
        re.IGNORECASE,
    )
    if cm:
        cust = cm.group(1).strip()

    start = None
    sm = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", prompt)
    if sm:
        start = sm.group(1)
    else:
        sm2 = re.search(r"\b(\d{1,2})[./](\d{1,2})[./](20\d{2})\b", prompt)
        if sm2:
            d, mo, y = int(sm2.group(1)), int(sm2.group(2)), sm2.group(3)
            start = f"{y}-{mo:02d}-{d:02d}"
    if not start:
        start = today

    return pname, cust, start


def _first_employee_id(client: TripletexClient) -> int | None:
    er = client.get(
        "/employee",
        params={"fields": "id,firstName,lastName", "count": 10},
    )
    if er.status_code != 200:
        return None
    vals = er.json().get("values") or []
    if not vals or not isinstance(vals[0], dict):
        return None
    try:
        return int(vals[0]["id"])
    except (TypeError, ValueError, KeyError):
        return None


def _resolve_customer_id(client: TripletexClient, name_hint: str) -> int | None:
    cr = client.get(
        "/customer",
        params={
            "name": name_hint,
            "fields": "id,name",
            "count": 20,
        },
    )
    if cr.status_code != 200:
        return None
    hint_l = name_hint.strip().lower()
    for row in cr.json().get("values") or []:
        if not isinstance(row, dict):
            continue
        n = row.get("name")
        if isinstance(n, str) and n.strip().lower() == hint_l:
            try:
                return int(row["id"])
            except (TypeError, ValueError, KeyError):
                continue
    return None


def handle_create_project(prompt: str, client: TripletexClient, today: str) -> bool:
    pname, cust_hint, start = _parse_project_fields(prompt, today)
    if not pname:
        log_event("INFO", "project_handler_skip", reason="no_name")
        return False

    pm_id = _first_employee_id(client)
    if pm_id is None:
        log_event("WARNING", "project_handler_no_manager")
        return False

    body: dict = {
        "name": pname,
        "startDate": start,
        "projectManager": {"id": pm_id},
    }
    if cust_hint:
        cid = _resolve_customer_id(client, cust_hint)
        if cid is not None:
            body["customer"] = {"id": cid}

    pr = client.post("/project", json=body)
    if pr.status_code not in (200, 201):
        log_event("WARNING", "project_handler_post_failed", status=pr.status_code)
        return False
    log_event("INFO", "project_handler_ok", name=pname, project_manager_id=pm_id)
    return True
