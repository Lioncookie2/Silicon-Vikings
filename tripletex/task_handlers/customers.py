"""Deterministic customer creation (GET-first, then POST)."""
from __future__ import annotations

import re

from ..structured_log import log_event
from ..tripletex_client import TripletexClient


def _parse_customer_fields(prompt: str) -> tuple[str | None, str | None, str | None, str | None]:
    """name, email, phone, organizationNumber"""
    email = None
    em = re.search(
        r"\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b",
        prompt,
    )
    if em:
        email = em.group(1).strip()

    name = None
    m = re.search(
        r"(?:kunde|customer)\s*[\"«]([^\"»]+)[\"»]",
        prompt,
        re.IGNORECASE,
    )
    if m:
        name = m.group(1).strip()
    if not name:
        m2 = re.search(
            r"(?:navn|name)\s*[:.]?\s*[\"«]([^\"»]+)[\"»]",
            prompt,
            re.IGNORECASE,
        )
        name = m2.group(1).strip() if m2 else None

    phone = None
    pm = re.search(
        r"(?:telefon|phone|tlf\.?)\s*[:.]?\s*([+0-9][\d\s-]{6,})",
        prompt,
        re.IGNORECASE,
    )
    if pm:
        phone = re.sub(r"\s+", "", pm.group(1).strip())

    org = None
    om = re.search(
        r"(?:org\.?\s*nr|organisasjonsnummer|organization\s*number)\s*[:.]?\s*(\d{9})",
        prompt,
        re.IGNORECASE,
    )
    if om:
        org = om.group(1)

    return name, email, phone, org


def handle_create_customer(prompt: str, client: TripletexClient) -> bool:
    name, email, phone, org = _parse_customer_fields(prompt)
    if not name:
        log_event("INFO", "customer_handler_skip", reason="no_name")
        return False

    gr = client.get(
        "/customer",
        params={
            "name": name,
            "fields": "id,name,email,organizationNumber",
            "count": 20,
        },
    )
    if gr.status_code == 200:
        for row in gr.json().get("values") or []:
            if not isinstance(row, dict):
                continue
            n = row.get("name")
            if isinstance(n, str) and n.strip().lower() == name.strip().lower():
                log_event("INFO", "customer_handler_exists", customer_id=row.get("id"))
                return True

    body: dict = {"name": name, "isCustomer": True}
    if email:
        body["email"] = email
    if phone:
        body["phone"] = phone
    if org:
        body["organizationNumber"] = org

    pr = client.post("/customer", json=body)
    if pr.status_code not in (200, 201):
        log_event("WARNING", "customer_handler_post_failed", status=pr.status_code)
        return False
    log_event("INFO", "customer_handler_ok", name=name)
    return True
