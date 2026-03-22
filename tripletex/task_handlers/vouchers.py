"""Deterministic voucher creation via LLM extraction."""
from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel, Field

from ..structured_log import log_event
from ..tripletex_client import TripletexClient


class PostingExtractor(BaseModel):
    description: str = Field(description="Description of the posting line")
    account_number: int = Field(description="The 4-digit ledger account number (e.g. 1920, 2400, 6300)")
    amount: float = Field(description="The amount in NOK. Positive for debit, negative for credit.")
    vat_percentage: float = Field(description="The VAT percentage (0, 15, 25). If unsure, assume 0.")
    supplier_name_hint: str | None = Field(default=None, description="If this is a supplier/vendor line, their name")
    customer_name_hint: str | None = Field(default=None, description="If this is a customer line, their name")


class VoucherExtractor(BaseModel):
    date: str = Field(description="The date of the voucher in YYYY-MM-DD format")
    description: str = Field(description="The main description of the voucher")
    postings: list[PostingExtractor] = Field(description="The accounting lines. Sum of amounts must be 0.")


def _extract_voucher_data(prompt: str, today: str, year_start: str) -> VoucherExtractor | None:
    """Use Gemini structured outputs to parse the voucher request."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        log_event("WARNING", "voucher_handler_no_gemini_key")
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

        system_instruction = (
            f"You are a Norwegian accountant. Extract voucher details from the user's text. "
            f"Today is {today}. The year started on {year_start}. "
            f"If a date is not specified, use {today}. "
            f"Ensure the sum of posting amounts equals exactly 0. "
            f"Typical accounts: 1920 (Bank), 1500 (Accounts Receivable), 2400 (Accounts Payable), "
            f"6300 (Rent), 6800 (Office supplies). "
            f"Return ONLY valid JSON matching the schema."
        )

        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=VoucherExtractor,
            ),
        )

        if not resp.text:
            return None

        data = json.loads(resp.text)
        return VoucherExtractor.model_validate(data)

    except Exception as e:
        log_event("ERROR", "voucher_handler_extraction_failed", error=str(e))
        return None


def _resolve_account_id(client: TripletexClient, number: int) -> int | None:
    r = client.get("/ledger/account", params={"number": str(number), "fields": "id", "count": 1})
    if r.status_code == 200:
        vals = r.json().get("values") or []
        if vals:
            return vals[0].get("id")
    return None


def _resolve_vat_type_id(client: TripletexClient, percentage: float) -> int | None:
    r = client.get("/ledger/vatType", params={"fields": "id,percentage", "count": 50})
    if r.status_code == 200:
        vals = r.json().get("values") or []
        for v in vals:
            if v.get("percentage") == percentage:
                return v.get("id")
    return None


def _resolve_supplier_id(client: TripletexClient, name: str) -> int | None:
    r = client.get("/supplier", params={"name": name, "fields": "id", "count": 1})
    if r.status_code == 200:
        vals = r.json().get("values") or []
        if vals:
            return vals[0].get("id")
    return None


def _resolve_customer_id(client: TripletexClient, name: str) -> int | None:
    r = client.get("/customer", params={"name": name, "fields": "id", "count": 1})
    if r.status_code == 200:
        vals = r.json().get("values") or []
        if vals:
            return vals[0].get("id")
    return None


def _get_voucher_type_id(client: TripletexClient) -> int | None:
    # Usually id 1 or 2 is "Journal" / "Bilag"
    r = client.get("/ledger/voucherType", params={"fields": "id,name", "count": 10})
    if r.status_code == 200:
        vals = r.json().get("values") or []
        for v in vals:
            if v.get("id") in (1, 2, 3):
                return v.get("id")
        if vals:
            return vals[0].get("id")
    return None


def handle_create_voucher(prompt: str, client: TripletexClient, today: str, year_start: str) -> bool:
    """End-to-end extraction and API execution for creating a voucher."""
    extracted = _extract_voucher_data(prompt, today, year_start)
    if not extracted:
        log_event("WARNING", "voucher_handler_extraction_null")
        return False

    # 1. Validate sum == 0
    total = sum(p.amount for p in extracted.postings)
    if abs(total) > 0.01:
        log_event("WARNING", "voucher_handler_unbalanced", sum=total)
        return False

    # 2. Get voucher type
    vt_id = _get_voucher_type_id(client)
    if not vt_id:
        log_event("WARNING", "voucher_handler_no_voucher_type")
        return False

    # 3. Build postings
    api_postings = []
    for p in extracted.postings:
        acc_id = _resolve_account_id(client, p.account_number)
        if not acc_id:
            log_event("WARNING", "voucher_handler_missing_account", account=p.account_number)
            return False

        vat_id = _resolve_vat_type_id(client, p.vat_percentage)
        # Fallback to 0% if missing
        if not vat_id and p.vat_percentage == 0:
            vat_id = _resolve_vat_type_id(client, 0.0)

        line: dict[str, Any] = {
            "account": {"id": acc_id},
            "amountGross": p.amount,
            "amountGrossCurrency": p.amount,
            "description": p.description,
            "date": extracted.date,
        }
        if vat_id:
            line["vatType"] = {"id": vat_id}

        if p.supplier_name_hint:
            supp_id = _resolve_supplier_id(client, p.supplier_name_hint)
            if supp_id:
                line["supplier"] = {"id": supp_id}
            else:
                log_event("WARNING", "voucher_handler_supplier_not_found", name=p.supplier_name_hint)
                return False

        if p.customer_name_hint:
            cust_id = _resolve_customer_id(client, p.customer_name_hint)
            if cust_id:
                line["customer"] = {"id": cust_id}
            else:
                log_event("WARNING", "voucher_handler_customer_not_found", name=p.customer_name_hint)
                return False

        api_postings.append(line)

    body = {
        "date": extracted.date,
        "description": extracted.description,
        "voucherType": {"id": vt_id},
        "postings": api_postings,
    }

    resp = client.post("/ledger/voucher", json=body)
    if resp.status_code in (200, 201):
        vid = resp.json().get("value", {}).get("id")
        log_event("INFO", "voucher_handler_ok", id=vid)
        return True

    log_event("WARNING", "voucher_handler_failed", status=resp.status_code, response=resp.text[:500])
    return False
