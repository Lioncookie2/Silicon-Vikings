"""Deterministic bank reconciliation using LLM."""
from __future__ import annotations

import csv
import io
import json
import os
import re
from typing import Any

from pydantic import BaseModel, Field

from ..structured_log import log_event
from ..tripletex_client import TripletexClient


class ReconcileMatch(BaseModel):
    date: str = Field(description="The date of the payment in YYYY-MM-DD")
    amount: float = Field(description="The amount paid (positive for incoming/customer, negative for outgoing/supplier)")
    description: str = Field(description="The description from the bank statement")
    is_incoming: bool = Field(description="True if this is an incoming payment from a customer, False if outgoing to a supplier")
    invoice_or_vendor_hint: str | None = Field(description="Any extracted name or invoice number to help find the match")


class ReconcileExtractor(BaseModel):
    matches: list[ReconcileMatch] = Field(description="The list of payments found in the bank statement that need to be recorded")


def _extract_reconciliation_data(prompt: str, file_context: str) -> ReconcileExtractor | None:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key or not file_context:
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

        system_instruction = (
            "You are an accountant reconciling a bank statement. "
            "Extract all payments from the attached CSV/text. "
            "Identify if they are incoming (from customers, positive amount) or outgoing (to suppliers, negative amount). "
            "Return the list of payments."
        )

        resp = client.models.generate_content(
            model=model_name,
            contents=f"Prompt:\n{prompt}\n\nFile:\n{file_context}",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=ReconcileExtractor,
            ),
        )
        if resp.text:
            data = json.loads(resp.text)
            return ReconcileExtractor.model_validate(data)
    except Exception as e:
        log_event("WARNING", "reconcile_llm_extract_failed", error=str(e))
        pass

    return None


def _find_matching_customer_invoice(client: TripletexClient, match: ReconcileMatch, year_start: str, today: str) -> int | None:
    """Find an unpaid outgoing invoice that matches the amount."""
    params = {
        "invoiceDateFrom": "2020-01-01", # broad search
        "invoiceDateTo": today,
        "fields": "id,amount,amountOutstanding,invoiceNumber,customer(name)",
        "count": 100,
    }
    r = client.get("/invoice", params=params)
    if r.status_code != 200:
        return None

    best_match_id = None
    for inv in r.json().get("values", []):
        outstanding = inv.get("amountOutstanding", 0)
        # Match amount (tolerate tiny float diffs)
        if abs(outstanding - match.amount) < 0.01:
            best_match_id = inv["id"]
            # If we also match the invoice number in the description, it's a perfect match
            if match.invoice_or_vendor_hint and str(inv.get("invoiceNumber", "")) in match.invoice_or_vendor_hint:
                return inv["id"]

    return best_match_id


def handle_bank_reconciliation(prompt: str, client: TripletexClient, today: str, year_start: str, file_context: str) -> bool:
    extracted = _extract_reconciliation_data(prompt, file_context)
    if not extracted or not extracted.matches:
        return False

    success = True
    for match in extracted.matches:
        if match.is_incoming:
            # Try to pay a customer invoice
            inv_id = _find_matching_customer_invoice(client, match, year_start, today)
            if inv_id:
                body = {
                    "paymentDate": match.date or today,
                    "paymentTypeId": 2, # Bank
                    "paidAmount": match.amount,
                    "currency": {"id": 1},
                }
                
                # Tripletex accepts either /payment or /:payment
                r = client.post(f"/invoice/{inv_id}/:payment", json=body)
                if r.status_code == 404:
                    r = client.put(f"/invoice/{inv_id}/:payment", json=body)
                if r.status_code == 404:
                    r = client.post(f"/invoice/{inv_id}/payment", json=body)
                if r.status_code == 404:
                    r = client.put(f"/invoice/{inv_id}/payment", json=body)

                if r.status_code in (200, 201, 204):
                    log_event("INFO", "reconcile_paid_invoice", invoice_id=inv_id, amount=match.amount)
                else:
                    log_event("WARNING", "reconcile_pay_invoice_failed", invoice_id=inv_id, status=r.status_code)
                    success = False
            else:
                # If we can't find an invoice, we might need to do a manual voucher, but let's fall back to LLM or just skip
                log_event("WARNING", "reconcile_no_invoice_found", amount=match.amount)
                success = False
        else:
            # Outgoing (supplier) payment. Usually we'd need a voucher for this or /supplierInvoice payment
            # For now, if there are supplier payments, we let the LLM handle the whole task to be safe, 
            # or we could implement supplier invoice payment if we knew the endpoints.
            log_event("WARNING", "reconcile_supplier_payment_unsupported_in_deterministic", amount=match.amount)
            return False

    return success