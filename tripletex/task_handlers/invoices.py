"""
Deterministic invoice task handlers.

Handles patterns:
  - "list invoices" / "finn fakturaer" — GET /invoice with correct required params
  - "send invoice {id}" / "send faktura {id}" — PUT /invoice/{id}/send
  - "pay invoice {id}" / "betal faktura {id}" — POST /invoice/{id}/payment

Returns True if the task was fully handled, False to fall through to the LLM agent.
"""
from __future__ import annotations

import re

from ..tripletex_client import TripletexClient


# ── helpers ──────────────────────────────────────────────────────────────────

def _list_invoices(
    client: TripletexClient,
    today: str,
    year_start: str,
    customer_id: int | None = None,
) -> list[dict]:
    """Fetch all invoices in the current year. Returns the values list."""
    params: dict = {
        "invoiceDateFrom": year_start,
        "invoiceDateTo": today,
        "fields": "id,invoiceNumber,customer,amountCurrency,invoiceDate,dueDate,amount",
        "count": 100,
    }
    if customer_id is not None:
        params["customerId"] = customer_id

    resp = client.get("/invoice", params=params)
    if resp.status_code == 200:
        data = resp.json()
        return data.get("values") or []
    return []


# ── pattern handlers ──────────────────────────────────────────────────────────

def _try_list_invoices(
    prompt: str,
    client: TripletexClient,
    today: str,
    year_start: str,
) -> bool:
    """Handle "list invoices" / "vis fakturaer" — no mutations needed."""
    list_patterns = [
        r"\blist\b.*\binvoice",
        r"\bshow\b.*\binvoice",
        r"\bget\b.*\binvoice",
        r"\bfetch\b.*\binvoice",
        r"\bvis\b.*\bfaktura",
        r"\bhent\b.*\bfaktura",
        r"\bfinn\b.*\bfaktura",
        r"\blist\b.*\bfaktura",
        r"\balle faktura",
        r"\ball invoice",
    ]
    if not any(re.search(p, prompt, re.IGNORECASE) for p in list_patterns):
        return False

    invoices = _list_invoices(client, today, year_start)
    print(f"[task_handler:invoices] listed {len(invoices)} invoices (0 write calls)")
    return True


def _try_send_invoice(
    prompt: str,
    client: TripletexClient,
    today: str,
    year_start: str,
) -> bool:
    """
    Handle "send invoice {id/number}" or "send faktura {id/number}".
    If no explicit ID, look up the invoice first.
    """
    send_patterns = [
        r"\bsend\b.*\binvoice\b",
        r"\bsend\b.*\bfaktura\b",
        r"\bemail\b.*\binvoice\b",
        r"\bsend faktura\b",
        r"\bsend invoice\b",
    ]
    if not any(re.search(p, prompt, re.IGNORECASE) for p in send_patterns):
        return False

    # Try to extract an explicit invoice ID or number from prompt
    id_match = re.search(r"\b(?:id|#|nr\.?|number)?\s*(\d{3,})\b", prompt, re.IGNORECASE)
    if not id_match:
        # Can't deterministically resolve which invoice — let LLM handle it
        return False

    candidate = int(id_match.group(1))

    # Try treating it as a direct invoice ID first
    resp = client.put(f"/invoice/{candidate}/send", json=None)
    if resp.status_code in (200, 204):
        print(f"[task_handler:invoices] sent invoice id={candidate} (1 write call)")
        return True

    # Candidate may be an invoice number — look it up
    invoices = _list_invoices(client, today, year_start)
    matched = [inv for inv in invoices if str(inv.get("invoiceNumber")) == str(candidate)]
    if len(matched) == 1:
        inv_id = matched[0]["id"]
        resp2 = client.put(f"/invoice/{inv_id}/send", json=None)
        if resp2.status_code in (200, 204):
            print(f"[task_handler:invoices] sent invoice id={inv_id} (number={candidate}) (2 calls total)")
            return True

    # Could not resolve deterministically — fall through
    return False


def _try_pay_invoice(
    prompt: str,
    client: TripletexClient,
    today: str,
    year_start: str,
) -> bool:
    """
    Handle "pay invoice {id}" / "betal faktura {id}" with a known amount.
    Only handles cases where both invoice ID and amount are explicit in the prompt.
    """
    pay_patterns = [
        r"\bpay\b.*\binvoice\b",
        r"\bpayment\b.*\binvoice\b",
        r"\bbetal\b.*\bfaktura\b",
        r"\bregistrer betaling\b",
    ]
    if not any(re.search(p, prompt, re.IGNORECASE) for p in pay_patterns):
        return False

    id_match = re.search(r"\b(?:invoice|faktura)\b[^\d]*(\d{3,})", prompt, re.IGNORECASE)
    amount_match = re.search(r"\b(\d[\d\s]*(?:[.,]\d+)?)\s*(?:kr|nok|,-)?", prompt, re.IGNORECASE)
    if not id_match or not amount_match:
        return False

    inv_id = int(id_match.group(1))
    raw_amount = amount_match.group(1).replace(" ", "").replace(",", ".")
    try:
        amount = float(raw_amount)
    except ValueError:
        return False

    if amount <= 0:
        return False

    body = {
        "paymentDate": today,
        "paymentTypeId": 2,
        "amount": amount,
        "currency": {"id": 1},
    }
    resp = None
    for url in (f"/invoice/{inv_id}/payment", f"/invoice/{inv_id}/:payment"):
        resp = client.post(url, json=body)
        if resp.status_code != 404:
            break
        resp = client.put(url, json=body)
        if resp.status_code != 404:
            break
    if resp is not None and resp.status_code in (200, 201):
        print(f"[task_handler:invoices] registered payment {amount} on invoice {inv_id} (1 write call)")
        return True

    return False


# ── public entry point ────────────────────────────────────────────────────────

def handle_invoice_task(
    prompt: str,
    client: TripletexClient,
    today: str,
    year_start: str,
) -> bool:
    """
    Try each deterministic invoice handler in order.
    Returns True if fully handled, False to fall through to LLM.
    """
    for handler in (_try_list_invoices, _try_send_invoice, _try_pay_invoice):
        if handler(prompt, client, today, year_start):
            return True
    return False
