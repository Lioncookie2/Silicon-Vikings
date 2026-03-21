"""
Optional explicit handlers per task category.

The default agent (agent.py) uses an LLM to emit a JSON call plan.
Deterministic handlers here bypass the LLM entirely for well-known patterns,
saving both LLM cost and API steps (better competition score).

Usage in solve():
    from .task_handlers import try_handle_deterministically
    if try_handle_deterministically(prompt, client, today, year_start):
        return
"""
from __future__ import annotations

from datetime import date

from ..tripletex_client import TripletexClient
from .invoices import handle_invoice_task


def try_handle_deterministically(
    prompt: str,
    client: TripletexClient,
    today: str,
    year_start: str,
) -> bool:
    """
    Try to handle the task with a deterministic handler.
    Returns True if the task was handled (caller should return immediately),
    False if the LLM agent loop should take over.
    """
    prompt_lower = prompt.lower()

    # Invoice-related tasks
    invoice_keywords = [
        "faktura", "invoice", "send faktura", "send invoice",
        "betal", "payment", "kreditnota", "credit note",
    ]
    if any(kw in prompt_lower for kw in invoice_keywords):
        if handle_invoice_task(prompt, client, today, year_start):
            return True

    return False
