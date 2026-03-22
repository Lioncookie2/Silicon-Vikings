"""
Optional explicit handlers per task category.

The default agent (agent.py) uses an LLM to emit a JSON call plan.
Deterministic handlers here bypass the LLM entirely for well-known patterns,
saving both LLM cost and API steps (better competition score).

Usage in solve():
    from .task_handlers import classify_task, try_handle_deterministically
    if not files and try_handle_deterministically(prompt, client, today, year_start, file_context):
        return
"""
from __future__ import annotations

from .activities import handle_create_activity
from .classifier import (
    ANALYSIS_ONLY,
    CREATE_ACTIVITY,
    CREATE_CUSTOMER,
    CREATE_DEPARTMENTS,
    CREATE_DIMENSIONS,
    CREATE_EMPLOYEE,
    CREATE_ORDER,
    CREATE_PAYROLL,
    CREATE_PROJECT,
    CREATE_VOUCHER,
    BANK_RECONCILIATION,
    EMPLOYEE_ONBOARDING,
    INVOICE_PAYMENT,
    classify_task,
)
from .customers import handle_create_customer
from .departments import handle_department_batch_task
from .employees import handle_create_employee
from .invoices import handle_invoice_task
from .projects import handle_create_project
from .orders import handle_create_order
from .payroll import handle_payroll
from .vouchers import handle_create_voucher
from .reconciliation import handle_bank_reconciliation
from .dimensions import handle_dimensions
from ..tripletex_client import TripletexClient

__all__ = [
    "ANALYSIS_ONLY",
    "classify_task",
    "try_handle_deterministically",
]


def try_handle_deterministically(
    prompt: str,
    client: TripletexClient,
    today: str,
    year_start: str,
    file_context: str = "",
) -> bool:
    """
    Try to handle the task with a deterministic handler.
    Returns True if the task was handled (caller should return immediately),
    False if the LLM agent loop should take over.
    """
    prompt_lower = prompt.lower()
    task_type = classify_task(prompt, file_context)

    if handle_department_batch_task(prompt, client):
        return True

    if task_type == EMPLOYEE_ONBOARDING:
        if handle_create_employee(prompt, client, today, full_onboarding=True, file_context=file_context):
            return True
    elif task_type == CREATE_EMPLOYEE:
        if handle_create_employee(prompt, client, today, full_onboarding=False, file_context=file_context):
            return True

    if task_type == CREATE_CUSTOMER:
        if handle_create_customer(prompt, client):
            return True

    if task_type == CREATE_PROJECT:
        if handle_create_project(prompt, client, today):
            return True

    if task_type == CREATE_ACTIVITY:
        if handle_create_activity(prompt, client):
            return True

    if task_type == CREATE_ORDER:
        if handle_create_order(prompt, client, today):
            return True

    if task_type == BANK_RECONCILIATION:
        if handle_bank_reconciliation(prompt, client, today, year_start, file_context):
            return True

    if task_type == CREATE_PAYROLL:
        if handle_payroll(prompt, client, today):
            return True

    if task_type == CREATE_DIMENSIONS:
        if handle_dimensions(prompt, client):
            return True

    if task_type == CREATE_VOUCHER:
        if handle_create_voucher(prompt, client, today, year_start):
            return True

    # Invoice-related tasks (keyword + classifier)
    invoice_keywords = [
        "faktura",
        "invoice",
        "send faktura",
        "send invoice",
        "betal",
        "payment",
        "kreditnota",
        "credit note",
    ]
    if task_type == INVOICE_PAYMENT or any(kw in prompt_lower for kw in invoice_keywords):
        if handle_invoice_task(prompt, client, today, year_start):
            return True

    return False
