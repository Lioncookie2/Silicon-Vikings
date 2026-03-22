"""
Lightweight task classification before the LLM loop (regex + keyword lists).

Returns a coarse label so deterministic handlers and prompt hints can route correctly.
"""
from __future__ import annotations

import re

# Labels returned by classify_task
ANALYSIS_ONLY = "analysis_only"
CREATE_DEPARTMENTS = "create_departments"
CREATE_EMPLOYEE = "create_employee"
EMPLOYEE_ONBOARDING = "employee_onboarding"
CREATE_CUSTOMER = "create_customer"
CREATE_PROJECT = "create_project"
CREATE_ACTIVITY = "create_activity"
CREATE_VOUCHER = "create_voucher"
SUPPLIER_INVOICE = "supplier_invoice"
CREATE_ORDER = "create_order"
BANK_RECONCILIATION = "bank_reconciliation"
INVOICE_PAYMENT = "invoice_payment"
LEDGER_CORRECTION = "ledger_correction"
UNKNOWN = "unknown"

_ANALYSIS_VERBS = (
    r"\banalysere\b",
    r"\banalyser\b",
    r"\banalyse\b",
    r"\banalysieren\b",
    r"\bidentifisere\b",
    r"\bidentifiser\b",
    r"\bidentifizieren\b",
    r"\bfinn feil\b",
    r"\bfinn\b.*\bfeil\b",
    r"\bcompare\b",
    r"\bvergleichen\b",
    r"\bsammenlign\b",
    r"\brapporter\b",
    r"\breport\b",
    r"\bsummarize\b",
    r"\boppsummer\b",
    r"\breview\b",
    r"\bhovedbok\b",
    r"\bbilag\b",
    r"\bgeneral ledger\b",
    r"\bvouchers?\b",
    r"\bposteringer\b",
)

_ACTION_VERBS = (
    r"\bopprett\b",
    r"\bopprette\b",
    r"\bcreate\b",
    r"\berstell",
    r"\bregistrer\b",
    r"\bregister\b",
    r"\bpost\b",
    r"\blag\b.*\b(faktura|invoice|bilag|voucher)\b",
    r"\bsend\b",
    r"\bbetal\b",
    r"\bpay\b",
    r"\badd\b",
    r"\blegg til\b",
    r"\bslett\b",
    r"\bdelete\b",
    r"\bput\b",
    r"\bupdate\b",
    r"\boppdater\b",
)


def _text_for_classification(prompt: str, file_context: str) -> str:
    return f"{prompt}\n{file_context or ''}".lower()


def _matches_any(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def classify_task(prompt: str, file_context: str = "") -> str:
    """
    Classify the user task into a coarse type.

    Order matters: more specific labels before generic ones.
    """
    t = _text_for_classification(prompt, file_context)

    # Ledger / supplier / invoice families (before generic analysis)
    if _matches_any(
        (
            r"\bleverandørfaktura\b",
            r"\binngående faktura\b",
            r"\bsupplier invoice\b",
            r"\blieferantenrechnung\b",
            r"\bfactura de proveedor\b",
            r"\bincoming invoice\b",
        ),
        t,
    ):
        return SUPPLIER_INVOICE

    if _matches_any(
        (
            r"\brette\b",
            r"\bkorriger\b",
            r"\bfix\b",
            r"\bcorrect\b",
            r"\bposteringsfeil\b",
            r"\bduplikat\b.*\bbilag\b",
            r"\bremove duplicate\b",
            r"\bledger correction\b",
            r"\bwrong account\b",
            r"\bslett\b.*\bbilag\b",
            r"\bdelete\b.*\bvoucher\b",
        ),
        t,
    ) and _matches_any(
        (r"\bhovedbok\b", r"\bbilag\b", r"\bledger\b", r"\bvoucher\b", r"\bposting\b"),
        t,
    ):
        return LEDGER_CORRECTION

    if _matches_any(
        (
            r"\bpay\b.*\binvoice\b",
            r"\bpayment\b.*\binvoice\b",
            r"\bbetal\b.*\bfaktura\b",
            r"\bregistrer betaling\b",
            r"\bzahlung\b",
        ),
        t,
    ):
        return INVOICE_PAYMENT

    if _matches_any(
        (r"\bavstemming\b", r"\breconcile\b", r"\bbank statement\b", r"\bkontoutskrift\b", r"\bmatch\b.*\bpayments\b"),
        t,
    ):
        return BANK_RECONCILIATION

    # Employee onboarding (salary, employment, stillingsprosent)
    if _matches_any(
        (
            r"\btilbud\b",
            r"\bstillingsprosent\b",
            r"\blønn\b",
            r"\bemployment details\b",
            r"\bemployee onboarding\b",
            r"\bny ansatt\b",
            r"\bnew employee\b",
            r"\bannualsalary\b",
            r"\bmonthly wage\b",
        ),
        t,
    ) and _matches_any((r"\bansatt\b", r"\bemployee\b", r"\bmedarbeider\b"), t):
        return EMPLOYEE_ONBOARDING

    if _matches_any(
        (r"\bopprett\b.*\bansatt\b", r"\bcreate\b.*\bemployee\b", r"\bny ansatt\b"),
        t,
    ):
        return CREATE_EMPLOYEE

    # Departments batch (also matched by handler's own regex)
    if _matches_any((r"\btre\s+avdeling", r"\bthree\s+departments?\b"), t):
        return CREATE_DEPARTMENTS

    if _matches_any(
        (r"\bopprett\b.*\bkunde\b", r"\bcreate\b.*\bcustomer\b", r"\bny kunde\b"),
        t,
    ):
        return CREATE_CUSTOMER

    if _matches_any(
        (r"\bopprett\b.*\bprosjekt\b", r"\bcreate\b.*\bproject\b", r"\bnytt prosjekt\b"),
        t,
    ):
        return CREATE_PROJECT

    if _matches_any(
        (r"\bopprett\b.*\bbilag\b", r"\bcreate\b.*\bvoucher\b", r"\bnytt bilag\b", r"\bregistrer\b.*\bbilag\b"),
        t,
    ):
        return CREATE_VOUCHER

    if _matches_any(
        (r"\bopprett\b.*\baktivitet\b",
         r"\bcreate\b.*\bactivity\b",
         r"\bny aktivitet\b",
         r"\btimesheet\b.*\bactivity\b",),
        t,
    ):
        return CREATE_ACTIVITY

    if _matches_any(
        (r"\bopprett\b.*\bordre\b", r"\bcreate\b.*\border\b", r"\bny ordre\b", r"\bopprett\b.*\border\b"),
        t,
    ):
        return CREATE_ORDER

    # Analysis-only: analysis verbs without strong action verbs
    if _matches_any(_ANALYSIS_VERBS, t) and not _matches_any(_ACTION_VERBS, t):
        return ANALYSIS_ONLY

    return UNKNOWN
