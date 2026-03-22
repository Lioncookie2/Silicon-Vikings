"""
Tripletex AI accounting agent — agentic loop with response-chaining.

Flow per request:
  1. Decode any attached files (PDF → text via pypdf or Gemini multimodal).
  2. Feed task prompt + file content into an LLM conversation.
  3. LLM returns ONE action at a time: {"action":"call", ...} or {"action":"done"}.
  4. Execute the call, append result (incl. returned IDs) to conversation.
  5. Repeat up to MAX_STEPS or until action == "done".

Supported LLM backends (checked in order):
  GEMINI_API_KEY / GOOGLE_API_KEY  → google-generativeai (Gemini 2.0 Flash)
  OPENAI_API_KEY                   → openai (GPT-4o-mini by default)
"""
from __future__ import annotations

import base64
import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Any

import requests as _requests

from .employee_post import execute_post_employee
from .structured_log import log_event, log_api_error, set_request_id
from .task_handlers import try_handle_deterministically
from .task_handlers import ANALYSIS_ONLY, classify_task
from .task_handlers.activities import validate_activity_body
from .tripletex_client import TripletexClient

# ── constants ────────────────────────────────────────────────────────────────
MAX_STEPS = 20              # hard cap — never loop forever
RESPONSE_TRUNCATE_GET = 4000   # GET list responses can be large — agent needs to see all IDs
RESPONSE_TRUNCATE_WRITE = 800  # POST/PUT/DELETE only return {value:{id:X}}, so 800 is fine
RESPONSE_TRUNCATE_ERROR = 2500  # non-422 errors / fallback cap when serialising API body for LLM
RESPONSE_TRUNCATE_ERROR_422 = 8000  # 422: prioritize full validationMessages in JSON sent to LLM


def _hard_stop_path_streak_limit() -> int:
    """Consecutive failures on same method+path (any body) before forced done. Override: TRIPLETEX_HARD_STOP_PATH_STREAK."""
    raw = os.environ.get("TRIPLETEX_HARD_STOP_PATH_STREAK", "8").strip()
    try:
        n = int(raw, 10)
        return max(4, min(n, 50))
    except ValueError:
        return 8


def _compact_error_body_for_llm(status: int, body_text: Any) -> Any:
    """Prefer keeping message + validationMessages for 422; trim other keys to a short tail."""
    if status != 422 or not isinstance(body_text, dict):
        return body_text
    msgs = body_text.get("validationMessages")
    if msgs is None:
        return body_text
    other = {k: v for k, v in body_text.items() if k not in ("message", "validationMessages")}
    out: dict[str, Any] = {
        "message": body_text.get("message"),
        "validationMessages": msgs,
    }
    if other:
        try:
            tail = json.dumps(_truncate_strings_in_obj(other, 800), ensure_ascii=False)
        except TypeError:
            tail = str(other)[:2000]
        if len(tail) > 2000:
            tail = tail[:2000] + "…"
        out["_additionalFields"] = tail
    return out


def _truncate_strings_in_obj(obj: Any, max_len: int) -> Any:
    if isinstance(obj, dict):
        return {k: _truncate_strings_in_obj(v, max_len) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate_strings_in_obj(v, max_len) for v in obj]
    if isinstance(obj, str) and len(obj) > max_len:
        return obj[:max_len] + "…"
    return obj

# ── system prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Tripletex v2 REST API agent for NM i AI 2026.

## Your job
Complete the accounting task described by the user by making Tripletex API calls.
You control the API one step at a time. After each call you will see the API response
and must decide what to do next.

## Output format (STRICT — no markdown fences, pure JSON only)
Each turn output EXACTLY ONE of:

Call action:
{"action":"call","method":"GET|POST|PUT|DELETE","path":"/endpoint","params":{},"json":{}}

Done action (task complete):
{"action":"done","reasoning":"short explanation"}

Rules:
- "params" and "json" can be null if not needed.
- "method" must be uppercase.
- Never output anything except ONE JSON object — no text before/after, no second JSON, no trailing commentary
    (parsing fails if you add prose after the closing brace).

## Key Tripletex v2 endpoints

### Employees
  POST /employee          body: {firstName, lastName, email, employeeNumber, userType:"STANDARD_USER"}
                          — **userType is a STRING ENUM, not an object**. Valid values:
                              "STANDARD_USER"   — regular employee with login (use this as default for onboarding)
                              "EMPLOYEE"        — employee without system access (no login)
                              "ADMINISTRATOR"   — admin access
                            Send the string directly: `"userType": "STANDARD_USER"` (NOT `{id:X}`, NOT `0`)
                          — DO NOT call GET /employee/userType — that endpoint does NOT exist (API interprets
                            "userType" as a numeric ID and returns 422 "Expected number").
                          — employeeNumber: **integer**, must be **unique** in the company.
                          — Optional: division:{id:X}, department:{id:Y} — GET /division and GET /department first.
                          — Optional nested address: {addressLine1, postalCode, city}
  GET  /employee          params: {fields:"id,firstName,lastName,email", count:100}
                          NOTE: do NOT use dot notation in fields (e.g. "userType.id") — use parentheses if
                          you need nested fields: e.g. fields="id,firstName,userType(id,name)"
  GET  /division          params: {fields:"id,name", count:50}
  PUT  /employee/{id}     body: {version:N, ...}  ← GET first for version when updating

### Employee onboarding (tilbud, PDF, ny ansatt, stillingsprosent, lønn)
  Typical sequence (adjust to validationMessages):
  1. GET /department — resolve department id by name/number from the task/PDF
  2. GET /division — if the task implies a division
  3. POST /employee — include userType:"STANDARD_USER" (string, not object); fix 422 before continuing
  4. POST /employee/employment — {employee:{id}, startDate:"YYYY-MM-DD"}
  5. POST /employee/employment/details — salary: annualSalary, percentageOfFullTimeEquivalent, remunerationType
  7. Standard working hours may be part of employment details or require additional PUTs — follow API errors

### Roles (assign after creating employee)
  PUT /employee/{id}/employment/... — use employeeId link
  To set admin role: PUT /employee/{id} body: {"administrator":true}

### Customers (kjøpere — entities you sell TO)
  POST /customer          body: {name, email, isCustomer:true, phone(optional)}
  GET  /customer          params: {name:"...", fields:"id,name,email", count:100}
  PUT  /customer/{id}     body: partial update
  NOTE: Use /customer for OUTGOING invoices (sales). For bills you RECEIVE from vendors, use /supplier.

### Suppliers (leverandører — vendors you buy FROM)
  POST /supplier          body: {name:"...", organizationNumber:"...", email(optional)}
  GET  /supplier          params: {name:"...", organizationNumber:"...",
                                   fields:"id,name,organizationNumber,version", count:10}
  PUT  /supplier/{id}     body: {version:N, ...fields to update...}
  NOTE: Use /supplier when the task involves an incoming invoice, purchase, or bill from a vendor.

### Products
  POST /product           body: {name, number(optional), costExcludingVatCurrency,
                                  priceExcludingVatCurrency, vatType:{id:X}}
                          — vatType is optional but use it when a specific VAT rate is mentioned.
                            Look up the correct vatType ID first with GET /ledger/vatType.
  GET  /product           params: {fields:"id,name,number,priceExcludingVatCurrency,vatType", count:100}
  PUT  /product/{id}      body: {version:N, ...fields to update...}

### VAT types
  GET  /ledger/vatType    params: {fields:"id,name,number,percentage", count:50}
                          — returns all available VAT types. Match by percentage or name.
                            Common Norwegian rates: 25% (standard), 15% (food/mat), 12% (transport/hotel), 0% (exempt)
                          — always do this GET first when a task mentions a specific VAT rate (%)

### Orders
  POST /order             body: {customer:{id:X}, orderDate:"YYYY-MM-DD", deliveryDate:"YYYY-MM-DD"}
  POST /order/orderline   body: {order:{id:Y}, product:{id:X}, count:1,
                                 unitPriceExcludingVatCurrency:N, vatType:{id:Z}}
                          — vatType per line when the task specifies different VAT % per product line.
                            Resolve Z via GET /ledger/vatType (match percentage: 25, 15, 0).
                          For **sales** lines, pick an **outgoing / sales** VAT type (e.g. "utgående", "25%"),
                            NOT purchase/input VAT (e.g. "Fradrag inngående avgift") — that causes 422 on order lines.
                          Each orderline needs a **product** OR a **description** (validation error if both missing).
                          NOTE: /order/{id}/orderline does NOT exist (404). Use /order/orderline.
  GET  /product           params: {number:"7733", fields:"id,name,number,vatType", count:10}
                          — filter by product number to avoid repeating generic GET /product.

### Invoices (OUTGOING / sales only — to customers)
  GET  /invoice           params: {invoiceDateFrom:"YYYY-MM-DD", invoiceDateTo:"YYYY-MM-DD",
                                    fields:"id,invoiceNumber,customer,amountCurrency,invoiceDate,invoiceDueDate",
                                    count:100}
                          NOTE: invoiceDateFrom AND invoiceDateTo are REQUIRED (400 without them).
                          NOTE: do NOT use "dueDate" in fields — it is NOT a field on InvoiceDTO (400 Illegal field).
                            Use invoiceDueDate if you need the due date column.
                          NOTE: do NOT use amountOutstandingCurrency or other non-InvoiceDTO names in `fields` (400).
                            Use fields:"*" to see all columns, or GET /invoice/{id}?fields=* for one invoice.
                          Default range: use year-start to today (both provided in context as TODAY and YEAR_START).
                          OVERDUE SEARCH: If the task mentions an overdue, unpaid, or late invoice and the
                            current-year search returns zero results, immediately retry with a wider range:
                            invoiceDateFrom: HISTORY_START (provided in context, 3 years back)
                            invoiceDateTo: TODAY
                            Never give up after a single empty GET /invoice — try the wider range first.
  POST /invoice           body: {invoiceDate:"YYYY-MM-DD", invoiceDueDate:"YYYY-MM-DD",
                                  customer:{id:X}, orders:[{id:Y}]}
                          — Only ONE order id per invoice is supported; put all lines on that order first.
                          — This endpoint is for **customer sales invoices** only. Valid keys are those in the API
                            (customer, orders, dates, …). Do NOT add supplier fields, invoiceLines, or
                            supplierInvoiceNumber here — 422 "unknown field" means wrong API or wrong object type.
  GET  /invoice/paymentType  params: {fields:"id,description", count:20}
                          — look up valid payment types if needed; do NOT pass paymentType as a top-level field
                            on POST /invoice — "paymentType" does not exist on InvoiceDTO (will get 422).
  PUT  /invoice/{id}/send     — send invoice by email (no request body needed)
                          If 404: the invoice id does not exist in this company — GET /invoice with dates first
                            and use an id from that list; do not invent ids from old responses.

### Payments (customer invoice — register incoming payment / German: Zahlung registrieren)
  Tripletex documents the action as **/invoice/{id}/:payment** (colon before `payment`). The runtime tries, in order:
  POST /payment, PUT /payment, POST /:payment, PUT /:payment (and PUT-first if you used PUT).
  Body: {paymentDate:"YYYY-MM-DD", paymentTypeId:X, paidAmount:AMOUNT, currency:{id:1}}
                          — Look up paymentTypeId via GET /invoice/paymentType (bank transfer is often id 2 or similar).
                          — **paidAmount**: prefer **amountOutstanding** / totals from GET /invoice/{id}?fields=*
                            (do not invent field names like amountOutstandingCurrency in `fields` filter).
                            If the task says "ohne MwSt." / ex-VAT, still pay the **full** open balance the API returns.
  Fallback — record payment via POST /ledger/voucher **only** if all payment URLs return 404:
    **Exactly two lines**: debit **1920** (bank), credit **1500** (kundefordringer). **vatType 0%** on BOTH.
    Do **NOT** post to **1600** (utgående MVA) or **2708** (inngående MVA) for a payment — those are wrong; payment
    clears AR against bank without re-booking VAT.
    Include **customer:{id}** on **both** postings. Same absolute amount (e.g. debit 1920 +amount, credit 1500 -amount).

### Reminder fees / overdue invoice handling
  When the task asks to "post a reminder fee", "purregebyr", "inkassogebyr", or any late fee
  on an overdue invoice:
  Step 1: GET /invoice (wide date range: HISTORY_START to TODAY) — find the overdue invoice id
  Step 2: GET /ledger/account (fields:"id,number,name", count:200)
            — locate account 1500 (kundefordringer / accounts receivable) and the reminder fee income account
  Step 3: GET /ledger/vatType — find VAT type id for 0% (reminder fees are VAT-exempt)
  Step 4: GET /ledger/voucherType — get a voucher type id
  Step 5: POST /ledger/voucher with two postings:
            body: {
              description: "Purregebyr",
              date: "YYYY-MM-DD",
              voucherType: {id: X},
              postings: [
                {account:{id:<1500-account-id>}, amountCurrency: 70, vatType:{id:<0%-vat-id>}},
                {account:{id:<reminder-income-account-id>}, amountCurrency: -70, vatType:{id:<0%-vat-id>}}
              ]
            }
  NOTE: Do NOT attempt to add a reminder fee via POST /invoice or order lines.
        Use ledger/voucher as shown above.

### Credit notes
  POST /invoice/{id}/createCreditNote  body: {date:"YYYY-MM-DD"}

### Travel expenses
  POST /travelExpense      body: {
                             employee:{id:X},
                             description:"purpose of the trip",
                             travelDetails:{
                               departureDate:"YYYY-MM-DD",
                               returnDate:"YYYY-MM-DD",
                               departureFrom:"City A",
                               destination:"City B",
                               purpose:"Business meeting",
                               isForeignTravel:false,
                               isDayTrip:false,
                               isCompensationFromRates:false
                             }
                           }
                           NOTE: employee, description and travelDetails are required.
                           travelDetails.departureDate and returnDate must be valid dates.
  GET  /travelExpense      params: {fields:"id,description,employee", count:10}
  DELETE /travelExpense/{id}

### Projects
  POST /project           body: {name:"...", customer:{id:X}, startDate:"YYYY-MM-DD",
                                  projectManager:{id:Y}}
                          — **projectManager is REQUIRED** (422 "Feltet «Prosjektleder» må fylles ut" if missing).
                            Always GET /employee first to find an employee id, then use that as projectManager.
  GET  /project           params: {fields:"id,name,version", count:10}
  PUT  /project/{id}      body: {version:N, name:"...", ...}  ← version is REQUIRED for all PUTs
                          Always GET the resource first to obtain its current version number.
  NOTE: Neither **fixedPrice** nor **budget** are fields on ProjectDTO — 422 if you try to set them.
        Fixed-price project tasks: create an order line (POST /order/orderline) linked to the project's
        order with the fixed price amount. If you cannot, document in reasoning and move on.
  NOTE: GET /project/projectType does NOT exist (API treats "projectType" as numeric ID → 422).
        Do not call that path.

### Activities (for timesheet entries)
  GET  /activity                    params: {fields:"id,name,isGeneral", count:100}
                                    — list all available activities
  GET  /activity/>forTimeSheet      params: {projectId:X, employeeId:Y, date:"YYYY-MM-DD",
                                             fields:"id,name"}
                                    — find valid activities for a specific project/employee/date
  POST /activity                    body: {name:"...", number:"...", isGeneral:true,
                                           activityType:{id:X}}
                                    — activityType is REQUIRED (422 if null/missing).
                                      Look up valid types with GET /activity/type first:
                                      GET /activity/type  params: {fields:"id,name", count:20}
                                      Then use one of the returned ids as activityType:{id:X}.
  NOTE: /timesheet/activity does NOT exist — always use /activity

### Timesheet entries (hours worked)
  GET  /timesheet/entry             params: {dateFrom:"YYYY-MM-DD", dateTo:"YYYY-MM-DD",
                                             employeeId:X, fields:"id,date,hours,activity,project"}
                                    NOTE: dateFrom AND dateTo are REQUIRED
  POST /timesheet/entry             body: {date:"YYYY-MM-DD", hours:7.5,
                                           employee:{id:X}, project:{id:Y}, activity:{id:Z},
                                           comment:"optional"}
                                    — activity ID must come from GET /activity first
  PUT  /timesheet/entry/{id}        body: {version:N, date:"...", hours:N, ...}

### Departments
  POST /department        body: {name:"...", departmentNumber:"..."}
  GET  /department        params: {fields:"id,name", count:10}

### Incoming invoices / supplier invoices (leverandørfaktura — bills you RECEIVE)
  IMPORTANT: When the task mentions "leverandørfaktura", "inngående faktura", "supplier invoice",
  "Lieferantenrechnung", "factura de proveedor", or similar — do NOT use /invoice.
  Use this flow instead:
  Step 1: GET /supplier?name=...    — find or verify supplier exists
  Step 2: POST /supplier            — create supplier if not found
            body: {name:"...", organizationNumber:"..."}
  Step 3: GET /ledger/account       — find account numbers (e.g. account 6300 for office services,
            params: {fields:"id,number,name", count:100}   2711 for input VAT, 2400 for accounts payable)
  Step 4: GET /ledger/vatType       — find correct VAT type by percentage
  Step 5: GET /ledger/voucherType   — pick a voucher type id (required for POST)
  Step 6: POST /ledger/voucher      — create the accounting entry (bilag)
            body: {
              voucherType: { id: VOUCHER_TYPE_ID },
              date: "YYYY-MM-DD",
              description: "Invoice description",
              externalVoucherNumber: "INV-...",
              postings: [
                {date:"YYYY-MM-DD", description:"Expense description",
                 account:{id:EXPENSE_ACCOUNT_ID}, supplier:{id:SUPPLIER_ID},
                 vatType:{id:VAT_TYPE_ID}, amountGross:TOTAL_INCL_VAT},
                {date:"YYYY-MM-DD", description:"Accounts payable",
                 account:{id:PAYABLE_ACCOUNT_ID}, supplier:{id:SUPPLIER_ID},
                 amountGross:-TOTAL_INCL_VAT}
              ]
            }
  CRITICAL: Do NOT include a "row" field in posting objects — row 0 is system-generated and causes 422.
            Omit "row" entirely from every posting object.
  NOTE: amountGross is the gross amount including VAT. Tripletex calculates net/VAT split automatically.
  NOTE: If you set amountGross, also set amountGrossCurrency to the **same numeric value** (422 if they differ).
  NOTE: **VAT type per account**: each ledger account has a default/locked VAT treatment in the chart.
            If validationMessages say «låst til mva-kode 0» / "locked to VAT code 0", you MUST use the 0% VAT type id
            from GET /ledger/vatType (e.g. "Ingen avgiftsbehandling") for **that posting line** — do not use 25% in
            on accounts locked to 0%.
  NOTE: For the **credit (kreditor)** line, use a real **accounts payable** account (typically **2400** —
            «Leverandørgjeld» / trade payables), NOT random balance-sheet accounts (2290, 2180, 2360, …) which are
            often locked to 0% and will reject wrong VAT.
  NOTE: /ledger/voucher GET requires dateFrom and dateTo (REQUIRED params).

### Ledger review / correction (hovedbok, finn feil i bilag, rette posteringsfeil)
  When the task asks to FIND ERRORS, REVIEW vouchers, FIX wrong account, REMOVE duplicate bilag,
  or similar (Norwegian: "hovedbok", "bilag", "finn feil", "rett", "duplikat" / English: "general ledger",
  "vouchers", "errors", "wrong account") — **correction = PUT existing voucher**, not a new POST unless the task
  explicitly says to create a new bilag.
  Flow:
  1. GET /ledger/account?fields=id,number,name&count=100  — ONCE, map account numbers to IDs
  2. GET /ledger/posting?dateFrom=YYYY-MM-DD&dateTo=YYYY-MM-DD&fields=*,account,voucher,amount,vatType&count=1000
     — dateFrom/dateTo REQUIRED. Use the task period (e.g. Jan–Feb 2026 → 2026-01-01 .. 2026-02-29 or end of Feb).
     Use **voucher.id** (or nested voucher on posting) only as a **hint** — always confirm against step 3.
  3. GET /ledger/voucher?dateFrom=...&dateTo=...&fields=*&count=200  — list all bilag in the period.
     **Voucher id for GET/PUT/DELETE** = each element's **top-level `id` in `values[]`** from THIS response only.
     Do NOT invent ids, do NOT use posting ids, gui numbers, or ids from truncated/hallucinated JSON.
  4. GET /ledger/voucher/{id}?fields=*  — only if {id} appears in step 3's `values[].id`. If **404**: re-run step 3
     with wider count or dates — do NOT retry the same id more than once; pick another id from the fresh list.
  5. PUT /ledger/voucher/{id}?sendToLedger=true  body: {version:N, voucherType:{id:...}, date:..., description:...,
     postings:[...]}  — copy structure from GET in step 4; change only what the task requires (wrong account → new
     account:{id}; wrong vatType → fix per line). **version** from GET is mandatory.
     **VAT per line**: system accounts like **2700, 2712, 2710, 2703** (MVA-kontoer) and many balance-sheet lines are
     **locked to mva-kode 0** («Ingen avgiftsbehandling»). Other accounts may be locked to **mva-kode 6** (e.g.
     «Ingen utgående avgift (utenfor mva-loven)»). Use GET /ledger/vatType and pick the **exact** id whose name matches
     **each** validation message — **different lines may need different vatType ids**.
  6. DELETE /ledger/voucher/{id}  — only if the task explicitly says duplicate/remove; id must be from step 3.
     If DELETE returns **404**, the id is wrong — get a new list (step 3), do NOT repeat the same DELETE path.
  Do **NOT** spam POST /ledger/voucher with synthetic multi-line entries to "fix" hovedbok — that causes 422 on
  locked VAT accounts. Prefer PUT after a successful GET of the real voucher.
  Keywords: "manglande MVA" = fix vatType/account on existing posting via PUT; wrong konto = PUT with new account id.

### Ledger / accounts
  GET  /ledger/account    params: {fields:"id,number,name", count:100}
  Call at most **ONCE** per task — you get all accounts in one call with count:100.
  Do NOT repeat GET /ledger/account multiple times; use the result from the first call to map all needed IDs.
  GET  /ledger/voucherType  params: {fields:"id,name", count:50}  — bilagstyper; needed for POST /ledger/voucher
  GET  /ledger/voucher       params: {dateFrom:"YYYY-MM-DD", dateTo:"YYYY-MM-DD", fields:"*", count:100}
  GET  /ledger/voucher/{id}  params: {fields:"*"}
  GET  /ledger/posting       params: {dateFrom:"YYYY-MM-DD", dateTo:"YYYY-MM-DD", fields:"*", count:1000}
                               — dateFrom and dateTo are REQUIRED

### Modules (enable accounting features)
  PUT  /company/settings/accounting  body: {use_department_accounting:true} (example)

### Free accounting dimensions (custom dimensions — «egen dimensjon» / French: dimension comptable)
  Tripletex v2.73+ exposes **user-defined dimensions** (up to 3 slots). Requires Pro package and API role
  «Regnskapsinnstillinger, kontoplan og historisk balanse» for POST/PUT.
  Flow when the task asks to create a dimension named X with values A, B, …:
  1. GET /ledger/accountingDimensionName   params: {fields:"id,name,dimensionIndex", count:10}
     — see existing dimensions and free slots (index 1–3).
  2. POST /ledger/accountingDimensionName  body: {name:"Produktlinje"}  — creates next free slot (adjust fields per validationMessages if 422).
  3. GET /ledger/accountingDimensionValue  params: {fields:"id,name,accountingDimensionName", count:50}
  4. POST /ledger/accountingDimensionValue  body: {name:"Basis", accountingDimensionName:{id:DIMENSION_NAME_ID}}
  5. POST /ledger/accountingDimensionValue  body: {name:"Avansert", accountingDimensionName:{id:SAME_ID}}
  When posting a voucher, attach the chosen value to each posting line using PostingDTO fields (writable since v2.72.05):
    freeAccountingDimension1:{id:VALUE_ID}   — use slot 1, 2, or 3 matching the dimensionIndex from step 1
    (or freeAccountingDimension2 / freeAccountingDimension3 for the other slots)
  If these endpoints return 403/404, the company may lack Pro or the integration key lacks rights — document in done reasoning.

### Employment & salary (set/update salary for an employee)
  GET  /employee/employment          params: {employeeId:X, fields:"id,employee,startDate"}
                                     — get the employment record ID for an employee
  POST /employee/employment          body: {employee:{id:X}, startDate:"YYYY-MM-DD"}
                                     — create employment if none exists
  GET  /employee/employment/details  params: {employmentId:Y, fields:"id,employment,annualSalary,hourlyWage,remunerationType"}
                                     — get current salary details
  POST /employee/employment/details  body: {employment:{id:Y}, date:"YYYY-MM-DD",
                                            remunerationType:"MONTHLY_WAGE",
                                            annualSalary:360000,
                                            percentageOfFullTimeEquivalent:100}
                                     — set new salary (remunerationType: MONTHLY_WAGE or HOURLY_WAGE)
  PUT  /employee/employment/details/{id}  body: same as POST above (update existing)
  GET  /employee/employment/employmentType/salaryType  — list salary type IDs

### Salary transactions / payroll runs
  POST /salary/transaction           body: {date:"YYYY-MM-DD", year:YYYY, month:M,
                                            payslips:[{employee:{id:X}}]}
                                     — create a payroll run for one or more employees
  GET  /salary/transaction/{id}      — get a payroll run
  GET  /salary/payslip               params: {employeeId:X, yearFrom:YYYY, monthFrom:M,
                                              yearTo:YYYY, monthTo:M, fields:"id,employee,amount"}
                                     — get payslips for an employee

## Response format from Tripletex
  Success POST/PUT:  {"value": {"id": 123, ...}}     ← use value.id in next call
  Success GET:       {"values": [...], "count": N}    ← use values[i].id
  Error:             {"message":"...", "validationMessages":[...]}

## Important rules
1. ALWAYS use the "id" from a previous response when referencing entities.
2. Never guess IDs — fetch them with GET first if you don't have them.
3. On 4xx errors: read the message and fix the request in the next step.
4. Use today's date (from the task or context) for all date fields.
5. If the sandbox is empty, create prerequisites first (e.g. customer before invoice).
6. Minimize write calls — every unnecessary POST/PUT/DELETE reduces your score.
7. GET calls are free — use them to look up IDs or verify.
8. When done, output {"action":"done","reasoning":"..."} — never loop unnecessarily.
9. Never call the same endpoint **with the same params/body** more than twice in a row.
   If you need another GET on the same path, change params (e.g. product number filter) so it is not identical.
   If you received a 200 response, extract the IDs from it and move to the next step.
10. NEVER output {"action":"done"} at step 0 without making at least one API call.
    If you are unsure how to proceed, start by fetching the relevant entity (e.g. GET /employee).
    Always attempt the task — do not refuse based on assumed API limitations.
11. For ALL PUT requests: first GET the resource to obtain its current "version" number.
    Include version in the PUT body: {"version": N, ...other fields...}
    Without version, Tripletex returns 422.
12. On 404 errors: assume wrong endpoint/path first.
    Do NOT call the same 404 endpoint again. Switch to a different documented endpoint.
13. Tasks about reviewing or fixing the general ledger (hovedbok, bilag, posteringsfeil, "general ledger", "vouchers"):
    Use GET /ledger/posting + GET /ledger/voucher (list) in the task period. Voucher ids for GET/PUT/DELETE must come
    **only** from `values[].id` in the voucher list response — never guessed. Prefer **PUT /ledger/voucher/{id}** with
    `version` from GET to fix errors; avoid POST /ledger/voucher unless creating a genuinely new bilag. On 404 for
    GET/DELETE voucher, refresh the list once — do not retry the same wrong id. MVA lines: match each account's locked
    code via GET /ledger/vatType (0% vs «utenfor mva-loven» / code 6 — can differ per posting line).
14. On POST /invoice → 422: read validationMessages exactly.
    - "bankkontonummer" = the company has no bank account registered — this CANNOT be fixed via API.
      Output {"action":"done","reasoning":"invoice creation blocked: company has no bank account (cannot fix via API)"}.
    - "paymentType" = do NOT add paymentType as a top-level field on POST /invoice; it is not a valid field.
    - Other 422: retry with corrected fields.
15. POST /invoice is ONLY for **outgoing customer invoices** (body: customer + orders). If validationMessages
    mention unknown fields like invoiceLines or supplierInvoiceNumber, you are using the wrong API — use the
    supplier + POST /ledger/voucher flow instead (see "Incoming invoices").
16. POST /ledger/voucher for **new** vouchers requires voucherType:{id} from GET /ledger/voucherType.
    For **hovedbok correction** tasks, prefer GET /ledger/posting + GET /ledger/voucher + PUT/DELETE — do not
    spam POST /ledger/voucher without a voucherType.
17. POST /employee: userType is a **string enum** — send `"userType": "STANDARD_USER"` directly.
    Do NOT send `{id:X}`, do NOT call GET /employee/userType (that endpoint does not exist).
    Do NOT use dot notation in `fields` param — use parentheses for nested fields.
18. postings in POST/PUT /ledger/voucher must **never** include a "row" field.
    Row 0 is system-generated; any posting with row=0 (or row omitted but defaulting to 0) causes
    «Posteringene på rad 0 er systemgenererte». Strip "row" from every posting object.
19. For tasks about overdue/unpaid/late invoices: ALWAYS search invoices with HISTORY_START (3 years
    back) as invoiceDateFrom, not just YEAR_START. If the first GET /invoice returns zero results,
    immediately retry with invoiceDateFrom=HISTORY_START before giving up.
    NEVER output done after a single empty invoice search without trying the wider date range.
20. ANALYSIS-ONLY tasks: if the task only asks to analyze, identify, find, compare, summarize,
    or report (German: analysieren, identifizieren, vergleichen, erkennen / Norwegian: analysere,
    identifisere, finn, rapporter) — and contains NO action verbs like "create", "register",
    "post", "add" — use ONLY GET calls and then output {"action":"done","reasoning":"<findings>"}
    with the full analysis in the reasoning field. Do NOT create projects, activities, vouchers,
    or any other objects just because the analysis revealed data about them.
21. POST /ledger/voucher postings: match **vatType** on each line to that account's locked default
    (read validationMessages — «låst til mva-kode 0» → use 0% vatType id). Use **2400** (leverandørgjeld)
    for supplier/AP credit lines unless the task names another payable account. If using amountGross,
    set amountGrossCurrency to the same value. For custom dimensions, use /ledger/accountingDimensionName
    + /ledger/accountingDimensionValue + freeAccountingDimension1/2/3 on postings (see section above).
22. Registering **customer invoice payment**: use POST or PUT on /invoice/{id}/payment **or** /invoice/{id}/:payment
    (server tries all combinations). For **ledger/voucher** fallback: **only** 1920↔1500, **MVA 0%** on both lines,
    **customer on both lines**. Never use **1600** or **2708** for payment vouchers.
    In GET /invoice `fields`, never use illegal names like amountOutstandingCurrency — use * or valid InvoiceDTO fields."""


# ── LLM backends ─────────────────────────────────────────────────────────────

def _call_gemini(messages: list[dict[str, str]]) -> str:
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY")

    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    client = genai.Client(api_key=key)

    # Build contents list for google-genai SDK
    contents: list[types.Content] = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

    resp = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1,
        ),
    )
    return resp.text or ""


def _call_openai(messages: list[dict[str, str]]) -> str:
    from openai import OpenAI  # type: ignore

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    openai_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    r = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=openai_messages,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    return r.choices[0].message.content or ""


def _llm_complete(messages: list[dict[str, str]]) -> str:
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return _call_gemini(messages)
    if os.environ.get("OPENAI_API_KEY"):
        return _call_openai(messages)
    raise RuntimeError("No LLM API key: set GEMINI_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY")


# ── action parsing ────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    # Strip markdown fences if LLM disobeys
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text.strip())
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON in LLM output: {text[:400]}")
    # First complete JSON object only — greedy regex breaks on trailing text / two objects
    decoder = json.JSONDecoder()
    try:
        obj, _end = decoder.raw_decode(text[start:])
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in LLM output: {e}: {text[:400]}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"JSON root must be object, got {type(obj)}")
    return obj


# ── single API call execution ─────────────────────────────────────────────────

def _call_signature(action: dict[str, Any]) -> tuple[str, str, str, str]:
    """Stable key for loop detection — includes params/json so GET /product?number=A vs B differs."""
    method = (action.get("method") or "GET").upper()
    path = action.get("path") or "/"
    params = action.get("params")
    body = action.get("json")
    p_s = (
        json.dumps(params, sort_keys=True, ensure_ascii=False)
        if isinstance(params, dict) and params
        else ""
    )
    j_s = (
        json.dumps(body, sort_keys=True, ensure_ascii=False)
        if isinstance(body, dict) and body
        else ""
    )
    return (method, path, p_s, j_s)


_VALID_USER_TYPES = {"STANDARD_USER", "EMPLOYEE", "ADMINISTRATOR", "EXTERNAL_USER"}


def _sanitize_employee_body(
    body: dict[str, Any] | None, method: str, path: str
) -> dict[str, Any] | None:
    """Normalize userType for POST /employee.

    Do **not** inject userType when the LLM omits it — some proxies/API builds expect the
    field absent (server default) and reject string enums like \"STANDARD_USER\".

    - Missing userType: leave unchanged.
    - int → {\"id\": n}; dict with id: keep.
    - Valid enum string: keep (fallback retries handle proxies that reject strings).
    - Otherwise: drop userType so POST can succeed with server default."""
    if body is None:
        return body
    if method.upper() != "POST" or not (path or "").rstrip("/").endswith("/employee"):
        return body
    out = dict(body)
    ut = out.get("userType")
    if ut is None:
        return out
    if isinstance(ut, dict) and ut.get("id") is not None:
        return out
    if isinstance(ut, int):
        out["userType"] = {"id": int(ut)}
        return out
    if isinstance(ut, str) and ut in _VALID_USER_TYPES:
        return out
    out.pop("userType", None)
    return out


def _execute_invoice_payment(
    client: TripletexClient,
    method: str,
    path: str,
    body: dict[str, Any],
) -> _requests.Response | None:
    """Tripletex OpenAPI uses /invoice/{id}/:payment; many proxies use /invoice/{id}/payment.
    Try both path variants with POST and PUT until one returns non-404."""
    p_norm = (path or "").rstrip("/")
    m = re.fullmatch(r"/invoice/(\d+)(?:/payment|/:payment)", p_norm)
    if not m:
        return None
    inv_id = m.group(1)
    urls = [f"/invoice/{inv_id}/payment", f"/invoice/{inv_id}/:payment"]
    attempts: list[tuple[str, str]] = []
    if method.upper() == "POST":
        for u in urls:
            attempts.extend([(u, "POST"), (u, "PUT")])
    else:
        for u in urls:
            attempts.extend([(u, "PUT"), (u, "POST")])
    last: _requests.Response | None = None
    seen: set[tuple[str, str]] = set()
    for u, verb in attempts:
        key = (u, verb)
        if key in seen:
            continue
        seen.add(key)
        last = client.post(u, json=body) if verb == "POST" else client.put(u, json=body)
        if last.status_code != 404:
            return last
    return last


def _sanitize_ledger_voucher_postings(
    body: dict[str, Any] | None, method: str, path: str
) -> dict[str, Any] | None:
    """Align with Tripletex + SYSTEM_PROMPT: do not send `row` on new vouchers.

    POST /ledger/voucher: strip `row` from every posting (row 0 is system-generated; forcing
    row 1..n caused widespread 422s).
    PUT /ledger/voucher/{id}: keep rows from GET when present; only strip row if it is 0."""
    if body is None or "ledger/voucher" not in (path or ""):
        return body
    postings = body.get("postings")
    if not isinstance(postings, list):
        return body
    p_norm = (path or "").rstrip("/")
    is_new_voucher = method.upper() == "POST" and p_norm == "/ledger/voucher"
    out: list[Any] = []
    for item in postings:
        if not isinstance(item, dict):
            out.append(item)
            continue
        d = dict(item)
        if is_new_voucher:
            d.pop("row", None)
        elif d.get("row") == 0:
            d.pop("row", None)
        out.append(d)
    return {**body, "postings": out}


class _SyntheticApiResponse:
    """Minimal response-like object for preflight validation failures (no HTTP round-trip)."""

    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self) -> Any:
        return self._payload


def _validate_write_call(
    method: str,
    path: str,
    params: dict[str, Any] | None,
    body: dict[str, Any] | None,
) -> str | None:
    """
    Return a human-readable rejection reason, or None if the call may proceed.
    Checked before sanitizers mutate the body.
    """
    m = method.upper()
    p_norm = (path or "").rstrip("/")

    if m == "PUT":
        # Do not require version for action endpoints that just perform an operation,
        # rather than updating the resource representation itself.
        is_action_endpoint = p_norm.endswith("/:payment") or p_norm.endswith("/payment") or p_norm.endswith("/send")
        if not is_action_endpoint and (not isinstance(body, dict) or "version" not in body):
            return (
                "For ALL PUT requests, you MUST include the current 'version' number "
                "in the JSON body (e.g., {\"version\": 2, ...}). "
                "Do a GET request first to obtain the current version."
            )

    if m == "GET" and p_norm == "/employee" and isinstance(params, dict):
        fields = params.get("fields")
        if isinstance(fields, str) and re.search(r"[A-Za-z0-9_]\.[A-Za-z_]", fields):
            return (
                "GET /employee: do not use dot notation in `fields` "
                '(e.g. userType.id). Use parentheses: userType(id,name).'
            )

    if m == "POST" and p_norm == "/activity":
        return validate_activity_body(body)

    if m == "POST" and p_norm == "/ledger/voucher":
        if not isinstance(body, dict):
            return "POST /ledger/voucher requires a JSON body with voucherType:{id} and postings."
        vt = body.get("voucherType")
        if not isinstance(vt, dict) or vt.get("id") is None:
            return "POST /ledger/voucher requires voucherType:{id} from GET /ledger/voucherType."
        postings = body.get("postings")
        if isinstance(postings, list):
            for i, line in enumerate(postings):
                if isinstance(line, dict) and "row" in line:
                    return (
                        "POST /ledger/voucher: remove `row` from postings — "
                        f"row keys are system-generated (posting index {i})."
                    )

    return None


def _execute_call(
    client: TripletexClient, action: dict[str, Any]
) -> _requests.Response | _SyntheticApiResponse:
    method = (action.get("method") or "GET").upper()
    path = action.get("path") or "/"
    params = action.get("params")
    body = action.get("json")

    if not isinstance(params, dict):
        params = None
    if not isinstance(body, dict):
        body = None

    pre_err = _validate_write_call(method, path, params, body)
    if pre_err is not None:
        log_event(
            "WARNING",
            "api_precheck_rejected",
            method=method,
            path=path,
            detail=pre_err[:500],
        )
        return _SyntheticApiResponse(
            422,
            {
                "message": "Preflight validation rejected request (not sent to Tripletex).",
                "validationMessages": [{"message": pre_err}],
            },
        )

    body = _sanitize_ledger_voucher_postings(body, method, path)
    body = _sanitize_employee_body(body, method, path)

    if isinstance(body, dict) and method in ("POST", "PUT"):
        pay = _execute_invoice_payment(client, method, path, body)
        if pay is not None:
            return pay

    if method == "GET":
        return client.get(path, params=params)
    elif method == "POST":
        p_norm = (path or "").rstrip("/")
        if p_norm == "/employee":
            return execute_post_employee(client, path, body)
        return client.post(path, json=body)
    elif method == "PUT":
        return client.put(path, json=body)
    elif method == "DELETE":
        return client.delete(path)
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")


# ── file handling ─────────────────────────────────────────────────────────────

def _extract_docx_text(raw: bytes) -> str:
    """Extract text from .docx (Office Open XML) using stdlib zipfile + ElementTree."""
    import zipfile, io
    import xml.etree.ElementTree as ET
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            with z.open("word/document.xml") as f:
                tree = ET.parse(f)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        texts = [el.text for el in tree.findall(".//w:t", ns) if el.text]
        return " ".join(texts)[:5000]
    except Exception as e:
        return f"(docx — could not extract: {e})"


def _extract_xlsx_text(raw: bytes) -> str:
    """Extract text from .xlsx (Office Open XML) using stdlib zipfile + ElementTree."""
    import zipfile, io
    import xml.etree.ElementTree as ET
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names = z.namelist()
            # Load shared strings (text cells are stored here by index)
            shared: list[str] = []
            if "xl/sharedStrings.xml" in names:
                with z.open("xl/sharedStrings.xml") as f:
                    tree = ET.parse(f)
                ns = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                for si in tree.findall(".//ns:si", ns):
                    parts = [el.text or "" for el in si.findall(".//ns:t", ns)]
                    shared.append("".join(parts))
            # Read the first worksheet
            sheets = sorted(n for n in names if n.startswith("xl/worksheets/sheet"))
            rows: list[str] = []
            if sheets:
                with z.open(sheets[0]) as f:
                    tree = ET.parse(f)
                ns = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                for row in tree.findall(".//ns:row", ns):
                    cells = []
                    for c in row.findall("ns:c", ns):
                        v = c.find("ns:v", ns)
                        if v is not None and v.text:
                            if c.get("t") == "s":
                                idx = int(v.text)
                                cells.append(shared[idx] if idx < len(shared) else v.text)
                            else:
                                cells.append(v.text)
                    if cells:
                        rows.append("\t".join(cells))
        return "\n".join(rows)[:5000]
    except Exception as e:
        return f"(xlsx — could not extract: {e})"


def _extract_pdf_text(raw: bytes) -> str:
    """Extract text from PDF bytes.

    Strategy:
    1. Try pypdf (fast, local, works on text-based PDFs).
    2. If pypdf returns empty/very short text AND Gemini is available, use
       Gemini Vision (gemini-2.0-flash) to OCR the first page as an image.
       This handles scanned PDFs, receipts, and hand-written invoices.
    """
    text = ""
    try:
        from pypdf import PdfReader  # type: ignore
        import io
        reader = PdfReader(io.BytesIO(raw))
        parts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
        text = "\n".join(parts)
    except Exception:
        pass

    if len(text.strip()) >= 200:
        return text[:5000]
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore

            client = genai.Client(api_key=key)
            model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
            resp = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                inline_data=types.Blob(mime_type="application/pdf", data=raw)
                            ),
                            types.Part(
                                text=(
                                    "Extract all text from this document. "
                                    "Return ONLY the raw text, no commentary."
                                )
                            ),
                        ],
                    )
                ],
            )
            extracted = resp.text or ""
            if extracted.strip():
                return extracted[:5000]
        except Exception:
            pass

    return text[:5000] if text else "(PDF — could not extract text)"


def _extract_image_text(raw: bytes, mime_type: str) -> str:
    """Use Gemini Vision to read text/info from an image file."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return "(image — no Gemini key for vision)"
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore

        client = genai.Client(api_key=key)
        model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        resp = client.models.generate_content(
            model=model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            inline_data=types.Blob(mime_type=mime_type, data=raw)
                        ),
                        types.Part(
                            text=(
                                "Describe this image and extract ALL text visible in it. "
                                "Return ONLY the extracted/described content, no commentary."
                            )
                        ),
                    ],
                )
            ],
        )
        return (resp.text or "")[:2000]
    except Exception as e:
        return f"(image — vision extraction failed: {e})"


def _process_files(files: list[dict[str, str]], workdir: Path) -> str:
    """Decode base64 files, extract text from PDFs/images/Office docs, return context string."""
    workdir.mkdir(parents=True, exist_ok=True)
    context_parts: list[str] = []

    _IMAGE_MIME: dict[str, str] = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    _TEXT_EXTS = {".txt", ".csv", ".json", ".html", ".htm", ".xml"}

    for f in files:
        name = f.get("filename") or "attachment.bin"
        content_b64 = f.get("content_base64") or ""
        try:
            raw = base64.b64decode(content_b64)
        except Exception:
            continue
        path = workdir / Path(name).name  # strip any subdirectory from filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

        ext = Path(name).suffix.lower()
        if ext == ".pdf":
            text = _extract_pdf_text(raw)
            context_parts.append(f"[File: {name}]\n{text}")
        elif ext in {".docx", ".docm"}:
            text = _extract_docx_text(raw)
            context_parts.append(f"[File: {name}]\n{text}")
        elif ext in {".xlsx", ".xlsm"}:
            text = _extract_xlsx_text(raw)
            context_parts.append(f"[File: {name}]\n{text}")
        elif ext in _TEXT_EXTS:
            try:
                context_parts.append(f"[File: {name}]\n{raw.decode('utf-8', errors='replace')[:3000]}")
            except Exception:
                pass
        elif ext in _IMAGE_MIME:
            mime = _IMAGE_MIME[ext]
            text = _extract_image_text(raw, mime)
            context_parts.append(f"[File: {name}]\n{text}")
        else:
            context_parts.append(f"[File: {name} — binary, {len(raw)} bytes]")
    return "\n\n".join(context_parts)


# ── agentic loop ──────────────────────────────────────────────────────────────

def solve(
    prompt: str,
    files: list[dict[str, str]],
    tripletex_credentials: dict[str, str],
    workdir: Path,
    request_id: str | None = None,
) -> None:
    """
    Main entry point. Runs the agentic loop until done or MAX_STEPS reached.
    The TripletexClient uses base_url + session_token from credentials.
    """
    if request_id:
        set_request_id(request_id)

    base_url = tripletex_credentials["base_url"]
    token = tripletex_credentials["session_token"]
    client = TripletexClient(base_url, token)

    # Process attached files
    file_context = _process_files(files, workdir)

    today = date.today().isoformat()
    year_start = f"{date.today().year}-01-01"
    history_start = f"{date.today().year - 3}-01-01"

    # Try a deterministic handler before falling back to the LLM agent loop
    if not files and try_handle_deterministically(
        prompt, client, today, year_start, file_context=file_context
    ):
        log_event("INFO", "deterministic_handler", result="task handled without LLM")
        log_event(
            "INFO",
            "run_summary",
            outcome="deterministic",
            steps_used=1,
            api_error_count=0,
            json_parse_error_count=0,
            hard_stop_count=0,
            client_error_count=0,
            error_loop_count=0,
            success_loop_count=0,
            had_max_steps=False,
            last_error_path="",
            last_error_status=0,
            task_preview=prompt[:200],
        )
        return

    # Build initial user message
    user_message_parts = [
        f"TODAY: {today}\nYEAR_START: {year_start}\nHISTORY_START: {history_start}",
    ]
    if classify_task(prompt, file_context) == ANALYSIS_ONLY:
        user_message_parts.append(
            "IMPORTANT: This task is classified as ANALYSIS-ONLY. "
            "Use ONLY GET API calls. Do NOT create projects, activities, vouchers, "
            "employees, customers, or any other objects. "
            "When finished, output {\"action\":\"done\",\"reasoning\":\"...\"} with your findings."
        )
    user_message_parts.append("TASK:\n" + prompt)
    if file_context:
        user_message_parts.append("ATTACHED FILES:\n" + file_context)
    user_message_parts.append(
        "Solve this task step by step. "
        "Output one JSON action per turn. "
        "Use IDs from previous API responses. "
        "When finished output {\"action\":\"done\"}."
    )
    initial_user = "\n\n".join(user_message_parts)

    # Conversation history (role: "user" | "assistant")
    messages: list[dict[str, str]] = [
        {"role": "user", "content": initial_user}
    ]

    log_event("INFO", "agent_task", task_preview=prompt[:300], max_steps=MAX_STEPS)

    recent_calls: list[tuple[str, str]] = []  # (method, path) of successful calls
    path_fail_streak: dict[tuple[str, str], int] = {}  # (method, path) → consecutive failures

    run_stats = {
        "api_error_count": 0,
        "json_parse_error_count": 0,
        "hard_stop_count": 0,
        "client_error_count": 0,
        "error_loop_count": 0,
        "success_loop_count": 0,
        "last_error_path": "",
        "last_error_status": 0,
    }
    finished_via_done = False
    hit_max_steps = False

    for step in range(MAX_STEPS):
        log_event("DEBUG", "llm_call", step=step, context_messages=len(messages))
        raw_llm = _llm_complete(messages)

        try:
            action = _extract_json(raw_llm)
        except (ValueError, json.JSONDecodeError) as e:
            run_stats["json_parse_error_count"] += 1
            log_event("WARNING", "json_parse_error", step=step, error=str(e))
            messages.append({"role": "assistant", "content": raw_llm})
            messages.append({
                "role": "user",
                "content": f"ERROR: Could not parse your response as JSON: {e}. "
                           "Output ONLY a valid JSON object."
            })
            continue

        messages.append({"role": "assistant", "content": json.dumps(action)})

        action_type = (action.get("action") or "call").lower()
        method = action.get("method", "")
        path = action.get("path", "")

        log_event(
            "INFO", "agent_action",
            step=step, action=action_type, method=method, path=path,
        )

        if action_type == "done":
            finished_via_done = True
            reasoning = str(action.get("reasoning", ""))
            if run_stats["api_error_count"] > 0:
                reasoning += (
                    f" Completed despite {run_stats['api_error_count']} API error(s); "
                    f"last failing path: {run_stats['last_error_path'] or 'n/a'} "
                    f"(HTTP {run_stats['last_error_status']})."
                )
                action["reasoning"] = reasoning
                messages[-1] = {"role": "assistant", "content": json.dumps(action)}
            log_event("INFO", "agent_done", step=step, reasoning=reasoning[:400])
            break

        if action_type != "call":
            log_event("WARNING", "unknown_action", step=step, action=action_type)
            messages.append({
                "role": "user",
                "content": f"Unknown action type '{action_type}'. Use 'call' or 'done'."
            })
            continue

        # Execute the API call
        try:
            resp = _execute_call(client, action)
        except Exception as e:
            run_stats["client_error_count"] += 1
            log_event("ERROR", "client_error", step=step, error=str(e))
            messages.append({
                "role": "user",
                "content": f"CLIENT ERROR: {e}\nFix your action and try again."
            })
            continue

        # Summarise response for conversation (truncate large bodies)
        status = resp.status_code
        try:
            body_text = resp.json()
        except Exception:
            body_text = resp.text

        is_error = status >= 400
        body_for_llm = _compact_error_body_for_llm(status, body_text) if is_error else body_text
        truncate_limit = (
            RESPONSE_TRUNCATE_GET if method.upper() == "GET" and not is_error
            else RESPONSE_TRUNCATE_WRITE if not is_error
            else RESPONSE_TRUNCATE_ERROR_422 if status == 422
            else RESPONSE_TRUNCATE_ERROR
        )
        resp_summary = json.dumps({
            "status": status,
            "body": body_for_llm,
        }, ensure_ascii=False)[:truncate_limit]

        if is_error:
            run_stats["api_error_count"] += 1
            run_stats["last_error_path"] = str(path or "")
            run_stats["last_error_status"] = int(status)
            log_api_error(step, method, path, status, body_text)
            # Track per-path consecutive failures (body-independent)
            pk = (method.upper(), (path or "").rstrip("/"))
            path_fail_streak[pk] = path_fail_streak.get(pk, 0) + 1
            if path_fail_streak[pk] >= 3:
                log_event(
                    "ERROR", "path_fail_streak",
                    step=step, method=method, path=path, streak=path_fail_streak[pk],
                )
        else:
            log_event("INFO", "api_call", step=step, method=method, path=path, status=status)
            # Reset failure streak on success
            pk = (method.upper(), (path or "").rstrip("/"))
            path_fail_streak[pk] = 0

        if is_error:
            feedback = (
                f"API ERROR {status}:\n{resp_summary}\n"
                "Read the error message carefully and fix the request."
            )
            # Targeted hints (do not replace validationMessages — append context for the LLM)
            p_low = (path or "").lower()
            if status == 422 and method.upper() == "POST" and p_low.rstrip("/").endswith("/invoice"):
                if "bankkontonummer" in str(body_text).lower() or "bank account" in str(body_text).lower():
                    feedback += (
                        "\n\nHINT: Company has no bank account registered — this CANNOT be fixed via API. "
                        "Output {\"action\":\"done\",\"reasoning\":\"invoice blocked: company has no bank account\"} immediately."
                    )
                elif "paymenttype" in str(body_text).lower():
                    feedback += (
                        "\n\nHINT: paymentType is NOT a valid field on POST /invoice body. Remove it. "
                        "Valid fields: invoiceDate, invoiceDueDate, customer:{id}, orders:[{id}]."
                    )
                else:
                    feedback += (
                        "\n\nHINT: POST /invoice is for OUTGOING customer invoices (customer + orders). "
                        "If errors mention unknown fields (invoiceLines, supplier, …), use supplier + "
                        "POST /ledger/voucher instead."
                    )
            if ("fixedprice" in str(body_text).lower() or "fixedPrice" in str(body_text)) and status in (400, 422):
                feedback += (
                    "\n\nHINT: fixedPrice is NOT a field on ProjectDTO. "
                    "Do not try to set fixedPrice via POST/PUT /project — it will always fail. "
                    "Use POST /order/orderline on an order linked to the project, or document limitation in done."
                )
            if status == 422 and method.upper() in ("POST", "PUT") and (
                (path or "").rstrip("/") == "/ledger/voucher"
                or re.search(r"/ledger/voucher/\d+", (path or ""))
            ):
                bt = str(body_text).lower()
                feedback += (
                    "\n\nHINT: POST /ledger/voucher usually needs voucherType:{id} from GET /ledger/voucherType. "
                    "For hovedbok correction, search with GET /ledger/posting + GET /ledger/voucher, then PUT/DELETE."
                )
                if "låst" in bt or "locked" in bt or "mva-kode" in bt or "vattype" in bt:
                    feedback += (
                        "\n\nHINT: Account is locked to a specific VAT code — GET /ledger/vatType and use the id "
                        "that matches **each** validation line (often 0% / «Ingen avgiftsbehandling» for 2700/2712/2710). "
                        "Do NOT use 25% VAT on accounts locked to 0%. "
                        "If the message says **mva-kode 6** / «utenfor mva-loven», pick the vatType whose name matches "
                        "that — it is different from code 0. **Each posting line may need a different vatType id.** "
                        "For supplier-style vouchers credit **2400** (leverandørgjeld), not random 22xx/23xx accounts."
                    )
                if "2700" in str(body_text) or "2712" in str(body_text) or "2710" in str(body_text) or "2703" in str(body_text):
                    feedback += (
                        "\n\nHINT: Tripletex **MVA systemkontoer** (2700, 2712, …) are almost always locked to "
                        "**Ingen avgiftsbehandling (0%)** — not high/low VAT rates. Use the 0% vatType id on those lines."
                    )
                if "utenfor mva-loven" in bt or "mva-kode 6" in bt:
                    feedback += (
                        "\n\nHINT: Account requires **mva-kode 6** (utenfor mva-loven) — select that exact vatType from "
                        "GET /ledger/vatType for the affected posting line(s), not 25% or generic 0% if the message says 6."
                    )
                if str(body_text).lower().count("låst") >= 2:
                    feedback += (
                        "\n\nHINT: Several postings failed VAT validation — fix **line by line**: each `vatType:{id}` "
                        "must match **that** account's error text; do not reuse one vatType for every row."
                    )
                if "amountgross" in bt and "amountgrosscurrency" in bt:
                    feedback += (
                        "\n\nHINT: amountGross and amountGrossCurrency must be equal on each posting. "
                        "Either set both to the same number or omit gross fields and use amountCurrency with vatType."
                    )
                if "kunde mangler" in bt or ("customer" in bt and "mangler" in bt):
                    feedback += (
                        "\n\nHINT: For customer (AR) payment vouchers, add customer:{id:X} to postings on "
                        "kundefordringer (1500) — and usually on the bank line (1920) too if validation requires it. "
                        "Use the customer id from GET /customer."
                    )
                if "1600" in str(body_text) or "utgående merverdiavgift" in bt:
                    feedback += (
                        "\n\nHINT: Recording a **payment** is NOT a VAT posting — do NOT use account 1600 or 2708. "
                        "Use only **1920 (bank)** and **1500 (AR)** with **vatType 0%** on both lines + customer:{id}."
                    )
            if status == 404 and method.upper() in ("POST", "PUT") and re.search(
                r"/invoice/\d+/(?:payment|:payment)", p_low
            ):
                feedback += (
                    "\n\nHINT: All /invoice/{id}/payment and /:payment variants returned 404 — use POST /ledger/voucher "
                    "with exactly two lines: debit 1920, credit 1500, vatType 0% on BOTH, customer:{id} on BOTH."
                )
            if status == 422 and method.upper() == "POST" and p_low.rstrip("/").endswith("/employee"):
                feedback += (
                    "\n\nHINT: userType 422 — try in order: (1) omit userType entirely for server default; "
                    "(2) \"userType\": \"STANDARD_USER\" | \"EMPLOYEE\" | \"ADMINISTRATOR\"; "
                    "(3) \"userType\": {\"id\": N} where N is from GET /employee?fields=id,userType(id). "
                    "The server may reject string enums; the agent retries automatically but your next "
                    "payload should vary if you still see this error."
                )
            if status == 422 and method.upper() == "POST" and p_low.rstrip("/").endswith("/activity"):
                if "activityType" in str(body_text).lower() or "activitytype" in str(body_text).lower():
                    feedback += (
                        "\n\nHINT: POST /activity requires activityType:{id:X}. "
                        "Call GET /activity/type (params: fields='id,name', count=20) first, "
                        "then use one of the returned ids: activityType:{id:X}. "
                        "Do NOT omit activityType or set it to null."
                    )
                if "project" in str(body_text).lower() and "eksisterer ikke" in str(body_text).lower():
                    feedback += (
                        "\n\nHINT: POST /activity does NOT accept a 'project' field — "
                        "activities are global, not per-project. Remove the 'project' field from the body."
                    )
            if status == 400 and method.upper() == "GET" and "/invoice" in p_low and "dueDate" in str(body_text):
                feedback += (
                    "\n\nHINT: GET /invoice `fields` must use only InvoiceDTO field names. "
                    "Remove dueDate — use invoiceDueDate instead (or omit fields filter)."
                )
            if status == 400 and method.upper() == "GET" and "/invoice" in p_low and (
                "amountoutstandingcurrency" in str(body_text).lower()
                or "does not match a field" in str(body_text).lower()
            ):
                feedback += (
                    "\n\nHINT: Remove invalid `fields` entries (e.g. amountOutstandingCurrency is not on InvoiceDTO). "
                    "Use fields=\"*\" or GET /invoice/{id}?fields=* for one invoice."
                )
            if status == 404 and method.upper() == "PUT" and "/invoice/" in p_low and "/send" in p_low:
                feedback += (
                    "\n\nHINT: PUT /invoice/{id}/send 404 — verify id with GET /invoice (date range) in this company."
                )
            if status == 404 and method.upper() in ("GET", "DELETE") and re.search(
                r"/ledger/voucher/\d+", p_low
            ):
                feedback += (
                    "\n\nHINT: /ledger/voucher/{id} 404 — use **id** only from GET /ledger/voucher list (`values[].id`) "
                    "in the task date range (count:200). Never use posting ids, invented ids, or ids from truncated JSON. "
                    "Re-list vouchers once; if still 404, pick a different id from the list. For corrections prefer "
                    "PUT after a successful GET, not DELETE loops."
                )
            # Also detect loops on error responses (e.g. repeated 404s / 422s)
            call_key = _call_signature(action)
            recent_calls.append(call_key)
            # Count consecutive identical calls from the end
            n_identical = 0
            for c in reversed(recent_calls):
                if c == call_key:
                    n_identical += 1
                else:
                    break
            # Per-path streak (body-independent) — catches loops where body changes but path keeps failing
            pk = (method.upper(), (path or "").rstrip("/"))
            path_streak = path_fail_streak.get(pk, 0)
            path_hard = _hard_stop_path_streak_limit()
            
            # Detect repeated loops on identical requests or GET/DELETE 404s
            identical_hard = n_identical >= 3
            if not identical_hard and status == 404 and method.upper() in ("GET", "DELETE") and n_identical >= 2:
                identical_hard = True
                
            path_only_hard = not identical_hard and path_streak >= path_hard
            if identical_hard or path_only_hard:
                run_stats["hard_stop_count"] += 1
                log_event(
                    "ERROR", "hard_stop",
                    step=step, call_key=str(call_key), n_identical=n_identical,
                    path_streak=path_streak, path_streak_limit=path_hard,
                    reason="identical_repeats" if identical_hard else "path_streak",
                )
                messages.append({"role": "user", "content": feedback})
                if identical_hard:
                    hard_txt = (
                        "HARD STOP: You have sent the **same** request repeatedly without success. "
                        "You MUST output {\"action\":\"done\",\"reasoning\":\"could not complete — same request failed repeatedly\"} RIGHT NOW. "
                        "Do NOT make any more API calls."
                    )
                else:
                    hard_txt = (
                        f"HARD STOP: The same URL has failed {path_streak} times in a row (limit {path_hard}) "
                        "with different payloads — change strategy or endpoint. "
                        "You MUST output {\"action\":\"done\",\"reasoning\":\"could not complete — endpoint failed repeatedly\"} RIGHT NOW. "
                        "Do NOT make any more API calls."
                    )
                messages.append({"role": "user", "content": hard_txt})
                continue
            elif n_identical >= 3 or path_streak >= 3:
                run_stats["error_loop_count"] += 1
                log_event(
                    "WARNING", "error_loop",
                    step=step, call_key=str(call_key), n_identical=n_identical,
                )
                feedback += (
                    "\n\nWARNING: You have called this exact endpoint 3+ times in a row and it keeps failing. "
                    "The request body or endpoint is wrong. "
                    "Either fix the body based on the error message, or output {\"action\":\"done\"}."
                )
        else:
            feedback = f"API SUCCESS {status}:\n{resp_summary}"
            # Track successful calls for loop detection
            call_key = _call_signature(action)
            recent_calls.append(call_key)
            if len(recent_calls) >= 3 and len(set(recent_calls[-3:])) == 1:
                run_stats["success_loop_count"] += 1
                log_event("WARNING", "success_loop", step=step, call_key=str(call_key))
                feedback += (
                    "\n\nWARNING: You have called this exact endpoint 3 times in a row with 200 responses. "
                    "You already have all the data you need. Do NOT call it again. "
                    "Proceed to the next action or output {\"action\":\"done\"} if the task is complete."
                )

        messages.append({"role": "user", "content": feedback})

    else:
        hit_max_steps = True
        log_event("WARNING", "max_steps_reached", max_steps=MAX_STEPS)

    steps_used = min(step + 1, MAX_STEPS)
    log_event("INFO", "agent_finished", steps_used=steps_used)

    if finished_via_done:
        outcome = "agent_done"
    elif hit_max_steps:
        outcome = "max_steps"
    else:
        outcome = "incomplete"

    log_event(
        "INFO",
        "run_summary",
        outcome=outcome,
        steps_used=steps_used,
        api_error_count=run_stats["api_error_count"],
        json_parse_error_count=run_stats["json_parse_error_count"],
        hard_stop_count=run_stats["hard_stop_count"],
        client_error_count=run_stats["client_error_count"],
        error_loop_count=run_stats["error_loop_count"],
        success_loop_count=run_stats["success_loop_count"],
        had_max_steps=hit_max_steps,
        last_error_path=run_stats["last_error_path"],
        last_error_status=run_stats["last_error_status"],
        task_preview=prompt[:200],
    )
