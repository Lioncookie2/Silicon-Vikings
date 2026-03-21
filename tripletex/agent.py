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

from .structured_log import log_event, log_api_error, set_request_id
from .task_handlers import try_handle_deterministically
from .tripletex_client import TripletexClient

# ── constants ────────────────────────────────────────────────────────────────
MAX_STEPS = 20              # hard cap — never loop forever
RESPONSE_TRUNCATE_GET = 4000   # GET list responses can be large — agent needs to see all IDs
RESPONSE_TRUNCATE_WRITE = 800  # POST/PUT/DELETE only return {value:{id:X}}, so 800 is fine
RESPONSE_TRUNCATE_ERROR = 2500  # error bodies may contain validationMessages — keep more

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
  GET  /employee          params: {fields:"id,firstName,lastName,email", count:10}
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
  GET  /customer          params: {name:"...", fields:"id,name,email", count:10}
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
  GET  /product           params: {fields:"id,name,number,priceExcludingVatCurrency,vatType", count:10}
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
                          Default range: use year-start to today (both provided in context as TODAY and YEAR_START).
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

### Payments
  POST /invoice/{id}/payment  body: {paymentDate:"YYYY-MM-DD", paymentTypeId:2,
                                      amount:X, currency:{id:1}}

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
  NOTE: /ledger/voucher GET requires dateFrom and dateTo (REQUIRED params).

### Ledger review / correction (hovedbok, finn feil i bilag, rette posteringsfeil)
  When the task asks to FIND ERRORS, REVIEW vouchers, FIX wrong account, REMOVE duplicate bilag,
  or similar (Norwegian: "hovedbok", "bilag", "finn feil", "rett", "duplikat") — do NOT blindly POST /ledger/voucher.
  Flow:
  1. GET /ledger/account?fields=id,number,name&count=100  — ONCE, map account numbers (6540, 6860, 7000, …) to IDs
  2. GET /ledger/posting?dateFrom=YYYY-MM-DD&dateTo=YYYY-MM-DD&fields=*,account,voucher,amount,vatType
     — dateFrom/dateTo are REQUIRED. Use the period from the task (e.g. Jan–Feb 2026 → 2026-01-01 to 2026-03-01 exclusive end).
     Filter mentally or by accountId after resolving account IDs from step 1.
  3. GET /ledger/voucher?dateFrom=...&dateTo=...&fields=*  — list bilag in the same period
  4. GET /ledger/voucher/{id}?fields=*  — full voucher with postings for a specific bilag you need to fix
  5. PUT /ledger/voucher/{id}?sendToLedger=true  body: {version:N, postings:[...]}  — update existing bilag (GET version first)
     CRITICAL: Do NOT include "row" field in posting objects — row 0 is system-generated and causes 422. Omit "row" entirely.
  6. DELETE /ledger/voucher/{id}  — remove a duplicate bilag if the task says so
  Keywords: "manglande MVA" = add/fix VAT line on posting; wrong konto = change account on posting.

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
13. Tasks about reviewing or fixing the general ledger (hovedbok, bilag, posteringsfeil):
    Search with GET /ledger/posting and GET /ledger/voucher using dateFrom/dateTo from the task period.
    Do NOT create new vouchers with POST until you have identified what to fix.
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
"""


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


def _assign_posting_rows(body: dict[str, Any] | None, path: str) -> dict[str, Any] | None:
    """Assign 1-based row numbers to postings in /ledger/voucher calls.
    Tripletex row 0 is system-generated; when row is missing the API defaults to 0 → 422.
    We always override row to 1, 2, 3... regardless of what the LLM sent."""
    if body is None or "ledger/voucher" not in path:
        return body
    postings = body.get("postings")
    if isinstance(postings, list):
        renumbered = []
        for i, p in enumerate(postings):
            if isinstance(p, dict):
                renumbered.append({**{k: v for k, v in p.items() if k != "row"}, "row": i + 1})
        return {**body, "postings": renumbered}
    return body


def _execute_call(
    client: TripletexClient, action: dict[str, Any]
) -> _requests.Response:
    method = (action.get("method") or "GET").upper()
    path = action.get("path") or "/"
    params = action.get("params")
    body = action.get("json")

    if not isinstance(params, dict):
        params = None
    if not isinstance(body, dict):
        body = None

    body = _assign_posting_rows(body, path)

    if method == "GET":
        return client.get(path, params=params)
    elif method == "POST":
        return client.post(path, json=body)
    elif method == "PUT":
        return client.put(path, json=body)
    elif method == "DELETE":
        return client.delete(path)
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")


# ── file handling ─────────────────────────────────────────────────────────────

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
        return text[:3000]

    # Fallback: send raw PDF bytes to Gemini Vision
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
                return extracted[:3000]
        except Exception:
            pass

    return text[:3000] if text else "(PDF — could not extract text)"


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
    """Decode base64 files, extract text from PDFs/images, return context string."""
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
        elif ext in {".txt", ".csv", ".json"}:
            try:
                context_parts.append(f"[File: {name}]\n{raw.decode('utf-8', errors='replace')[:2000]}")
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

    # Try a deterministic handler before falling back to the LLM agent loop
    if not files and try_handle_deterministically(prompt, client, today, year_start):
        log_event("INFO", "deterministic_handler", result="task handled without LLM")
        return

    # Build initial user message
    user_message_parts = [
        f"TODAY: {today}\nYEAR_START: {year_start}",
        "TASK:\n" + prompt,
    ]
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

    for step in range(MAX_STEPS):
        log_event("DEBUG", "llm_call", step=step, context_messages=len(messages))
        raw_llm = _llm_complete(messages)

        try:
            action = _extract_json(raw_llm)
        except (ValueError, json.JSONDecodeError) as e:
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
            log_event("INFO", "agent_done", step=step, reasoning=str(action.get("reasoning", ""))[:400])
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
        truncate_limit = (
            RESPONSE_TRUNCATE_ERROR if is_error
            else RESPONSE_TRUNCATE_GET if method.upper() == "GET"
            else RESPONSE_TRUNCATE_WRITE
        )
        resp_summary = json.dumps({
            "status": status,
            "body": body_text,
        }, ensure_ascii=False)[:truncate_limit]

        if is_error:
            log_api_error(step, method, path, status, body_text)
        else:
            log_event("INFO", "api_call", step=step, method=method, path=path, status=status)

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
                    "Try GET /project/projectType to see project types, or skip fixed-price if unavailable."
                )
            if status == 422 and method.upper() == "POST" and (path or "").rstrip("/") == "/ledger/voucher":
                feedback += (
                    "\n\nHINT: POST /ledger/voucher usually needs voucherType:{id} from GET /ledger/voucherType. "
                    "For hovedbok correction, search with GET /ledger/posting + GET /ledger/voucher, then PUT/DELETE."
                )
            if status == 422 and method.upper() == "POST" and p_low.rstrip("/").endswith("/employee"):
                feedback += (
                    "\n\nHINT: userType must be a string: \"STANDARD_USER\" (default), \"EMPLOYEE\", or \"ADMINISTRATOR\". "
                    "Do NOT send {id:X} or call GET /employee/userType (that path does not exist). "
                    "Fix: add \"userType\": \"STANDARD_USER\" to the body."
                )
            if status == 400 and method.upper() == "GET" and "/invoice" in p_low and "dueDate" in str(body_text):
                feedback += (
                    "\n\nHINT: GET /invoice `fields` must use only InvoiceDTO field names. "
                    "Remove dueDate — use invoiceDueDate instead (or omit fields filter)."
                )
            if status == 404 and method.upper() == "PUT" and "/invoice/" in p_low and "/send" in p_low:
                feedback += (
                    "\n\nHINT: PUT /invoice/{id}/send 404 — verify id with GET /invoice (date range) in this company."
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
            if n_identical >= 5:
                log_event(
                    "ERROR", "hard_stop",
                    step=step, call_key=str(call_key), n_identical=n_identical,
                )
                messages.append({"role": "user", "content": feedback})
                messages.append({
                    "role": "user",
                    "content": (
                        "HARD STOP: You have failed on this exact endpoint 5 times in a row. "
                        "You MUST output {\"action\":\"done\",\"reasoning\":\"could not complete — endpoint failed repeatedly\"} RIGHT NOW. "
                        "Do NOT make any more API calls."
                    )
                })
                continue
            elif n_identical >= 3:
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
                log_event("WARNING", "success_loop", step=step, call_key=str(call_key))
                feedback += (
                    "\n\nWARNING: You have called this exact endpoint 3 times in a row with 200 responses. "
                    "You already have all the data you need. Do NOT call it again. "
                    "Proceed to the next action or output {\"action\":\"done\"} if the task is complete."
                )

        messages.append({"role": "user", "content": feedback})

    else:
        log_event("WARNING", "max_steps_reached", max_steps=MAX_STEPS)

    log_event("INFO", "agent_finished", steps_used=min(step + 1, MAX_STEPS))
