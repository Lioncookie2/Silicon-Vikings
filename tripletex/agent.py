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

from .task_handlers import try_handle_deterministically
from .tripletex_client import TripletexClient

# ── constants ────────────────────────────────────────────────────────────────
MAX_STEPS = 20              # hard cap — never loop forever
RESPONSE_TRUNCATE_GET = 4000   # GET list responses can be large — agent needs to see all IDs
RESPONSE_TRUNCATE_WRITE = 800  # POST/PUT/DELETE only return {value:{id:X}}, so 800 is fine

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
- Never output anything except the JSON object.

## Key Tripletex v2 endpoints

### Employees
  POST /employee          body: {firstName, lastName, email, employeeNumber(optional)}
  GET  /employee          params: {fields:"id,firstName,lastName,email", count:10}
  PUT  /employee/{id}     body: partial update

### Roles (assign after creating employee)
  PUT /employee/{id}/employment/... — use employeeId link
  To set admin role: PUT /employee/{id} body: {"administrator":true}

### Customers
  POST /customer          body: {name, email, isCustomer:true, phone(optional)}
  GET  /customer          params: {name:"...", fields:"id,name,email", count:10}
  PUT  /customer/{id}     body: partial update

### Products
  POST /product           body: {name, number(optional), costExcludingVatCurrency, priceExcludingVatCurrency}
  GET  /product           params: {fields:"id,name,number", count:10}

### Orders
  POST /order             body: {customer:{id:X}, orderDate:"YYYY-MM-DD", deliveryDate:"YYYY-MM-DD"}
  POST /order/{id}/orderline   body: {product:{id:X}, count:N, unitPriceExcludingVat:N}

### Invoices
  GET  /invoice           params: {invoiceDateFrom:"YYYY-MM-DD", invoiceDateTo:"YYYY-MM-DD",
                                    fields:"id,invoiceNumber,customer,amountCurrency,invoiceDate,dueDate",
                                    count:100}
                          NOTE: invoiceDateFrom AND invoiceDateTo are REQUIRED (400 without them).
                          Default range: use year-start to today (both provided in context as TODAY and YEAR_START).
  POST /invoice           body: {invoiceDate:"YYYY-MM-DD", invoiceDueDate:"YYYY-MM-DD",
                                  customer:{id:X}, orders:[{id:Y}]}
  PUT  /invoice/{id}/send     — send invoice by email (no request body needed)

### Payments
  POST /invoice/{id}/payment  body: {paymentDate:"YYYY-MM-DD", paymentTypeId:2,
                                      amount:X, currency:{id:1}}

### Credit notes
  POST /invoice/{id}/createCreditNote  body: {date:"YYYY-MM-DD"}

### Travel expenses
  POST /travelExpense      body: {employee:{id:X}, description:"...", travelDetails:{...}}
  GET  /travelExpense      params: {fields:"id,description", count:10}
  DELETE /travelExpense/{id}

### Projects
  POST /project           body: {name:"...", customer:{id:X}, startDate:"YYYY-MM-DD"}
  GET  /project           params: {fields:"id,name,version", count:10}
  PUT  /project/{id}      body: {version:N, name:"...", ...}  ← version is REQUIRED for all PUTs
                          Always GET the resource first to obtain its current version number.

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

### Ledger / accounts
  GET  /ledger/account    params: {fields:"id,number,name", count:100}

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
9. Never call the same endpoint more than twice in a row — you already have the data.
   If you received a 200 response, extract the IDs from it and move to the next step.
10. NEVER output {"action":"done"} at step 0 without making at least one API call.
    If you are unsure how to proceed, start by fetching the relevant entity (e.g. GET /employee).
    Always attempt the task — do not refuse based on assumed API limitations.
11. For ALL PUT requests: first GET the resource to obtain its current "version" number.
    Include version in the PUT body: {"version": N, ...other fields...}
    Without version, Tripletex returns 422.
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
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"No JSON in LLM output: {text[:400]}")
    return json.loads(m.group(0))


# ── single API call execution ─────────────────────────────────────────────────

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
    """Extract text from PDF bytes using pypdf (lightweight, no network)."""
    try:
        from pypdf import PdfReader  # type: ignore
        import io
        reader = PdfReader(io.BytesIO(raw))
        parts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
        return "\n".join(parts)[:3000]
    except Exception:
        return "(PDF — could not extract text)"


def _process_files(files: list[dict[str, str]], workdir: Path) -> str:
    """Decode base64 files, extract text from PDFs, return context string."""
    workdir.mkdir(parents=True, exist_ok=True)
    context_parts: list[str] = []
    for f in files:
        name = f.get("filename") or "attachment.bin"
        content_b64 = f.get("content_base64") or ""
        try:
            raw = base64.b64decode(content_b64)
        except Exception:
            continue
        path = workdir / name
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
        else:
            context_parts.append(f"[File: {name} — binary, {len(raw)} bytes]")
    return "\n\n".join(context_parts)


# ── agentic loop ──────────────────────────────────────────────────────────────

def solve(
    prompt: str,
    files: list[dict[str, str]],
    tripletex_credentials: dict[str, str],
    workdir: Path,
) -> None:
    """
    Main entry point. Runs the agentic loop until done or MAX_STEPS reached.
    The TripletexClient uses base_url + session_token from credentials.
    """
    base_url = tripletex_credentials["base_url"]
    token = tripletex_credentials["session_token"]
    client = TripletexClient(base_url, token)

    # Process attached files
    file_context = _process_files(files, workdir)

    today = date.today().isoformat()
    year_start = f"{date.today().year}-01-01"

    # Try a deterministic handler before falling back to the LLM agent loop
    if not files and try_handle_deterministically(prompt, client, today, year_start):
        print("[agent] task handled deterministically — no LLM steps used")
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

    print(f"[agent] starting agentic loop, max_steps={MAX_STEPS}")

    recent_calls: list[tuple[str, str]] = []  # (method, path) of successful calls

    for step in range(MAX_STEPS):
        print(f"[step {step}] calling LLM ({len(messages)} messages in context)")
        raw_llm = _llm_complete(messages)

        try:
            action = _extract_json(raw_llm)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"[step {step}] JSON parse error: {e} — feeding back to LLM")
            messages.append({"role": "assistant", "content": raw_llm})
            messages.append({
                "role": "user",
                "content": f"ERROR: Could not parse your response as JSON: {e}. "
                           "Output ONLY a valid JSON object."
            })
            continue

        messages.append({"role": "assistant", "content": json.dumps(action)})

        action_type = (action.get("action") or "call").lower()
        print(f"[step {step}] action={action_type} method={action.get('method','')} path={action.get('path','')}")

        if action_type == "done":
            print(f"[step {step}] agent done: {action.get('reasoning','')}")
            break

        if action_type != "call":
            print(f"[step {step}] unknown action type: {action_type}")
            messages.append({
                "role": "user",
                "content": f"Unknown action type '{action_type}'. Use 'call' or 'done'."
            })
            continue

        # Execute the API call
        try:
            resp = _execute_call(client, action)
        except Exception as e:
            print(f"[step {step}] client error: {e}")
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

        truncate_limit = (
            RESPONSE_TRUNCATE_GET if action.get("method", "GET").upper() == "GET"
            else RESPONSE_TRUNCATE_WRITE
        )
        resp_summary = json.dumps({
            "status": status,
            "body": body_text,
        }, ensure_ascii=False)[:truncate_limit]

        print(f"[step {step}] API {action.get('method','')} {action.get('path','')} → {status}")

        if status >= 400:
            feedback = (
                f"API ERROR {status}:\n{resp_summary}\n"
                "Read the error message carefully and fix the request."
            )
            # Also detect loops on error responses (e.g. repeated 404s)
            call_key = (action.get("method", "GET").upper(), action.get("path", ""))
            recent_calls.append(call_key)
            if len(recent_calls) >= 3 and len(set(recent_calls[-3:])) == 1:
                print(f"[step {step}] error loop detected — same failing endpoint called 3+ times")
                feedback += (
                    "\n\nWARNING: You have called this exact endpoint 3 times in a row and it keeps failing. "
                    "This endpoint does NOT exist or you are using the wrong path. "
                    "Stop trying it. Use a different endpoint or output {\"action\":\"done\"}."
                )
        else:
            feedback = f"API SUCCESS {status}:\n{resp_summary}"
            # Track successful calls for loop detection
            call_key = (action.get("method", "GET").upper(), action.get("path", ""))
            recent_calls.append(call_key)
            if len(recent_calls) >= 3 and len(set(recent_calls[-3:])) == 1:
                print(f"[step {step}] loop detected — same endpoint called 3+ times in a row")
                feedback += (
                    "\n\nWARNING: You have called this exact endpoint 3 times in a row with 200 responses. "
                    "You already have all the data you need. Do NOT call it again. "
                    "Proceed to the next action or output {\"action\":\"done\"} if the task is complete."
                )

        messages.append({"role": "user", "content": feedback})

    else:
        print(f"[agent] reached MAX_STEPS={MAX_STEPS} without done action")

    print(f"[agent] finished after {min(step+1, MAX_STEPS)} steps")
