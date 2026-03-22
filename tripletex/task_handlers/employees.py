"""Deterministic employee create / onboarding flows using LLM extraction."""
from __future__ import annotations

import json
import os
import re
from typing import Any

from pydantic import BaseModel, Field

from ..employee_post import execute_post_employee
from ..structured_log import log_event
from ..tripletex_client import TripletexClient


class EmployeeExtractor(BaseModel):
    first_name: str = Field(description="First name of the employee")
    last_name: str = Field(description="Last name of the employee")
    email: str | None = Field(default=None, description="Email address, if any")
    date_of_birth: str | None = Field(default=None, description="Date of birth in YYYY-MM-DD format")
    department_name_hint: str | None = Field(default=None, description="Name of the department they should belong to")
    start_date: str | None = Field(default=None, description="Employment start date in YYYY-MM-DD")
    annual_salary: float | None = Field(default=None, description="Annual salary in NOK")
    percentage_fte: float | None = Field(default=None, description="Percentage of full time equivalent (e.g. 100 for 100%)")


def _extract_employee_data(prompt: str, file_context: str, today: str) -> EmployeeExtractor | None:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

        text_to_analyze = f"Prompt:\n{prompt}\n\nFile Context:\n{file_context}"

        resp = client.models.generate_content(
            model=model_name,
            contents=text_to_analyze,
            config=types.GenerateContentConfig(
                system_instruction=f"Extract employee details. Today is {today}.",
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=EmployeeExtractor,
            ),
        )
        if resp.text:
            data = json.loads(resp.text)
            return EmployeeExtractor.model_validate(data)
    except Exception as e:
        log_event("WARNING", "employee_llm_extract_failed", error=str(e))
        pass

    return None


def _next_employee_number(client: TripletexClient) -> int:
    r = client.get("/employee", params={"fields": "id,employeeNumber", "count": 200})
    mx = 100
    if r.status_code == 200:
        for row in r.json().get("values") or []:
            if not isinstance(row, dict):
                continue
            n = row.get("employeeNumber")
            try:
                n_int = int(n) if n is not None else 0
                mx = max(mx, n_int)
            except (TypeError, ValueError):
                pass
    return mx + 1


def _parse_email(text: str) -> str | None:
    m = re.search(
        r"\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b",
        text,
    )
    return m.group(1).strip() if m else None


def _parse_iso_or_nor_date(text: str) -> str | None:
    m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
    if m:
        return m.group(1)
    m2 = re.search(r"\b(\d{1,2})[./](\d{1,2})[./](20\d{2})\b", text)
    if m2:
        d, mo, y = int(m2.group(1)), int(m2.group(2)), m2.group(3)
        return f"{y}-{mo:02d}-{d:02d}"
    return None


def _parse_name(text: str) -> tuple[str, str] | None:
    # "Fornavn: A Etternavn: B"
    m = re.search(
        r"(?:fornavn|first\s*name)\s*[:.]?\s*([^\n,]+?)\s*(?:etternavn|last\s*name)\s*[:.]?\s*([^\n,]+)",
        text,
        re.IGNORECASE,
    )
    if m:
        a, b = m.group(1).strip(), m.group(2).strip()
        if a and b:
            return a.split()[0], b.split()[-1]

    m2 = re.search(
        r"(?:heter|named|name\s*is)\s+([A-ZÆØÅa-zæøå][A-ZÆØÅa-zæøå-]+)\s+([A-ZÆØÅa-zæøå][A-ZÆØÅa-zæøå-]+)",
        text,
    )
    if m2:
        return m2.group(1).strip(), m2.group(2).strip()

    quoted = re.findall(r'"([^"]+)"', text)
    for q in quoted:
        parts = q.strip().split()
        if len(parts) >= 2 and "@" not in q:
            return parts[0], parts[-1]

    french = re.findall(r"«([^»]+)»", text)
    for q in french:
        parts = q.strip().split()
        if len(parts) >= 2 and "@" not in q:
            return parts[0], parts[-1]

    return None


def _parse_department_hint(text: str) -> str | None:
    m = re.search(
        r"(?:avdeling|department)\s*[:.]?\s*[\"«]?([A-ZÆØÅa-zæøå0-9][^\"»\n,]{1,60})",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().rstrip('"»').strip()
    return None


def _resolve_department_id(client: TripletexClient, hint: str) -> int | None:
    hint_l = hint.strip().lower()
    r = client.get("/department", params={"fields": "id,name", "count": 100})
    if r.status_code != 200:
        return None
    best: int | None = None
    for row in r.json().get("values") or []:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        if isinstance(name, str) and name.strip().lower() == hint_l:
            try:
                return int(row["id"])
            except (TypeError, ValueError, KeyError):
                continue
        if isinstance(name, str) and hint_l in name.strip().lower():
            try:
                best = int(row["id"])
            except (TypeError, ValueError, KeyError):
                pass
    return best


def _parse_annual_salary(text: str) -> int | None:
    m = re.search(
        r"(?:annual|årslønn|lønn|salary)\s*[:.]?\s*(\d[\d\s]{3,})\s*(?:kr|NOK)?",
        text,
        re.IGNORECASE,
    )
    if not m:
        m = re.search(r"\b(\d{3,6})\s*(?:kr|NOK|,-)\b", text, re.IGNORECASE)
    if not m:
        return None
    raw = m.group(1).replace(" ", "").replace(".", "")
    try:
        return int(raw)
    except ValueError:
        return None


def _parse_percentage_fte(text: str) -> float | None:
    m = re.search(
        r"(?:stillingsprosent|prosent|fte|full\s*time\s*equivalent)\s*[:.]?\s*(\d+(?:[.,]\d+)?)\s*%?",
        text,
        re.IGNORECASE,
    )
    if m:
        try:
            return float(m.group(1).replace(",", "."))
        except ValueError:
            return None
    return None


def handle_create_employee(
    prompt: str,
    client: TripletexClient,
    today: str,
    *,
    full_onboarding: bool = False,
    file_context: str = "",
) -> bool:
    """
    Deterministic create (and optional employment/salary) using LLM extraction.
    """
    extracted = _extract_employee_data(prompt, file_context, today)
    if not extracted:
        return False

    first = extracted.first_name
    last = extracted.last_name
    email = extracted.email or f"{first.lower()}.{last.lower()}@example.invalid"

    dept_id: int | None = None
    if extracted.department_name_hint:
        dept_id = _resolve_department_id(client, extracted.department_name_hint)

    emp_no = _next_employee_number(client)
    body: dict[str, Any] = {
        "firstName": first,
        "lastName": last,
        "email": email,
        "employeeNumber": emp_no,
        "userType": "STANDARD_USER",
    }
    if dept_id is not None:
        body["department"] = {"id": dept_id}
    if extracted.date_of_birth:
        body["dateOfBirth"] = extracted.date_of_birth

    log_event("INFO", "employee_handler_post", employeeNumber=emp_no, has_department=dept_id is not None)
    resp = execute_post_employee(client, "/employee", body)
    if resp.status_code not in (200, 201):
        log_event("WARNING", "employee_handler_post_failed", status=resp.status_code, text=resp.text[:200])
        return False

    try:
        val = resp.json().get("value") or {}
        emp_id = int(val.get("id"))
    except (TypeError, ValueError, AttributeError):
        log_event("WARNING", "employee_handler_no_id")
        return False

    start = extracted.start_date or today
    if not full_onboarding and not extracted.annual_salary and not extracted.percentage_fte:
        log_event("INFO", "employee_handler_ok", employee_id=emp_id, onboarding=False)
        return True

    salary = extracted.annual_salary
    pct = extracted.percentage_fte

    er = client.post(
        "/employee/employment",
        json={"employee": {"id": emp_id}, "startDate": start},
    )
    if er.status_code not in (200, 201):
        log_event("WARNING", "employee_handler_employment_failed", status=er.status_code, text=er.text[:200])
        return False

    try:
        emp_row = er.json().get("value") or {}
        employment_id = int(emp_row.get("id"))
    except (TypeError, ValueError, AttributeError):
        log_event("WARNING", "employee_handler_no_employment_id")
        return False

    detail_body: dict[str, Any] = {
        "employment": {"id": employment_id},
        "date": start,
        "remunerationType": "MONTHLY_WAGE",
        "percentageOfFullTimeEquivalent": float(pct if pct is not None else 100),
    }
    if salary is not None:
        detail_body["annualSalary"] = salary

    dr = client.post("/employee/employment/details", json=detail_body)
    if dr.status_code not in (200, 201):
        log_event("WARNING", "employee_handler_details_failed", status=dr.status_code, text=dr.text[:200])
        return False
        
    log_event("INFO", "employee_handler_ok", employee_id=emp_id, onboarding=True)
    return True
