"""
Deterministic: create three departments when the task lists exactly three quoted names
(NM-style: «tre avdelingar» + "Produksjon", "Lager", …).
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from pydantic import BaseModel, Field

from ..structured_log import log_event
from ..tripletex_client import TripletexClient


class DepartmentBatchExtractor(BaseModel):
    is_department_creation_task: bool = Field(description="True if the user is asking to create departments")
    department_names: list[str] = Field(description="The names of the departments to create")


def _extract_department_names(prompt: str) -> list[str] | None:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=key)
            model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
            
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="Extract the department names the user wants to create. Ignore other entities. Return empty list if this is not a department creation task.",
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=DepartmentBatchExtractor,
                ),
            )
            if resp.text:
                data = json.loads(resp.text)
                ex = DepartmentBatchExtractor.model_validate(data)
                if ex.is_department_creation_task and ex.department_names:
                    return ex.department_names
        except Exception as e:
            log_event("WARNING", "dept_llm_extract_failed", error=str(e))
            pass
            
    # Fallback regex
    names = re.findall(r'"([^"]+)"', prompt)
    if len(names) < 2:
        names = re.findall(r"«([^»]+)»", prompt)
    return [n.strip() for n in names if n.strip()]


def handle_department_batch_task(prompt: str, client: TripletexClient) -> bool:
    if not any(kw in prompt.lower() for kw in ("avdeling", "department", "département", "departamentos")):
        return False

    names = _extract_department_names(prompt)
    if not names:
        return False

    resp = client.get(
        "/department",
        params={"fields": "id,name,departmentNumber", "count": 100},
    )
    existing_lower: set[str] = set()
    used_numbers: set[str] = set()
    if resp.status_code == 200:
        data = resp.json()
        for row in data.get("values") or []:
            if isinstance(row, dict):
                n = row.get("name")
                if isinstance(n, str):
                    existing_lower.add(n.strip().lower())
                num = row.get("departmentNumber")
                if num is not None:
                    used_numbers.add(str(num))

    created_any = False
    for i, name in enumerate(names):
        if name.lower() in existing_lower:
            continue
        # Unique departmentNumber in this company
        dept_num = ""
        for cand in range(5000 + i * 17, 99999, 3):
            s = str(cand)
            if s not in used_numbers:
                dept_num = s
                break
        else:
            continue

        body = {"name": name, "departmentNumber": dept_num}
        r = client.post("/department", json=body)
        if r.status_code in (200, 201):
            created_any = True
            used_numbers.add(dept_num)
            existing_lower.add(name.lower())

    if created_any:
        log_event("INFO", "task_handler_departments", msg=f"created {len(names)} departments")
        
    return True
