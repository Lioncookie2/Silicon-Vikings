"""
Deterministic: create three departments when the task lists exactly three quoted names
(NM-style: «tre avdelingar» + "Produksjon", "Lager", …).
"""
from __future__ import annotations

import re

from ..tripletex_client import TripletexClient


def handle_department_batch_task(prompt: str, client: TripletexClient) -> bool:
    if not re.search(r"\btre\s+avdeling", prompt, re.IGNORECASE) and not re.search(
        r"\bthree\s+departments?\b", prompt, re.IGNORECASE
    ):
        return False

    names = re.findall(r'"([^"]+)"', prompt)
    if len(names) < 3:
        names = re.findall(r"«([^»]+)»", prompt)
    if len(names) < 3:
        return False
    names = [n.strip() for n in names[:3] if n.strip()]
    if len(names) < 3:
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

    for i, name in enumerate(names):
        if name.lower() in existing_lower:
            continue
        # Unique departmentNumber in this company
        for cand in range(5000 + i * 17, 99999, 3):
            s = str(cand)
            if s not in used_numbers:
                dept_num = s
                break
        else:
            return False

        body = {"name": name, "departmentNumber": dept_num}
        r = client.post("/department", json=body)
        if r.status_code not in (200, 201):
            return False
        used_numbers.add(dept_num)
        existing_lower.add(name.lower())

    print("[task_handler:departments] ensured 3 departments exist (see logs for writes)")
    return True
