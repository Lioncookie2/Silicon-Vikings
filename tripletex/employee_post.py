"""
POST /employee with userType retry strategies (shared by agent loop and task handlers).
"""
from __future__ import annotations

import json
from typing import Any

import requests as _requests

from .tripletex_client import TripletexClient


def employee_post_usertype_422(body_text: Any) -> bool:
    """True if error payload suggests userType mapping/type failure on POST /employee."""
    if isinstance(body_text, dict):
        blob = json.dumps(body_text, ensure_ascii=False).lower()
    else:
        blob = str(body_text).lower()
    if "usertype" not in blob:
        return False
    return (
        "korrekt type" in blob
        or "correct type" in blob
        or "ikke av korrekt type" in blob
        or "mapping failed" in blob
        or "request mapping failed" in blob
    )


def execute_post_employee(
    client: TripletexClient, path: str, body: dict[str, Any] | None
) -> _requests.Response:
    """POST /employee only. Retries without userType and with userType {id} from GET /employee."""
    if body is None:
        return client.post(path, json=body)
    resp = client.post(path, json=body)
    if resp.status_code < 400:
        return resp
    if resp.status_code != 422:
        return resp
    try:
        err_json = resp.json()
    except Exception:
        err_json = {}
    if not employee_post_usertype_422(err_json):
        return resp

    if "userType" in body:
        stripped = {k: v for k, v in body.items() if k != "userType"}
        r2 = client.post(path, json=stripped)
        if r2.status_code < 400:
            return r2
        resp = r2
        try:
            err_json = r2.json()
        except Exception:
            err_json = {}
        if not employee_post_usertype_422(err_json):
            return r2

    try:
        gr = client.get("/employee", params={"fields": "id,userType(id)", "count": 30})
        if gr.status_code != 200:
            return resp
        data = gr.json()
        base = {k: v for k, v in body.items() if k != "userType"}
        for emp in data.get("values") or []:
            ut = emp.get("userType")
            uid: int | None = None
            if isinstance(ut, dict) and ut.get("id") is not None:
                try:
                    uid = int(ut["id"])
                except (TypeError, ValueError):
                    uid = None
            elif isinstance(ut, int):
                uid = ut
            if uid is None:
                continue
            r3 = client.post(path, json={**base, "userType": {"id": uid}})
            if r3.status_code < 400:
                return r3
            resp = r3

        if resp.status_code >= 400:
            try:
                ej = resp.json()
            except Exception:
                ej = {}
            if employee_post_usertype_422(ej):
                for guess_id in (1, 2, 3, 4, 5):
                    rx = client.post(path, json={**base, "userType": {"id": guess_id}})
                    if rx.status_code < 400:
                        return rx
                    try:
                        ej2 = rx.json()
                    except Exception:
                        ej2 = {}
                    if not employee_post_usertype_422(ej2):
                        return rx
                    resp = rx
    except Exception:
        pass
    return resp
