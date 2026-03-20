"""
LLM agent: parse accounting task prompt -> structured API plan -> execute with TripletexClient.

Set one of: GEMINI_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY (optional: OPENAI_BASE_URL).
"""
from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any

from .tripletex_client import TripletexClient


SYSTEM_PROMPT = """You are an expert Tripletex accounting API agent for NM i AI 2026.
The user message contains a task in one of several languages (Norwegian, English, etc.).
You must output ONLY valid JSON (no markdown fences) with this shape:
{
  "reasoning": "short internal plan",
  "calls": [
    {"method": "GET|POST|PUT|DELETE", "path": "/employee", "params": null, "json": null},
    ...
  ]
}
Rules:
- Paths are Tripletex REST v2 paths relative to base_url (e.g. /employee, /customer, /invoice).
- Order calls so prerequisites exist (e.g. create customer before invoice if needed).
- Use fields query param on GET when listing, e.g. params: {"fields": "id,firstName,lastName"}
- For POST/PUT, put body in "json".
- Minimize the number of calls. Avoid speculative GETs.
- If the task references attached files, the prompt may include extracted filenames — infer required data from the task text.
"""


def _call_gemini(prompt: str) -> str:
    import google.generativeai as genai

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY")
    genai.configure(api_key=key)
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content([SYSTEM_PROMPT, prompt])
    return resp.text or ""


def _call_openai(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    r = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    return r.choices[0].message.content or ""


def _llm_complete(user_prompt: str) -> str:
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return _call_gemini(user_prompt)
    if os.environ.get("OPENAI_API_KEY"):
        return _call_openai(user_prompt)
    raise RuntimeError(
        "No LLM API key: set GEMINI_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY"
    )


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"No JSON object in LLM output: {text[:500]}")
    return json.loads(m.group(0))


def _save_files(files: list[dict[str, str]], workdir: Path) -> list[str]:
    saved: list[str] = []
    workdir.mkdir(parents=True, exist_ok=True)
    for f in files:
        name = f.get("filename") or "attachment.bin"
        content_b64 = f.get("content_base64") or ""
        raw = base64.b64decode(content_b64)
        path = workdir / name
        path.write_bytes(raw)
        saved.append(str(path))
    return saved


def execute_plan(client: TripletexClient, plan: dict[str, Any]) -> None:
    calls = plan.get("calls") or []
    if not isinstance(calls, list):
        raise ValueError("Invalid plan: calls must be a list")

    for step in calls:
        method = (step.get("method") or "GET").upper()
        path = step.get("path") or "/"
        params = step.get("params")
        body = step.get("json")

        if method == "GET":
            r = client.get(path, params=params if isinstance(params, dict) else None)
        elif method == "POST":
            r = client.post(path, json=body if isinstance(body, dict) else None)
        elif method == "PUT":
            r = client.put(path, json=body if isinstance(body, dict) else None)
        elif method == "DELETE":
            r = client.delete(path)
        else:
            raise ValueError(f"Unsupported method: {method}")

        if r.status_code >= 400:
            raise RuntimeError(
                f"Tripletex API error {r.status_code} on {method} {path}: {r.text[:2000]}"
            )


def build_user_prompt(
    prompt: str,
    file_paths: list[str],
    tripletex_credentials: dict[str, str],
) -> str:
    parts = [
        "TASK PROMPT:\n" + prompt,
        f"base_url (for reference only): {tripletex_credentials.get('base_url', '')}",
    ]
    if file_paths:
        parts.append("Saved attachment paths on server:\n" + "\n".join(file_paths))
    return "\n\n".join(parts)


def solve(
    prompt: str,
    files: list[dict[str, str]],
    tripletex_credentials: dict[str, str],
    workdir: Path,
) -> None:
    base_url = tripletex_credentials["base_url"]
    token = tripletex_credentials["session_token"]
    client = TripletexClient(base_url, token)

    saved = _save_files(files, workdir)
    user_text = build_user_prompt(prompt, saved, tripletex_credentials)
    raw = _llm_complete(user_text)
    plan = _extract_json(raw)
    execute_plan(client, plan)

