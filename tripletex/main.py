from __future__ import annotations

import tempfile
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .agent import solve
from .structured_log import log_event, set_request_id

app = FastAPI(title="NM i AI — Tripletex Agent", version="1.0.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/solve")
async def solve_endpoint(request: Request) -> JSONResponse:
    request_id = str(uuid.uuid4())
    set_request_id(request_id)

    body = await request.json()
    prompt = body.get("prompt") or ""
    files = body.get("files") or []
    creds = body.get("tripletex_credentials") or {}
    if not isinstance(files, list):
        files = []
    if not isinstance(creds, dict) or "base_url" not in creds or "session_token" not in creds:
        log_event("WARNING", "solve_bad_request", detail="Missing tripletex_credentials")
        return JSONResponse(
            {"status": "error", "detail": "Missing tripletex_credentials.base_url or session_token"},
            status_code=400,
        )

    log_event(
        "INFO",
        "solve_start",
        prompt_len=len(prompt),
        file_count=len(files),
        file_names=[f.get("filename", "?") for f in files],
    )

    with tempfile.TemporaryDirectory(prefix="ttx_") as tmp:
        work = Path(tmp)
        try:
            solve(str(prompt), files, creds, work)
        except Exception as exc:
            log_event("ERROR", "solve_exception", error=str(exc))
            traceback.print_exc()
            return JSONResponse({"status": "error", "detail": str(exc)}, status_code=500)

    log_event("INFO", "solve_completed")
    return JSONResponse({"status": "completed"})
