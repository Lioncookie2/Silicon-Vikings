"""
Astar Island API client — https://api.ainm.no/astar-island/

Authentication: JWT from app.ainm.no (cookie `access_token` or Authorization: Bearer).
Also: ``ACCESS_TOKEN`` in environment, or ``.env`` in repo root (``python-dotenv``).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv_files() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # Kun `.env` lastes — ikke `.env.example` (mal uten hemmeligheter).
    for path in (_REPO_ROOT / ".env", Path.cwd() / ".env"):
        if path.is_file():
            load_dotenv(path, override=False)
            break


_load_dotenv_files()

BASE_URL = os.environ.get("AINM_API_BASE", "https://api.ainm.no")


class AstarClient:
    def __init__(self, access_token: str | None = None) -> None:
        self.token = access_token or os.environ.get("ACCESS_TOKEN") or ""
        if not self.token:
            env_path = _REPO_ROOT / ".env"
            hint = (
                f"Legg JWT i {env_path} som ACCESS_TOKEN=... (filen lastes automatisk). "
                "Kun `.env` brukes — ikke `.env.example`. Opprett med: cp .env.example .env"
            )
            if (_REPO_ROOT / ".env.example").is_file() and not env_path.is_file():
                hint += f" (mangler {env_path})"
            raise ValueError(hint + " Eller: export ACCESS_TOKEN=...")
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {self.token}"

    def list_rounds(self) -> list[dict[str, Any]]:
        r = self.session.get(f"{BASE_URL}/astar-island/rounds", timeout=60)
        r.raise_for_status()
        return r.json()

    def get_round(self, round_id: str) -> dict[str, Any]:
        r = self.session.get(f"{BASE_URL}/astar-island/rounds/{round_id}", timeout=60)
        r.raise_for_status()
        return r.json()

    def my_rounds(self) -> list[dict[str, Any]]:
        r = self.session.get(f"{BASE_URL}/astar-island/my-rounds", timeout=60)
        r.raise_for_status()
        return r.json()

    def simulate(
        self,
        *,
        round_id: str,
        seed_index: int,
        viewport_x: int,
        viewport_y: int,
        viewport_w: int,
        viewport_h: int,
    ) -> dict[str, Any]:
        payload = {
            "round_id": round_id,
            "seed_index": seed_index,
            "viewport_x": viewport_x,
            "viewport_y": viewport_y,
            "viewport_w": viewport_w,
            "viewport_h": viewport_h,
        }
        r = self.session.post(f"{BASE_URL}/astar-island/simulate", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()

    def submit_prediction(
        self,
        *,
        round_id: str,
        seed_index: int,
        prediction: list[list[list[float]]],
    ) -> requests.Response:
        """prediction: height x width x 6 probabilities per cell."""
        payload = {
            "round_id": round_id,
            "seed_index": seed_index,
            "prediction": prediction,
        }
        return self.session.post(f"{BASE_URL}/astar-island/submit", json=payload, timeout=120)

    def get_analysis(self, round_id: str, seed_index: int) -> dict[str, Any]:
        """
        Fetch post-round analysis for one seed.
        Raises for non-2xx responses.
        """
        r = self.session.get(f"{BASE_URL}/astar-island/analysis/{round_id}/{seed_index}", timeout=120)
        r.raise_for_status()
        return r.json()


def resolve_round_identifier(client: AstarClient, identifier: str) -> str:
    """
    API-et bruker UUID som round_id. Streng som *kun* er siffer tolkes som ``round_number``
    fra ``list_rounds()``; ved flere treff velges først ``status == "active"``.
    """
    t = str(identifier).strip()
    if not t.isdigit():
        return t
    n = int(t)
    rounds = client.list_rounds()
    matches = [r for r in rounds if int(r.get("round_number", -(10**9))) == n]
    if not matches:
        raise ValueError(
            f"Ingen runde med round_number={n} i /astar-island/rounds. "
            "Bruk full UUID som --round-id, eller sjekk innlogging."
        )
    actives = [r for r in matches if r.get("status") == "active"]
    chosen = actives[0] if actives else matches[0]
    return str(chosen["id"])


def get_active_round(client: AstarClient) -> tuple[str, dict[str, Any]]:
    """Return (round_id, round_detail) for the first round with status ``active``."""
    rounds = client.list_rounds()
    for r in rounds:
        if r.get("status") == "active":
            rid = str(r["id"])
            return rid, client.get_round(rid)
    raise RuntimeError("No active Astar Island round")


# Backwards compatibility
get_active_round_id = get_active_round
