"""
Astar Island API client — https://api.ainm.no/astar-island/

Authentication: JWT from app.ainm.no (cookie `access_token` or Authorization: Bearer).
Set environment variable ACCESS_TOKEN or pass token= to AstarClient.
"""
from __future__ import annotations

import os
from typing import Any

import requests

BASE_URL = os.environ.get("AINM_API_BASE", "https://api.ainm.no")


class AstarClient:
    def __init__(self, access_token: str | None = None) -> None:
        self.token = access_token or os.environ.get("ACCESS_TOKEN") or ""
        if not self.token:
            raise ValueError("Set ACCESS_TOKEN (JWT from browser) or pass access_token=")
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {self.token}"

    def list_rounds(self) -> list[dict[str, Any]]:
        r = self.session.get(f"{BASE_URL}/astar-island/rounds", timeout=60)
        r.raise_for_status()
        return r.json()

    def get_round(self, round_id: int) -> dict[str, Any]:
        r = self.session.get(f"{BASE_URL}/astar-island/rounds/{round_id}", timeout=60)
        r.raise_for_status()
        return r.json()

    def simulate(
        self,
        *,
        round_id: int,
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
        round_id: int,
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


def get_active_round_id(client: AstarClient) -> tuple[int, dict[str, Any]]:
    rounds = client.list_rounds()
    for r in rounds:
        if r.get("status") == "active":
            rid = int(r["id"])
            return rid, client.get_round(rid)
    raise RuntimeError("No active Astar Island round")
