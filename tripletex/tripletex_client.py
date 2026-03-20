"""
Minimal Tripletex v2 API client via competition proxy.

Auth: HTTP Basic with username "0" and session_token as password.
"""
from __future__ import annotations

from typing import Any

import requests


class TripletexClient:
    def __init__(self, base_url: str, session_token: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth = ("0", session_token)
        self.timeout = timeout

    def get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> requests.Response:
        url = f"{self.base_url}{path if path.startswith('/') else '/' + path}"
        return requests.get(url, auth=self.auth, params=params, timeout=self.timeout)

    def post(self, path: str, *, json: dict[str, Any] | None = None) -> requests.Response:
        url = f"{self.base_url}{path if path.startswith('/') else '/' + path}"
        return requests.post(url, auth=self.auth, json=json, timeout=self.timeout)

    def put(self, path: str, *, json: dict[str, Any] | None = None) -> requests.Response:
        url = f"{self.base_url}{path if path.startswith('/') else '/' + path}"
        return requests.put(url, auth=self.auth, json=json, timeout=self.timeout)

    def delete(self, path: str) -> requests.Response:
        url = f"{self.base_url}{path if path.startswith('/') else '/' + path}"
        return requests.delete(url, auth=self.auth, timeout=self.timeout)
