"""
Minimal Tripletex v2 API client via competition proxy.

Auth: HTTP Basic with username "0" and session_token as password.
"""
from __future__ import annotations

from typing import Any

import requests


from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class TripletexClient:
    def __init__(self, base_url: str, session_token: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth = ("0", session_token)
        self.timeout = timeout
        
        # Configure robust session with retries for common server/network errors
        self.session = requests.Session()
        self.session.auth = self.auth
        retries = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"]
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> requests.Response:
        url = f"{self.base_url}{path if path.startswith('/') else '/' + path}"
        return self.session.get(url, params=params, timeout=self.timeout)

    def post(self, path: str, *, json: dict[str, Any] | None = None) -> requests.Response:
        url = f"{self.base_url}{path if path.startswith('/') else '/' + path}"
        return self.session.post(url, json=json, timeout=self.timeout)

    def put(self, path: str, *, json: dict[str, Any] | None = None) -> requests.Response:
        url = f"{self.base_url}{path if path.startswith('/') else '/' + path}"
        return self.session.put(url, json=json, timeout=self.timeout)

    def delete(self, path: str) -> requests.Response:
        url = f"{self.base_url}{path if path.startswith('/') else '/' + path}"
        return self.session.delete(url, timeout=self.timeout)
