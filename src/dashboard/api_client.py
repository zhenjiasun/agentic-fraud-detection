"""Thin HTTP client to the FastAPI backend. Dash never touches the DB directly."""
from __future__ import annotations

import os

import requests

BASE = os.environ.get("FRAUDGUARD_API", "http://127.0.0.1:8000")
TIMEOUT = 60


def get(path: str, **params):
    r = requests.get(f"{BASE}{path}", params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def post(path: str, json: dict | None = None):
    r = requests.post(f"{BASE}{path}", json=json or {}, timeout=TIMEOUT)
    return r.status_code, (r.json() if r.content else {})
