"""Shared singletons for the API process (store, audit, queue, settings)."""
from __future__ import annotations

from functools import lru_cache

from src.audit.log import open_audit
from src.config import get_settings
from src.data.store import Store
from src.orchestrator.queue import ReviewQueue


@lru_cache(maxsize=1)
def settings():
    return get_settings()


@lru_cache(maxsize=1)
def store() -> Store:
    st = Store(settings().db_path)
    st.init_schema()
    return st


def audit():
    return open_audit(settings(), store())


def queue() -> ReviewQueue:
    return ReviewQueue(store(), audit())
