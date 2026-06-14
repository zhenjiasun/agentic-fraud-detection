"""Entity, transaction, and ring read endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.deps import settings, store
from src.graph.builder import build_graph
from src.graph.rings import detect_rings

router = APIRouter()

_ENTITY_TABLES = {
    "user": ("users", "user_id"), "merchant": ("merchants", "merchant_id"),
    "device": ("devices", "device_id"), "ip": ("ips", "ip_id"),
    "card": ("cards", "card_id"), "identity": ("identities", "identity_id"),
}

# rings are cheap-ish but rebuild the graph; cache for the API process lifetime
_rings_cache: list[dict] | None = None


@router.get("/entities/{etype}/{eid}")
def get_entity(etype: str, eid: str):
    if etype not in _ENTITY_TABLES:
        raise HTTPException(404, f"unknown entity type {etype}")
    table, key = _ENTITY_TABLES[etype]
    row = store().get_row(table, key, eid)
    if not row:
        raise HTTPException(404, "not found")
    return row


@router.get("/transactions")
def list_transactions(user_id: str | None = None, limit: int = 100):
    if user_id:
        return store().recent_transactions(user_id, limit)
    return store().query_df(
        "SELECT * FROM transactions ORDER BY ts DESC LIMIT ?", (limit,)
    ).to_dict("records")


@router.get("/transactions/{txn_id}")
def get_transaction(txn_id: str):
    row = store().get_row("transactions", "txn_id", txn_id)
    if not row:
        raise HTTPException(404, "not found")
    return row


def _rings():
    global _rings_cache
    if _rings_cache is None:
        g = build_graph(store())
        _rings_cache = detect_rings(g, settings(), store=store())
    return _rings_cache


@router.get("/rings")
def list_rings():
    return sorted(_rings(), key=lambda r: -r["risk_score"])


@router.get("/rings/{ring_id}")
def get_ring(ring_id: str):
    for r in _rings():
        if r["ring_id"] == ring_id:
            return r
    raise HTTPException(404, "not found")
