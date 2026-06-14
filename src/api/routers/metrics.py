"""Evaluation + monitoring metric endpoints (cached per API-process lifetime)."""
from __future__ import annotations

from fastapi import APIRouter

from src.analysis.drift import monitoring_snapshot
from src.analysis.evaluation import build_report
from src.api.deps import settings, store

router = APIRouter()

_cache: dict = {}


@router.get("/metrics/evaluation")
def evaluation(refresh: bool = False):
    if refresh or "evaluation" not in _cache:
        _cache["evaluation"] = build_report(store(), settings())
    return _cache["evaluation"]


@router.get("/metrics/monitoring")
def monitoring(refresh: bool = False):
    if refresh or "monitoring" not in _cache:
        _cache["monitoring"] = monitoring_snapshot(store(), settings())
    return _cache["monitoring"]


@router.get("/metrics/overview")
def overview():
    s = store()
    txn = s.query_df("SELECT COUNT(*) n FROM transactions").iloc[0]["n"]
    fraud = s.query_df("SELECT COUNT(*) n FROM transactions WHERE is_fraud_gt=1").iloc[0]["n"]
    actions = s.query_df(
        "SELECT action, COUNT(*) n FROM decisions WHERE source='orchestrator' GROUP BY action"
    ).set_index("action")["n"].to_dict()
    cases = s.query_df("SELECT status, COUNT(*) n FROM cases GROUP BY status"
                       ).set_index("status")["n"].to_dict()
    return {
        "transactions": int(txn),
        "fraud_rate": round(float(fraud) / max(1, int(txn)), 4),
        "actions": actions,
        "cases": cases,
    }
