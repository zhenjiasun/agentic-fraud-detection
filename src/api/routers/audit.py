"""Audit trail endpoints + hash-chain verification."""
from __future__ import annotations

from fastapi import APIRouter

from src.api.deps import store
from src.audit.query import query_audit, verify_chain

router = APIRouter()


@router.get("/audit")
def get_audit(subject_id: str | None = None, action_type: str | None = None,
              actor: str | None = None, limit: int = 200):
    df = query_audit(store(), subject_id=subject_id, action_type=action_type,
                     actor=actor, limit=limit)
    return df.to_dict("records")


@router.get("/audit/verify")
def audit_verify():
    return verify_chain(store())
