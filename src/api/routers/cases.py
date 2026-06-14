"""Case queue endpoints. /resolve is the only money-affecting write and requires
a human actor; /investigate produces a recommendation only."""
from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException

from src.api.deps import queue, settings, store
from src.orchestrator.queue import InvalidTransition

router = APIRouter()


@router.get("/cases")
def list_cases(status: str | None = None, limit: int = 200):
    return store().list_cases(status=status, limit=limit)


@router.get("/cases/{case_id}")
def get_case(case_id: str):
    case = store().get_case(case_id)
    if not case:
        raise HTTPException(404, "not found")
    inv = store().query_df(
        "SELECT * FROM investigations WHERE case_id=? ORDER BY investigation_id DESC",
        (case_id,),
    ).to_dict("records")
    decisions = store().query_df(
        "SELECT * FROM decisions WHERE case_id=? OR subject_id=? ORDER BY decision_id DESC LIMIT 50",
        (case_id, case["subject_id"]),
    ).to_dict("records")
    return {"case": case, "investigations": inv, "decisions": decisions}


@router.post("/cases/{case_id}/assign")
def assign_case(case_id: str, reviewer: str = Body(..., embed=True)):
    try:
        status = queue().assign(case_id, reviewer, actor=f"human:{reviewer}")
    except InvalidTransition as e:
        raise HTTPException(409, str(e))
    return {"case_id": case_id, "status": status}


@router.post("/cases/{case_id}/resolve")
def resolve_case(case_id: str, decision: str = Body(..., embed=True),
                 actor: str = Body(..., embed=True)):
    if decision not in ("block", "allow"):
        raise HTTPException(400, "decision must be 'block' or 'allow'")
    if not actor.startswith("human:"):
        raise HTTPException(403, "resolve requires a human actor (human:<name>)")
    try:
        status = queue().resolve(case_id, decision, actor=actor)
    except InvalidTransition as e:
        raise HTTPException(409, str(e))
    return {"case_id": case_id, "status": status}


@router.post("/cases/{case_id}/investigate")
def investigate(case_id: str):
    from src.agent.investigator import investigate_case
    result = investigate_case(store(), settings(), case_id)
    return result
