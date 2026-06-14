"""Queue state machine: illegal transitions rejected; agent cannot resolve."""
from __future__ import annotations

import pytest

from src.audit.log import AuditLog
from src.orchestrator.queue import (
    AWAITING_DECISION, InvalidTransition, OPEN, RESOLVED_BLOCK, ReviewQueue,
)


def _queue(store):
    return ReviewQueue(store, AuditLog(store))


_counter = [0]


def _a_case(store, status=OPEN):
    _counter[0] += 1
    cid = f"tc_{status}_{_counter[0]}"
    _queue(store).create_case(case_id=cid, subject_type="user", subject_id="u00001",
                              status=status, model_score=0.5, rule_codes=[],
                              graph_signals={})
    return cid


def test_agent_cannot_resolve(store):
    q = _queue(store)
    cid = _a_case(store, OPEN)
    with pytest.raises(InvalidTransition):
        q.resolve(cid, "block", actor="agent:investigator")


def test_human_can_resolve(store):
    q = _queue(store)
    cid = _a_case(store, AWAITING_DECISION)
    assert q.resolve(cid, "block", actor="human:alice") == RESOLVED_BLOCK


def test_illegal_transition_rejected(store):
    q = _queue(store)
    cid = _a_case(store, RESOLVED_BLOCK)
    with pytest.raises(InvalidTransition):
        q.assign(cid, "bob", actor="human:bob")  # terminal -> nothing allowed


def test_agent_may_recommend(store):
    q = _queue(store)
    cid = _a_case(store, OPEN)
    assert q.mark_awaiting(cid, actor="agent:investigator") == AWAITING_DECISION
