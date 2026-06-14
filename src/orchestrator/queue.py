"""Human-review queue state machine.

Transitions are the ONLY way to mutate case state; each writes a decision + an
audit row. The LLM investigator may only move a case to AWAITING_DECISION
(a recommendation). Money-affecting terminal states (RESOLVED_ALLOW/
RESOLVED_BLOCK) are reachable only by a human actor (or a high-confidence rule
acting through the orchestrator) — never by the agent. This boundary is enforced
here, independent of anything the agent does or any prompt it sees.
"""
from __future__ import annotations

import json
from datetime import datetime

OPEN = "OPEN"
IN_REVIEW = "IN_REVIEW"
AWAITING_DECISION = "AWAITING_DECISION"
RESOLVED_ALLOW = "RESOLVED_ALLOW"
RESOLVED_BLOCK = "RESOLVED_BLOCK"
ESCALATED = "ESCALATED"

TERMINAL_MONEY_STATES = {RESOLVED_ALLOW, RESOLVED_BLOCK}

ALLOWED = {
    # OPEN -> AWAITING_DECISION is the investigator recording a recommendation
    # on a not-yet-assigned case (still non-terminal; humans resolve from there).
    OPEN: {IN_REVIEW, AWAITING_DECISION, ESCALATED, RESOLVED_ALLOW, RESOLVED_BLOCK},
    IN_REVIEW: {AWAITING_DECISION, OPEN, RESOLVED_ALLOW, RESOLVED_BLOCK, ESCALATED},
    AWAITING_DECISION: {RESOLVED_ALLOW, RESOLVED_BLOCK, ESCALATED, IN_REVIEW},
    ESCALATED: {RESOLVED_ALLOW, RESOLVED_BLOCK},
    RESOLVED_ALLOW: set(),
    RESOLVED_BLOCK: set(),
}


class InvalidTransition(Exception):
    pass


class ReviewQueue:
    def __init__(self, store, audit):
        self.store = store
        self.audit = audit

    def create_case(self, *, case_id, subject_type, subject_id, status, model_score,
                    rule_codes, graph_signals, disposition=None):
        self.store.create_case({
            "case_id": case_id, "subject_type": subject_type, "subject_id": subject_id,
            "opened_at": datetime.now().isoformat(), "status": status,
            "model_score": float(model_score),
            "rule_codes_json": json.dumps(rule_codes),
            "graph_signals_json": json.dumps(graph_signals),
            "assigned_to": None, "resolved_at": None, "disposition": disposition,
        })

    def _transition(self, case_id, to_state, actor, *, disposition=None,
                    assigned_to=None, payload=None):
        case = self.store.get_case(case_id)
        if not case:
            raise InvalidTransition(f"no such case {case_id}")
        frm = case["status"]
        if to_state not in ALLOWED.get(frm, set()):
            raise InvalidTransition(f"{frm} -> {to_state} not allowed")
        # The agent can never reach a money-affecting terminal state.
        if actor.startswith("agent") and to_state in TERMINAL_MONEY_STATES:
            raise InvalidTransition("agent may not resolve a case (bounded action)")

        fields = {"status": to_state}
        if assigned_to is not None:
            fields["assigned_to"] = assigned_to
        if to_state in TERMINAL_MONEY_STATES or to_state == ESCALATED:
            fields["resolved_at"] = datetime.now().isoformat()
            fields["disposition"] = disposition or to_state
        self.store.update_case(case_id, **fields)

        action_map = {RESOLVED_ALLOW: "HUMAN_RESOLVE", RESOLVED_BLOCK: "HUMAN_RESOLVE",
                      IN_REVIEW: "CASE_ASSIGNED", ESCALATED: "HUMAN_RESOLVE",
                      AWAITING_DECISION: "INVESTIGATION_RUN"}
        self.audit.record(
            actor=actor, action_type=action_map.get(to_state, "ROUTE_TO_REVIEW"),
            subject_type=case["subject_type"], subject_id=case["subject_id"],
            payload={"case_id": case_id, "from": frm, "to": to_state, **(payload or {})},
        )
        return to_state

    # convenience wrappers
    def assign(self, case_id, reviewer, actor="human:reviewer"):
        return self._transition(case_id, IN_REVIEW, actor, assigned_to=reviewer)

    def mark_awaiting(self, case_id, actor):
        # used by the investigator agent after producing a recommendation
        return self._transition(case_id, AWAITING_DECISION, actor)

    def resolve(self, case_id, decision, actor):
        to = RESOLVED_BLOCK if decision == "block" else RESOLVED_ALLOW
        return self._transition(case_id, to, actor, disposition=to)

    def escalate(self, case_id, actor):
        return self._transition(case_id, ESCALATED, actor)
