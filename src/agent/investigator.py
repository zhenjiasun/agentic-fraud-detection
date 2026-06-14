"""Investigator agent: a bounded tool loop that gathers read-only evidence and
submits one validated recommendation per case.

Boundedness is enforced structurally, not by prompting:
- only read tools + the submit_finding output channel are offered
- any call to a forbidden (write) tool name is blocked and audited
- the finding is validated against InvestigationResult; invalid output (bad
  enum, hallucinated evidence ids) is rejected and never reaches the queue
- the agent transitions a case only to AWAITING_DECISION (a recommendation);
  it can never resolve a case (enforced again in queue.py)
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from src.agent.guards import detect_injection
from src.agent.llm_client import SUBMIT_FINDING_TOOL, get_client
from src.agent.schema import InvestigationResult
from src.agent.tools import FORBIDDEN_TOOL_NAMES, ToolRegistry
from src.audit.log import open_audit
from src.config import ROOT
from src.log import get_logger
from src.orchestrator.queue import ReviewQueue

log = get_logger("investigator")

ACTOR = "agent:investigator"


def _system_prompt() -> str:
    return (ROOT / "prompts" / "investigator_system.md").read_text()


def _ref_exists(store, ref: str) -> bool:
    for table, col in (("transactions", "txn_id"), ("cases", "case_id"),
                       ("users", "user_id"), ("merchants", "merchant_id")):
        if store.get_row(table, col, ref):
            return True
    return False


def investigate_case(store, settings, case_id: str, audit=None, queue=None) -> dict:
    case = store.get_case(case_id)
    if not case:
        return {"case_id": case_id, "status": "error", "reason": "no such case"}

    audit = audit or open_audit(settings, store)
    queue = queue or ReviewQueue(store, audit)
    client = get_client(settings)
    registry = ToolRegistry(store, case)
    tools = registry.specs() + [SUBMIT_FINDING_TOOL]
    system = _system_prompt()

    messages = [{
        "role": "user",
        "content": (f"Investigate case {case_id} (subject user {case['subject_id']}). "
                    f"Gather evidence with the read tools, then call submit_finding."),
    }]

    max_calls = int(settings.agent.get("max_tool_calls", 8))
    tool_calls_log: list[dict] = []
    injection_flags: list[str] = []
    finding: InvestigationResult | None = None
    raw_finding: dict | None = None
    rejected_reason: str | None = None

    for _ in range(max_calls):
        resp = client.complete(system, messages, tools)
        if resp.stop_reason != "tool_use" or not resp.tool_calls:
            # model answered without using the finding tool; nudge once then stop
            messages.append({"role": "assistant", "text": resp.text})
            messages.append({"role": "user",
                             "content": "Call submit_finding to conclude."})
            continue

        # process tool calls
        submit = next((c for c in resp.tool_calls if c.name == "submit_finding"), None)
        if submit is not None:
            raw_finding = submit.input
            try:
                finding = InvestigationResult(**submit.input)
            except ValidationError as e:
                rejected_reason = f"schema validation failed: {e.errors()[:2]}"
                break
            # reject hallucinated evidence ids (refs that look like ids but don't exist)
            bad = [r for r in finding.evidence_refs
                   if _looks_like_id(r) and not _ref_exists(store, r)]
            if bad:
                rejected_reason = f"hallucinated evidence_refs: {bad}"
                finding = None
            break

        messages.append({"role": "assistant", "text": resp.text,
                         "tool_calls": resp.tool_calls})
        results = []
        for call in resp.tool_calls:
            if call.name in FORBIDDEN_TOOL_NAMES:
                # structurally impossible (not offered) but audited if ever attempted
                audit.record(actor=ACTOR, action_type="TOOL_ABUSE_BLOCKED",
                             subject_type="user", subject_id=case["subject_id"],
                             payload={"case_id": case_id, "attempted_tool": call.name})
                results.append({"tool_call_id": call.id,
                                "content": json.dumps({"error": "forbidden tool"})})
                continue
            out = registry.execute(call.name, call.input)
            tool_calls_log.append({"name": call.name, "input": call.input})
            results.append({"tool_call_id": call.id, "content": out})
        injection_flags += detect_injection(json.dumps([r["content"] for r in results]))
        messages.append({"role": "tool_results", "results": results})

    injection_flags = sorted(set(injection_flags + registry.injection_flags))

    # persist the investigation (success or rejection) for the audit trail
    store.insert_investigation(
        case_id=case_id, ts=case.get("opened_at", ""),
        provider=client.provider, model=client.model,
        disposition=(finding.disposition.value if finding else "REJECTED"),
        confidence=(finding.confidence if finding else 0.0),
        rationale=(finding.rationale if finding else (rejected_reason or "no finding")),
        evidence_json=json.dumps(finding.evidence_refs if finding else []),
        tool_calls_json=json.dumps(tool_calls_log),
        injection_flags_json=json.dumps(injection_flags),
    )
    audit.record(actor=ACTOR, action_type="INVESTIGATION_RUN",
                 subject_type="user", subject_id=case["subject_id"],
                 payload={"case_id": case_id, "provider": client.provider,
                          "disposition": finding.disposition.value if finding else "REJECTED",
                          "n_tool_calls": len(tool_calls_log),
                          "injection_flags": injection_flags})
    if injection_flags:
        audit.record(actor=ACTOR, action_type="INJECTION_FLAGGED",
                     subject_type="user", subject_id=case["subject_id"],
                     payload={"case_id": case_id, "flags": injection_flags})

    # record the recommendation (NON-terminal transition; agent cannot resolve)
    if finding is not None and case["status"] in ("OPEN", "IN_REVIEW"):
        queue.mark_awaiting(case_id, actor=ACTOR)

    return {
        "case_id": case_id,
        "disposition": finding.disposition.value if finding else "REJECTED",
        "confidence": finding.confidence if finding else 0.0,
        "injection_flags": injection_flags,
        "rejected_reason": rejected_reason,
        "tool_calls": tool_calls_log,
        "raw_finding": raw_finding,
    }


def _looks_like_id(ref: str) -> bool:
    return any(ref.startswith(p) for p in ("u", "t", "c", "m", "d", "ip", "id", "case_"))


def investigate_open_cases(store, settings, limit: int = 15) -> dict:
    audit = open_audit(settings, store)
    queue = ReviewQueue(store, audit)
    open_cases = store.list_cases(status="OPEN", limit=limit)
    results = [investigate_case(store, settings, c["case_id"], audit, queue)
               for c in open_cases]
    dispositions: dict[str, int] = {}
    for r in results:
        dispositions[r["disposition"]] = dispositions.get(r["disposition"], 0) + 1
    return {
        "investigated": len(results),
        "dispositions": dispositions,
        "with_injection_flags": sum(1 for r in results if r["injection_flags"]),
        "provider": settings.llm.provider,
    }
