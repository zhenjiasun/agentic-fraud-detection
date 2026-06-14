"""The investigator is bounded by construction: no write tools, closed-enum
output, no store mutations from tool execution, hallucinated evidence rejected.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.agent.schema import Disposition, InvestigationResult
from src.agent.tools import FORBIDDEN_TOOL_NAMES, ToolRegistry


def _a_case(store):
    c = store.list_cases(limit=1)[0]
    return c


def test_no_write_tools_offered(store):
    reg = ToolRegistry(store, _a_case(store))
    assert reg.tool_names.isdisjoint(FORBIDDEN_TOOL_NAMES)
    # every offered tool is a known read tool
    assert reg.tool_names <= {"get_case", "get_account_profile", "get_transaction",
                              "get_recent_transactions", "get_graph_neighborhood",
                              "get_model_explanation"}


def test_tools_never_mutate_store(store, monkeypatch):
    reg = ToolRegistry(store, _a_case(store))
    writes = []
    for m in ("insert_decision", "insert_investigation", "create_case",
              "update_case", "insert_df", "execute"):
        monkeypatch.setattr(store, m,
                            lambda *a, _m=m, **k: writes.append(_m))
    for name in reg.tool_names:
        reg.execute(name, {})
    assert writes == [], f"read tools triggered writes: {writes}"


def test_out_of_enum_disposition_rejected():
    with pytest.raises(ValidationError):
        InvestigationResult(disposition="DEFINITELY_FINE", confidence=0.9,
                            rationale="x", evidence_refs=[])
    # valid one parses
    r = InvestigationResult(disposition=Disposition.LIKELY_FRAUD, confidence=0.9,
                            rationale="x", evidence_refs=[])
    assert r.disposition == Disposition.LIKELY_FRAUD


def test_confidence_bounds_enforced():
    with pytest.raises(ValidationError):
        InvestigationResult(disposition="ESCALATE", confidence=1.5,
                            rationale="x", evidence_refs=[])


def test_hallucinated_evidence_rejected(settings, store, monkeypatch):
    """A finding citing a non-existent txn id is rejected, not queued."""
    from src.agent import investigator
    from src.agent.llm_client import LLMResponse, ToolCall

    class HallucinatingClient:
        provider, model = "mock", "halluc"

        def complete(self, system, messages, tools):
            if not any(m["role"] == "tool_results" for m in messages):
                return LLMResponse(stop_reason="tool_use",
                                   tool_calls=[ToolCall("h1", "get_case", {})])
            return LLMResponse(stop_reason="tool_use", tool_calls=[ToolCall(
                "h2", "submit_finding", {"disposition": "LIKELY_FRAUD", "confidence": 0.9,
                 "rationale": "made up", "evidence_refs": ["t99999999"]})])

    monkeypatch.setattr(investigator, "get_client", lambda s: HallucinatingClient())
    case_id = store.list_cases(status="OPEN", limit=1)[0]["case_id"]
    result = investigator.investigate_case(store, settings, case_id)
    assert result["disposition"] == "REJECTED"
    assert "hallucinated" in (result["rejected_reason"] or "")
