"""Bounded output schema for the investigator agent.

The agent's ONLY decision surface is an InvestigationResult with a closed
disposition enum. Anything outside this schema (an invented disposition,
hallucinated evidence ids, missing fields) is rejected by validation and never
reaches the review queue. ESCALATE routes to a human; it does not resolve.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Disposition(str, Enum):
    LIKELY_FRAUD = "LIKELY_FRAUD"
    LIKELY_LEGIT = "LIKELY_LEGIT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    ESCALATE = "ESCALATE"


MAX_RATIONALE_CHARS = 1200


class InvestigationResult(BaseModel):
    disposition: Disposition
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=MAX_RATIONALE_CHARS)
    evidence_refs: list[str] = Field(default_factory=list)

    @field_validator("rationale")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()[:MAX_RATIONALE_CHARS]


# JSON schema for the `submit_finding` structured-output tool (provider-neutral).
SUBMIT_FINDING_SCHEMA = {
    "type": "object",
    "properties": {
        "disposition": {
            "type": "string",
            "enum": [d.value for d in Disposition],
            "description": "Your recommended disposition (closed set).",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence 0.0-1.0 in the disposition.",
        },
        "rationale": {
            "type": "string",
            "description": "Concise evidence-based justification (<=1200 chars).",
        },
        "evidence_refs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "IDs of transactions/cases/entities you actually inspected.",
        },
    },
    "required": ["disposition", "confidence", "rationale", "evidence_refs"],
}
