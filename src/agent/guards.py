"""Prompt-injection defenses for the investigator agent (defense in depth).

The LOAD-BEARING control is capability restriction (no write tools exist; output
is a closed enum — see tools.py / schema.py). These guards are the secondary
layer: they tag attacker-controlled fields (merchant names, memos) as untrusted
data, strip role-marker/control sequences, and flag known injection tokens so
the adversarial monitor and audit trail can see attempts. They are NOT relied on
as the sole defense.
"""
from __future__ import annotations

import re

# Patterns indicative of an injection attempt embedded in entity-controlled text.
INJECTION_PATTERNS = {
    "ignore_instructions": re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.I),
    "disregard": re.compile(r"disregard\s+(the\s+)?(above|previous|prior|system)", re.I),
    "role_marker": re.compile(r"\b(system|assistant)\s*:", re.I),
    "new_instructions": re.compile(r"(new|updated)\s+instructions?\s*:", re.I),
    "you_are_now": re.compile(r"you\s+are\s+now\b", re.I),
    "tool_injection": re.compile(r"\b(call|invoke|use)\s+(the\s+)?(block|allow|refund|approve)\b", re.I),
    "mark_as": re.compile(r"mark\s+(this\s+)?(as\s+)?(legit|safe|approved|not\s+fraud)", re.I),
    "override": re.compile(r"\boverride\b|\bbypass\b", re.I),
    "exfiltration": re.compile(
        r"(print|reveal|dump|list|show|send|leak|expose)\b.{0,40}"
        r"(ssn|social\s+security|card\s+number|password|secret|credential|"
        r"all\s+(other\s+)?users)", re.I),
}

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def detect_injection(text: str) -> list[str]:
    if not text:
        return []
    return [name for name, pat in INJECTION_PATTERNS.items() if pat.search(text)]


def sanitize(text: str) -> str:
    """Strip control chars and collapse role-marker sequences to neutral text."""
    if not text:
        return ""
    text = _CONTROL.sub(" ", text)
    text = re.sub(r"\b(system|assistant)\s*:", r"\1_", text, flags=re.I)
    return text


def tag_untrusted(text: str) -> tuple[str, list[str]]:
    """Return (delimited+sanitized text, injection flags) for an untrusted field."""
    flags = detect_injection(text or "")
    safe = sanitize(text or "")
    tagged = f"<untrusted_field>{safe}</untrusted_field>"
    return tagged, flags
