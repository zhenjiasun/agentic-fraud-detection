"""Load and validate declarative rules from YAML."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

VALID_ACTIONS = {"auto_block", "auto_allow", "route_to_review"}
VALID_CONFIDENCE = {"high", "medium", "low"}


@dataclass
class Rule:
    id: str
    when: str
    action: str
    reason_code: str
    confidence: str


def load_rules(path: str | Path) -> list[Rule]:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    rules = []
    for r in raw.get("rules", []):
        rule = Rule(id=r["id"], when=r["when"], action=r["action"],
                    reason_code=r["reason_code"], confidence=r.get("confidence", "medium"))
        if rule.action not in VALID_ACTIONS:
            raise ValueError(f"{rule.id}: invalid action {rule.action}")
        if rule.confidence not in VALID_CONFIDENCE:
            raise ValueError(f"{rule.id}: invalid confidence {rule.confidence}")
        rules.append(rule)
    return rules
