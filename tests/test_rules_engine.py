"""Rules fire on crafted contexts; malicious expressions are rejected (no eval)."""
from __future__ import annotations

import pytest

from src.rules.engine import RuleSyntaxError, RulesEngine, safe_eval
from src.rules.loader import Rule


def _ctx(**overrides):
    base = {"account": {"decline_rate": 0.0, "max_amount_to_limit": 0.0,
                        "spend_trajectory": 0.0, "account_age_days": 365},
            "graph": {"ring_member": 0}, "txn": {"max_cards_on_device": 0,
                                                 "new_geo_high_value": 0,
                                                 "n_users_on_identity": 1,
                                                 "datacenter_share": 0, "max_amount_z": 0},
            "model": {"account_score": 0.0}}
    for ns, vals in overrides.items():
        base[ns].update(vals)
    return base


def test_rule_fires():
    e = RulesEngine([Rule("R", "graph.ring_member == 1 and model.account_score > 0.7",
                          "auto_block", "RING", "high")])
    assert e.evaluate(_ctx(graph={"ring_member": 1}, model={"account_score": 0.9}))
    assert not e.evaluate(_ctx(model={"account_score": 0.9}))


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hi')",
    "account.__class__",
    "open('/etc/passwd')",
    "().__class__.__bases__",
    "account.decline_rate.__init__",
])
def test_malicious_expressions_rejected(expr):
    with pytest.raises(RuleSyntaxError):
        safe_eval(expr, _ctx())


def test_bare_name_rejected():
    with pytest.raises(RuleSyntaxError):
        safe_eval("foo > 1", _ctx())
