"""Same seed => identical world; different seed => different world."""
from __future__ import annotations

from src.data.simulator import SyntheticWorld


def _world(settings, seed):
    w = SyntheticWorld(settings, seed=seed)
    w.generate()
    return w


def test_same_seed_identical(settings):
    a = _world(settings, 42)
    b = _world(settings, 42)
    assert a.summary() == b.summary()
    assert [t["txn_id"] for t in a.txns] == [t["txn_id"] for t in b.txns]
    assert [t["amount"] for t in a.txns] == [t["amount"] for t in b.txns]


def test_different_seed_differs(settings):
    a = _world(settings, 1)
    b = _world(settings, 2)
    assert a.summary() != b.summary() or \
        [t["amount"] for t in a.txns] != [t["amount"] for t in b.txns]


def test_fraud_label_consistency(settings):
    w = _world(settings, 7)
    labeled = sum(t["is_fraud_gt"] for t in w.txns)
    assert labeled == w.summary()["fraud_txns"]
    assert 0 < w.summary()["fraud_rate"] < 0.20  # realistic, not toy-saturated
