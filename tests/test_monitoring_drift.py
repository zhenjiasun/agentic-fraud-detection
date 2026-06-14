"""PSI/KS sensitivity + data-quality catches injected problems."""
from __future__ import annotations

import numpy as np

from src.analysis.data_quality import data_quality_section
from src.analysis.drift import ks_stat, psi


def _deciles(x):
    return [float(q) for q in np.quantile(x, np.linspace(0, 1, 11))]


def test_psi_near_zero_for_identical():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert psi(_deciles(ref), ref) < 0.05


def test_psi_large_for_shifted():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    shifted = rng.normal(3, 1, 5000)
    assert psi(_deciles(ref), shifted) > 0.25


def test_ks_detects_shift():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 5000)
    assert ks_stat(_deciles(ref), rng.normal(0, 1, 5000)) < 0.1
    assert ks_stat(_deciles(ref), rng.normal(3, 1, 5000)) > 0.5


def test_data_quality_passes_clean(store):
    dq = data_quality_section(store)
    assert dq["status"] in ("pass", "warn")
    assert dq["n_fail"] == 0


def test_data_quality_catches_orphans(store):
    store.execute(
        "INSERT INTO transactions (txn_id, ts, user_id, merchant_id, amount, mcc, status, is_fraud_gt) "
        "VALUES ('orphan_x','2025-01-01T00:00:00','ghost_user','m00001',10,'5411','approved',0)")
    dq = data_quality_section(store)
    names = {c["name"]: c["status"] for c in dq["checks"]}
    assert names["orphan_transactions"] == "fail"
    store.execute("DELETE FROM transactions WHERE txn_id='orphan_x'")
