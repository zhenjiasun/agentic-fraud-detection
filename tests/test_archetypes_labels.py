"""Every fraud archetype is injected and correctly labeled."""
from __future__ import annotations

ARCHETYPES = {"account_takeover", "card_testing", "bust_out",
              "money_mule_ring", "synthetic_identity"}


def test_all_archetypes_present(store):
    df = store.query_df(
        "SELECT DISTINCT fraud_archetype_gt FROM transactions "
        "WHERE fraud_archetype_gt IS NOT NULL"
    )
    found = set(df["fraud_archetype_gt"])
    assert ARCHETYPES <= found, f"missing: {ARCHETYPES - found}"


def test_injected_rows_labeled(store):
    # every fraud-archetype transaction must carry is_fraud_gt=1
    bad = store.query_df(
        "SELECT COUNT(*) n FROM transactions "
        "WHERE fraud_archetype_gt IS NOT NULL AND is_fraud_gt=0"
    ).iloc[0]["n"]
    assert int(bad) == 0


def test_base_rate_realistic(store):
    n = store.query_df("SELECT COUNT(*) n FROM transactions").iloc[0]["n"]
    f = store.query_df("SELECT COUNT(*) n FROM transactions WHERE is_fraud_gt=1").iloc[0]["n"]
    rate = f / n
    assert 0.005 < rate < 0.15
