"""Audit trail: every action recorded, hash chain verifies, tampering detected."""
from __future__ import annotations

from src.audit.log import AuditLog
from src.audit.query import verify_chain


def test_actions_are_recorded(store):
    types = set(store.query_df("SELECT DISTINCT action_type FROM audit")["action_type"])
    # the full pipeline must have logged rules, model actions, and investigations
    assert {"RULE_FIRED", "AUTO_BLOCK", "AUTO_ALLOW", "ROUTE_TO_REVIEW"} <= types


def test_chain_verifies(store):
    assert verify_chain(store)["ok"] is True


def test_every_record_has_a_hash(store):
    nulls = store.query_df(
        "SELECT COUNT(*) n FROM audit WHERE hash IS NULL OR hash=''").iloc[0]["n"]
    assert int(nulls) == 0


def test_tampering_breaks_chain(store):
    # record a row, tamper it, confirm detection, then it's the last row so we
    # leave the store usable for other session-scoped reads (verify only)
    audit = AuditLog(store)
    audit.record(actor="system", action_type="WORLD_SEEDED",
                 subject_type="test", subject_id="t1", payload={"k": 1})
    last_id = store.query_df("SELECT MAX(audit_id) m FROM audit").iloc[0]["m"]
    store.execute("UPDATE audit SET payload_json='{\"k\":999}' WHERE audit_id=?", (int(last_id),))
    result = verify_chain(store)
    assert result["ok"] is False
    assert result["broken_at"] == int(last_id)
    # clean up the tampered row so chain is valid again for any later reads
    store.execute("DELETE FROM audit WHERE audit_id=?", (int(last_id),))
