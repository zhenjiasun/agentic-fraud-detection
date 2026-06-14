"""Audit querying + hash-chain verification."""
from __future__ import annotations

import hashlib
import json

from src.audit.log import _canonical


def verify_chain(store) -> dict:
    """Recompute the hash chain over all audit rows; report first break."""
    rows = store.conn.execute(
        "SELECT audit_id, ts, actor, action_type, subject_type, subject_id, "
        "payload_json, prev_hash, hash FROM audit ORDER BY audit_id ASC"
    ).fetchall()
    prev_hash = ""
    for r in rows:
        record = {
            "ts": r["ts"], "actor": r["actor"], "action_type": r["action_type"],
            "subject_type": r["subject_type"], "subject_id": r["subject_id"],
            "payload": json.loads(r["payload_json"]),
        }
        expected = hashlib.sha256((prev_hash + _canonical(record)).encode()).hexdigest()
        if r["prev_hash"] != prev_hash or r["hash"] != expected:
            return {"ok": False, "broken_at": int(r["audit_id"]), "n_rows": len(rows)}
        prev_hash = r["hash"]
    return {"ok": True, "broken_at": None, "n_rows": len(rows)}


def query_audit(store, *, subject_id=None, action_type=None, actor=None, limit=500):
    clauses, params = [], []
    if subject_id:
        clauses.append("subject_id = ?"); params.append(subject_id)
    if action_type:
        clauses.append("action_type = ?"); params.append(action_type)
    if actor:
        clauses.append("actor = ?"); params.append(actor)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)
    return store.query_df(
        f"SELECT * FROM audit {where} ORDER BY audit_id DESC LIMIT ?", params
    )
