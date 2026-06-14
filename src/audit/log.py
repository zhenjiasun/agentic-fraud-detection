"""Append-only, hash-chained audit log.

Every consequential action in the system records one row here. The table is
INSERT-only (no update/delete path) and each row is chained:
    hash = sha256(prev_hash + canonical_json(record))
so any later tampering is detectable via src/audit/query.py:verify_chain. Rows
are mirrored to an immutable JSONL ledger as a second, file-based witness.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Canonical action vocabulary (kept here so producers stay consistent).
ACTION_TYPES = {
    "WORLD_SEEDED", "MODELS_TRAINED", "RULE_FIRED", "AUTO_BLOCK", "AUTO_ALLOW",
    "ROUTE_TO_REVIEW", "CASE_ASSIGNED", "INVESTIGATION_RUN", "HUMAN_RESOLVE",
    "INJECTION_FLAGGED", "TOOL_ABUSE_BLOCKED", "DRIFT_ALERT", "DATA_QUALITY_ALERT",
    "ADVERSARIAL_ALERT",
}


def _canonical(record: dict) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)


class AuditLog:
    def __init__(self, store, jsonl_path: Optional[str | Path] = None):
        self.store = store
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self) -> str:
        cur = self.store.conn.execute(
            "SELECT hash FROM audit ORDER BY audit_id DESC LIMIT 1"
        ).fetchone()
        return cur["hash"] if cur else ""

    def record(self, *, actor: str, action_type: str, subject_type: str,
               subject_id: str, payload: Optional[dict[str, Any]] = None) -> str:
        payload = payload or {}
        ts = datetime.now().isoformat()
        prev_hash = self._last_hash()
        record = {
            "ts": ts, "actor": actor, "action_type": action_type,
            "subject_type": subject_type, "subject_id": subject_id,
            "payload": payload,
        }
        digest = hashlib.sha256((prev_hash + _canonical(record)).encode()).hexdigest()
        self.store.execute(
            "INSERT INTO audit (ts, actor, action_type, subject_type, subject_id, "
            "payload_json, prev_hash, hash) VALUES (?,?,?,?,?,?,?,?)",
            (ts, actor, action_type, subject_type, subject_id,
             json.dumps(payload, default=str), prev_hash, digest),
        )
        if self.jsonl_path:
            with open(self.jsonl_path, "a") as fh:
                fh.write(json.dumps({**record, "prev_hash": prev_hash, "hash": digest},
                                    default=str) + "\n")
        return digest


def open_audit(settings, store) -> AuditLog:
    return AuditLog(store, settings.audit_log_path)
