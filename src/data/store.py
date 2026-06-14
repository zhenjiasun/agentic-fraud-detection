"""SQLite store — the single source of truth.

Everything (simulator, models, orchestrator, agent, API, dashboard) reads and
writes through one SQLite database so the API and dashboard processes share
state for free. The audit table is append-only and hash-chained; see
src/audit/log.py for the writer that enforces that invariant.

Design notes:
- WAL mode so the API can read while bootstrap writes.
- `*_gt` (ground-truth) columns exist only because this is a simulator. They are
  consumed ONLY by src/analysis/evaluation.py. Models/rules must never read them
  (enforced by the feature whitelist in src/models/features.py + a leakage test).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    created_at TEXT,
    country TEXT,
    segment TEXT,
    email_hash TEXT,
    signup_ip TEXT,
    credit_limit REAL,
    is_fraud_gt INTEGER DEFAULT 0,
    fraud_archetype_gt TEXT
);
CREATE TABLE IF NOT EXISTS merchants (
    merchant_id TEXT PRIMARY KEY,
    name TEXT,
    mcc TEXT,
    country TEXT,
    risk_tier TEXT,
    is_fraud_gt INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS devices (
    device_id TEXT PRIMARY KEY,
    os TEXT,
    fingerprint_hash TEXT,
    first_seen TEXT
);
CREATE TABLE IF NOT EXISTS ips (
    ip_id TEXT PRIMARY KEY,
    ip TEXT,
    asn TEXT,
    geo_country TEXT,
    is_datacenter INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS cards (
    card_id TEXT PRIMARY KEY,
    user_id TEXT,
    bin TEXT,
    last4 TEXT,
    issuer_country TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS identities (
    identity_id TEXT PRIMARY KEY,
    ssn_hash TEXT,
    dob TEXT,
    name_hash TEXT
);
CREATE TABLE IF NOT EXISTS entity_links (
    link_id INTEGER PRIMARY KEY AUTOINCREMENT,
    src_type TEXT, src_id TEXT,
    dst_type TEXT, dst_id TEXT,
    link_type TEXT,
    first_seen TEXT
);
CREATE TABLE IF NOT EXISTS transactions (
    txn_id TEXT PRIMARY KEY,
    ts TEXT,
    user_id TEXT,
    merchant_id TEXT,
    card_id TEXT,
    device_id TEXT,
    ip_id TEXT,
    amount REAL,
    currency TEXT,
    mcc TEXT,
    txn_type TEXT,
    status TEXT,
    is_fraud_gt INTEGER DEFAULT 0,
    fraud_archetype_gt TEXT
);
CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    subject_type TEXT,
    subject_id TEXT,
    opened_at TEXT,
    status TEXT,
    model_score REAL,
    rule_codes_json TEXT,
    graph_signals_json TEXT,
    assigned_to TEXT,
    resolved_at TEXT,
    disposition TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    subject_type TEXT,
    subject_id TEXT,
    source TEXT,
    action TEXT,
    reason_codes_json TEXT,
    score REAL,
    case_id TEXT
);
CREATE TABLE IF NOT EXISTS investigations (
    investigation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT,
    ts TEXT,
    provider TEXT,
    model TEXT,
    disposition TEXT,
    confidence REAL,
    rationale TEXT,
    evidence_json TEXT,
    tool_calls_json TEXT,
    injection_flags_json TEXT
);
CREATE TABLE IF NOT EXISTS account_scores (
    user_id TEXT PRIMARY KEY,
    raw_score REAL,
    score REAL
);
CREATE TABLE IF NOT EXISTS txn_scores (
    txn_id TEXT PRIMARY KEY,
    score REAL
);
CREATE TABLE IF NOT EXISTS audit (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    actor TEXT,
    action_type TEXT,
    subject_type TEXT,
    subject_id TEXT,
    payload_json TEXT,
    prev_hash TEXT,
    hash TEXT
);
CREATE INDEX IF NOT EXISTS idx_txn_user ON transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_txn_ts ON transactions(ts);
CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);
CREATE INDEX IF NOT EXISTS idx_links_src ON entity_links(src_type, src_id);
CREATE INDEX IF NOT EXISTS idx_audit_subject ON audit(subject_type, subject_id);
"""

# Tables the simulator regenerates wholesale on each seed.
WORLD_TABLES = [
    "users", "merchants", "devices", "ips", "cards", "identities",
    "entity_links", "transactions",
]
# Operational tables cleared alongside a reseed (decisions reference txns/users).
OPERATIONAL_TABLES = ["cases", "decisions", "investigations", "audit",
                      "account_scores", "txn_scores"]


class Store:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")

    # --- schema lifecycle ---
    def init_schema(self) -> None:
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def reset(self) -> None:
        """Drop all rows (world + operational) for a clean reseed."""
        self.init_schema()
        for table in WORLD_TABLES + OPERATIONAL_TABLES:
            self.conn.execute(f"DELETE FROM {table};")
        self.conn.commit()

    # --- generic helpers ---
    def insert_df(self, table: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        df.to_sql(table, self.conn, if_exists="append", index=False)

    def query_df(self, sql: str, params: Iterable[Any] = ()) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.conn, params=list(params))

    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        cur = self.conn.execute(sql, list(params))
        self.conn.commit()
        return cur

    def table(self, name: str) -> pd.DataFrame:
        return self.query_df(f"SELECT * FROM {name}")

    # --- entity getters ---
    def get_row(self, table: str, key_col: str, key: str) -> Optional[dict]:
        cur = self.conn.execute(f"SELECT * FROM {table} WHERE {key_col} = ?", (key,))
        row = cur.fetchone()
        return dict(row) if row else None

    def recent_transactions(self, user_id: str, n: int = 20) -> list[dict]:
        cur = self.conn.execute(
            "SELECT * FROM transactions WHERE user_id = ? ORDER BY ts DESC LIMIT ?",
            (user_id, n),
        )
        return [dict(r) for r in cur.fetchall()]

    # --- cases ---
    def create_case(self, case: dict) -> None:
        cols = ", ".join(case.keys())
        ph = ", ".join("?" for _ in case)
        self.execute(f"INSERT OR REPLACE INTO cases ({cols}) VALUES ({ph})", tuple(case.values()))

    def get_case(self, case_id: str) -> Optional[dict]:
        return self.get_row("cases", "case_id", case_id)

    def list_cases(self, status: Optional[str] = None, limit: int = 200) -> list[dict]:
        if status:
            cur = self.conn.execute(
                "SELECT * FROM cases WHERE status = ? ORDER BY model_score DESC LIMIT ?",
                (status, limit),
            )
        else:
            cur = self.conn.execute(
                "SELECT * FROM cases ORDER BY opened_at DESC LIMIT ?", (limit,)
            )
        return [dict(r) for r in cur.fetchall()]

    def update_case(self, case_id: str, **fields) -> None:
        if not fields:
            return
        assignments = ", ".join(f"{k} = ?" for k in fields)
        self.execute(
            f"UPDATE cases SET {assignments} WHERE case_id = ?",
            tuple(fields.values()) + (case_id,),
        )

    # --- decisions / investigations ---
    def insert_decision(self, **fields) -> None:
        fields.setdefault("reason_codes_json", json.dumps([]))
        cols = ", ".join(fields.keys())
        ph = ", ".join("?" for _ in fields)
        self.execute(f"INSERT INTO decisions ({cols}) VALUES ({ph})", tuple(fields.values()))

    def insert_investigation(self, **fields) -> int:
        cols = ", ".join(fields.keys())
        ph = ", ".join("?" for _ in fields)
        cur = self.execute(
            f"INSERT INTO investigations ({cols}) VALUES ({ph})", tuple(fields.values())
        )
        return int(cur.lastrowid)

    def close(self) -> None:
        self.conn.close()


def open_store(settings) -> Store:
    """Convenience: open + ensure schema using a Settings object."""
    store = Store(settings.db_path)
    store.init_schema()
    return store
