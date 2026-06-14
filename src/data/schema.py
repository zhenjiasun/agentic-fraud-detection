"""Entity schema: dataclasses + shared constants for the synthetic world.

Column order in each dataclass matches the SQLite schema in store.py. The
simulator builds these and converts to dicts/DataFrames for persistence.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional

# Fixed base date so timestamps are deterministic regardless of wall-clock run time.
BASE_DATE = datetime(2025, 1, 1)

COUNTRIES = ["US", "GB", "DE", "FR", "CA", "BR", "NG", "IN", "RU", "CN"]
# Countries we treat as higher base risk (used for plausibility, not as a label).
HIGH_RISK_COUNTRIES = ["NG", "RU", "CN"]
SEGMENTS = ["budget", "standard", "premium"]
MCC_CODES = {
    "5411": "grocery",
    "5812": "restaurant",
    "5732": "electronics",
    "5999": "misc_retail",
    "4829": "money_transfer",   # high-risk: money-mule funnel target
    "6051": "crypto",           # high-risk
    "7995": "gambling",         # high-risk
    "5944": "jewelry",
}
HIGH_RISK_MCC = ["4829", "6051", "7995"]


def h(value: str) -> str:
    """Short stable hash used for PII-like fields (emails, ssn, names)."""
    return hashlib.sha256(str(value).encode()).hexdigest()[:16]


def iso(dt: datetime) -> str:
    return dt.isoformat()


@dataclass
class User:
    user_id: str
    created_at: str
    country: str
    segment: str
    email_hash: str
    signup_ip: str
    credit_limit: float
    is_fraud_gt: int = 0
    fraud_archetype_gt: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Merchant:
    merchant_id: str
    name: str
    mcc: str
    country: str
    risk_tier: str
    is_fraud_gt: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Device:
    device_id: str
    os: str
    fingerprint_hash: str
    first_seen: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Ip:
    ip_id: str
    ip: str
    asn: str
    geo_country: str
    is_datacenter: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Card:
    card_id: str
    user_id: str
    bin: str
    last4: str
    issuer_country: str
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Identity:
    identity_id: str
    ssn_hash: str
    dob: str
    name_hash: str

    def to_dict(self) -> dict:
        return asdict(self)
