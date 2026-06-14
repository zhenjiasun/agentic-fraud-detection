"""SyntheticWorld — reproducible generator of users, merchants, payments,
identities, devices and IPs, with injected fraud archetypes and ground-truth
labels.

The world is built in three passes:
1. base population (legit users/merchants/devices/ips/cards/identities)
2. normal transactions (Poisson arrivals, geo-consistent, diurnal)
3. fraud archetype injection (src/data/archetypes.py)

Realism: a fraction of legit users get travel/burst behavior (hard negatives)
and some fraud campaigns run at reduced intensity (soft positives), so a genuine
ambiguous middle band reaches the review queue and the LLM investigator instead
of every case being trivially separable.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.data import archetypes
from src.data.schema import (
    BASE_DATE, COUNTRIES, HIGH_RISK_MCC, MCC_CODES, SEGMENTS,
    Card, Device, Identity, Ip, Merchant, User, h, iso,
)
from src.data.seeds import spawn
from src.log import get_logger

log = get_logger("simulator")

SEGMENT_AMOUNT = {"budget": 40.0, "standard": 90.0, "premium": 220.0}
SEGMENT_LIMIT = {"budget": 1500.0, "standard": 6000.0, "premium": 25000.0}


class SyntheticWorld:
    def __init__(self, settings, seed: int):
        self.settings = settings
        self.cfg = settings.simulator
        self.seed = seed

        # ordered records
        self.users: dict[str, User] = {}
        self.merchants: dict[str, Merchant] = {}
        self.devices: dict[str, Device] = {}
        self.ips: dict[str, Ip] = {}
        self.cards: dict[str, Card] = {}
        self.identities: dict[str, Identity] = {}
        self.links: list[dict] = []
        self.txns: list[dict] = []

        # per-user infra (for normal txn generation + archetype reuse)
        self.user_cards: dict[str, list[str]] = {}
        self.user_devices: dict[str, list[str]] = {}
        self.user_ips: dict[str, list[str]] = {}

        self._counters: dict[str, int] = {}

    # ------------------------------------------------------------------ ids
    def _next(self, prefix: str) -> str:
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        return f"{prefix}{self._counters[prefix]:05d}"

    def rng(self, name: str) -> np.random.Generator:
        return spawn(self.seed, name)

    # ----------------------------------------------------------- entity adds
    def add_device(self, rng, geo: Optional[str] = None, datacenter: bool = False,
                   day: float = 0.0) -> str:
        did = self._next("d")
        self.devices[did] = Device(
            device_id=did,
            os=rng.choice(["iOS", "Android", "Windows", "macOS", "Linux"]),
            fingerprint_hash=h(f"fp-{did}-{rng.integers(1e9)}"),
            first_seen=iso(BASE_DATE + timedelta(days=float(day))),
        )
        return did

    def add_ip(self, rng, geo: Optional[str] = None, datacenter: bool = False) -> str:
        iid = self._next("ip")
        geo = geo or rng.choice(COUNTRIES)
        octets = rng.integers(1, 255, size=4)
        self.ips[iid] = Ip(
            ip_id=iid,
            ip=".".join(str(int(o)) for o in octets),
            asn=f"AS{int(rng.integers(1000, 60000))}",
            geo_country=geo,
            is_datacenter=int(datacenter),
        )
        return iid

    def add_card(self, rng, user_id: str, day: float = 0.0,
                 issuer_country: Optional[str] = None) -> str:
        cid = self._next("c")
        self.cards[cid] = Card(
            card_id=cid,
            user_id=user_id,
            bin=str(int(rng.integers(400000, 499999))),
            last4=f"{int(rng.integers(0, 9999)):04d}",
            issuer_country=issuer_country or self.users[user_id].country,
            created_at=iso(BASE_DATE + timedelta(days=float(day))),
        )
        self.user_cards.setdefault(user_id, []).append(cid)
        self.link("user", user_id, "card", cid, "owns_card", day)
        return cid

    def add_identity(self, rng, ssn: str, dob: str, name: str) -> str:
        idid = self._next("id")
        self.identities[idid] = Identity(
            identity_id=idid, ssn_hash=h(ssn), dob=dob, name_hash=h(name),
        )
        return idid

    def link(self, src_type, src_id, dst_type, dst_id, link_type, day: float = 0.0):
        self.links.append({
            "src_type": src_type, "src_id": src_id,
            "dst_type": dst_type, "dst_id": dst_id,
            "link_type": link_type,
            "first_seen": iso(BASE_DATE + timedelta(days=float(day))),
        })

    def add_txn(self, *, user_id, merchant_id, card_id, device_id, ip_id, amount,
                ts, status="approved", fraud=0, archetype=None, txn_type="purchase"):
        tid = self._next("t")
        m = self.merchants[merchant_id]
        self.txns.append({
            "txn_id": tid, "ts": ts, "user_id": user_id, "merchant_id": merchant_id,
            "card_id": card_id, "device_id": device_id, "ip_id": ip_id,
            "amount": round(float(amount), 2), "currency": "USD", "mcc": m.mcc,
            "txn_type": txn_type, "status": status,
            "is_fraud_gt": int(fraud), "fraud_archetype_gt": archetype,
        })
        return tid

    # --------------------------------------------------------------- passes
    def generate(self) -> None:
        self._gen_merchants()
        self._gen_users()
        self._gen_normal_transactions()
        archetypes.inject_all(self)
        log.info("generate() produced %d txns across %d users", len(self.txns), len(self.users))

    def _gen_merchants(self) -> None:
        rng = self.rng("merchants")
        mccs = list(MCC_CODES.keys())
        for _ in range(self.cfg["n_merchants"]):
            mid = self._next("m")
            mcc = rng.choice(mccs, p=self._mcc_probs(mccs))
            tier = "high" if mcc in HIGH_RISK_MCC else rng.choice(["low", "medium"], p=[0.7, 0.3])
            self.merchants[mid] = Merchant(
                merchant_id=mid,
                name=f"{MCC_CODES[mcc].title()} Store {mid[-3:]}",
                mcc=mcc, country=rng.choice(COUNTRIES), risk_tier=tier,
            )

    @staticmethod
    def _mcc_probs(mccs):
        # high-risk MCCs are rarer among merchants
        w = np.array([0.3 if m in HIGH_RISK_MCC else 1.0 for m in mccs])
        return w / w.sum()

    def create_user(self, rng, *, country=None, segment=None, created_day=None,
                    n_devices=1, n_cards=1, fraud=0, archetype=None,
                    identity_id=None) -> str:
        """Create one user + their infra (ip, device(s), card(s), identity).

        Used by both base-population generation and archetype injectors so the
        two never drift. Pass `identity_id` to share an existing identity node
        (synthetic-identity rings).
        """
        uid = self._next("u")
        seg = segment or rng.choice(SEGMENTS, p=[0.4, 0.45, 0.15])
        country = country or rng.choice(COUNTRIES, p=self._country_probs())
        created_day = created_day if created_day is not None else -float(rng.integers(1, 730))
        ip_id = self.add_ip(rng, geo=country)
        self.users[uid] = User(
            user_id=uid,
            created_at=iso(BASE_DATE + timedelta(days=float(created_day))),
            country=country, segment=seg,
            email_hash=h(f"user-{uid}"),
            signup_ip=self.ips[ip_id].ip,
            credit_limit=SEGMENT_LIMIT[seg] * float(rng.uniform(0.7, 1.3)),
            is_fraud_gt=int(fraud), fraud_archetype_gt=archetype,
        )
        self.user_ips[uid] = [ip_id]
        self.link("user", uid, "ip", ip_id, "uses_ip", 0)
        self.user_devices[uid] = []
        for _ in range(n_devices):
            did = self.add_device(rng, geo=country)
            self.user_devices[uid].append(did)
            self.link("user", uid, "device", did, "uses_device", 0)
        for _ in range(n_cards):
            self.add_card(rng, uid)
        if identity_id is None:
            identity_id = self.add_identity(
                rng, ssn=f"ssn-{uid}", dob=f"19{int(rng.integers(50, 99))}-01-01",
                name=f"name-{uid}",
            )
        self.link("user", uid, "identity", identity_id, "shares_identity", 0)
        return uid

    def _gen_users(self) -> None:
        rng = self.rng("users")
        for _ in range(self.cfg["n_users"]):
            self.create_user(rng, n_devices=1 + int(rng.random() < 0.35),
                             n_cards=1 + int(rng.random() < 0.4))

    @staticmethod
    def _country_probs():
        # US-heavy population
        w = np.array([6, 2, 2, 2, 2, 1.2, 0.6, 1.5, 0.6, 0.6])
        return w / w.sum()

    def _gen_normal_transactions(self) -> None:
        rng = self.rng("normal_txns")
        days = self.cfg["days"]
        base = self.cfg["base_txn_per_user"]
        hard_neg_rate = self.cfg["hard_negative_rate"]
        normal_merch = [m for m in self.merchants.values() if m.risk_tier != "high"]
        for uid, user in self.users.items():
            n = int(rng.poisson(base))
            mean_amt = SEGMENT_AMOUNT[user.segment]
            is_hard_neg = rng.random() < hard_neg_rate
            for _ in range(n):
                day = float(rng.uniform(0, days))
                hour = int(np.clip(rng.normal(14, 4), 0, 23))  # diurnal
                ts = iso(BASE_DATE + timedelta(days=day, hours=hour,
                                               minutes=int(rng.integers(0, 60))))
                merchant = normal_merch[int(rng.integers(len(normal_merch)))]
                amount = float(rng.lognormal(np.log(mean_amt), 0.6))
                status = "approved" if rng.random() > 0.03 else "declined"
                device_id = self.user_devices[uid][int(rng.integers(len(self.user_devices[uid])))]
                ip_id = self.user_ips[uid][0]
                card_id = self.user_cards[uid][int(rng.integers(len(self.user_cards[uid])))]
                # hard negative: a legit "travel + spend burst" that looks risky
                if is_hard_neg and rng.random() < 0.25:
                    ip_id = self.add_ip(rng, geo=rng.choice(COUNTRIES))
                    self.user_ips[uid].append(ip_id)
                    self.link("user", uid, "ip", ip_id, "uses_ip", day)
                    amount *= float(rng.uniform(2.0, 4.0))
                self.add_txn(user_id=uid, merchant_id=merchant.merchant_id,
                             card_id=card_id, device_id=device_id, ip_id=ip_id,
                             amount=amount, ts=ts, status=status)

    # -------------------------------------------------------------- persist
    def persist(self, store) -> None:
        store.insert_df("users", self._df(self.users))
        store.insert_df("merchants", self._df(self.merchants))
        store.insert_df("devices", self._df(self.devices))
        store.insert_df("ips", self._df(self.ips))
        store.insert_df("cards", self._df(self.cards))
        store.insert_df("identities", self._df(self.identities))
        store.insert_df("entity_links", pd.DataFrame(self.links))
        store.insert_df("transactions", pd.DataFrame(self.txns))

    @staticmethod
    def _df(records: dict) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in records.values()])

    def summary(self) -> dict:
        fraud_txns = sum(t["is_fraud_gt"] for t in self.txns)
        fraud_users = sum(1 for u in self.users.values() if u.is_fraud_gt)
        return {
            "users": len(self.users),
            "merchants": len(self.merchants),
            "devices": len(self.devices),
            "ips": len(self.ips),
            "cards": len(self.cards),
            "transactions": len(self.txns),
            "fraud_txns": int(fraud_txns),
            "fraud_rate": round(fraud_txns / max(1, len(self.txns)), 4),
            "fraud_users": int(fraud_users),
        }
