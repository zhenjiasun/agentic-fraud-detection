"""Fraud archetype injectors.

Each injector mutates a SyntheticWorld in place: it adds entities/transactions
and stamps ground-truth labels (`is_fraud_gt`, `fraud_archetype_gt`). Five
distinct, separable-but-overlapping signatures:

- account_takeover  : new device+IP+geo on an aged account, then a high-value burst
- card_testing      : one device -> many cards, many tiny auths, high decline rate
- bust_out          : good history, then a spike toward the credit limit, then dark
- money_mule_ring   : cluster of mules sharing infra, funneling to high-risk merchants
- synthetic_identity: fabricated identities sharing SSN/DOB fragments, thin then active

`soft_positive_intensity` in config dampens a fraction of campaigns so they land
in the ambiguous band that feeds the review queue + investigator.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np

from src.data.schema import BASE_DATE, COUNTRIES, HIGH_RISK_COUNTRIES, iso
from src.log import get_logger

log = get_logger("archetypes")


def inject_all(world) -> None:
    fraud_cfg = world.cfg["fraud"]
    inject_account_takeover(world, fraud_cfg["account_takeover"])
    inject_card_testing(world, fraud_cfg["card_testing"])
    inject_bust_out(world, fraud_cfg["bust_out"])
    inject_money_mule_ring(world, fraud_cfg["money_mule_ring"])
    inject_synthetic_identity(world, fraud_cfg["synthetic_identity"])


# ----------------------------------------------------------------- helpers
def _mark_user_fraud(world, uid, archetype):
    u = world.users[uid]
    u.is_fraud_gt = 1
    u.fraud_archetype_gt = archetype


def _aged_legit_users(world, rng, n):
    pool = [uid for uid, u in world.users.items()
            if u.is_fraud_gt == 0 and u.created_at < iso(BASE_DATE)]
    if not pool:
        return []
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[int(i)] for i in idx]


def _high_risk_merchants(world):
    return [m.merchant_id for m in world.merchants.values() if m.risk_tier == "high"] \
        or [next(iter(world.merchants))]


def _any_merchants(world):
    return list(world.merchants.keys())


def _ts(day, rng, hour=None):
    hour = hour if hour is not None else int(rng.integers(0, 24))
    return iso(BASE_DATE + timedelta(days=float(day), hours=hour,
                                     minutes=int(rng.integers(0, 60))))


# -------------------------------------------------------- account takeover
def inject_account_takeover(world, n):
    rng = world.rng("ato")
    days = world.cfg["days"]
    soft = world.cfg["soft_positive_intensity"]
    victims = _aged_legit_users(world, rng, n)
    merchants = _any_merchants(world)
    for uid in victims:
        is_soft = rng.random() < soft
        takeover_day = float(rng.uniform(days * 0.6, days * 0.95))
        # attacker's fresh device + IP, foreign geo, often datacenter
        new_geo = rng.choice(HIGH_RISK_COUNTRIES) if not is_soft else world.users[uid].country
        did = world.add_device(rng, datacenter=not is_soft, day=takeover_day)
        iid = world.add_ip(rng, geo=new_geo, datacenter=not is_soft)
        world.user_devices[uid].append(did)
        world.user_ips[uid].append(iid)
        world.link("user", uid, "device", did, "uses_device", takeover_day)
        world.link("user", uid, "ip", iid, "uses_ip", takeover_day)
        card_id = world.user_cards[uid][0]
        mult = rng.uniform(2.0, 3.0) if is_soft else rng.uniform(5.0, 10.0)
        from src.data.simulator import SEGMENT_AMOUNT
        base_amt = SEGMENT_AMOUNT[world.users[uid].segment]
        n_burst = int(rng.integers(3, 9))
        for j in range(n_burst):
            day = takeover_day + j * 0.02
            world.add_txn(
                user_id=uid, merchant_id=merchants[int(rng.integers(len(merchants)))],
                card_id=card_id, device_id=did, ip_id=iid,
                amount=base_amt * mult * float(rng.uniform(0.7, 1.4)),
                ts=_ts(day, rng), status="approved",
                fraud=1, archetype="account_takeover",
            )
        _mark_user_fraud(world, uid, "account_takeover")
    log.info("injected account_takeover on %d accounts", len(victims))


# ------------------------------------------------------------- card testing
def inject_card_testing(world, n):
    rng = world.rng("card_testing")
    days = world.cfg["days"]
    soft = world.cfg["soft_positive_intensity"]
    merchants = _any_merchants(world)
    for _ in range(n):
        is_soft = rng.random() < soft
        uid = world.create_user(rng, country=rng.choice(HIGH_RISK_COUNTRIES),
                                n_devices=1, n_cards=0, fraud=1, archetype="card_testing")
        did = world.user_devices[uid][0]
        iid = world.add_ip(rng, geo=rng.choice(HIGH_RISK_COUNTRIES), datacenter=not is_soft)
        world.user_ips[uid] = [iid]
        world.link("user", uid, "ip", iid, "uses_ip", 0)
        # one device tests many card numbers -> strong graph signal
        n_cards = int(rng.integers(6, 14)) if is_soft else int(rng.integers(15, 30))
        decline_p = 0.4 if is_soft else 0.65
        start_day = float(rng.uniform(0, days))
        for k in range(n_cards):
            cid = world.add_card(rng, uid, day=start_day)
            day = start_day + k * 0.005  # all within minutes
            world.add_txn(
                user_id=uid, merchant_id=merchants[int(rng.integers(len(merchants)))],
                card_id=cid, device_id=did, ip_id=iid,
                amount=float(rng.uniform(0.5, 3.0)),
                ts=_ts(day, rng), txn_type="auth",
                status="declined" if rng.random() < decline_p else "approved",
                fraud=1, archetype="card_testing",
            )
    log.info("injected card_testing x%d", n)


# ---------------------------------------------------------------- bust out
def inject_bust_out(world, n):
    rng = world.rng("bust_out")
    days = world.cfg["days"]
    soft = world.cfg["soft_positive_intensity"]
    users = _aged_legit_users(world, rng, n)
    merchants = _any_merchants(world)
    for uid in users:
        is_soft = rng.random() < soft
        limit = world.users[uid].credit_limit
        did = world.user_devices[uid][0]
        iid = world.user_ips[uid][0]
        card_id = world.user_cards[uid][0]
        spike_day = float(rng.uniform(days * 0.7, days * 0.92))
        # purchases that ramp toward the limit, then the account goes dark
        target = limit * (0.5 if is_soft else 0.9)
        n_spike = int(rng.integers(4, 9))
        for j in range(n_spike):
            day = spike_day + j * 0.3
            world.add_txn(
                user_id=uid, merchant_id=merchants[int(rng.integers(len(merchants)))],
                card_id=card_id, device_id=did, ip_id=iid,
                amount=target / n_spike * float(rng.uniform(0.8, 1.2)),
                ts=_ts(day, rng), status="approved",
                fraud=1, archetype="bust_out",
            )
        _mark_user_fraud(world, uid, "bust_out")
    log.info("injected bust_out on %d accounts", len(users))


# ------------------------------------------------------- money mule ring
def inject_money_mule_ring(world, n_rings):
    rng = world.rng("money_mule")
    days = world.cfg["days"]
    soft = world.cfg["soft_positive_intensity"]
    collectors = _high_risk_merchants(world)
    for r in range(n_rings):
        is_soft = rng.random() < soft
        ring_size = int(rng.integers(4, 9))
        geo = rng.choice(COUNTRIES)
        # mules share a small pool of devices + IPs -> the ring's graph signature
        shared_devices = [world.add_device(rng, datacenter=not is_soft) for _ in range(2)]
        shared_ips = [world.add_ip(rng, geo=geo, datacenter=not is_soft) for _ in range(2)]
        collector = collectors[int(rng.integers(len(collectors)))]
        mules = []
        for _ in range(ring_size):
            uid = world.create_user(rng, country=geo, n_devices=0, n_cards=1,
                                    fraud=1, archetype="money_mule_ring")
            mules.append(uid)
            # attach shared infra (not the user's own) -> connects the ring
            for did in shared_devices:
                world.user_devices.setdefault(uid, []).append(did)
                world.link("user", uid, "device", did, "uses_device", 0)
            for iid in shared_ips:
                world.user_ips.setdefault(uid, []).append(iid)
                world.link("user", uid, "ip", iid, "uses_ip", 0)
        # funnel: each mule pushes funds to the collector merchant
        for uid in mules:
            card_id = world.user_cards[uid][0]
            n_tx = int(rng.integers(2, 6))
            for j in range(n_tx):
                day = float(rng.uniform(days * 0.3, days * 0.95))
                world.add_txn(
                    user_id=uid, merchant_id=collector,
                    card_id=card_id,
                    device_id=shared_devices[int(rng.integers(len(shared_devices)))],
                    ip_id=shared_ips[int(rng.integers(len(shared_ips)))],
                    amount=float(rng.uniform(300, 1500)),
                    ts=_ts(day, rng), txn_type="transfer",
                    status="approved", fraud=1, archetype="money_mule_ring",
                )
    log.info("injected money_mule_ring x%d", n_rings)


# --------------------------------------------------- synthetic identity
def inject_synthetic_identity(world, n):
    rng = world.rng("synthetic_id")
    days = world.cfg["days"]
    merchants = _any_merchants(world)
    created = 0
    remaining = n
    while remaining > 0:
        cluster = int(min(remaining, rng.integers(2, 4)))
        # one fabricated identity shared across the cluster -> shares_identity edges
        shared_identity = world.add_identity(
            rng, ssn=f"synthssn-{rng.integers(1e9)}",
            dob=f"199{int(rng.integers(0,9))}-0{int(rng.integers(1,9))}-01",
            name=f"synthname-{rng.integers(1e6)}",
        )
        for _ in range(cluster):
            # thin file: brand-new account, sudden activity
            uid = world.create_user(rng, created_day=-float(rng.integers(5, 40)),
                                    n_devices=1, n_cards=1, fraud=1,
                                    archetype="synthetic_identity",
                                    identity_id=shared_identity)
            did = world.user_devices[uid][0]
            iid = world.user_ips[uid][0]
            card_id = world.user_cards[uid][0]
            burst_day = float(rng.uniform(days * 0.5, days * 0.95))
            for j in range(int(rng.integers(3, 8))):
                world.add_txn(
                    user_id=uid, merchant_id=merchants[int(rng.integers(len(merchants)))],
                    card_id=card_id, device_id=did, ip_id=iid,
                    amount=float(rng.uniform(150, 900)),
                    ts=_ts(burst_day + j * 0.1, rng), status="approved",
                    fraud=1, archetype="synthetic_identity",
                )
            created += 1
        remaining -= cluster
    log.info("injected synthetic_identity x%d", created)
