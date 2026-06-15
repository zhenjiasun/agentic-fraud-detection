"""Decision orchestrator: combine rules + calibrated model score + graph signals
into one action per account, populate the review queue, and audit everything.

Precedence:
1. high-confidence rule (auto_block / auto_allow) wins, short-circuit
2. else calibrated account score vs t_low / t_high thresholds
3. any route_to_review rule, or ring membership on an otherwise-allowed account,
   escalates to human review
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.audit.log import open_audit
from src.config import ROOT
from src.data.schema import HIGH_RISK_MCC
from src.graph.builder import build_graph
from src.graph.features import graph_features
from src.graph.rings import detect_rings
from src.log import get_logger
from src.models.features import build_account_features
from src.orchestrator.queue import OPEN, RESOLVED_BLOCK, ReviewQueue
from src.rules.engine import RulesEngine
from src.rules.loader import load_rules

log = get_logger("orchestrator")


def build_account_context(store) -> pd.DataFrame:
    """One row per user with account/graph/txn/model fields the rules reference."""
    graph = build_graph(store)
    rings = detect_rings(graph, _settings_from_store(store), store=store)
    gf = graph_features(graph, rings)
    acct = build_account_features(store, gf)

    # model scores
    a_scores = store.table("account_scores").set_index("user_id")["score"]
    t = store.table("transactions").merge(
        store.table("ips")[["ip_id", "geo_country", "is_datacenter"]], on="ip_id", how="left"
    ).merge(store.table("users")[["user_id", "country"]], on="user_id", how="left")
    tx_scores = store.table("txn_scores").set_index("txn_id")["score"]
    t["txn_score"] = t["txn_id"].map(tx_scores).fillna(0)

    # extra per-account txn aggregates referenced by rules
    dev_cards = t.groupby("device_id")["card_id"].nunique()
    t["dev_cards"] = t["device_id"].map(dev_cards)
    user_mean = t.groupby("user_id")["amount"].transform("mean")
    user_std = t.groupby("user_id")["amount"].transform("std").replace(0, 1).fillna(1)
    t["amount_z"] = (t["amount"] - user_mean) / user_std
    t["geo_mismatch"] = (t["geo_country"] != t["country"]).astype(int)
    t["new_geo_high_value"] = ((t["geo_mismatch"] == 1) & (t["amount"] > 3 * user_mean)).astype(int)

    agg = t.groupby("user_id").agg(
        max_cards_on_device=("dev_cards", "max"),
        datacenter_share=("is_datacenter", "mean"),
        max_amount_z=("amount_z", "max"),
        new_geo_high_value=("new_geo_high_value", "max"),
        max_txn_score=("txn_score", "max"),
    )

    # n_users_on_identity from shared-identity links
    links = store.query_df(
        "SELECT src_id user_id, dst_id identity_id FROM entity_links WHERE link_type='shares_identity'"
    )
    users_per_identity = links.groupby("identity_id")["user_id"].nunique()
    links["upi"] = links["identity_id"].map(users_per_identity)
    n_users_on_identity = links.groupby("user_id")["upi"].max()

    ctx = acct.set_index("user_id").join(agg).fillna(0)
    ctx["account_score"] = ctx.index.map(a_scores).fillna(0)
    ctx["n_users_on_identity"] = ctx.index.map(n_users_on_identity).fillna(1)
    # dollars at stake per account (observable; reverses the log1p in features) — the
    # exposure term in amount_aware thresholding. Not a *_gt column: no leakage.
    ctx["exposure"] = np.expm1(ctx["log_total_spend"].clip(lower=0))
    return ctx.reset_index()


def _ctx_namespaces(row) -> dict[str, dict]:
    return {
        "account": {
            "decline_rate": row["decline_rate"],
            "max_amount_to_limit": row["max_amount_to_limit"],
            "spend_trajectory": row["spend_trajectory"],
            "account_age_days": row["account_age_days"],
            "n_cards": row["n_cards"], "n_devices": row["n_devices"],
            "high_risk_mcc_share": row["high_risk_mcc_share"],
            "foreign_ip_share": row["foreign_ip_share"],
        },
        "graph": {
            "ring_member": int(row["g_ring_member"]),
            "ring_risk": row["g_ring_risk"],
            "n_users_on_device": row["g_n_users_on_device"],
            "n_users_on_ip": row["g_n_users_on_ip"],
        },
        "txn": {
            "max_cards_on_device": row["max_cards_on_device"],
            "datacenter_share": row["datacenter_share"],
            "max_amount_z": row["max_amount_z"],
            "new_geo_high_value": int(row["new_geo_high_value"]),
            "n_users_on_identity": row["n_users_on_identity"],
        },
        "model": {
            "account_score": row["account_score"],
            "max_txn_score": row["max_txn_score"],
        },
    }


def _decide(fired: list[dict], score: float, ring_member: int, t_low, t_high,
            *, mode: str = "global", exposure: float = 0.0, fp_cost: float = 25.0) -> str:
    high = [f for f in fired if f["confidence"] == "high"]
    if any(f["action"] == "auto_block" for f in high):
        return "auto_block"
    # amount_aware: block when the expected fraud loss saved (score*exposure) beats
    # the friction cost of a wrongful block ((1-score)*fp_cost). t_high stays a safety
    # ceiling so a near-certain account still blocks even on thin recorded exposure.
    block_by_loss = mode == "amount_aware" and score * exposure > (1.0 - score) * fp_cost
    if any(f["action"] == "auto_allow" for f in high):
        # a high-confidence allow still defers to a hard block rule (handled above)
        action = "auto_allow"
    elif score >= t_high or block_by_loss:
        action = "auto_block"
    elif score <= t_low:
        action = "auto_allow"
    else:
        action = "route_to_review"
    # escalation: review rules or ring membership pull a borderline case to humans
    if any(f["action"] == "route_to_review" for f in fired) and action != "auto_block":
        action = "route_to_review"
    if ring_member and action == "auto_allow":
        action = "route_to_review"
    return action


def run_orchestrator(store, settings) -> dict:
    audit = open_audit(settings, store)
    engine = RulesEngine(load_rules(ROOT / "config" / "rules.yaml"))
    ocfg = settings.orchestrator
    t_low, t_high = ocfg["t_low"], ocfg["t_high"]
    mode = ocfg.get("threshold_mode", "global")
    fp_cost = ocfg.get("fp_cost", 25.0)

    ctx_df = build_account_context(store)
    queue = ReviewQueue(store, audit)

    counts = {"auto_block": 0, "auto_allow": 0, "route_to_review": 0, "cases": 0,
              "rules_fired": 0}
    for _, row in ctx_df.iterrows():
        uid = row["user_id"]
        ns = _ctx_namespaces(row)
        fired = engine.evaluate(ns)
        counts["rules_fired"] += len(fired)
        for f in fired:
            audit.record(actor="rules", action_type="RULE_FIRED",
                         subject_type="user", subject_id=uid,
                         payload={"rule": f["id"], "reason_code": f["reason_code"]})

        action = _decide(fired, row["account_score"], int(row["g_ring_member"]),
                         t_low, t_high, mode=mode, exposure=float(row["exposure"]),
                         fp_cost=fp_cost)
        counts[action] += 1
        reason_codes = [f["reason_code"] for f in fired]
        store.insert_decision(
            ts=row.get("opened_at", ""), subject_type="user", subject_id=uid,
            source="orchestrator", action=action,
            reason_codes_json=_json(reason_codes), score=float(row["account_score"]),
            case_id=None,
        )
        audit.record(actor="orchestrator", action_type=action.upper(),
                     subject_type="user", subject_id=uid,
                     payload={"score": round(float(row["account_score"]), 4),
                              "reason_codes": reason_codes})

        graph_signals = {"ring_member": int(row["g_ring_member"]),
                         "ring_risk": float(row["g_ring_risk"]),
                         "n_users_on_device": int(row["g_n_users_on_device"])}
        if action == "route_to_review":
            queue.create_case(case_id=f"case_{uid}", subject_type="user", subject_id=uid,
                              status=OPEN, model_score=row["account_score"],
                              rule_codes=reason_codes, graph_signals=graph_signals)
            counts["cases"] += 1
        elif action == "auto_block":
            queue.create_case(case_id=f"case_{uid}", subject_type="user", subject_id=uid,
                              status=RESOLVED_BLOCK, model_score=row["account_score"],
                              rule_codes=reason_codes, graph_signals=graph_signals,
                              disposition="auto_block")
            counts["cases"] += 1
    return counts


def _json(obj):
    import json
    return json.dumps(obj)


def _settings_from_store(store):
    from src.config import get_settings
    return get_settings()
