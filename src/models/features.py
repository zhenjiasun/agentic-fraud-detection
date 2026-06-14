"""Feature engineering for the transaction- and account-risk models.

LEAKAGE GUARD: TXN_FEATURES and ACCOUNT_FEATURES are explicit whitelists. The
build functions assert the returned feature matrix contains exactly these
columns, and a unit test asserts no `*_gt` column ever appears here. Ground-truth
labels are returned separately and only the evaluation module consumes them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import BASE_DATE, HIGH_RISK_MCC

TXN_FEATURES = [
    "log_amount", "amount_z", "amount_to_limit", "hour",
    "is_high_risk_mcc", "is_datacenter_ip", "geo_mismatch", "is_declined",
    "merchant_high_risk", "user_decline_rate", "velocity_24h",
    "g_n_users_on_device", "g_n_users_on_ip", "g_n_shared_devices",
    "g_n_shared_ips", "g_ring_member", "g_ring_risk",
]

ACCOUNT_FEATURES = [
    "account_age_days", "n_txns", "log_total_spend", "mean_amount", "std_amount",
    "max_amount_to_limit", "n_devices", "n_ips", "n_cards", "decline_rate",
    "spend_trajectory", "high_risk_mcc_share", "foreign_ip_share",
    "g_n_users_on_device", "g_n_users_on_ip", "g_n_shared_devices",
    "g_n_shared_ips", "g_degree", "g_component_size", "g_pagerank",
    "g_ring_member", "g_ring_risk",
]


def _assert_no_leakage(cols) -> None:
    bad = [c for c in cols if c.endswith("_gt") or "fraud" in c.lower()]
    if bad:
        raise ValueError(f"Leakage: ground-truth columns in feature set: {bad}")


def _epoch(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts).astype("int64") // 10**9


def build_txn_features(store, graph_feats: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with TXN_FEATURES + meta columns (txn_id,user_id,ts,amount,is_fraud_gt)."""
    t = store.table("transactions")
    users = store.table("users")[["user_id", "country", "credit_limit"]]
    ips = store.table("ips")[["ip_id", "geo_country", "is_datacenter"]]

    df = t.merge(users, on="user_id", how="left").merge(ips, on="ip_id", how="left")
    df["log_amount"] = np.log1p(df["amount"])
    # user amount stats (window-level; documented mild non-causality, acceptable for MVP)
    grp = df.groupby("user_id")["amount"]
    df["user_mean_amount"] = grp.transform("mean")
    df["user_std_amount"] = grp.transform("std").fillna(1.0).replace(0, 1.0)
    df["amount_z"] = (df["amount"] - df["user_mean_amount"]) / df["user_std_amount"]
    df["amount_to_limit"] = df["amount"] / df["credit_limit"].clip(lower=1.0)
    df["hour"] = pd.to_datetime(df["ts"]).dt.hour
    df["is_high_risk_mcc"] = df["mcc"].isin(HIGH_RISK_MCC).astype(int)
    df["is_datacenter_ip"] = df["is_datacenter"].fillna(0).astype(int)
    df["geo_mismatch"] = (df["geo_country"] != df["country"]).astype(int)
    df["is_declined"] = (df["status"] == "declined").astype(int)
    df["merchant_high_risk"] = df["is_high_risk_mcc"]
    df["user_decline_rate"] = df.groupby("user_id")["is_declined"].transform("mean")
    df["velocity_24h"] = _velocity_24h(df)

    df = df.merge(graph_feats, on="user_id", how="left")
    for c in ["g_n_users_on_device", "g_n_users_on_ip", "g_n_shared_devices",
              "g_n_shared_ips", "g_ring_member", "g_ring_risk"]:
        df[c] = df[c].fillna(0)

    _assert_no_leakage(TXN_FEATURES)
    keep = TXN_FEATURES + ["txn_id", "user_id", "ts", "amount", "is_fraud_gt"]
    return df[keep].fillna(0)


def _velocity_24h(df: pd.DataFrame) -> pd.Series:
    """Count of the same user's transactions in the prior 24h (causal)."""
    out = pd.Series(0, index=df.index, dtype=int)
    sec = _epoch(df["ts"])
    for _, idx in df.groupby("user_id").groups.items():
        order = sec.loc[idx].sort_values()
        times = order.values
        # for each txn, number of prior txns within 24h
        lo = np.searchsorted(times, times - 86400, side="left")
        pos = np.arange(len(times))
        counts = pos - lo
        out.loc[order.index] = counts
    return out


def build_account_features(store, graph_feats: pd.DataFrame) -> pd.DataFrame:
    """Per-user frame with ACCOUNT_FEATURES + meta (user_id, is_fraud_gt)."""
    users = store.table("users")
    t = store.table("transactions").merge(
        users[["user_id", "credit_limit", "country"]], on="user_id", how="left"
    )
    ips = store.table("ips")[["ip_id", "geo_country"]]
    t = t.merge(ips, on="ip_id", how="left")
    t["is_declined"] = (t["status"] == "declined").astype(int)
    t["high_risk_mcc"] = t["mcc"].isin(HIGH_RISK_MCC).astype(int)
    t["foreign_ip"] = (t["geo_country"] != t["country"]).astype(int)
    t["ts_dt"] = pd.to_datetime(t["ts"])
    t["amount_to_limit"] = t["amount"] / t["credit_limit"].clip(lower=1.0)

    last_ts = t["ts_dt"].max()

    agg = t.groupby("user_id").agg(
        n_txns=("txn_id", "count"),
        total_spend=("amount", "sum"),
        mean_amount=("amount", "mean"),
        std_amount=("amount", "std"),
        max_amount_to_limit=("amount_to_limit", "max"),
        n_devices=("device_id", "nunique"),
        n_ips=("ip_id", "nunique"),
        n_cards=("card_id", "nunique"),
        decline_rate=("is_declined", "mean"),
        high_risk_mcc_share=("high_risk_mcc", "mean"),
        foreign_ip_share=("foreign_ip", "mean"),
    ).reset_index()

    # spend trajectory: last-7-day spend vs prior mean weekly spend (bust-out signal)
    cutoff = last_ts - pd.Timedelta(days=7)
    recent = t[t["ts_dt"] >= cutoff].groupby("user_id")["amount"].sum()
    traj = (recent / agg.set_index("user_id")["total_spend"].clip(lower=1.0)).fillna(0)
    agg["spend_trajectory"] = agg["user_id"].map(traj).fillna(0)

    df = users.merge(agg, on="user_id", how="left")
    df["account_age_days"] = (BASE_DATE - pd.to_datetime(df["created_at"])).dt.days.abs()
    df["log_total_spend"] = np.log1p(df["total_spend"].fillna(0))
    df = df.merge(graph_feats, on="user_id", how="left")

    _assert_no_leakage(ACCOUNT_FEATURES)
    for c in ACCOUNT_FEATURES:
        if c not in df.columns:
            df[c] = 0
    keep = ACCOUNT_FEATURES + ["user_id", "is_fraud_gt"]
    return df[keep].fillna(0)
