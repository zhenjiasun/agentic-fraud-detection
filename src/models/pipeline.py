"""Train + calibrate + score the risk models, persisting scores to the store.

Splits avoid leakage two ways:
- transaction model: TEMPORAL split (train on earlier, test on later, calibrate
  on the train tail) so there is no look-ahead.
- account model: RING-GROUPED split (GroupShuffleSplit on ring_id) so a single
  fraud ring never straddles train and test.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from src.graph.features import graph_features
from src.graph.rings import ring_membership
from src.log import get_logger
from src.models.account_risk import AccountRiskModel
from src.models.calibration import Calibrator, expected_calibration_error
from src.models.features import (
    ACCOUNT_FEATURES, TXN_FEATURES, build_account_features, build_txn_features,
)
from src.models.registry import save_model
from src.models.txn_risk import TxnRiskModel

log = get_logger("pipeline")


def _safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def _reference_snapshot(X: pd.DataFrame) -> dict:
    """Per-feature quantile edges + mean/std for drift monitoring."""
    ref = {}
    for c in X.columns:
        col = X[c].astype(float)
        ref[c] = {
            "quantiles": [float(q) for q in np.quantile(col, np.linspace(0, 1, 11))],
            "mean": float(col.mean()),
            "std": float(col.std() or 1.0),
        }
    return ref


def train_and_score(store, graph, rings, settings) -> dict:
    gf = graph_features(graph, rings)
    mcfg = settings.models

    # ---------------- transaction-risk: temporal split ----------------
    txn = build_txn_features(store, gf).sort_values("ts").reset_index(drop=True)
    n = len(txn)
    test_n = int(n * mcfg["test_fraction"])
    train_part = txn.iloc[: n - test_n]
    test_part = txn.iloc[n - test_n:]
    cal_n = int(len(train_part) * mcfg["calibration_fraction"])
    fit_part = train_part.iloc[: len(train_part) - cal_n]
    cal_part = train_part.iloc[len(train_part) - cal_n:]

    txn_model = TxnRiskModel(
        n_estimators=mcfg["n_estimators"], max_depth=mcfg["max_depth"],
        learning_rate=mcfg["learning_rate"],
    )
    txn_model.fit(fit_part[TXN_FEATURES], fit_part["is_fraud_gt"].values)
    cal_raw = txn_model.predict_proba(cal_part[TXN_FEATURES])
    txn_cal, txn_cal_metrics = Calibrator.fit_best(cal_part["is_fraud_gt"].values, cal_raw)

    test_raw = txn_model.predict_proba(test_part[TXN_FEATURES])
    test_calp = txn_cal.transform(test_raw)
    txn_metrics = {
        "roc_auc": round(_safe_auc(test_part["is_fraud_gt"], test_raw), 4),
        "pr_auc": round(float(average_precision_score(test_part["is_fraud_gt"], test_raw)), 4),
        **txn_cal_metrics,
        "n_train": int(len(fit_part)), "n_test": int(len(test_part)),
    }
    save_model(settings.saved_models_dir, "txn_risk", txn_model, txn_cal, {
        "model_id": "latest", "trained_at": datetime.now().isoformat(),
        "features": TXN_FEATURES, "metrics": txn_metrics,
        "reference": _reference_snapshot(fit_part[TXN_FEATURES]),
    })

    # score ALL transactions
    all_raw = txn_model.predict_proba(txn[TXN_FEATURES])
    all_cal = txn_cal.transform(all_raw)
    store.insert_df("txn_scores", pd.DataFrame({"txn_id": txn["txn_id"], "score": all_cal}))

    # ---------------- account-risk: ring-grouped split ----------------
    acct = build_account_features(store, gf).reset_index(drop=True)
    membership = ring_membership(rings)
    groups = acct["user_id"].map(lambda u: membership.get(u, {}).get("ring_id", f"solo_{u}"))

    gss = GroupShuffleSplit(n_splits=1, test_size=mcfg["test_fraction"], random_state=0)
    train_idx, test_idx = next(gss.split(acct, acct["is_fraud_gt"], groups))
    a_train, a_test = acct.iloc[train_idx], acct.iloc[test_idx]
    # carve calibration from train by group too
    g2 = GroupShuffleSplit(n_splits=1, test_size=mcfg["calibration_fraction"], random_state=1)
    fit_i, cal_i = next(g2.split(a_train, a_train["is_fraud_gt"], groups.iloc[train_idx]))
    a_fit, a_cal = a_train.iloc[fit_i], a_train.iloc[cal_i]

    acct_model = AccountRiskModel(
        n_estimators=mcfg["n_estimators"], max_depth=mcfg["max_depth"],
        learning_rate=mcfg["learning_rate"],
    )
    acct_model.fit(a_fit[ACCOUNT_FEATURES], a_fit["is_fraud_gt"].values)
    a_cal_raw = acct_model.predict_proba(a_cal[ACCOUNT_FEATURES])
    acct_cal, acct_cal_metrics = Calibrator.fit_best(a_cal["is_fraud_gt"].values, a_cal_raw)

    a_test_raw = acct_model.predict_proba(a_test[ACCOUNT_FEATURES])
    acct_metrics = {
        "roc_auc": round(_safe_auc(a_test["is_fraud_gt"], a_test_raw), 4),
        "pr_auc": round(float(average_precision_score(a_test["is_fraud_gt"], a_test_raw)), 4),
        **acct_cal_metrics,
        "n_train": int(len(a_fit)), "n_test": int(len(a_test)),
    }
    save_model(settings.saved_models_dir, "account_risk", acct_model, acct_cal, {
        "model_id": "latest", "trained_at": datetime.now().isoformat(),
        "features": ACCOUNT_FEATURES, "metrics": acct_metrics,
        "reference": _reference_snapshot(a_fit[ACCOUNT_FEATURES]),
    })

    # score ALL accounts
    raw = acct_model.predict_proba(acct[ACCOUNT_FEATURES])
    cal = acct_cal.transform(raw)
    store.insert_df("account_scores", pd.DataFrame({
        "user_id": acct["user_id"], "raw_score": raw, "score": cal,
    }))

    return {
        "txn": {k: txn_metrics[k] for k in ("roc_auc", "pr_auc", "ece_raw", "ece_calibrated")},
        "account": {k: acct_metrics[k] for k in ("roc_auc", "pr_auc", "ece_raw", "ece_calibrated")},
        "scored_txns": int(len(txn)), "scored_accounts": int(len(acct)),
    }
