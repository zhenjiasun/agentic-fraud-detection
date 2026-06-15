"""Evaluation: confusion metrics, precision/recall, PR/ROC-AUC, and expected $
loss — including the loss-vs-threshold curve that feeds the orchestrator's
threshold choice. Ground truth (`is_fraud_gt`) is consumed ONLY here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.analysis.calibration_report import calibration_section
from src.analysis.disparity import disparity_section


def _account_frame(store) -> pd.DataFrame:
    users = store.table("users")[["user_id", "is_fraud_gt", "country", "segment"]]
    scores = store.table("account_scores")[["user_id", "score"]]
    # latest orchestrator decision per user
    dec = store.query_df(
        "SELECT subject_id user_id, action FROM decisions WHERE source='orchestrator'"
    )
    # per-user realized fraud loss (sum of fraudulent transaction amounts)
    loss = store.query_df(
        "SELECT user_id, SUM(amount) fraud_amount FROM transactions "
        "WHERE is_fraud_gt=1 GROUP BY user_id"
    )
    # per-user exposure = total spend (observable, all txns) — the $ at stake used by
    # the amount_aware policy comparison. Distinct from fraud_amount (ground truth).
    expo = store.query_df(
        "SELECT user_id, SUM(amount) exposure FROM transactions GROUP BY user_id"
    )
    df = (users.merge(scores, on="user_id", how="left")
          .merge(dec, on="user_id", how="left")
          .merge(loss, on="user_id", how="left")
          .merge(expo, on="user_id", how="left"))
    df["score"] = df["score"].fillna(0)
    df["action"] = df["action"].fillna("auto_allow")
    df["fraud_amount"] = df["fraud_amount"].fillna(0)
    df["exposure"] = df["exposure"].fillna(0)
    df["pred_pos"] = df["action"].isin(["auto_block", "route_to_review"]).astype(int)
    return df


def _confusion(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4),
            "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) else 0.0}


def expected_loss_curve(df: pd.DataFrame, fp_cost: float) -> dict:
    thresholds = np.round(np.linspace(0.02, 0.98, 49), 3)
    losses = []
    for t in thresholds:
        pred = (df["score"] >= t).astype(int)
        fn_loss = df.loc[(df["is_fraud_gt"] == 1) & (pred == 0), "fraud_amount"].sum()
        fp_count = int(((df["is_fraud_gt"] == 0) & (pred == 1)).sum())
        losses.append(float(fn_loss + fp_count * fp_cost))
    best_i = int(np.argmin(losses))
    return {"thresholds": thresholds.tolist(), "expected_loss": losses,
            "recommended_threshold": float(thresholds[best_i]),
            "min_expected_loss": round(losses[best_i], 2)}


def amount_aware_loss(df: pd.DataFrame, fp_cost: float) -> dict:
    """Realized loss of the per-account amount_aware policy vs the best single cutoff.

    The amount_aware rule blocks when score*exposure > (1-score)*fp_cost (the
    loss-minimizing decision, cutoff = fp_cost/exposure). Reported alongside the
    global-threshold optimum so the two policies are A/B-comparable in the report.
    """
    score = df["score"].to_numpy()
    expo = df["exposure"].to_numpy()
    y = df["is_fraud_gt"].to_numpy()
    fraud_amt = df["fraud_amount"].to_numpy()

    block = score * expo > (1.0 - score) * fp_cost
    fn_loss = float(fraud_amt[(y == 1) & (~block)].sum())
    fp_count = int(((y == 0) & block).sum())
    aa_loss = fn_loss + fp_count * fp_cost

    best_global = expected_loss_curve(df, fp_cost)["min_expected_loss"]
    improvement = (best_global - aa_loss) / best_global if best_global else 0.0
    return {
        "policy": "amount_aware",
        "expected_loss": round(aa_loss, 2),
        "blocked": int(block.sum()),
        "fp": fp_count,
        "vs_best_global": round(best_global, 2),
        "improvement_pct": round(100.0 * improvement, 1),
    }


def build_report(store, settings) -> dict:
    df = _account_frame(store)
    fp_cost = settings.orchestrator["fp_cost"]
    conf = _confusion(df["is_fraud_gt"], df["pred_pos"])

    y, s = df["is_fraud_gt"].values, df["score"].values
    roc = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")
    pr = float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else float("nan")

    # realized expected loss at the current operating point
    fn_loss = df.loc[(df["is_fraud_gt"] == 1) & (df["pred_pos"] == 0), "fraud_amount"].sum()
    fp_count = conf["fp"]
    operating_loss = float(fn_loss + fp_count * fp_cost)
    loss_curve = expected_loss_curve(df, fp_cost)

    report = {
        "confusion": conf,
        "roc_auc": round(roc, 4), "pr_auc": round(pr, 4),
        "expected_loss": {
            "operating_point": round(operating_loss, 2),
            "fp_cost": fp_cost,
            **loss_curve,
            "amount_aware": amount_aware_loss(df, fp_cost),
        },
        "calibration": calibration_section(df),
        "disparity": disparity_section(df),
        "n_accounts": int(len(df)),
        "fraud_rate": round(float(df["is_fraud_gt"].mean()), 4),
    }
    report["headline"] = (
        f"PR-AUC={report['pr_auc']} recall={conf['recall']} precision={conf['precision']} "
        f"FP={conf['fp']} FN={conf['fn']} exp_loss=${operating_loss:,.0f} "
        f"(min ${loss_curve['min_expected_loss']:,.0f} @ t={loss_curve['recommended_threshold']})"
    )
    return report
