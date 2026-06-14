"""Test FraudGuard's modeling pipeline on the real ULB credit-card fraud dataset.

This exercises the *supervised* half of FraudGuard end to end on genuinely real,
severely imbalanced data (284,807 European card transactions, 0.173% fraud):

    TxnRiskModel (XGBoost + imbalance weighting)  ->  Calibrator.fit_best
    (isotonic vs Platt)  ->  PR/ROC-AUC, ECE/Brier, reliability, expected-$loss.

It deliberately does NOT touch the graph-ring / agent / orchestrator stack: the
ULB data is a flat transaction log with no device/IP/identity entities, so those
parts have nothing to operate on. Features here are the dataset's own anonymized
PCA components V1-V28 + Amount (Time is used only to order the temporal split,
never as a feature).

Split discipline mirrors src/models/pipeline.py: temporal train/test, with the
calibrator fit on the tail of the train head (no look-ahead).

Dataset: "Credit Card Fraud Detection" by the ULB Machine Learning Group,
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud (CC BY-SA 4.0). Fetched
on demand into data/real/creditcard.csv (gitignored); not committed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.txn_risk import TxnRiskModel
from src.models.calibration import (
    Calibrator, expected_calibration_error, brier_score, reliability_curve,
)
from sklearn.metrics import average_precision_score, roc_auc_score

CSV = ROOT / "data" / "real" / "creditcard.csv"
TEST_FRACTION = 0.25          # mirrors config.models.test_fraction
CALIBRATION_FRACTION = 0.15   # mirrors config.models.calibration_fraction
FP_COST = 25.0                # mirrors config.orchestrator.fp_cost ($ per false block)
DEFAULT_T_HIGH = 0.80         # mirrors config.orchestrator.t_high (auto_block line)


def main() -> None:
    df = pd.read_csv(CSV)
    feature_cols = [c for c in df.columns if c not in ("Time", "Class")]
    df = df.sort_values("Time").reset_index(drop=True)   # temporal order

    n = len(df)
    test_n = int(n * TEST_FRACTION)
    train_part = df.iloc[: n - test_n]
    test_part = df.iloc[n - test_n:]
    cal_n = int(len(train_part) * CALIBRATION_FRACTION)
    fit_part = train_part.iloc[: len(train_part) - cal_n]
    cal_part = train_part.iloc[len(train_part) - cal_n:]

    def Xy(part):
        return part[feature_cols], part["Class"].to_numpy()

    X_fit, y_fit = Xy(fit_part)
    X_cal, y_cal = Xy(cal_part)
    X_test, y_test = Xy(test_part)

    print("=" * 72)
    print("FraudGuard model pipeline on REAL data — ULB creditcard.csv")
    print("=" * 72)
    print(f"rows={n:,}  features={len(feature_cols)} (V1-V28 + Amount)")
    print(f"split (temporal by Time):  fit={len(X_fit):,}  cal={len(X_cal):,}  test={len(X_test):,}")
    print(f"fraud rate  fit={y_fit.mean():.4%}  cal={y_cal.mean():.4%}  test={y_test.mean():.4%}")
    print()

    # ---- train (FraudGuard's TxnRiskModel; same hyperparams as config.yaml) ----
    model = TxnRiskModel(n_estimators=300, max_depth=5, learning_rate=0.08)
    model.fit(X_fit, y_fit)

    p_cal = model.predict_proba(X_cal)
    p_test_raw = model.predict_proba(X_test)

    # ---- calibrate on held-out cal split, apply to test ----
    calibrator, cal_metrics = Calibrator.fit_best(y_cal, p_cal)
    p_test_cal = calibrator.transform(p_test_raw)

    # ---- ranking metrics (calibration is monotonic -> AUCs unchanged) ----
    roc = roc_auc_score(y_test, p_test_raw)
    pr = average_precision_score(y_test, p_test_raw)

    ece_raw = expected_calibration_error(y_test, p_test_raw)
    ece_cal = expected_calibration_error(y_test, p_test_cal)
    brier_raw = brier_score(y_test, p_test_raw)
    brier_cal = brier_score(y_test, p_test_cal)

    print("--- ranking (test set) ---")
    print(f"ROC-AUC = {roc:.4f}")
    print(f"PR-AUC  = {pr:.4f}   (baseline = fraud rate = {y_test.mean():.4%})")
    print()
    print(f"--- calibration (chosen: {calibrator.method}) ---")
    print(f"ECE   raw={ece_raw:.4f}  ->  calibrated={ece_cal:.4f}")
    print(f"Brier raw={brier_raw:.4f}  ->  calibrated={brier_cal:.4f}")
    print()

    # ---- expected-$loss curve (FraudGuard's cost model: FN=Amount, FP=$25) ----
    amount = test_part["Amount"].to_numpy()
    thresholds = np.round(np.linspace(0.02, 0.98, 49), 3)
    losses = []
    for t in thresholds:
        pred = (p_test_cal >= t).astype(int)
        fn_loss = amount[(y_test == 1) & (pred == 0)].sum()
        fp_count = int(((y_test == 0) & (pred == 1)).sum())
        losses.append(float(fn_loss + fp_count * FP_COST))
    best_i = int(np.argmin(losses))
    t_star = float(thresholds[best_i])

    def op(t):
        pred = (p_test_cal >= t).astype(int)
        tp = int(((y_test == 1) & (pred == 1)).sum())
        fp = int(((y_test == 0) & (pred == 1)).sum())
        fn = int(((y_test == 1) & (pred == 0)).sum())
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        fn_loss = amount[(y_test == 1) & (pred == 0)].sum()
        loss = fn_loss + fp * FP_COST
        return tp, fp, fn, recall, prec, loss

    total_fraud_dollars = amount[y_test == 1].sum()
    print("--- expected $loss (FN cost = txn amount, FP cost = $25) ---")
    print(f"total fraud $ exposure in test = ${total_fraud_dollars:,.0f}")
    for label, t in [("default t_high=0.80", DEFAULT_T_HIGH),
                     (f"loss-optimal t*={t_star}", t_star)]:
        tp, fp, fn, rec, prec, loss = op(t)
        print(f"{label:24s}  recall={rec:.3f} precision={prec:.3f} "
              f"TP={tp} FP={fp} FN={fn}  exp_loss=${loss:,.0f}")
    print()

    # ---- reliability curve (calibrated) ----
    rc = reliability_curve(y_test, p_test_cal)
    print("--- reliability (calibrated, non-empty bins) ---")
    print("  conf  ->  actual   (count)")
    for c, a, k in zip(rc["confidence"], rc["accuracy"], rc["count"]):
        print(f"  {c:.3f} -> {a:.3f}   ({k:,})")
    print()

    # ---- top feature importances ----
    imp = sorted(model.feature_importance().items(), key=lambda kv: -kv[1])[:8]
    print("--- top XGBoost feature importances ---")
    print("  " + ", ".join(f"{f}={v}" for f, v in imp))


if __name__ == "__main__":
    main()
