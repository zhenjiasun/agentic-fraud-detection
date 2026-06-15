"""Sensitivity experiments on the real ULB test set.

Answers a concrete question: which operating decisions are easily testable in
this environment, and how sensitive are they to the configurable knobs?

Two sweeps, both on the held-out temporal test set, reusing the exact
train/calibrate path from scripts/real_data_test.py:

  1. FP-cost sweep   — how the loss-optimal block threshold (and the recall /
                       precision / false-positive count it implies) moves as the
                       false-positive friction cost (config.orchestrator.fp_cost)
                       changes. This is the single biggest lever and it is a
                       *business assumption*, not a model property.

  2. Review-band sweep — how many accounts get routed to human review as the
                       (t_low, t_high) band widens. This trades fraud caught
                       against human-review labor (queue volume).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.txn_risk import TxnRiskModel
from src.models.calibration import Calibrator

CSV = ROOT / "data" / "real" / "creditcard.csv"
TEST_FRACTION, CALIBRATION_FRACTION = 0.25, 0.15


def load_train_score():
    df = pd.read_csv(CSV).sort_values("Time").reset_index(drop=True)
    feats = [c for c in df.columns if c not in ("Time", "Class")]
    n = len(df); test_n = int(n * TEST_FRACTION)
    train, test = df.iloc[: n - test_n], df.iloc[n - test_n:]
    cal_n = int(len(train) * CALIBRATION_FRACTION)
    fit, cal = train.iloc[: len(train) - cal_n], train.iloc[len(train) - cal_n:]

    model = TxnRiskModel(n_estimators=300, max_depth=5, learning_rate=0.08)
    model.fit(fit[feats], fit["Class"].to_numpy())
    calibrator, _ = Calibrator.fit_best(cal["Class"].to_numpy(),
                                        model.predict_proba(cal[feats]))
    p = calibrator.transform(model.predict_proba(test[feats]))
    return p, test["Class"].to_numpy(), test["Amount"].to_numpy()


def at_threshold(p, y, amount, t):
    pred = (p >= t).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    fn_loss = amount[(y == 1) & (pred == 0)].sum()
    return tp, fp, fn, recall, prec, fn_loss


def loss_optimal_threshold(p, y, amount, fp_cost):
    ts = np.round(np.linspace(0.01, 0.99, 99), 3)
    best_t, best_loss = None, float("inf")
    for t in ts:
        _, fp, _, _, _, fn_loss = at_threshold(p, y, amount, t)
        loss = fn_loss + fp * fp_cost
        if loss < best_loss:
            best_loss, best_t = loss, t
    return best_t, best_loss


def main():
    p, y, amount = load_train_score()
    fraud_exposure = amount[y == 1].sum()
    print(f"test rows={len(y):,}  frauds={int(y.sum())}  fraud $ exposure=${fraud_exposure:,.0f}\n")

    # ---- 1. FP-cost sensitivity ----
    print("=== 1. Loss-optimal threshold vs false-positive cost ===")
    print(f"{'fp_cost':>8} {'t*':>6} {'recall':>7} {'prec':>6} {'TP':>4} {'FP':>5} {'FN':>4} {'min_loss':>10}")
    for fp_cost in [1, 5, 10, 25, 50, 100, 250, 500]:
        t, _ = loss_optimal_threshold(p, y, amount, fp_cost)
        tp, fp, fn, rec, prec, fn_loss = at_threshold(p, y, amount, t)
        loss = fn_loss + fp * fp_cost
        print(f"{fp_cost:>8} {t:>6.2f} {rec:>7.3f} {prec:>6.3f} {tp:>4} {fp:>5} {fn:>4} ${loss:>9,.0f}")
    print("  -> t* is driven by the COST ASSUMPTION, not the model. Cheap FPs => block aggressively.\n")

    # ---- 2. Review-band volume (t_low, t_high) ----
    print("=== 2. Human-review queue volume vs (t_low, t_high) band ===")
    print(f"{'t_low':>6} {'t_high':>7} {'auto_allow':>11} {'review':>7} {'auto_block':>11} "
          f"{'fraud_in_review':>15} {'fraud_missed_allow':>18}")
    for t_low, t_high in [(0.05, 0.95), (0.10, 0.90), (0.15, 0.80), (0.20, 0.60), (0.30, 0.50)]:
        allow = p <= t_low
        block = p >= t_high
        review = (~allow) & (~block)
        fraud_in_review = int(y[review].sum())
        fraud_missed = int(y[allow].sum())   # fraud auto-allowed (the dangerous error)
        print(f"{t_low:>6.2f} {t_high:>7.2f} {int(allow.sum()):>11,} {int(review.sum()):>7,} "
              f"{int(block.sum()):>11,} {fraud_in_review:>15} {fraud_missed:>18}")
    print("  -> wider band => more fraud reaches a human (safer) but more review labor (costlier).\n")

    # ---- 3. Amount-aware per-txn decision vs best global threshold ----
    # Expected loss of ALLOW = p*amount ; of BLOCK = (1-p)*fp_cost.
    # Optimal per-txn rule: block iff p*amount > (1-p)*fp_cost  (threshold depends on amount).
    print("=== 3. Amount-aware per-transaction threshold vs best single global threshold ===")
    print(f"{'fp_cost':>8} {'global_loss':>12} {'amount_aware_loss':>18} {'improvement':>12}")
    for fp_cost in [10, 25, 50, 100]:
        # best single global threshold
        t, gloss = loss_optimal_threshold(p, y, amount, fp_cost)
        # amount-aware: per-row optimal decision using the calibrated probability
        block = p * amount > (1.0 - p) * fp_cost
        aw_loss = amount[(y == 1) & (~block)].sum() + int(((y == 0) & block).sum()) * fp_cost
        impr = (gloss - aw_loss) / gloss if gloss else 0.0
        print(f"{fp_cost:>8} ${gloss:>10,.0f} ${aw_loss:>16,.0f} {impr:>11.1%}")
    print("  -> a single threshold is provably suboptimal: the loss-minimizing cutoff is")
    print("     fp_cost/amount, so high-value txns should be blocked at lower probability.")


if __name__ == "__main__":
    main()
