"""Evaluation arithmetic + report structure."""
from __future__ import annotations

import pandas as pd

from src.analysis.evaluation import _confusion, build_report, expected_loss_curve


def test_confusion_arithmetic():
    c = _confusion([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])
    assert c["tp"] == 2 and c["fn"] == 1 and c["fp"] == 1 and c["tn"] == 1
    assert c["precision"] == round(2 / 3, 4)
    assert c["recall"] == round(2 / 3, 4)


def test_expected_loss_curve_picks_min():
    df = pd.DataFrame({
        "score": [0.9, 0.1, 0.95, 0.05],
        "is_fraud_gt": [1, 0, 1, 0],
        "fraud_amount": [1000, 0, 1000, 0],
    })
    curve = expected_loss_curve(df, fp_cost=10.0)
    # blocking only the two fraud accounts (high threshold but below 0.9) is optimal
    assert curve["min_expected_loss"] == 0.0


def test_build_report_shape(store, settings):
    rep = build_report(store, settings)
    for key in ("confusion", "roc_auc", "pr_auc", "expected_loss",
                "calibration", "disparity", "headline"):
        assert key in rep
    assert "recommended_threshold" in rep["expected_loss"]
    assert 0 <= rep["pr_auc"] <= 1
