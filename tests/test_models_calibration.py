"""No ground-truth leakage into features; calibration improves ECE."""
from __future__ import annotations

import numpy as np

from src.models.calibration import Calibrator, expected_calibration_error
from src.models.features import ACCOUNT_FEATURES, TXN_FEATURES


def test_no_gt_in_feature_whitelists():
    for feat in TXN_FEATURES + ACCOUNT_FEATURES:
        assert not feat.endswith("_gt")
        assert "fraud" not in feat.lower()


def test_built_features_exclude_gt(pipeline):
    from src.graph.features import graph_features
    from src.models.features import build_txn_features
    gf = graph_features(pipeline["graph"], pipeline["rings"])
    df = build_txn_features(pipeline["store"], gf)
    feature_cols = [c for c in df.columns if c in TXN_FEATURES]
    assert all(not c.endswith("_gt") for c in feature_cols)


def test_scores_persisted(store):
    n = store.query_df("SELECT COUNT(*) n FROM account_scores").iloc[0]["n"]
    assert n > 0
    rng = store.query_df("SELECT MIN(score) lo, MAX(score) hi FROM account_scores").iloc[0]
    assert 0.0 <= rng["lo"] <= rng["hi"] <= 1.0


def test_calibration_reduces_or_matches_ece():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=2000)
    # deliberately miscalibrated raw scores
    raw = np.clip(y * 0.5 + rng.uniform(0, 0.5, size=2000), 0, 1)
    cal, metrics = Calibrator.fit_best(y, raw)
    assert metrics["ece_calibrated"] <= metrics["ece_raw"] + 1e-9
