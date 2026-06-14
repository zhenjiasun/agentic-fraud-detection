"""Planted money-mule rings (shared infrastructure) are recovered."""
from __future__ import annotations


def test_rings_detected(pipeline):
    rings = pipeline["rings"]
    assert len(rings) >= 1


def test_rings_are_mostly_fraud(pipeline):
    """Detected rings should concentrate fraud (high ground-truth saturation)."""
    rings = pipeline["rings"]
    saturated = [r for r in rings if r.get("fraud_saturation_gt", 0) >= 0.5]
    assert len(saturated) >= 1
    # the highest-risk ring should be essentially all fraud (a mule cluster)
    top = max(rings, key=lambda r: r["risk_score"])
    assert top["fraud_saturation_gt"] >= 0.8


def test_ring_features_present(pipeline):
    from src.graph.features import graph_features
    gf = graph_features(pipeline["graph"], pipeline["rings"])
    assert "g_ring_member" in gf.columns
    assert gf["g_ring_member"].sum() >= 1
    # no ground-truth leaked into graph features
    assert not any(c.endswith("_gt") for c in gf.columns)
