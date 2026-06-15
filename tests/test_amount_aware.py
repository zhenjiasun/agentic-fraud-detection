"""Amount-aware thresholding: the decision rule and the policy-loss comparison."""
import pandas as pd

from src.analysis.evaluation import amount_aware_loss
from src.orchestrator.decision import _decide

T_LOW, T_HIGH = 0.15, 0.80


def test_amount_aware_blocks_high_exposure_below_t_high():
    # score 0.5 sits in the review band under the global policy...
    assert _decide([], 0.5, 0, T_LOW, T_HIGH, mode="global") == "route_to_review"
    # ...but with large dollars at stake the loss rule blocks it.
    got = _decide([], 0.5, 0, T_LOW, T_HIGH, mode="amount_aware",
                  exposure=1000.0, fp_cost=25.0)
    assert got == "auto_block"


def test_amount_aware_spares_thin_exposure():
    # same score, tiny exposure: 0.5*10 = 5 < 0.5*25 = 12.5 -> not worth blocking.
    got = _decide([], 0.5, 0, T_LOW, T_HIGH, mode="amount_aware",
                  exposure=10.0, fp_cost=25.0)
    assert got == "route_to_review"


def test_t_high_stays_a_safety_ceiling():
    # near-certain account with no recorded spend still blocks via the t_high ceiling.
    got = _decide([], 0.9, 0, T_LOW, T_HIGH, mode="amount_aware",
                  exposure=0.0, fp_cost=25.0)
    assert got == "auto_block"


def test_global_mode_is_unchanged():
    assert _decide([], 0.9, 0, T_LOW, T_HIGH, mode="global", exposure=1e9) == "auto_block"
    assert _decide([], 0.05, 0, T_LOW, T_HIGH, mode="global", exposure=1e9) == "auto_allow"


def test_amount_aware_loss_report():
    df = pd.DataFrame({
        "score":        [0.60, 0.05, 0.50],
        "exposure":     [1000.0, 50.0, 2000.0],
        "is_fraud_gt":  [1, 0, 0],
        "fraud_amount": [1000.0, 0.0, 0.0],
    })
    out = amount_aware_loss(df, fp_cost=25.0)
    # row0 (fraud) and row2 (legit, big exposure) get blocked; row2 is the one FP.
    assert out["blocked"] == 2
    assert out["fp"] == 1
    assert out["expected_loss"] == 25.0          # one $25 false positive, no missed fraud
    assert set(out) >= {"policy", "vs_best_global", "improvement_pct"}
