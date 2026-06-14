# Cost-Sensitive Thresholding & Amount-Aware Decisions

How the block/allow threshold is chosen, why it is a *business* decision rather
than a model property, and the optional **amount-aware** policy that blocks based
on dollars at stake. Experiments are in
[`scripts/threshold_experiments.py`](../scripts/threshold_experiments.py); the
policy lives behind `config.yaml: orchestrator.threshold_mode`.

---

## 1. The threshold is a knob, not a constant

The "loss-optimal threshold" reported anywhere in this project is a function of
the **false-positive cost assumption**, not of the model. Sweeping `fp_cost` on
the real ULB test set (94 frauds, $10,839 exposure) moves the optimum across the
entire range:

| fp_cost | t\* | recall | precision | FP | expected loss |
|--------:|----:|-------:|----------:|---:|--------------:|
| $1   | 0.02 | 0.83 | 0.28 | 205 | $1,961 |
| $25  | 0.03 | 0.81 | 0.59 |  54 | $3,742 |
| $50  | 0.05 | 0.75 | 0.91 |   7 | $3,991 |
| $100 | 0.67 | 0.72 | 0.99 |   1 | $4,067 |
| $500 | 0.84 | 0.61 | 1.00 |   0 | $4,391 |

The loss surface is **not smooth** — note the regime jump between $50 and $100
(t\* leaps 0.05 → 0.67). Eyeballing a threshold is therefore risky; it should be
derived from the real friction cost. This is the single highest-leverage knob in
the system, and it requires no retraining to test.

## 2. The review band has low leverage when calibration is sharp

Widening the `(t_low, t_high)` human-review band barely changes outcomes, because
calibrated scores are **bimodal** (≈0 or ≈1):

| (t_low, t_high) | →review | frauds reaching a human | frauds auto-allowed |
|---|---:|---:|---:|
| (0.05, 0.95) | 20 | 13 | **24** |
| (0.15, 0.80) | 12 |  6 | **24** |
| (0.30, 0.50) |  0 |  0 | **24** |

The constant **24** is the real signal: a quarter of frauds are scored so low
that *no threshold* recovers them. That is a model/feature problem, not a
threshold problem — see [real-data-evaluation.md](real-data-evaluation.md) §6.

## 3. Amount-aware thresholding

A single global cutoff is provably suboptimal under an asymmetric cost model. The
expected loss of each decision is:

```
ALLOW: P(fraud) · amount_at_stake          # we eat the fraud if it is one
BLOCK: (1 - P(fraud)) · fp_cost            # we annoy a legit customer if it isn't
```

Blocking is worth it exactly when `P(fraud) · exposure > (1 - P(fraud)) · fp_cost`,
i.e. the loss-minimizing cutoff is **`fp_cost / exposure`** — high-exposure
accounts should be blocked at a *lower* probability. This is why calibrated
probabilities matter: the rule only works if `P(fraud)` is a real probability.

**Implementation.** `orchestrator.threshold_mode: amount_aware` adds this rule to
`_decide` (`src/orchestrator/decision.py`); `t_high` remains a safety ceiling so a
near-certain account still blocks even on thin recorded spend. `build_report`
emits an `expected_loss.amount_aware` section comparing it head-to-head with the
best single global threshold, so the two policies are A/B-measurable on any run.

## 4. It is data-dependent — which is why it is a flag, not a default

The policy is **not universally better**. Measured both ways:

| Evaluation | "exposure" used | amount-aware vs best global threshold |
|---|---|---|
| Real ULB test set (per **transaction**) | transaction amount | **−21% loss** (better) |
| Synthetic world (per **account**) | account total spend | **+6.7% loss** (worse) |

Why the reversal:

- On the **per-transaction** ULB test, `exposure` is the actual amount at risk on
  that authorization, and fraud amounts are small relative to the $25 FP cost — so
  blocking low-probability-but-cheap-to-block items pays off.
- On the **per-account** synthetic world, `exposure` is a coarse proxy (lifetime
  spend). It conflates "big customer" with "big risk," so the rule over-blocks
  legitimate high-spenders and *adds* false-positive cost. The synthetic fraud
  archetypes (high-value ATO bursts, bust-out toward the credit limit) are already
  well captured by the model, leaving little for the exposure term to add.

**Conclusion.** Ship it **off by default**, instrumented for comparison. Whether
it helps depends on (a) how well `exposure` approximates true dollars-at-risk at
the decision granularity, and (b) the FP/FN cost ratio. Measure it on your data
and cost structure before enabling — the report tells you the answer for free.

## 5. How to A/B test it

```bash
# baseline
#   config.yaml -> orchestrator.threshold_mode: global
uv run python bootstrap.py        # read Evaluation line + expected_loss.amount_aware

# candidate
#   config.yaml -> orchestrator.threshold_mode: amount_aware
uv run python bootstrap.py        # compare the same numbers

# real-data sweeps (fp_cost sensitivity, review band, amount-aware win)
uv run python scripts/threshold_experiments.py
```

The `amount_aware` block in `expected_loss` reports `expected_loss`,
`vs_best_global`, and `improvement_pct` on every run, so the comparison is always
visible regardless of which mode is active.
