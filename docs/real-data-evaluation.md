# Real-Data Evaluation: Methodology & Rationale

This report documents how FraudGuard's modeling stack is validated on a real,
external fraud dataset, and — more importantly — **why each procedure was
chosen**. The accompanying code is [`scripts/real_data_test.py`](../scripts/real_data_test.py).

---

## 1. Purpose

FraudGuard is developed and CI-tested entirely on a *deterministic synthetic
world* (seeded simulator + injected fraud archetypes). That guarantees
reproducible tests, but it raises a fair question:

> Does the modeling code actually work on **real**, severely imbalanced fraud
> data — or only on a world we built to be detectable?

This evaluation answers that by running the project's own model components,
unchanged, against a public real-world dataset and reporting honest,
production-relevant metrics.

**Headline result (held-out temporal tail):**

| Metric | Value |
|---|---|
| ROC-AUC | 0.972 |
| PR-AUC | 0.800 (vs 0.13% base rate) |
| Expected Calibration Error (raw → calibrated) | 0.0004 → 0.0002 |
| Cost-optimal threshold | 0.80 → 0.04, expected loss **$4,290 → $3,742** |

---

## 2. Dataset

**Credit Card Fraud Detection**, ULB (Université Libre de Bruxelles) Machine
Learning Group — <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>
(CC BY-SA 4.0).

- 284,807 real transactions by European cardholders, September 2013.
- 492 frauds → **0.173%** positive rate (severe imbalance).
- Features: `Time`, `Amount`, and `V1`–`V28` (anonymized PCA components — the
  original features were transformed to protect cardholder privacy).
- Label: `Class` (1 = fraud).

The CSV is fetched on demand into `data/real/` (gitignored, ~100 MB); it is not
committed.

> **Why this dataset.** It is the canonical, genuinely-real fraud benchmark with
> a published label and a realistic class imbalance. Its limitation — it is a
> *flat* transaction log with no relational entities — is exactly why only part
> of FraudGuard can be tested on it (see §6).

---

## 3. Procedure, step by step

Each step lists **what** was done and **why**.

### 3.1 Feature selection

**What.** Use `V1`–`V28` + `Amount` (29 features). Exclude `Time` from the
feature set.

**Why.** `Time` is *seconds elapsed since the first transaction* — it encodes
absolute position in the data window. Because we split the data by time
(§3.2), feeding `Time` as a feature would let the model key on "how late in the
window" a row is, which is a proxy for the split boundary rather than a genuine
fraud signal. We therefore use `Time` **only to order the split**, never as a
predictor. `Amount` is kept raw: the model is tree-based and invariant to
monotonic transforms, so log-scaling buys nothing.

### 3.2 Temporal train / calibration / test split

**What.** Sort all rows by `Time`, then cut:

```
|------------------ train (75%) ------------------|---- test (25%) ----|
|-------- fit (≈85% of train) --------|-- cal --|
```

- fit  = 181,566 rows  (train the model)
- cal  =  32,040 rows  (fit the probability calibrator)
- test =  71,201 rows  (final, untouched evaluation)

The fractions mirror `config.yaml` (`test_fraction: 0.25`,
`calibration_fraction: 0.15`) and the discipline mirrors
`src/models/pipeline.py`.

**Why temporal, not random.** Fraud is a *moving target* — tactics drift over
time. A random (shuffled) split lets transactions from the future leak into
training, which inflates every metric and is impossible in production (you
cannot train on transactions that haven't happened yet). A temporal split
simulates the real task: *train on the past, predict the future.* It is strictly
harder, and it surfaces real distribution shift — here the fraud rate drifts
from **0.20% in the fit window to 0.13% in the test window**. A random split
hides that drift entirely.

**Why a separate calibration split.** The calibrator (§3.5) must be fit on data
the *model* has never seen, otherwise the model's overconfidence on its own
training rows would be baked into the calibration map. Carving `cal` from the
**tail of the train block** (not from test) keeps the test set pristine and
preserves the no-look-ahead ordering.

### 3.3 Model and imbalance strategy

**What.** Train `TxnRiskModel` — an XGBoost classifier with
`scale_pos_weight = n_neg / n_pos` — using the same hyperparameters as the
synthetic pipeline (`n_estimators=300, max_depth=5, learning_rate=0.08`). **No
resampling of the data.**

**Why cost-sensitive learning instead of under/over-sampling.** Two common
imbalance fixes are random undersampling (throw away most negatives to reach
50/50) and SMOTE (synthesize minority examples). Both **change the base rate the
model sees**, which has three costs:

1. **Discarded signal / synthetic artifacts.** Undersampling throws away
   >99% of the legitimate transactions; SMOTE invents minority points that may
   not lie on the real fraud manifold.
2. **Broken calibration.** After resampling, a predicted probability no longer
   reflects the true 0.17% prior — a model trained on 50/50 data thinks fraud is
   a coin flip. That makes cost-based decisions (§3.7) meaningless without
   ad-hoc correction.
3. **Optimistic metrics.** Scores computed on a balanced subsample look great
   but don't transfer to the real distribution (the classic "high recall, but a
   flood of false positives in production" failure).

`scale_pos_weight` instead reweights the *loss function* so each fraud counts as
much as the ~580 negatives it's outnumbered by. The model still sees the true
distribution, so its probabilities stay meaningful and calibratable.

### 3.4 Scoring

**What.** Produce raw fraud probabilities for `cal` and `test` via
`predict_proba`.

**Why split raw vs. calibrated.** FraudGuard deliberately keeps the
*uncalibrated* model output and the *calibrated* probability as distinct stages
(`src/models/base.py` docstring). This makes the calibration improvement
explicit and measurable rather than hidden inside the classifier.

### 3.5 Probability calibration

**What.** `Calibrator.fit_best(y_cal, p_cal)` fits **both** isotonic regression
and Platt (sigmoid) scaling on the calibration split, then keeps whichever gives
the lower Expected Calibration Error. Here it chose **isotonic**.

**Why calibrate at all.** A raw classifier score is good for *ranking* but not
necessarily a true probability. Any downstream **cost** decision ("block if
P(fraud) high enough to beat the friction cost") requires the score to *mean*
what it says — a 0.8 should correspond to ~80% empirical fraud rate. Calibration
enforces that.

**Why try both methods.** Isotonic is flexible (non-parametric, monotonic) but
can overfit on small calibration sets; Platt is a stable 2-parameter fit but
assumes a sigmoid shape. Fitting both and selecting by ECE picks the right
bias/variance tradeoff for the data at hand instead of hard-coding a guess.

### 3.6 Evaluation metrics

**What.** Report ROC-AUC, **PR-AUC (average precision)**, ECE and Brier score
(raw vs. calibrated), and a reliability curve.

**Why PR-AUC is the headline, not ROC-AUC.** Under 0.17% prevalence, ROC-AUC is
nearly saturated for any competent model — it rewards correctly ranking the tens
of thousands of obvious negatives, which is easy, and it is insensitive to the
false-positive *volume* that actually hurts at scale. Precision-Recall AUC
focuses on the minority class and directly reflects the precision/recall
tradeoff a fraud team lives with. We report ROC-AUC too, but only for comparison
with the literature.

**Why ECE + Brier.** ECE measures *calibration* (do predicted probabilities
match empirical frequencies?); Brier measures *sharpness + calibration*
together. Reporting both, before and after calibration, quantifies what §3.5
bought.

### 3.7 Cost-based decision threshold

**What.** Sweep the calibrated score over candidate thresholds and pick the one
minimizing **expected dollar loss**, using FraudGuard's cost model:

- **False negative** (missed fraud) costs the **transaction's dollar amount**.
- **False positive** (blocking a legit transaction) costs a fixed **$25**
  friction penalty (`config.yaml: fp_cost`).

**Why optimize loss, not accuracy or F1.** Accuracy is useless at 0.17%
prevalence (predicting "never fraud" scores 99.83%). F1 treats precision and
recall as equally important, which is arbitrary. The business actually trades
*dollars of fraud* against *dollars of customer friction*, and those costs are
asymmetric and known. Minimizing expected loss turns the model score into an
**operational policy** with an explicit, defensible threshold — not just an
offline number.

On this data the loss-optimal threshold is **0.04**, far below the default
auto-block line of 0.80: because the average fraud here is small in dollars
relative to a $25 false-positive cost, the optimizer accepts more false
positives to recover more fraud — and that is the *correct* economic call given
the stated costs.

---

## 4. Results in detail

```
rows=284,807  features=29 (V1-V28 + Amount)
split (temporal by Time):  fit=181,566  cal=32,040  test=71,201
fraud rate  fit=0.2005%  cal=0.1061%  test=0.1320%

ROC-AUC = 0.972
PR-AUC  = 0.800     (baseline = fraud rate = 0.132%)
calibration (isotonic):  ECE 0.0004 -> 0.0002 ; Brier 0.0004 -> 0.0004

expected $loss (FN = txn amount, FP = $25):
  total fraud $ exposure in test = $10,839
  default t=0.80   recall=0.681 precision=0.985  TP=64 FP=1  FN=30  loss=$4,290
  optimal t=0.04   recall=0.809 precision=0.585  TP=76 FP=54 FN=18  loss=$3,742

top features: V14 (0.49), V4, V10, V12, V8 ...
```

**Reading the results.**

- **PR-AUC 0.80** against a 0.13% base rate is strong for an *honest temporal
  split* — the model recovers most of the precision/recall frontier without
  resampling tricks.
- **Calibration was already excellent** (ECE 0.0004): the model correctly
  assigns ≈0 probability to the 71,124 obvious negatives. Isotonic still halves
  the residual error. The reliability curve confirms the high-confidence bins
  (predicted ≈0.999) are ~100% fraud.
- **The cost optimizer changes the decision**, not the model. Moving the
  threshold from 0.80 to 0.04 cuts expected loss ~13% by catching 12 more frauds
  at the price of 53 extra false positives — a trade the $25-vs-amount cost
  structure justifies.
- **V14 dominates** feature importance, matching every published analysis of
  this dataset — a sanity check that the pipeline learns real signal, not an
  artifact of our harness.

---

## 5. Why these choices, in one paragraph

Every decision points the same direction: **measure performance the way
production would experience it.** Temporal split (not random) reflects that we
predict the future. Cost-sensitive learning (not resampling) keeps the true base
rate so probabilities stay meaningful. Calibration makes those probabilities
trustworthy. PR-AUC and expected-dollar-loss (not accuracy or ROC-AUC alone)
measure what actually matters under extreme imbalance. The result is a number we
can defend to a risk owner, not a leaderboard score.

---

## 6. Scope and limitations

This evaluation exercises only the **supervised modeling half** of FraudGuard:

- ✅ tested: `TxnRiskModel`, `Calibrator`, the metrics suite, and the
  expected-loss threshold logic.
- ❌ **not** tested: graph-ring detection, the agent investigator, and the
  orchestrator's routing/decision layer.

**Why the gap is unavoidable here.** Those components operate on *relationships*
between entities — shared devices, IPs, identities, and the rings they form. The
ULB dataset is a flat, fully-anonymized transaction log with none of those
entities, so there is literally nothing for the graph or agent stack to consume.
Validating them requires a **relational** dataset (e.g., a transaction log that
retains merchant / device / geo / account links); that is a separate evaluation.

Two further caveats:

1. **Features are the dataset's PCA components**, not FraudGuard's named feature
   engineering (`velocity_24h`, `geo_mismatch`, graph features). So this
   validates the *model + calibration + cost* code, not the feature pipeline.
2. **Metrics are not directly comparable to resampling-based notebooks** that
   report scores on balanced subsamples or random splits — those setups are
   easier and tend to overstate real-world performance.

---

## 7. Reproducibility

```bash
# dataset is fetched on demand into data/real/ (gitignored)
uv run python scripts/real_data_test.py
```

The script is deterministic (fixed model `random_state`, fixed split fractions),
reuses the project's own components, and prints every number in this report.

---

## 8. Conclusion

On the metric that matters under severe imbalance — PR-AUC on an honest temporal
split — FraudGuard's model code performs strongly on real data (0.80), with
well-calibrated probabilities and a cost-justified operating point. ROC-AUC
(0.972) is competitive with the best published models on this dataset while
being measured on a deliberately harder split. The evaluation is intentionally
conservative: it makes no resampling shortcuts, hides no future leakage, and is
explicit about the half of the system it cannot test on flat data.
