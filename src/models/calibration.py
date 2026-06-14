"""Probability calibration + calibration metrics (ECE, Brier).

Fits both isotonic and Platt (sigmoid) calibrators on a held-out calibration
split and picks the one with the lower ECE. Keeps the raw vs calibrated
distinction explicit so the evaluation module can report the improvement.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def brier_score(y_true, y_prob) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def reliability_curve(y_true, y_prob, n_bins: int = 10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    conf, acc, count = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf.append(float(y_prob[mask].mean()))
        acc.append(float(y_true[mask].mean()))
        count.append(int(mask.sum()))
    return {"confidence": conf, "accuracy": acc, "count": count}


class Calibrator:
    """Wraps the chosen 1-D calibrator. method in {isotonic, platt}."""

    def __init__(self, method: str, model):
        self.method = method
        self.model = model

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self.method == "isotonic":
            return np.clip(self.model.predict(p), 0, 1)
        return self.model.predict_proba(p.reshape(-1, 1))[:, 1]

    @classmethod
    def fit_best(cls, y_true, p_raw) -> tuple["Calibrator", dict]:
        y_true = np.asarray(y_true)
        p_raw = np.asarray(p_raw, dtype=float)

        iso = IsotonicRegression(out_of_bounds="clip").fit(p_raw, y_true)
        platt = LogisticRegression(max_iter=1000).fit(p_raw.reshape(-1, 1), y_true)

        p_iso = np.clip(iso.predict(p_raw), 0, 1)
        p_platt = platt.predict_proba(p_raw.reshape(-1, 1))[:, 1]

        ece_raw = expected_calibration_error(y_true, p_raw)
        ece_iso = expected_calibration_error(y_true, p_iso)
        ece_platt = expected_calibration_error(y_true, p_platt)

        if ece_iso <= ece_platt:
            chosen = cls("isotonic", iso)
            ece_cal = ece_iso
        else:
            chosen = cls("platt", platt)
            ece_cal = ece_platt

        metrics = {
            "method": chosen.method,
            "ece_raw": round(ece_raw, 4),
            "ece_calibrated": round(ece_cal, 4),
            "brier_raw": round(brier_score(y_true, p_raw), 4),
            "brier_calibrated": round(
                brier_score(y_true, chosen.transform(p_raw)), 4),
        }
        return chosen, metrics
