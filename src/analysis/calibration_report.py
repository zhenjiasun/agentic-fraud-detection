"""Calibration reporting on the deployed account scores: reliability curve,
ECE, and Brier (reuses the metric implementations in src/models/calibration.py).
"""
from __future__ import annotations

import pandas as pd

from src.models.calibration import (
    brier_score, expected_calibration_error, reliability_curve,
)


def calibration_section(df: pd.DataFrame) -> dict:
    y = df["is_fraud_gt"].values
    p = df["score"].values
    return {
        "ece": round(expected_calibration_error(y, p), 4),
        "brier": round(brier_score(y, p), 4),
        "reliability_curve": reliability_curve(y, p),
    }
