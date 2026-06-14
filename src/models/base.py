"""Abstract risk-model base (mirrors the project's existing model conventions)."""
from __future__ import annotations

import abc

import numpy as np
import pandas as pd


class RiskModel(abc.ABC):
    """A binary risk classifier producing an uncalibrated fraud probability.

    Calibration is applied separately (src/models/calibration.py) so the raw vs
    calibrated distinction stays explicit and evaluable.
    """

    name: str = "risk_model"

    def __init__(self) -> None:
        self._fitted = False
        self.feature_columns: list[str] = []
        self.training_metrics: dict = {}

    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RiskModel": ...

    @abc.abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(fraud) in [0,1] for each row."""

    @property
    def is_fitted(self) -> bool:
        return self._fitted
