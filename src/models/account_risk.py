"""Account-risk model (XGBoost). Same shape as TxnRiskModel, different features."""
from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.models.base import RiskModel


class AccountRiskModel(RiskModel):
    name = "account_risk"

    def __init__(self, **params):
        super().__init__()
        defaults = dict(n_estimators=250, max_depth=4, learning_rate=0.08,
                        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                        n_jobs=4, random_state=0)
        defaults.update(params)
        self.clf = XGBClassifier(**defaults)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "AccountRiskModel":
        self.feature_columns = list(X.columns)
        pos = max(1, int(np.sum(y)))
        neg = max(1, int(len(y) - pos))
        self.clf.set_params(scale_pos_weight=neg / pos)
        self.clf.fit(X.values, np.asarray(y))
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(X[self.feature_columns].values)[:, 1]

    def feature_importance(self) -> dict[str, float]:
        if not self._fitted:
            return {}
        imp = self.clf.feature_importances_
        return {f: float(round(v, 4)) for f, v in zip(self.feature_columns, imp)}
