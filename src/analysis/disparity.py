"""Segment disparity (model governance / fairness): per-slice false-positive,
false-negative, and flag rates across geography, customer segment, and merchant
category, plus the max/min disparity ratio per dimension.
"""
from __future__ import annotations

import pandas as pd


def _slice_rates(df: pd.DataFrame, col: str) -> list[dict]:
    rows = []
    for value, g in df.groupby(col):
        legit = g[g["is_fraud_gt"] == 0]
        fraud = g[g["is_fraud_gt"] == 1]
        fpr = float(legit["pred_pos"].mean()) if len(legit) else 0.0
        fnr = float((fraud["pred_pos"] == 0).mean()) if len(fraud) else 0.0
        rows.append({
            "segment": str(value), "n": int(len(g)),
            "flag_rate": round(float(g["pred_pos"].mean()), 4),
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
        })
    return sorted(rows, key=lambda r: -r["false_positive_rate"])


def _ratio(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if r["n"] >= 20]
    vals = [v for v in vals if v > 0]
    if len(vals) < 2:
        return 1.0
    return round(max(vals) / min(vals), 2)


def disparity_section(df: pd.DataFrame) -> dict:
    out = {}
    for dim in ("country", "segment"):
        if dim in df.columns:
            rows = _slice_rates(df, dim)
            out[dim] = {
                "rows": rows,
                "fpr_disparity_ratio": _ratio(rows, "false_positive_rate"),
            }
    return out
