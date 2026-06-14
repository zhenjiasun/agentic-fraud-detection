"""Feature/score drift (PSI + KS) plus the monitoring snapshot that bundles
drift, data-quality, and adversarial-behavior checks.

Drift compares the most recent scoring window against the training-distribution
snapshot stored in the model's meta.json (the deciles captured at train time).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.adversarial import adversarial_section
from src.analysis.data_quality import data_quality_section
from src.graph.builder import build_graph
from src.graph.features import graph_features
from src.graph.rings import detect_rings
from src.log import get_logger
from src.models.features import TXN_FEATURES, build_txn_features

log = get_logger("drift")


def psi(reference_quantiles: list[float], current: np.ndarray) -> float:
    """Population Stability Index using the reference deciles as bin edges."""
    edges = np.array(reference_quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    cur_counts, _ = np.histogram(current, bins=edges)
    cur_pct = np.clip(cur_counts / max(1, cur_counts.sum()), 1e-4, None)
    ref_pct = np.full(len(cur_pct), 1.0 / len(cur_pct))  # deciles => ~uniform mass
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_stat(reference_quantiles: list[float], current: np.ndarray) -> float:
    """Approximate KS: max gap between reference and current CDF at decile edges."""
    edges = np.array(reference_quantiles)
    n = len(current)
    if n == 0:
        return 0.0
    ref_cdf = np.linspace(0, 1, len(edges))
    cur_cdf = np.array([(current <= e).mean() for e in edges])
    return float(np.max(np.abs(ref_cdf - cur_cdf)))


def _band(psi_val: float, warn: float, alert: float) -> str:
    if psi_val >= alert:
        return "alert"
    if psi_val >= warn:
        return "warn"
    return "stable"


def feature_drift(store, settings) -> dict:
    meta_path = settings.saved_models_dir / "txn_risk" / "latest" / "meta.json"
    if not Path(meta_path).exists():
        return {"available": False}
    reference = json.loads(Path(meta_path).read_text()).get("reference", {})

    graph = build_graph(store)
    rings = detect_rings(graph, settings, store=store)
    gf = graph_features(graph, rings)
    txn = build_txn_features(store, gf).sort_values("ts")
    current = txn.iloc[int(len(txn) * 0.75):]  # most recent scoring window

    warn = settings.monitoring["psi_warn"]
    alert = settings.monitoring["psi_alert"]
    rows = []
    for feat in TXN_FEATURES:
        if feat not in reference or feat not in current.columns:
            continue
        vals = current[feat].astype(float).values
        p = round(psi(reference[feat]["quantiles"], vals), 4)
        rows.append({"feature": feat, "psi": p, "ks": round(
            ks_stat(reference[feat]["quantiles"], vals), 4),
            "band": _band(p, warn, alert)})
    rows.sort(key=lambda r: -r["psi"])
    return {"available": True, "features": rows,
            "n_alert": sum(1 for r in rows if r["band"] == "alert"),
            "n_warn": sum(1 for r in rows if r["band"] == "warn")}


def monitoring_snapshot(store, settings) -> dict:
    drift = feature_drift(store, settings)
    dq = data_quality_section(store)
    adv = adversarial_section(store, settings)
    headline = (
        f"drift: {drift.get('n_alert', 0)} alert / {drift.get('n_warn', 0)} warn | "
        f"data-quality: {dq['status']} ({dq['n_fail']} fail) | "
        f"adversarial: {len(adv['alerts'])} alert(s)"
    )
    return {"drift": drift, "data_quality": dq, "adversarial": adv, "headline": headline}
