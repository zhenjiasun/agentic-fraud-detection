"""Evaluation page: confusion, calibration reliability curve, segment disparity."""
from __future__ import annotations

import dash_mantine_components as dmc
import plotly.graph_objects as go
from dash import dcc

from src.dashboard.theme import badge, section_title, stat_card


def render(ev: dict):
    conf = ev.get("confusion", {})
    cal = ev.get("calibration", {})
    disp = ev.get("disparity", {})

    cards = dmc.SimpleGrid(cols={"base": 2, "sm": 4}, children=[
        stat_card("Precision", conf.get("precision", "—"), "teal"),
        stat_card("Recall", conf.get("recall", "—"), "teal"),
        stat_card("PR-AUC", ev.get("pr_auc", "—"), "teal"),
        stat_card("ROC-AUC", ev.get("roc_auc", "—"), "teal"),
        stat_card("False positives", conf.get("fp", "—"), "orange"),
        stat_card("False negatives", conf.get("fn", "—"), "red"),
        stat_card("ECE", cal.get("ece", "—"), "grape"),
        stat_card("Brier", cal.get("brier", "—"), "grape"),
    ])

    rc = cal.get("reliability_curve", {})
    rel = go.Figure()
    rel.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color="#888"), name="perfect"))
    rel.add_trace(go.Scatter(x=rc.get("confidence", []), y=rc.get("accuracy", []),
                             mode="lines+markers", name="model", line=dict(color="#4dabf7")))
    rel.update_layout(template="plotly_dark", height=320, title="Calibration reliability",
                      xaxis_title="predicted", yaxis_title="observed",
                      margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")

    disparity_blocks = []
    for dim, d in disp.items():
        head = ["Segment", "N", "Flag rate", "FPR", "FNR"]
        body = [[r["segment"], r["n"], r["flag_rate"], r["false_positive_rate"],
                 r["false_negative_rate"]] for r in d["rows"]]
        disparity_blocks.append(dmc.Stack([
            dmc.Group([dmc.Text(f"Disparity by {dim}", fw=600),
                       badge(f"FPR ratio {d['fpr_disparity_ratio']}×",
                             "red" if d["fpr_disparity_ratio"] > 2 else "green")]),
            dmc.Table(data={"head": head, "body": body}, striped=True),
        ]))

    return dmc.Stack([
        cards,
        dcc.Graph(figure=rel),
        section_title("Segment disparity (model governance)"),
        dmc.SimpleGrid(cols={"base": 1, "md": 2}, children=disparity_blocks),
    ])
