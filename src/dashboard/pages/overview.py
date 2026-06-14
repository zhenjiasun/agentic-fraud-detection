"""Overview page: KPI cards + action breakdown + expected-loss headline."""
from __future__ import annotations

import dash_mantine_components as dmc
import plotly.graph_objects as go
from dash import dcc

from src.dashboard.theme import ACTION_COLOR, STATUS_COLOR, badge, stat_card


def render(overview: dict, evaluation: dict, monitoring: dict):
    actions = overview.get("actions", {})
    cases = overview.get("cases", {})
    conf = evaluation.get("confusion", {})
    el = evaluation.get("expected_loss", {})

    kpis = dmc.SimpleGrid(cols={"base": 2, "sm": 4}, spacing="md", children=[
        stat_card("Transactions", f"{overview.get('transactions', 0):,}"),
        stat_card("Fraud rate", f"{overview.get('fraud_rate', 0) * 100:.1f}%", "red"),
        stat_card("Auto-blocked", actions.get("auto_block", 0), "red"),
        stat_card("In review queue", cases.get("OPEN", 0), "yellow"),
        stat_card("PR-AUC", evaluation.get("pr_auc", "—"), "teal"),
        stat_card("Recall", conf.get("recall", "—"), "teal"),
        stat_card("Expected loss", f"${el.get('operating_point', 0):,.0f}", "orange"),
        stat_card("Min loss @ t", f"${el.get('min_expected_loss', 0):,.0f} "
                  f"@ {el.get('recommended_threshold', '—')}", "green"),
    ])

    fig = go.Figure(go.Bar(
        x=list(actions.keys()), y=list(actions.values()),
        marker_color=[{"auto_block": "#e03131", "auto_allow": "#2f9e44",
                       "route_to_review": "#7048e8"}.get(k, "#888") for k in actions],
    ))
    fig.update_layout(template="plotly_dark", height=300, margin=dict(l=10, r=10, t=30, b=10),
                      title="Orchestrator actions", paper_bgcolor="rgba(0,0,0,0)")

    loss_fig = go.Figure(go.Scatter(
        x=el.get("thresholds", []), y=el.get("expected_loss", []), mode="lines",
        line=dict(color="#f08c00")))
    loss_fig.update_layout(template="plotly_dark", height=300,
                           margin=dict(l=10, r=10, t=30, b=10),
                           title="Expected $ loss vs block threshold",
                           paper_bgcolor="rgba(0,0,0,0)")

    return dmc.Stack([
        kpis,
        dmc.Group([badge(f"monitoring: {monitoring.get('headline', '')}", "blue")]),
        dmc.SimpleGrid(cols={"base": 1, "md": 2}, children=[
            dcc.Graph(figure=fig), dcc.Graph(figure=loss_fig)]),
    ])
