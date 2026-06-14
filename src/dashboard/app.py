"""FraudGuard Dash dashboard (dark theme). Pure client of the FastAPI backend.

Tabs: Overview · Review Queue · Rings · Evaluation · Monitoring · Audit.
The Review Queue tab drives the human-in-the-loop flow: pick a case, run the
bounded investigator, read its recommendation, then resolve as a human.
"""
from __future__ import annotations

import dash_mantine_components as dmc
from dash import Dash, Input, Output, callback, ctx, dcc, html

from src.dashboard import api_client as api
from src.dashboard.pages import audit, evaluation, monitoring, overview
from src.dashboard.pages import queue as queue_page
from src.dashboard.pages import rings as rings_page

app = Dash(__name__, suppress_callback_exceptions=True, title="FraudGuard")

TABS = [("Overview", "overview"), ("Review Queue", "queue"), ("Rings", "rings"),
        ("Evaluation", "evaluation"), ("Monitoring", "monitoring"), ("Audit", "audit")]

app.layout = dmc.MantineProvider(
    forceColorScheme="dark",
    children=dmc.Container(fluid=True, p="lg", children=[
        dmc.Group([
            dmc.Title("FraudGuard", order=2),
            dmc.Badge("Agentic Fraud & Abuse Detection", color="indigo", variant="light"),
        ], mb="xs"),
        dmc.Text("Simulated payments · fraud-ring graph · risk models · rules engine · "
                 "bounded LLM investigator · evaluation · monitoring · audited review queue",
                 size="sm", c="dimmed", mb="md"),
        dmc.Tabs(value="overview", id="tabs", children=[
            dmc.TabsList([dmc.TabsTab(label, value=val) for label, val in TABS]),
        ]),
        dmc.Space(h="md"),
        dcc.Loading(html.Div(id="content")),
    ]),
)


def _error(msg: str):
    return dmc.Alert(f"Could not reach the API at {api.BASE}. Start it with "
                     f"`python run_api.py`. ({msg})", title="API unavailable",
                     color="red")


@callback(Output("content", "children"), Input("tabs", "value"))
def render_tab(tab):
    try:
        if tab == "overview":
            return overview.render(api.get("/metrics/overview"),
                                   api.get("/metrics/evaluation"),
                                   api.get("/metrics/monitoring"))
        if tab == "queue":
            cases = api.get("/cases", limit=300)
            actionable = [c for c in cases if c["status"] in
                          ("OPEN", "IN_REVIEW", "AWAITING_DECISION")] or cases
            return queue_page.render_picker(actionable)
        if tab == "rings":
            return rings_page.render(api.get("/rings"))
        if tab == "evaluation":
            return evaluation.render(api.get("/metrics/evaluation"))
        if tab == "monitoring":
            return monitoring.render(api.get("/metrics/monitoring"))
        if tab == "audit":
            return audit.render(api.get("/audit", limit=200), api.get("/audit/verify"))
    except Exception as e:  # API down or transient error
        return _error(str(e))
    return dmc.Text("Select a tab")


@callback(
    Output("case-detail", "children"),
    Input("case-select", "value"),
    Input("btn-investigate", "n_clicks"),
    Input("btn-block", "n_clicks"),
    Input("btn-allow", "n_clicks"),
)
def update_case_detail(case_id, _inv, _block, _allow):
    if not case_id:
        return dmc.Text("Select a case to review.", c="dimmed")
    try:
        trig = ctx.triggered_id
        if trig == "btn-investigate":
            api.post(f"/cases/{case_id}/investigate")
        elif trig == "btn-block":
            api.post(f"/cases/{case_id}/resolve",
                     {"decision": "block", "actor": "human:dashboard"})
        elif trig == "btn-allow":
            api.post(f"/cases/{case_id}/resolve",
                     {"decision": "allow", "actor": "human:dashboard"})
        return queue_page.render_case_detail(api.get(f"/cases/{case_id}"))
    except Exception as e:
        return _error(str(e))


server = app.server
