"""Review queue page: case picker + case detail with evidence, investigator,
and human resolve actions."""
from __future__ import annotations

import json

import dash_mantine_components as dmc

from src.dashboard.theme import DISPOSITION_COLOR, STATUS_COLOR, badge


def render_picker(cases: list[dict]):
    options = [{"value": c["case_id"],
                "label": f"{c['case_id']} · {c['status']} · score={c['model_score']:.2f}"}
               for c in cases]
    return dmc.Stack([
        dmc.Group([
            dmc.Select(id="case-select", data=options, searchable=True,
                       placeholder="Select a case to review", w=420,
                       value=options[0]["value"] if options else None),
            dmc.Button("Refresh queue", id="queue-refresh", variant="light"),
        ]),
        dmc.Divider(),
        dmc.Box(id="case-detail"),
    ])


def render_case_detail(detail: dict):
    case = detail["case"]
    invs = detail.get("investigations", [])
    rule_codes = json.loads(case.get("rule_codes_json") or "[]")
    graph_signals = json.loads(case.get("graph_signals_json") or "{}")

    facts = dmc.Card(withBorder=True, p="md", children=[
        dmc.Group([
            dmc.Title(case["case_id"], order=4),
            badge(case["status"], STATUS_COLOR.get(case["status"], "gray")),
        ]),
        dmc.Text(f"Subject: {case['subject_id']}", size="sm"),
        dmc.Text(f"Model score: {case['model_score']:.3f}", size="sm"),
        dmc.Group([badge(rc, "grape") for rc in rule_codes] or [dmc.Text("no rules fired", size="sm", c="dimmed")]),
        dmc.Text(f"Graph: ring_member={graph_signals.get('ring_member', 0)}, "
                 f"users_on_device={graph_signals.get('n_users_on_device', '—')}", size="sm"),
    ])

    inv_card = _investigation_card(invs[0]) if invs else dmc.Card(
        withBorder=True, p="md", children=[dmc.Text(
            "No investigation yet. Run the bounded LLM investigator.", c="dimmed")])

    is_terminal = case["status"].startswith("RESOLVED")
    actions = dmc.Group([
        dmc.Button("Run Investigator", id="btn-investigate", color="violet"),
        dmc.Button("Resolve: BLOCK", id="btn-block", color="red", disabled=is_terminal),
        dmc.Button("Resolve: ALLOW", id="btn-allow", color="green", disabled=is_terminal),
        dmc.Text(id="action-feedback", size="sm", c="dimmed"),
    ])

    return dmc.Stack([dmc.SimpleGrid(cols={"base": 1, "md": 2}, children=[facts, inv_card]),
                      actions])


def _investigation_card(inv: dict):
    disp = inv.get("disposition", "—")
    flags = json.loads(inv.get("injection_flags_json") or "[]")
    tool_calls = json.loads(inv.get("tool_calls_json") or "[]")
    children = [
        dmc.Group([dmc.Text("Investigator finding", fw=600),
                   badge(disp, DISPOSITION_COLOR.get(disp, "gray"))]),
        dmc.Text(f"Confidence: {inv.get('confidence', 0):.2f} · "
                 f"provider: {inv.get('provider', '—')} ({inv.get('model', '—')})", size="sm"),
        dmc.Text(inv.get("rationale", ""), size="sm"),
        dmc.Text("Tools used: " + ", ".join(t["name"] for t in tool_calls), size="xs", c="dimmed"),
    ]
    if flags:
        children.append(badge(f"⚠ prompt-injection flagged: {', '.join(flags)}", "red"))
    children.append(dmc.Text("Recommendation only — agent cannot resolve the case.",
                             size="xs", c="dimmed", fs="italic"))
    return dmc.Card(withBorder=True, p="md", children=children)
