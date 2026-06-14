"""Audit page: hash-chain verification badge + searchable audit log."""
from __future__ import annotations

import json

import dash_mantine_components as dmc

from src.dashboard.theme import badge, section_title

_ACTOR_COLOR = {"rules": "grape", "orchestrator": "blue", "system": "gray"}


def render(rows: list[dict], verify: dict):
    chain = (badge("✓ hash chain intact", "green") if verify.get("ok")
             else badge(f"✗ chain broken at #{verify.get('broken_at')}", "red"))
    head = ["ts", "actor", "action", "subject", "payload"]
    body = []
    for r in rows[:150]:
        actor = r["actor"]
        color = _ACTOR_COLOR.get(actor.split(":")[0], "teal")
        payload = json.dumps(json.loads(r["payload_json"]))[:90]
        body.append([r["ts"][11:19], badge(actor, color), r["action_type"],
                     f"{r['subject_type']}:{r['subject_id']}", payload])
    return dmc.Stack([
        dmc.Group([section_title("Audit trail"), chain,
                   dmc.Text(f"{verify.get('n_rows', 0)} append-only records", c="dimmed")]),
        dmc.Table(data={"head": head, "body": body}, striped=True, highlightOnHover=True),
        dmc.Text("Every rule fire, model action, investigation, and human decision is "
                 "recorded and hash-chained — the whole system is reconstructable from "
                 "this log.", size="xs", c="dimmed"),
    ])
