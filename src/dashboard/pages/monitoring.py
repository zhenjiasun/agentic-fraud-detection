"""Monitoring page: drift (PSI/KS), data-quality, adversarial alerts."""
from __future__ import annotations

import dash_mantine_components as dmc

from src.dashboard.theme import badge, section_title

_BAND = {"stable": "green", "warn": "yellow", "alert": "red"}
_DQ = {"pass": "green", "warn": "yellow", "fail": "red"}


def render(mon: dict):
    drift = mon.get("drift", {})
    dq = mon.get("data_quality", {})
    adv = mon.get("adversarial", {})

    drift_rows = drift.get("features", [])[:12]
    drift_table = dmc.Table(data={
        "head": ["Feature", "PSI", "KS", "Band"],
        "body": [[r["feature"], r["psi"], r["ks"],
                  badge(r["band"], _BAND.get(r["band"], "gray"))] for r in drift_rows]
    }, striped=True) if drift.get("available") else dmc.Text("No drift reference yet.", c="dimmed")

    dq_table = dmc.Table(data={
        "head": ["Check", "Status", "Detail"],
        "body": [[c["name"], badge(c["status"], _DQ.get(c["status"], "gray")), c["detail"]]
                 for c in dq.get("checks", [])]}, striped=True)

    alerts = adv.get("alerts", [])
    adv_block = (dmc.Stack([badge(f"{a['signal']}: {a['detail']}", "red") for a in alerts])
                 if alerts else dmc.Text("No adversarial-behavior alerts.", c="green"))

    return dmc.Stack([
        dmc.Group([badge(f"drift alerts: {drift.get('n_alert', 0)}", "red"),
                   badge(f"data-quality: {dq.get('status', '—')}", _DQ.get(dq.get('status'), 'gray')),
                   badge(f"adversarial alerts: {len(alerts)}", "orange")]),
        section_title("Feature / score drift (PSI vs training snapshot)"),
        drift_table,
        section_title("Data quality"),
        dq_table,
        section_title("Adversarial behavior"),
        adv_block,
    ])
