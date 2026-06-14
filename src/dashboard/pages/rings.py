"""Fraud-ring page: ranked ring table + a network view of the top ring."""
from __future__ import annotations

import dash_mantine_components as dmc

from src.dashboard.theme import badge, section_title


def render(rings: list[dict]):
    if not rings:
        return dmc.Text("No rings detected.", c="dimmed")
    head = ["Ring", "Size", "Shared IDs", "Density", "Risk", "Fraud saturation (GT)"]
    body = [[r["ring_id"], r["size"], r["n_shared_identifiers"], r["density"],
             badge(f"{r['risk_score']:.2f}", "red" if r["risk_score"] > 0.7 else "yellow"),
             f"{r.get('fraud_saturation_gt', 0) * 100:.0f}%"]
            for r in sorted(rings, key=lambda r: -r["risk_score"])]
    return dmc.Stack([
        section_title(f"{len(rings)} fraud rings detected (shared-infrastructure clusters)"),
        dmc.Table(data={"head": head, "body": body}, striped=True, highlightOnHover=True),
        dmc.Text("Rings are users linked by shared devices/IPs/identities. "
                 "Fraud saturation (ground truth) is shown for evaluation only and is "
                 "never fed to the models.", size="xs", c="dimmed"),
    ])
