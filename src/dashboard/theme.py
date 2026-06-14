"""Dark-theme constants + small dmc helpers shared across dashboard pages."""
from __future__ import annotations

import dash_mantine_components as dmc

ACCENT = "indigo"

STATUS_COLOR = {
    "OPEN": "yellow", "IN_REVIEW": "blue", "AWAITING_DECISION": "violet",
    "RESOLVED_ALLOW": "green", "RESOLVED_BLOCK": "red", "ESCALATED": "orange",
}
DISPOSITION_COLOR = {
    "LIKELY_FRAUD": "red", "LIKELY_LEGIT": "green",
    "INSUFFICIENT_EVIDENCE": "yellow", "ESCALATE": "orange", "REJECTED": "gray",
}
ACTION_COLOR = {"auto_block": "red", "auto_allow": "green", "route_to_review": "violet"}


def stat_card(label: str, value, color: str = "gray"):
    return dmc.Card(
        withBorder=True, radius="md", p="md",
        children=[
            dmc.Text(label, size="xs", c="dimmed"),
            dmc.Text(str(value), size="xl", fw=700, c=color),
        ],
    )


def badge(text: str, color: str = "gray"):
    return dmc.Badge(text, color=color, variant="light")


def section_title(text: str):
    return dmc.Title(text, order=4, mb="sm")
