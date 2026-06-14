"""Fraud-ring detection on the user-projection graph.

Layered cheap -> expensive:
1. connected components of the user-projection (users linked by shared infra)
2. community detection (greedy modularity) to split large components into rings
3. score each ring by size, internal density, and shared-identifier concentration

`fraud_saturation_gt` is computed for reporting only (how much of a detected ring
is truly fraudulent); it is NEVER fed back into models or rules.
"""
from __future__ import annotations

import networkx as nx

from src.graph.builder import build_user_projection


def detect_rings(graph: nx.Graph, settings, store=None) -> list[dict]:
    proj = build_user_projection(graph)
    min_size = settings.graph.get("community_min_size", 3)

    rings: list[dict] = []
    ring_id = 0
    for comp in nx.connected_components(proj):
        if len(comp) < 2:
            continue
        sub = proj.subgraph(comp)
        # split large components into tighter communities
        if len(comp) > 12:
            communities = nx.community.greedy_modularity_communities(sub, weight="weight")
        else:
            communities = [comp]
        for community in communities:
            members = list(community)
            if len(members) < min_size:
                continue
            ring_id += 1
            csub = proj.subgraph(members)
            shared_ids: set[str] = set()
            for _, _, data in csub.edges(data=True):
                shared_ids.update(data.get("shared", []))
            density = nx.density(csub) if len(members) > 1 else 0.0
            risk = _ring_risk(len(members), density, len(shared_ids))
            rings.append({
                "ring_id": f"ring_{ring_id:03d}",
                "members": [m.split(":", 1)[1] for m in members],
                "size": len(members),
                "n_shared_identifiers": len(shared_ids),
                "density": round(density, 3),
                "risk_score": round(risk, 3),
            })

    if store is not None:
        _attach_fraud_saturation(rings, store)
    return rings


def _ring_risk(size: int, density: float, n_shared: int) -> float:
    """Label-free heuristic in [0,1]: bigger, denser, more-shared = riskier."""
    size_term = min(size / 8.0, 1.0)
    shared_term = min(n_shared / 4.0, 1.0)
    return 0.4 * size_term + 0.35 * density + 0.25 * shared_term


def _attach_fraud_saturation(rings: list[dict], store) -> None:
    users = store.table("users").set_index("user_id")["is_fraud_gt"].to_dict()
    for r in rings:
        labels = [users.get(m, 0) for m in r["members"]]
        r["fraud_saturation_gt"] = round(sum(labels) / max(1, len(labels)), 3)


def ring_membership(rings: list[dict]) -> dict[str, dict]:
    """user_id -> {ring_id, ring_risk_score} for the highest-risk ring it's in."""
    out: dict[str, dict] = {}
    for r in rings:
        for uid in r["members"]:
            cur = out.get(uid)
            if cur is None or r["risk_score"] > cur["ring_risk_score"]:
                out[uid] = {"ring_id": r["ring_id"], "ring_risk_score": r["risk_score"]}
    return out
