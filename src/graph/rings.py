"""Fraud-ring detection.

The hot path (union-find clustering on the user/infrastructure graph) runs in
the Rust core `fraudguard_core` when the extension is built; otherwise it falls
back to an equivalent pure-Python/networkx implementation. Both produce the same
ring shape. `fraud_saturation_gt` is computed for reporting only — never fed to
models or rules.
"""
from __future__ import annotations

import networkx as nx

from src.graph.builder import INFRA_NTYPES, build_user_projection

try:  # Rust core is optional — pure-Python fallback keeps the repo runnable
    import fraudguard_core as _fc
    _CORE = True
except ImportError:  # pragma: no cover
    _fc = None
    _CORE = False


def detect_rings(graph: nx.Graph, settings, store=None) -> list[dict]:
    min_size = settings.graph.get("community_min_size", 3)
    if _CORE:
        rings = _detect_rings_rust(graph, min_size)
    else:
        rings = _detect_rings_networkx(graph, min_size)
    if store is not None:
        _attach_fraud_saturation(rings, store)
    return rings


def _user_infra_edges(graph: nx.Graph) -> list[tuple[str, str]]:
    edges = []
    for n, attrs in graph.nodes(data=True):
        if attrs.get("ntype") != "user":
            continue
        for nb in graph.neighbors(n):
            if graph.nodes[nb].get("ntype") in INFRA_NTYPES:
                edges.append((n, nb))
    return edges


def _detect_rings_rust(graph: nx.Graph, min_size: int) -> list[dict]:
    raw = _fc.detect_rings(_user_infra_edges(graph), min_size)
    rings = []
    for i, (members, size, n_shared, density, risk) in enumerate(
            sorted(raw, key=lambda r: -r[4]), 1):
        rings.append({
            "ring_id": f"ring_{i:03d}",
            "members": [m.split(":", 1)[1] for m in members],
            "size": size, "n_shared_identifiers": n_shared,
            "density": round(density, 3), "risk_score": round(risk, 3),
        })
    return rings


def _detect_rings_networkx(graph: nx.Graph, min_size: int) -> list[dict]:
    proj = build_user_projection(graph)
    rings: list[dict] = []
    ring_id = 0
    for comp in nx.connected_components(proj):
        if len(comp) < 2:
            continue
        sub = proj.subgraph(comp)
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
                "size": len(members), "n_shared_identifiers": len(shared_ids),
                "density": round(density, 3), "risk_score": round(risk, 3),
            })
    return rings


def _ring_risk(size: int, density: float, n_shared: int) -> float:
    size_term = min(size / 8.0, 1.0)
    shared_term = min(n_shared / 4.0, 1.0)
    return 0.4 * size_term + 0.35 * density + 0.25 * shared_term


def _attach_fraud_saturation(rings: list[dict], store) -> None:
    users = store.table("users").set_index("user_id")["is_fraud_gt"].to_dict()
    for r in rings:
        labels = [users.get(m, 0) for m in r["members"]]
        r["fraud_saturation_gt"] = round(sum(labels) / max(1, len(labels)), 3)


def ring_membership(rings: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rings:
        for uid in r["members"]:
            cur = out.get(uid)
            if cur is None or r["risk_score"] > cur["ring_risk_score"]:
                out[uid] = {"ring_id": r["ring_id"], "ring_risk_score": r["risk_score"]}
    return out
