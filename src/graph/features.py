"""Per-user graph features joined into the model feature frame.

Computed once per scoring batch (per-txn recomputation is the cost trap). All
features are derived from infrastructure links only — never from `*_gt`.
"""
from __future__ import annotations

import networkx as nx
import pandas as pd

from src.graph.builder import build_user_projection
from src.graph.rings import ring_membership


def graph_features(graph: nx.Graph, rings: list[dict]) -> pd.DataFrame:
    proj = build_user_projection(graph)

    # how many distinct users sit on each infra node
    users_on_infra: dict[str, int] = {}
    for n, attrs in graph.nodes(data=True):
        if attrs.get("ntype") in {"device", "ip", "card", "identity"}:
            users_on_infra[n] = sum(
                1 for nb in graph.neighbors(n) if graph.nodes[nb].get("ntype") == "user"
            )

    pagerank = nx.pagerank(proj, weight="weight") if proj.number_of_edges() else {}
    clustering = nx.clustering(proj)
    comp_size = {}
    for comp in nx.connected_components(proj):
        for n in comp:
            comp_size[n] = len(comp)

    membership = ring_membership(rings)

    rows = []
    for n, attrs in graph.nodes(data=True):
        if attrs.get("ntype") != "user":
            continue
        uid = n.split(":", 1)[1]
        devices = [nb for nb in graph.neighbors(n) if graph.nodes[nb].get("ntype") == "device"]
        ips = [nb for nb in graph.neighbors(n) if graph.nodes[nb].get("ntype") == "ip"]
        dev_counts = [users_on_infra.get(d, 1) for d in devices]
        ip_counts = [users_on_infra.get(i, 1) for i in ips]
        ring = membership.get(uid, {})
        rows.append({
            "user_id": uid,
            "g_n_users_on_device": max(dev_counts) if dev_counts else 1,
            "g_n_users_on_ip": max(ip_counts) if ip_counts else 1,
            "g_n_shared_devices": sum(1 for c in dev_counts if c > 1),
            "g_n_shared_ips": sum(1 for c in ip_counts if c > 1),
            "g_degree": proj.degree(n) if n in proj else 0,
            "g_component_size": comp_size.get(n, 1),
            "g_clustering": round(clustering.get(n, 0.0), 4),
            "g_pagerank": round(pagerank.get(n, 0.0), 6),
            "g_ring_member": int(bool(ring)),
            "g_ring_risk": ring.get("ring_risk_score", 0.0),
        })
    return pd.DataFrame(rows)
