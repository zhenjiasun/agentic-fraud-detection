"""Build the heterogeneous entity graph from the store.

Nodes are typed (`ntype` attribute) and namespaced by id (`user:u00001`,
`device:d00012`, ...) to avoid cross-type id collisions. Edges come from
entity_links (shared infrastructure) plus transacted_with edges from
transactions. The user-projection (users linked when they share infrastructure)
is the workhorse for ring detection and graph features.
"""
from __future__ import annotations

import networkx as nx
import pandas as pd

# Infra link types that connect users to shared identifiers.
INFRA_LINK_TYPES = {"uses_device", "uses_ip", "owns_card", "shares_identity"}
INFRA_NTYPES = {"device", "ip", "card", "identity"}


def node(ntype: str, _id: str) -> str:
    return f"{ntype}:{_id}"


def build_graph(store) -> nx.Graph:
    g = nx.Graph()
    links = store.table("entity_links")
    for _, r in links.iterrows():
        s = node(r["src_type"], r["src_id"])
        d = node(r["dst_type"], r["dst_id"])
        g.add_node(s, ntype=r["src_type"])
        g.add_node(d, ntype=r["dst_type"])
        g.add_edge(s, d, link_type=r["link_type"])

    # transacted_with: user -> merchant, weighted by count (context for rings/centrality)
    txns = store.query_df(
        "SELECT user_id, merchant_id, COUNT(*) n FROM transactions GROUP BY user_id, merchant_id"
    )
    for _, r in txns.iterrows():
        u = node("user", r["user_id"])
        m = node("merchant", r["merchant_id"])
        g.add_node(u, ntype="user")
        g.add_node(m, ntype="merchant")
        g.add_edge(u, m, link_type="transacted_with", weight=int(r["n"]))
    return g


def user_infra_map(g: nx.Graph) -> dict[str, set[str]]:
    """user node -> set of infra node ids it touches (device/ip/card/identity)."""
    out: dict[str, set[str]] = {}
    for n, attrs in g.nodes(data=True):
        if attrs.get("ntype") != "user":
            continue
        infra = {nb for nb in g.neighbors(n) if g.nodes[nb].get("ntype") in INFRA_NTYPES}
        out[n] = infra
    return out


def build_user_projection(g: nx.Graph) -> nx.Graph:
    """Users connected when they share an infra node; edge weight = #shared."""
    proj = nx.Graph()
    for n, attrs in g.nodes(data=True):
        if attrs.get("ntype") == "user":
            proj.add_node(n)
    # for each infra node, connect all user-neighbors pairwise
    for n, attrs in g.nodes(data=True):
        if attrs.get("ntype") not in INFRA_NTYPES:
            continue
        users = [nb for nb in g.neighbors(n) if g.nodes[nb].get("ntype") == "user"]
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                a, b = users[i], users[j]
                if proj.has_edge(a, b):
                    proj[a][b]["weight"] += 1
                    proj[a][b]["shared"].append(n)
                else:
                    proj.add_edge(a, b, weight=1, shared=[n])
    return proj
