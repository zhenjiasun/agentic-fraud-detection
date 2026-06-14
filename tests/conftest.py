"""Hermetic test fixtures.

A small synthetic world is built once per session into a temp SQLite DB and run
through the full pipeline (seed -> graph -> models -> orchestrator). LLM_PROVIDER
is forced to `mock` so the whole suite runs offline and deterministically.
"""
from __future__ import annotations

import copy
import os

import pytest

os.environ["LLM_PROVIDER"] = "mock"


def _test_settings(tmp_path):
    from src.config import LLMSettings, Settings, get_settings
    base = get_settings()
    raw = copy.deepcopy(base.raw)
    # smaller, faster world but enough to exercise every archetype + rings
    raw["simulator"]["n_users"] = 300
    raw["simulator"]["days"] = 30
    raw["paths"]["db"] = str(tmp_path / "test.db")
    raw["paths"]["audit_log"] = str(tmp_path / "audit.jsonl")
    return Settings(seed=int(raw["seed"]), raw=raw, llm=LLMSettings(provider="mock"))


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    return _test_settings(tmp_path_factory.mktemp("fraudguard"))


@pytest.fixture(scope="session")
def pipeline(settings):
    """Fully built world + store after the end-to-end pipeline."""
    from src.data.simulator import SyntheticWorld
    from src.data.store import open_store
    from src.graph.builder import build_graph
    from src.graph.rings import detect_rings
    from src.models.pipeline import train_and_score
    from src.orchestrator.decision import run_orchestrator

    store = open_store(settings)
    store.reset()
    world = SyntheticWorld(settings, seed=settings.seed)
    world.generate()
    world.persist(store)
    graph = build_graph(store)
    rings = detect_rings(graph, settings, store=store)
    train_and_score(store, graph, rings, settings)
    run_orchestrator(store, settings)
    return {"settings": settings, "store": store, "graph": graph, "rings": rings,
            "world": world}


@pytest.fixture(scope="session")
def store(pipeline):
    return pipeline["store"]
