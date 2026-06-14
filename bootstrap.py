"""Single end-to-end bootstrap: seed -> graph -> train -> score -> queue ->
investigate -> evaluate -> monitor.

Runs fully offline with the default `mock` LLM provider (no API key). Phases are
filled in as the build progresses; each phase is guarded so partial builds still
run what exists.

Usage:
    python bootstrap.py [--seed N]
"""
from __future__ import annotations

import argparse

from src.config import get_settings
from src.log import get_logger

log = get_logger("bootstrap")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap the fraudguard pipeline")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    args = parser.parse_args()

    settings = get_settings()
    seed = args.seed if args.seed is not None else settings.seed
    log.info("Bootstrapping fraudguard (seed=%s, llm=%s)", seed, settings.llm.provider)

    # --- P1: seed the synthetic world ---
    from src.data.store import open_store
    from src.data.simulator import SyntheticWorld

    from src.audit.log import open_audit

    store = open_store(settings)
    store.reset()
    audit = open_audit(settings, store)
    # clear the file ledger so the JSONL mirror matches the freshly-reset DB
    if settings.audit_log_path.exists():
        settings.audit_log_path.unlink()

    world = SyntheticWorld(settings, seed=seed)
    world.generate()
    world.persist(store)
    counts = world.summary()
    audit.record(actor="system", action_type="WORLD_SEEDED", subject_type="world",
                 subject_id=str(seed), payload=counts)
    log.info("Seeded world: %s", counts)

    # --- P2: graph + rings ---
    from src.graph.builder import build_graph
    from src.graph.rings import detect_rings

    graph = build_graph(store)
    rings = detect_rings(graph, settings)
    log.info("Graph: %d nodes / %d edges, %d rings detected", graph.number_of_nodes(),
             graph.number_of_edges(), len(rings))

    # --- P3: train + calibrate + score ---
    from src.models.pipeline import train_and_score

    score_summary = train_and_score(store, graph, rings, settings)
    audit.record(actor="system", action_type="MODELS_TRAINED", subject_type="models",
                 subject_id="txn_risk+account_risk", payload=score_summary)
    log.info("Scoring: %s", score_summary)

    # --- P4: rules + orchestrator + queue ---
    from src.orchestrator.decision import run_orchestrator

    orch_summary = run_orchestrator(store, settings)
    log.info("Orchestrator: %s", orch_summary)

    # --- P5: investigate ambiguous (review-queued) cases ---
    from src.agent.investigator import investigate_open_cases

    inv_summary = investigate_open_cases(store, settings, limit=15)
    log.info("Investigations: %s", inv_summary)

    # --- P6: evaluation + monitoring ---
    from src.analysis.evaluation import build_report
    from src.analysis.drift import monitoring_snapshot

    report = build_report(store, settings)
    log.info("Evaluation: %s", report["headline"])
    mon = monitoring_snapshot(store, settings)
    log.info("Monitoring: %s", mon["headline"])

    log.info("Bootstrap complete. Run `python run_api.py` then `python run_dashboard.py`.")


if __name__ == "__main__":
    main()
