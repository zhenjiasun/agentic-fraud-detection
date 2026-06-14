# CLAUDE.md

Context for AI agents working in this repo. Read before making changes.

## What this is

FraudGuard — an end-to-end **Agentic Fraud & Abuse Detection System** (a public
showcase of fraud ML + graph analysis + agent security + human-in-the-loop + AI
evaluation + model governance + monitoring). Not a library; a runnable system.
Runs fully offline (mock LLM) and must stay that way for CI.

## Run / verify

- `python bootstrap.py` — full pipeline (seed→graph→train→score→queue→investigate→evaluate→monitor).
- `python run_api.py` (FastAPI :8000) then `python run_dashboard.py` (Dash :8050).
- `pytest` — 58 hermetic tests; `conftest.py` forces `LLM_PROVIDER=mock`.

## Architecture invariants (do not break)

- **SQLite is the single source of truth** (`src/data/store.py`). The Dash app
  never touches the DB — it goes through the FastAPI backend only.
- **Capability restriction is the agent's real security boundary.** Never add a
  write tool to `src/agent/tools.py`. The investigator can only read + call
  `submit_finding` (validated, closed-enum). It must never resolve a case — only
  a human actor (or a high-confidence rule) reaches a terminal money state
  (enforced in `src/orchestrator/queue.py` and the `/resolve` endpoint).
- **No `eval`/`exec` in the rules engine.** `when` expressions go through the AST
  allowlist interpreter in `src/rules/engine.py`. Adding `eval` is a vulnerability.
- **No ground-truth leakage.** `*_gt` columns are simulator-only and consumed ONLY
  by `src/analysis/evaluation.py`. Models read from the `TXN_FEATURES` /
  `ACCOUNT_FEATURES` whitelists in `src/models/features.py`; a leakage test guards it.
- **Audit is append-only + hash-chained** (`src/audit/`). Never add an update/delete
  path. Every consequential action records exactly one row.
- **Determinism.** The whole world derives from one seed via `src/data/seeds.py`.
  Keep new randomness flowing through `spawn(seed, name)`.

## Key functions

- `SyntheticWorld.generate()/persist()` — builds + writes the world.
- `detect_rings()` / `graph_features()` — ring detection + per-user graph features.
- `train_and_score()` — trains both models, calibrates, writes `account_scores`/`txn_scores`.
- `run_orchestrator()` — rules + score + graph → action; populates `cases`.
- `investigate_case()` — bounded agent tool loop → validated recommendation.
- `build_report()` / `monitoring_snapshot()` — evaluation + monitoring bundles.
- `verify_chain()` — audit integrity check.

## Conventions

- Python, `src/{data,graph,models,rules,orchestrator,agent,analysis,audit}` layout.
- snake_case funcs / PascalCase classes; stdlib `logging` via `src/log.py`.
- Dash UI is `dash-mantine-components` dark theme only — no raw HTML styling.
- New LLM providers go behind the `LLMClient` interface in `src/agent/llm_client.py`
  (keep the normalization layer; the mock must keep working with no key).
- Comments explain *why* for non-obvious invariants only (security boundary,
  leakage guard, no-eval), not *what*.

## Gotchas

- The mock investigator keys its disposition on numeric evidence (model score,
  ring flag), never on free text — this is what makes it injection-resistant in
  tests. Don't make it parse attacker-controllable strings for its decision.
- `bootstrap.py` clears the JSONL audit ledger on reseed to keep it in sync with
  the reset DB. The hash chain restarts each seed.
- Tests share a session-scoped seeded store; tests that mutate it clean up after
  themselves (see `test_audit_completeness.py`, `test_monitoring_drift.py`).
