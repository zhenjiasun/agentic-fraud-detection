# FraudGuard — Agentic Fraud & Abuse Detection System

A single, runnable project that closes the **AI-security gap**: it combines fraud
ML, graph analysis, a bounded LLM investigator, human-in-the-loop review, AI
evaluation, model governance, and production monitoring — and treats the agent
itself as an attack surface.

Everything runs **offline with no API key** (the LLM investigator defaults to a
deterministic mock). Switch to Claude, OpenAI, or DeepSeek with one env var.

Tooling is **uv** (Python env/deps) + a **Rust core** (`fraudguard_core`, built
with maturin) for the CPU-bound graph + feature hot paths.

```bash
# Python env + deps (uv)
uv venv && uv sync                       # or: uv pip install -e .
cp .env.example .env                     # LLM_PROVIDER=mock by default — no key

# Build the Rust core (needs a Rust toolchain: `brew install rust` or rustup)
uv pip install maturin
uv run maturin develop -m rust/Cargo.toml --release

# Run
uv run python bootstrap.py               # seed→graph→train→score→queue→investigate→evaluate→monitor
uv run python run_api.py                 # FastAPI backend on :8000
uv run python run_dashboard.py           # Dash dark dashboard on :8050
uv run pytest                            # 58 hermetic tests, fully offline
```

The Rust core is **optional** — if the extension isn't built, the graph ring
detection and transaction-velocity kernels transparently fall back to equivalent
pure-Python implementations, so `uv pip install -e .` alone still runs everything
(plain `pip install -r requirements.txt` + `python bootstrap.py` also works).

## What it does

A synthetic payments world (users, merchants, cards, devices, IPs, identities)
is generated with five injected fraud archetypes and ground-truth labels. Each
account is scored, run through a rules engine, and routed to one of
`auto_allow` / `auto_block` / `route_to_review`. Ambiguous cases land in a
human-review queue, where a **bounded LLM investigator** gathers read-only
evidence and produces a *recommendation* — it can never block, allow, or move
money. Every action is written to a hash-chained audit trail.

## The nine pillars

| # | Pillar | Where |
|---|--------|-------|
| 1 | Simulated users / merchants / payments / identities / devices / IPs | `src/data/simulator.py`, `archetypes.py` |
| 2 | Fraud rings as a graph | `src/graph/` (networkx, community detection) |
| 3 | Transaction- & account-risk models | `src/models/` (XGBoost + isotonic/Platt calibration) |
| 4 | Rules engine for high-confidence actions | `src/rules/` (declarative YAML, **AST allowlist — no `eval`**) |
| 5 | Human-review queue | `src/orchestrator/queue.py` (state machine) |
| 6 | Bounded LLM investigator | `src/agent/` (read-only tools, closed-enum output) |
| 7 | Evaluation: FP, expected loss, calibration, disparity | `src/analysis/evaluation.py`, `disparity.py` |
| 8 | Drift / data-quality / adversarial monitoring | `src/analysis/drift.py`, `data_quality.py`, `adversarial.py` |
| 9 | Prompt-injection & tool-abuse tests + audit trail | `tests/test_prompt_injection.py`, `src/audit/` |

## The agent-security model (the crux)

The investigator's bound is **capability restriction, not prompt obedience**:

- **No write tools exist.** The tool registry (`src/agent/tools.py`) exposes only
  six read functions. There is no `block` / `allow` / `refund` / `move_money`
  tool, so a fully hijacked agent still cannot act.
- **Closed-enum output.** The agent concludes by calling `submit_finding`, whose
  result is validated against `InvestigationResult` (disposition ∈ a fixed set,
  bounded confidence, **evidence ids must exist**). Invalid output is rejected
  and never reaches the queue.
- **Server-side resolve boundary.** Only a human actor can move a case to a
  money-affecting terminal state (`queue.py` + the API's `/cases/{id}/resolve`).
  The agent can only recommend (`AWAITING_DECISION`).
- **Untrusted-field tagging.** Attacker-controlled text (merchant names) is wrapped
  and sanitized before entering the prompt; injection tokens are flagged and audited.

`tests/test_prompt_injection.py` runs an attack corpus (`prompts/injection_corpus.yaml`)
through the live agent and asserts: no write tool is ever invoked, output is always
a valid disposition, injected text alone cannot flip a fraud case, and every
attempt is audited.

## Provider-agnostic investigator

Set `LLM_PROVIDER` in `.env`: `mock` (default, offline), `anthropic`,
`openai`, or `deepseek`. One normalization layer (`src/agent/llm_client.py`)
makes the tool loop provider-blind; DeepSeek/OpenAI share an OpenAI-compatible
client (`base_url` switch) and Anthropic uses its native SDK. The mock exercises
the identical tool loop and validation so CI stays hermetic.

## Evaluation & governance

`build_report()` produces confusion metrics, PR/ROC-AUC, calibration (ECE/Brier
+ reliability curve), **expected-$-loss vs threshold** (which feeds the
orchestrator's threshold choice), and **segment disparity** (FPR/FNR per
geography and customer segment). Ground-truth labels are consumed *only* here; a
feature whitelist + leakage test keep `*_gt` columns out of the models.

## Real-data validation

The supervised modeling stack is also validated on a real external dataset (ULB
[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud),
284,807 transactions, 0.173% fraud) — not just the synthetic world. On an honest
temporal split it reaches **PR-AUC 0.80 / ROC-AUC 0.972** with calibrated
probabilities and a cost-optimal threshold. Run it with
`uv run python scripts/real_data_test.py`; the full methodology and rationale are
in [docs/real-data-evaluation.md](docs/real-data-evaluation.md).

## Audit trail

`src/audit/` is append-only and hash-chained
(`hash = sha256(prev_hash + canonical(record))`). `verify_chain()` detects any
tampering; the dashboard shows a green/red integrity badge. Every rule fire,
model action, investigation, injection attempt, and human decision is
reconstructable from the log alone.

## Rust core

`rust/` is a pyo3 crate (`fraudguard_core`) built with maturin into the uv venv.
It implements the two CPU-bound hot paths with results identical to the Python
fallbacks:
- `detect_rings` — union-find clustering on the user/infrastructure graph (the
  graph layer's heavy loop), used by `src/graph/rings.py`.
- `velocity_24h` — per-user count of prior transactions within 24h (the feature
  kernel run right after the simulator), used by `src/models/features.py`.

Each call site does `import fraudguard_core` behind a `try/except` and falls back
to pure Python when the extension isn't present.

## Layout

```
bootstrap.py            one command for the whole pipeline
rust/                   pyo3 crate (fraudguard_core): ring detection + velocity
src/data/               simulator, archetypes, SQLite store (single source of truth)
src/graph/              entity graph, ring detection, graph features
src/models/             XGBoost risk models, calibration, registry, pipeline
src/rules/              declarative rules + safe AST evaluator
src/orchestrator/       decision logic + review-queue state machine
src/agent/              provider-agnostic client, read-only tools, guards, investigator
src/analysis/           evaluation, calibration, disparity, drift, data-quality, adversarial
src/audit/              append-only hash-chained log + verification
src/api/                FastAPI backend (the only writer)
src/dashboard/          Dash dark UI (pure API client)
tests/                  one suite per pillar (hermetic)
```

## License

MIT.
