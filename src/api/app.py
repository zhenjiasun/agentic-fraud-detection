"""FastAPI app — the single backend over the store. Dash talks only to this."""
from __future__ import annotations

from fastapi import FastAPI

from src.api.routers import audit, cases, entities, metrics

app = FastAPI(title="fraudguard", version="0.1.0",
              description="Agentic Fraud & Abuse Detection System")

app.include_router(entities.router, tags=["entities"])
app.include_router(cases.router, tags=["cases"])
app.include_router(metrics.router, tags=["metrics"])
app.include_router(audit.router, tags=["audit"])


@app.get("/health")
def health():
    return {"status": "ok"}
