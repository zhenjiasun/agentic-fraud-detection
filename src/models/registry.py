"""Model persistence: joblib artifact + meta.json sidecar.

Mirrors the project convention of saving models under
saved_models/{name}/{id}/ with a meta.json carrying metrics and the training
distribution snapshot used later by drift monitoring.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib


def save_model(saved_models_dir: Path, name: str, model, calibrator, meta: dict) -> str:
    model_id = meta.get("model_id", "latest")
    out = Path(saved_models_dir) / name / model_id
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "calibrator": calibrator}, out / "model.joblib")
    with open(out / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    return str(out)


def load_model(saved_models_dir: Path, name: str, model_id: str = "latest"):
    out = Path(saved_models_dir) / name / model_id
    bundle = joblib.load(out / "model.joblib")
    with open(out / "meta.json") as fh:
        meta = json.load(fh)
    return bundle["model"], bundle["calibrator"], meta
