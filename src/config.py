"""Typed settings loaded from config.yaml + .env.

config.yaml carries non-secret defaults; .env carries secrets and the LLM
provider switch. Everything downstream reads from a single `get_settings()`
singleton so the simulator, models, API and dashboard agree on one config.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass
class LLMSettings:
    provider: str = "mock"
    anthropic_model: str = "claude-opus-4-8"
    openai_model: str = "gpt-4o-mini"
    deepseek_model: str = "deepseek-chat"
    anthropic_key: str = ""
    openai_key: str = ""
    deepseek_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    @classmethod
    def from_env(cls) -> "LLMSettings":
        return cls(
            provider=os.environ.get("LLM_PROVIDER", "mock").strip().lower(),
            anthropic_model=os.environ.get("ANTHROPIC_MODEL", cls.anthropic_model),
            openai_model=os.environ.get("OPENAI_MODEL", cls.openai_model),
            deepseek_model=os.environ.get("DEEPSEEK_MODEL", cls.deepseek_model),
            anthropic_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            openai_key=os.environ.get("OPENAI_API_KEY", ""),
            deepseek_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            openai_base_url=os.environ.get("OPENAI_BASE_URL", cls.openai_base_url),
            deepseek_base_url=os.environ.get("DEEPSEEK_BASE_URL", cls.deepseek_base_url),
        )


@dataclass
class Settings:
    seed: int
    raw: dict[str, Any]
    llm: LLMSettings = field(default_factory=LLMSettings.from_env)

    # --- resolved absolute paths ---
    @property
    def db_path(self) -> Path:
        return ROOT / self.raw["paths"]["db"]

    @property
    def audit_log_path(self) -> Path:
        return ROOT / self.raw["paths"]["audit_log"]

    @property
    def saved_models_dir(self) -> Path:
        return ROOT / self.raw["paths"]["saved_models"]

    # --- section accessors (thin wrappers, keep call sites readable) ---
    @property
    def simulator(self) -> dict[str, Any]:
        return self.raw["simulator"]

    @property
    def graph(self) -> dict[str, Any]:
        return self.raw["graph"]

    @property
    def models(self) -> dict[str, Any]:
        return self.raw["models"]

    @property
    def orchestrator(self) -> dict[str, Any]:
        return self.raw["orchestrator"]

    @property
    def monitoring(self) -> dict[str, Any]:
        return self.raw["monitoring"]

    @property
    def agent(self) -> dict[str, Any]:
        return self.raw["agent"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    with open(ROOT / "config.yaml") as fh:
        raw = yaml.safe_load(fh)
    return Settings(seed=int(raw.get("seed", 0)), raw=raw)
