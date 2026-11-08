"""Model settings from environment variables, .env or Streamlit secrets.
LLM_* is the default for every stage; SCORE_*, JUDGE_* and RANK_* override one stage (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass

STAGES = ("SCORE", "JUDGE", "RANK")
DEFAULT_TEMPERATURE = 0.2


@dataclass
class ModelConfig:
    stage: str
    base_url: str
    api_key: str
    model: str
    reasoning_effort: str | None
    temperature: float | None = DEFAULT_TEMPERATURE
    rpm: float | None = None

    def public(self) -> dict:
        """Settings safe to write into a run folder (no key)."""
        return {"stage": self.stage, "base_url": self.base_url, "model": self.model,
                "reasoning_effort": self.reasoning_effort, "temperature": self.temperature, "rpm": self.rpm}


def _get(name: str, env: dict) -> str:
    return (env.get(name) or "").strip()


def _env(env: dict | None) -> dict:
    if env is None:
        try:
            from dotenv import load_dotenv
            load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
        except ImportError:
            pass
        env = dict(os.environ)
    return env


def _temperature(v: str) -> float | None:
    if not v:
        return DEFAULT_TEMPERATURE
    return None if v.lower() in ("none", "omit", "off") else float(v)


def _build(stage: str, chain: list[str], env: dict) -> ModelConfig:
    pick = lambda field: next((_get(f"{p}_{field}", env) for p in chain if _get(f"{p}_{field}", env)), "")
    rpm = pick("RPM")
    return ModelConfig(stage, pick("BASE_URL"), pick("API_KEY"), pick("MODEL"), pick("REASONING_EFFORT") or None,
                       _temperature(pick("TEMPERATURE")), float(rpm) if rpm else None)


def load_configs(env: dict | None = None) -> dict[str, ModelConfig]:
    env = _env(env)
    return {stage: _build(stage, [stage, "JUDGE", "LLM"] if stage == "RANK" else [stage, "LLM"], env)
            for stage in STAGES}


def load_provider(prefix: str, env: dict | None = None) -> ModelConfig:
    """Settings under one prefix only, e.g. GEMINI_BASE_URL, GEMINI_API_KEY, GEMINI_MODEL."""
    return _build(prefix.upper(), [prefix.upper()], _env(env))
