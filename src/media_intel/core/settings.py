"""Type-safe runtime configuration."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field

Provider = Literal["mock", "openai"]


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


class Settings(BaseModel):
    """Typed, validated application settings (12-factor friendly)."""

    model_config = {"frozen": True}

    # LLM
    llm_provider: Provider = "mock"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    model_large: str = "mock-large"
    model_small: str = "mock-small"
    llm_timeout_s: float = 45.0

    # Embeddings
    embedding_provider: Provider = "mock"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 128

    # Governance
    auto_approve: bool = False
    max_llm_calls_per_run: int = 24
    max_total_tokens_per_run: int = 60_000
    run_deadline_ms: int = 20_000

    # API
    api_key: str = ""

    # Databricks
    databricks_host: str = ""
    databricks_token: str = ""
    databricks_vector_endpoint: str = ""
    databricks_warehouse_id: str = ""
    databricks_catalog: str = "media_intel"
    databricks_schema: str = "default"

    # Observability
    mlflow_enabled: bool = False
    mlflow_tracking_uri: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            llm_provider=_env("LLM_PROVIDER", "mock"),  # type: ignore[arg-type]
            openai_base_url=_env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_api_key=_env("OPENAI_API_KEY"),
            model_large=_env("LLM_MODEL_LARGE", "mock-large"),
            model_small=_env("LLM_MODEL_SMALL", "mock-small"),
            llm_timeout_s=float(_env("LLM_TIMEOUT_S", "45")),
            embedding_provider=_env("EMBEDDING_PROVIDER", "mock"),  # type: ignore[arg-type]
            embedding_model=_env("EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dim=_env_int("EMBEDDING_DIM", 128),
            auto_approve=_env_bool("AUTO_APPROVE", False),
            max_llm_calls_per_run=_env_int("MAX_LLM_CALLS_PER_RUN", 24),
            max_total_tokens_per_run=_env_int("MAX_TOTAL_TOKENS_PER_RUN", 60_000),
            run_deadline_ms=_env_int("RUN_DEADLINE_MS", 20_000),
            api_key=_env("API_KEY"),
            databricks_host=_env("DATABRICKS_HOST"),
            databricks_token=_env("DATABRICKS_TOKEN"),
            databricks_vector_endpoint=_env("DATABRICKS_VECTOR_ENDPOINT"),
            databricks_warehouse_id=_env("DATABRICKS_WAREHOUSE_ID"),
            databricks_catalog=_env("DATABRICKS_CATALOG", "media_intel"),
            databricks_schema=_env("DATABRICKS_SCHEMA", "default"),
            mlflow_enabled=_env_bool("MLFLOW_ENABLED", False),
            mlflow_tracking_uri=_env("MLFLOW_TRACKING_URI"),
        )
