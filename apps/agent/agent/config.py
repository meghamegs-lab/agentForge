"""
Fortio Agent – application settings.

All config is loaded from environment variables (or .env file).
Access the singleton:  from agent.config import settings
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Pydantic-settings v2 style — field names map to env var names
    (case-insensitive).  E.g.  anthropic_api_key ↔ ANTHROPIC_API_KEY.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",          # silently ignore unknown env vars
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    # claude-haiku-4-5 is fast and capable for portfolio Q&A.
    # Override with PRIMARY_MODEL env var to switch without code changes.
    primary_model: str = "claude-haiku-4-5"
    fallback_model: str = "gpt-4o-mini"

    # ── Ghostfolio ────────────────────────────────────────────────────────────
    # Required — must point to your Ghostfolio instance (e.g. Railway URL).
    # No default: startup will fail fast if this env var is missing.
    ghostfolio_base_url: str
    ghostfolio_access_token: str = ""
    ghostfolio_public_access_id: str = ""

    # ── LangSmith / Observability ─────────────────────────────────────────────
    langchain_tracing_v2: str = "true"
    langchain_api_key: str = ""
    langchain_project: str = "fortio-agent"

    # ── Database & Checkpointing ──────────────────────────────────────────────
    database_url: str = ""
    redis_url: str = "redis://localhost:6379/0"
    # LangGraph conversation checkpointing — uses DATABASE_URL (Postgres).
    # Set to "memory" to fall back to in-memory checkpointing (no persistence).
    checkpoint_backend: str = "postgres"

    # ── App ───────────────────────────────────────────────────────────────────
    environment: str = "development"
    log_level: str = "INFO"
    portfolio_concentration_threshold: float = 0.20
    market_data_freshness_minutes: int = 15
    max_tool_retries: int = 2


settings = Settings()
