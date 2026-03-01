# Loads all application settings from environment variables and the .env file into a typed singleton.
"""
Fortio Agent – application settings.

All config is loaded from environment variables (or .env file).
Access the singleton:  from agent.config import settings

CORS_ORIGINS in .env or shell accepts either format:
  Comma-separated:  CORS_ORIGINS=http://localhost:4200,https://localhost:4200
  JSON array:       CORS_ORIGINS=["http://localhost:4200","https://localhost:4200"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict
from pydantic_settings.sources.providers.dotenv import DotEnvSettingsSource
from pydantic_settings.sources.providers.env import EnvSettingsSource

# Resolve the `apps/agent/` directory from this file's location so that
# `.env` is found correctly regardless of the working directory the server
# is started from (repo root, apps/agent/, etc.).
_AGENT_DIR = Path(__file__).resolve().parent.parent


# ── Custom env source that accepts comma-separated strings for list fields ─────


class _CommaSeparatedListMixin:
    """
    Mixin for pydantic-settings env sources.

    pydantic-settings v2 always calls json.loads() on list/dict fields before
    any pydantic validator runs.  This mixin overrides `decode_complex_value` so
    that a plain comma-separated string (e.g. "a,b,c") is accepted in addition
    to the standard JSON-array format (e.g. '["a","b","c"]').
    """

    # Splits "a,b,c" comma strings into a Python list; bypasses pydantic-settings' default JSON-only parser.
    def decode_complex_value(self, field_name: str, field: Any, value: Any) -> Any:
        if isinstance(value, str) and not value.strip().startswith(("[", "{")):
            # Comma-separated → split into list; strip whitespace around each item
            return [item.strip() for item in value.split(",") if item.strip()]
        return super().decode_complex_value(field_name, field, value)  # type: ignore[misc]


class _CSEnvSource(_CommaSeparatedListMixin, EnvSettingsSource):
    """OS-level environment variable source with comma-separated list support."""


class _CSDotEnvSource(_CommaSeparatedListMixin, DotEnvSettingsSource):
    """.env file source with comma-separated list support."""


# ── Settings ───────────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    """
    Pydantic-settings v2 style — field names map to env var names
    (case-insensitive).  E.g.  anthropic_api_key ↔ ANTHROPIC_API_KEY.
    """

    model_config = SettingsConfigDict(
        env_file=str(_AGENT_DIR / ".env"),  # always reads apps/agent/.env
        case_sensitive=False,
        extra="ignore",  # silently ignore unknown env vars
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
    # Redis logical DB allocation:
    #   DB 0 → Ghostfolio (default when only REDIS_HOST/REDIS_PORT are set)
    #   DB 1 → Fortio agent (caching, rate-limit counters)
    # Using separate logical DBs keeps the two services' keyspaces isolated
    # without needing two Redis instances.
    redis_url: str = "redis://localhost:6379/1"
    # LangGraph conversation checkpointing — uses DATABASE_URL (Postgres).
    # Set to "memory" to fall back to in-memory checkpointing (no persistence).
    checkpoint_backend: str = "postgres"

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Comma-separated list of allowed origins for the FastAPI CORS middleware.
    # Defaults cover local Angular dev server (http + https) and Ghostfolio.
    # Override CORS_ORIGINS in .env or Railway to add/replace domains.
    #
    # Accepted .env formats (both work):
    #   CORS_ORIGINS=http://localhost:4200,https://localhost:4200
    #   CORS_ORIGINS=["http://localhost:4200","https://localhost:4200"]
    cors_origins: list[str] = [
        "http://localhost:4200",
        "https://localhost:4200",
        "http://localhost:3333",
        "https://localhost:3333",
        "https://ghostfolio-production-453e.up.railway.app",
    ]

    # ── App ───────────────────────────────────────────────────────────────────
    environment: str = "development"
    log_level: str = "INFO"
    portfolio_concentration_threshold: float = 0.20
    market_data_freshness_minutes: int = 15
    max_tool_retries: int = 2

    # ── FIRE Goal Tracker (optional add-on) ──────────────────────────────────
    # Set FIRE_TRACKER_ENABLED=true in .env to activate the retirement planning feature.
    # When false (default): no DB table is created, no tools are registered,
    # no FRED API calls are ever made — safe to deploy without any .env change.
    fire_tracker_enabled: bool = False
    # Free FRED API key — sign up at https://fred.stlouisfed.org/docs/api/api_key.html
    # Required when fire_tracker_enabled=true. If empty, get_macro_data returns an error.
    fred_api_key: str = ""

    # ── Optimisation knobs ────────────────────────────────────────────────────
    # 1. Tool schema compression — strip verbose Args/Returns/Use-this-when sections
    #    from tool descriptions sent to the LLM. Saves ~40% of the 3,161 tool-schema
    #    tokens per LLM call. Set to False to restore original verbose descriptions.
    tool_schema_compression: bool = True

    # 2. Sliding-window history — number of past *turns* (HumanMessage boundaries)
    #    kept in the context sent to the LLM. Older turns are dropped silently.
    #    Set to 0 to disable (full history, original behaviour).
    history_window_turns: int = 6

    # 3. Semantic query cache — Redis-backed exact-match cache for repeated queries.
    #    Only caches single-turn (fresh-session) queries to avoid stale context.
    semantic_cache_enabled: bool = True
    semantic_cache_ttl_seconds: int = 300  # 5 minutes

    # 4. Tool result truncation — cap each ToolMessage at ~N tokens before the
    #    next LLM call. Uses a 4-chars-per-token heuristic.
    #    Set to 0 to disable.
    #    Raised from 300 → 800: 300 tokens (~1 200 chars) was too aggressive —
    #    fee-drag and health-scorecard results were being silently truncated,
    #    causing the LLM to see incomplete data and sometimes skip synthesis.
    max_tool_result_tokens: int = 800

    # ── Custom sources: teach pydantic-settings to parse comma-separated lists ─
    # Swaps the default pydantic-settings env/dotenv sources for comma-aware custom variants.
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        **kwargs: Any,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Replace the default env + dotenv sources with comma-aware variants.
        The custom sources accept "a,b,c" in addition to '["a","b","c"]' for
        list[str] fields — specifically CORS_ORIGINS.

        Uses **kwargs for the trailing sources (secrets_dir / file_secret_settings)
        so this works across pydantic-settings 2.x minor versions.
        """
        remaining = tuple(kwargs.values())  # secrets source(s), version-agnostic
        return (
            init_settings,
            _CSEnvSource(settings_cls),
            _CSDotEnvSource(settings_cls),
            *remaining,
        )


settings = Settings()
