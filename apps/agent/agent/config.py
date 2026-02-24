from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # LLM
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    # claude-sonnet-4-5 is the model your Anthropic account has access to.
    # Override with PRIMARY_MODEL env var to switch models without code changes.
    primary_model: str = Field(default="claude-haiku-4-5", env="PRIMARY_MODEL")
    fallback_model: str = Field(default="gpt-4o-mini", env="FALLBACK_MODEL")

    # Ghostfolio
    ghostfolio_base_url: str = Field(default="https://ghostfol.io", env="GHOSTFOLIO_BASE_URL")
    ghostfolio_access_token: str = Field(default="", env="GHOSTFOLIO_ACCESS_TOKEN")
    ghostfolio_public_access_id: str = Field(default="", env="GHOSTFOLIO_PUBLIC_ACCESS_ID")

    # Observability
    langchain_tracing_v2: str = Field(default="true", env="LANGCHAIN_TRACING_V2")
    langchain_api_key: str = Field(default="", env="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="fortio-agent", env="LANGCHAIN_PROJECT")

    # Database
    database_url: str = Field(default="", env="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")

    # App
    environment: str = Field(default="development", env="ENVIRONMENT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    portfolio_concentration_threshold: float = Field(default=0.20, env="PORTFOLIO_CONCENTRATION_THRESHOLD")
    market_data_freshness_minutes: int = Field(default=15, env="MARKET_DATA_FRESHNESS_MINUTES")
    max_tool_retries: int = Field(default=2, env="MAX_TOOL_RETRIES")

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
