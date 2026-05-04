from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Load .env with override=True so the project's .env always wins over
# any stale system-level environment variables with the same name.
_ENV_FILE = Path(__file__).parent.parent / ".env"
load_dotenv(_ENV_FILE, override=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API keys
    groq_api_key: str = Field(..., alias="GROQ_API_KEY")
    tavily_api_key: str = Field(..., alias="TAVILY_API_KEY")

    # SEC EDGAR
    edgar_user_agent: str = Field(
        "FinancialResearchAgent research@example.com",
        alias="EDGAR_USER_AGENT",
    )

    # LLM models
    primary_model: str = Field("llama-3.3-70b-versatile", alias="PRIMARY_MODEL")
    fast_model: str = Field("llama-3.1-8b-instant", alias="FAST_MODEL")

    # Storage
    chroma_persist_dir: str = Field("data/chroma_db", alias="CHROMA_PERSIST_DIR")
    sec_filings_dir: str = Field("data/sec_filings", alias="SEC_FILINGS_DIR")

    # RAG
    chunk_size: int = 1500
    chunk_overlap: int = 200
    top_k: int = 8
    hybrid_alpha: float = 0.6  # 1.0 = pure semantic, 0.0 = pure BM25

    # Hard-gate thresholds (Article 3)
    min_health_score: float = Field(50.0, alias="MIN_HEALTH_SCORE")
    min_growth_score: float = Field(50.0, alias="MIN_GROWTH_SCORE")
    min_daily_liquidity: float = Field(500_000.0, alias="MIN_DAILY_LIQUIDITY")
    max_analyst_coverage: int = Field(15, alias="MAX_ANALYST_COVERAGE")


settings = Settings()
