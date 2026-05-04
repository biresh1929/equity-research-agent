"""State TypedDicts for all LangGraph graphs."""

from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    ticker: str
    messages: Annotated[list[BaseMessage], add_messages]

    # Phase 1 — parallel data gathering
    fundamentals: dict
    technicals: dict
    news_sentiment: dict

    # Phase 2 — data quality gate (Article 3)
    data_quality: str          # GOOD / MARGINAL / POOR
    fields_available: int
    fields_total: int

    # Phase 3 — synthesis
    research_summary: str

    # Phase 4 — bull/bear debate (Article 3)
    bull_argument: str
    bear_argument: str

    # Scoring (Article 3)
    health_score: float
    growth_score: float
    daily_liquidity: float
    health_breakdown: dict
    growth_breakdown: dict

    # Phase 5 — risk sizing (Article 3)
    conservative_sizing: str
    neutral_sizing: str
    aggressive_sizing: str

    # Phase 6 — portfolio manager decision
    decision: str              # BUY / HOLD / SELL
    conviction: str            # HIGH / MEDIUM / LOW
    hard_fails: list
    pm_rationale: str

    # Phase 7 — final output
    investment_brief: str      # markdown
    structured_output: dict    # JSON

    error: str | None


class SECState(TypedDict):
    ticker: str
    cik: str
    company_name: str
    filing_type: str           # 10-K or 10-Q
    filing_path: str
    collection_name: str       # ChromaDB + BM25 namespace

    # Parallel expert outputs
    risk_analysis: str
    sentiment_analysis: str
    fundamental_analysis: str
    math_results: dict

    # Output
    filing_report: str
    error: str | None


class SupervisorState(TypedDict):
    ticker: str
    mode: Literal["stock", "sec", "combined"]
    filing_type: str

    # Sub-graph outputs
    stock_brief: str
    structured_output: dict
    filing_report: str

    # Final output
    comprehensive_report: str
    error: str | None
