"""PlaybookEntry — structured record of a completed investment analysis."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class PlaybookEntry(BaseModel):
    id: str                         # f"{ticker}_{date:%Y%m%d_%H%M}_{mode}"
    ticker: str
    company_name: str
    date: datetime
    mode: str                       # stock / sec / combined
    sector: str = ""

    # Decision
    decision: str                   # BUY / HOLD / SELL / REJECT
    conviction: str                 # HIGH / MEDIUM / LOW
    health_score: float = 0.0
    growth_score: float = 0.0
    hard_fails: list[str] = Field(default_factory=list)

    # Narrative (searchable)
    bull_thesis: str = ""
    bear_thesis: str = ""
    research_summary: str = ""
    key_risks: list[str] = Field(default_factory=list)

    # Metric snapshot for similarity matching
    pe_ratio: Optional[float] = None
    revenue_growth: Optional[float] = None
    profit_margin: Optional[float] = None
    debt_to_equity: Optional[float] = None
    rsi_14: Optional[float] = None

    # Risk sizing
    conservative_position: str = ""
    neutral_position: str = ""
    aggressive_position: str = ""

    # Quality / audit
    eval_scores: dict = Field(default_factory=dict)
    guardrail_flags: list[str] = Field(default_factory=list)

    # Feedback loop
    user_feedback: Optional[str] = None    # "correct" / "incorrect"
    feedback_notes: Optional[str] = None
