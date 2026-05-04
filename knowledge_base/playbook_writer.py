"""Save a completed analysis to the AI playbook."""

import re
from datetime import datetime, timezone

from .models import PlaybookEntry
from .playbook_store import PlaybookStore


def _extract_key_risks(state: dict) -> list[str]:
    """Pull top 3 risk sentences from the bear argument."""
    bear = state.get("bear_argument", "")
    if not bear:
        return state.get("hard_fails", [])[:3]
    sentences = re.split(r"[.!?•\n]", bear)
    risks = [s.strip() for s in sentences if len(s.strip()) > 20][:3]
    return risks or state.get("hard_fails", [])[:3]


def _extract_sector(state: dict) -> str:
    return state.get("fundamentals", {}).get("sector", "")


def write_to_playbook(state: dict) -> str:
    """Build a PlaybookEntry from completed ResearchState and persist it. Returns entry_id."""
    ticker = state.get("ticker", "UNKNOWN")
    now = datetime.now(timezone.utc)
    mode = state.get("mode", "stock")

    fundamentals = state.get("fundamentals", {})
    technicals = state.get("technicals", {})

    entry = PlaybookEntry(
        id=f"{ticker}_{now:%Y%m%d_%H%M}_{mode}",
        ticker=ticker,
        company_name=fundamentals.get("company_name", ticker),
        date=now,
        mode=mode,
        sector=_extract_sector(state),
        decision=state.get("decision", "UNKNOWN"),
        conviction=state.get("conviction", "LOW"),
        health_score=state.get("health_score", 0.0),
        growth_score=state.get("growth_score", 0.0),
        hard_fails=state.get("hard_fails", []),
        bull_thesis=state.get("bull_argument", ""),
        bear_thesis=state.get("bear_argument", ""),
        research_summary=state.get("research_summary", ""),
        key_risks=_extract_key_risks(state),
        pe_ratio=fundamentals.get("trailingPE"),
        revenue_growth=fundamentals.get("revenueGrowth"),
        profit_margin=fundamentals.get("profitMargins"),
        debt_to_equity=fundamentals.get("debtToEquity"),
        rsi_14=technicals.get("rsi_14"),
        conservative_position=state.get("conservative_sizing", ""),
        neutral_position=state.get("neutral_sizing", ""),
        aggressive_position=state.get("aggressive_sizing", ""),
    )

    store = PlaybookStore()
    store.add_entry(entry)
    return entry.id
