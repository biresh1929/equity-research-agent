"""Retrieve relevant past analyses to inject as context before a new run."""

import logging
from typing import Optional

from .models import PlaybookEntry
from .playbook_store import PlaybookStore

logger = logging.getLogger(__name__)


def _format_entry(entry: PlaybookEntry) -> str:
    date_str = entry.date.strftime("%Y-%m-%d")
    pe = f"P/E {entry.pe_ratio:.1f}" if entry.pe_ratio else ""
    rev = f"rev growth {entry.revenue_growth:.1%}" if entry.revenue_growth else ""
    metrics = ", ".join(filter(None, [pe, rev]))
    feedback = f" | Feedback: {entry.user_feedback}" if entry.user_feedback else ""
    bull_snippet = entry.bull_thesis[:100].replace("\n", " ") if entry.bull_thesis else ""
    bear_snippet = entry.bear_thesis[:100].replace("\n", " ") if entry.bear_thesis else ""
    return (
        f"• {date_str}: Decision={entry.decision} ({entry.conviction}). "
        f"{metrics}. "
        f"Bull: \"{bull_snippet}...\". "
        f"Bear: \"{bear_snippet}...\""
        f"{feedback}."
    )


def retrieve_playbook_context(
    ticker: str,
    sector: Optional[str] = None,
    max_entries: int = 5,
) -> str:
    """
    Retrieve and format past analyses as context string.

    Tries same ticker first, falls back to sector if < 2 results.
    Returns empty string if playbook is empty.
    """
    store = PlaybookStore()
    query = f"investment analysis {ticker}"

    same_ticker = store.query_similar(query, ticker=ticker, n_results=3)

    if len(same_ticker) < 2 and sector:
        sector_entries = store.query_similar(query, sector=sector, n_results=3)
        # Avoid duplicates
        seen = {e.id for e in same_ticker}
        extra = [e for e in sector_entries if e.id not in seen]
    else:
        extra = []

    entries = (same_ticker + extra)[:max_entries]
    if not entries:
        return ""

    lines = []
    if same_ticker:
        lines.append(f"Previous analyses of {ticker}:")
        for e in same_ticker:
            lines.append(_format_entry(e))

    if extra:
        lines.append(f"\nSimilar sector analyses ({sector}):")
        for e in extra:
            lines.append(_format_entry(e))

    context = "\n".join(lines)
    logger.info("Retrieved %d playbook entries for %s", len(entries), ticker)
    return context
