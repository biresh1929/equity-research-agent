"""Chainlit chat UI for the Financial Research Agent."""

import sys
import re
import uuid
import logging
from pathlib import Path
from typing import Literal

# Chainlit changes cwd — ensure project root is always on sys.path
sys.path.insert(0, str(Path(__file__).parent))

import chainlit as cl

from guardrails.input_guard import guard_input
from guardrails.output_guard import guard_output
from guardrails.audit_logger import log_interaction, log_guardrail_block
from ui.charts import (
    make_score_chart,
    make_technical_chart,
    make_risk_chart,
    make_history_chart,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parse user message → ticker + mode
# ---------------------------------------------------------------------------

# Strip leading instruction prefixes that appear when users copy test instructions verbatim
_INSTRUCTION_PREFIX_RE = re.compile(
    r"^(?:type|click|press|enter|send|write|input|ask)\s*:?\s*[\"']?",
    re.IGNORECASE,
)

_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")
_SEC_KEYWORDS = re.compile(
    r"\b(10-k|10-q|sec|filing|annual report|quarterly report|"
    r"risk factor|edgar|10k|10q)\b",
    re.IGNORECASE,
)
_COMBINED_KEYWORDS = re.compile(
    r"\b(full|complete|comprehensive|combined|both|deep.?dive)\b",
    re.IGNORECASE,
)


# Common company names → canonical tickers (used when user types name instead of ticker)
_NAME_TO_TICKER = {
    "APPLE": "AAPL", "MICROSOFT": "MSFT", "GOOGLE": "GOOGL", "ALPHABET": "GOOGL",
    "AMAZON": "AMZN", "TESLA": "TSLA", "META": "META", "FACEBOOK": "META",
    "NVIDIA": "NVDA", "NETFLIX": "NFLX", "INTEL": "INTC", "BROADCOM": "AVGO",
    "BERKSHIRE": "BRK-B", "JPMORGAN": "JPM", "GOLDMAN": "GS", "MORGAN": "MS",
    "RELIANCE": "RELIANCE.NS", "INFOSYS": "INFY", "PALANTIR": "PLTR",
    "SHOPIFY": "SHOP", "SALESFORCE": "CRM", "ORACLE": "ORCL", "ADOBE": "ADBE",
}


def parse_request(text: str) -> tuple[str, Literal["stock", "sec", "combined"], str]:
    """Return (ticker, mode, filing_type)."""
    # Strip instruction prefixes (e.g. "Type: " when users copy test instructions)
    text = _INSTRUCTION_PREFIX_RE.sub("", text).strip().strip("\"'")

    text_upper = text.upper()

    # Check for company name mentions before ticker regex (e.g. "Tesla" → TSLA)
    for name, mapped_ticker in _NAME_TO_TICKER.items():
        if re.search(rf"\b{name}\b", text_upper):
            ticker = mapped_ticker
            break
    else:
        tickers = _TICKER_RE.findall(text_upper)
        stop_words = {
            "A", "I", "IN", "AT", "BY", "BE", "DO", "GO", "IF", "IS", "IT",
            "ME", "MY", "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US", "WE",
            "GET", "FOR", "AND", "THE", "BUY", "CAN", "ALL", "SEC", "HAS",
            # Common English words that look like tickers
            "TYPE", "FULL", "BOTH", "GIVE", "SHOW", "FIND", "WHAT", "WITH",
            "FROM", "THIS", "THAT", "WILL", "YOUR", "HAVE", "MORE", "ALSO",
            "JUST", "INTO", "OVER", "EACH", "ONLY", "SOME", "BEEN", "LIKE",
            "WELL", "LOOK", "WANT", "KNOW", "TELL", "DOES", "MANY", "LONG",
            "DEEP", "DIVE", "DONE", "OPEN", "READ", "HELP", "LIST", "NEXT",
            "LAST", "REAL", "NEWS", "PART", "TERM", "NOTE", "VIEW", "THEN",
            "THAN", "WHEN", "MOST", "MUCH", "LETS", "HIGH", "GOOD", "MAKE",
            "TAKE", "BACK", "EVEN", "VERY", "TIME", "HERE", "SUCH", "COMBINED",
            "ANALYSIS", "RESEARCH", "STOCK", "MARKET", "ANNUAL", "COMPLETE",
        }
        tickers = [t for t in tickers if t not in stop_words]
        ticker = tickers[0] if tickers else "AAPL"

    has_sec = bool(_SEC_KEYWORDS.search(text))
    has_combined = bool(_COMBINED_KEYWORDS.search(text))

    # "combined" keyword alone is enough — user asking for "full/comprehensive/combined"
    # analysis implies they want both stock research AND SEC filing.
    # SEC-only mode requires an explicit SEC/filing keyword without a combined modifier.
    if has_combined:
        mode: Literal["stock", "sec", "combined"] = "combined"
    elif has_sec:
        mode = "sec"
    else:
        mode = "stock"

    filing_type = "10-Q" if "10-q" in text.lower() or "quarterly" in text.lower() else "10-K"
    return ticker, mode, filing_type


def _mode_label(mode: str) -> str:
    return {"stock": "Stock Research", "sec": "SEC Filing Analysis", "combined": "Combined Research"}.get(mode, mode)


# ---------------------------------------------------------------------------
# UI helper functions (display only — no backend logic)
# ---------------------------------------------------------------------------

def _guardrail_badge(input_scan: dict) -> str:
    """Format the input guardrail scan result as a one-line badge."""
    if input_scan.get("token_limit"):
        return "🛡️ *Guardrails: input truncated to 2000 tokens — scan passed.*"
    return "🛡️ *Guardrails: input scan clean — injection, off-topic, and content checks passed.*"


def _decision_emoji(decision: str) -> str:
    return {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴", "REJECT": "⛔"}.get(decision.upper(), "⚪")


def _score_bar(pct: float) -> str:
    """Return a 10-char ASCII bar for a 0-100 score."""
    filled = round(pct / 10)
    return "█" * filled + "░" * (10 - filled)


def _decision_card(ticker: str, structured: dict) -> str:
    """
    Build a markdown scorecard shown after the investment brief.
    Reads from structured_output which is already populated by generate_report node.
    """
    decision = structured.get("decision", "")
    conviction = structured.get("conviction", "")
    health = structured.get("health_score", 0) or 0
    growth = structured.get("growth_score", 0) or 0
    hard_fails = structured.get("hard_fails") or []
    conservative = structured.get("conservative_position", "—")
    neutral = structured.get("neutral_position", "—")
    aggressive = structured.get("aggressive_position", "—")

    if not decision:
        return ""

    emoji = _decision_emoji(decision)
    conviction_emoji = {"HIGH": "🔥", "MEDIUM": "💡", "LOW": "🌱"}.get(conviction.upper(), "")

    lines = [
        f"---",
        f"### 📊 Research Scorecard — {ticker}",
        f"",
        f"| | |",
        f"|---|---|",
        f"| **Decision** | {emoji} **{decision}** |",
        f"| **Conviction** | {conviction_emoji} {conviction} |",
        f"| **Health Score** | `{health:.1f}%` {_score_bar(health)} |",
        f"| **Growth Score** | `{growth:.1f}%` {_score_bar(growth)} |",
    ]

    if hard_fails:
        lines.append(f"| **Hard Gates Failed** | {', '.join(hard_fails)} |")

    lines += [
        f"",
        f"**Risk Sizing**",
        f"",
        f"| Profile | Position |",
        f"|---|---|",
        f"| 🛡️ Conservative | {conservative} |",
        f"| ⚖️ Neutral | {neutral} |",
        f"| 🚀 Aggressive | {aggressive} |",
        f"",
        f"---",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chainlit lifecycle
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_start():
    session_id = str(uuid.uuid4())[:8]
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("last_ticker", None)
    cl.user_session.set("last_entry_id", None)

    await cl.Message(
        content=(
            "# Financial Research Agent\n\n"
            "Ask me to research any stock. Examples:\n"
            "- `Research AAPL` — stock fundamentals, technicals, and Bull/Bear analysis\n"
            "- `Analyse Tesla's 10-K` — deep-dive into SEC annual filing\n"
            "- `Full combined analysis for MSFT` — stock + SEC together\n"
            "- `/history` — browse all past analyses\n"
            "- `/history AAPL` — filter history to one ticker\n\n"
            "⚠️ *For informational purposes only. Not financial advice.*"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("session_id") or str(uuid.uuid4())[:8]
    user_text = message.content.strip()

    # --- /history command (no guardrail needed) ---
    if user_text.lower().startswith("/history"):
        parts = user_text.strip().split()
        ticker_filter = parts[1].upper() if len(parts) > 1 else None
        await _show_history(ticker_filter)
        return

    # --- Input guard ---
    sanitized, is_safe, input_scan = guard_input(user_text)
    if not is_safe:
        reason = next(iter(input_scan), "unknown")
        log_guardrail_block(sanitized, str(input_scan), reason, session_id)
        block_messages = {
            "injection": "⛔ **Guardrails blocked this request** — possible prompt injection detected.",
            "off_topic": "⛔ **Off-topic request blocked** — I specialise in financial research. Please ask about stocks or SEC filings.",
            "llamaguard": "⛔ **Request blocked** by content moderation.",
        }
        await cl.Message(content=block_messages.get(reason, "⛔ Request blocked.")).send()
        return

    # --- Guardrail badge (only shown when input passes) ---
    await cl.Message(content=_guardrail_badge(input_scan)).send()

    ticker, mode, filing_type = parse_request(user_text)
    cl.user_session.set("last_ticker", ticker)

    # --- Status message ---
    await cl.Message(
        content=f"Starting **{_mode_label(mode)}** for **{ticker}**..."
    ).send()

    # --- Run graph with streaming steps ---
    from graph.supervisor import build_supervisor_graph, build_initial_supervisor_state

    graph = build_supervisor_graph()
    initial_state = build_initial_supervisor_state(ticker, mode, filing_type)

    comprehensive_report = ""
    structured_output = {}
    # Extra state captured for richer chart display
    _bull_arg: str = ""
    _bear_arg: str = ""
    _technicals: dict = {}

    try:
        async for event in graph.astream(initial_state, stream_mode="updates"):
            for node_name, output in event.items():
                if not output:
                    continue

                # Capture bull/bear args before summarising so the step shows them
                if node_name == "bull_bear_debate":
                    _bull_arg = output.get("bull_argument", "")
                    _bear_arg = output.get("bear_argument", "")

                # Capture technicals for the RSI/signal chart
                if node_name == "technicals_node" and output.get("technicals"):
                    _technicals = output["technicals"]

                step_text = _summarise_node(node_name, output, _bull_arg, _bear_arg)
                if step_text:
                    async with cl.Step(name=node_name) as step:
                        step.output = step_text

                # Capture final outputs
                if "comprehensive_report" in output:
                    comprehensive_report = output["comprehensive_report"]
                if "structured_output" in output:
                    structured_output = output.get("structured_output", {})
                # structured_output may also be set on generate_report node
                if node_name == "generate_report" and not structured_output:
                    structured_output = output.get("structured_output", {})

    except Exception as e:
        logger.error("Graph error: %s", e)
        comprehensive_report = f"An error occurred during analysis: {e}"

    # --- Output guard ---
    final_response, _out_safe, output_scan = guard_output(user_text, comprehensive_report)

    # --- Audit log ---
    log_interaction(
        user_input=sanitized,
        agent_output=final_response,
        ticker=ticker,
        decision=structured_output.get("decision", ""),
        guardrail_results={**input_scan, **output_scan},
        session_id=session_id,
    )

    # --- Send final report ---
    await cl.Message(content=final_response).send()

    # --- Decision scorecard (shown after report for stock/combined modes) ---
    if mode in ("stock", "combined") and structured_output.get("decision"):
        card = _decision_card(ticker, structured_output)
        if card:
            await cl.Message(content=card).send()

    # --- Charts (stock / combined modes only) ---
    if mode in ("stock", "combined") and structured_output.get("decision"):
        await _send_analysis_charts(ticker, structured_output, _technicals)

    # --- Feedback buttons ---
    entry_id = structured_output.get("playbook_entry_id") or f"{ticker}_{mode}"
    cl.user_session.set("last_entry_id", entry_id)

    actions = [
        cl.Action(name="feedback_correct", payload={"value": "correct"}, label="✓ Analysis looks correct"),
        cl.Action(name="feedback_incorrect", payload={"value": "incorrect"}, label="✗ Analysis seems wrong"),
    ]
    await cl.Message(
        content="*Was this analysis helpful? Your feedback improves future analyses.*",
        actions=actions,
    ).send()


# ---------------------------------------------------------------------------
# Node step summaries
# ---------------------------------------------------------------------------

def _summarise_node(node_name: str, output: dict, bull_arg: str = "", bear_arg: str = "") -> str:
    # Bull/Bear debate gets full content, not just a one-liner
    if node_name == "bull_bear_debate":
        bull = bull_arg or output.get("bull_argument", "")
        bear = bear_arg or output.get("bear_argument", "")
        parts = ["**Adversarial debate between Bull and Bear analysts:**\n"]
        if bull:
            parts.append(f"**🐂 Bull Case:**\n{bull}")
        if bear:
            parts.append(f"\n**🐻 Bear Case:**\n{bear}")
        return "\n".join(parts) if len(parts) > 1 else "Bull/Bear adversarial debate completed."

    summaries = {
        "run_stock_research": "Completed stock research (fundamentals, technicals, news, debate).",
        "run_sec_analysis": "Completed SEC filing analysis (risk, sentiment, fundamentals, math).",
        "combine_reports": "Combined stock and SEC reports into comprehensive brief.",
        "load_playbook_context": "Retrieved relevant past analyses from institutional memory.",
        "fundamentals_node": "Fetched fundamentals: P/E, margins, growth rates.",
        "technicals_node": "Computed RSI, MACD, SMAs, volume trend.",
        "news_sentiment_node": "Analysed recent news sentiment.",
        "data_quality_check": f"Data quality: {output.get('data_quality', 'checked')} ({output.get('fields_available', '?')}/{output.get('fields_total', '?')} fields).",
        "research_manager": "Research Manager synthesised all data sources.",
        "compute_scores": (
            f"Health score: **{output.get('health_score', 0):.1f}%** {_score_bar(output.get('health_score', 0))}  "
            f"| Growth score: **{output.get('growth_score', 0):.1f}%** {_score_bar(output.get('growth_score', 0))}"
        ),
        "conservative_analyst_node": f"Conservative sizing: {output.get('conservative_sizing', 'generated')}.",
        "neutral_analyst_node": f"Neutral sizing: {output.get('neutral_sizing', 'generated')}.",
        "aggressive_analyst_node": f"Aggressive sizing: {output.get('aggressive_sizing', 'generated')}.",
        "portfolio_manager": (
            f"Portfolio Manager: {_decision_emoji(output.get('decision', ''))} **{output.get('decision', 'pending')}** "
            f"— conviction {output.get('conviction', '')}."
        ),
        "generate_report": "Investment brief generated.",
        "save_to_playbook": "Analysis saved to institutional memory (playbook).",
        "early_reject": f"Early reject: insufficient data ({output.get('fields_available', 0)}/{output.get('fields_total', 12)} fields).",
        "resolve_cik": f"Resolved CIK: {output.get('cik', '')}.",
        "download_filing": f"Downloaded {output.get('filing_path', 'filing')}.",
        "chunk_filing": "Chunked filing into sections.",
        "enrich_chunks": "Enriched chunks with descriptions and search queries.",
        "build_index": "Built ChromaDB + BM25 hybrid index.",
        "risk_analyst": "Risk factor analysis complete.",
        "sentiment_analyst": "Management sentiment analysis complete.",
        "fundamental_analyst": "Fundamental analysis complete.",
        "math_agent": f"Math computations: {list(output.get('math_results', {}).keys()) or 'complete'}.",
        "generate_filing_report": "Filing report generated.",
    }
    return summaries.get(node_name, "")


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

async def _send_analysis_charts(
    ticker: str,
    structured: dict,
    technicals: dict,
) -> None:
    """Generate and send the three analysis charts after an analysis completes."""
    try:
        # 1. Score donuts
        health = structured.get("health_score") or 0.0
        growth = structured.get("growth_score") or 0.0
        score_bytes = make_score_chart(health, growth)
        img_scores = cl.Image(name="scores.png", display="inline", content=score_bytes)
        await cl.Message(content="", elements=[img_scores]).send()
    except Exception as e:
        logger.warning("Score chart failed: %s", e)

    try:
        # 2. RSI + technical signals
        if technicals:
            tech_bytes = make_technical_chart(technicals)
            img_tech = cl.Image(name="technicals.png", display="inline", content=tech_bytes)
            await cl.Message(content="", elements=[img_tech]).send()
    except Exception as e:
        logger.warning("Technical chart failed: %s", e)

    try:
        # 3. Risk-tier sizing
        conservative = structured.get("conservative_position") or ""
        neutral      = structured.get("neutral_position") or ""
        aggressive   = structured.get("aggressive_position") or ""
        if any([conservative, neutral, aggressive]):
            risk_bytes = make_risk_chart(conservative, neutral, aggressive)
            img_risk = cl.Image(name="risk.png", display="inline", content=risk_bytes)
            await cl.Message(content="", elements=[img_risk]).send()
    except Exception as e:
        logger.warning("Risk chart failed: %s", e)


async def _show_history(ticker_filter: str | None = None) -> None:
    """Handle /history [TICKER] command — show playbook table + chart."""
    try:
        from knowledge_base.playbook_store import PlaybookStore
        store = PlaybookStore()
        entries = store._get_all_entries()
    except Exception as e:
        await cl.Message(content=f"Could not load playbook: {e}").send()
        return

    if ticker_filter:
        entries = [e for e in entries if e.ticker == ticker_filter]

    if not entries:
        msg = f"No analyses found{f' for **{ticker_filter}**' if ticker_filter else ''}."
        await cl.Message(content=msg).send()
        return

    entries.sort(key=lambda e: e.date, reverse=True)

    decision_emoji = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴", "REJECT": "⛔"}
    feedback_emoji = {"correct": "✓", "incorrect": "✗"}

    header = (
        f"## 📚 Playbook History"
        + (f" — {ticker_filter}" if ticker_filter else "")
        + f"\n*{len(entries)} {'entries' if len(entries) != 1 else 'entry'} stored*\n\n"
    )
    table_lines = [
        "| # | Ticker | Date | Mode | Decision | Conv. | Health | Growth | Feedback |",
        "|---|--------|------|------|----------|-------|--------|--------|----------|",
    ]
    for i, e in enumerate(entries[:20], 1):
        de = decision_emoji.get(e.decision, "⚪")
        fb = feedback_emoji.get(e.user_feedback or "", "—")
        table_lines.append(
            f"| {i} | **{e.ticker}** | {e.date.strftime('%Y-%m-%d')} | {e.mode} "
            f"| {de} {e.decision} | {e.conviction} "
            f"| {e.health_score:.0f}% | {e.growth_score:.0f}% | {fb} |"
        )

    await cl.Message(content=header + "\n".join(table_lines)).send()

    # History chart (only if ≥ 2 unique tickers or entries)
    try:
        if len(entries) >= 2:
            chart_bytes = make_history_chart(entries)
            if chart_bytes:
                img = cl.Image(name="history.png", display="inline", content=chart_bytes)
                await cl.Message(content="", elements=[img]).send()
    except Exception as e:
        logger.warning("History chart failed: %s", e)


# ---------------------------------------------------------------------------
# Feedback callbacks
# ---------------------------------------------------------------------------

@cl.action_callback("feedback_correct")
async def on_correct(action: cl.Action):
    entry_id = cl.user_session.get("last_entry_id")
    if entry_id:
        _update_feedback(entry_id, "correct")
    await cl.Message(content="Thank you! Feedback saved — this analysis will strengthen future research.").send()


@cl.action_callback("feedback_incorrect")
async def on_incorrect(action: cl.Action):
    entry_id = cl.user_session.get("last_entry_id")
    if entry_id:
        _update_feedback(entry_id, "incorrect")
    await cl.Message(content="Thank you for the feedback. This will be noted for future improvement.").send()


def _update_feedback(entry_id: str, feedback: str) -> None:
    try:
        from knowledge_base.playbook_store import PlaybookStore
        store = PlaybookStore()
        store.update_feedback(entry_id, feedback)
    except Exception as e:
        logger.warning("Feedback update failed: %s", e)
