"""7-phase stock research LangGraph — Articles 1 + 3 + Playbook context (C)."""

import json
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.types import Send

from config.settings import settings
from config.prompts import (
    RESEARCH_MANAGER_SYSTEM,
    RESEARCH_MANAGER_HUMAN,
    PORTFOLIO_MANAGER_SYSTEM,
    PORTFOLIO_MANAGER_HUMAN,
    REPORT_TEMPLATE,
)
from graph.state import ResearchState
from tools import get_fundamentals, get_technicals, get_news_sentiment, get_analyst_ratings
from scoring import compute_health_score, compute_growth_score, compute_daily_liquidity
from debate import bull_bear_debate_node
from risk_sizing import conservative_analyst_node, neutral_analyst_node, aggressive_analyst_node

# ---------------------------------------------------------------------------
# Phase 0 — Load playbook context (Mandatory C)
# ---------------------------------------------------------------------------

def load_playbook_context_node(state: ResearchState) -> dict:
    """Retrieve relevant past analyses before starting data gathering."""
    try:
        from knowledge_base.playbook_retriever import retrieve_playbook_context
        context = retrieve_playbook_context(state["ticker"])
        if context:
            return {"messages": [SystemMessage(content=f"[Playbook Context]\n{context}")]}
    except Exception:
        pass  # playbook is empty or not yet built — that's fine
    return {}


# ---------------------------------------------------------------------------
# Phase 1 — Parallel data gathering
# ---------------------------------------------------------------------------

def fundamentals_node(state: ResearchState) -> dict:
    result_str = get_fundamentals.invoke({"ticker": state["ticker"]})
    try:
        data = json.loads(result_str)
    except Exception:
        data = {"error": result_str}
    return {"fundamentals": data}


def technicals_node(state: ResearchState) -> dict:
    result_str = get_technicals.invoke({"ticker": state["ticker"]})
    try:
        data = json.loads(result_str)
    except Exception:
        data = {"error": result_str}
    return {"technicals": data}


def news_sentiment_node(state: ResearchState) -> dict:
    result_str = get_news_sentiment.invoke({"ticker": state["ticker"]})
    try:
        data = json.loads(result_str)
    except Exception:
        data = {"error": result_str}
    return {"news_sentiment": data}


def fan_out_data_gather(state: ResearchState) -> list[Send]:
    return [
        Send("fundamentals_node", state),
        Send("technicals_node", state),
        Send("news_sentiment_node", state),
    ]


# ---------------------------------------------------------------------------
# Phase 2 — Data quality check (Article 3)
# ---------------------------------------------------------------------------

HEALTH_FIELDS = [
    "trailingPE", "trailingEps", "revenueGrowth", "profitMargins",
    "debtToEquity", "returnOnEquity", "currentRatio", "freeCashflow",
    "operatingMargins", "earningsGrowth", "marketCap", "current_price",
]


def data_quality_check_node(state: ResearchState) -> dict:
    fundamentals = state.get("fundamentals", {})
    available = sum(1 for f in HEALTH_FIELDS if fundamentals.get(f) is not None)
    total = len(HEALTH_FIELDS)
    coverage = available / total

    if coverage >= 0.66:
        quality = "GOOD"
    elif coverage >= 0.33:
        quality = "MARGINAL"
    else:
        quality = "POOR"

    return {
        "data_quality": quality,
        "fields_available": available,
        "fields_total": total,
    }


def should_continue_after_quality(state: ResearchState) -> str:
    return "research_manager" if state["data_quality"] != "POOR" else "early_reject"


def early_reject_node(state: ResearchState) -> dict:
    available = state.get("fields_available", 0)
    total = state.get("fields_total", 0)
    return {
        "decision": "REJECT",
        "conviction": "HIGH",
        "hard_fails": [f"INSUFFICIENT_DATA: only {available}/{total} fields available"],
        "investment_brief": (
            f"## Research Rejected: {state['ticker']}\n\n"
            f"**Reason:** Insufficient data — only {available}/{total} required fields available.\n\n"
            f"This typically occurs for tickers on emerging market exchanges with limited API coverage.\n\n"
            "⚠️ *Not financial advice.*"
        ),
        "structured_output": {
            "ticker": state["ticker"],
            "decision": "REJECT",
            "conviction": "HIGH",
            "reason": "insufficient_data",
            "fields_available": available,
            "fields_total": total,
        },
    }


# ---------------------------------------------------------------------------
# Phase 3 — Research Manager synthesis
# ---------------------------------------------------------------------------

def research_manager_node(state: ResearchState) -> dict:
    llm = ChatGroq(
        model=settings.primary_model,
        temperature=0.1,
        api_key=settings.groq_api_key,
    )

    # Extract playbook context from messages if present
    playbook_ctx = ""
    for msg in state.get("messages", []):
        if isinstance(msg, SystemMessage) and "[Playbook Context]" in msg.content:
            playbook_ctx = msg.content

    human_content = RESEARCH_MANAGER_HUMAN.format(
        ticker=state["ticker"],
        fundamentals=json.dumps(state.get("fundamentals", {}), indent=2),
        technicals=json.dumps(state.get("technicals", {}), indent=2),
        news_sentiment=json.dumps(state.get("news_sentiment", {}), indent=2),
        playbook_context=playbook_ctx,
    )

    response = llm.invoke([
        SystemMessage(content=RESEARCH_MANAGER_SYSTEM),
        HumanMessage(content=human_content),
    ])
    return {"research_summary": response.content}


# ---------------------------------------------------------------------------
# Phase 4 — Scores + Debate
# ---------------------------------------------------------------------------

def compute_scores_node(state: ResearchState) -> dict:
    fundamentals = state.get("fundamentals", {})
    technicals = state.get("technicals", {})

    health_score, health_breakdown = compute_health_score(fundamentals)
    growth_score, growth_breakdown = compute_growth_score(fundamentals)
    daily_liquidity = compute_daily_liquidity(fundamentals, technicals)

    return {
        "health_score": health_score,
        "growth_score": growth_score,
        "daily_liquidity": daily_liquidity,
        "health_breakdown": health_breakdown,
        "growth_breakdown": growth_breakdown,
    }


# ---------------------------------------------------------------------------
# Phase 5 — Risk sizing fan-out
# ---------------------------------------------------------------------------

def fan_out_risk_sizing(state: ResearchState) -> list[Send]:
    return [
        Send("conservative_analyst_node", state),
        Send("neutral_analyst_node", state),
        Send("aggressive_analyst_node", state),
    ]


# ---------------------------------------------------------------------------
# Phase 6 — Portfolio Manager hard gates (Article 3)
# ---------------------------------------------------------------------------

def portfolio_manager_node(state: ResearchState) -> dict:
    health = state.get("health_score", 0.0)
    growth = state.get("growth_score", 0.0)
    liquidity = state.get("daily_liquidity", 0.0)
    coverage = (state.get("fundamentals", {}).get("numberOfAnalystOpinions") or 0)

    hard_fails = []
    if health < settings.min_health_score:
        hard_fails.append(f"health_score={health:.1f}% < {settings.min_health_score}%")
    if growth < settings.min_growth_score:
        hard_fails.append(f"growth_score={growth:.1f}% < {settings.min_growth_score}%")
    if liquidity < settings.min_daily_liquidity:
        hard_fails.append(f"daily_liquidity=${liquidity:,.0f} < ${settings.min_daily_liquidity:,.0f}")

    if hard_fails:
        return {
            "decision": "SELL",
            "conviction": "HIGH",
            "hard_fails": hard_fails,
            "pm_rationale": f"Hard gate failure: {'; '.join(hard_fails)}",
        }

    # All gates passed — ask LLM for nuanced decision
    llm = ChatGroq(
        model=settings.primary_model,
        temperature=0.1,
        api_key=settings.groq_api_key,
    )

    human = PORTFOLIO_MANAGER_HUMAN.format(
        ticker=state["ticker"],
        health_score=health,
        growth_score=growth,
        hard_fails=hard_fails,
        bull_argument=state.get("bull_argument", ""),
        bear_argument=state.get("bear_argument", ""),
        conservative_sizing=state.get("conservative_sizing", ""),
        neutral_sizing=state.get("neutral_sizing", ""),
        aggressive_sizing=state.get("aggressive_sizing", ""),
    )

    response = llm.invoke([
        SystemMessage(content=PORTFOLIO_MANAGER_SYSTEM),
        HumanMessage(content=human),
    ])

    try:
        parsed = json.loads(response.content)
        decision = parsed.get("decision", "HOLD")
        conviction = parsed.get("conviction", "MEDIUM")
        rationale = parsed.get("rationale", "")
    except json.JSONDecodeError:
        decision = "HOLD"
        conviction = "LOW"
        rationale = response.content[:300]

    return {
        "decision": decision,
        "conviction": conviction,
        "hard_fails": [],
        "pm_rationale": rationale,
    }


# ---------------------------------------------------------------------------
# Phase 7 — Generate report
# ---------------------------------------------------------------------------

def _build_bullets(data: dict, keys: list[str]) -> str:
    lines = []
    for k in keys:
        val = data.get(k)
        if val is not None:
            lines.append(f"- **{k}**: {val}")
    return "\n".join(lines) if lines else "- Data unavailable"


def generate_report_node(state: ResearchState) -> dict:
    fundamentals = state.get("fundamentals", {})
    technicals = state.get("technicals", {})
    news = state.get("news_sentiment", {})

    fund_bullets = _build_bullets(fundamentals, [
        "company_name", "sector", "current_price", "currency",
        "trailingPE", "forwardPE", "revenueGrowth", "profitMargins",
        "debtToEquity", "returnOnEquity", "freeCashflow",
    ])

    tech_bullets = _build_bullets(technicals, [
        "rsi_14", "rsi_signal", "macd_interpretation",
        "cross_status", "sma_50", "sma_200", "volume_note",
    ])

    news_headlines = news.get("headlines", [])
    news_bullets = "\n".join(
        f"- [{h['title'][:80]}]({h['url']})" for h in news_headlines[:3]
    ) or "- No recent headlines found"

    analyst_data = get_analyst_ratings.invoke({"ticker": state["ticker"]})
    try:
        analyst = json.loads(analyst_data)
    except Exception:
        analyst = {}

    analyst_bullets = _build_bullets(analyst, [
        "consensus", "num_analysts", "target_mean", "implied_upside_pct",
    ])

    brief = REPORT_TEMPLATE.format(
        ticker=state["ticker"],
        company_name=fundamentals.get("company_name", state["ticker"]),
        date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        decision=state.get("decision", "HOLD"),
        conviction=state.get("conviction", "LOW"),
        bull_argument=state.get("bull_argument", "No bull argument generated."),
        bear_argument=state.get("bear_argument", "No bear argument generated."),
        fundamentals_bullets=fund_bullets,
        technicals_bullets=tech_bullets,
        sentiment_label=news.get("overall_sentiment", "NEUTRAL"),
        sentiment_score=news.get("sentiment_score", 0.0),
        news_bullets=news_bullets,
        analyst_bullets=analyst_bullets,
        conservative_sizing=state.get("conservative_sizing", "N/A"),
        neutral_sizing=state.get("neutral_sizing", "N/A"),
        aggressive_sizing=state.get("aggressive_sizing", "N/A"),
        rationale=state.get("pm_rationale", ""),
    )

    structured = {
        "ticker": state["ticker"],
        "company": fundamentals.get("company_name", state["ticker"]),
        "date": datetime.now(timezone.utc).isoformat(),
        "decision": state.get("decision"),
        "conviction": state.get("conviction"),
        "health_score": state.get("health_score"),
        "growth_score": state.get("growth_score"),
        "hard_fails": state.get("hard_fails", []),
        "bull_summary": (state.get("bull_argument") or "")[:300],
        "bear_summary": (state.get("bear_argument") or "")[:300],
        "conservative_position": state.get("conservative_sizing"),
        "neutral_position": state.get("neutral_sizing"),
        "aggressive_position": state.get("aggressive_sizing"),
        "sentiment": news.get("overall_sentiment"),
        "pe_ratio": fundamentals.get("trailingPE"),
        "revenue_growth": fundamentals.get("revenueGrowth"),
    }

    return {"investment_brief": brief, "structured_output": structured}


# ---------------------------------------------------------------------------
# Phase 8 — Save to playbook (Mandatory C)
# ---------------------------------------------------------------------------

def save_to_playbook_node(state: ResearchState) -> dict:
    """Index this completed analysis into the knowledge base."""
    try:
        from knowledge_base.playbook_writer import write_to_playbook
        entry_id = write_to_playbook(state)
        existing = state.get("structured_output") or {}
        return {"structured_output": {**existing, "playbook_entry_id": entry_id}}
    except Exception:
        pass  # never fail the main flow due to playbook write error
    return {}


# ---------------------------------------------------------------------------
# Build the compiled graph
# ---------------------------------------------------------------------------

def build_research_graph():
    g = StateGraph(ResearchState)

    # Register all nodes
    g.add_node("load_playbook_context", load_playbook_context_node)
    g.add_node("fundamentals_node", fundamentals_node)
    g.add_node("technicals_node", technicals_node)
    g.add_node("news_sentiment_node", news_sentiment_node)
    g.add_node("data_quality_check", data_quality_check_node)
    g.add_node("research_manager", research_manager_node)
    g.add_node("compute_scores", compute_scores_node)
    g.add_node("bull_bear_debate", bull_bear_debate_node)
    g.add_node("conservative_analyst_node", conservative_analyst_node)
    g.add_node("neutral_analyst_node", neutral_analyst_node)
    g.add_node("aggressive_analyst_node", aggressive_analyst_node)
    g.add_node("portfolio_manager", portfolio_manager_node)
    g.add_node("generate_report", generate_report_node)
    g.add_node("save_to_playbook", save_to_playbook_node)
    g.add_node("early_reject", early_reject_node)

    # Edges
    g.set_entry_point("load_playbook_context")
    g.add_conditional_edges("load_playbook_context", fan_out_data_gather,
                            ["fundamentals_node", "technicals_node", "news_sentiment_node"])
    g.add_edge("fundamentals_node", "data_quality_check")
    g.add_edge("technicals_node", "data_quality_check")
    g.add_edge("news_sentiment_node", "data_quality_check")
    g.add_conditional_edges(
        "data_quality_check",
        should_continue_after_quality,
        {"research_manager": "research_manager", "early_reject": "early_reject"},
    )
    g.add_edge("early_reject", END)
    g.add_edge("research_manager", "compute_scores")
    g.add_edge("compute_scores", "bull_bear_debate")
    g.add_conditional_edges("bull_bear_debate", fan_out_risk_sizing,
                            ["conservative_analyst_node", "neutral_analyst_node", "aggressive_analyst_node"])
    g.add_edge("conservative_analyst_node", "portfolio_manager")
    g.add_edge("neutral_analyst_node", "portfolio_manager")
    g.add_edge("aggressive_analyst_node", "portfolio_manager")
    g.add_edge("portfolio_manager", "generate_report")
    g.add_edge("generate_report", "save_to_playbook")
    g.add_edge("save_to_playbook", END)

    return g.compile()
