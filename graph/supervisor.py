"""Supervisor graph — routes between stock research, SEC analysis, and combined mode."""

import logging
from typing import Literal

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from config.settings import settings
from graph.state import SupervisorState
from graph.research_graph import build_research_graph
from graph.sec_graph import build_sec_graph

logger = logging.getLogger(__name__)

_COMBINED_REPORT_SYSTEM = """You are a senior investment director writing a comprehensive research report.
You have two inputs:
1. Stock research brief (fundamentals, technicals, news, bull/bear debate, recommendation)
2. SEC filing deep-dive (risk factors, management sentiment, financial analysis)

Synthesise these into a single cohesive report. Highlight:
- Where the two analyses agree (conviction booster)
- Where they diverge (risk flag)
- Final investment thesis in one clear paragraph

Keep under 600 words. Use headers."""

_COMBINED_REPORT_HUMAN = """STOCK RESEARCH BRIEF:
{stock_brief}

SEC FILING ANALYSIS:
{filing_report}

Ticker: {ticker}

Write the comprehensive combined report."""


def route_to_subgraph(state: SupervisorState) -> str:
    mode = state.get("mode", "stock")
    if mode == "stock":
        return "run_stock_research"
    elif mode == "sec":
        return "run_sec_analysis"
    return "run_stock_research"  # combined starts with stock


def run_stock_research_node(state: SupervisorState) -> dict:
    graph = build_research_graph()
    from graph.state import ResearchState

    initial: dict = {
        "ticker": state["ticker"],
        "messages": [],
    }
    result = graph.invoke(initial)

    return {
        "stock_brief": result.get("investment_brief", ""),
        "structured_output": result.get("structured_output", {}),
    }


def run_sec_analysis_node(state: SupervisorState) -> dict:
    graph = build_sec_graph()
    filing_type = state.get("filing_type", "10-K")

    initial = {
        "ticker": state["ticker"],
        "filing_type": filing_type,
        "cik": "",
        "company_name": "",
        "filing_path": "",
        "collection_name": "",
        "risk_analysis": "",
        "sentiment_analysis": "",
        "fundamental_analysis": "",
        "math_results": {},
        "filing_report": "",
        "error": None,
    }
    result = graph.invoke(initial)
    return {"filing_report": result.get("filing_report", "")}


def combine_reports_node(state: SupervisorState) -> dict:
    mode = state.get("mode", "stock")

    if mode == "stock":
        return {"comprehensive_report": state.get("stock_brief", "")}

    if mode == "sec":
        return {"comprehensive_report": state.get("filing_report", "")}

    # Combined mode — run SEC if not done yet
    if not state.get("filing_report"):
        sec_result = run_sec_analysis_node(state)
        filing_report = sec_result.get("filing_report", "")
    else:
        filing_report = state.get("filing_report", "")

    stock_brief = state.get("stock_brief", "")

    if not stock_brief and not filing_report:
        return {"comprehensive_report": "No analysis available.", "error": "Both sub-graphs returned empty results."}

    if not filing_report:
        return {"comprehensive_report": stock_brief}

    if not stock_brief:
        return {"comprehensive_report": filing_report}

    llm = ChatGroq(
        model=settings.primary_model,
        temperature=0.1,
        api_key=settings.groq_api_key,
    )

    human = _COMBINED_REPORT_HUMAN.format(
        stock_brief=stock_brief[:3000],
        filing_report=filing_report[:3000],
        ticker=state["ticker"],
    )

    response = llm.invoke([
        SystemMessage(content=_COMBINED_REPORT_SYSTEM),
        HumanMessage(content=human),
    ])

    return {"comprehensive_report": response.content}


def should_run_sec(state: SupervisorState) -> str:
    mode = state.get("mode", "stock")
    if mode in ("sec", "combined"):
        return "run_sec_analysis"
    return "combine_reports"


def _entry_route(state: SupervisorState) -> str:
    """Route the initial request: skip stock research for sec-only mode."""
    mode = state.get("mode", "stock")
    if mode == "sec":
        return "run_sec_analysis"
    return "run_stock_research"


def build_supervisor_graph():
    g = StateGraph(SupervisorState)

    g.add_node("router", lambda s: s)  # pass-through routing node
    g.add_node("run_stock_research", run_stock_research_node)
    g.add_node("run_sec_analysis", run_sec_analysis_node)
    g.add_node("combine_reports", combine_reports_node)

    g.set_entry_point("router")

    # Entry routing: sec-only skips stock research
    g.add_conditional_edges(
        "router",
        _entry_route,
        {
            "run_stock_research": "run_stock_research",
            "run_sec_analysis": "run_sec_analysis",
        },
    )

    # After stock research, decide whether to also do SEC (for combined mode)
    g.add_conditional_edges(
        "run_stock_research",
        should_run_sec,
        {
            "run_sec_analysis": "run_sec_analysis",
            "combine_reports": "combine_reports",
        },
    )
    g.add_edge("run_sec_analysis", "combine_reports")
    g.add_edge("combine_reports", END)

    return g.compile()


def build_initial_supervisor_state(
    ticker: str,
    mode: Literal["stock", "sec", "combined"] = "stock",
    filing_type: str = "10-K",
) -> dict:
    return {
        "ticker": ticker.upper().strip(),
        "mode": mode,
        "filing_type": filing_type,
        "stock_brief": "",
        "structured_output": {},
        "filing_report": "",
        "comprehensive_report": "",
        "error": None,
    }
