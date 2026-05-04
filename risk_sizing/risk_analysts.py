"""Three risk analyst personas — Conservative, Neutral, Aggressive (Article 3)."""

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import settings
from config.prompts import (
    CONSERVATIVE_RISK_SYSTEM,
    NEUTRAL_RISK_SYSTEM,
    AGGRESSIVE_RISK_SYSTEM,
    RISK_SIZING_HUMAN,
)


def _risk_node(system_prompt: str, state: dict) -> dict:
    llm = ChatGroq(
        model=settings.primary_model,
        temperature=0.2,
        api_key=settings.groq_api_key,
    )

    human_content = RISK_SIZING_HUMAN.format(
        ticker=state["ticker"],
        decision=state.get("decision", "PENDING"),
        conviction=state.get("conviction", "UNKNOWN"),
        health_score=state.get("health_score", 0.0),
        growth_score=state.get("growth_score", 0.0),
        hard_fails=state.get("hard_fails", []),
        bull_argument=state.get("bull_argument", ""),
        bear_argument=state.get("bear_argument", ""),
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content),
    ])
    return response.content


def conservative_analyst_node(state: dict) -> dict:
    """Conservative risk analyst — recommends minimum position size (0–3% range)."""
    result = _risk_node(CONSERVATIVE_RISK_SYSTEM, state)
    return {"conservative_sizing": result}


def neutral_analyst_node(state: dict) -> dict:
    """Neutral risk analyst — balanced risk/reward position size (0–5% range)."""
    result = _risk_node(NEUTRAL_RISK_SYSTEM, state)
    return {"neutral_sizing": result}


def aggressive_analyst_node(state: dict) -> dict:
    """Aggressive risk analyst — maximum upside position size (0–8% range)."""
    result = _risk_node(AGGRESSIVE_RISK_SYSTEM, state)
    return {"aggressive_sizing": result}
