"""Bull/Bear adversarial debate — Article 3 pattern."""

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from config.settings import settings
from config.prompts import BULL_PROMPT, BEAR_PROMPT


def bull_bear_debate_node(state: dict) -> dict:
    """
    Two-round adversarial debate:
    1. Bull makes the strongest case for buying.
    2. Bear directly refutes Bull and argues the strongest case for avoiding.

    Returns state updates: bull_argument, bear_argument.
    """
    llm = ChatGroq(
        model=settings.primary_model,
        temperature=0.4,
        api_key=settings.groq_api_key,
    )

    ticker = state["ticker"]
    summary = state.get("research_summary", "No summary available.")

    # Round 1: Bull
    bull_response = llm.invoke([
        HumanMessage(content=BULL_PROMPT.format(ticker=ticker, summary=summary))
    ])
    bull_arg = bull_response.content

    # Round 2: Bear (with full visibility of Bull's argument)
    bear_response = llm.invoke([
        HumanMessage(content=BEAR_PROMPT.format(
            ticker=ticker,
            summary=summary,
            bull_argument=bull_arg,
        ))
    ])
    bear_arg = bear_response.content

    return {
        "bull_argument": bull_arg,
        "bear_argument": bear_arg,
    }
