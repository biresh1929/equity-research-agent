"""Pillar 1 — Routing accuracy evaluation."""

import json
import re
import logging
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from config.settings import settings

logger = logging.getLogger(__name__)

_DATASET = Path(__file__).parent / "datasets" / "routing_cases.json"

_ROUTING_PROMPT = """Given the following user query about stock market research, classify which agent should handle it.

Options:
- "stock": For queries about stock fundamentals, technicals, news, price analysis, buy/sell recommendations
- "sec": For queries specifically about SEC filings (10-K, 10-Q, annual report, risk factors in filings)
- "combined": For queries explicitly requesting both stock analysis AND SEC filing analysis together

Query: {query}

Respond with ONLY one word: stock, sec, or combined."""


def classify_query(query: str) -> str:
    """Use Groq to classify the routing intent of a query."""
    llm = ChatGroq(
        model=settings.fast_model,
        temperature=0.0,
        api_key=settings.groq_api_key,
    )
    response = llm.invoke([HumanMessage(content=_ROUTING_PROMPT.format(query=query))])
    result = response.content.strip().lower()

    # Extract clean label
    for label in ("combined", "stock", "sec"):
        if label in result:
            return label
    return "stock"  # default


def evaluate_routing(test_cases: list[dict] | None = None) -> dict:
    """
    Run routing evaluation against dataset.

    Returns metrics dict including accuracy and per-case results.
    """
    if test_cases is None:
        test_cases = json.loads(_DATASET.read_text(encoding="utf-8"))

    correct = 0
    results = []

    for case in test_cases:
        predicted = classify_query(case["query"])
        expected = case["expected_agent"]
        passed = predicted == expected

        if passed:
            correct += 1

        results.append({
            "id": case["id"],
            "query": case["query"],
            "expected": expected,
            "predicted": predicted,
            "passed": passed,
        })

    accuracy = correct / len(test_cases) if test_cases else 0.0
    logger.info("Routing eval: %d/%d correct (%.1f%%)", correct, len(test_cases), accuracy * 100)

    return {
        "routing_accuracy": accuracy,
        "correct": correct,
        "total": len(test_cases),
        "cases": results,
    }
