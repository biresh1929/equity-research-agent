"""Four SEC filing expert personas — Risk, Sentiment, Fundamental, Math Agent."""

import json
import logging
import subprocess
import tempfile
import textwrap
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import settings
from config.prompts import (
    RISK_ANALYST_SYSTEM,
    SENTIMENT_ANALYST_SYSTEM,
    FUNDAMENTAL_ANALYST_SYSTEM,
    SEC_ANALYST_HUMAN,
    MATH_AGENT_SYSTEM,
    MATH_AGENT_HUMAN,
)
from .hybrid_search import hybrid_search

logger = logging.getLogger(__name__)

_ANALYST_QUERIES = {
    "risk": [
        "material risk factors regulatory litigation",
        "cybersecurity risks data breaches",
        "liquidity risk debt covenants going concern",
        "market risk interest rate foreign exchange",
        "operational risk supply chain disruption",
    ],
    "sentiment": [
        "management outlook forward-looking statements",
        "CEO letter to shareholders confidence language",
        "guidance uncertainty macro headwinds tailwinds",
        "strategic priorities growth initiatives",
        "competitive landscape market share",
    ],
    "fundamental": [
        "revenue net income operating income margins",
        "earnings per share diluted shares outstanding",
        "capital expenditure free cash flow",
        "balance sheet total assets liabilities equity",
        "segment performance geographic revenue breakdown",
    ],
}


def _build_context(collection_name: str, queries: list[str], top_k: int | None = None) -> str:
    top_k = top_k or settings.top_k
    seen: set[str] = set()
    chunks = []

    for query in queries:
        results = hybrid_search(collection_name, query, n_results=top_k // len(queries) + 1)
        for r in results:
            if r["chunk_id"] not in seen:
                seen.add(r["chunk_id"])
                chunks.append(r["text"])

    return "\n\n---\n\n".join(chunks[:top_k])


def _analyst_node(system: str, queries: list[str], state: dict, key: str) -> dict:
    collection_name = state["collection_name"]
    company_name = state.get("company_name", state["ticker"])
    filing_type = state.get("filing_type", "10-K")

    context = _build_context(collection_name, queries)

    llm = ChatGroq(
        model=settings.primary_model,
        temperature=0.1,
        api_key=settings.groq_api_key,
    )
    human = SEC_ANALYST_HUMAN.format(
        company_name=company_name,
        filing_type=filing_type,
        context=context[:6000],  # stay within context limits
    )
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    return {key: response.content}


def risk_analyst_node(state: dict) -> dict:
    return _analyst_node(RISK_ANALYST_SYSTEM, _ANALYST_QUERIES["risk"], state, "risk_analysis")


def sentiment_analyst_node(state: dict) -> dict:
    return _analyst_node(SENTIMENT_ANALYST_SYSTEM, _ANALYST_QUERIES["sentiment"], state, "sentiment_analysis")


def fundamental_analyst_node(state: dict) -> dict:
    return _analyst_node(FUNDAMENTAL_ANALYST_SYSTEM, _ANALYST_QUERIES["fundamental"], state, "fundamental_analysis")


# ---------------------------------------------------------------------------
# Math Agent — subprocess sandbox
# ---------------------------------------------------------------------------

_BLOCKED_IMPORTS = ["os", "subprocess", "socket", "requests", "pathlib", "shutil", "sys"]
_SAFE_IMPORTS = ["math", "statistics", "json", "re", "decimal"]


def _is_safe_code(code: str) -> bool:
    for blocked in _BLOCKED_IMPORTS:
        if f"import {blocked}" in code or f"from {blocked}" in code:
            return False
    return True


def _run_code_sandbox(code: str, timeout: int = 30) -> dict:
    """Execute code in a subprocess and return the `result` dict."""
    if not _is_safe_code(code):
        return {"error": "Blocked import detected"}

    # Inject result extractor
    full_code = textwrap.dedent(f"""
import json, math, statistics, re
result = {{}}
{code}
print(json.dumps(result))
""")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        proc = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode == 0:
            last_line = proc.stdout.strip().split("\n")[-1]
            return json.loads(last_line)
        return {"error": proc.stderr[:500]}
    except subprocess.TimeoutExpired:
        return {"error": "Code execution timed out"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON decode error: {e}"}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def math_agent_node(state: dict) -> dict:
    collection_name = state["collection_name"]
    company_name = state.get("company_name", state["ticker"])
    filing_type = state.get("filing_type", "10-K")

    math_queries = [
        "revenue net income earnings per share growth rate",
        "total assets liabilities equity debt ratio",
        "operating cash flow capital expenditure free cash flow",
    ]
    context = _build_context(collection_name, math_queries, top_k=5)

    llm = ChatGroq(
        model=settings.primary_model,
        temperature=0.0,
        api_key=settings.groq_api_key,
    )
    human = MATH_AGENT_HUMAN.format(
        context=context[:4000],
        query=f"Compute key financial ratios and growth rates for {company_name} from this {filing_type}.",
    )
    response = llm.invoke([SystemMessage(content=MATH_AGENT_SYSTEM), HumanMessage(content=human)])
    code = response.content.strip()

    # Strip any accidental markdown fences
    if code.startswith("```"):
        code = "\n".join(code.split("\n")[1:])
    if code.endswith("```"):
        code = "\n".join(code.split("\n")[:-1])

    math_results = _run_code_sandbox(code)
    return {"math_results": math_results}
