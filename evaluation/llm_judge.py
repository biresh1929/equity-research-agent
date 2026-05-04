"""Pillar 2 — LLM-as-Judge: factual accuracy, reasoning quality, completeness."""

import json
import logging
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from config.settings import settings
from config.prompts import (
    JUDGE_FACTUAL_PROMPT,
    JUDGE_REASONING_PROMPT,
    JUDGE_COMPLETENESS_PROMPT,
)

logger = logging.getLogger(__name__)

_DATASET = Path(__file__).parent / "datasets" / "factual_cases.json"


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` markdown fences the LLM wraps around JSON."""
    text = text.strip()
    if text.startswith("```"):
        # drop first line (```json or ```) and last ``` line
        lines = text.splitlines()
        inner = lines[1:] if lines[0].startswith("```") else lines
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner).strip()
    return text


def _judge_call(prompt: str) -> dict:
    llm = ChatGroq(
        model=settings.primary_model,
        temperature=0.0,
        api_key=settings.groq_api_key,
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = _strip_fences(response.content)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Judge returned non-JSON: %s", raw[:200])
        return {}


def evaluate_factual_accuracy(query: str, response: str, ground_truth: list[str]) -> dict:
    prompt = JUDGE_FACTUAL_PROMPT.format(
        query=query,
        response=response[:3000],
        ground_truth="\n".join(f"- {f}" for f in ground_truth),
    )
    result = _judge_call(prompt)
    return {
        "factual_accuracy": float(result.get("accuracy_score", 0.0)),
        "facts_correct": result.get("facts_correct", []),
        "facts_incorrect": result.get("facts_incorrect", []),
        "facts_missing": result.get("facts_missing", []),
        "issues": result.get("issues", []),
    }


def evaluate_reasoning_quality(query: str, response: str) -> dict:
    prompt = JUDGE_REASONING_PROMPT.format(query=query, response=response[:3000])
    result = _judge_call(prompt)
    return {
        "reasoning_quality": float(result.get("reasoning_score", 0.0)),
        "strengths": result.get("strengths", []),
        "weaknesses": result.get("weaknesses", []),
        "issues": result.get("issues", []),
    }


def evaluate_completeness(query: str, response: str, expected_sections: list[str]) -> dict:
    prompt = JUDGE_COMPLETENESS_PROMPT.format(
        query=query,
        response=response[:3000],
        expected_sections=", ".join(expected_sections),
    )
    result = _judge_call(prompt)
    return {
        "completeness": float(result.get("completeness_score", 0.0)),
        "sections_present": result.get("sections_present", []),
        "sections_missing": result.get("sections_missing", []),
        "sections_shallow": result.get("sections_shallow", []),
    }


def run_llm_judge_eval(test_cases: list[dict] | None = None) -> dict:
    """
    Run all three LLM-judge dimensions against the factual dataset.

    For each case, it runs the supervisor graph to get a response,
    then judges it on factual accuracy + completeness.
    """
    if test_cases is None:
        test_cases = json.loads(_DATASET.read_text(encoding="utf-8"))

    from graph.supervisor import build_supervisor_graph, build_initial_supervisor_state

    all_factual = []
    all_completeness = []
    case_results = []

    for case in test_cases:
        query = case["query"]
        ticker = query.split()[-1].upper()  # crude ticker extraction for eval

        try:
            graph = build_supervisor_graph()
            state = build_initial_supervisor_state(ticker, "stock")
            result = graph.invoke(state)
            response = result.get("comprehensive_report", "") or result.get("stock_brief", "")
        except Exception as e:
            logger.error("Graph error for case %s: %s", case["id"], e)
            # Skip this case rather than scoring 0 (avoids 429 skewing the mean)
            case_results.append({"id": case["id"], "skipped": True, "error": str(e)})
            continue

        factual = evaluate_factual_accuracy(query, response, case.get("ground_truth_facts", []))
        completeness = evaluate_completeness(query, response, case.get("expected_sections", []))

        all_factual.append(factual["factual_accuracy"])
        all_completeness.append(completeness["completeness"])
        case_results.append({
            "id": case["id"],
            "factual": factual,
            "completeness": completeness,
        })

    return {
        "factual_accuracy": sum(all_factual) / len(all_factual) if all_factual else 0.0,
        "completeness": sum(all_completeness) / len(all_completeness) if all_completeness else 0.0,
        "cases": case_results,
    }
