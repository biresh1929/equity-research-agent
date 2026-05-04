"""Pillar 3 — RAG evaluation using DeepEval with Groq as judge."""

import json
import logging
from pathlib import Path
logger = logging.getLogger(__name__)

_DATASET = Path(__file__).parent / "datasets" / "rag_cases.json"


def _build_deepeval_llm():
    """Build a DeepEvalBaseLLM subclass backed by Groq.

    Defined inline so the class properly inherits DeepEvalBaseLLM at definition
    time — avoids the __class__ reassignment TypeError caused by layout mismatch.
    """
    try:
        from deepeval.models import DeepEvalBaseLLM
        from langchain_groq import ChatGroq
        from config.settings import settings

        class _GroqDeepEvalLLM(DeepEvalBaseLLM):
            def __init__(self):
                self._chat = ChatGroq(
                    model=settings.primary_model,
                    temperature=0.0,
                    api_key=settings.groq_api_key,
                )

            def generate(self, prompt: str) -> str:
                return self._chat.invoke(prompt).content

            async def a_generate(self, prompt: str) -> str:
                response = await self._chat.ainvoke(prompt)
                return response.content

            def get_model_name(self) -> str:
                return settings.primary_model

            def load_model(self):
                return self

        return _GroqDeepEvalLLM()
    except ImportError:
        logger.warning("deepeval not installed — RAG eval skipped")
        return None


def run_rag_eval(test_cases: list[dict] | None = None) -> dict:
    """
    Run RAG evaluation using DeepEval metrics with Groq as judge.

    Metrics: Faithfulness, AnswerRelevancy, ContextualRelevancy,
             ContextualRecall, ContextualPrecision.
    """
    try:
        import deepeval
        from deepeval.metrics import (
            FaithfulnessMetric,
            AnswerRelevancyMetric,
            ContextualRelevancyMetric,
            ContextualRecallMetric,
            ContextualPrecisionMetric,
        )
        from deepeval.test_case import LLMTestCase
    except ImportError:
        logger.warning("deepeval not installed — returning empty RAG eval results")
        return {"rag_faithfulness": 0.0, "rag_answer_relevancy": 0.0, "skipped": True}

    if test_cases is None:
        test_cases = json.loads(_DATASET.read_text(encoding="utf-8"))

    groq_llm = _build_deepeval_llm()
    if groq_llm is None:
        return {"skipped": True}

    metrics_map = {
        "rag_faithfulness": FaithfulnessMetric(threshold=0.80, model=groq_llm),
        "rag_answer_relevancy": AnswerRelevancyMetric(threshold=0.75, model=groq_llm),
        "rag_contextual_relevancy": ContextualRelevancyMetric(threshold=0.70, model=groq_llm),
        "rag_contextual_recall": ContextualRecallMetric(threshold=0.70, model=groq_llm),
        "rag_contextual_precision": ContextualPrecisionMetric(threshold=0.70, model=groq_llm),
    }

    from graph.sec_graph import build_sec_graph

    scores: dict[str, list[float]] = {k: [] for k in metrics_map}
    case_results = []

    for case in test_cases:
        ticker = case.get("query", "").split()[-1].upper()

        try:
            graph = build_sec_graph()
            initial = {
                "ticker": ticker, "filing_type": "10-K",
                "cik": "", "company_name": "", "filing_path": "",
                "collection_name": "", "risk_analysis": "",
                "sentiment_analysis": "", "fundamental_analysis": "",
                "math_results": {}, "filing_report": "", "error": None,
            }
            result = graph.invoke(initial)
            actual_output = result.get("filing_report", "")
            # For retrieval context, we use the filing report sections as proxies
            retrieval_context = [actual_output[:2000]]
        except Exception as e:
            logger.error("SEC graph error for %s: %s", case["id"], e)
            actual_output = ""
            retrieval_context = [""]

        test_case = LLMTestCase(
            input=case["query"],
            actual_output=actual_output,
            expected_output=case.get("expected_output", ""),
            retrieval_context=retrieval_context,
        )

        case_score = {}
        for metric_name, metric in metrics_map.items():
            try:
                metric.measure(test_case)
                score = metric.score
                scores[metric_name].append(score)
                case_score[metric_name] = score
            except Exception as e:
                logger.warning("Metric %s failed for %s: %s", metric_name, case["id"], e)

        case_results.append({"id": case["id"], "scores": case_score})

    aggregated = {
        k: sum(v) / len(v) if v else 0.0
        for k, v in scores.items()
    }
    aggregated["cases"] = case_results
    return aggregated
