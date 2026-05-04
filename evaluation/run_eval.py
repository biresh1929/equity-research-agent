"""CLI runner for all three evaluation pillars."""

import sys
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path when run as: python evaluation/run_eval.py
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_RESULTS_DIR = Path("evaluation/results")


def _save_results(pillar: str, data: dict) -> Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = _RESULTS_DIR / f"{pillar}_{ts}.json"
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def run_routing(args) -> dict:
    from evaluation.routing_eval import evaluate_routing
    logger.info("Running routing evaluation...")
    results = evaluate_routing()
    saved = _save_results("routing", results)
    print(f"\nRouting accuracy: {results['routing_accuracy']:.1%} ({results['correct']}/{results['total']})")
    print(f"Results saved to: {saved}")
    return {"routing_accuracy": results["routing_accuracy"]}


def run_llm_judge(args) -> dict:
    from evaluation.llm_judge import run_llm_judge_eval
    logger.info("Running LLM-judge evaluation (factual + completeness)...")
    results = run_llm_judge_eval()
    saved = _save_results("llm_judge", results)
    print(f"\nFactual accuracy: {results['factual_accuracy']:.1%}")
    print(f"Completeness: {results['completeness']:.1%}")
    print(f"Results saved to: {saved}")
    return {k: v for k, v in results.items() if k != "cases"}


def run_rag(args) -> dict:
    from evaluation.rag_eval import run_rag_eval
    logger.info("Running RAG evaluation (DeepEval + Groq)...")
    results = run_rag_eval()
    saved = _save_results("rag", results)
    if results.get("skipped"):
        print("\nRAG evaluation skipped (deepeval not installed)")
        return {}
    for k, v in results.items():
        if k != "cases" and isinstance(v, float):
            print(f"{k}: {v:.1%}")
    print(f"Results saved to: {saved}")
    return {k: v for k, v in results.items() if k != "cases" and isinstance(v, float)}


def run_data_quality(args) -> dict:
    from evaluation.quality_gates import run_data_quality_assertions, print_data_quality_report
    logger.info("Running data quality gate assertions (deterministic, no API calls)...")
    results = run_data_quality_assertions()
    print_data_quality_report(results)
    saved = _save_results("data_quality", results)
    print(f"Results saved to: {saved}")
    # Express as a 0–1 score so it fits the combined quality-gate format
    return {"data_quality_gate": results["passed"] / results["total"]}


def main():
    parser = argparse.ArgumentParser(description="Run Financial Research Agent evaluation suite")
    parser.add_argument(
        "--pillar",
        choices=["routing", "llm_judge", "rag", "data_quality", "all"],
        default="all",
        help="Which evaluation pillar to run (default: all)",
    )
    args = parser.parse_args()

    all_scores: dict[str, float] = {}

    # Data quality assertions run first — deterministic, no API cost
    if args.pillar in ("data_quality", "all"):
        all_scores.update(run_data_quality(args))

    if args.pillar in ("routing", "all"):
        all_scores.update(run_routing(args))

    if args.pillar in ("llm_judge", "all"):
        all_scores.update(run_llm_judge(args))

    if args.pillar in ("rag", "all"):
        all_scores.update(run_rag(args))

    if args.pillar == "all" and all_scores:
        from evaluation.quality_gates import format_report, all_passed
        print("\n" + "=" * 60)
        print(format_report(all_scores))
        if not all_passed(all_scores):
            exit(1)


if __name__ == "__main__":
    main()
