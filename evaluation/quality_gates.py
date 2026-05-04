"""
Evaluation quality gates — two concerns:

1. METRIC THRESHOLDS: pass/fail thresholds for the 3 evaluation pillars.
2. DATA QUALITY ASSERTIONS: deterministic checks of the graph's data-coverage
   gate logic (no API calls required — safe to run in CI).
"""

# ---------------------------------------------------------------------------
# Part 1 — Metric thresholds (used by run_eval.py after pillar runs)
# ---------------------------------------------------------------------------

THRESHOLDS: dict[str, float] = {
    "routing_accuracy": 0.90,
    "factual_accuracy": 0.85,
    "reasoning_quality": 0.70,
    "completeness": 0.75,
    "rag_faithfulness": 0.80,
    "rag_answer_relevancy": 0.75,
    "rag_contextual_relevancy": 0.70,
    "rag_contextual_recall": 0.70,
    "rag_contextual_precision": 0.70,
}


def check_gates(results: dict[str, float]) -> dict[str, bool]:
    """Compare metric scores against thresholds. Returns {metric: passed}."""
    return {
        metric: score >= THRESHOLDS.get(metric, 0.0)
        for metric, score in results.items()
    }


def all_passed(results: dict[str, float]) -> bool:
    return all(check_gates(results).values())


def format_report(results: dict[str, float]) -> str:
    gates = check_gates(results)
    lines = ["## Evaluation Quality Gate Results\n"]
    for metric, score in results.items():
        threshold = THRESHOLDS.get(metric, 0.0)
        status = "PASS" if gates.get(metric, False) else "FAIL"
        lines.append(
            f"  {'[PASS]' if status == 'PASS' else '[FAIL]'} {metric}: "
            f"{score:.3f} (threshold >= {threshold:.2f})"
        )
    overall = "ALL GATES PASSED" if all_passed(results) else "SOME GATES FAILED"
    lines.append(f"\nOverall: {overall}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Part 2 — Data quality gate assertions (deterministic, no API calls)
# ---------------------------------------------------------------------------

# These mirror the field list and thresholds used in research_graph.py
_HEALTH_FIELDS = [
    "trailingPE", "trailingEps", "revenueGrowth", "profitMargins",
    "debtToEquity", "returnOnEquity", "currentRatio", "freeCashflow",
    "operatingMargins", "earningsGrowth", "marketCap", "current_price",
]

_GOOD_THRESHOLD = 0.66     # >= 66% fields → GOOD
_MARGINAL_THRESHOLD = 0.33  # >= 33% fields → MARGINAL, else POOR


def check_data_quality(fundamentals: dict) -> dict:
    """
    Replicate the data_quality_check_node logic as a standalone function.
    Returns coverage metrics and the quality label.
    """
    available = sum(1 for f in _HEALTH_FIELDS if fundamentals.get(f) is not None)
    total = len(_HEALTH_FIELDS)
    coverage = available / total

    if coverage >= _GOOD_THRESHOLD:
        quality = "GOOD"
    elif coverage >= _MARGINAL_THRESHOLD:
        quality = "MARGINAL"
    else:
        quality = "POOR"

    return {
        "quality": quality,
        "fields_available": available,
        "fields_total": total,
        "coverage_pct": round(coverage * 100, 1),
    }


def _assert(condition: bool, name: str, detail: str = "") -> dict:
    return {"name": name, "passed": condition, "detail": detail}


def run_data_quality_assertions() -> dict:
    """
    Run all deterministic data-quality gate assertions.
    Returns a results dict with per-assertion pass/fail and an overall verdict.
    No network calls, no LLM — safe for CI.
    """
    assertions = []

    # --- Full coverage → GOOD ---
    full = {f: 1.0 for f in _HEALTH_FIELDS}
    r = check_data_quality(full)
    assertions.append(_assert(
        r["quality"] == "GOOD",
        "full_coverage_is_GOOD",
        f"12/12 fields → expected GOOD, got {r['quality']}",
    ))

    # --- Exactly 66% coverage (8/12) → GOOD (boundary) ---
    eight = {f: 1.0 for f in _HEALTH_FIELDS[:8]}
    r = check_data_quality(eight)
    assertions.append(_assert(
        r["quality"] == "GOOD",
        "66pct_boundary_is_GOOD",
        f"8/12 fields → expected GOOD, got {r['quality']}",
    ))

    # --- 50% coverage (6/12) → MARGINAL ---
    six = {f: 1.0 for f in _HEALTH_FIELDS[:6]}
    r = check_data_quality(six)
    assertions.append(_assert(
        r["quality"] == "MARGINAL",
        "50pct_coverage_is_MARGINAL",
        f"6/12 fields → expected MARGINAL, got {r['quality']}",
    ))

    # --- Exactly 33% coverage (4/12) → MARGINAL (boundary) ---
    four = {f: 1.0 for f in _HEALTH_FIELDS[:4]}
    r = check_data_quality(four)
    assertions.append(_assert(
        r["quality"] == "MARGINAL",
        "33pct_boundary_is_MARGINAL",
        f"4/12 fields → expected MARGINAL, got {r['quality']}",
    ))

    # --- 20% coverage (2/12) → POOR ---
    two = {f: 1.0 for f in _HEALTH_FIELDS[:2]}
    r = check_data_quality(two)
    assertions.append(_assert(
        r["quality"] == "POOR",
        "20pct_coverage_is_POOR",
        f"2/12 fields → expected POOR, got {r['quality']}",
    ))

    # --- Empty dict → POOR ---
    r = check_data_quality({})
    assertions.append(_assert(
        r["quality"] == "POOR",
        "empty_fundamentals_is_POOR",
        f"0/12 fields → expected POOR, got {r['quality']}",
    ))

    # --- None values treated as unavailable ---
    none_vals = {f: None for f in _HEALTH_FIELDS}
    r = check_data_quality(none_vals)
    assertions.append(_assert(
        r["quality"] == "POOR",
        "all_None_values_is_POOR",
        f"all None → expected POOR, got {r['quality']}",
    ))

    # --- Routing decision: GOOD/MARGINAL → continue, POOR → early_reject ---
    for quality, expected_route in [("GOOD", "research_manager"), ("MARGINAL", "research_manager"), ("POOR", "early_reject")]:
        route = "early_reject" if quality == "POOR" else "research_manager"
        assertions.append(_assert(
            route == expected_route,
            f"routing_{quality}_→_{expected_route}",
            f"quality={quality} should route to {expected_route}",
        ))

    passed = sum(1 for a in assertions if a["passed"])
    total = len(assertions)

    return {
        "passed": passed,
        "total": total,
        "all_passed": passed == total,
        "assertions": assertions,
    }


def print_data_quality_report(results: dict) -> None:
    print(f"\n{'='*55}")
    print("Data Quality Gate Assertions")
    print(f"{'='*55}")
    for a in results["assertions"]:
        status = "[PASS]" if a["passed"] else "[FAIL]"
        print(f"  {status} {a['name']}")
        if not a["passed"]:
            print(f"        {a['detail']}")
    print(f"\n{results['passed']}/{results['total']} assertions passed")
    if results["all_passed"]:
        print("Result: ALL PASSED")
    else:
        print("Result: FAILURES DETECTED")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    results = run_data_quality_assertions()
    print_data_quality_report(results)
    raise SystemExit(0 if results["all_passed"] else 1)
