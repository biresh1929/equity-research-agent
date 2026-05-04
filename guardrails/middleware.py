"""Guardrail middleware — wraps the supervisor graph with input/output guards."""

import logging
from typing import Literal

from .input_guard import guard_input
from .output_guard import guard_output
from .audit_logger import log_interaction, log_guardrail_block

logger = logging.getLogger(__name__)

_BLOCK_MESSAGES = {
    "injection": (
        "⛔ I cannot process this request. It appears to contain instructions "
        "that attempt to override my safety guidelines."
    ),
    "off_topic": (
        "I'm specialised for financial research. Please ask about stocks, "
        "SEC filings, or investment analysis."
    ),
    "llamaguard": (
        "⛔ This request was flagged by content moderation and cannot be processed."
    ),
}


def _block_reason(scan_results: dict) -> str:
    for key in ("injection", "llamaguard", "off_topic"):
        if key in scan_results:
            return key
    return "unknown"


def run_with_guardrails(
    user_input: str,
    ticker: str,
    mode: Literal["stock", "sec", "combined"],
    filing_type: str = "10-K",
    session_id: str | None = None,
) -> tuple[str, dict]:
    """
    Full pipeline with guardrails:
      input_guard → supervisor_graph → output_guard → audit_log

    Returns (final_response, metadata).
    """
    from graph.supervisor import build_supervisor_graph, build_initial_supervisor_state

    # --- Input guard ---
    sanitized_input, is_safe, input_scan = guard_input(user_input)

    if not is_safe:
        reason_key = _block_reason(input_scan)
        block_msg = _BLOCK_MESSAGES.get(reason_key, "⛔ Request blocked by safety guardrails.")
        log_guardrail_block(sanitized_input, str(input_scan), reason_key, session_id)
        return block_msg, {"blocked": True, "reason": reason_key, "scan": input_scan}

    # --- Run supervisor graph ---
    try:
        graph = build_supervisor_graph()
        state = build_initial_supervisor_state(ticker, mode, filing_type)
        result = graph.invoke(state)
        raw_response = result.get("comprehensive_report", "Analysis completed but no report was generated.")
        decision = result.get("structured_output", {}).get("decision", "")
    except Exception as e:
        logger.error("Supervisor graph error: %s", e)
        raw_response = f"An error occurred during analysis: {e}"
        decision = ""

    # --- Output guard ---
    final_response, out_safe, output_scan = guard_output(sanitized_input, raw_response)

    # --- Audit log ---
    sid = log_interaction(
        user_input=sanitized_input,
        agent_output=final_response,
        ticker=ticker,
        decision=decision,
        guardrail_results={**input_scan, **output_scan},
        session_id=session_id,
    )

    metadata = {
        "session_id": sid,
        "blocked": False,
        "input_scan": input_scan,
        "output_scan": output_scan,
        "ticker": ticker,
        "mode": mode,
    }
    return final_response, metadata
