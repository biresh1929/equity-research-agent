"""JSONL audit logger — records all interactions for compliance."""

import json
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_AUDIT_LOG = Path("data/audit_log.jsonl")


def _ensure_dir() -> None:
    _AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)


def log_interaction(
    user_input: str,          # already anonymized before passing here
    agent_output: str,
    ticker: str = "",
    decision: str = "",
    guardrail_results: dict | None = None,
    session_id: str | None = None,
) -> str:
    """Write a JSONL audit record. Returns the session_id used."""
    _ensure_dir()
    sid = session_id or str(uuid.uuid4())[:8]

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": sid,
        "ticker": ticker,
        "input_length": len(user_input),
        "input_preview": user_input[:100],          # short preview, not full PII
        "output_length": len(agent_output),
        "decision": decision,
        "guardrail_scan": guardrail_results or {},
    }

    try:
        with _AUDIT_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.error("Audit log write failed: %s", e)

    return sid


def log_guardrail_block(
    user_input: str,
    reason: str,
    scanner: str,
    session_id: str | None = None,
) -> None:
    """Log a blocked request."""
    _ensure_dir()
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id or "anon",
        "event": "BLOCKED",
        "scanner": scanner,
        "reason": reason,
        "input_preview": user_input[:100],
    }
    try:
        with _AUDIT_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.error("Audit log write failed: %s", e)
