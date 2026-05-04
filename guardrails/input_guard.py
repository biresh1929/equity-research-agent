"""Input guardrails — LlamaGuard fast check + pattern-based injection detection."""

import re
import logging

from groq import Groq
from config.settings import settings
from .financial_guard import is_off_topic, is_finance_related

logger = logging.getLogger(__name__)

_LLAMAGUARD_MODEL = "meta-llama/llama-guard-3-8b"

_INJECTION_PATTERNS = [
    r"ignore (previous|all|your) instructions",
    r"forget everything (above|before|previously)",
    r"you are now (a|an) (?!financial|equity|investment)",
    r"jailbreak|DAN mode|developer mode|unrestricted mode",
    r"reveal (your|the|api|secret) (prompt|key|instructions|system)",
    r"act as (a|an) (?!financial|research|equity)",
]

_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

_TOKEN_LIMIT = 2000


def _check_injection(text: str) -> tuple[bool, str]:
    for pat in _INJECTION_RE:
        if pat.search(text):
            return True, f"Prompt injection pattern: {pat.pattern}"
    return False, ""


def llamaguard_check(user_input: str) -> tuple[bool, str]:
    """
    Use LlamaGuard on Groq for fast content moderation.
    Returns (is_safe, category_if_unsafe).
    Falls back gracefully if the model is unavailable on the current plan.
    """
    try:
        client = Groq(api_key=settings.groq_api_key)
        response = client.chat.completions.create(
            model=_LLAMAGUARD_MODEL,
            messages=[{"role": "user", "content": user_input}],
            max_tokens=20,
        )
        result = response.choices[0].message.content.strip().lower()
        is_safe = result.startswith("safe")
        category = result if not is_safe else ""
        return is_safe, category
    except Exception:
        # LlamaGuard may not be available on all Groq plans — fail open silently
        return True, ""


def llm_guard_check(user_input: str) -> tuple[bool, str]:
    """
    Run LLM Guard PromptInjection scanner as a second injection layer.
    Falls back gracefully if llm-guard is not installed or model unavailable.
    """
    try:
        from llm_guard import scan_prompt
        from llm_guard.input_scanners import PromptInjection as LLMGuardInjection

        scanner = LLMGuardInjection(threshold=0.75)
        _sanitized, results, scores = scan_prompt([scanner], user_input)
        is_valid = all(results.values())
        if not is_valid:
            score_str = ", ".join(f"{k}={v:.2f}" for k, v in scores.items())
            return False, f"llm_guard_injection ({score_str})"
        return True, ""
    except ImportError:
        return True, ""  # llm-guard not installed — fail open
    except Exception as exc:
        logger.debug("LLM Guard check skipped: %s", exc)
        return True, ""


def guard_input(user_input: str) -> tuple[str, bool, dict]:
    """
    Run all input guardrails.

    Returns:
        (sanitized_input, is_safe, scan_results)

    Checks performed (in order):
    1. Token limit
    2. Regex injection patterns
    3. LLM Guard prompt-injection scanner
    4. Off-topic detection
    5. LlamaGuard content moderation
    """
    scan_results: dict = {}

    # 1. Token limit
    if len(user_input) > _TOKEN_LIMIT:
        user_input = user_input[:_TOKEN_LIMIT]
        scan_results["token_limit"] = "truncated"

    # 2. Regex injection patterns (fast, no network)
    injected, reason = _check_injection(user_input)
    if injected:
        scan_results["injection"] = reason
        return user_input, False, scan_results

    # 3. LLM Guard prompt-injection scanner (transformer-based, optional)
    is_safe_llmg, llmg_reason = llm_guard_check(user_input)
    if not is_safe_llmg:
        scan_results["injection"] = llmg_reason
        return user_input, False, scan_results

    # 4. Off-topic check (warn but don't block — be permissive for finance-adjacent)
    off_topic, category = is_off_topic(user_input)
    if off_topic and not is_finance_related(user_input):
        scan_results["off_topic"] = category
        return user_input, False, scan_results

    # 5. LlamaGuard
    is_safe, category = llamaguard_check(user_input)
    if not is_safe:
        scan_results["llamaguard"] = category
        return user_input, False, scan_results

    scan_results["status"] = "clean"
    return user_input, True, scan_results
