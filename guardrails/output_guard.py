"""Output guardrails — secrets scan, PII anonymisation, advice flags, disclaimer injection."""

import re
import logging

from .financial_guard import inject_disclaimer, check_unsolicited_advice

logger = logging.getLogger(__name__)

# Patterns that suggest leaked secrets
_SECRET_PATTERNS = [
    r"sk-[a-zA-Z0-9]{20,}",         # OpenAI-style keys
    r"gsk_[a-zA-Z0-9]{40,}",         # Groq keys
    r"Bearer\s+[a-zA-Z0-9._-]{20,}",
    r"password\s*[:=]\s*\S+",
    r"api[_-]?key\s*[:=]\s*\S+",
]
_SECRET_RE = [re.compile(p, re.IGNORECASE) for p in _SECRET_PATTERNS]

# PII patterns — ordered from most specific to least to avoid partial matches
_PII_PATTERNS: list[tuple[str, str]] = [
    # SSN: 123-45-6789
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN REDACTED]"),
    # Credit/debit card: 4 groups of 4 digits optionally separated by space/dash
    (r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "[CARD REDACTED]"),
    # US/international phone: +1 (555) 555-5555, 555-555-5555, (555) 555-5555
    (r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b", "[PHONE REDACTED]"),
    # Email address
    (r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", "[EMAIL REDACTED]"),
]
_PII_RE: list[tuple[re.Pattern, str]] = [
    (re.compile(pat), label) for pat, label in _PII_PATTERNS
]


def _scrub_secrets(text: str) -> tuple[str, list[str]]:
    found = []
    for pat in _SECRET_RE:
        if pat.search(text):
            found.append(pat.pattern)
            text = pat.sub("[REDACTED]", text)
    return text, found


def _anonymise_pii(text: str) -> tuple[str, list[str]]:
    """Replace PII (emails, phones, SSNs, card numbers) with labelled placeholders."""
    found: list[str] = []
    for pat, label in _PII_RE:
        if pat.search(text):
            found.append(label)
            text = pat.sub(label, text)
    return text, found


def guard_output(prompt: str, response: str) -> tuple[str, bool, dict]:
    """
    Run all output guardrails.

    Returns:
        (sanitized_response, is_safe, scan_results)

    Checks performed (in order):
    1. Secrets scrubbing
    2. PII anonymisation
    3. Unsolicited advice detection (flag only, don't block)
    4. Disclaimer injection
    """
    scan_results: dict = {}

    # 1. Secrets
    response, leaked = _scrub_secrets(response)
    if leaked:
        scan_results["secrets_scrubbed"] = leaked
        logger.warning("Secrets found and scrubbed from output: %s", leaked)

    # 2. PII anonymisation
    response, pii_found = _anonymise_pii(response)
    if pii_found:
        scan_results["pii_anonymised"] = pii_found
        logger.info("PII anonymised in output: %s", pii_found)

    # 3. Advice patterns (flag but don't block — research briefs are acceptable)
    advice_flags = check_unsolicited_advice(response)
    if advice_flags:
        scan_results["advice_flags"] = advice_flags

    # 4. LLM Guard sensitive-data scanner (optional deep-scan layer)
    response, llmg_flags = _llm_guard_sensitive_scan(prompt, response)
    if llmg_flags:
        scan_results["llm_guard_sensitive"] = llmg_flags

    # 5. Always inject disclaimer for financial output
    response = inject_disclaimer(response)
    scan_results["disclaimer_injected"] = True

    is_safe = not bool(leaked)  # only hard-fail on leaked secrets
    return response, is_safe, scan_results


def _llm_guard_sensitive_scan(prompt: str, response: str) -> tuple[str, list[str]]:
    """
    Run LLM Guard Sensitive scanner for any residual PII the regex layer may miss.
    Returns (possibly_redacted_response, list_of_flagged_entities).
    Fails open (returns original response) if llm-guard is not installed.
    """
    try:
        from llm_guard import scan_output
        from llm_guard.output_scanners import Sensitive

        scanner = Sensitive(redact=True)
        sanitized, results, scores = scan_output([scanner], prompt, response)
        flagged = [k for k, passed in results.items() if not passed]
        return sanitized, flagged
    except ImportError:
        return response, []
    except Exception as exc:
        logger.debug("LLM Guard sensitive scan skipped: %s", exc)
        return response, []
