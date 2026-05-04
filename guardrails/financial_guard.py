"""Domain-specific financial guardrails — disclaimer injection + advice detection."""

import re

FINANCIAL_DISCLAIMER = (
    "\n\n---\n"
    "⚠️ *This analysis is for informational purposes only and does not constitute "
    "financial advice. Past performance is not indicative of future results. "
    "Always consult a qualified financial advisor before making investment decisions.*"
)

_ADVICE_PATTERNS = [
    r"\byou (should|must|need to) (buy|sell|invest|short|hold)\b",
    r"\bi (strongly )?recommend (buying|selling|shorting)\b",
    r"\bguaranteed (return|profit|gain|upside)\b",
    r"\byou (will|can) (definitely|certainly|surely) (make|profit|gain)\b",
    r"\bthis is a (sure thing|no-brainer|guaranteed winner)\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _ADVICE_PATTERNS]

_OFF_TOPIC_PATTERNS = [
    r"\b(politics|election|president|congress|senate)\b",
    r"\b(medical advice|diagnos|prescri|treat|cure|symptom|illness|disease)\b",
    r"\b(personal relationship|dating|romance|marriage|boyfriend|girlfriend)\b",
    r"\b(religion|god|allah|jesus|church|mosque|temple|prayer)\b",
    # Food / cooking / general lifestyle
    r"\b(recipe|cook(ing)?|food|meal|diet|nutrition|ingredient|bake|baking)\b",
    r"\b(weather|forecast|temperature|rain|snow|sunny|climate)\b",
    r"\b(sport(s)?|football|soccer|basketball|cricket|tennis|game score)\b",
    r"\b(movie|film|series|tv show|netflix show|music|song|artist|concert)\b",
    r"\b(travel|holiday|vacation|hotel|flight|tourism|restaurant)\b",
    r"\b(homework|essay|assignment|exam|study tips|tutoring)\b",
]

_OFF_TOPIC_COMPILED = [re.compile(p, re.IGNORECASE) for p in _OFF_TOPIC_PATTERNS]


def inject_disclaimer(response: str) -> str:
    """Always append the financial disclaimer if not already present."""
    if "not constitute financial advice" in response:
        return response
    return response + FINANCIAL_DISCLAIMER


def check_unsolicited_advice(response: str) -> list[str]:
    """Return list of flagged advice patterns found in response."""
    return [p.pattern for p in _COMPILED if p.search(response)]


def is_off_topic(user_input: str) -> tuple[bool, str]:
    """Return (is_off_topic, matched_category)."""
    for pattern in _OFF_TOPIC_COMPILED:
        if pattern.search(user_input):
            return True, pattern.pattern
    return False, ""


def is_finance_related(user_input: str) -> bool:
    """Quick check that the query is about stocks/finance."""
    finance_terms = re.compile(
        r"\b(stock|ticker|share|invest|portfolio|market|nasdaq|nyse|sec|"
        r"earnings|revenue|dividend|fund|etf|bond|equity|analysis|"
        r"research|buy|sell|hold|price|valuation|financial)\b",
        re.IGNORECASE,
    )
    return bool(finance_terms.search(user_input))
