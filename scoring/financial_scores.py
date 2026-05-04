"""Health and growth scoring — Article 3 hard-gate logic."""

import json


def compute_health_score(fundamentals: dict) -> tuple[float, dict]:
    """
    Scores financial health 0–100%.
    Returns (score_pct, breakdown) where breakdown shows each check's result.
    """
    raw = json.loads(fundamentals) if isinstance(fundamentals, str) else fundamentals

    checks = {
        "positive_eps": (raw.get("trailingEps") or 0) > 0,
        "positive_profit_margin": (raw.get("profitMargins") or 0) > 0,
        "manageable_debt": (raw.get("debtToEquity") or 0) < 200,
        "positive_roe": (raw.get("returnOnEquity") or 0) > 0,
        "current_ratio_ok": (raw.get("currentRatio") or 0) >= 1.0,
        "positive_free_cashflow": (raw.get("freeCashflow") or 0) > 0,
    }

    # Only count checks where data is available (not None in source)
    available = {
        k: v for k, v in checks.items()
        if _has_data(raw, k)
    }

    if not available:
        return 0.0, {"error": "no_data_available", "checks": checks}

    score = (sum(available.values()) / len(available)) * 100
    return round(score, 1), {
        "score_pct": round(score, 1),
        "checks_passed": sum(available.values()),
        "checks_available": len(available),
        "breakdown": {k: ("PASS" if v else "FAIL") for k, v in available.items()},
    }


def compute_growth_score(fundamentals: dict) -> tuple[float, dict]:
    """
    Scores growth profile 0–100%.
    Returns (score_pct, breakdown).
    """
    raw = json.loads(fundamentals) if isinstance(fundamentals, str) else fundamentals

    rev_growth = raw.get("revenueGrowth") or 0
    earn_growth = raw.get("earningsGrowth") or 0

    checks = {
        "revenue_growth_positive": rev_growth > 0,
        "earnings_growth_positive": earn_growth > 0,
        "revenue_growth_strong": rev_growth > 0.05,    # >5% YoY
        "earnings_growth_strong": earn_growth > 0.10,  # >10% YoY
    }

    available = {
        k: v for k, v in checks.items()
        if _growth_data_available(raw, k)
    }

    if not available:
        return 0.0, {"error": "no_growth_data", "checks": checks}

    score = (sum(available.values()) / len(available)) * 100
    return round(score, 1), {
        "score_pct": round(score, 1),
        "checks_passed": sum(available.values()),
        "checks_available": len(available),
        "revenue_growth_yoy": round(rev_growth * 100, 2),
        "earnings_growth_yoy": round(earn_growth * 100, 2),
        "breakdown": {k: ("PASS" if v else "FAIL") for k, v in available.items()},
    }


def compute_daily_liquidity(fundamentals: dict, technicals: dict) -> float:
    """
    Estimates average daily liquidity in USD from volume * price.
    Uses technicals volume_20d_avg × current_price.
    Falls back to 0 if data unavailable.
    """
    t = json.loads(technicals) if isinstance(technicals, str) else technicals
    vol = t.get("volume_20d_avg") or 0
    price = t.get("current_price") or 0
    return round(vol * price, 0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _has_data(raw: dict, check_key: str) -> bool:
    mapping = {
        "positive_eps": "trailingEps",
        "positive_profit_margin": "profitMargins",
        "manageable_debt": "debtToEquity",
        "positive_roe": "returnOnEquity",
        "current_ratio_ok": "currentRatio",
        "positive_free_cashflow": "freeCashflow",
    }
    field = mapping.get(check_key)
    return field is not None and raw.get(field) is not None


def _growth_data_available(raw: dict, check_key: str) -> bool:
    if "revenue" in check_key:
        return raw.get("revenueGrowth") is not None
    if "earnings" in check_key:
        return raw.get("earningsGrowth") is not None
    return False
