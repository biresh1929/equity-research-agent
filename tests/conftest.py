"""Shared pytest fixtures."""

import pytest


@pytest.fixture
def full_fundamentals():
    """All 12 health fields populated with healthy values."""
    return {
        "trailingPE": 25.0,
        "trailingEps": 6.50,
        "revenueGrowth": 0.12,
        "profitMargins": 0.24,
        "debtToEquity": 45.0,
        "returnOnEquity": 0.18,
        "currentRatio": 1.8,
        "freeCashflow": 9_000_000_000,
        "operatingMargins": 0.30,
        "earningsGrowth": 0.15,
        "marketCap": 2_000_000_000_000,
        "current_price": 175.0,
        "company_name": "Test Corp",
        "sector": "Technology",
        "currency": "USD",
    }


@pytest.fixture
def weak_fundamentals():
    """Fundamentals that fail every health and growth check."""
    return {
        "trailingPE": 25.0,
        "trailingEps": -1.0,       # negative EPS → FAIL
        "revenueGrowth": -0.05,    # negative → FAIL
        "profitMargins": -0.02,    # negative → FAIL
        "debtToEquity": 300.0,     # > 200 → FAIL
        "returnOnEquity": -0.05,   # negative → FAIL
        "currentRatio": 0.8,       # < 1.0 → FAIL
        "freeCashflow": -500_000,  # negative → FAIL
        "operatingMargins": 0.05,
        "earningsGrowth": -0.10,
        "marketCap": 1_000_000_000,
        "current_price": 15.0,
    }


@pytest.fixture
def full_technicals():
    return {
        "rsi_14": 45.0,
        "rsi_signal": "NEUTRAL",
        "macd_interpretation": "BULLISH",
        "cross_status": "ABOVE_50SMA",
        "sma_50": 170.0,
        "sma_200": 155.0,
        "volume_note": "Average volume",
        "volume_20d_avg": 60_000_000,
        "current_price": 175.0,
    }
