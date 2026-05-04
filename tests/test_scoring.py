"""Tests for scoring/financial_scores.py — no API calls."""

import pytest
from scoring.financial_scores import (
    compute_health_score,
    compute_growth_score,
    compute_daily_liquidity,
)


class TestHealthScore:
    def test_perfect_health_is_100(self, full_fundamentals):
        score, breakdown = compute_health_score(full_fundamentals)
        assert score == 100.0
        assert breakdown["checks_passed"] == breakdown["checks_available"]

    def test_all_failing_is_0(self, weak_fundamentals):
        score, breakdown = compute_health_score(weak_fundamentals)
        assert score == 0.0

    def test_partial_fields_partial_score(self, full_fundamentals):
        partial = {k: v for k, v in full_fundamentals.items() if k in ("trailingEps", "profitMargins")}
        score, breakdown = compute_health_score(partial)
        assert 0.0 < score <= 100.0
        assert breakdown["checks_available"] == 2

    def test_no_data_returns_zero(self):
        score, breakdown = compute_health_score({})
        assert score == 0.0
        assert "error" in breakdown

    def test_none_values_are_unavailable(self, full_fundamentals):
        nulled = {k: None for k in full_fundamentals}
        score, _ = compute_health_score(nulled)
        assert score == 0.0

    def test_json_string_input_parsed(self):
        import json
        data = {"trailingEps": 5.0, "profitMargins": 0.10}
        score, _ = compute_health_score(json.dumps(data))
        assert score > 0.0

    def test_score_in_valid_range(self, full_fundamentals):
        score, _ = compute_health_score(full_fundamentals)
        assert 0.0 <= score <= 100.0

    def test_debt_boundary_exactly_200(self):
        # debtToEquity == 200 is NOT < 200, so it should FAIL the manageable_debt check
        data = {"debtToEquity": 200.0}
        score, breakdown = compute_health_score(data)
        assert breakdown["breakdown"]["manageable_debt"] == "FAIL"

    def test_debt_under_200_passes(self):
        data = {"debtToEquity": 199.9}
        _, breakdown = compute_health_score(data)
        assert breakdown["breakdown"]["manageable_debt"] == "PASS"


class TestGrowthScore:
    def test_strong_growth_is_100(self, full_fundamentals):
        score, breakdown = compute_growth_score(full_fundamentals)
        assert score == 100.0

    def test_negative_growth_is_0(self, weak_fundamentals):
        score, _ = compute_growth_score(weak_fundamentals)
        assert score == 0.0

    def test_weak_positive_growth_is_partial(self):
        # revenue +2% and earnings +3% → passes "positive" checks but fails "strong" checks
        data = {"revenueGrowth": 0.02, "earningsGrowth": 0.03}
        score, breakdown = compute_growth_score(data)
        assert score == 50.0  # 2 of 4 checks pass

    def test_no_growth_data_returns_zero(self):
        score, breakdown = compute_growth_score({})
        assert score == 0.0
        assert "error" in breakdown

    def test_revenue_growth_threshold_at_5pct(self):
        # exactly 5% is NOT > 5%, so strong check fails
        data = {"revenueGrowth": 0.05, "earningsGrowth": 0.20}
        _, breakdown = compute_growth_score(data)
        assert breakdown["breakdown"]["revenue_growth_strong"] == "FAIL"

    def test_revenue_growth_above_5pct_passes(self):
        data = {"revenueGrowth": 0.051, "earningsGrowth": 0.20}
        _, breakdown = compute_growth_score(data)
        assert breakdown["breakdown"]["revenue_growth_strong"] == "PASS"

    def test_score_in_valid_range(self, full_fundamentals):
        score, _ = compute_growth_score(full_fundamentals)
        assert 0.0 <= score <= 100.0


class TestDailyLiquidity:
    def test_standard_calculation(self, full_fundamentals, full_technicals):
        liquidity = compute_daily_liquidity(full_fundamentals, full_technicals)
        # 60_000_000 vol * 175.0 price = 10_500_000_000
        assert liquidity == 10_500_000_000.0

    def test_missing_volume_returns_zero(self, full_fundamentals):
        technicals = {"current_price": 175.0}
        liquidity = compute_daily_liquidity(full_fundamentals, technicals)
        assert liquidity == 0.0

    def test_missing_price_returns_zero(self, full_fundamentals):
        technicals = {"volume_20d_avg": 60_000_000}
        liquidity = compute_daily_liquidity(full_fundamentals, technicals)
        assert liquidity == 0.0

    def test_empty_technicals_returns_zero(self, full_fundamentals):
        assert compute_daily_liquidity(full_fundamentals, {}) == 0.0
