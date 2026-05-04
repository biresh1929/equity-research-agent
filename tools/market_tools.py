"""Fundamental financials and analyst ratings tools — yfinance with Tavily fallback."""

import json
import time
from functools import lru_cache

import yfinance as yf
from langchain_core.tools import tool

CRITICAL_FIELDS = [
    "trailingPE", "revenueGrowth", "profitMargins",
    "debtToEquity", "returnOnEquity",
]


def _safe_div(a, b):
    try:
        return a / b if b and b != 0 else None
    except Exception:
        return None


def _fetch_from_statements(t: yf.Ticker, current_price=None, shares_override=None) -> dict:
    """
    Compute fundamental metrics from income statement + balance sheet + cash flow.
    Used as fallback when yf.Ticker.info returns 401.
    """
    result = {}
    try:
        inc = t.income_stmt          # columns = annual dates, index = line items
        if inc is not None and not inc.empty:
            # Use most recent two annual columns for YoY growth
            cols = list(inc.columns)
            rev_key = next((k for k in inc.index if "Total Revenue" in str(k)), None)
            ni_key = next((k for k in inc.index if "Net Income" in str(k) and "Common" not in str(k)), None)
            op_key = next((k for k in inc.index if "Operating Income" in str(k)), None)
            gross_key = next((k for k in inc.index if "Gross Profit" in str(k)), None)

            if rev_key and len(cols) >= 1:
                rev_recent = inc.loc[rev_key, cols[0]]
                result["profitMargins"] = _safe_div(
                    inc.loc[ni_key, cols[0]] if ni_key else None, rev_recent
                )
                result["operatingMargins"] = _safe_div(
                    inc.loc[op_key, cols[0]] if op_key else None, rev_recent
                )
                result["grossMargins"] = _safe_div(
                    inc.loc[gross_key, cols[0]] if gross_key else None, rev_recent
                )
                if len(cols) >= 2:
                    rev_prev = inc.loc[rev_key, cols[1]]
                    result["revenueGrowth"] = _safe_div(rev_recent - rev_prev, abs(rev_prev) if rev_prev else None)
                if ni_key and len(cols) >= 2:
                    ni_recent = inc.loc[ni_key, cols[0]]
                    ni_prev = inc.loc[ni_key, cols[1]]
                    result["earningsGrowth"] = _safe_div(ni_recent - ni_prev, abs(ni_prev) if ni_prev else None)
    except Exception:
        pass

    try:
        bs = t.balance_sheet
        if bs is not None and not bs.empty:
            cols = list(bs.columns)
            eq_key = next((k for k in bs.index if "Stockholders Equity" in str(k) or "Total Equity" in str(k)), None)
            debt_key = next((k for k in bs.index if "Total Debt" in str(k) or "Long Term Debt" in str(k)), None)
            curr_a_key = next((k for k in bs.index if "Current Assets" in str(k) and "Total" in str(k)), None)
            curr_l_key = next((k for k in bs.index if "Current Liabilities" in str(k) and "Total" in str(k)), None)

            eq = bs.loc[eq_key, cols[0]] if eq_key else None
            debt = bs.loc[debt_key, cols[0]] if debt_key else None
            curr_a = bs.loc[curr_a_key, cols[0]] if curr_a_key else None
            curr_l = bs.loc[curr_l_key, cols[0]] if curr_l_key else None

            result["debtToEquity"] = _safe_div(debt, eq / 100 if eq else None)  # yfinance uses % scale
            result["currentRatio"] = _safe_div(curr_a, curr_l)

            # ROE from net income / equity
            try:
                ni = t.income_stmt
                if ni is not None and not ni.empty:
                    ni_key = next((k for k in ni.index if "Net Income" in str(k) and "Common" not in str(k)), None)
                    if ni_key and eq:
                        result["returnOnEquity"] = _safe_div(ni.loc[ni_key, list(ni.columns)[0]], eq)
            except Exception:
                pass
    except Exception:
        pass

    try:
        cf = t.cash_flow
        if cf is not None and not cf.empty:
            cols = list(cf.columns)
            ocf_key = next((k for k in cf.index if "Operating Cash Flow" in str(k) or "Cash From Operations" in str(k)), None)
            capex_key = next((k for k in cf.index if "Capital Expenditure" in str(k) or "Purchase Of PP" in str(k)), None)
            if ocf_key:
                ocf = cf.loc[ocf_key, cols[0]]
                capex = cf.loc[capex_key, cols[0]] if capex_key else 0
                result["freeCashflow"] = float(ocf + capex) if ocf is not None else None
    except Exception:
        pass

    # Trailing EPS from net income / shares
    try:
        shares = shares_override
        if not shares:
            fi = t.fast_info
            try:
                shares = fi.shares
            except Exception:
                shares = None
        ni = t.income_stmt
        if ni is not None and not ni.empty and shares:
            ni_key = next((k for k in ni.index if "Net Income" in str(k) and "Common" not in str(k)), None)
            if ni_key:
                result["trailingEps"] = _safe_div(float(ni.loc[ni_key, list(ni.columns)[0]]), shares)
        if current_price and result.get("trailingEps") and result["trailingEps"] > 0:
            result["trailingPE"] = _safe_div(current_price, result["trailingEps"])
    except Exception:
        pass

    return result


def _fetch_price_history_fallback(t: yf.Ticker) -> float | None:
    """Try progressively shorter periods to get the latest close price."""
    import time as _time
    for period in ("5d", "1mo"):
        try:
            _time.sleep(0.3)
            df = t.history(period=period, auto_adjust=True)
            if not df.empty:
                return float(df["Close"].iloc[-1])
        except Exception:
            pass
    return None


def _fetch_yfinance(ticker: str) -> dict:
    t = yf.Ticker(ticker)

    # .info can 401 on Yahoo Finance's newer API — catch and fall through to fast_info
    info: dict = {}
    try:
        info = t.info or {}
    except Exception:
        pass

    # fast_info is a lightweight endpoint unaffected by the Yahoo Finance .info 401
    fi = None
    try:
        fi = t.fast_info
    except Exception:
        pass

    def _fi(key):
        try:
            return getattr(fi, key, None) if fi else None
        except Exception:
            return None

    current_price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or _fi("last_price")
        or _fetch_price_history_fallback(t)
    )

    # When .info fails, compute key ratios from financial statements
    # These endpoints use different Yahoo Finance routes (usually unaffected by 401)
    stmt = _fetch_from_statements(
        t, current_price, shares_override=info.get("sharesOutstanding")
    )

    def _best(*keys_and_computed):
        """Return first non-None value from info, then stmt fallback."""
        for k in keys_and_computed:
            v = info.get(k) if isinstance(k, str) else k
            if v is not None:
                return v
        return None

    return {
        "company_name": info.get("longName") or info.get("shortName", ticker),
        "sector": info.get("sector", "Unknown"),
        "industry": info.get("industry", "Unknown"),
        "market_cap": (
            info.get("marketCap")
            or _fi("market_cap")
            or (
                current_price * info["sharesOutstanding"]
                if current_price and info.get("sharesOutstanding")
                else None
            )
        ),
        "current_price": current_price,
        "currency": info.get("currency", "USD"),
        "trailingPE": _best("trailingPE", _fi("pe_trailing"), stmt.get("trailingPE")),
        "forwardPE": _best("forwardPE", _fi("pe_forward")),
        "trailingEps": _best("trailingEps", stmt.get("trailingEps")),
        "revenueGrowth": _best("revenueGrowth", stmt.get("revenueGrowth")),
        "earningsGrowth": _best("earningsGrowth", stmt.get("earningsGrowth")),
        "grossMargins": _best("grossMargins", stmt.get("grossMargins")),
        "operatingMargins": _best("operatingMargins", stmt.get("operatingMargins")),
        "profitMargins": _best("profitMargins", stmt.get("profitMargins")),
        "debtToEquity": _best("debtToEquity", stmt.get("debtToEquity")),
        "currentRatio": _best("currentRatio", stmt.get("currentRatio")),
        "returnOnEquity": _best("returnOnEquity", stmt.get("returnOnEquity")),
        "returnOnAssets": _best("returnOnAssets", stmt.get("returnOnAssets")),
        "freeCashflow": _best("freeCashflow", stmt.get("freeCashflow")),
        "enterpriseValue": info.get("enterpriseValue") or _fi("enterprise_value"),
        "priceToBook": info.get("priceToBook"),
        "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh") or _fi("year_high"),
        "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow") or _fi("year_low"),
        "dividendYield": info.get("dividendYield"),
        "numberOfAnalystOpinions": info.get("numberOfAnalystOpinions"),
        "targetMeanPrice": info.get("targetMeanPrice"),
        "targetHighPrice": info.get("targetHighPrice"),
        "targetLowPrice": info.get("targetLowPrice"),
        "recommendationKey": info.get("recommendationKey"),
        "beta": info.get("beta"),
    }


def _missing_critical_fields(data: dict) -> bool:
    present = sum(1 for f in CRITICAL_FIELDS if data.get(f) is not None)
    return present < 3


def _fetch_via_tavily(ticker: str, existing: dict) -> dict:
    """Web search fallback for missing critical fields."""
    try:
        from tavily import TavilyClient
        from config.settings import settings

        company = existing.get("company_name", ticker)
        client = TavilyClient(api_key=settings.tavily_api_key)
        query = f"{company} {ticker} P/E ratio revenue growth profit margin 2024 2025"
        results = client.search(query=query, max_results=3, search_depth="basic")

        snippets = " ".join(
            r.get("content", "")[:500] for r in results.get("results", [])
        )
        return {"web_search_context": snippets[:1000]}
    except Exception:
        return {}


def _merge_keeping_best(primary: dict, secondary: dict) -> dict:
    merged = {**primary}
    for k, v in secondary.items():
        if merged.get(k) is None and v is not None:
            merged[k] = v
    return merged


@tool
def get_fundamentals(ticker: str) -> str:
    """
    Fetch key fundamental financial metrics for a stock ticker.
    Returns P/E, forward P/E, EPS, revenue growth, earnings growth, margins,
    debt-to-equity, ROE, free cash flow, market cap, 52-week range.
    Falls back to Tavily web search if yfinance returns fewer than 3 critical fields.
    Use this tool first to assess business quality before price-based analysis.
    Works with NSE tickers (e.g. RELIANCE.NS), NYSE/NASDAQ (e.g. AAPL), and most global exchanges.
    """
    result = _fetch_yfinance(ticker)

    if _missing_critical_fields(result):
        web_data = _fetch_via_tavily(ticker, result)
        result = _merge_keeping_best(result, web_data)

    # Flag automatic concerns
    concerns = []
    if (result.get("debtToEquity") or 0) > 200:
        concerns.append("HIGH_LEVERAGE: debt/equity > 200")
    if (result.get("profitMargins") or 0) < 0:
        concerns.append("UNPROFITABLE: negative profit margin")
    if (result.get("revenueGrowth") or 0) < -0.05:
        concerns.append("DECLINING_REVENUE: YoY revenue falling > 5%")
    if result.get("currentRatio") and result["currentRatio"] < 1.0:
        concerns.append("LIQUIDITY_RISK: current ratio < 1.0")

    result["auto_flags"] = concerns
    return json.dumps(result, indent=2, default=str)


@tool
def get_analyst_ratings(ticker: str) -> str:
    """
    Fetch analyst consensus ratings and price targets for a stock.
    Returns: consensus recommendation (buy/hold/sell), number of analysts,
    mean/high/low price targets, implied upside, and rating distribution breakdown.
    Use this to understand professional analyst sentiment and valuation consensus.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info
        current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        target_mean = info.get("targetMeanPrice")

        implied_upside = None
        if target_mean and current_price:
            implied_upside = round((target_mean - current_price) / current_price * 100, 2)

        # Recent recommendations breakdown
        rec_breakdown = {}
        try:
            recs = t.recommendations
            if recs is not None and not recs.empty:
                latest = recs.tail(1)
                for col in ["strongBuy", "buy", "hold", "sell", "strongSell"]:
                    if col in latest.columns:
                        rec_breakdown[col] = int(latest[col].values[0])
        except Exception:
            pass

        return json.dumps({
            "ticker": ticker,
            "current_price": current_price,
            "consensus": info.get("recommendationKey", "N/A"),
            "consensus_mean_score": info.get("recommendationMean"),
            "num_analysts": info.get("numberOfAnalystOpinions", 0),
            "target_mean": target_mean,
            "target_high": info.get("targetHighPrice"),
            "target_low": info.get("targetLowPrice"),
            "implied_upside_pct": implied_upside,
            "rating_breakdown": rec_breakdown,
        }, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})
