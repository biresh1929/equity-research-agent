"""News sentiment tool — Tavily search + keyword scoring."""

import json

from langchain_core.tools import tool

BULLISH_KEYWORDS = [
    "beats", "beat", "exceeded", "record", "growth", "upgrade", "buy",
    "outperform", "surge", "rally", "positive", "robust", "strong",
    "raised guidance", "raised forecast", "buyback", "acquisition",
    "expansion", "profit", "upside",
]

BEARISH_KEYWORDS = [
    "miss", "missed", "disappointed", "weak", "decline", "downgrade", "sell",
    "underperform", "concern", "risk", "drop", "fell", "negative",
    "lawsuit", "investigation", "layoff", "guidance cut", "warning",
    "loss", "debt", "regulation", "fine", "sanction", "recall",
]


@tool
def get_news_sentiment(ticker: str) -> str:
    """
    Search for recent news about a stock ticker and compute sentiment.
    Returns: top 5 headlines with URLs, overall sentiment (BULLISH/BEARISH/NEUTRAL),
    sentiment score [-1 to +1], and keyword signal counts.
    Use this to understand the current market narrative and any recent events
    (earnings, regulatory issues, product launches, macro exposure) that
    could affect near-term price action.
    Works for any publicly traded company.
    """
    try:
        from tavily import TavilyClient
        from config.settings import settings
        import yfinance as yf

        # Get company name for better search
        try:
            info = yf.Ticker(ticker).info
            company_name = info.get("longName", ticker)
        except Exception:
            company_name = ticker

        client = TavilyClient(api_key=settings.tavily_api_key)
        results = client.search(
            query=f"{company_name} {ticker} stock news earnings 2025 2026",
            max_results=10,
            search_depth="basic",
        )

        articles = results.get("results", [])
        headlines = []
        combined_text = ""

        for art in articles[:5]:
            headlines.append({
                "title": art.get("title", ""),
                "url": art.get("url", ""),
                "published_date": art.get("published_date", ""),
                "snippet": art.get("content", "")[:300],
            })
            combined_text += " " + art.get("title", "") + " " + art.get("content", "")

        combined_lower = combined_text.lower()
        bull_count = sum(combined_lower.count(kw) for kw in BULLISH_KEYWORDS)
        bear_count = sum(combined_lower.count(kw) for kw in BEARISH_KEYWORDS)

        total = bull_count + bear_count or 1
        score = round((bull_count - bear_count) / total, 3)

        if bull_count > bear_count * 1.5:
            sentiment = "BULLISH"
        elif bear_count > bull_count * 1.5:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"

        return json.dumps({
            "ticker": ticker,
            "company": company_name,
            "overall_sentiment": sentiment,
            "sentiment_score": score,
            "bullish_signals": bull_count,
            "bearish_signals": bear_count,
            "articles_analyzed": len(articles),
            "headlines": headlines,
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})
