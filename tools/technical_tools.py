"""Technical indicator tools — manual RSI/MACD/SMA implementation (pandas 2.x safe)."""

import json

import numpy as np
import pandas as pd
import yfinance as yf
from langchain_core.tools import tool


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance 0.2.x returns MultiIndex columns for single-ticker downloads. Flatten."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _compute_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return None if np.isnan(f) else round(f, 4)
    except (TypeError, ValueError):
        return None


@tool
def get_technicals(ticker: str, period: str = "1y") -> str:
    """
    Calculate technical analysis indicators using the last year of price data.
    Returns: RSI-14 (overbought >70, oversold <30), MACD and signal line,
    50-day and 200-day SMAs, Golden/Death Cross status, current price,
    and volume trend vs 20-day average.
    Use this tool to assess price momentum and entry timing.
    Works with NSE tickers (e.g. RELIANCE.NS) and US tickers (e.g. AAPL).
    """
    try:
        # Ticker.history() avoids the 401 that yf.download() hits on Yahoo Finance's new API
        t = yf.Ticker(ticker)
        df = t.history(period=period, auto_adjust=True)

        # Yahoo Finance intermittently rejects longer periods — fall back to shorter windows
        if df.empty:
            for fallback_period in ("6mo", "3mo", "1mo"):
                import time as _time
                _time.sleep(0.5)
                df = t.history(period=fallback_period, auto_adjust=True)
                if not df.empty:
                    break

        if df.empty:
            return json.dumps({"error": "No price data returned", "ticker": ticker})

        # .history() returns simple columns (not MultiIndex), but flatten just in case
        df = _flatten_columns(df)

        if len(df) < 30:
            return json.dumps({"error": f"Insufficient data: only {len(df)} rows", "ticker": ticker})

        close = df["Close"].squeeze()

        rsi = _compute_rsi(close, 14)
        macd_line, signal_line, histogram = _compute_macd(close)

        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean() if len(df) >= 200 else pd.Series([None] * len(df))

        current_price = _safe_float(close.iloc[-1])
        rsi_val = _safe_float(rsi.iloc[-1])
        macd_val = _safe_float(macd_line.iloc[-1])
        signal_val = _safe_float(signal_line.iloc[-1])
        hist_val = _safe_float(histogram.iloc[-1])
        sma50_val = _safe_float(sma_50.iloc[-1])
        sma200_val = _safe_float(sma_200.iloc[-1]) if len(df) >= 200 else None

        # Cross status
        if sma50_val and sma200_val:
            cross = "GOLDEN_CROSS" if sma50_val > sma200_val else "DEATH_CROSS"
        else:
            cross = "INSUFFICIENT_DATA"

        # RSI interpretation
        if rsi_val is None:
            rsi_signal = "UNKNOWN"
        elif rsi_val > 70:
            rsi_signal = "OVERBOUGHT"
        elif rsi_val < 30:
            rsi_signal = "OVERSOLD"
        else:
            rsi_signal = "NEUTRAL"

        # MACD interpretation
        if macd_val is not None and signal_val is not None:
            macd_signal = "BULLISH" if macd_val > signal_val else "BEARISH"
        else:
            macd_signal = "UNKNOWN"

        # Volume trend
        volume = df["Volume"].squeeze()
        vol_20d_avg = _safe_float(volume.tail(20).mean())
        vol_current = _safe_float(volume.iloc[-1])
        if vol_20d_avg and vol_current:
            vol_ratio = round(vol_current / vol_20d_avg, 2)
            vol_note = (
                "ABOVE_AVERAGE" if vol_ratio > 1.2
                else "BELOW_AVERAGE" if vol_ratio < 0.8
                else "AVERAGE"
            )
        else:
            vol_ratio, vol_note = None, "UNKNOWN"

        # Price vs SMAs
        price_vs_sma50 = (
            round((current_price - sma50_val) / sma50_val * 100, 2)
            if current_price and sma50_val else None
        )

        return json.dumps({
            "ticker": ticker,
            "current_price": current_price,
            "rsi_14": rsi_val,
            "rsi_signal": rsi_signal,
            "macd": macd_val,
            "macd_signal_line": signal_val,
            "macd_histogram": hist_val,
            "macd_interpretation": macd_signal,
            "sma_50": sma50_val,
            "sma_200": sma200_val,
            "cross_status": cross,
            "price_vs_sma50_pct": price_vs_sma50,
            "volume_20d_avg": vol_20d_avg,
            "volume_latest": vol_current,
            "volume_ratio": vol_ratio,
            "volume_note": vol_note,
            "data_points": len(df),
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})
