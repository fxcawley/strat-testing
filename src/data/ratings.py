"""
Analyst-rating fetcher.

Pulls consensus recommendation data (strong buy / buy / hold / sell /
strong sell) from yfinance.  This is the primary signal source for the
analyst-rating strategy.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import yfinance as yf


def fetch_recommendations(ticker: str) -> pd.DataFrame:
    """Return the historical analyst recommendations for *ticker*.

    Columns (int counts): strongBuy, buy, hold, sell, strongSell, period
    Index: date of the recommendation snapshot.
    """
    tk = yf.Ticker(ticker)
    rec = tk.recommendations
    if rec is None or rec.empty:
        raise ValueError(f"No recommendation data for {ticker}")
    return rec


def recommendation_trend(ticker: str) -> pd.DataFrame:
    """Return the recommendation trend (monthly aggregates)."""
    tk = yf.Ticker(ticker)
    trend = tk.recommendations_summary
    if trend is None or trend.empty:
        raise ValueError(f"No recommendation trend for {ticker}")
    return trend


def current_consensus(ticker: str) -> dict:
    """Return current consensus as a dict with counts and a label.

    The label is the bucket with the highest count among
    {strongBuy, buy, hold, sell, strongSell}.
    """
    rec = fetch_recommendations(ticker)
    latest = rec.iloc[-1]

    buckets = ["strongBuy", "buy", "hold", "sell", "strongSell"]
    available = [b for b in buckets if b in latest.index]
    if not available:
        raise ValueError(f"Unexpected recommendation columns for {ticker}: {list(latest.index)}")

    counts = {b: int(latest[b]) for b in available}
    consensus_label = max(counts, key=counts.get)
    return {"counts": counts, "consensus": consensus_label, "ticker": ticker}


def screen_universe(
    tickers: list[str],
    min_strong_buy: int = 5,
) -> pd.DataFrame:
    """Screen a list of tickers and return those meeting a threshold.

    Returns a DataFrame with one row per ticker that has at least
    *min_strong_buy* strong-buy ratings in its latest snapshot.
    """
    rows = []
    for t in tickers:
        try:
            info = current_consensus(t)
            if info["counts"].get("strongBuy", 0) >= min_strong_buy:
                rows.append(info)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.set_index("ticker")
    return df
