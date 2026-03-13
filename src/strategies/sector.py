"""
Sector rotation strategies -- Research Track (c).

These operate on sector ETFs rather than individual stocks.  The thesis:
cross-sectional signals may be stronger at the sector level because
sector returns are driven by macro factors (rates, oil, growth/value
rotation) that are more persistent and less noisy than single-stock moves.

Three sector strategies:
  1. SectorMomentum: classic cross-sectional momentum on sector ETFs.
  2. SectorMeanReversion: buy the most beaten-down sectors.
  3. SectorRelativeStrength: multi-timeframe relative strength with
     a trend filter (only go long sectors above their 50d SMA).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# The investable universe for all sector strategies
SECTOR_TICKERS = [
    "XLK",   # Technology
    "XLV",   # Healthcare
    "XLF",   # Financials
    "XLE",   # Energy
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLI",   # Industrials
    "XLB",   # Materials
    "XLU",   # Utilities
    "XLRE",  # Real Estate
    "XLC",   # Communications
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# 1. Sector Momentum
# ---------------------------------------------------------------------------

@dataclass
class SectorMomentum:
    """Go long the top-N sectors by trailing return.

    Skips sectors below their 50d SMA (trend filter).
    """
    lookback_days: int = 60
    trend_sma: int = 50
    top_n: int = 4
    use_trend_filter: bool = True

    def generate_signals(self, date, universe, lookback):
        scores = {}

        for ticker in universe:
            df = lookback.get(ticker)
            if df is None or len(df) < max(self.lookback_days, self.trend_sma) + 5:
                continue

            close = df["Close"]

            # Trend filter
            if self.use_trend_filter:
                sma = close.rolling(self.trend_sma).mean().iloc[-1]
                if pd.isna(sma) or close.iloc[-1] < sma:
                    continue

            ret = close.iloc[-1] / close.iloc[-self.lookback_days] - 1
            if ret > 0:
                scores[ticker] = float(ret)

        if not scores:
            return {}

        sorted_t = sorted(scores, key=scores.get, reverse=True)
        keep = set(sorted_t[:self.top_n])
        scores = {t: s for t, s in scores.items() if t in keep}

        total = sum(scores.values())
        if total == 0:
            return {}
        return {t: s / total for t, s in scores.items() if s > 1e-9}


# ---------------------------------------------------------------------------
# 2. Sector Mean Reversion
# ---------------------------------------------------------------------------

@dataclass
class SectorMeanReversion:
    """Go long the most beaten-down sectors (lowest trailing return).

    Contrarian bet that lagging sectors will catch up.
    Requires RSI < threshold (oversold confirmation).
    """
    lookback_days: int = 20
    rsi_period: int = 14
    rsi_threshold: float = 45.0
    top_n: int = 3

    def generate_signals(self, date, universe, lookback):
        scores = {}

        for ticker in universe:
            df = lookback.get(ticker)
            if df is None or len(df) < max(self.lookback_days, self.rsi_period) + 10:
                continue

            close = df["Close"]
            ret = close.iloc[-1] / close.iloc[-self.lookback_days] - 1

            rsi = _rsi(close, self.rsi_period).iloc[-1]
            if pd.isna(rsi) or rsi > self.rsi_threshold:
                continue

            # More negative return = stronger buy signal
            if ret < 0:
                scores[ticker] = float(-ret)

        if not scores:
            return {}

        sorted_t = sorted(scores, key=scores.get, reverse=True)
        keep = set(sorted_t[:self.top_n])
        scores = {t: s for t, s in scores.items() if t in keep}

        total = sum(scores.values())
        if total == 0:
            return {}
        return {t: s / total for t, s in scores.items() if s > 1e-9}


# ---------------------------------------------------------------------------
# 3. Sector Relative Strength (Multi-Timeframe)
# ---------------------------------------------------------------------------

@dataclass
class SectorRelativeStrength:
    """Multi-timeframe relative strength on sectors.

    Scores each sector by a blend of short (20d), medium (60d), and
    long (120d) trailing returns, each percentile-ranked.  Only invests
    in sectors above their 50d SMA.

    This captures sectors with persistent strength across timeframes,
    filtering out one-off spikes.
    """
    short_period: int = 20
    medium_period: int = 60
    long_period: int = 120
    trend_sma: int = 50
    weight_short: float = 0.4
    weight_medium: float = 0.35
    weight_long: float = 0.25
    top_n: int = 4

    def generate_signals(self, date, universe, lookback):
        raw: dict[str, dict[str, float]] = {}

        for ticker in universe:
            df = lookback.get(ticker)
            if df is None or len(df) < self.long_period + 10:
                continue

            close = df["Close"]

            # Trend filter
            sma = close.rolling(self.trend_sma).mean().iloc[-1]
            if pd.isna(sma) or close.iloc[-1] < sma:
                continue

            raw[ticker] = {
                "short": close.iloc[-1] / close.iloc[-self.short_period] - 1,
                "medium": close.iloc[-1] / close.iloc[-self.medium_period] - 1,
                "long": close.iloc[-1] / close.iloc[-self.long_period] - 1,
            }

        if len(raw) < 2:
            return {}

        # Percentile rank each timeframe
        tickers = list(raw.keys())
        ranked = {}
        for tf in ["short", "medium", "long"]:
            vals = sorted(tickers, key=lambda t: raw[t][tf])
            n = len(vals)
            for rank, t in enumerate(vals):
                ranked.setdefault(t, {})[tf] = rank / max(n - 1, 1)

        # Composite score
        scores = {}
        for t in tickers:
            scores[t] = (
                self.weight_short * ranked[t]["short"]
                + self.weight_medium * ranked[t]["medium"]
                + self.weight_long * ranked[t]["long"]
            )

        sorted_t = sorted(scores, key=scores.get, reverse=True)
        keep = set(sorted_t[:self.top_n])
        scores = {t: s for t, s in scores.items() if t in keep}

        total = sum(scores.values())
        if total == 0:
            return {}
        return {t: s / total for t, s in scores.items() if s > 1e-9}
