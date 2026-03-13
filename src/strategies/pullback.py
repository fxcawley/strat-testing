"""
Pullback trading strategy.

Signal: buy stocks that are in an uptrend but have temporarily pulled back.
"Buy the dip" in strong trends.

Filters for uptrend (price above long-term moving average), then scores
by the depth of the pullback relative to the short-term average.
Uses RSI to confirm the stock is in an oversold-but-trending condition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI (Relative Strength Index)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


@dataclass
class PullbackStrategy:
    """Buy dips in uptrends.

    Parameters
    ----------
    trend_sma : int
        Long-term SMA to define uptrend (price must be above this).
    pullback_sma : int
        Short-term SMA; pullback is measured relative to this.
    rsi_period : int
        RSI calculation period.
    rsi_threshold : float
        RSI must be below this to count as a pullback (oversold in trend).
    max_pullback_pct : float
        Maximum pullback depth from recent high (filter out crashes).
    top_n : int
        Number of positions to hold.
    """
    trend_sma: int = 50
    pullback_sma: int = 10
    rsi_period: int = 14
    rsi_threshold: float = 45.0
    max_pullback_pct: float = 0.15  # skip if dropped more than 15%
    top_n: int = 10

    def generate_signals(
        self,
        date: pd.Timestamp,
        universe: list[str],
        lookback: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        scores: dict[str, float] = {}

        for ticker in universe:
            df = lookback.get(ticker)
            if df is None or len(df) < self.trend_sma + 20:
                continue

            close = df["Close"]

            # Uptrend filter: price above long-term SMA
            trend_ma = close.rolling(self.trend_sma).mean().iloc[-1]
            current_price = close.iloc[-1]
            if current_price < trend_ma:
                continue

            # Trend strength: how far above the long SMA (as %)
            trend_strength = (current_price - trend_ma) / trend_ma

            # RSI filter: must be in pullback zone
            rsi = _rsi(close, self.rsi_period)
            current_rsi = rsi.iloc[-1]
            if pd.isna(current_rsi) or current_rsi > self.rsi_threshold:
                continue

            # Pullback depth: distance from recent high
            recent_high = close.iloc[-self.trend_sma:].max()
            pullback_depth = (recent_high - current_price) / recent_high
            if pullback_depth > self.max_pullback_pct:
                continue  # too deep -- might be a trend reversal, not a pullback
            if pullback_depth < 0.02:
                continue  # hasn't actually pulled back

            # Score: combine trend strength with pullback depth
            # Higher trend strength + deeper (but not too deep) pullback = better
            score = trend_strength * pullback_depth * (1 - current_rsi / 100)
            scores[ticker] = float(score)

        if not scores:
            return None  # no pullback signals -- keep existing positions

        sorted_tickers = sorted(scores, key=scores.get, reverse=True)
        keep = set(sorted_tickers[: self.top_n])
        scores = {t: s for t, s in scores.items() if t in keep}

        total = sum(scores.values())
        if total == 0:
            return {}
        return {t: s / total for t, s in scores.items() if abs(s) > 1e-9}
