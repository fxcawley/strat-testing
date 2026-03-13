"""
Swing trading strategy.

Signal: stochastic oscillator crossovers combined with MACD confirmation.
Go long when %K crosses above %D from oversold territory and MACD histogram
is turning positive.  This captures short-term turning points.

Best suited for range-bound markets with clear oscillations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series,
    k_period: int = 14, d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Compute stochastic %K and %D."""
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(d_period).mean()
    return k, d


def _macd(
    close: pd.Series,
    fast: int = 12, slow: int = 26, signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD line, signal line, and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


@dataclass
class SwingStrategy:
    """Swing trade using stochastic + MACD confirmation.

    Parameters
    ----------
    k_period : int
        Stochastic %K lookback.
    d_period : int
        Stochastic %D smoothing.
    oversold : float
        %K threshold for oversold zone (buy when crossing up from below).
    overbought : float
        %K threshold for overbought zone (avoid/exit).
    macd_fast, macd_slow, macd_signal : int
        MACD parameters.
    require_macd_confirm : bool
        If True, require MACD histogram turning positive to confirm.
    top_n : int
        Max positions.
    """
    k_period: int = 14
    d_period: int = 3
    oversold: float = 25.0
    overbought: float = 75.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    require_macd_confirm: bool = True
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
            if df is None or len(df) < self.macd_slow + 20:
                continue

            close = df["Close"]
            high = df["High"]
            low = df["Low"]

            # Stochastic
            k, d = _stochastic(high, low, close, self.k_period, self.d_period)
            k_now, k_prev = k.iloc[-1], k.iloc[-2]
            d_now, d_prev = d.iloc[-1], d.iloc[-2]

            if pd.isna(k_now) or pd.isna(d_now) or pd.isna(k_prev) or pd.isna(d_prev):
                continue

            # Buy signal: %K crosses above %D from oversold territory
            crossover = (k_prev <= d_prev) and (k_now > d_now)
            from_oversold = k_prev < self.oversold or d_prev < self.oversold

            if not (crossover and from_oversold):
                continue

            # Avoid overbought
            if k_now > self.overbought:
                continue

            # MACD confirmation
            if self.require_macd_confirm:
                _, _, histogram = _macd(close, self.macd_fast, self.macd_slow, self.macd_signal)
                hist_now = histogram.iloc[-1]
                hist_prev = histogram.iloc[-2]
                if pd.isna(hist_now) or pd.isna(hist_prev):
                    continue
                # Histogram turning positive or accelerating
                if not (hist_now > hist_prev):
                    continue

            # Score by distance from oversold (deeper oversold = stronger signal)
            score = (self.oversold - min(k_prev, d_prev)) / self.oversold
            score = max(score, 0.01)  # floor
            scores[ticker] = float(score)

        if not scores:
            return None  # no swing signals -- keep existing positions

        sorted_tickers = sorted(scores, key=scores.get, reverse=True)
        keep = set(sorted_tickers[: self.top_n])
        scores = {t: s for t, s in scores.items() if t in keep}

        total = sum(scores.values())
        if total == 0:
            return {}
        return {t: s / total for t, s in scores.items() if abs(s) > 1e-9}
