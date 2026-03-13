"""
Breakout trading strategy.

Signal: price breaks above the N-day high (Donchian channel upper band),
confirmed by above-average volume.  Go long stocks that just broke out.

This is a trend-following strategy that tries to catch the start of new
moves.  Works well in trending markets; generates many false signals
in choppy markets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BreakoutStrategy:
    """Go long stocks breaking above their N-day Donchian channel high.

    Parameters
    ----------
    channel_period : int
        Lookback for Donchian channel (N-day high/low).
    volume_multiplier : float
        Require volume >= this multiple of the 20-day average to confirm.
    top_n : int
        Max positions to hold.
    min_breakout_pct : float
        Minimum % above the prior channel high to qualify.
    """
    channel_period: int = 20
    volume_multiplier: float = 1.5
    top_n: int = 10
    min_breakout_pct: float = 0.005  # 0.5%

    def generate_signals(
        self,
        date: pd.Timestamp,
        universe: list[str],
        lookback: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        scores: dict[str, float] = {}

        for ticker in universe:
            df = lookback.get(ticker)
            if df is None or len(df) < self.channel_period + 5:
                continue

            close = df["Close"]
            high = df["High"]
            volume = df["Volume"]

            # Donchian channel: N-day high (excluding today)
            channel_high = high.iloc[-(self.channel_period + 1):-1].max()
            current_close = close.iloc[-1]

            # Breakout: close above channel high by minimum threshold
            breakout_pct = (current_close - channel_high) / channel_high
            if breakout_pct < self.min_breakout_pct:
                continue

            # Volume confirmation
            avg_volume = volume.iloc[-21:-1].mean()  # 20-day avg excluding today
            current_volume = volume.iloc[-1]
            if avg_volume > 0 and current_volume < avg_volume * self.volume_multiplier:
                continue

            # Score by breakout strength (% above channel)
            scores[ticker] = float(breakout_pct)

        if not scores:
            return {}  # no breakouts today

        sorted_tickers = sorted(scores, key=scores.get, reverse=True)
        keep = set(sorted_tickers[: self.top_n])
        scores = {t: s for t, s in scores.items() if t in keep}

        total = sum(scores.values())
        if total == 0:
            return {}
        return {t: s / total for t, s in scores.items() if abs(s) > 1e-9}
