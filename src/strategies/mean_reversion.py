"""
Mean reversion strategy.

Signal: z-score of price relative to its rolling mean.  Go long the most
oversold stocks (most negative z-scores), expecting reversion to the mean.

This is a contrarian strategy -- it buys losers and sells winners.
Best suited to range-bound or mean-reverting markets; gets crushed
during strong trends.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MeanReversionStrategy:
    """Buy oversold, sell overbought based on z-score of price vs rolling mean.

    Parameters
    ----------
    lookback : int
        Rolling window for mean/std calculation.
    entry_z : float
        Z-score threshold for entry (buy when z < -entry_z).
    top_n : int
        Number of positions to hold.
    long_only : bool
        If True, only go long oversold stocks.  If False, also short
        overbought stocks.
    """
    lookback: int = 20
    entry_z: float = 1.5
    top_n: int = 10
    long_only: bool = True

    def generate_signals(
        self,
        date: pd.Timestamp,
        universe: list[str],
        lookback: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        scores: dict[str, float] = {}

        for ticker in universe:
            df = lookback.get(ticker)
            if df is None or len(df) < self.lookback + 5:
                continue

            close = df["Close"]
            rolling_mean = close.rolling(self.lookback).mean()
            rolling_std = close.rolling(self.lookback).std()

            if rolling_std.iloc[-1] == 0 or pd.isna(rolling_std.iloc[-1]):
                continue

            z = (close.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]

            # Oversold: negative z-score below threshold
            if z < -self.entry_z:
                # Score by how oversold -- more negative = stronger signal
                scores[ticker] = -float(z)
            elif not self.long_only and z > self.entry_z:
                # Overbought: short signal
                scores[ticker] = -float(z)  # negative weight for shorts

        if not scores:
            return None  # no signals -- keep existing positions

        if self.long_only:
            scores = {t: max(s, 0.0) for t, s in scores.items()}

        # Keep top N by absolute signal strength
        sorted_tickers = sorted(scores, key=lambda t: abs(scores[t]), reverse=True)
        keep = set(sorted_tickers[: self.top_n])
        scores = {t: s for t, s in scores.items() if t in keep}

        total = sum(abs(v) for v in scores.values())
        if total == 0:
            return {}
        return {t: s / total for t, s in scores.items() if abs(s) > 1e-9}
