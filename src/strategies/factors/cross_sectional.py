"""
Cross-sectional momentum on ETFs.

Classic Jegadeesh-Titman (1993) momentum: rank assets by trailing return
(12 months, skip last month), go long the top quartile, underweight/short
the bottom quartile.

On ETFs rather than individual stocks: lower turnover, no single-stock
risk, more liquid, easier to short.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CrossSectionalMomentum:
    """Go long the top-ranked ETFs by trailing return, short the bottom.

    Parameters
    ----------
    lookback_days : int
        Trailing return period (252 = 12 months).
    skip_days : int
        Skip most recent N days (21 = 1 month, standard).
    top_frac : float
        Fraction of universe to go long (0.25 = top quartile).
    long_only : bool
        If True, only go long. If False, also short bottom quartile.
    equal_weight : bool
        If True, equal-weight within long/short legs.
        If False, weight by relative strength (return rank).
    """
    lookback_days: int = 252
    skip_days: int = 21
    top_frac: float = 0.30
    long_only: bool = True
    equal_weight: bool = True

    def generate_signals(self, date, universe, lookback):
        scores: dict[str, float] = {}

        for ticker in universe:
            df = lookback.get(ticker)
            if df is None or len(df) < self.lookback_days + self.skip_days + 5:
                continue

            close = df["Close"]
            n = len(close)
            end_pos = n - self.skip_days if self.skip_days > 0 else n
            start_pos = end_pos - self.lookback_days
            if start_pos < 0 or end_pos < 1:
                continue

            ret = close.iloc[end_pos - 1] / close.iloc[start_pos] - 1
            scores[ticker] = float(ret)

        if len(scores) < 3:
            return None

        # Rank by return
        sorted_tickers = sorted(scores, key=scores.get, reverse=True)
        n = len(sorted_tickers)
        n_long = max(1, int(n * self.top_frac))
        n_short = max(1, int(n * self.top_frac))

        longs = sorted_tickers[:n_long]
        shorts = sorted_tickers[-n_short:] if not self.long_only else []

        weights: dict[str, float] = {}

        if self.equal_weight:
            for t in longs:
                weights[t] = 1.0 / n_long
            for t in shorts:
                weights[t] = -1.0 / n_short
        else:
            # Weight by relative return (within each leg)
            long_scores = {t: scores[t] for t in longs if scores[t] > 0}
            long_total = sum(long_scores.values()) or 1.0
            for t in longs:
                weights[t] = max(scores[t], 0) / long_total

            if shorts:
                short_scores = {t: -scores[t] for t in shorts if scores[t] < 0}
                short_total = sum(short_scores.values()) or 1.0
                for t in shorts:
                    weights[t] = -max(-scores[t], 0) / short_total

        # Scale so long leg sums to ~0.5 if long/short, ~1.0 if long-only
        if not self.long_only and shorts:
            total_long = sum(w for w in weights.values() if w > 0)
            total_short = sum(-w for w in weights.values() if w < 0)
            if total_long > 0:
                for t in longs:
                    weights[t] = weights.get(t, 0) * 0.5 / total_long
            if total_short > 0:
                for t in shorts:
                    weights[t] = weights.get(t, 0) * 0.5 / total_short

        return {t: w for t, w in weights.items() if abs(w) > 1e-6}
