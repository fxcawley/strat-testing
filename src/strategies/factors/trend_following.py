"""
Trend following (time-series momentum).

Each asset is evaluated independently: if its trailing return is positive,
go long; if negative, go to cash (or short if allowed).  This is absolute
momentum -- the decision is "is this asset trending?" not "is it trending
more than others?"

Classic references:
  - Moskowitz, Ooi, Pedersen (2012) "Time Series Momentum"
  - Hurst, Ooi, Pedersen (2017) "A Century of Evidence on Trend Following"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TrendFollowing:
    """Time-series momentum: go long assets with positive trailing return.

    Parameters
    ----------
    lookback_days : int
        Trailing return period.
    vol_target : float | None
        If set, scale each position by inverse volatility to target
        roughly equal risk contribution.  Standard in trend-following.
    vol_lookback : int
        Days for volatility estimation (for vol targeting).
    long_only : bool
        If True, go to cash when signal is negative.
        If False, short when signal is negative.
    """
    lookback_days: int = 252  # 12-month
    skip_days: int = 21  # skip most recent month (standard in momentum lit)
    vol_target: float | None = 0.10  # 10% annualized vol per position
    vol_lookback: int = 60
    long_only: bool = True

    def generate_signals(self, date, universe, lookback):
        raw_weights: dict[str, float] = {}

        for ticker in universe:
            df = lookback.get(ticker)
            if df is None or len(df) < self.lookback_days + self.skip_days + 5:
                continue

            close = df["Close"]

            # Trailing return, skipping most recent month
            n = len(close)
            end_pos = n - self.skip_days if self.skip_days > 0 else n
            start_pos = end_pos - self.lookback_days
            if start_pos < 0 or end_pos < 1:
                continue

            ret = close.iloc[end_pos - 1] / close.iloc[start_pos] - 1

            # Signal: direction of trailing return
            if ret > 0:
                direction = 1.0
            elif not self.long_only:
                direction = -1.0
            else:
                continue  # skip, go to cash for this asset

            # Vol scaling
            if self.vol_target is not None:
                recent_vol = close.pct_change().iloc[-self.vol_lookback:].std() * np.sqrt(252)
                if recent_vol > 0.001:
                    size = self.vol_target / recent_vol
                else:
                    size = 1.0
                size = min(size, 3.0)  # cap leverage at 3x per position
            else:
                size = 1.0

            raw_weights[ticker] = direction * size

        if not raw_weights:
            return None

        # Normalize to sum of abs weights <= 1.0
        total = sum(abs(w) for w in raw_weights.values())
        if total > 1.0:
            raw_weights = {t: w / total for t, w in raw_weights.items()}

        return {t: w for t, w in raw_weights.items() if abs(w) > 1e-6}
