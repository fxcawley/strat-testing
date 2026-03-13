"""
Adaptive parameter strategies -- Research Track (b).

These strategies dynamically adjust their parameters based on the
current volatility regime.  The thesis: a fixed-parameter strategy
is always wrong for some market condition; adapting to the regime
should improve robustness.

The volatility regime is measured as the trailing 20d annualized
realized volatility of the benchmark.  Parameters shift along a
continuum from "tight" (low-vol regime) to "wide" (high-vol regime).

Two adaptive strategies:
  1. AdaptiveMeanReversion: widens z-score thresholds and extends
     lookback in high-vol regimes (requires deeper oversold to trigger).
  2. AdaptiveMomentum: shortens lookback and reduces position count
     in high-vol regimes (faster, more concentrated).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _regime_vol(lookback: dict[str, pd.DataFrame], benchmark_key: str | None = None,
                window: int = 20) -> float:
    """Compute annualized realized vol of the benchmark from lookback data."""
    if benchmark_key and benchmark_key in lookback:
        close = lookback[benchmark_key]["Close"]
    else:
        # Use the ticker with the most data as proxy
        best = max(lookback, key=lambda t: len(lookback[t]))
        close = lookback[best]["Close"]

    if len(close) < window + 1:
        return 0.15  # fallback to moderate

    return float(close.pct_change().iloc[-window:].std() * np.sqrt(252))


def _interpolate(vol: float, low_vol: float, high_vol: float,
                 val_at_low: float, val_at_high: float) -> float:
    """Linearly interpolate a parameter between low-vol and high-vol values."""
    frac = (vol - low_vol) / max(high_vol - low_vol, 1e-9)
    frac = max(0.0, min(1.0, frac))
    return val_at_low + frac * (val_at_high - val_at_low)


# ---------------------------------------------------------------------------
# 1. Adaptive Mean Reversion
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveMeanReversion:
    """Mean reversion with regime-adaptive parameters.

    In low vol (< 12% ann): tight params (lookback=15, z=1.0)
    In high vol (> 30% ann): wide params (lookback=40, z=2.5)
    Interpolates linearly between.
    """
    low_vol: float = 0.12
    high_vol: float = 0.30
    # Lookback range
    lookback_low: int = 15
    lookback_high: int = 40
    # Z-score threshold range
    z_low: float = 1.0
    z_high: float = 2.5
    top_n: int = 10

    def generate_signals(self, date, universe, lookback):
        vol = _regime_vol(lookback)

        # Adapt parameters
        lb = int(_interpolate(vol, self.low_vol, self.high_vol,
                              self.lookback_low, self.lookback_high))
        z_thresh = _interpolate(vol, self.low_vol, self.high_vol,
                                self.z_low, self.z_high)

        scores = {}
        for ticker in universe:
            df = lookback.get(ticker)
            if df is None or len(df) < lb + 5:
                continue

            close = df["Close"]
            roll_mean = close.rolling(lb).mean().iloc[-1]
            roll_std = close.rolling(lb).std().iloc[-1]

            if roll_std == 0 or pd.isna(roll_std):
                continue

            z = (close.iloc[-1] - roll_mean) / roll_std

            if z < -z_thresh:
                scores[ticker] = float(-z)

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
# 2. Adaptive Momentum
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveMomentum:
    """Momentum with regime-adaptive parameters.

    In low vol (< 12% ann): long lookback (120d), wider portfolio (15 names)
    In high vol (> 30% ann): short lookback (20d), concentrated (5 names)

    Intuition: in calm markets, long-term trends persist (long lookback
    captures them); in volatile markets, only very recent momentum matters,
    and you want to concentrate in the few clear winners.
    """
    low_vol: float = 0.12
    high_vol: float = 0.30
    # Lookback range
    lookback_low: int = 120
    lookback_high: int = 20
    # Top-N range
    topn_low: int = 15
    topn_high: int = 5

    def generate_signals(self, date, universe, lookback):
        vol = _regime_vol(lookback)

        lb = int(_interpolate(vol, self.low_vol, self.high_vol,
                              self.lookback_low, self.lookback_high))
        top_n = int(_interpolate(vol, self.low_vol, self.high_vol,
                                 self.topn_low, self.topn_high))

        scores = {}
        for ticker in universe:
            df = lookback.get(ticker)
            if df is None or len(df) < lb + 5:
                continue

            close = df["Close"]
            ret = close.iloc[-1] / close.iloc[-lb] - 1
            if ret > 0:
                scores[ticker] = float(ret)

        if not scores:
            return {}

        sorted_t = sorted(scores, key=scores.get, reverse=True)
        keep = set(sorted_t[:top_n])
        scores = {t: s for t, s in scores.items() if t in keep}

        total = sum(scores.values())
        if total == 0:
            return {}
        return {t: s / total for t, s in scores.items() if s > 1e-9}
