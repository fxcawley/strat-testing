"""
Cross-asset momentum.

Applies momentum across distinct asset classes (equities, bonds,
commodities, real estate).  The key insight: momentum across asset
classes is driven by different macro factors, so cross-asset momentum
captures regime-level trends that within-equity momentum misses.

Two variants:
  1. CrossAssetTimeSeries: time-series momentum per asset, vol-scaled
  2. CrossAssetRelative: rank assets by return, overweight top, underweight bottom

Reference:
  - Asness, Moskowitz, Pedersen (2013) "Value and Momentum Everywhere"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# Asset class labels for grouping
ASSET_CLASSES = {
    "SPY": "equity", "QQQ": "equity", "IWM": "equity",
    "EFA": "equity", "EEM": "equity",
    "TLT": "bond", "IEF": "bond", "SHY": "bond",
    "LQD": "bond", "HYG": "bond",
    "GLD": "commodity", "SLV": "commodity", "DBA": "commodity",
    "VNQ": "real_estate",
}


@dataclass
class CrossAssetTimeSeries:
    """Time-series momentum across asset classes with vol scaling.

    Goes long assets with positive 12-1 momentum, sized by inverse vol.
    The cross-asset diversification is the primary source of value.
    """
    lookback_days: int = 252
    skip_days: int = 21
    vol_lookback: int = 60
    vol_target: float = 0.10

    def generate_signals(self, date, universe, lookback):
        raw_weights: dict[str, float] = {}

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

            if ret <= 0:
                continue  # only go long positive trends

            # Vol scaling
            recent_vol = close.pct_change().iloc[-self.vol_lookback:].std() * np.sqrt(252)
            if recent_vol > 0.001:
                size = self.vol_target / recent_vol
            else:
                size = 1.0
            size = min(size, 3.0)

            raw_weights[ticker] = size

        if not raw_weights:
            return None

        total = sum(raw_weights.values())
        if total > 1.0:
            raw_weights = {t: w / total for t, w in raw_weights.items()}

        return {t: w for t, w in raw_weights.items() if abs(w) > 1e-6}


@dataclass
class CrossAssetRelative:
    """Cross-asset relative momentum.

    Ranks all assets by trailing return, goes long the top fraction.
    Diversifies across asset classes by capping exposure per class.

    Parameters
    ----------
    lookback_days : int
        Trailing return period.
    skip_days : int
        Skip most recent N days.
    top_frac : float
        Fraction to hold long.
    max_class_weight : float
        Max portfolio weight in any single asset class.
    """
    lookback_days: int = 252
    skip_days: int = 21
    top_frac: float = 0.35
    max_class_weight: float = 0.40

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

            scores[ticker] = float(close.iloc[end_pos - 1] / close.iloc[start_pos] - 1)

        if len(scores) < 3:
            return None

        # Rank and select top fraction
        sorted_t = sorted(scores, key=scores.get, reverse=True)
        n_hold = max(1, int(len(sorted_t) * self.top_frac))
        top = sorted_t[:n_hold]

        # Equal weight initially
        weights = {t: 1.0 / n_hold for t in top}

        # Apply asset class cap
        class_weights: dict[str, float] = {}
        for t, w in weights.items():
            ac = ASSET_CLASSES.get(t, "other")
            class_weights[ac] = class_weights.get(ac, 0.0) + w

        # Scale down over-exposed classes
        for ac, cw in class_weights.items():
            if cw > self.max_class_weight:
                scale = self.max_class_weight / cw
                for t in list(weights.keys()):
                    if ASSET_CLASSES.get(t, "other") == ac:
                        weights[t] *= scale

        # Re-normalize
        total = sum(weights.values())
        if total > 0:
            weights = {t: w / total for t, w in weights.items()}

        return {t: w for t, w in weights.items() if abs(w) > 1e-6}
