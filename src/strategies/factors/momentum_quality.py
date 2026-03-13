"""
Momentum + Quality (AQR-style).

Combines cross-sectional momentum with a quality filter based on
observable price-based metrics (since we don't have fundamentals
from yfinance for ETFs).

Quality proxy for ETFs:
  - Low realized volatility (stable assets are "higher quality")
  - High risk-adjusted return (Sharpe over trailing period)
  - Low max drawdown (resilience)

This approximates AQR's QMJ (Quality Minus Junk) using price data.
The idea: buy assets with strong momentum AND high quality scores,
avoiding high-momentum-but-fragile assets.

Reference:
  - Asness, Frazzini, Pedersen (2019) "Quality Minus Junk"
  - AQR: momentum + quality reduces crash risk of pure momentum
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MomentumQuality:
    """Momentum weighted by quality (low-vol, high Sharpe, low drawdown).

    Parameters
    ----------
    mom_lookback : int
        Momentum lookback (days).
    mom_skip : int
        Skip most recent N days.
    quality_lookback : int
        Period for quality metrics.
    mom_weight : float
        Weight on momentum score (0-1).
    quality_weight : float
        Weight on quality score (0-1). Must sum to 1 with mom_weight.
    top_frac : float
        Fraction of universe to hold.
    """
    mom_lookback: int = 252
    mom_skip: int = 21
    quality_lookback: int = 252
    mom_weight: float = 0.5
    quality_weight: float = 0.5
    top_frac: float = 0.30

    def generate_signals(self, date, universe, lookback):
        raw: dict[str, dict[str, float]] = {}

        for ticker in universe:
            df = lookback.get(ticker)
            min_data = max(self.mom_lookback + self.mom_skip, self.quality_lookback) + 10
            if df is None or len(df) < min_data:
                continue

            close = df["Close"]
            returns = close.pct_change()

            # Momentum score (12-1)
            n = len(close)
            end_pos = n - self.mom_skip if self.mom_skip > 0 else n
            start_pos = end_pos - self.mom_lookback
            if start_pos < 0 or end_pos < 1:
                continue
            mom = close.iloc[end_pos - 1] / close.iloc[start_pos] - 1

            # Quality metrics (over quality_lookback)
            recent_returns = returns.iloc[-self.quality_lookback:]
            recent_close = close.iloc[-self.quality_lookback:]

            # 1. Inverse volatility (lower vol = higher quality)
            vol = recent_returns.std() * np.sqrt(252)
            inv_vol = 1.0 / max(vol, 0.01)

            # 2. Sharpe ratio (risk-adjusted return)
            ann_ret = recent_returns.mean() * 252
            sharpe = ann_ret / max(vol, 0.01)

            # 3. Inverse max drawdown (lower DD = higher quality)
            rolling_max = recent_close.cummax()
            dd = ((recent_close - rolling_max) / rolling_max).min()
            inv_dd = 1.0 / max(-dd, 0.01)

            raw[ticker] = {
                "momentum": mom,
                "inv_vol": inv_vol,
                "sharpe": sharpe,
                "inv_dd": inv_dd,
            }

        if len(raw) < 3:
            return None

        # Percentile-rank each metric
        tickers = list(raw.keys())
        n = len(tickers)
        ranked = {}

        for metric in ["momentum", "inv_vol", "sharpe", "inv_dd"]:
            sorted_by = sorted(tickers, key=lambda t: raw[t][metric])
            for rank, t in enumerate(sorted_by):
                ranked.setdefault(t, {})[metric] = rank / max(n - 1, 1)

        # Composite: momentum rank * mom_weight + quality rank * quality_weight
        # Quality = average of inv_vol, sharpe, inv_dd ranks
        composite = {}
        for t in tickers:
            mom_rank = ranked[t]["momentum"]
            quality_rank = (ranked[t]["inv_vol"] + ranked[t]["sharpe"] + ranked[t]["inv_dd"]) / 3
            composite[t] = self.mom_weight * mom_rank + self.quality_weight * quality_rank

        # Top fraction
        n_hold = max(1, int(n * self.top_frac))
        sorted_t = sorted(composite, key=composite.get, reverse=True)
        top = sorted_t[:n_hold]

        weights = {t: composite[t] for t in top}
        total = sum(weights.values())
        if total == 0:
            return None
        return {t: w / total for t, w in weights.items()}
