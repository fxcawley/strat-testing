"""
Composite signal strategies -- Research Track (a).

These strategies combine multiple independent signals into a single
scoring function.  The thesis: individual signals are noisy, but
signals that agree (confluence) are more reliable.

Three composites:
  1. MomentumMeanRevFilter: momentum signal filtered by mean-reversion
     z-score (only buy momentum stocks that aren't overbought).
  2. PullbackSentiment: pullback-in-uptrend signal weighted by analyst
     consensus (buy dips in stocks analysts love).
  3. MultiSignalComposite: kitchen-sink scorer using momentum, mean
     reversion z-score, RSI, trend, and analyst sentiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.ratings import consensus_at_date


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# 1. Momentum + Mean Reversion Filter
# ---------------------------------------------------------------------------

@dataclass
class MomentumMeanRevFilter:
    """Go long momentum winners that aren't overbought.

    Signal: trailing 60d return (momentum), filtered out if z-score
    of price vs 20d mean is above a ceiling (overbought).  This avoids
    chasing momentum stocks that have already run too far.
    """
    momentum_days: int = 60
    zscore_lookback: int = 20
    zscore_ceiling: float = 2.0  # skip if z > this
    top_n: int = 10

    def generate_signals(self, date, universe, lookback):
        scores = {}

        for ticker in universe:
            df = lookback.get(ticker)
            if df is None or len(df) < max(self.momentum_days, self.zscore_lookback) + 5:
                continue

            close = df["Close"]

            # Momentum score
            mom = close.iloc[-1] / close.iloc[-self.momentum_days] - 1
            if mom <= 0:
                continue  # only positive momentum

            # Mean reversion filter: skip overbought
            roll_mean = close.rolling(self.zscore_lookback).mean().iloc[-1]
            roll_std = close.rolling(self.zscore_lookback).std().iloc[-1]
            if roll_std == 0 or pd.isna(roll_std):
                continue
            z = (close.iloc[-1] - roll_mean) / roll_std
            if z > self.zscore_ceiling:
                continue  # overbought, skip

            # Score: momentum, penalized by how stretched (higher z = lower score)
            scores[ticker] = float(mom * (1 - z / (self.zscore_ceiling * 2)))

        return self._normalize(scores)

    def _normalize(self, scores):
        if not scores:
            return {}
        scores = {t: max(s, 0.0) for t, s in scores.items()}
        sorted_t = sorted(scores, key=scores.get, reverse=True)
        keep = set(sorted_t[:self.top_n])
        scores = {t: s for t, s in scores.items() if t in keep}
        total = sum(scores.values())
        if total == 0:
            return {}
        return {t: s / total for t, s in scores.items() if s > 1e-9}


# ---------------------------------------------------------------------------
# 2. Pullback + Analyst Sentiment
# ---------------------------------------------------------------------------

@dataclass
class PullbackSentiment:
    """Buy dips in uptrends, weighted by analyst consensus.

    Signal: stock is above 50d SMA (uptrend) and RSI < 45 (pullback).
    Weight: pullback score * analyst sentiment score (point-in-time).
    Stocks with stronger analyst consensus get larger positions.
    """
    trend_sma: int = 50
    rsi_period: int = 14
    rsi_threshold: float = 45.0
    max_pullback_pct: float = 0.15
    consensus_cache: dict = field(default_factory=dict)
    min_analysts: int = 3
    top_n: int = 10

    def generate_signals(self, date, universe, lookback):
        scores = {}

        for ticker in universe:
            df = lookback.get(ticker)
            if df is None or len(df) < self.trend_sma + 20:
                continue

            close = df["Close"]

            # Uptrend filter
            trend_ma = close.rolling(self.trend_sma).mean().iloc[-1]
            current = close.iloc[-1]
            if current < trend_ma:
                continue

            trend_strength = (current - trend_ma) / trend_ma

            # RSI pullback
            rsi = _rsi(close, self.rsi_period).iloc[-1]
            if pd.isna(rsi) or rsi > self.rsi_threshold:
                continue

            # Pullback depth
            recent_high = close.iloc[-self.trend_sma:].max()
            pullback_depth = (recent_high - current) / recent_high
            if pullback_depth > self.max_pullback_pct or pullback_depth < 0.02:
                continue

            # Technical score
            tech_score = trend_strength * pullback_depth * (1 - rsi / 100)

            # Analyst sentiment multiplier
            sentiment_mult = 1.0
            history = self.consensus_cache.get(ticker)
            if history is not None:
                try:
                    info = consensus_at_date(history, date)
                    if info["n_analysts"] >= self.min_analysts:
                        # Score ranges from -2 (strong sell) to +2 (strong buy)
                        # Map to a multiplier: 0.2 (strong sell) to 2.0 (strong buy)
                        sentiment_mult = max(0.2, (info["score"] + 2) / 2)
                except ValueError:
                    pass  # no analyst data yet -- use neutral multiplier

            scores[ticker] = float(tech_score * sentiment_mult)

        return self._normalize(scores)

    def _normalize(self, scores):
        if not scores:
            return {}
        sorted_t = sorted(scores, key=scores.get, reverse=True)
        keep = set(sorted_t[:self.top_n])
        scores = {t: s for t, s in scores.items() if t in keep and s > 0}
        total = sum(scores.values())
        if total == 0:
            return {}
        return {t: s / total for t, s in scores.items()}


# ---------------------------------------------------------------------------
# 3. Multi-Signal Composite
# ---------------------------------------------------------------------------

@dataclass
class MultiSignalComposite:
    """Kitchen-sink composite: scores each ticker across 5 dimensions,
    then combines using equal weights.

    Dimensions:
      1. Momentum (60d return, percentile-ranked)
      2. Mean reversion (inverted z-score, percentile-ranked)
      3. RSI (inverted -- lower is more attractive)
      4. Trend (distance above 200d SMA)
      5. Analyst sentiment (consensus score)

    Each dimension is percentile-ranked across the universe, then the
    composite score is the average rank.  This handles scale differences
    and is robust to outliers.
    """
    momentum_days: int = 60
    zscore_lookback: int = 20
    rsi_period: int = 14
    trend_sma: int = 200
    consensus_cache: dict = field(default_factory=dict)
    dimension_weights: dict = field(default_factory=lambda: {
        "momentum": 0.25,
        "mean_rev": 0.15,
        "rsi": 0.15,
        "trend": 0.20,
        "sentiment": 0.25,
    })
    top_n: int = 10

    def generate_signals(self, date, universe, lookback):
        # Compute raw signals for each ticker
        raw: dict[str, dict[str, float]] = {}

        for ticker in universe:
            df = lookback.get(ticker)
            if df is None or len(df) < max(self.trend_sma, self.momentum_days) + 10:
                continue

            close = df["Close"]
            signals = {}

            # 1. Momentum
            signals["momentum"] = close.iloc[-1] / close.iloc[-self.momentum_days] - 1

            # 2. Mean reversion (inverted z-score: negative z = oversold = high score)
            roll_mean = close.rolling(self.zscore_lookback).mean().iloc[-1]
            roll_std = close.rolling(self.zscore_lookback).std().iloc[-1]
            if roll_std > 0 and pd.notna(roll_std):
                signals["mean_rev"] = -(close.iloc[-1] - roll_mean) / roll_std
            else:
                continue

            # 3. RSI (inverted: lower RSI = more attractive)
            rsi_val = _rsi(close, self.rsi_period).iloc[-1]
            if pd.isna(rsi_val):
                continue
            signals["rsi"] = 100 - rsi_val  # invert so higher = more oversold

            # 4. Trend
            sma = close.rolling(self.trend_sma).mean().iloc[-1]
            if pd.isna(sma) or sma <= 0:
                continue
            signals["trend"] = (close.iloc[-1] - sma) / sma

            # 5. Analyst sentiment
            sentiment = 0.0  # neutral default
            history = self.consensus_cache.get(ticker)
            if history is not None:
                try:
                    info = consensus_at_date(history, date)
                    if info["n_analysts"] >= 3:
                        sentiment = info["score"]
                except ValueError:
                    pass
            signals["sentiment"] = sentiment

            raw[ticker] = signals

        if len(raw) < 5:
            return {}

        # Percentile-rank each dimension across the universe
        tickers = list(raw.keys())
        dims = list(self.dimension_weights.keys())
        ranked = {dim: {} for dim in dims}

        for dim in dims:
            vals = [(t, raw[t].get(dim, 0.0)) for t in tickers]
            vals.sort(key=lambda x: x[1])
            n = len(vals)
            for rank, (t, _) in enumerate(vals):
                ranked[dim][t] = rank / max(n - 1, 1)  # 0-1 percentile

        # Composite score: weighted average of percentile ranks
        composite = {}
        for t in tickers:
            score = sum(
                self.dimension_weights[dim] * ranked[dim].get(t, 0.5)
                for dim in dims
            )
            composite[t] = score

        # Top N
        sorted_t = sorted(composite, key=composite.get, reverse=True)
        keep = set(sorted_t[:self.top_n])
        scores = {t: composite[t] for t in keep}

        total = sum(scores.values())
        if total == 0:
            return {}
        return {t: s / total for t, s in scores.items()}
