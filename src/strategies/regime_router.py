"""
Regime-based routing model.

A meta-strategy that allocates across sub-strategies based on observable
market conditions and trailing performance.  Think of it as a balanced
bagger: each sub-strategy is a specialist, and the router dynamically
weights the ensemble using only point-in-time signals.

Observable regime indicators (computed at each rebalance):
  1. Market volatility: rolling 20d realized vol of benchmark
  2. Market trend: benchmark return over trailing 50 days
  3. Market breadth: fraction of universe above their 50d SMA
  4. Dispersion: cross-sectional std of trailing 20d returns
  5. Trailing strategy performance: 63d rolling return of each sub-strategy

Routing logic:
  - Each sub-strategy has a regime affinity profile (which conditions suit it)
  - The router scores each strategy as:
      regime_match * trailing_sharpe_contribution
  - Weights are softmax-normalized for smooth allocation
  - The final signal is the weighted average of all sub-strategy signals

No future information is used.  The router can only observe regime
indicators and past performance of each sub-strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.backtest.engine import Strategy


@dataclass
class RegimeRouter:
    """Signal-driven routing model across sub-strategies.

    Parameters
    ----------
    strategies : dict[str, Strategy]
        Named sub-strategies to route across.
    vol_lookback : int
        Days for volatility calculation.
    trend_lookback : int
        Days for trend signal.
    breadth_sma : int
        SMA period for breadth calculation.
    perf_lookback : int
        Days for trailing strategy performance evaluation.
    temperature : float
        Softmax temperature for weight blending.  Lower = more concentrated
        on the best strategy.  Higher = more uniform.
    min_weight : float
        Minimum weight for any strategy (prevents total exclusion).
    """
    strategies: dict[str, object] = field(default_factory=dict)
    vol_lookback: int = 20
    trend_lookback: int = 50
    breadth_sma: int = 50
    perf_lookback: int = 63
    temperature: float = 1.0
    min_weight: float = 0.05

    # Internal state
    _virtual_equity: dict[str, list[float]] = field(default_factory=dict, repr=False)
    _last_weights: dict[str, dict[str, float]] = field(default_factory=dict, repr=False)
    _rebalance_count: int = field(default=0, repr=False)

    def __post_init__(self):
        for name in self.strategies:
            self._virtual_equity[name] = [1.0]  # normalized starting equity
            self._last_weights[name] = {}

    def _compute_regime(
        self, date: pd.Timestamp, universe: list[str],
        lookback: dict[str, pd.DataFrame], benchmark_key: str | None = None,
    ) -> dict[str, float]:
        """Compute observable regime indicators."""
        # Find the benchmark (use the ticker with the most data, or SPY if present)
        bench_key = benchmark_key
        if bench_key is None or bench_key not in lookback:
            bench_key = max(lookback, key=lambda t: len(lookback[t]))

        bench = lookback[bench_key]["Close"]

        # 1. Volatility regime (annualized)
        if len(bench) >= self.vol_lookback:
            vol = bench.pct_change().iloc[-self.vol_lookback:].std() * np.sqrt(252)
        else:
            vol = 0.15  # default moderate

        # 2. Trend (trailing return)
        if len(bench) >= self.trend_lookback:
            trend = bench.iloc[-1] / bench.iloc[-self.trend_lookback] - 1
        else:
            trend = 0.0

        # 3. Breadth (% of universe above 50d SMA)
        above_sma = 0
        total_counted = 0
        for ticker in universe:
            df = lookback.get(ticker)
            if df is None or len(df) < self.breadth_sma:
                continue
            sma = df["Close"].rolling(self.breadth_sma).mean().iloc[-1]
            if pd.notna(sma) and df["Close"].iloc[-1] > sma:
                above_sma += 1
            total_counted += 1
        breadth = above_sma / max(total_counted, 1)

        # 4. Cross-sectional dispersion
        returns_20d = []
        for ticker in universe:
            df = lookback.get(ticker)
            if df is None or len(df) < 21:
                continue
            ret = df["Close"].iloc[-1] / df["Close"].iloc[-21] - 1
            returns_20d.append(ret)
        dispersion = np.std(returns_20d) if len(returns_20d) > 5 else 0.0

        return {
            "volatility": float(vol),
            "trend": float(trend),
            "breadth": float(breadth),
            "dispersion": float(dispersion),
        }

    def _regime_affinity(self, strategy_name: str, regime: dict[str, float]) -> float:
        """Score how well a strategy fits the current regime.

        Returns a positive score.  Higher = better fit.
        These mappings encode domain knowledge about when each strategy works:
          - Mean reversion: high vol, low trend, low breadth (oversold markets)
          - Breakout: moderate vol, strong trend, high breadth (expanding markets)
          - Pullback: uptrend with moderate breadth (healthy trend with dips)
          - Swing: low-moderate vol, low trend magnitude (range-bound)
          - Momentum: strong trend, high breadth
        """
        vol = regime["volatility"]
        trend = regime["trend"]
        breadth = regime["breadth"]
        dispersion = regime["dispersion"]

        name = strategy_name.lower()

        if "mean_rev" in name or "reversion" in name:
            # Likes: high vol, weak/negative trend, low breadth
            score = (vol / 0.20) * (1 - breadth) * max(0.1, 1 - trend * 5)
        elif "breakout" in name:
            # Likes: moderate vol, strong positive trend, rising breadth
            score = max(0.1, trend * 3 + 1) * breadth * min(vol / 0.15, 1.5)
        elif "pullback" in name:
            # Likes: uptrend (moderate), moderate breadth (dips available)
            score = max(0.1, trend * 2 + 0.5) * (1 - abs(breadth - 0.6))
        elif "swing" in name:
            # Likes: low vol, weak trend (range-bound), moderate dispersion
            score = (1 / max(vol, 0.05)) * (1 - abs(trend) * 3) * 0.1
        elif "momentum" in name:
            # Likes: strong trend, high breadth, low-moderate vol
            score = max(0.1, trend * 4 + 1) * breadth
        else:
            score = 1.0  # neutral

        return max(score, 0.01)

    def _trailing_performance(self, strategy_name: str) -> float:
        """Get trailing normalized performance for a sub-strategy.

        Returns annualized Sharpe-like ratio from virtual equity tracking.
        """
        equity = self._virtual_equity.get(strategy_name, [1.0])
        if len(equity) < 3:
            return 0.0

        # Use last perf_lookback entries (or all if fewer)
        window = equity[-min(self.perf_lookback, len(equity)):]
        returns = np.diff(window) / np.array(window[:-1])

        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0

        return float(np.mean(returns) / np.std(returns))

    def _update_virtual_equity(
        self, strategy_name: str, weights: dict[str, float],
        lookback: dict[str, pd.DataFrame],
    ):
        """Track virtual performance of a sub-strategy using its last signals."""
        prev_weights = self._last_weights.get(strategy_name, {})
        if not prev_weights:
            return

        # Compute return of the virtual portfolio over the last period
        ret = 0.0
        for tkr, w in prev_weights.items():
            df = lookback.get(tkr)
            if df is None or len(df) < 2:
                continue
            period_ret = df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1
            ret += w * period_ret

        eq = self._virtual_equity[strategy_name]
        eq.append(eq[-1] * (1 + ret))

    def generate_signals(
        self,
        date: pd.Timestamp,
        universe: list[str],
        lookback: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        """Generate blended signals from all sub-strategies."""
        self._rebalance_count += 1

        # 1. Compute regime indicators
        regime = self._compute_regime(date, universe, lookback)

        # 2. Get signals from each sub-strategy
        strategy_signals: dict[str, dict[str, float]] = {}
        for name, strat in self.strategies.items():
            try:
                sig = strat.generate_signals(date, universe, lookback)
                if sig is None:
                    sig = {}
            except (ValueError, Exception):
                sig = {}
            strategy_signals[name] = sig

            # Update virtual tracking
            self._update_virtual_equity(name, sig, lookback)
            self._last_weights[name] = sig

        # 3. Compute strategy weights
        raw_scores: dict[str, float] = {}
        for name in self.strategies:
            affinity = self._regime_affinity(name, regime)
            trailing = self._trailing_performance(name)
            # Combine: regime fit * (1 + clipped trailing perf)
            raw_scores[name] = affinity * (1 + max(min(trailing, 2.0), -1.0))

        # Softmax to get strategy allocation weights
        scores_arr = np.array(list(raw_scores.values()))
        scores_arr = scores_arr / max(self.temperature, 0.01)
        exp_scores = np.exp(scores_arr - scores_arr.max())  # numerical stability
        softmax_weights = exp_scores / exp_scores.sum()

        # Apply minimum weight floor
        strat_weights = {}
        for i, name in enumerate(raw_scores):
            strat_weights[name] = max(softmax_weights[i], self.min_weight)

        # Re-normalize after floor
        total_sw = sum(strat_weights.values())
        strat_weights = {n: w / total_sw for n, w in strat_weights.items()}

        # 4. Blend sub-strategy ticker signals
        blended: dict[str, float] = {}
        for name, sw in strat_weights.items():
            for tkr, tw in strategy_signals[name].items():
                blended[tkr] = blended.get(tkr, 0.0) + sw * tw

        # Normalize so weights sum to <= 1.0
        total = sum(abs(v) for v in blended.values())
        if total > 1.0:
            blended = {t: v / total for t, v in blended.items()}

        # Remove negligible weights
        blended = {t: v for t, v in blended.items() if abs(v) > 1e-6}

        if not blended:
            return {}

        return blended
