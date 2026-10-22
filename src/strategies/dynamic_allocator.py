"""
Dynamic Portfolio Allocator.

A meta-strategy that converts each sub-strategy's forecast into a
utility score, then dynamically allocates capital across strategies
based on those scores.

Key ideas:
  - Each sub-strategy produces raw signals (ticker -> weight).
  - The allocator tracks virtual equity for each sub-strategy to
    compute trailing realized returns.
  - For each strategy, it computes a utility score using:
      (a) CRRA (constant relative risk aversion) utility of trailing returns
      (b) Trailing Sharpe ratio (risk-adjusted performance)
      (c) Signal conviction: how concentrated/confident the strategy's signals are
  - These three components are combined into a composite utility score.
  - Strategy allocation weights are set proportional to utility scores
    (softmax-normalized with temperature control).
  - The final portfolio is the utility-weighted blend of sub-strategy signals.

No look-ahead.  Virtual performance tracking uses only realized returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DynamicAllocator:
    """Utility-score-based dynamic strategy allocator.

    Parameters
    ----------
    strategies : dict[str, object]
        Named sub-strategies, each implementing generate_signals().
    risk_aversion : float
        CRRA risk aversion parameter gamma.
        gamma=1 -> log utility; gamma=2 -> moderate risk aversion;
        gamma=5 -> high risk aversion.
    perf_lookback : int
        Number of rebalance periods for trailing performance evaluation.
    temperature : float
        Softmax temperature.  Lower = more concentrated on best strategy.
    min_weight : float
        Minimum allocation to any strategy (prevents total exclusion).
    sharpe_weight : float
        Weight on trailing Sharpe in composite utility (0-1).
    conviction_weight : float
        Weight on signal conviction in composite utility (0-1).
    utility_weight : float
        Weight on CRRA utility in composite utility (0-1).
    """
    strategies: dict[str, object] = field(default_factory=dict)
    risk_aversion: float = 2.0
    perf_lookback: int = 12
    temperature: float = 0.5
    min_weight: float = 0.05
    sharpe_weight: float = 0.35
    conviction_weight: float = 0.15
    utility_weight: float = 0.50

    # Internal state
    _virtual_equity: dict[str, list[float]] = field(default_factory=dict, repr=False)
    _virtual_returns: dict[str, list[float]] = field(default_factory=dict, repr=False)
    _last_weights: dict[str, dict[str, float]] = field(default_factory=dict, repr=False)
    _last_prices: dict[str, dict[str, float]] = field(default_factory=dict, repr=False)
    _rebalance_count: int = field(default=0, repr=False)

    def __post_init__(self):
        for name in self.strategies:
            self._virtual_equity[name] = [1.0]
            self._virtual_returns[name] = []
            self._last_weights[name] = {}
            self._last_prices[name] = {}

    def _update_virtual_performance(
        self, strategy_name: str, lookback: dict[str, pd.DataFrame],
    ):
        """Track virtual return of a sub-strategy since last rebalance."""
        prev_weights = self._last_weights.get(strategy_name, {})
        prev_prices = self._last_prices.get(strategy_name, {})
        if not prev_weights or not prev_prices:
            return

        ret = 0.0
        for tkr, w in prev_weights.items():
            df = lookback.get(tkr)
            if df is None or len(df) < 1:
                continue
            current_price = float(df["Close"].iloc[-1])
            prev_price = prev_prices.get(tkr)
            if prev_price is not None and prev_price > 0:
                ret += w * (current_price / prev_price - 1)

        eq = self._virtual_equity[strategy_name]
        eq.append(eq[-1] * (1 + ret))
        self._virtual_returns[strategy_name].append(ret)

    def _crra_utility(self, returns: list[float]) -> float:
        """Compute CRRA (power) utility of a return sequence.

        U(W) = W^(1-gamma) / (1-gamma)  for gamma != 1
        U(W) = log(W)                    for gamma = 1

        We compute expected utility over the trailing return distribution.
        """
        if len(returns) < 2:
            return 0.0

        gamma = self.risk_aversion
        # Use trailing window
        window = returns[-self.perf_lookback:]

        # Convert returns to wealth levels (1+r)
        wealth = np.array([1.0 + r for r in window])
        wealth = np.maximum(wealth, 0.01)  # floor to avoid negative/zero

        if abs(gamma - 1.0) < 1e-6:
            # Log utility
            utilities = np.log(wealth)
        else:
            utilities = (wealth ** (1 - gamma) - 1) / (1 - gamma)

        return float(np.mean(utilities))

    def _trailing_sharpe(self, returns: list[float]) -> float:
        """Compute annualized Sharpe-like ratio from trailing returns."""
        window = returns[-self.perf_lookback:]
        if len(window) < 3:
            return 0.0
        arr = np.array(window)
        std = arr.std()
        if std < 1e-10:
            return 0.0
        # Annualize assuming monthly rebalancing
        return float(arr.mean() / std * np.sqrt(12))

    def _signal_conviction(self, signals: dict[str, float]) -> float:
        """Measure how concentrated/confident a strategy's signals are.

        Higher conviction = fewer positions with larger weights.
        Uses Herfindahl-Hirschman Index (HHI) of absolute weights.
        """
        if not signals:
            return 0.0
        weights = np.array([abs(w) for w in signals.values()])
        total = weights.sum()
        if total < 1e-10:
            return 0.0
        normalized = weights / total
        hhi = float(np.sum(normalized ** 2))
        # Scale: equal-weight N positions -> HHI = 1/N
        # Concentrated single position -> HHI = 1
        return hhi

    def _compute_utility_scores(
        self, strategy_signals: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Compute composite utility score for each strategy."""
        scores = {}

        for name in self.strategies:
            # CRRA utility component
            rets = self._virtual_returns.get(name, [])
            crra = self._crra_utility(rets)

            # Trailing Sharpe component
            sharpe = self._trailing_sharpe(rets)

            # Signal conviction component
            sigs = strategy_signals.get(name, {})
            conviction = self._signal_conviction(sigs)

            # Composite utility score
            # Normalize components to comparable scales
            # CRRA is typically in [-1, 0.5] range, Sharpe in [-2, 3], conviction in [0, 1]
            composite = (
                self.utility_weight * crra * 10  # scale up
                + self.sharpe_weight * sharpe
                + self.conviction_weight * conviction
            )
            scores[name] = composite

        return scores

    def generate_signals(self, date, universe, lookback):
        self._rebalance_count += 1

        # Step 1: update virtual performance for each strategy
        for name in self.strategies:
            self._update_virtual_performance(name, lookback)

        # Step 2: get signals from each sub-strategy
        strategy_signals: dict[str, dict[str, float]] = {}
        for name, strat in self.strategies.items():
            try:
                sig = strat.generate_signals(date, universe, lookback)
                if sig is None:
                    sig = {}
            except Exception:
                sig = {}
            strategy_signals[name] = sig

            # Store weights and prices for next period's virtual tracking
            self._last_weights[name] = sig
            self._last_prices[name] = {
                tkr: float(lookback[tkr]["Close"].iloc[-1])
                for tkr in sig if tkr in lookback and len(lookback[tkr]) > 0
            }

        # Step 3: compute utility scores
        utility_scores = self._compute_utility_scores(strategy_signals)

        # Step 4: convert utility scores to allocation weights via softmax
        # During warm-up (first few rebalances), use equal weights
        if self._rebalance_count <= 3 or all(
            len(self._virtual_returns.get(n, [])) < 3 for n in self.strategies
        ):
            n_strat = len(self.strategies)
            strat_weights = {n: 1.0 / n_strat for n in self.strategies}
        else:
            scores_arr = np.array(list(utility_scores.values()))
            scores_arr = scores_arr / max(self.temperature, 0.01)
            # Softmax with numerical stability
            exp_scores = np.exp(scores_arr - scores_arr.max())
            softmax_w = exp_scores / exp_scores.sum()

            strat_weights = {}
            for i, name in enumerate(utility_scores):
                strat_weights[name] = max(float(softmax_w[i]), self.min_weight)

            # Re-normalize after floor
            total_sw = sum(strat_weights.values())
            strat_weights = {n: w / total_sw for n, w in strat_weights.items()}

        # Step 5: blend sub-strategy signals using dynamic weights
        blended: dict[str, float] = {}
        for name, sw in strat_weights.items():
            for tkr, tw in strategy_signals[name].items():
                blended[tkr] = blended.get(tkr, 0.0) + sw * tw

        if not blended:
            return None

        # Cap total at 1.0
        total = sum(abs(w) for w in blended.values())
        if total > 1.0:
            blended = {t: w / total for t, w in blended.items()}

        # Remove negligible weights
        blended = {t: w for t, w in blended.items() if abs(w) > 1e-6}

        return blended if blended else None
