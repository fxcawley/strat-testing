"""
Core backtesting engine.

Design principles:
  - Strategies emit a Signal (target weights per ticker) each rebalance day.
  - The engine walks forward through time, applies those weights, and tracks
    a synthetic portfolio equity curve.
  - Benchmark comparison (e.g. SPY buy-and-hold) is built in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from src.data.prices import fetch_prices, fetch_benchmark


# ---------------------------------------------------------------------------
# Signal / Strategy interface
# ---------------------------------------------------------------------------

class Strategy(Protocol):
    """Any object that can produce portfolio weights on a given date."""

    def generate_signals(
        self, date: pd.Timestamp, universe: list[str], lookback: dict[str, pd.DataFrame]
    ) -> dict[str, float]:
        """Return target portfolio weights keyed by ticker.

        Weights should sum to <= 1.0 (remainder is cash).
        Negative weights represent short positions.
        """
        ...


# ---------------------------------------------------------------------------
# Portfolio bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class PortfolioSnapshot:
    date: pd.Timestamp
    weights: dict[str, float]
    equity: float
    cash_weight: float


@dataclass
class BacktestResult:
    """Container for everything a backtest produces."""
    equity_curve: pd.Series  # indexed by date
    benchmark_curve: pd.Series
    trades: list[dict]
    snapshots: list[PortfolioSnapshot]
    metadata: dict = field(default_factory=dict)

    # --- Derived metrics (computed lazily) ----------------------------------

    @property
    def returns(self) -> pd.Series:
        return self.equity_curve.pct_change().dropna()

    @property
    def benchmark_returns(self) -> pd.Series:
        return self.benchmark_curve.pct_change().dropna()

    @property
    def excess_returns(self) -> pd.Series:
        r = self.returns
        b = self.benchmark_returns.reindex(r.index)
        return (r - b).dropna()

    def summary(self) -> dict:
        """Compute standard performance metrics."""
        r = self.returns
        er = self.excess_returns
        n_years = len(r) / 252

        total_ret = self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1
        bench_ret = self.benchmark_curve.iloc[-1] / self.benchmark_curve.iloc[0] - 1
        cagr = (1 + total_ret) ** (1 / max(n_years, 1e-9)) - 1

        ann_vol = r.std() * np.sqrt(252)
        sharpe = (r.mean() * 252) / ann_vol if ann_vol > 0 else 0.0

        rolling_max = self.equity_curve.cummax()
        drawdowns = (self.equity_curve - rolling_max) / rolling_max
        max_dd = drawdowns.min()

        ann_alpha = er.mean() * 252
        ir = ann_alpha / (er.std() * np.sqrt(252)) if er.std() > 0 else 0.0

        return {
            "total_return": total_ret,
            "benchmark_return": bench_ret,
            "cagr": cagr,
            "annual_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "annualized_alpha": ann_alpha,
            "information_ratio": ir,
            "n_trades": len(self.trades),
            "n_days": len(r),
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def run_backtest(
    strategy: Strategy,
    universe: list[str],
    start: str = "2018-01-01",
    end: str | None = None,
    benchmark: str = "SPY",
    rebalance_freq: str = "ME",  # ME = month-end, W = weekly, D = daily
    initial_capital: float = 100_000.0,
    lookback_buffer_days: int = 252,
) -> BacktestResult:
    """Run a walk-forward backtest.

    Parameters
    ----------
    strategy : Strategy
        An object implementing ``generate_signals``.
    universe : list[str]
        Tickers the strategy can trade.
    start, end : str
        Date range.
    benchmark : str
        Ticker used as the passive benchmark.
    rebalance_freq : str
        Pandas offset alias for rebalance cadence.
    initial_capital : float
        Starting cash.
    lookback_buffer_days : int
        Extra calendar days of price history to fetch before *start*
        so strategies have lookback data on the first rebalance.
    """
    # 1. Fetch all price data up front -- every ticker must succeed
    # Fetch extra history so strategies have lookback on day one
    from datetime import datetime, timedelta
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    fetch_start = (start_dt - timedelta(days=lookback_buffer_days)).strftime("%Y-%m-%d")

    price_data: dict[str, pd.DataFrame] = {}
    for t in set(universe + [benchmark]):
        price_data[t] = fetch_prices(t, start=fetch_start, end=end)

    bench_close = price_data[benchmark]["Close"]

    # 2. Build rebalance schedule -- only trade from `start` onward
    #    (earlier data is available for lookback but not for trading)
    all_dates = bench_close.loc[start:].index.sort_values()
    rebal_dates = all_dates.to_series().groupby(pd.Grouper(freq=rebalance_freq)).last().dropna().values
    rebal_set = set(pd.DatetimeIndex(rebal_dates))

    # 3. Walk forward
    equity = initial_capital
    current_weights: dict[str, float] = {}
    prev_prices: dict[str, float] = {}

    equity_series: dict[pd.Timestamp, float] = {}
    trades: list[dict] = []
    snapshots: list[PortfolioSnapshot] = []

    for date in all_dates:
        # --- mark-to-market existing positions ---
        if current_weights:
            pnl = 0.0
            for tkr, w in current_weights.items():
                if tkr not in price_data:
                    raise RuntimeError(
                        f"Ticker {tkr} in portfolio weights but missing from price data"
                    )
                if date not in price_data[tkr].index:
                    continue  # legitimate: non-trading day for this ticker
                p_now = float(price_data[tkr].loc[date, "Close"])
                p_prev = prev_prices.get(tkr, p_now)
                if p_prev > 0:
                    pnl += w * equity * (p_now / p_prev - 1)
                prev_prices[tkr] = p_now
            equity += pnl

        equity_series[date] = equity

        # --- rebalance ---
        if date in rebal_set:
            lookback = {t: df.loc[:date] for t, df in price_data.items() if t in universe}
            new_weights = strategy.generate_signals(date, universe, lookback)

            # Record trades
            for tkr, nw in new_weights.items():
                ow = current_weights.get(tkr, 0.0)
                if abs(nw - ow) > 1e-6:
                    trades.append({"date": date, "ticker": tkr, "old_weight": ow, "new_weight": nw})

            current_weights = new_weights
            cash_w = 1.0 - sum(current_weights.values())

            # Refresh prev_prices to current
            for tkr in current_weights:
                if tkr in price_data and date in price_data[tkr].index:
                    prev_prices[tkr] = float(price_data[tkr].loc[date, "Close"])

            snapshots.append(PortfolioSnapshot(date=date, weights=dict(current_weights), equity=equity, cash_weight=cash_w))

    # 4. Build result
    eq_curve = pd.Series(equity_series, name="equity").sort_index()
    bench_curve = (bench_close.reindex(eq_curve.index).ffill() / bench_close.reindex(eq_curve.index).ffill().iloc[0] * initial_capital)
    bench_curve.name = "benchmark"

    return BacktestResult(
        equity_curve=eq_curve,
        benchmark_curve=bench_curve,
        trades=trades,
        snapshots=snapshots,
        metadata={"strategy": type(strategy).__name__, "universe_size": len(universe), "benchmark": benchmark, "rebalance_freq": rebalance_freq},
    )
