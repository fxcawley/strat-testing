"""
Run all three strategies head-to-head and produce a full comparison.

Usage:
    source .venv/Scripts/activate   # or bin/activate on Linux
    python run_backtest.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.data.prices import fetch_prices, BENCHMARKS
from src.data.ratings import current_consensus
from src.data.universe import SP500_SAMPLE
from src.backtest.engine import run_backtest, BacktestResult
from src.backtest.metrics import (
    sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio,
    alpha_beta, rolling_beta, plot_equity, plot_drawdown,
)
from src.analysis.alpha import t_test_alpha, bootstrap_alpha, rolling_alpha
from src.strategies.buy_and_hold import BuyAndHoldStrategy
from src.strategies.momentum import MomentumStrategy
from src.utils.display import print_summary


# ── Config ──────────────────────────────────────────────────────────────────
UNIVERSE = SP500_SAMPLE
START = "2020-01-01"
END = None  # through today
BENCHMARK = "SPY"
REBALANCE = "ME"  # month-end
INITIAL_CAPITAL = 100_000.0


# ── Pre-fetch analyst ratings (point-in-time snapshot) ─────────────────────
# The analyst-rating strategy needs live consensus data. Since the backtest
# engine calls generate_signals() at each rebalance, and we can only observe
# *current* ratings (not historical), we pre-fetch once and build a strategy
# that uses this fixed snapshot. This is the honest approach: we are testing
# "what if I screened today's ratings and held that portfolio since START?"

print("=" * 60)
print("  FETCHING ANALYST RATINGS FOR UNIVERSE")
print("=" * 60)

ratings_cache: dict[str, dict] = {}
for ticker in UNIVERSE:
    try:
        info = current_consensus(ticker)
        ratings_cache[ticker] = info
        print(f"  {ticker:>6s}: {info['consensus']:>12s}  {info['counts']}")
    except (ValueError, Exception) as e:
        print(f"  {ticker:>6s}: FAILED - {e}")
        raise

# Build a strategy class that uses the pre-fetched ratings
DEFAULT_SCORE_MAP = {
    "strongBuy": 2.0,
    "buy": 1.0,
    "hold": 0.0,
    "sell": -1.0,
    "strongSell": -2.0,
}


class PreloadedAnalystRatingStrategy:
    """Analyst rating strategy using pre-fetched consensus data."""

    def __init__(self, ratings: dict[str, dict], score_map: dict | None = None,
                 long_only: bool = True, top_n: int = 10):
        self._ratings = ratings
        self._score_map = score_map or dict(DEFAULT_SCORE_MAP)
        self._long_only = long_only
        self._top_n = top_n

    def generate_signals(self, date, universe, lookback):
        scores = {}
        for ticker in universe:
            info = self._ratings[ticker]
            label = info["consensus"]
            score = self._score_map.get(label)
            if score is None:
                raise ValueError(f"Consensus label '{label}' for {ticker} not in score_map")
            scores[ticker] = score

        if self._long_only:
            scores = {t: max(s, 0.0) for t, s in scores.items()}

        if self._top_n is not None:
            sorted_tickers = sorted(scores, key=scores.get, reverse=True)
            keep = set(sorted_tickers[:self._top_n])
            scores = {t: s for t, s in scores.items() if t in keep}

        total = sum(abs(v) for v in scores.values())
        if total == 0:
            raise ValueError("All scores are zero after filtering -- no tradeable signal")
        return {t: s / total for t, s in scores.items() if abs(s) > 1e-9}


# ── Run backtests ──────────────────────────────────────────────────────────

strategies = {
    "Analyst Ratings": PreloadedAnalystRatingStrategy(ratings_cache, top_n=10, long_only=True),
    "Momentum (60d)": MomentumStrategy(lookback_days=60, top_n=10, long_only=True),
    "Buy & Hold (EW)": BuyAndHoldStrategy(),
}

results: dict[str, BacktestResult] = {}

for name, strat in strategies.items():
    print(f"\n{'=' * 60}")
    print(f"  RUNNING: {name}")
    print(f"{'=' * 60}")
    result = run_backtest(
        strategy=strat,
        universe=UNIVERSE,
        start=START,
        end=END,
        benchmark=BENCHMARK,
        rebalance_freq=REBALANCE,
        initial_capital=INITIAL_CAPITAL,
    )
    results[name] = result
    print_summary(result.summary(), title=name)


# ── Detailed analysis ─────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  DETAILED COMPARATIVE ANALYSIS")
print("=" * 60)

# Build comparison table
rows = []
for name, result in results.items():
    s = result.summary()
    a, b = alpha_beta(result.returns, result.benchmark_returns)
    sr = sortino_ratio(result.returns)
    cr = calmar_ratio(result.equity_curve)

    t_res = t_test_alpha(result.excess_returns)
    boot = bootstrap_alpha(result.excess_returns, n_bootstrap=5000)

    rows.append({
        "Strategy": name,
        "Total Return": s["total_return"],
        "CAGR": s["cagr"],
        "Benchmark Return": s["benchmark_return"],
        "Annual Vol": s["annual_volatility"],
        "Sharpe": s["sharpe_ratio"],
        "Sortino": sr,
        "Max Drawdown": s["max_drawdown"],
        "Calmar": cr,
        "Alpha (ann)": a,
        "Beta": b,
        "Info Ratio": s["information_ratio"],
        "Alpha t-stat": t_res["t_statistic"],
        "Alpha p-value": t_res["p_value"],
        "Alpha significant (5%)": t_res["significant_5pct"],
        "Bootstrap Alpha Mean": boot["annualized_alpha_mean"],
        "Bootstrap CI Low": boot["ci_lower"],
        "Bootstrap CI High": boot["ci_upper"],
        "P(alpha > 0)": boot["pct_positive"],
        "N Trades": s["n_trades"],
        "N Days": s["n_days"],
    })

comp = pd.DataFrame(rows).set_index("Strategy")

# Print the table
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)
print("\n" + comp.to_string())


# ── Generate plots ─────────────────────────────────────────────────────────

output_dir = Path("data/results")
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Combined equity curves
fig, ax = plt.subplots(figsize=(14, 7))
for name, result in results.items():
    ax.plot(result.equity_curve.index, result.equity_curve.values, label=name, linewidth=1.5)
bench = list(results.values())[0]
ax.plot(bench.benchmark_curve.index, bench.benchmark_curve.values, label=f"Benchmark ({BENCHMARK})", linewidth=1.2, alpha=0.6, linestyle="--", color="gray")
ax.set_title("Strategy Equity Curves vs Benchmark")
ax.set_ylabel("Portfolio Value ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir / "equity_curves.png", dpi=150)
print(f"\nSaved: {output_dir / 'equity_curves.png'}")

# 2. Drawdown comparison
fig, axes = plt.subplots(len(results), 1, figsize=(14, 4 * len(results)), sharex=True)
for ax, (name, result) in zip(axes, results.items()):
    eq = result.equity_curve
    rm = eq.cummax()
    dd = (eq - rm) / rm
    ax.fill_between(dd.index, dd.values, 0, alpha=0.4, color="red")
    ax.set_title(f"Drawdown: {name}")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir / "drawdowns.png", dpi=150)
print(f"Saved: {output_dir / 'drawdowns.png'}")

# 3. Rolling alpha for each strategy
fig, axes = plt.subplots(len(results), 1, figsize=(14, 4 * len(results)), sharex=True)
for ax, (name, result) in zip(axes, results.items()):
    try:
        ra = rolling_alpha(result.returns, result.benchmark_returns, window=63)
        ax.plot(ra.index, ra.values, linewidth=1.0)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"Rolling Alpha (63d): {name}")
        ax.set_ylabel("Ann. Alpha")
        ax.grid(True, alpha=0.3)
    except ValueError as e:
        ax.set_title(f"Rolling Alpha: {name} -- {e}")
fig.tight_layout()
fig.savefig(output_dir / "rolling_alpha.png", dpi=150)
print(f"Saved: {output_dir / 'rolling_alpha.png'}")

# 4. Rolling beta
fig, axes = plt.subplots(len(results), 1, figsize=(14, 4 * len(results)), sharex=True)
for ax, (name, result) in zip(axes, results.items()):
    rb = rolling_beta(result.returns, result.benchmark_returns, window=63)
    ax.plot(rb.index, rb.values, linewidth=1.0)
    ax.axhline(1, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(f"Rolling Beta (63d): {name}")
    ax.set_ylabel("Beta")
    ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir / "rolling_beta.png", dpi=150)
print(f"Saved: {output_dir / 'rolling_beta.png'}")

# Save the comparison table
comp.to_csv(output_dir / "comparison.csv")
print(f"Saved: {output_dir / 'comparison.csv'}")

print(f"\n{'=' * 60}")
print("  ALL BACKTESTS COMPLETE")
print(f"{'=' * 60}")
