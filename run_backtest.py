"""
Run all three strategies head-to-head and produce a full comparison.

Usage:
    source .venv/Scripts/activate   # or bin/activate on Linux
    python run_backtest.py
"""

import sys
import time
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.data.prices import fetch_prices, BENCHMARKS
from src.data.ratings import build_consensus_history
from src.data.universe import SP500_CURRENT
from src.backtest.engine import run_backtest, BacktestResult
from src.backtest.metrics import (
    sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio,
    alpha_beta, rolling_beta,
)
from src.analysis.alpha import t_test_alpha, bootstrap_alpha, rolling_alpha
from src.strategies.analyst_ratings import AnalystRatingStrategy
from src.strategies.buy_and_hold import BuyAndHoldStrategy
from src.strategies.momentum import MomentumStrategy
from src.utils.display import print_summary


# ── Config ──────────────────────────────────────────────────────────────────
# Use a 100-stock random sample from the full S&P 500 to reduce survivorship
# bias vs the old hand-picked 25, while keeping runtime reasonable.
# Seed for reproducibility.
random.seed(42)
UNIVERSE = sorted(random.sample(SP500_CURRENT, 100))
START = "2020-01-01"
END = None  # through today
BENCHMARK = "SPY"
REBALANCE = "ME"  # month-end
INITIAL_CAPITAL = 100_000.0
COST_PCT = 0.001  # 10bps round-trip (slippage + spread)


# ── Pre-fetch analyst consensus histories ──────────────────────────────────
print("=" * 60)
print("  BUILDING POINT-IN-TIME ANALYST CONSENSUS HISTORIES")
print("=" * 60)

consensus_cache: dict[str, pd.DataFrame] = {}
failed_tickers = []
for i, ticker in enumerate(UNIVERSE):
    try:
        history = build_consensus_history(ticker)
        consensus_cache[ticker] = history
        n_events = len(history)
        date_range = f"{history.index[0].date()} to {history.index[-1].date()}"
        print(f"  [{i+1:3d}/{len(UNIVERSE)}] {ticker:>6s}: {n_events:4d} snapshots ({date_range})")
    except (ValueError, Exception) as e:
        failed_tickers.append(ticker)
        print(f"  [{i+1:3d}/{len(UNIVERSE)}] {ticker:>6s}: FAILED - {e}")
    # Small delay to avoid rate limiting
    if (i + 1) % 20 == 0:
        time.sleep(1)

# Remove tickers that failed from the universe
if failed_tickers:
    print(f"\n  Removed {len(failed_tickers)} tickers with no analyst data: {failed_tickers}")
    UNIVERSE = [t for t in UNIVERSE if t not in failed_tickers]

print(f"\n  Final universe: {len(UNIVERSE)} tickers")
print(f"  Tickers with consensus history: {len(consensus_cache)}")


# ── Build strategies ───────────────────────────────────────────────────────

strategies = {
    "Analyst Ratings (PIT)": AnalystRatingStrategy(
        consensus_cache=consensus_cache,
        top_n=10,
        long_only=True,
        min_analysts=3,
    ),
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
        cost_pct=COST_PCT,
    )
    results[name] = result
    print_summary(result.summary(), title=name)


# ── Detailed analysis ─────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  DETAILED COMPARATIVE ANALYSIS")
print("=" * 60)

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
        "Total Costs": s["total_costs"],
        "N Trades": s["n_trades"],
        "N Days": s["n_days"],
    })

comp = pd.DataFrame(rows).set_index("Strategy")

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
ax.set_title("Strategy Equity Curves vs Benchmark (with 10bps transaction costs)")
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

# 3. Rolling alpha
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

# Save comparison
comp.to_csv(output_dir / "comparison.csv")
print(f"Saved: {output_dir / 'comparison.csv'}")

# Save universe for reproducibility
with open(output_dir / "universe.txt", "w") as f:
    f.write(f"# Universe: {len(UNIVERSE)} tickers (random sample from SP500_CURRENT, seed=42)\n")
    f.write(f"# Failed/excluded: {failed_tickers}\n")
    for t in UNIVERSE:
        f.write(t + "\n")
print(f"Saved: {output_dir / 'universe.txt'}")

print(f"\n{'=' * 60}")
print("  ALL BACKTESTS COMPLETE")
print(f"{'=' * 60}")
