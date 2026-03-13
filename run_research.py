"""
Research runner: backtest mean reversion, breakout, pullback, swing, and
the regime-routing ensemble.

All strategies use weekly rebalancing and 10bps transaction costs.

Usage:
    source .venv/Scripts/activate
    python run_research.py
"""

import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.data.universe import SP500_CURRENT
from src.backtest.engine import run_backtest, BacktestResult
from src.backtest.metrics import (
    sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio,
    alpha_beta, rolling_beta,
)
from src.analysis.alpha import t_test_alpha, bootstrap_alpha, rolling_alpha
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.breakout import BreakoutStrategy
from src.strategies.pullback import PullbackStrategy
from src.strategies.swing import SwingStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.regime_router import RegimeRouter
from src.utils.display import print_summary


# ── Config ──────────────────────────────────────────────────────────────────
random.seed(42)
UNIVERSE_RAW = sorted(random.sample(SP500_CURRENT, 100))
START = "2020-01-01"
END = None
BENCHMARK = "SPY"
REBALANCE = "W-FRI"  # weekly on Fridays
INITIAL_CAPITAL = 100_000.0
COST_PCT = 0.001  # 10bps

# Pre-filter universe: remove tickers that can't be fetched (delisted/acquired)
from src.data.prices import fetch_prices
print("Validating universe tickers...")
UNIVERSE = []
for t in UNIVERSE_RAW:
    try:
        fetch_prices(t, start="2024-01-01")
        UNIVERSE.append(t)
    except (ValueError, Exception) as e:
        print(f"  Dropping {t}: {e}")
print(f"  Universe: {len(UNIVERSE)} tickers (dropped {len(UNIVERSE_RAW) - len(UNIVERSE)})")

# ── Build strategies ───────────────────────────────────────────────────────

mean_rev = MeanReversionStrategy(lookback=20, entry_z=1.5, top_n=10)
breakout = BreakoutStrategy(channel_period=20, volume_multiplier=1.5, top_n=10)
pullback = PullbackStrategy(trend_sma=50, rsi_threshold=45, top_n=10)
swing = SwingStrategy(k_period=14, oversold=25, top_n=10)
momentum = MomentumStrategy(lookback_days=60, top_n=10)

# The regime router: an ensemble of all sub-strategies
router = RegimeRouter(
    strategies={
        "mean_reversion": mean_rev,
        "breakout": breakout,
        "pullback": pullback,
        "swing": swing,
        "momentum": momentum,
    },
    temperature=1.0,
    min_weight=0.05,
)

all_strategies = {
    "Mean Reversion": mean_rev,
    "Breakout": breakout,
    "Pullback": pullback,
    "Swing": swing,
    "Momentum (60d)": momentum,
    "Regime Router": router,
}

# ── Run backtests ──────────────────────────────────────────────────────────

results: dict[str, BacktestResult] = {}

for name, strat in all_strategies.items():
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
        "Alpha sig (5%)": t_res["significant_5pct"],
        "Boot Alpha": boot["annualized_alpha_mean"],
        "Boot CI Low": boot["ci_lower"],
        "Boot CI High": boot["ci_upper"],
        "P(alpha>0)": boot["pct_positive"],
        "Total Costs": s["total_costs"],
        "N Trades": s["n_trades"],
        "N Days": s["n_days"],
    })

comp = pd.DataFrame(rows).set_index("Strategy")
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 220)
print("\n" + comp.to_string())


# ── Plots ─────────────────────────────────────────────────────────────────

output_dir = Path("data/results/research")
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Equity curves
fig, ax = plt.subplots(figsize=(16, 8))
for name, result in results.items():
    lw = 2.0 if name == "Regime Router" else 1.2
    ls = "-" if name == "Regime Router" else "-"
    ax.plot(result.equity_curve.index, result.equity_curve.values,
            label=name, linewidth=lw, linestyle=ls)
bench = list(results.values())[0]
ax.plot(bench.benchmark_curve.index, bench.benchmark_curve.values,
        label=f"Benchmark ({BENCHMARK})", linewidth=1.0, alpha=0.5, linestyle="--", color="gray")
ax.set_title("Research Strategies: Equity Curves (weekly rebalance, 10bps costs)")
ax.set_ylabel("Portfolio Value ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir / "equity_curves.png", dpi=150)
print(f"\nSaved: {output_dir / 'equity_curves.png'}")

# 2. Drawdowns
n_strats = len(results)
fig, axes = plt.subplots(n_strats, 1, figsize=(16, 3 * n_strats), sharex=True)
for ax, (name, result) in zip(axes, results.items()):
    eq = result.equity_curve
    rm = eq.cummax()
    dd = (eq - rm) / rm
    ax.fill_between(dd.index, dd.values, 0, alpha=0.4, color="red")
    ax.set_title(f"Drawdown: {name}", fontsize=10)
    ax.set_ylabel("DD")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir / "drawdowns.png", dpi=150)
print(f"Saved: {output_dir / 'drawdowns.png'}")

# 3. Rolling alpha (63d) -- just router vs individual
fig, ax = plt.subplots(figsize=(16, 6))
for name in ["Regime Router", "Momentum (60d)", "Mean Reversion", "Pullback"]:
    result = results.get(name)
    if result is None:
        continue
    try:
        ra = rolling_alpha(result.returns, result.benchmark_returns, window=63)
        lw = 2.0 if name == "Regime Router" else 1.0
        ax.plot(ra.index, ra.values, label=name, linewidth=lw, alpha=0.8)
    except ValueError:
        continue
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_title("Rolling Alpha (63d): Router vs Key Strategies")
ax.set_ylabel("Annualized Alpha")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir / "rolling_alpha_comparison.png", dpi=150)
print(f"Saved: {output_dir / 'rolling_alpha_comparison.png'}")

# Save comparison
comp.to_csv(output_dir / "comparison.csv")
print(f"Saved: {output_dir / 'comparison.csv'}")

print(f"\n{'=' * 60}")
print("  ALL RESEARCH BACKTESTS COMPLETE")
print(f"{'=' * 60}")
