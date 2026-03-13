"""
Research runner for tracks (a), (b), (c).

  (a) Composite signals: MomentumMeanRevFilter, PullbackSentiment, MultiSignalComposite
  (b) Adaptive params: AdaptiveMeanReversion, AdaptiveMomentum
  (c) Sector rotation: SectorMomentum, SectorMeanReversion, SectorRelativeStrength

All compared against SPY benchmark.  Stock-level strategies use monthly
rebalancing on 100-stock universe.  Sector strategies use weekly
rebalancing on 11 sector ETFs.

Usage:
    source .venv/Scripts/activate
    python run_research_v2.py
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

from src.data.prices import fetch_prices
from src.data.ratings import build_consensus_history
from src.data.universe import SP500_CURRENT
from src.backtest.engine import run_backtest, BacktestResult
from src.backtest.metrics import (
    sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio, alpha_beta,
)
from src.analysis.alpha import t_test_alpha, bootstrap_alpha
from src.strategies.composite import (
    MomentumMeanRevFilter, PullbackSentiment, MultiSignalComposite,
)
from src.strategies.adaptive import AdaptiveMeanReversion, AdaptiveMomentum
from src.strategies.sector import (
    SectorMomentum, SectorMeanReversion, SectorRelativeStrength, SECTOR_TICKERS,
)
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.utils.display import print_summary


# ── Config ──────────────────────────────────────────────────────────────────
random.seed(42)
UNIVERSE_RAW = sorted(random.sample(SP500_CURRENT, 100))
START = "2020-01-01"
END = None
BENCHMARK = "SPY"
INITIAL_CAPITAL = 100_000.0
COST_PCT = 0.001  # 10bps


# ── Validate stock universe ───────────────────────────────────────────────
print("Validating stock universe...")
STOCK_UNIVERSE = []
for t in UNIVERSE_RAW:
    try:
        fetch_prices(t, start="2024-01-01")
        STOCK_UNIVERSE.append(t)
    except (ValueError, Exception):
        print(f"  Dropping {t}")
print(f"  Stock universe: {len(STOCK_UNIVERSE)} tickers")


# ── Pre-fetch analyst consensus for composites ─────────────────────────────
print("\nBuilding analyst consensus histories...")
consensus_cache = {}
for i, t in enumerate(STOCK_UNIVERSE):
    try:
        consensus_cache[t] = build_consensus_history(t)
    except (ValueError, Exception):
        pass
    if (i + 1) % 25 == 0:
        print(f"  [{i+1}/{len(STOCK_UNIVERSE)}] done")
        time.sleep(1)
print(f"  Consensus data for {len(consensus_cache)}/{len(STOCK_UNIVERSE)} tickers")


# ════════════════════════════════════════════════════════════════════════════
# TRACK (a): COMPOSITE SIGNALS
# ════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("  TRACK (a): COMPOSITE SIGNAL STRATEGIES")
print(f"{'=' * 70}")

track_a_strategies = {
    # Baselines
    "Momentum (baseline)": MomentumStrategy(lookback_days=60, top_n=10),
    "MeanRev (baseline)": MeanReversionStrategy(lookback=20, entry_z=1.5, top_n=10),
    # Composites
    "Momentum+MeanRevFilter": MomentumMeanRevFilter(
        momentum_days=60, zscore_lookback=20, zscore_ceiling=2.0, top_n=10,
    ),
    "Pullback+Sentiment": PullbackSentiment(
        trend_sma=50, rsi_threshold=45, consensus_cache=consensus_cache, top_n=10,
    ),
    "MultiSignal Composite": MultiSignalComposite(
        consensus_cache=consensus_cache, top_n=10,
    ),
}

track_a_results = {}
for name, strat in track_a_strategies.items():
    print(f"\n  Running: {name}")
    result = run_backtest(
        strategy=strat, universe=STOCK_UNIVERSE, start=START, end=END,
        benchmark=BENCHMARK, rebalance_freq="ME", initial_capital=INITIAL_CAPITAL,
        cost_pct=COST_PCT,
    )
    track_a_results[name] = result
    s = result.summary()
    print(f"    Return: {s['total_return']:+.1%}  Sharpe: {s['sharpe_ratio']:.2f}  "
          f"MaxDD: {s['max_drawdown']:.1%}  Costs: ${s['total_costs']:,.0f}  "
          f"Trades: {s['n_trades']}")


# ════════════════════════════════════════════════════════════════════════════
# TRACK (b): ADAPTIVE PARAMETERS
# ════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("  TRACK (b): ADAPTIVE PARAMETER STRATEGIES")
print(f"{'=' * 70}")

track_b_strategies = {
    # Baselines (fixed params)
    "MeanRev Fixed": MeanReversionStrategy(lookback=20, entry_z=1.5, top_n=10),
    "Momentum Fixed (60d)": MomentumStrategy(lookback_days=60, top_n=10),
    # Adaptive
    "MeanRev Adaptive": AdaptiveMeanReversion(top_n=10),
    "Momentum Adaptive": AdaptiveMomentum(),
}

track_b_results = {}
for name, strat in track_b_strategies.items():
    print(f"\n  Running: {name}")
    result = run_backtest(
        strategy=strat, universe=STOCK_UNIVERSE, start=START, end=END,
        benchmark=BENCHMARK, rebalance_freq="ME", initial_capital=INITIAL_CAPITAL,
        cost_pct=COST_PCT,
    )
    track_b_results[name] = result
    s = result.summary()
    print(f"    Return: {s['total_return']:+.1%}  Sharpe: {s['sharpe_ratio']:.2f}  "
          f"MaxDD: {s['max_drawdown']:.1%}  Costs: ${s['total_costs']:,.0f}  "
          f"Trades: {s['n_trades']}")


# ════════════════════════════════════════════════════════════════════════════
# TRACK (c): SECTOR ROTATION
# ════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("  TRACK (c): SECTOR ROTATION STRATEGIES")
print(f"{'=' * 70}")

track_c_strategies = {
    "Sector Momentum": SectorMomentum(lookback_days=60, top_n=4),
    "Sector MeanRev": SectorMeanReversion(lookback_days=20, top_n=3),
    "Sector RelStrength": SectorRelativeStrength(top_n=4),
}

track_c_results = {}
for name, strat in track_c_strategies.items():
    print(f"\n  Running: {name}")
    result = run_backtest(
        strategy=strat, universe=SECTOR_TICKERS, start=START, end=END,
        benchmark=BENCHMARK, rebalance_freq="W-FRI", initial_capital=INITIAL_CAPITAL,
        cost_pct=COST_PCT,
    )
    track_c_results[name] = result
    s = result.summary()
    print(f"    Return: {s['total_return']:+.1%}  Sharpe: {s['sharpe_ratio']:.2f}  "
          f"MaxDD: {s['max_drawdown']:.1%}  Costs: ${s['total_costs']:,.0f}  "
          f"Trades: {s['n_trades']}")


# ════════════════════════════════════════════════════════════════════════════
# FULL COMPARISON TABLE
# ════════════════════════════════════════════════════════════════════════════

all_results = {}
for label, res_dict in [("(a)", track_a_results), ("(b)", track_b_results), ("(c)", track_c_results)]:
    for name, result in res_dict.items():
        all_results[f"{label} {name}"] = result

print(f"\n\n{'=' * 70}")
print("  FULL COMPARISON TABLE")
print(f"{'=' * 70}")

rows = []
for name, result in all_results.items():
    s = result.summary()
    a, b = alpha_beta(result.returns, result.benchmark_returns)
    t_res = t_test_alpha(result.excess_returns)

    rows.append({
        "Strategy": name,
        "Return": s["total_return"],
        "CAGR": s["cagr"],
        "Sharpe": s["sharpe_ratio"],
        "Sortino": sortino_ratio(result.returns),
        "MaxDD": s["max_drawdown"],
        "Calmar": calmar_ratio(result.equity_curve),
        "Alpha": a,
        "Beta": b,
        "Alpha p": t_res["p_value"],
        "Sig?": t_res["significant_5pct"],
        "Costs": s["total_costs"],
        "Trades": s["n_trades"],
    })

comp = pd.DataFrame(rows).set_index("Strategy")
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 220)
print("\n" + comp.to_string())
print(f"\nBenchmark ({BENCHMARK}) return: {list(all_results.values())[0].summary()['benchmark_return']:.2%}")


# ════════════════════════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════════════════════════

output_dir = Path("data/results/research_v2")
output_dir.mkdir(parents=True, exist_ok=True)

# Plot helper
def plot_track(track_results, title, filename, bench_result):
    fig, ax = plt.subplots(figsize=(15, 7))
    for name, result in track_results.items():
        ax.plot(result.equity_curve.index, result.equity_curve.values,
                label=name, linewidth=1.3)
    ax.plot(bench_result.benchmark_curve.index, bench_result.benchmark_curve.values,
            label="SPY", linewidth=1.0, alpha=0.5, linestyle="--", color="gray")
    ax.set_title(title)
    ax.set_ylabel("Portfolio Value ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / filename}")


bench_ref = list(track_a_results.values())[0]
plot_track(track_a_results, "Track (a): Composite Signals (monthly, 10bps)", "track_a.png", bench_ref)
plot_track(track_b_results, "Track (b): Adaptive Parameters (monthly, 10bps)", "track_b.png", bench_ref)

bench_ref_c = list(track_c_results.values())[0]
plot_track(track_c_results, "Track (c): Sector Rotation (weekly, 10bps)", "track_c.png", bench_ref_c)

# Combined best-of-each
fig, ax = plt.subplots(figsize=(15, 7))
highlights = {}
for track_res in [track_a_results, track_b_results, track_c_results]:
    best_name = max(track_res, key=lambda n: track_res[n].summary()["sharpe_ratio"])
    highlights[best_name] = track_res[best_name]
for name, result in highlights.items():
    ax.plot(result.equity_curve.index, result.equity_curve.values, label=name, linewidth=1.5)
ax.plot(bench_ref.benchmark_curve.index, bench_ref.benchmark_curve.values,
        label="SPY", linewidth=1.0, alpha=0.5, linestyle="--", color="gray")
ax.set_title("Best Strategy Per Research Track vs SPY")
ax.set_ylabel("Portfolio Value ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir / "best_per_track.png", dpi=150)
plt.close(fig)
print(f"Saved: {output_dir / 'best_per_track.png'}")

comp.to_csv(output_dir / "comparison.csv")
print(f"Saved: {output_dir / 'comparison.csv'}")

print(f"\n{'=' * 70}")
print("  ALL RESEARCH TRACKS COMPLETE")
print(f"{'=' * 70}")
