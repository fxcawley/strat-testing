"""
Factor research: well-known momentum signals on ETFs.

Tests each strategy on TWO non-overlapping periods:
  - Period 1: 2014-01-01 to 2019-12-31 (out-of-sample validation)
  - Period 2: 2020-01-01 to present (in-sample, what we've been testing)

A strategy is interesting only if it performs consistently across both.

All strategies:
  - Monthly rebalancing
  - 3bps transaction costs (realistic for highly liquid ETFs)
  - Benchmarked against SPY

Usage:
    source .venv/Scripts/activate
    python run_factors.py
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

from src.backtest.engine import run_backtest, BacktestResult
from src.backtest.metrics import sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio, alpha_beta
from src.analysis.alpha import t_test_alpha, bootstrap_alpha
from src.strategies.factors.universes import MULTI_ASSET_TICKERS, EQUITY_ETF_TICKERS
from src.strategies.factors.trend_following import TrendFollowing
from src.strategies.factors.cross_sectional import CrossSectionalMomentum
from src.strategies.factors.momentum_quality import MomentumQuality
from src.strategies.factors.cross_asset import CrossAssetTimeSeries, CrossAssetRelative
from src.utils.display import print_summary


# ── Config ──────────────────────────────────────────────────────────────────
BENCHMARK = "SPY"
REBALANCE = "ME"  # monthly
INITIAL_CAPITAL = 100_000.0
COST_PCT = 0.0003  # 3bps -- realistic for liquid ETFs
LOOKBACK_BUFFER = 500  # extra calendar days to fetch before start (covers 12-1 month lookback)

PERIODS = {
    "2014-2019 (OOS)": ("2014-01-01", "2019-12-31"),
    "2020-2026 (IS)": ("2020-01-01", None),
}


# ── Build strategies ───────────────────────────────────────────────────────

def build_strategies():
    return {
        # Trend following variants
        "Trend 12-1 (multi-asset)": (
            TrendFollowing(lookback_days=252, skip_days=21, vol_target=0.10),
            MULTI_ASSET_TICKERS,
        ),
        "Trend 6-1 (multi-asset)": (
            TrendFollowing(lookback_days=126, skip_days=21, vol_target=0.10),
            MULTI_ASSET_TICKERS,
        ),
        # Cross-sectional momentum on equity ETFs
        "XS Mom 12-1 (equity ETFs)": (
            CrossSectionalMomentum(lookback_days=252, skip_days=21, top_frac=0.30),
            EQUITY_ETF_TICKERS,
        ),
        "XS Mom 6-1 (equity ETFs)": (
            CrossSectionalMomentum(lookback_days=126, skip_days=21, top_frac=0.30),
            EQUITY_ETF_TICKERS,
        ),
        # Momentum + Quality
        "Mom+Quality (equity ETFs)": (
            MomentumQuality(mom_lookback=252, mom_skip=21, mom_weight=0.5, quality_weight=0.5,
                            top_frac=0.30),
            EQUITY_ETF_TICKERS,
        ),
        # Cross-asset
        "Cross-Asset TS Mom": (
            CrossAssetTimeSeries(lookback_days=252, skip_days=21, vol_target=0.10),
            MULTI_ASSET_TICKERS,
        ),
        "Cross-Asset Relative": (
            CrossAssetRelative(lookback_days=252, skip_days=21, top_frac=0.35,
                               max_class_weight=0.40),
            MULTI_ASSET_TICKERS,
        ),
    }


# ── Run backtests across both periods ─────────────────────────────────────

all_results: dict[str, dict[str, BacktestResult]] = {}  # {strategy: {period: result}}

for period_name, (start, end) in PERIODS.items():
    print(f"\n{'=' * 70}")
    print(f"  PERIOD: {period_name}")
    print(f"{'=' * 70}")

    strategies = build_strategies()

    for strat_name, (strat, universe) in strategies.items():
        print(f"\n  {strat_name}...", end=" ", flush=True)
        result = run_backtest(
            strategy=strat,
            universe=universe,
            start=start,
            end=end,
            benchmark=BENCHMARK,
            rebalance_freq=REBALANCE,
            initial_capital=INITIAL_CAPITAL,
            cost_pct=COST_PCT,
            lookback_buffer_days=LOOKBACK_BUFFER,
        )
        all_results.setdefault(strat_name, {})[period_name] = result
        s = result.summary()
        print(f"Return: {s['total_return']:+.1%}  Sharpe: {s['sharpe_ratio']:.2f}  "
              f"MaxDD: {s['max_drawdown']:.1%}  Costs: ${s['total_costs']:,.0f}")


# ── Comparison tables ─────────────────────────────────────────────────────

print(f"\n\n{'=' * 70}")
print("  CROSS-PERIOD COMPARISON")
print(f"{'=' * 70}")

for period_name in PERIODS:
    print(f"\n--- {period_name} ---")
    rows = []
    for strat_name in all_results:
        result = all_results[strat_name].get(period_name)
        if result is None:
            continue
        s = result.summary()
        a, b = alpha_beta(result.returns, result.benchmark_returns)
        t_res = t_test_alpha(result.excess_returns)

        rows.append({
            "Strategy": strat_name,
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
    print(comp.to_string())
    print(f"  SPY return: {list(all_results.values())[0][period_name].summary()['benchmark_return']:.2%}")


# ── Robustness check: Sharpe consistency across periods ────────────────────

print(f"\n\n{'=' * 70}")
print("  SHARPE RATIO CONSISTENCY CHECK")
print(f"{'=' * 70}")
print(f"\n{'Strategy':<30s}  {'2014-2019':>10s}  {'2020-2026':>10s}  {'Diff':>8s}  {'Consistent?':>12s}")
print("-" * 75)

period_names = list(PERIODS.keys())
for strat_name in all_results:
    sharpes = []
    for pn in period_names:
        r = all_results[strat_name].get(pn)
        if r:
            sharpes.append(r.summary()["sharpe_ratio"])
        else:
            sharpes.append(float("nan"))

    diff = abs(sharpes[0] - sharpes[1]) if len(sharpes) == 2 else float("nan")
    # "Consistent" if both periods have same-sign Sharpe and diff < 0.5
    consistent = (sharpes[0] > 0 and sharpes[1] > 0 and diff < 0.5)
    flag = "YES" if consistent else "NO"

    print(f"{strat_name:<30s}  {sharpes[0]:>10.2f}  {sharpes[1]:>10.2f}  {diff:>8.2f}  {flag:>12s}")


# ── Plots ─────────────────────────────────────────────────────────────────

output_dir = Path("data/results/factors")
output_dir.mkdir(parents=True, exist_ok=True)

for period_name, (start, end) in PERIODS.items():
    safe_name = period_name.replace(" ", "_").replace("(", "").replace(")", "")
    fig, ax = plt.subplots(figsize=(15, 7))

    for strat_name in all_results:
        result = all_results[strat_name].get(period_name)
        if result is None:
            continue
        ax.plot(result.equity_curve.index, result.equity_curve.values,
                label=strat_name, linewidth=1.3)

    bench = list(all_results.values())[0][period_name]
    ax.plot(bench.benchmark_curve.index, bench.benchmark_curve.values,
            label="SPY", linewidth=1.0, alpha=0.5, linestyle="--", color="gray")

    ax.set_title(f"Factor Strategies: {period_name} (monthly, 3bps)")
    ax.set_ylabel("Portfolio Value ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"equity_{safe_name}.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {output_dir / f'equity_{safe_name}.png'}")

# Sharpe bar chart comparison
fig, ax = plt.subplots(figsize=(14, 6))
strat_names = list(all_results.keys())
x = np.arange(len(strat_names))
width = 0.35

sharpes_p1 = [all_results[s][period_names[0]].summary()["sharpe_ratio"] for s in strat_names]
sharpes_p2 = [all_results[s][period_names[1]].summary()["sharpe_ratio"] for s in strat_names]

bars1 = ax.bar(x - width/2, sharpes_p1, width, label="2014-2019 (OOS)", color="steelblue")
bars2 = ax.bar(x + width/2, sharpes_p2, width, label="2020-2026 (IS)", color="coral")
ax.set_ylabel("Sharpe Ratio")
ax.set_title("Sharpe Ratio Consistency: Out-of-Sample vs In-Sample")
ax.set_xticks(x)
ax.set_xticklabels(strat_names, rotation=30, ha="right", fontsize=8)
ax.axhline(0, color="gray", linewidth=0.8)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(output_dir / "sharpe_consistency.png", dpi=150)
plt.close(fig)
print(f"Saved: {output_dir / 'sharpe_consistency.png'}")

print(f"\n{'=' * 70}")
print("  FACTOR RESEARCH COMPLETE")
print(f"{'=' * 70}")
