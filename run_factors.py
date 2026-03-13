"""
Factor research v2: three-period validation + XS Mom / Trend blend.

Periods:
  - 2005-2013: stress test (GFC, 2009 momentum crash, Euro crisis)
  - 2014-2019: post-GFC expansion
  - 2020-2026: COVID and rate-hiking cycle

Strategies:
  - Trend 12-1 (multi-asset, vol-scaled)
  - Trend 6-1 (multi-asset, vol-scaled)
  - XS Mom 12-1 (equity ETFs, top 30%)
  - XS Mom 6-1 (equity ETFs, top 30%)
  - Mom+Quality (equity ETFs)
  - Cross-Asset Relative (multi-asset, class-capped)
  - XS Mom + Trend 50/50 blend

All: monthly rebalance, 3bps costs, SPY benchmark.
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
from src.analysis.alpha import t_test_alpha
from src.strategies.factors.trend_following import TrendFollowing
from src.strategies.factors.cross_sectional import CrossSectionalMomentum
from src.strategies.factors.momentum_quality import MomentumQuality
from src.strategies.factors.cross_asset import CrossAssetRelative
from src.strategies.factors.blend import StaticBlend
from src.utils.display import print_summary


BENCHMARK = "SPY"
REBALANCE = "ME"
INITIAL_CAPITAL = 100_000.0
COST_PCT = 0.0003  # 3bps
LOOKBACK_BUFFER = 500

# Universes that work across all three periods.
# Some ETFs didn't exist before 2005, so the early period uses a reduced set.
EQUITY_ETFS_CORE = [
    "SPY", "QQQ", "IWM", "IWD", "IWF", "EFA", "EEM", "EWJ",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU",
]

MULTI_ASSET_CORE = [
    "SPY", "QQQ", "IWM", "EFA", "EEM",
    "TLT", "IEF", "SHY", "LQD",
    "GLD",
]

# Combined universe (superset) for the blend strategy
BLEND_UNIVERSE = sorted(set(EQUITY_ETFS_CORE + MULTI_ASSET_CORE))

PERIODS = {
    "2005-2013 (GFC)": ("2005-01-01", "2013-12-31"),
    "2014-2019": ("2014-01-01", "2019-12-31"),
    "2020-2026": ("2020-01-01", None),
}


def build_strategies():
    trend_12 = TrendFollowing(lookback_days=252, skip_days=21, vol_target=0.10)
    trend_6 = TrendFollowing(lookback_days=126, skip_days=21, vol_target=0.10)
    xs_mom_12 = CrossSectionalMomentum(lookback_days=252, skip_days=21, top_frac=0.30)
    xs_mom_6 = CrossSectionalMomentum(lookback_days=126, skip_days=21, top_frac=0.30)
    mom_qual = MomentumQuality(mom_lookback=252, mom_skip=21, mom_weight=0.5,
                               quality_weight=0.5, top_frac=0.30)
    xasset_rel = CrossAssetRelative(lookback_days=252, skip_days=21, top_frac=0.35,
                                    max_class_weight=0.40)

    # The blend: 50% XS Mom 12-1 on equity ETFs + 50% Trend 12-1 on multi-asset
    # Uses the combined universe so both sub-strategies can find their tickers.
    blend = StaticBlend(strategies={
        "xs_mom": (CrossSectionalMomentum(lookback_days=252, skip_days=21, top_frac=0.30), 0.5),
        "trend": (TrendFollowing(lookback_days=252, skip_days=21, vol_target=0.10), 0.5),
    })

    return {
        "Trend 12-1": (trend_12, MULTI_ASSET_CORE),
        "Trend 6-1": (trend_6, MULTI_ASSET_CORE),
        "XS Mom 12-1": (xs_mom_12, EQUITY_ETFS_CORE),
        "XS Mom 6-1": (xs_mom_6, EQUITY_ETFS_CORE),
        "Mom+Quality": (mom_qual, EQUITY_ETFS_CORE),
        "Cross-Asset Rel": (xasset_rel, MULTI_ASSET_CORE),
        "XS Mom + Trend (50/50)": (blend, BLEND_UNIVERSE),
    }


# ── Run ─────────────────────────────────────────────────────────────────────

all_results: dict[str, dict[str, BacktestResult]] = {}

for period_name, (start, end) in PERIODS.items():
    print(f"\n{'=' * 70}")
    print(f"  PERIOD: {period_name}")
    print(f"{'=' * 70}")

    strategies = build_strategies()
    for strat_name, (strat, universe) in strategies.items():
        print(f"\n  {strat_name}...", end=" ", flush=True)
        result = run_backtest(
            strategy=strat, universe=universe, start=start, end=end,
            benchmark=BENCHMARK, rebalance_freq=REBALANCE,
            initial_capital=INITIAL_CAPITAL, cost_pct=COST_PCT,
            lookback_buffer_days=LOOKBACK_BUFFER,
        )
        all_results.setdefault(strat_name, {})[period_name] = result
        s = result.summary()
        print(f"Return: {s['total_return']:+.1%}  Sharpe: {s['sharpe_ratio']:.2f}  "
              f"MaxDD: {s['max_drawdown']:.1%}  Costs: ${s['total_costs']:,.0f}")


# ── Tables ──────────────────────────────────────────────────────────────────

print(f"\n\n{'=' * 70}")
print("  CROSS-PERIOD COMPARISON")
print(f"{'=' * 70}")

for period_name in PERIODS:
    print(f"\n--- {period_name} ---")
    rows = []
    for strat_name in all_results:
        r = all_results[strat_name].get(period_name)
        if r is None:
            continue
        s = r.summary()
        a, b = alpha_beta(r.returns, r.benchmark_returns)
        t_res = t_test_alpha(r.excess_returns)
        rows.append({
            "Strategy": strat_name,
            "Return": s["total_return"],
            "CAGR": s["cagr"],
            "Sharpe": s["sharpe_ratio"],
            "Sortino": sortino_ratio(r.returns),
            "MaxDD": s["max_drawdown"],
            "Calmar": calmar_ratio(r.equity_curve),
            "Alpha": a,
            "Beta": b,
            "Alpha p": t_res["p_value"],
            "Costs": s["total_costs"],
            "Trades": s["n_trades"],
        })
    comp = pd.DataFrame(rows).set_index("Strategy")
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)
    print(comp.to_string())
    bench_ret = list(all_results.values())[0][period_name].summary()["benchmark_return"]
    print(f"  SPY: {bench_ret:.2%}")


# ── Consistency ─────────────────────────────────────────────────────────────

print(f"\n\n{'=' * 70}")
print("  THREE-PERIOD SHARPE CONSISTENCY")
print(f"{'=' * 70}")

period_names = list(PERIODS.keys())
header = f"{'Strategy':<28s}" + "".join(f"  {p:>14s}" for p in period_names) + "  Min    Max    Consistent?"
print(f"\n{header}")
print("-" * len(header))

for strat_name in all_results:
    sharpes = []
    for pn in period_names:
        r = all_results[strat_name].get(pn)
        sharpes.append(r.summary()["sharpe_ratio"] if r else float("nan"))

    s_min, s_max = min(sharpes), max(sharpes)
    all_positive = all(s > 0 for s in sharpes)
    spread = s_max - s_min
    consistent = all_positive and spread < 0.5
    flag = "YES" if consistent else "NO"
    vals = "".join(f"  {s:>14.2f}" for s in sharpes)
    print(f"{strat_name:<28s}{vals}  {s_min:.2f}   {s_max:.2f}   {flag:>12s}")


# ── Plots ───────────────────────────────────────────────────────────────────

output_dir = Path("data/results/factors")
output_dir.mkdir(parents=True, exist_ok=True)

for period_name, (start, end) in PERIODS.items():
    safe_name = period_name.replace(" ", "_").replace("(", "").replace(")", "")
    fig, ax = plt.subplots(figsize=(15, 7))
    for strat_name in all_results:
        r = all_results[strat_name].get(period_name)
        if r is None:
            continue
        lw = 2.0 if "blend" in strat_name.lower() or "50/50" in strat_name else 1.2
        ax.plot(r.equity_curve.index, r.equity_curve.values, label=strat_name, linewidth=lw)
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

# Sharpe bar chart
fig, ax = plt.subplots(figsize=(14, 6))
strat_names = list(all_results.keys())
x = np.arange(len(strat_names))
width = 0.25

for i, pn in enumerate(period_names):
    sharpes = [all_results[s][pn].summary()["sharpe_ratio"] for s in strat_names]
    ax.bar(x + (i - 1) * width, sharpes, width, label=pn)

ax.set_ylabel("Sharpe Ratio")
ax.set_title("Three-Period Sharpe Consistency")
ax.set_xticks(x)
ax.set_xticklabels(strat_names, rotation=25, ha="right", fontsize=8)
ax.axhline(0, color="gray", linewidth=0.8)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(output_dir / "sharpe_3period.png", dpi=150)
plt.close(fig)
print(f"Saved: {output_dir / 'sharpe_3period.png'}")

print(f"\n{'=' * 70}")
print("  FACTOR RESEARCH COMPLETE")
print(f"{'=' * 70}")
