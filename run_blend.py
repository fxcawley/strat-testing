"""
Production backtest: XS Momentum + Trend Following blend.

This is the final output of the research project. A single strategy
combining cross-sectional equity momentum with multi-asset trend
following, run end-to-end through the engine with:
  - Share-count-based position tracking (no implicit rebalancing)
  - 3bps transaction costs (realistic for liquid ETFs)
  - 2% rebalance threshold (skip trades below this weight change)
  - Monthly rebalancing
  - Three-period validation

Usage:
    source .venv/Scripts/activate
    python run_blend.py
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
from src.strategies.factors.trend_following import TrendFollowing
from src.strategies.factors.cross_sectional import CrossSectionalMomentum
from src.strategies.factors.blend import StaticBlend
from src.utils.display import print_summary


# ── Universe ────────────────────────────────────────────────────────────────
# Core ETFs available since 2004, covering equities + bonds + gold.
EQUITY_ETFS = [
    "SPY", "QQQ", "IWM", "IWD", "IWF", "EFA", "EEM", "EWJ",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU",
]
MULTI_ASSET = [
    "SPY", "QQQ", "IWM", "EFA", "EEM",
    "TLT", "IEF", "SHY", "LQD",
    "GLD",
]
FULL_UNIVERSE = sorted(set(EQUITY_ETFS + MULTI_ASSET))

# ── Strategy ────────────────────────────────────────────────────────────────
blend = StaticBlend(strategies={
    "xs_momentum": (
        CrossSectionalMomentum(lookback_days=252, skip_days=21, top_frac=0.30),
        0.5,
    ),
    "trend_following": (
        TrendFollowing(lookback_days=252, skip_days=21, vol_target=0.10),
        0.5,
    ),
})

# ── Periods ─────────────────────────────────────────────────────────────────
PERIODS = {
    "2005-2013 (GFC)": ("2005-01-01", "2013-12-31"),
    "2014-2019": ("2014-01-01", "2019-12-31"),
    "2020-2026": ("2020-01-01", None),
}

# ── Run ─────────────────────────────────────────────────────────────────────
results: dict[str, BacktestResult] = {}

for period_name, (start, end) in PERIODS.items():
    print(f"\n{'=' * 60}")
    print(f"  {period_name}")
    print(f"{'=' * 60}")

    # Fresh strategy instance per period (resets internal state)
    strat = StaticBlend(strategies={
        "xs_momentum": (
            CrossSectionalMomentum(lookback_days=252, skip_days=21, top_frac=0.30),
            0.5,
        ),
        "trend_following": (
            TrendFollowing(lookback_days=252, skip_days=21, vol_target=0.10),
            0.5,
        ),
    })

    result = run_backtest(
        strategy=strat,
        universe=FULL_UNIVERSE,
        start=start,
        end=end,
        benchmark="SPY",
        rebalance_freq="ME",
        initial_capital=100_000.0,
        cost_pct=0.0003,          # 3bps
        lookback_buffer_days=500,
        rebalance_threshold=0.02, # 2% threshold
    )
    results[period_name] = result
    print_summary(result.summary(), title=f"XS Mom + Trend (50/50) -- {period_name}")


# ── Summary table ───────────────────────────────────────────────────────────

print(f"\n\n{'=' * 60}")
print("  THREE-PERIOD SUMMARY: XS Mom + Trend (50/50)")
print(f"{'=' * 60}")

rows = []
for period_name, result in results.items():
    s = result.summary()
    a, b = alpha_beta(result.returns, result.benchmark_returns)
    t_res = t_test_alpha(result.excess_returns)
    boot = bootstrap_alpha(result.excess_returns, n_bootstrap=5000)

    rows.append({
        "Period": period_name,
        "Return": f"{s['total_return']:+.1%}",
        "CAGR": f"{s['cagr']:.1%}",
        "Sharpe": f"{s['sharpe_ratio']:.2f}",
        "Sortino": f"{sortino_ratio(result.returns):.2f}",
        "MaxDD": f"{s['max_drawdown']:.1%}",
        "Calmar": f"{calmar_ratio(result.equity_curve):.2f}",
        "Alpha": f"{a:+.4f}",
        "Beta": f"{b:.2f}",
        "Costs": f"${s['total_costs']:,.0f}",
        "Trades": f"{s['n_trades']}",
        "SPY": f"{s['benchmark_return']:+.1%}",
    })

comp = pd.DataFrame(rows)
print("\n" + comp.to_string(index=False))


# ── Plot ────────────────────────────────────────────────────────────────────

output_dir = Path("data/results/blend")
output_dir.mkdir(parents=True, exist_ok=True)

for period_name, result in results.items():
    safe = period_name.replace(" ", "_").replace("(", "").replace(")", "")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[3, 1], sharex=True)

    # Equity curve
    ax1.plot(result.equity_curve.index, result.equity_curve.values,
             label="XS Mom + Trend (50/50)", linewidth=2, color="steelblue")
    ax1.plot(result.benchmark_curve.index, result.benchmark_curve.values,
             label="SPY", linewidth=1.2, alpha=0.6, linestyle="--", color="gray")
    ax1.set_title(f"XS Mom + Trend Blend: {period_name} (monthly, 3bps, 2% threshold)")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Drawdown
    eq = result.equity_curve
    dd = (eq - eq.cummax()) / eq.cummax()
    ax2.fill_between(dd.index, dd.values, 0, alpha=0.4, color="red")
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / f"blend_{safe}.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / f'blend_{safe}.png'}")

print(f"\n{'=' * 60}")
print("  DONE")
print(f"{'=' * 60}")
