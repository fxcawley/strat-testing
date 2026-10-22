"""
MiniWeekly validation: rigorous walk-forward holdout on extended data.

Runs the compact 15-feature weekly model through purged walk-forward CV
with ~20 years of data (2005-2026), then compares to XS 12-1 Momentum
on the same folds.

Validation design:
  - 3-year minimum training window (expanding)
  - 12-month OOS test windows
  - 2-week purge gap between train and test
  - ~14 non-overlapping folds from 2008 to 2026
  - Reports per-fold OOS metrics + aggregate t-tests

Also runs full-period backtests (weekly and monthly) for comparison.

Usage:
    source .venv/Scripts/activate
    python run_mini_weekly.py
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
from src.utils.display import print_summary
from src.validation.walk_forward import walk_forward_validate

from src.strategies.mini_weekly import MiniWeeklyStrategy
from src.strategies.factors.cross_sectional import CrossSectionalMomentum
from src.strategies.shock_propagation import ShockPropagationStrategy
from src.strategies.dynamic_allocator import DynamicAllocator
from src.strategies.causal_temporal import CausalTemporalStrategy
from src.strategies.graph_diffusion import GraphDiffusionStrategy


# ── Universe (available since 2003+) ───────────────────────────────────────
UNIVERSE = sorted([
    "SPY", "QQQ", "IWM", "IWD", "IWF",  # US equity styles
    "EFA", "EEM", "EWJ",                  # international
    "XLK", "XLF", "XLE", "XLV", "XLI",   # sectors
    "XLP", "XLY", "XLU",                  # sectors
    "TLT", "IEF", "SHY", "LQD",          # fixed income
    "GLD",                                 # commodities
])

output_dir = Path("data/results/mini_weekly")
output_dir.mkdir(parents=True, exist_ok=True)


# ── Strategy factories ──────────────────────────────────────────────────────

def make_mini_weekly():
    return MiniWeeklyStrategy(
        top_frac=0.30,
        min_train_weeks=52,
        ridge_alpha=10.0,
        retrain_every=4,
    )

def make_xs_momentum():
    return CrossSectionalMomentum(
        lookback_days=252, skip_days=21, top_frac=0.30,
    )

def make_dynamic_allocator():
    return DynamicAllocator(
        strategies={
            "causal_temporal": CausalTemporalStrategy(
                top_frac=0.30, min_train_months=36, ridge_alpha=10.0,
            ),
            "graph_diffusion": GraphDiffusionStrategy(
                corr_lookback=126, momentum_lookback=252, skip_days=21,
                diffusion_time=0.5, corr_threshold=0.15, top_frac=0.30,
            ),
            "shock_propagation": ShockPropagationStrategy(
                shock_threshold=2.0, max_lag=5, corr_lookback=126,
                shock_lookback=21, momentum_lookback=63,
                propagation_weight=0.4, top_frac=0.30,
            ),
        },
        risk_aversion=2.0, perf_lookback=12,
        temperature=0.5, min_weight=0.05,
    )


# =====================================================================
#  PART 1: Walk-forward validation (purged k-fold)
# =====================================================================

def run_validation():
    print("\n" + "=" * 80)
    print("  PART 1: PURGED WALK-FORWARD VALIDATION (2005-2026)")
    print("=" * 80)

    common_kwargs = dict(
        universe=UNIVERSE,
        full_start="2005-01-01",
        full_end=None,
        min_train_years=3.0,
        test_months=12,
        purge_weeks=2,
        step_months=12,
        benchmark="SPY",
        cost_pct=0.0003,
        lookback_buffer_days=500,
    )

    # 1a. MiniWeekly (weekly rebalance)
    print("\n>>> MiniWeekly (weekly rebalance)")
    mini_report = walk_forward_validate(
        strategy_factory=make_mini_weekly,
        rebalance_freq="W",
        rebalance_threshold=0.01,
        strategy_name="MiniWeekly (15-feat, Ridge, weekly)",
        **common_kwargs,
    )
    mini_report.print_report()

    # 1b. XS 12-1 Momentum baseline (weekly rebalance for fair comparison)
    print("\n>>> XS 12-1 Momentum baseline (weekly rebalance)")
    xs_report = walk_forward_validate(
        strategy_factory=make_xs_momentum,
        rebalance_freq="W",
        rebalance_threshold=0.01,
        strategy_name="XS 12-1 Momentum (baseline, weekly)",
        **common_kwargs,
    )
    xs_report.print_report()

    # 1c. Dynamic Utility Allocator (weekly rebalance)
    print("\n>>> Dynamic Utility Allocator (weekly rebalance)")
    dyn_report = walk_forward_validate(
        strategy_factory=make_dynamic_allocator,
        rebalance_freq="W",
        rebalance_threshold=0.01,
        strategy_name="Dynamic Utility Allocator (weekly)",
        **common_kwargs,
    )
    dyn_report.print_report()

    return mini_report, xs_report, dyn_report


# =====================================================================
#  PART 2: Full-period extended backtests
# =====================================================================

def run_extended_backtests():
    print("\n" + "=" * 80)
    print("  PART 2: EXTENDED FULL-PERIOD BACKTESTS")
    print("=" * 80)

    PERIODS = {
        "2005-2012 (GFC era)": ("2005-01-01", "2012-12-31"),
        "2013-2019 (bull)": ("2013-01-01", "2019-12-31"),
        "2020-2026 (COVID+)": ("2020-01-01", None),
        "Full 2005-2026": ("2005-01-01", None),
    }

    strat_configs = {
        "XS 12-1 Momentum": (make_xs_momentum, "W"),
        "MiniWeekly": (make_mini_weekly, "W"),
        "Dynamic Allocator": (make_dynamic_allocator, "W"),
    }

    all_results = {}
    for period_name, (start, end) in PERIODS.items():
        print(f"\n{'-' * 70}")
        print(f"  {period_name}")
        print(f"{'-' * 70}")

        period_results = {}
        for strat_name, (factory, freq) in strat_configs.items():
            print(f"  Running: {strat_name} ...")
            strat = factory()
            try:
                result = run_backtest(
                    strategy=strat,
                    universe=UNIVERSE,
                    start=start,
                    end=end,
                    benchmark="SPY",
                    rebalance_freq=freq,
                    initial_capital=100_000.0,
                    cost_pct=0.0003,
                    lookback_buffer_days=500,
                    rebalance_threshold=0.01,
                )
                period_results[strat_name] = result
                print_summary(result.summary(), title=f"{strat_name} -- {period_name}")
            except Exception as e:
                print(f"  ERROR: {e}")

        all_results[period_name] = period_results

    return all_results


# =====================================================================
#  PART 3: Feature importance + model diagnostics
# =====================================================================

def run_diagnostics():
    print("\n" + "=" * 80)
    print("  PART 3: MODEL DIAGNOSTICS")
    print("=" * 80)

    # Run on full period to get final model state
    strat = make_mini_weekly()
    result = run_backtest(
        strategy=strat,
        universe=UNIVERSE,
        start="2005-01-01",
        end=None,
        benchmark="SPY",
        rebalance_freq="W",
        initial_capital=100_000.0,
        cost_pct=0.0003,
        lookback_buffer_days=500,
        rebalance_threshold=0.01,
    )

    # Feature importance
    importance = strat.get_feature_importance()
    if importance is not None:
        print("\nRidge Feature Coefficients (absolute magnitude):")
        print("-" * 50)
        for feat, coef in importance.items():
            bar = "+" * int(min(abs(coef) * 100, 40))
            sign = "+" if coef > 0 else "-"
            print(f"  {feat:>18s}: {sign}{abs(coef):.4f}  {bar}")

    # Training history size
    print(f"\n  Training samples collected: {len(strat._history)}")
    print(f"  Rebalance count: {strat._rebalance_count}")

    return strat, result


# =====================================================================
#  PART 4: Comparison plots
# =====================================================================

def make_plots(all_results, mini_report, xs_report, dyn_report):
    colors = {
        "XS 12-1 Momentum": "#888888",
        "MiniWeekly": "#2196F3",
        "Dynamic Allocator": "#E91E63",
    }

    for period_name, period_results in all_results.items():
        safe = period_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").replace("+","")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1], sharex=True)

        any_r = next(iter(period_results.values()), None)
        if any_r:
            ax1.plot(any_r.benchmark_curve.index, any_r.benchmark_curve.values,
                     label="SPY", linewidth=1, alpha=0.5, linestyle=":", color="black")

        for sn, res in period_results.items():
            c = colors.get(sn, "#666")
            lw = 2.5 if "Mini" in sn else 1.8
            ls = "--" if "Momentum" in sn else "-"
            ax1.plot(res.equity_curve.index, res.equity_curve.values,
                     label=sn, linewidth=lw, color=c, linestyle=ls)
            eq = res.equity_curve
            dd = (eq - eq.cummax()) / eq.cummax()
            ax2.plot(dd.index, dd.values, color=c, alpha=0.6, linewidth=1)

        ax1.set_title(f"MiniWeekly vs Baseline: {period_name}\n(weekly, 3bps, 1% threshold)", fontsize=13)
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax1.legend(loc="upper left", fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax2.set_ylabel("Drawdown")
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"mini_{safe}.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {output_dir / f'mini_{safe}.png'}")

    # Walk-forward fold comparison chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (report, label, color) in zip(axes, [
        (mini_report, "MiniWeekly", "#2196F3"),
        (xs_report, "XS Momentum", "#888888"),
        (dyn_report, "Dynamic Alloc", "#E91E63"),
    ]):
        valid = [f for f in report.folds if f.backtest is not None]
        folds = [f.fold_id for f in valid]
        sharpes = [f.oos_sharpe for f in valid]
        returns = [f.oos_return * 100 for f in valid]

        ax.bar(folds, sharpes, color=color, alpha=0.7)
        ax.axhline(y=0, color="black", linewidth=0.5)
        mean_s = np.mean(sharpes) if sharpes else 0
        ax.axhline(y=mean_s, color=color, linewidth=2, linestyle="--", alpha=0.8)
        ax.set_title(f"{label}\nMean OOS Sharpe: {mean_s:.2f}", fontsize=11)
        ax.set_xlabel("Fold")
        ax.set_ylabel("OOS Sharpe")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Walk-Forward OOS Sharpe by Fold", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "wf_fold_sharpes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / 'wf_fold_sharpes.png'}")


# =====================================================================
#  PART 5: Comparison summary table
# =====================================================================

def print_final_comparison(all_results, mini_report, xs_report, dyn_report):
    print(f"\n\n{'=' * 90}")
    print("  FINAL COMPARISON SUMMARY")
    print(f"{'=' * 90}")

    for period_name, period_results in all_results.items():
        print(f"\n  {period_name}")
        print(f"  {'-' * 85}")
        rows = []
        for sn, res in period_results.items():
            s = res.summary()
            try:
                a, b = alpha_beta(res.returns, res.benchmark_returns)
            except:
                a, b = 0.0, 0.0
            rows.append({
                "Strategy": sn,
                "Return": f"{s['total_return']:+.1%}",
                "CAGR": f"{s['cagr']:.1%}",
                "Sharpe": f"{s['sharpe_ratio']:.2f}",
                "Sortino": f"{sortino_ratio(res.returns):.2f}",
                "MaxDD": f"{s['max_drawdown']:.1%}",
                "Calmar": f"{calmar_ratio(res.equity_curve):.2f}",
                "Alpha": f"{a:+.4f}",
                "Beta": f"{b:.2f}",
                "Trades": f"{s['n_trades']}",
                "Costs": f"${s['total_costs']:,.0f}",
            })
        print(pd.DataFrame(rows).to_string(index=False))

    # WF validation comparison
    print(f"\n\n  Walk-Forward OOS Aggregate (12-month folds, 2008-2026)")
    print(f"  {'-' * 85}")
    rows = []
    for report in [mini_report, xs_report, dyn_report]:
        agg = report.aggregate()
        rows.append({
            "Strategy": report.strategy_name,
            "Mean Sharpe": f"{agg.get('oos_sharpe_mean', 0):.2f}",
            "Std Sharpe": f"{agg.get('oos_sharpe_std', 0):.2f}",
            "Mean Return": f"{agg.get('oos_return_mean', 0):+.1%}",
            "Mean Alpha": f"{agg.get('oos_alpha_mean', 0):+.4f}",
            "Mean MaxDD": f"{agg.get('oos_max_dd_mean', 0):.1%}",
            "Win Rate": f"{agg.get('win_rate', 0):.0%}",
            "Sharpe p": f"{agg.get('sharpe_p_value', 1):.3f}",
            "Alpha p": f"{agg.get('alpha_p_value', 1):.3f}",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # Save all to CSV
    all_rows = []
    for period_name, period_results in all_results.items():
        for sn, res in period_results.items():
            s = res.summary()
            try:
                a, b = alpha_beta(res.returns, res.benchmark_returns)
            except:
                a, b = 0.0, 0.0
            all_rows.append({
                "Period": period_name, "Strategy": sn,
                "Total Return": s["total_return"], "CAGR": s["cagr"],
                "Sharpe": s["sharpe_ratio"], "Sortino": sortino_ratio(res.returns),
                "Max Drawdown": s["max_drawdown"], "Calmar": calmar_ratio(res.equity_curve),
                "Alpha": a, "Beta": b,
                "Trades": s["n_trades"], "Costs": s["total_costs"],
            })
    csv_path = output_dir / "full_comparison.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")


# =====================================================================
#  Main
# =====================================================================

if __name__ == "__main__":
    # Part 1: Walk-forward validation
    mini_report, xs_report, dyn_report = run_validation()

    # Part 2: Extended backtests
    all_results = run_extended_backtests()

    # Part 3: Model diagnostics
    strat, diag_result = run_diagnostics()

    # Part 4: Plots
    make_plots(all_results, mini_report, xs_report, dyn_report)

    # Part 5: Final comparison
    print_final_comparison(all_results, mini_report, xs_report, dyn_report)

    print(f"\n{'=' * 80}")
    print("  DONE -- all results in data/results/mini_weekly/")
    print(f"{'=' * 80}")
