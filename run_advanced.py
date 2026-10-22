"""
Advanced strategies backtest: Causal Temporal, Graph Diffusion,
Shock Propagation, and Dynamic Utility Allocator.

Compares all four against the XS 12-1 Momentum baseline on the same
universe, same engine settings, same time periods.

Usage:
    source .venv/Scripts/activate
    python run_advanced.py
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

# Strategies
from src.strategies.factors.cross_sectional import CrossSectionalMomentum
from src.strategies.causal_temporal import CausalTemporalStrategy
from src.strategies.graph_diffusion import GraphDiffusionStrategy
from src.strategies.shock_propagation import ShockPropagationStrategy
from src.strategies.dynamic_allocator import DynamicAllocator


# ── Universe ────────────────────────────────────────────────────────────────
# Same universe as production blend for fair comparison.
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


# ── Engine settings (match production) ──────────────────────────────────────
ENGINE_KWARGS = dict(
    universe=FULL_UNIVERSE,
    benchmark="SPY",
    rebalance_freq="ME",
    initial_capital=100_000.0,
    cost_pct=0.0003,           # 3bps
    lookback_buffer_days=500,
    rebalance_threshold=0.02,  # 2% threshold
)


def make_strategies() -> dict[str, object]:
    """Create fresh strategy instances (resets internal state)."""
    # Baseline
    xs_mom = CrossSectionalMomentum(
        lookback_days=252, skip_days=21, top_frac=0.30,
    )

    # New strategies
    causal = CausalTemporalStrategy(
        top_frac=0.30, min_train_months=36, ridge_alpha=10.0,
    )
    graph = GraphDiffusionStrategy(
        corr_lookback=126, momentum_lookback=252, skip_days=21,
        diffusion_time=0.5, corr_threshold=0.15, top_frac=0.30,
    )
    shock = ShockPropagationStrategy(
        shock_threshold=2.0, max_lag=5, corr_lookback=126,
        shock_lookback=21, momentum_lookback=63,
        propagation_weight=0.4, top_frac=0.30,
    )

    # Dynamic allocator combining all three new strategies
    allocator = DynamicAllocator(
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
        risk_aversion=2.0,
        perf_lookback=12,
        temperature=0.5,
        min_weight=0.05,
    )

    return {
        "XS 12-1 Momentum (baseline)": xs_mom,
        "Causal Temporal CNN": causal,
        "Graph Diffusion": graph,
        "Shock Propagation": shock,
        "Dynamic Utility Allocator": allocator,
    }


# ── Periods ─────────────────────────────────────────────────────────────────
PERIODS = {
    "2010-2019": ("2010-01-01", "2019-12-31"),
    "2020-2026": ("2020-01-01", None),
    "Full (2010-2026)": ("2010-01-01", None),
}


# ── Run ─────────────────────────────────────────────────────────────────────

def run_all():
    all_results: dict[str, dict[str, BacktestResult]] = {}

    for period_name, (start, end) in PERIODS.items():
        print(f"\n{'=' * 70}")
        print(f"  PERIOD: {period_name}")
        print(f"{'=' * 70}")

        strategies = make_strategies()
        period_results: dict[str, BacktestResult] = {}

        for strat_name, strat in strategies.items():
            print(f"\n  Running: {strat_name} ...")
            try:
                result = run_backtest(
                    strategy=strat,
                    start=start,
                    end=end,
                    **ENGINE_KWARGS,
                )
                period_results[strat_name] = result
                print_summary(result.summary(), title=f"{strat_name} -- {period_name}")
            except Exception as e:
                print(f"  ERROR running {strat_name}: {e}")

        all_results[period_name] = period_results

    # ── Comparison tables ───────────────────────────────────────────────────
    for period_name, period_results in all_results.items():
        print(f"\n\n{'=' * 90}")
        print(f"  COMPARISON: {period_name}")
        print(f"{'=' * 90}")

        rows = []
        for strat_name, result in period_results.items():
            s = result.summary()
            try:
                a, b = alpha_beta(result.returns, result.benchmark_returns)
            except Exception:
                a, b = 0.0, 0.0
            try:
                t_res = t_test_alpha(result.excess_returns)
                sig = "Yes" if t_res.get("significant_5pct", False) else "No"
            except Exception:
                sig = "N/A"

            rows.append({
                "Strategy": strat_name,
                "Return": f"{s['total_return']:+.1%}",
                "CAGR": f"{s['cagr']:.1%}",
                "Sharpe": f"{s['sharpe_ratio']:.2f}",
                "Sortino": f"{sortino_ratio(result.returns):.2f}",
                "MaxDD": f"{s['max_drawdown']:.1%}",
                "Calmar": f"{calmar_ratio(result.equity_curve):.2f}",
                "Alpha": f"{a:+.4f}",
                "Beta": f"{b:.2f}",
                "Sig@5%": sig,
                "Trades": f"{s['n_trades']}",
                "Costs": f"${s['total_costs']:,.0f}",
            })

        comp = pd.DataFrame(rows)
        print("\n" + comp.to_string(index=False))

    # ── Plots ───────────────────────────────────────────────────────────────
    output_dir = Path("data/results/advanced")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Color palette
    colors = {
        "XS 12-1 Momentum (baseline)": "#888888",
        "Causal Temporal CNN": "#2196F3",
        "Graph Diffusion": "#4CAF50",
        "Shock Propagation": "#FF9800",
        "Dynamic Utility Allocator": "#E91E63",
    }

    for period_name, period_results in all_results.items():
        safe = period_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")

        # Equity curves comparison
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(16, 10), height_ratios=[3, 1], sharex=True,
        )

        # Plot benchmark
        any_result = next(iter(period_results.values()), None)
        if any_result is not None:
            ax1.plot(
                any_result.benchmark_curve.index,
                any_result.benchmark_curve.values,
                label="SPY (benchmark)", linewidth=1.0, alpha=0.5,
                linestyle=":", color="black",
            )

        for strat_name, result in period_results.items():
            color = colors.get(strat_name, "#666666")
            lw = 2.5 if "Allocator" in strat_name else 1.8
            ls = "-" if "baseline" not in strat_name else "--"

            ax1.plot(
                result.equity_curve.index, result.equity_curve.values,
                label=strat_name, linewidth=lw, color=color, linestyle=ls,
            )

            # Drawdown on second axis
            eq = result.equity_curve
            dd = (eq - eq.cummax()) / eq.cummax()
            ax2.plot(dd.index, dd.values, color=color, alpha=0.6, linewidth=1.0)

        ax1.set_title(
            f"Advanced Strategies Comparison: {period_name}\n"
            f"(monthly rebal, 3bps costs, 2% threshold)",
            fontsize=13,
        )
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(True, alpha=0.3)

        ax2.set_ylabel("Drawdown")
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / f"advanced_{safe}.png", dpi=150)
        plt.close(fig)
        print(f"\nSaved: {output_dir / f'advanced_{safe}.png'}")

    # ── Save comparison CSV ─────────────────────────────────────────────────
    all_rows = []
    for period_name, period_results in all_results.items():
        for strat_name, result in period_results.items():
            s = result.summary()
            try:
                a, b = alpha_beta(result.returns, result.benchmark_returns)
            except Exception:
                a, b = 0.0, 0.0
            all_rows.append({
                "Period": period_name,
                "Strategy": strat_name,
                "Total Return": s["total_return"],
                "CAGR": s["cagr"],
                "Sharpe": s["sharpe_ratio"],
                "Sortino": sortino_ratio(result.returns),
                "Max Drawdown": s["max_drawdown"],
                "Calmar": calmar_ratio(result.equity_curve),
                "Alpha": a,
                "Beta": b,
                "Trades": s["n_trades"],
                "Costs": s["total_costs"],
            })

    csv_path = output_dir / "comparison.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    print(f"\n{'=' * 70}")
    print("  DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_all()
