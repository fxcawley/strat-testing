"""
ML return prediction vs rule-based momentum.

Trains a model at each monthly rebalance using only past data,
predicts next-month ETF returns, and compares the ML signal to
the rule-based XS Momentum baseline.

Walk-forward: no look-ahead. The model is retrained monthly on
an expanding window of historical features and realized returns.

Usage:
    source .venv/Scripts/activate
    python run_ml.py
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
from src.strategies.factors.cross_sectional import CrossSectionalMomentum
from src.ml.strategy import MLReturnPredictor
from src.utils.display import print_summary


EQUITY_ETFS = [
    "SPY", "QQQ", "IWM", "IWD", "IWF", "EFA", "EEM", "EWJ",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU",
]
# Include bonds/gold in universe so ML can use market-level features
# but the strategy primarily trades equities
FULL_UNIVERSE = EQUITY_ETFS + ["TLT", "IEF", "SHY", "LQD", "GLD"]

BENCHMARK = "SPY"
REBALANCE = "ME"
INITIAL_CAPITAL = 100_000.0
COST_PCT = 0.0003  # 3bps
LOOKBACK_BUFFER = 500

# Two periods: the ML model needs ~3 years to accumulate training data
# before it starts trading, so results in early months will be cash.
PERIODS = {
    "2008-2013 (GFC+recovery)": ("2008-01-01", "2013-12-31"),
    "2014-2019": ("2014-01-01", "2019-12-31"),
    "2020-2026": ("2020-01-01", None),
}


def build_strategies():
    return {
        "XS Mom 12-1 (baseline)": CrossSectionalMomentum(
            lookback_days=252, skip_days=21, top_frac=0.30,
        ),
        "ML Ridge": MLReturnPredictor(
            model_type="ridge", top_frac=0.30, min_train_months=36,
            ridge_alpha=1.0,
        ),
        "ML GBM": MLReturnPredictor(
            model_type="gbm", top_frac=0.30, min_train_months=36,
            gbm_max_depth=3, gbm_n_estimators=100,
        ),
    }


all_results: dict[str, dict[str, BacktestResult]] = {}

for period_name, (start, end) in PERIODS.items():
    print(f"\n{'=' * 70}")
    print(f"  PERIOD: {period_name}")
    print(f"{'=' * 70}")

    strategies = build_strategies()
    for strat_name, strat in strategies.items():
        print(f"\n  {strat_name}...", end=" ", flush=True)
        result = run_backtest(
            strategy=strat,
            universe=FULL_UNIVERSE,
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
              f"MaxDD: {s['max_drawdown']:.1%}  Costs: ${s['total_costs']:,.0f}  "
              f"Trades: {s['n_trades']}")


# ── Tables ──────────────────────────────────────────────────────────────────

print(f"\n\n{'=' * 70}")
print("  ML vs RULE-BASED COMPARISON")
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
    bench = list(all_results.values())[0][period_name].summary()["benchmark_return"]
    print(f"  SPY: {bench:.2%}")


# ── Consistency ─────────────────────────────────────────────────────────────

print(f"\n\n{'=' * 70}")
print("  SHARPE CONSISTENCY")
print(f"{'=' * 70}")

period_names = list(PERIODS.keys())
print(f"\n{'Strategy':<28s}" + "".join(f"  {p:>20s}" for p in period_names))
print("-" * 90)
for strat_name in all_results:
    sharpes = []
    for pn in period_names:
        r = all_results[strat_name].get(pn)
        sharpes.append(r.summary()["sharpe_ratio"] if r else float("nan"))
    vals = "".join(f"  {s:>20.2f}" for s in sharpes)
    print(f"{strat_name:<28s}{vals}")


# ── Plots ───────────────────────────────────────────────────────────────────

output_dir = Path("data/results/ml")
output_dir.mkdir(parents=True, exist_ok=True)

for period_name, (start, end) in PERIODS.items():
    safe = period_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_")
    fig, ax = plt.subplots(figsize=(15, 7))
    for strat_name in all_results:
        r = all_results[strat_name].get(period_name)
        if r is None:
            continue
        lw = 1.5 if "ML" in strat_name else 1.0
        ax.plot(r.equity_curve.index, r.equity_curve.values, label=strat_name, linewidth=lw)
    bench = list(all_results.values())[0][period_name]
    ax.plot(bench.benchmark_curve.index, bench.benchmark_curve.values,
            label="SPY", linewidth=1.0, alpha=0.5, linestyle="--", color="gray")
    ax.set_title(f"ML vs Rule-Based: {period_name} (monthly, 3bps)")
    ax.set_ylabel("Portfolio Value ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"ml_{safe}.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {output_dir / f'ml_{safe}.png'}")

print(f"\n{'=' * 70}")
print("  ML RESEARCH COMPLETE")
print(f"{'=' * 70}")
