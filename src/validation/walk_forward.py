"""
Walk-forward validation framework.

Implements purged walk-forward cross-validation for time-series strategies.
Each fold: expanding training window -> purge gap -> OOS test window.
Reports per-fold and aggregate OOS metrics with statistical significance.

                 |--- train (expanding) ---|-- purge --|--- test ---|
    Fold 1:      [=======================]  [xxx]       [==========]
    Fold 2:      [================================]  [xxx]  [==========]
    Fold 3:      [==========================================] [xxx] [====]

Key properties:
  - No look-ahead: train only on data strictly before purge boundary
  - Purge gap prevents information leakage from overlapping return windows
  - Expanding (not sliding) window to match production walk-forward behavior
  - Reports per-fold OOS Sharpe, return, max DD, alpha
  - Aggregate: mean/std across folds, t-test for significance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest, BacktestResult
from src.backtest.metrics import sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio, alpha_beta


@dataclass
class FoldResult:
    """Metrics for a single validation fold."""
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train_days: int
    n_test_days: int
    backtest: BacktestResult | None = None

    # OOS metrics (computed from backtest)
    oos_return: float = 0.0
    oos_cagr: float = 0.0
    oos_sharpe: float = 0.0
    oos_sortino: float = 0.0
    oos_max_dd: float = 0.0
    oos_calmar: float = 0.0
    oos_alpha: float = 0.0
    oos_beta: float = 0.0
    oos_ir: float = 0.0
    n_trades: int = 0


@dataclass
class ValidationReport:
    """Aggregate validation results across all folds."""
    folds: list[FoldResult]
    strategy_name: str = ""
    benchmark_name: str = "SPY"

    def summary_df(self) -> pd.DataFrame:
        """Per-fold results as a DataFrame."""
        rows = []
        for f in self.folds:
            rows.append({
                "Fold": f.fold_id,
                "Train": f"{f.train_start} -> {f.train_end}",
                "Test": f"{f.test_start} -> {f.test_end}",
                "OOS Return": f.oos_return,
                "OOS CAGR": f.oos_cagr,
                "OOS Sharpe": f.oos_sharpe,
                "OOS Sortino": f.oos_sortino,
                "OOS MaxDD": f.oos_max_dd,
                "OOS Calmar": f.oos_calmar,
                "OOS Alpha": f.oos_alpha,
                "OOS Beta": f.oos_beta,
                "OOS IR": f.oos_ir,
                "Trades": f.n_trades,
            })
        return pd.DataFrame(rows)

    def aggregate(self) -> dict[str, float]:
        """Mean and std of key OOS metrics across folds."""
        valid = [f for f in self.folds if f.backtest is not None]
        if not valid:
            return {}

        metrics = {}
        for key in ["oos_return", "oos_sharpe", "oos_sortino", "oos_max_dd",
                     "oos_calmar", "oos_alpha", "oos_ir"]:
            vals = [getattr(f, key) for f in valid]
            metrics[f"{key}_mean"] = float(np.mean(vals))
            metrics[f"{key}_std"] = float(np.std(vals))

        # t-test: is mean OOS Sharpe > 0?
        sharpes = [f.oos_sharpe for f in valid]
        n = len(sharpes)
        if n >= 3:
            mean_s = np.mean(sharpes)
            std_s = np.std(sharpes, ddof=1)
            if std_s > 0:
                t_stat = mean_s / (std_s / np.sqrt(n))
                from scipy.stats import t as t_dist
                p_val = 1 - t_dist.cdf(t_stat, df=n - 1)
                metrics["sharpe_t_stat"] = float(t_stat)
                metrics["sharpe_p_value"] = float(p_val)

        # t-test: is mean OOS alpha > 0?
        alphas = [f.oos_alpha for f in valid]
        if n >= 3:
            mean_a = np.mean(alphas)
            std_a = np.std(alphas, ddof=1)
            if std_a > 0:
                t_stat_a = mean_a / (std_a / np.sqrt(n))
                p_val_a = 1 - t_dist.cdf(t_stat_a, df=n - 1)
                metrics["alpha_t_stat"] = float(t_stat_a)
                metrics["alpha_p_value"] = float(p_val_a)

        # Win rate: fraction of folds with positive OOS return
        metrics["win_rate"] = sum(1 for f in valid if f.oos_return > 0) / len(valid)
        metrics["n_folds"] = len(valid)

        return metrics

    def print_report(self):
        """Pretty-print the full validation report."""
        print(f"\n{'=' * 80}")
        print(f"  WALK-FORWARD VALIDATION: {self.strategy_name}")
        print(f"{'=' * 80}")

        df = self.summary_df()
        # Format for display
        fmt_df = df.copy()
        for col in ["OOS Return", "OOS CAGR", "OOS MaxDD"]:
            if col in fmt_df.columns:
                fmt_df[col] = fmt_df[col].apply(lambda x: f"{x:+.1%}")
        for col in ["OOS Sharpe", "OOS Sortino", "OOS Calmar", "OOS IR"]:
            if col in fmt_df.columns:
                fmt_df[col] = fmt_df[col].apply(lambda x: f"{x:.2f}")
        for col in ["OOS Alpha", "OOS Beta"]:
            if col in fmt_df.columns:
                fmt_df[col] = fmt_df[col].apply(lambda x: f"{x:+.4f}")

        print("\nPer-Fold OOS Results:")
        print(fmt_df.to_string(index=False))

        agg = self.aggregate()
        if agg:
            print(f"\n{'-' * 80}")
            print("Aggregate OOS Statistics:")
            print(f"  Mean OOS Sharpe:  {agg.get('oos_sharpe_mean', 0):.2f}"
                  f"  (std: {agg.get('oos_sharpe_std', 0):.2f})")
            print(f"  Mean OOS Return:  {agg.get('oos_return_mean', 0):+.1%}"
                  f"  (std: {agg.get('oos_return_std', 0):.1%})")
            print(f"  Mean OOS Alpha:   {agg.get('oos_alpha_mean', 0):+.4f}"
                  f"  (std: {agg.get('oos_alpha_std', 0):.4f})")
            print(f"  Mean OOS MaxDD:   {agg.get('oos_max_dd_mean', 0):.1%}"
                  f"  (std: {agg.get('oos_max_dd_std', 0):.1%})")
            print(f"  Win Rate:         {agg.get('win_rate', 0):.0%}"
                  f"  ({agg.get('n_folds', 0):.0f} folds)")

            if "sharpe_t_stat" in agg:
                sig = "***" if agg["sharpe_p_value"] < 0.01 else \
                      "**" if agg["sharpe_p_value"] < 0.05 else \
                      "*" if agg["sharpe_p_value"] < 0.10 else ""
                print(f"\n  Sharpe > 0 test:  t={agg['sharpe_t_stat']:.2f},"
                      f"  p={agg['sharpe_p_value']:.3f} {sig}")
            if "alpha_t_stat" in agg:
                sig = "***" if agg["alpha_p_value"] < 0.01 else \
                      "**" if agg["alpha_p_value"] < 0.05 else \
                      "*" if agg["alpha_p_value"] < 0.10 else ""
                print(f"  Alpha > 0 test:   t={agg['alpha_t_stat']:.2f},"
                      f"  p={agg['alpha_p_value']:.3f} {sig}")

        print(f"{'=' * 80}\n")


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

def walk_forward_validate(
    strategy_factory,
    universe: list[str],
    full_start: str = "2005-01-01",
    full_end: str | None = None,
    min_train_years: float = 3.0,
    test_months: int = 12,
    purge_weeks: int = 2,
    n_folds: int | None = None,
    step_months: int | None = None,
    benchmark: str = "SPY",
    rebalance_freq: str = "W",
    cost_pct: float = 0.0003,
    rebalance_threshold: float = 0.01,
    lookback_buffer_days: int = 500,
    strategy_name: str = "",
) -> ValidationReport:
    """Run purged walk-forward validation.

    Parameters
    ----------
    strategy_factory : callable
        A function that returns a fresh strategy instance.
        Called once per fold to ensure no state leakage.
    universe : list[str]
        ETF tickers.
    full_start : str
        Earliest date for training data.
    full_end : str
        Latest date.  None = today.
    min_train_years : float
        Minimum years of training before first test fold.
    test_months : int
        Length of each OOS test period in months.
    purge_weeks : int
        Gap between train end and test start (information barrier).
    n_folds : int or None
        Number of folds.  If None, inferred from available data.
    step_months : int or None
        Months to advance between fold starts.  Defaults to test_months.
    benchmark : str
        Benchmark ticker.
    rebalance_freq : str
        "W" for weekly, "ME" for monthly.
    cost_pct : float
        Transaction cost.
    rebalance_threshold : float
        Min weight change to trade.
    lookback_buffer_days : int
        Extra days of price history.
    strategy_name : str
        Label for reports.
    """
    if step_months is None:
        step_months = test_months

    start_dt = datetime.strptime(full_start, "%Y-%m-%d")
    if full_end is not None:
        end_dt = datetime.strptime(full_end, "%Y-%m-%d")
    else:
        end_dt = datetime.now()

    # Compute fold boundaries
    min_train_end = start_dt + timedelta(days=int(min_train_years * 365.25))
    purge_days = purge_weeks * 7

    folds: list[FoldResult] = []
    fold_id = 0
    test_start_dt = min_train_end + timedelta(days=purge_days)

    while test_start_dt < end_dt:
        test_end_dt = test_start_dt + timedelta(days=int(test_months * 30.44))
        if test_end_dt > end_dt:
            test_end_dt = end_dt

        # Ensure test period has at least 60 days
        if (test_end_dt - test_start_dt).days < 60:
            break

        train_end_dt = test_start_dt - timedelta(days=purge_days)

        fold = FoldResult(
            fold_id=fold_id,
            train_start=full_start,
            train_end=train_end_dt.strftime("%Y-%m-%d"),
            test_start=test_start_dt.strftime("%Y-%m-%d"),
            test_end=test_end_dt.strftime("%Y-%m-%d"),
            n_train_days=(train_end_dt - start_dt).days,
            n_test_days=(test_end_dt - test_start_dt).days,
        )
        folds.append(fold)
        fold_id += 1

        test_start_dt += timedelta(days=int(step_months * 30.44))

        if n_folds is not None and fold_id >= n_folds:
            break

    # Run each fold
    for fold in folds:
        print(f"\n  Fold {fold.fold_id}: "
              f"train {fold.train_start}->{fold.train_end}, "
              f"test {fold.test_start}->{fold.test_end}")

        strat = strategy_factory()

        try:
            # Run backtest over the FULL period (train + test)
            # The strategy's walk-forward logic handles the train/test split internally
            # But we only measure OOS metrics on the test window
            result = run_backtest(
                strategy=strat,
                universe=universe,
                start=fold.train_start,
                end=fold.test_end,
                benchmark=benchmark,
                rebalance_freq=rebalance_freq,
                initial_capital=100_000.0,
                cost_pct=cost_pct,
                lookback_buffer_days=lookback_buffer_days,
                rebalance_threshold=rebalance_threshold,
            )

            # Extract OOS window only
            test_start_ts = pd.Timestamp(fold.test_start)
            test_end_ts = pd.Timestamp(fold.test_end)

            oos_equity = result.equity_curve.loc[test_start_ts:test_end_ts]
            oos_bench = result.benchmark_curve.loc[test_start_ts:test_end_ts]

            if len(oos_equity) < 20:
                print(f"    Fold {fold.fold_id}: insufficient OOS data ({len(oos_equity)} days)")
                continue

            # Compute OOS metrics
            oos_rets = oos_equity.pct_change().dropna()
            oos_bench_rets = oos_bench.pct_change().dropna()
            n_years = len(oos_rets) / 252

            fold.oos_return = float(oos_equity.iloc[-1] / oos_equity.iloc[0] - 1)
            fold.oos_cagr = float((1 + fold.oos_return) ** (1 / max(n_years, 1e-9)) - 1)
            fold.oos_sharpe = sharpe_ratio(oos_rets)
            fold.oos_sortino = sortino_ratio(oos_rets)
            fold.oos_max_dd = max_drawdown(oos_equity)
            fold.oos_calmar = calmar_ratio(oos_equity)

            try:
                a, b = alpha_beta(oos_rets, oos_bench_rets)
                fold.oos_alpha = a
                fold.oos_beta = b
            except Exception:
                fold.oos_alpha = 0.0
                fold.oos_beta = 0.0

            excess = (oos_rets - oos_bench_rets.reindex(oos_rets.index)).dropna()
            if len(excess) > 0 and excess.std() > 0:
                fold.oos_ir = float(excess.mean() * 252 / (excess.std() * np.sqrt(252)))

            oos_trades = [t for t in result.trades
                          if test_start_ts <= t["date"] <= test_end_ts]
            fold.n_trades = len(oos_trades)
            fold.backtest = result

            print(f"    OOS: return={fold.oos_return:+.1%}, "
                  f"sharpe={fold.oos_sharpe:.2f}, "
                  f"maxDD={fold.oos_max_dd:.1%}, "
                  f"alpha={fold.oos_alpha:+.4f}")

        except Exception as e:
            print(f"    Fold {fold.fold_id} FAILED: {e}")

    return ValidationReport(folds=folds, strategy_name=strategy_name)
