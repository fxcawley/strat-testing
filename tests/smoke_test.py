"""
Smoke test -- verify all imports resolve and core computations work.

Run: python -m tests.smoke_test
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_imports():
    print("Testing imports...")
    from src.data.prices import fetch_prices, fetch_benchmark, BENCHMARKS
    from src.data.ratings import (
        fetch_upgrades_downgrades, build_consensus_history,
        consensus_at_date, current_consensus, GRADE_MAP, BUCKET_SCORES,
    )
    from src.data.universe import SP500_CURRENT, SECTOR_ETFS
    from src.backtest.engine import run_backtest, BacktestResult
    from src.backtest.metrics import sharpe_ratio, max_drawdown, alpha_beta
    from src.strategies.analyst_ratings import AnalystRatingStrategy
    from src.strategies.buy_and_hold import BuyAndHoldStrategy
    from src.strategies.momentum import MomentumStrategy
    from src.analysis.alpha import t_test_alpha, bootstrap_alpha, information_coefficient, rolling_alpha
    from src.analysis.anomaly import zscore_anomalies, isolation_forest_anomalies, build_anomaly_features
    from src.analysis.priced_in import event_study, priced_in_score, surprise_regression
    from src.utils.display import print_summary
    print("  All imports OK.")


def test_metrics():
    """Verify metrics functions on synthetic data."""
    import numpy as np
    import pandas as pd
    from src.backtest.metrics import sharpe_ratio, max_drawdown, sortino_ratio, calmar_ratio, alpha_beta

    print("Testing metrics on synthetic data...")
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0005, 0.01, 500))
    equity = (1 + returns).cumprod() * 100_000

    sr = sharpe_ratio(returns)
    assert isinstance(sr, float)
    print(f"  Sharpe ratio: {sr:.2f}")

    mdd = max_drawdown(equity)
    assert mdd <= 0
    print(f"  Max drawdown: {mdd:.2%}")

    so = sortino_ratio(returns)
    assert isinstance(so, float)
    print(f"  Sortino ratio: {so:.2f}")

    cr = calmar_ratio(equity)
    assert isinstance(cr, float)
    print(f"  Calmar ratio: {cr:.2f}")

    bench_returns = pd.Series(rng.normal(0.0003, 0.01, 500))
    a, b = alpha_beta(returns, bench_returns)
    assert isinstance(a, float) and isinstance(b, float)
    print(f"  Alpha: {a:.4f}, Beta: {b:.4f}")

    print("  All metrics OK.")


def test_alpha_tools():
    """Verify alpha assessment on synthetic excess returns."""
    import numpy as np
    import pandas as pd
    from src.analysis.alpha import t_test_alpha, bootstrap_alpha, rolling_alpha

    print("Testing alpha assessment tools...")
    rng = np.random.default_rng(1)
    excess = pd.Series(rng.normal(0.0002, 0.005, 500))

    t_result = t_test_alpha(excess)
    assert "p_value" in t_result
    print(f"  t-test p-value: {t_result['p_value']:.4f}")

    boot = bootstrap_alpha(excess, n_bootstrap=1000)
    assert "ci_lower" in boot
    print(f"  Bootstrap CI: [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")

    strat_ret = pd.Series(rng.normal(0.0005, 0.01, 300))
    bench_ret = pd.Series(rng.normal(0.0003, 0.01, 300))
    ra = rolling_alpha(strat_ret, bench_ret, window=60)
    assert len(ra) == 300
    assert pd.notna(ra.iloc[-1])
    print(f"  Rolling alpha (last value): {ra.iloc[-1]:.4f}")

    print("  All alpha tools OK.")


def test_strict_failures():
    """Verify that functions raise instead of returning fallback values."""
    import pandas as pd
    from src.analysis.alpha import information_coefficient
    from src.backtest.metrics import alpha_beta as ab
    from src.analysis.priced_in import surprise_regression

    print("Testing strict failure modes...")

    try:
        information_coefficient(pd.Series([1, 2]), pd.Series([3, 4]))
        assert False, "Should have raised"
    except ValueError:
        pass
    print("  information_coefficient raises on insufficient data: OK")

    try:
        ab(pd.Series(dtype=float), pd.Series(dtype=float))
        assert False, "Should have raised"
    except ValueError:
        pass
    print("  alpha_beta raises on insufficient data: OK")

    try:
        surprise_regression(pd.Series([1, 2]), pd.Series([3, 4]))
        assert False, "Should have raised"
    except ValueError:
        pass
    print("  surprise_regression raises on insufficient data: OK")

    print("  All strict failure modes OK.")


def test_buy_and_hold_returns_none():
    """Verify Buy & Hold returns weights once, then None."""
    import pandas as pd
    from src.strategies.buy_and_hold import BuyAndHoldStrategy

    print("Testing Buy & Hold returns None after first call...")
    bh = BuyAndHoldStrategy()
    universe = ["A", "B", "C"]
    date = pd.Timestamp("2024-01-01")

    w1 = bh.generate_signals(date, universe, {})
    assert w1 is not None
    assert abs(sum(w1.values()) - 1.0) < 1e-9

    w2 = bh.generate_signals(date, universe, {})
    assert w2 is None

    print("  Buy & Hold: first call returns weights, second returns None: OK")


if __name__ == "__main__":
    test_imports()
    test_metrics()
    test_alpha_tools()
    test_strict_failures()
    test_buy_and_hold_returns_none()
    print("\n=== All smoke tests passed ===")
