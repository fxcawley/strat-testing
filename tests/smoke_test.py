"""
Quick smoke test -- verify all imports resolve and core types work.

Run: python -m tests.smoke_test
"""

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_imports():
    print("Testing imports...")
    from src.data.prices import fetch_prices, fetch_benchmark, BENCHMARKS
    from src.data.ratings import fetch_recommendations, current_consensus, screen_universe
    from src.data.universe import SP500_SAMPLE, SECTOR_ETFS
    from src.backtest.engine import run_backtest, BacktestResult
    from src.backtest.metrics import sharpe_ratio, max_drawdown, alpha_beta, plot_equity
    from src.strategies.analyst_ratings import AnalystRatingStrategy
    from src.strategies.buy_and_hold import BuyAndHoldStrategy
    from src.strategies.momentum import MomentumStrategy
    from src.analysis.alpha import t_test_alpha, bootstrap_alpha, information_coefficient
    from src.analysis.anomaly import zscore_anomalies, isolation_forest_anomalies, build_anomaly_features
    from src.analysis.priced_in import event_study, priced_in_score, surprise_regression
    from src.utils.display import print_summary
    print("  All imports OK.")


def test_strategy_interface():
    """Verify strategies conform to the expected interface."""
    import pandas as pd

    print("Testing strategy interfaces...")
    date = pd.Timestamp("2024-01-01")
    universe = ["AAPL", "MSFT", "GOOGL"]
    lookback = {}

    for StratClass in []:
        # Skip strategies that need live data; just verify instantiation
        pass

    from src.strategies.buy_and_hold import BuyAndHoldStrategy
    bh = BuyAndHoldStrategy()
    weights = bh.generate_signals(date, universe, lookback)
    assert isinstance(weights, dict)
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    print("  BuyAndHoldStrategy OK.")

    from src.strategies.momentum import MomentumStrategy
    mom = MomentumStrategy()
    # With empty lookback it should return empty weights
    weights = mom.generate_signals(date, universe, lookback)
    assert isinstance(weights, dict)
    print("  MomentumStrategy OK.")

    print("  All strategy interfaces OK.")


def test_metrics():
    """Verify metrics functions on synthetic data."""
    import numpy as np
    import pandas as pd
    from src.backtest.metrics import sharpe_ratio, max_drawdown, sortino_ratio, calmar_ratio

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

    print("  All metrics OK.")


def test_alpha_tools():
    """Verify alpha assessment on synthetic excess returns."""
    import numpy as np
    import pandas as pd
    from src.analysis.alpha import t_test_alpha, bootstrap_alpha

    print("Testing alpha assessment tools...")
    rng = np.random.default_rng(1)
    excess = pd.Series(rng.normal(0.0002, 0.005, 500))

    t_result = t_test_alpha(excess)
    assert "p_value" in t_result
    print(f"  t-test p-value: {t_result['p_value']:.4f}")

    boot = bootstrap_alpha(excess, n_bootstrap=1000)
    assert "ci_lower" in boot
    print(f"  Bootstrap CI: [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")

    print("  All alpha tools OK.")


if __name__ == "__main__":
    test_imports()
    test_strategy_interface()
    test_metrics()
    test_alpha_tools()
    print("\n=== All smoke tests passed ===")
