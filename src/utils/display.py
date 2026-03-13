"""
Logging and display helpers.
"""

from __future__ import annotations

import pandas as pd


def print_summary(summary: dict, title: str = "Backtest Summary") -> None:
    """Pretty-print a backtest summary dict."""
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")
    fmt = {
        "total_return": "{:.2%}",
        "benchmark_return": "{:.2%}",
        "cagr": "{:.2%}",
        "annual_volatility": "{:.2%}",
        "sharpe_ratio": "{:.2f}",
        "max_drawdown": "{:.2%}",
        "annualized_alpha": "{:.4f}",
        "information_ratio": "{:.2f}",
        "n_trades": "{:,}",
        "n_days": "{:,}",
    }
    for key, val in summary.items():
        f = fmt.get(key, "{}")
        print(f"  {key:>25s}:  {f.format(val)}")
    print(f"{'=' * 50}\n")
