"""
Performance metrics and reporting.

Standalone functions that operate on return series, useful outside the
backtesting engine too (e.g. quick analysis in notebooks).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# Risk-free rate helper
# ---------------------------------------------------------------------------

_rf_rate_cache: dict[str, float] = {}


def get_risk_free_rate() -> float:
    """Fetch annualized risk-free rate from 13-week T-bill (^IRX).

    Returns the most recent yield as a decimal (e.g. 0.05 for 5%).
    Cached per session.  Falls back to 0.0 if the fetch fails.
    """
    if "rate" not in _rf_rate_cache:
        try:
            import yfinance as yf
            from src.data.session import get_session
            irx = yf.download(
                "^IRX", period="5d", progress=False, session=get_session(),
            )
            if isinstance(irx.columns, pd.MultiIndex):
                irx.columns = irx.columns.get_level_values(0)
            if not irx.empty:
                _rf_rate_cache["rate"] = float(irx["Close"].dropna().iloc[-1]) / 100.0
            else:
                _rf_rate_cache["rate"] = 0.0
        except Exception:
            _rf_rate_cache["rate"] = 0.0
    return _rf_rate_cache["rate"]


def sharpe_ratio(returns: pd.Series, risk_free: float | None = None, periods: int = 252) -> float:
    if risk_free is None:
        risk_free = get_risk_free_rate()
    excess = returns - risk_free / periods
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(periods))


def sortino_ratio(returns: pd.Series, risk_free: float | None = None, periods: int = 252) -> float:
    if risk_free is None:
        risk_free = get_risk_free_rate()
    excess = returns - risk_free / periods
    downside = excess[excess < 0]
    if downside.std() == 0:
        return 0.0
    return float(excess.mean() / downside.std() * np.sqrt(periods))


def max_drawdown(equity_curve: pd.Series) -> float:
    rolling_max = equity_curve.cummax()
    dd = (equity_curve - rolling_max) / rolling_max
    return float(dd.min())


def calmar_ratio(equity_curve: pd.Series, periods: int = 252) -> float:
    returns = equity_curve.pct_change().dropna()
    ann_ret = returns.mean() * periods
    mdd = abs(max_drawdown(equity_curve))
    return float(ann_ret / mdd) if mdd > 0 else 0.0


def rolling_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Rolling beta vs benchmark."""
    aligned = pd.DataFrame({"s": strategy_returns, "b": benchmark_returns}).dropna()
    cov = aligned["s"].rolling(window).cov(aligned["b"])
    var = aligned["b"].rolling(window).var()
    return (cov / var).rename("rolling_beta")


def alpha_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> tuple[float, float]:
    """OLS alpha (annualized) and beta."""
    aligned = pd.DataFrame({"s": strategy_returns, "b": benchmark_returns}).dropna()
    if len(aligned) < 2:
        raise ValueError(
            f"Insufficient aligned data for alpha/beta: need >= 2, got {len(aligned)}"
        )
    cov_matrix = np.cov(aligned["s"], aligned["b"])
    beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 0.0
    alpha = (aligned["s"].mean() - beta * aligned["b"].mean()) * 252
    return float(alpha), float(beta)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_equity(result, figsize=(14, 6)):
    """Plot strategy equity curve vs benchmark."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(result.equity_curve.index, result.equity_curve.values, label="Strategy", linewidth=1.5)
    ax.plot(result.benchmark_curve.index, result.benchmark_curve.values, label="Benchmark", linewidth=1.2, alpha=0.7)
    ax.set_title("Equity Curve")
    ax.set_ylabel("Portfolio Value ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_drawdown(result, figsize=(14, 4)):
    """Plot drawdown chart."""
    eq = result.equity_curve
    rolling_max = eq.cummax()
    dd = (eq - rolling_max) / rolling_max

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(dd.index, dd.values, 0, alpha=0.4, color="red", label="Drawdown")
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_monthly_returns(result, figsize=(12, 6)):
    """Heatmap of monthly returns."""
    import seaborn as sns

    returns = result.returns
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    table = monthly.groupby([monthly.index.year, monthly.index.month]).first().unstack()
    table.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(table, annot=True, fmt=".1%", center=0, cmap="RdYlGn", ax=ax)
    ax.set_title("Monthly Returns")
    fig.tight_layout()
    return fig
