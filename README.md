# strat-testing

A trading strategy backtesting sandbox. Started as infrastructure for testing analyst-rating signals and evolved into a systematic exploration of what works, what doesn't, and why.

## The short version

After testing ~20 strategy variants across 20 years of data, the finding is: **there is no alpha, but there are robust risk premia.** The only strategies that survive three-period out-of-sample validation are well-known academic factors (cross-sectional momentum, trend following) implemented on liquid ETFs. A simple 50/50 blend of these two delivered 0.60-0.67 Sharpe with -19% to -25% max drawdown across 2005-2013, 2014-2019, and 2020-2026 -- at ~$1K/year in transaction costs on a $100K account.

## What broke (and what we learned)

This project went through four major iterations. Each one exposed errors in the previous version:

### Iteration 1: "Everything shows significant alpha"
- **Bug**: engine applied fixed target weights daily (implicit daily rebalancing)
- **Bug**: Buy & Hold returned same weights every call (constant-mix, not buy-and-hold)
- **Bug**: analyst ratings used today's consensus applied retroactively (look-ahead bias)
- **Bug**: 25-stock universe was hand-picked mega-cap winners (survivorship bias)
- **Result**: all strategies showed "statistically significant alpha." This was fake.

### Iteration 2: Fixed engine, honest results
- Share-count-based position tracking with natural drift between rebalances
- Transaction costs (10bps per trade on dollar value traded)
- Point-in-time analyst consensus reconstructed from upgrade/downgrade events
- 100-stock random sample from S&P 500
- **Result**: no strategy showed significant alpha. Buy & Hold underperformed SPY.

### Iteration 3: Research strategies (mean reversion, breakout, pullback, swing)
- All underperformed SPY after costs
- Transaction costs dominated at weekly frequency ($20-93K on $100K)
- Regime router (balanced bagger) achieved lowest drawdown but no alpha
- **Key insight**: the `{}` vs `None` return semantics caused intermittent strategies to whipsaw between fully invested and 100% cash

### Iteration 4: Factor strategies on ETFs
- Academic momentum signals (Jegadeesh-Titman 12-1, Moskowitz trend following)
- Multi-asset universe (equities, bonds, gold) -- liquid, no single-stock risk
- 3bps costs (realistic for ETFs)
- **Three-period validation: all strategies positive Sharpe across 2005-2013, 2014-2019, 2020-2026**
- XS Momentum + Trend Following blend: the most robust result

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows/Git Bash
# source .venv/bin/activate    # macOS/Linux
pip install -r requirements.txt
```

## Running

```bash
# The definitive result: blend strategy across three periods
python run_blend.py

# Factor research (all 7 strategies, three periods)
python run_factors.py

# Stock-level research (mean reversion, breakout, etc.)
python run_research.py

# Original strategies (analyst ratings, momentum, buy & hold)
python run_backtest.py

# Smoke tests
python -m tests.smoke_test
```

## Project structure

```
src/
  backtest/
    engine.py          # Walk-forward backtester (share-count tracking,
                       # transaction costs, rebalance threshold filter)
    metrics.py         # Sharpe, Sortino, Calmar, alpha/beta, plotting
  data/
    prices.py          # yfinance OHLCV fetcher with parquet cache
    ratings.py         # Point-in-time analyst consensus reconstruction
    session.py         # SSL/proxy session handler
    universe.py        # S&P 500 ticker lists (with survivorship bias note)
  strategies/
    factors/           # ETF factor strategies (the robust findings)
      trend_following.py    # Time-series momentum, vol-scaled
      cross_sectional.py    # Cross-sectional momentum (12-1)
      momentum_quality.py   # Momentum + quality (AQR-style)
      cross_asset.py        # Cross-asset momentum
      blend.py              # Static blend of sub-strategies
      universes.py          # Multi-asset and equity ETF universes
    analyst_ratings.py # PIT consensus from upgrade/downgrade events
    momentum.py        # Simple trailing-return momentum
    mean_reversion.py  # Z-score buy-oversold
    breakout.py        # Donchian channel + volume
    pullback.py        # Buy dips in uptrends
    swing.py           # Stochastic/MACD crossovers
    buy_and_hold.py    # True buy-and-hold (allocate once, never rebalance)
    composite.py       # Multi-signal composites
    adaptive.py        # Vol-regime parameter adaptation
    sector.py          # Sector ETF rotation
    regime_router.py   # Signal-driven strategy ensemble
  analysis/
    alpha.py           # t-test, bootstrap CI, rolling alpha, IC
    anomaly.py         # Z-score, Isolation Forest, volume spikes
    priced_in.py       # Event study, surprise regression
  utils/
    display.py         # Pretty-print helpers
```

## Key results

### Blend: XS Momentum + Trend Following (50/50)

All prices are dividend-adjusted (total return). Costs at 3bps, 2% rebalance threshold.

| Period | Return | CAGR | Sharpe | Max DD | Costs | SPY |
|---|---|---|---|---|---|---|
| 2005-2013 (GFC) | +126% | 9.5% | 0.84 | -21% | $1,099 | +83% |
| 2014-2019 | +57% | 7.8% | 0.78 | -19% | $839 | +97% |
| 2020-2026 | +75% | 9.5% | 0.69 | -24% | $1,033 | +124% |

The blend underperforms SPY in the two bull-market periods (2014-2019, 2020-2026) because it holds bonds and gold via the trend component. It outperforms in 2005-2013 because trend following exits equities during the 2008 crash. The max drawdown (-19% to -25%) is far shallower than SPY (-55% in 2008).

### What strategies didn't work

| Strategy | Problem |
|---|---|
| Stock-level momentum (60d) | High Sharpe but not significant alpha; driven by beta exposure |
| Mean reversion | Transaction costs destroy the edge at weekly frequency |
| Breakout (Donchian) | Significant *negative* alpha -- false signals on individual stocks |
| Pullback + sentiment | Crowded-trade amplification; falling knives |
| Adaptive parameters | Vol-regime signal too noisy; parameter shifts add lag |
| Regime router | Diversifies drawdowns but blends losing strategies |

## Engine design

The backtesting engine (`src/backtest/engine.py`) uses:
- **Share-count tracking**: positions are held as fractional shares, drifting with prices between rebalances
- **Transaction costs**: proportional (bps on dollar value) + per-share (commission)
- **Rebalance threshold**: skip trades where the weight change is below a configurable minimum (reduces churn)
- **Carry-forward pricing**: last known price used for non-trading days
- **`None` vs `{}`**: strategies return `None` to keep existing positions, `{}` to go to cash

## Lessons

1. **Test out of sample before adding complexity.** The stock-level research produced increasingly complex strategies (composites, adaptive params, regime routing) without first confirming the base signal was robust. The ETF factor strategies -- much simpler -- were the first thing that survived multi-period validation.

2. **Transaction costs are a strategy.** At weekly rebalancing with 10bps costs, even a positive signal gets eaten. Monthly rebalancing with 3bps on liquid ETFs makes costs negligible. The choice of instrument and frequency matters more than the signal.

3. **Survivorship and look-ahead bias are easy to introduce and hard to detect.** The original analyst-rating strategy showed "significant alpha" because it used today's ratings retroactively. The original universe showed "alpha" because it was hand-picked winners. Both passed statistical tests. The only detection method that worked was asking "does Buy & Hold on this universe also show alpha?" -- if the benchmark portfolio beats the benchmark, the universe is biased.

4. **Robustness > Sharpe.** The blend has a moderate Sharpe (0.60-0.67) but a 0.07 spread across three very different market regimes. The stock-level momentum strategy had a higher Sharpe (0.88) in one period but was never tested out of sample. The blend is the one you could actually run.
