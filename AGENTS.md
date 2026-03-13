# Project Notes (AGENTS.md)

## Build / Test Commands
- **Activate venv**: `source .venv/Scripts/activate` (Windows/Git Bash) or `source .venv/bin/activate` (Linux/Mac)
- **Install deps**: `pip install -r requirements.txt`
- **Smoke test**: `python -m tests.smoke_test`
- **Production backtest**: `python run_blend.py`
- **Clear cache**: `rm -f data/cache/*.parquet` (required after changing price fetch settings)

## Architecture
- All strategies implement `generate_signals(date, universe, lookback) -> dict[str, float] | None`
- Return `None` to keep existing positions; return `{}` to go to cash
- Price data uses `auto_adjust=True` (dividend-adjusted OHLCV) and is cached as parquet in `data/cache/`
- `src/backtest/engine.py` is the core loop: fetch data -> walk forward -> rebalance -> compute equity
- Engine tracks share counts (not weights) between rebalances -- positions drift with prices
- `rebalance_threshold` parameter skips trades below a weight-change minimum (default 0, set to 0.02 for production)
- Analysis modules in `src/analysis/` are standalone and can be used outside the backtest engine
