# Project Notes (AGENTS.md)

## Build / Test Commands
- **Activate venv**: `source .venv/Scripts/activate` (Windows/Git Bash) or `source .venv/bin/activate` (Linux/Mac)
- **Install deps**: `pip install -r requirements.txt`
- **Smoke test**: `python -m tests.smoke_test`

## Architecture
- All strategies implement `generate_signals(date, universe, lookback) -> dict[str, float]`
- Price data is cached as parquet in `data/cache/` to avoid redundant API calls
- `src/backtest/engine.py` is the core loop: fetch data -> walk forward -> rebalance -> compute equity
- Analysis modules in `src/analysis/` are standalone and can be used outside the backtest engine
