"""
Buy-and-hold benchmark strategy.

Allocates equal weight on the first rebalance, then returns None on
all subsequent calls so the engine never rebalances.  Positions drift
with prices -- winners grow, losers shrink -- which is what buy-and-hold
actually means.
"""

from __future__ import annotations

import pandas as pd


class BuyAndHoldStrategy:
    """Equal-weight buy-and-hold across the universe.

    Allocates once, then holds.  No rebalancing.
    """

    def __init__(self, tickers: list[str] | None = None):
        self._tickers = tickers
        self._initialized = False

    def generate_signals(
        self,
        date: pd.Timestamp,
        universe: list[str],
        lookback: dict[str, pd.DataFrame],
    ) -> dict[str, float] | None:
        if not self._initialized:
            tickers = self._tickers or universe
            n = len(tickers)
            self._initialized = True
            return {t: 1.0 / n for t in tickers}
        # After initial allocation, never rebalance
        return None
