"""
Buy-and-hold benchmark strategy.

Useful as a control: equal-weight the universe and never rebalance
(weights drift with prices).
"""

from __future__ import annotations

import pandas as pd


class BuyAndHoldStrategy:
    """Equal-weight buy-and-hold across the universe."""

    def __init__(self, tickers: list[str] | None = None):
        self._tickers = tickers
        self._initialized = False
        self._weights: dict[str, float] = {}

    def generate_signals(
        self,
        date: pd.Timestamp,
        universe: list[str],
        lookback: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        if not self._initialized:
            tickers = self._tickers or universe
            n = len(tickers)
            self._weights = {t: 1.0 / n for t in tickers}
            self._initialized = True
        return dict(self._weights)
