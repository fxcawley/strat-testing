"""
Analyst-rating strategy.

Core idea: go long stocks that have consensus strong-buy / buy ratings,
underweight or short those with sell / strong-sell.  Rebalance on the
chosen cadence (default monthly).

This is deliberately simple so it serves as both a real signal to test
and a template for more sophisticated strategies.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.ratings import current_consensus


# Score map: maps consensus label -> portfolio tilt
DEFAULT_SCORE_MAP = {
    "strongBuy": 2.0,
    "buy": 1.0,
    "hold": 0.0,
    "sell": -1.0,
    "strongSell": -2.0,
}


@dataclass
class AnalystRatingStrategy:
    """Long/short strategy driven by analyst consensus ratings.

    Parameters
    ----------
    score_map : dict
        Maps consensus label to a raw score.  Positive = long, negative = short.
    long_only : bool
        If True, zero out negative weights (no shorting).
    top_n : int | None
        If set, only hold the top-N scoring names.
    """
    score_map: dict[str, float] | None = None
    long_only: bool = True
    top_n: int | None = 10

    def __post_init__(self):
        if self.score_map is None:
            self.score_map = dict(DEFAULT_SCORE_MAP)

    def generate_signals(
        self,
        date: pd.Timestamp,
        universe: list[str],
        lookback: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        """Score each ticker by analyst consensus, return target weights."""
        scores: dict[str, float] = {}

        for ticker in universe:
            try:
                info = current_consensus(ticker)
                label = info["consensus"]
                scores[ticker] = self.score_map.get(label, 0.0)
            except Exception:
                scores[ticker] = 0.0

        if self.long_only:
            scores = {t: max(s, 0.0) for t, s in scores.items()}

        # Keep top N
        if self.top_n is not None:
            sorted_tickers = sorted(scores, key=scores.get, reverse=True)
            keep = set(sorted_tickers[: self.top_n])
            scores = {t: s for t, s in scores.items() if t in keep}

        # Normalize to sum to 1.0 (equal-ish risk)
        total = sum(abs(v) for v in scores.values())
        if total > 0:
            weights = {t: s / total for t, s in scores.items() if abs(s) > 1e-9}
        else:
            weights = {}

        return weights
