"""
Analyst-rating strategy using point-in-time consensus.

Reconstructs historical analyst consensus from individual upgrade/downgrade
events.  At each rebalance, it looks up what the consensus was *as of that
date* -- no look-ahead bias.

Scoring: each ticker gets a weighted average score from its active analyst
ratings (strongBuy=+2, buy=+1, hold=0, sell=-1, strongSell=-2).
The strategy goes long the top-N scoring names.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.data.ratings import build_consensus_history, consensus_at_date, BUCKET_SCORES


@dataclass
class AnalystRatingStrategy:
    """Long/short strategy driven by point-in-time analyst consensus.

    Pre-fetches and caches historical consensus for each ticker in the
    universe at construction time.  At each rebalance, scores are looked
    up as of that date.

    Parameters
    ----------
    consensus_cache : dict
        Pre-built consensus history DataFrames keyed by ticker.
        Built by calling build_consensus_history() for each ticker.
    long_only : bool
        If True, zero out negative weights (no shorting).
    top_n : int | None
        If set, only hold the top-N scoring names.
    min_analysts : int
        Minimum number of active analysts required to include a ticker.
    """
    consensus_cache: dict[str, pd.DataFrame] = field(default_factory=dict)
    long_only: bool = True
    top_n: int | None = 10
    min_analysts: int = 3

    @classmethod
    def from_universe(cls, tickers: list[str], **kwargs) -> "AnalystRatingStrategy":
        """Build the strategy by pre-fetching consensus history for all tickers."""
        cache = {}
        for t in tickers:
            cache[t] = build_consensus_history(t)
        return cls(consensus_cache=cache, **kwargs)

    def generate_signals(
        self,
        date: pd.Timestamp,
        universe: list[str],
        lookback: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        """Score each ticker by its point-in-time analyst consensus."""
        scores: dict[str, float] = {}

        for ticker in universe:
            history = self.consensus_cache.get(ticker)
            if history is None:
                raise ValueError(f"No consensus history for {ticker} -- not in cache")

            try:
                info = consensus_at_date(history, date)
            except ValueError:
                # No analyst data available before this date for this ticker
                # This is legitimate for early dates -- skip the ticker
                continue

            if info["n_analysts"] < self.min_analysts:
                continue

            scores[ticker] = info["score"]

        if not scores:
            raise ValueError(
                f"No tickers had sufficient analyst data on {date}"
            )

        if self.long_only:
            scores = {t: max(s, 0.0) for t, s in scores.items()}

        if self.top_n is not None:
            sorted_tickers = sorted(scores, key=scores.get, reverse=True)
            keep = set(sorted_tickers[: self.top_n])
            scores = {t: s for t, s in scores.items() if t in keep}

        total = sum(abs(v) for v in scores.values())
        if total == 0:
            raise ValueError(
                "All scores are zero after filtering -- no tradeable signal"
            )
        weights = {t: s / total for t, s in scores.items() if abs(s) > 1e-9}

        return weights
