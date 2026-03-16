"""
Momentum strategy (example of a price-based signal).

Goes long the top-N tickers ranked by trailing return over *lookback_days*.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class MomentumStrategy:
    lookback_days: int = 60
    top_n: int = 10
    long_only: bool = True

    def generate_signals(
        self,
        date: pd.Timestamp,
        universe: list[str],
        lookback: dict[str, pd.DataFrame],
    ) -> dict[str, float] | None:
        scores: dict[str, float] = {}

        for ticker in universe:
            df = lookback.get(ticker)
            if df is None or len(df) < self.lookback_days:
                continue
            recent = df["Close"].iloc[-self.lookback_days :]
            ret = recent.iloc[-1] / recent.iloc[0] - 1
            scores[ticker] = float(ret)

        if not scores:
            return None  # insufficient data -- keep existing positions

        if self.long_only:
            scores = {t: max(s, 0.0) for t, s in scores.items()}

        sorted_tickers = sorted(scores, key=scores.get, reverse=True)
        keep = set(sorted_tickers[: self.top_n])
        scores = {t: s for t, s in scores.items() if t in keep}

        total = sum(abs(v) for v in scores.values())
        if total == 0:
            return {}  # all scores zero -- go to cash
        weights = {t: s / total for t, s in scores.items() if abs(s) > 1e-9}

        return weights
