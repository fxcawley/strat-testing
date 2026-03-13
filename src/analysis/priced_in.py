"""
"Priced In" analysis module.

Attempt to quantify the degree to which known information (earnings,
macro events, analyst upgrades/downgrades) is already reflected in
the current price.

This is inherently tricky -- the scaffolding below provides a few
angles of attack that you can flesh out:

1. **Event study**: measure abnormal returns around known catalyst dates.
   If the move happens *before* the event, the market priced it in.

2. **Implied vol vs realized vol**: if IV is low heading into an event,
   the options market isn't expecting a surprise (i.e. it's priced in).

3. **Consensus drift**: track how analyst estimates drift into an event.
   If estimates converge to the actual number, the market had time to
   adjust.

4. **Price-surprise correlation**: regress post-event return on the
   magnitude of surprise.  A steep slope => market was NOT pricing it in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def event_study(
    prices: pd.Series,
    event_dates: list[pd.Timestamp],
    pre_window: int = 20,
    post_window: int = 5,
    benchmark: pd.Series | None = None,
) -> pd.DataFrame:
    """Run a simple event study.

    For each event date, compute the cumulative abnormal return (CAR)
    in the *pre_window* leading up to the event and the *post_window*
    after.

    If *benchmark* is given, abnormal return = stock return - benchmark return.
    Otherwise raw returns are used.

    Returns a DataFrame with columns:
        event_date, car_pre, car_post, total_car
    """
    returns = prices.pct_change()
    if benchmark is not None:
        bench_ret = benchmark.pct_change()
        abnormal = returns - bench_ret.reindex(returns.index, fill_value=0)
    else:
        abnormal = returns

    rows = []
    for edate in event_dates:
        if edate not in abnormal.index:
            # Find nearest trading day
            idx = abnormal.index.get_indexer([edate], method="ffill")[0]
            if idx < 0:
                continue
            edate = abnormal.index[idx]

        loc = abnormal.index.get_loc(edate)
        pre_start = max(0, loc - pre_window)
        post_end = min(len(abnormal), loc + post_window + 1)

        car_pre = abnormal.iloc[pre_start:loc].sum()
        car_post = abnormal.iloc[loc:post_end].sum()

        rows.append({
            "event_date": edate,
            "car_pre": float(car_pre),
            "car_post": float(car_post),
            "total_car": float(car_pre + car_post),
        })

    return pd.DataFrame(rows)


def priced_in_score(
    car_pre: float,
    car_post: float,
) -> float:
    """Heuristic: fraction of total absolute move that happened pre-event.

    Returns a value in [0, 1].  1 = fully priced in (all the move was pre-event).
    0 = total surprise (all the move was post-event).
    """
    total = abs(car_pre) + abs(car_post)
    if total < 1e-9:
        return 0.5  # no move at all -- ambiguous
    return abs(car_pre) / total


def surprise_regression(
    surprises: pd.Series,
    post_event_returns: pd.Series,
) -> dict:
    """Regress post-event returns on the surprise magnitude.

    A steep positive slope means the market is NOT pricing the event in.
    A flat slope means the market anticipated the magnitude correctly.
    """
    aligned = pd.DataFrame({"surprise": surprises, "ret": post_event_returns}).dropna()
    if len(aligned) < 5:
        return {"slope": 0.0, "r_squared": 0.0, "p_value": 1.0, "n": len(aligned)}

    slope, intercept, r, p, se = stats.linregress(aligned["surprise"], aligned["ret"])
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r ** 2),
        "p_value": float(p),
        "n": len(aligned),
    }
