"""
Analyst-rating fetcher.

Provides both current consensus and historical point-in-time consensus
reconstructed from individual analyst upgrade/downgrade events.
"""

from __future__ import annotations

from pathlib import Path
import hashlib

import pandas as pd
import yfinance as yf

from src.data.session import get_session

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"

# Map the many brokerage-specific grade names to a 5-bucket scale.
# Unmapped grades are treated as "hold" (score 0).
GRADE_MAP = {
    # Strong buy
    "Strong Buy": "strongBuy",
    # Buy
    "Buy": "buy",
    "Outperform": "buy",
    "Overweight": "buy",
    "Positive": "buy",
    "Sector Outperform": "buy",
    "Long-Term Buy": "buy",
    "Top Pick": "buy",
    "Accumulate": "buy",
    "Add": "buy",
    # Hold / neutral
    "Hold": "hold",
    "Neutral": "hold",
    "Equal-Weight": "hold",
    "Market Perform": "hold",
    "Sector Perform": "hold",
    "Sector Weight": "hold",
    "Peer Perform": "hold",
    "Perform": "hold",
    "In-Line": "hold",
    "Mixed": "hold",
    "Fair Value": "hold",
    # Sell
    "Underperform": "sell",
    "Underweight": "sell",
    "Reduce": "sell",
    "Negative": "sell",
    "Sector Underperform": "sell",
    # Strong sell
    "Sell": "strongSell",
    "Strong Sell": "strongSell",
}

# Numeric scores for the standardized buckets
BUCKET_SCORES = {
    "strongBuy": 2.0,
    "buy": 1.0,
    "hold": 0.0,
    "sell": -1.0,
    "strongSell": -2.0,
}


def fetch_upgrades_downgrades(ticker: str) -> pd.DataFrame:
    """Fetch individual analyst upgrade/downgrade events for *ticker*.

    Returns a DataFrame indexed by GradeDate with columns:
        Firm, ToGrade, FromGrade, Action
    Sorted oldest-to-newest.
    """
    tk = yf.Ticker(ticker, session=get_session())
    ud = tk.upgrades_downgrades
    if ud is None or ud.empty:
        raise ValueError(f"No upgrade/downgrade data for {ticker}")
    # Sort oldest first for accumulation
    return ud.sort_index()


def build_consensus_history(
    ticker: str,
    stale_days: int = 365,
) -> pd.DataFrame:
    """Reconstruct point-in-time analyst consensus from upgrade/downgrade events.

    For each analyst action, we update that firm's current rating.  On any
    given date, the consensus is the distribution of active ratings across
    all firms that have issued a rating in the last *stale_days* days.

    Returns a DataFrame indexed by date with columns:
        strongBuy, buy, hold, sell, strongSell, consensus, score
    One row per date on which an analyst event occurred.
    """
    # Check cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.md5(f"consensus_{ticker}_{stale_days}".encode()).hexdigest()
    cache_path = CACHE_DIR / f"consensus_{ticker}_{cache_key}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    ud = fetch_upgrades_downgrades(ticker)

    # Track each firm's most recent rating and when it was issued
    firm_ratings: dict[str, tuple[str, pd.Timestamp]] = {}  # firm -> (bucket, date)

    snapshots = []

    for grade_date, row in ud.iterrows():
        firm = row["Firm"]
        to_grade = row["ToGrade"]

        bucket = GRADE_MAP.get(to_grade, "hold")
        ts = pd.Timestamp(grade_date)
        firm_ratings[firm] = (bucket, ts)

        # Compute current consensus: only include non-stale ratings
        cutoff = ts - pd.Timedelta(days=stale_days)
        active = {f: b for f, (b, d) in firm_ratings.items() if d >= cutoff}

        if not active:
            continue

        counts = {"strongBuy": 0, "buy": 0, "hold": 0, "sell": 0, "strongSell": 0}
        for b in active.values():
            counts[b] += 1

        consensus_label = max(counts, key=counts.get)
        total_analysts = sum(counts.values())
        weighted_score = sum(BUCKET_SCORES[b] * c for b, c in counts.items()) / total_analysts

        snapshots.append({
            "date": ts.normalize(),  # strip time, keep date only
            **counts,
            "consensus": consensus_label,
            "score": weighted_score,
            "n_analysts": total_analysts,
        })

    if not snapshots:
        raise ValueError(f"Could not build consensus history for {ticker}")

    df = pd.DataFrame(snapshots)
    # Multiple events on the same date -> keep the last one (end-of-day state)
    df = df.drop_duplicates(subset="date", keep="last")
    df = df.set_index("date").sort_index()

    df.to_parquet(cache_path)
    return df


def consensus_at_date(
    consensus_history: pd.DataFrame,
    date: pd.Timestamp,
) -> dict:
    """Look up the consensus as of a specific date (point-in-time).

    Uses the most recent consensus snapshot on or before *date*.
    """
    date = pd.Timestamp(date)
    available = consensus_history.loc[:date]
    if available.empty:
        raise ValueError(f"No consensus data available on or before {date}")
    latest = available.iloc[-1]
    return {
        "counts": {
            "strongBuy": int(latest["strongBuy"]),
            "buy": int(latest["buy"]),
            "hold": int(latest["hold"]),
            "sell": int(latest["sell"]),
            "strongSell": int(latest["strongSell"]),
        },
        "consensus": latest["consensus"],
        "score": float(latest["score"]),
        "n_analysts": int(latest["n_analysts"]),
    }


def fetch_recommendations(ticker: str) -> pd.DataFrame:
    """Return the recent analyst recommendations for *ticker*.

    Columns (int counts): strongBuy, buy, hold, sell, strongSell, period
    """
    tk = yf.Ticker(ticker, session=get_session())
    rec = tk.recommendations
    if rec is None or rec.empty:
        raise ValueError(f"No recommendation data for {ticker}")
    return rec


def current_consensus(ticker: str) -> dict:
    """Return current consensus as a dict with counts and a label."""
    rec = fetch_recommendations(ticker)
    latest = rec.iloc[-1]

    buckets = ["strongBuy", "buy", "hold", "sell", "strongSell"]
    available = [b for b in buckets if b in latest.index]
    if not available:
        raise ValueError(f"Unexpected recommendation columns for {ticker}: {list(latest.index)}")

    counts = {b: int(latest[b]) for b in available}
    consensus_label = max(counts, key=counts.get)
    return {"counts": counts, "consensus": consensus_label, "ticker": ticker}
