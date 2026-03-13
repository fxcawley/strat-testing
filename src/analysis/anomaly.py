"""
Anomaly detection for market signals.

Scaffolding for identifying unusual patterns in price, volume, or
sentiment data that might precede significant moves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def zscore_anomalies(
    series: pd.Series,
    window: int = 20,
    threshold: float = 2.5,
) -> pd.Series:
    """Flag points where the rolling z-score exceeds *threshold*.

    Returns a boolean Series (True = anomaly).
    """
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    z = (series - rolling_mean) / rolling_std
    return z.abs() > threshold


def isolation_forest_anomalies(
    features: pd.DataFrame,
    contamination: float = 0.05,
) -> pd.Series:
    """Use Isolation Forest to flag anomalous rows.

    Parameters
    ----------
    features : DataFrame
        Numeric feature matrix (e.g. returns, volume change, volatility).
    contamination : float
        Expected fraction of anomalies.

    Returns a boolean Series aligned to the input index.
    """
    clean = features.dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(clean)

    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(X)

    result = pd.Series(False, index=features.index)
    result.loc[clean.index] = preds == -1
    return result


def volume_spike(
    volume: pd.Series,
    window: int = 20,
    multiplier: float = 3.0,
) -> pd.Series:
    """Detect volume spikes relative to trailing average."""
    avg = volume.rolling(window).mean()
    return volume > (avg * multiplier)


def build_anomaly_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Build a standard feature set for anomaly detection from OHLCV data.

    Returns columns: return_1d, return_5d, volatility_10d, volume_ratio,
                     high_low_range, gap (open vs prev close).
    """
    close = prices["Close"]
    volume = prices["Volume"]

    features = pd.DataFrame(index=prices.index)
    features["return_1d"] = close.pct_change()
    features["return_5d"] = close.pct_change(5)
    features["volatility_10d"] = features["return_1d"].rolling(10).std()
    features["volume_ratio"] = volume / volume.rolling(20).mean()
    features["high_low_range"] = (prices["High"] - prices["Low"]) / close
    features["gap"] = (prices["Open"] - close.shift(1)) / close.shift(1)

    return features
