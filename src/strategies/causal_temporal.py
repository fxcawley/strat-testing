"""
Causal Temporal Neural Network strategy.

Uses causal (dilated) 1D convolutions to extract multi-scale temporal
patterns from ETF price data, then feeds them into a walk-forward
Ridge regression to predict next-period returns.

Key ideas:
  - Causal convolutions: each filter only looks at past data (no leakage).
  - Dilated filters at scales 1, 2, 4, 8, 16 capture short-to-long
    temporal dependencies (inspired by WaveNet / TCN architectures).
  - Learned filters are replaced by fixed analytical kernels (derivative,
    momentum, mean-reversion, volatility-break) to avoid neural network
    training overhead while preserving the multi-scale causal structure.
  - Walk-forward Ridge regression maps the convolutional feature vector
    to next-month return predictions.
  - Go long the top-predicted ETFs, equal-weight.

No look-ahead. All convolutions are strictly causal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Causal 1D convolution helpers
# ---------------------------------------------------------------------------

def _causal_conv1d(signal: np.ndarray, kernel: np.ndarray, dilation: int = 1) -> np.ndarray:
    """Apply a causal 1D convolution with dilation.

    Only uses past values -- output[t] depends on signal[t], signal[t-d],
    signal[t-2d], ..., where d = dilation.
    """
    k_len = len(kernel)
    n = len(signal)
    receptive = (k_len - 1) * dilation + 1
    if n < receptive:
        return np.full(n, np.nan)

    out = np.full(n, np.nan)
    for t in range(receptive - 1, n):
        indices = [t - i * dilation for i in range(k_len)]
        out[t] = np.dot(kernel, signal[indices])
    return out


def _causal_conv1d_fast(signal: np.ndarray, kernel: np.ndarray, dilation: int = 1) -> np.ndarray:
    """Vectorized causal 1D convolution with dilation."""
    k_len = len(kernel)
    n = len(signal)
    receptive = (k_len - 1) * dilation + 1
    if n < receptive:
        return np.full(n, np.nan)

    # Build index matrix: shape (n - receptive + 1, k_len)
    start = receptive - 1
    t_indices = np.arange(start, n)
    offsets = np.arange(k_len) * dilation
    idx_matrix = t_indices[:, None] - offsets[None, :]  # (valid_len, k_len)

    gathered = signal[idx_matrix]  # (valid_len, k_len)
    conv_vals = gathered @ kernel   # (valid_len,)

    out = np.full(n, np.nan)
    out[start:] = conv_vals
    return out


# Fixed analytical kernels (no training required)
KERNELS = {
    "derivative": np.array([1.0, -1.0]),                       # first difference
    "momentum": np.array([1.0, 0.0, 0.0, -1.0]) / 2.0,        # skip-difference
    "smooth_deriv": np.array([1.0, 1.0, -1.0, -1.0]) / 2.0,   # smoothed gradient
    "curvature": np.array([1.0, -2.0, 1.0]),                   # second difference
    "vol_break": np.array([1.0, -1.0, -1.0, 1.0]) / 2.0,      # change in gradient
}

DILATIONS = [1, 2, 4, 8, 16]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_causal_features(close: np.ndarray, volume: np.ndarray | None = None) -> dict[str, float]:
    """Extract multi-scale causal temporal features from a price series.

    Returns a dict of feature_name -> feature_value using the most
    recent output of each (kernel, dilation) combination.
    """
    if len(close) < 260:
        return {}

    log_price = np.log(close)
    returns = np.diff(log_price)
    abs_returns = np.abs(returns)

    features = {}
    for series, series_name in [
        (log_price, "logp"),
        (returns, "ret"),
        (abs_returns, "absret"),
    ]:
        for k_name, kernel in KERNELS.items():
            for d in DILATIONS:
                conv = _causal_conv1d_fast(series, kernel, dilation=d)
                if np.all(np.isnan(conv)):
                    continue
                # Take the most recent valid value as the feature
                last_val = conv[-1]
                if np.isfinite(last_val):
                    features[f"{series_name}_{k_name}_d{d}"] = float(last_val)

                # Also take a short trailing statistic for richer representation
                tail = conv[-21:]
                valid_tail = tail[np.isfinite(tail)]
                if len(valid_tail) >= 5:
                    features[f"{series_name}_{k_name}_d{d}_mean21"] = float(np.mean(valid_tail))
                    features[f"{series_name}_{k_name}_d{d}_std21"] = float(np.std(valid_tail))

    # Volume features (if available)
    if volume is not None and len(volume) > 42:
        v = volume.astype(float)
        v_safe = np.where(v > 0, v, np.nan)
        log_vol = np.log(np.where(np.isfinite(v_safe), v_safe, 1.0))
        for k_name, kernel in [("derivative", KERNELS["derivative"]),
                                ("smooth_deriv", KERNELS["smooth_deriv"])]:
            for d in [1, 2, 4]:
                conv = _causal_conv1d_fast(log_vol, kernel, dilation=d)
                last = conv[-1]
                if np.isfinite(last):
                    features[f"vol_{k_name}_d{d}"] = float(last)

    return features


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

@dataclass
class CausalTemporalStrategy:
    """Walk-forward causal temporal convolution strategy.

    Parameters
    ----------
    top_frac : float
        Fraction of universe to go long.
    min_train_months : int
        Minimum training observations before trading.
    ridge_alpha : float
        L2 regularization for Ridge regression.
    retrain_every : int
        Retrain every N rebalances.
    """
    top_frac: float = 0.30
    min_train_months: int = 36
    ridge_alpha: float = 10.0
    retrain_every: int = 1

    # Internal state
    _history: list[dict] = field(default_factory=list, repr=False)
    _model: object = field(default=None, repr=False)
    _scaler: object = field(default=None, repr=False)
    _feature_cols: list[str] = field(default_factory=list, repr=False)
    _rebalance_count: int = field(default=0, repr=False)
    _last_features: pd.DataFrame | None = field(default=None, repr=False)
    _last_date: pd.Timestamp | None = field(default=None, repr=False)

    def generate_signals(self, date, universe, lookback):
        self._rebalance_count += 1

        # Step 1: extract causal temporal features for every ETF
        rows = []
        for ticker in universe:
            df = lookback.get(ticker)
            if df is None or len(df) < 260:
                continue
            close = df["Close"].values.astype(float)
            vol = df["Volume"].values.astype(float) if "Volume" in df.columns else None
            feats = extract_causal_features(close, vol)
            if not feats:
                continue
            feats["_ticker"] = ticker
            rows.append(feats)

        if not rows:
            return None

        current_features = pd.DataFrame(rows).set_index("_ticker")

        # Step 2: if we have last rebalance's features, compute realized
        # returns and add to training set
        if self._last_features is not None and self._last_date is not None:
            for ticker in self._last_features.index:
                if ticker in lookback:
                    close = lookback[ticker]["Close"]
                    prev_mask = close.index <= self._last_date
                    if prev_mask.any():
                        prev_price = close.loc[prev_mask].iloc[-1]
                        current_price = close.iloc[-1]
                        if prev_price > 0:
                            realized_ret = current_price / prev_price - 1
                            feat_row = self._last_features.loc[ticker].to_dict()
                            feat_row["_target"] = realized_ret
                            self._history.append(feat_row)

        self._last_features = current_features.copy()
        self._last_date = date

        # Step 3: check training data sufficiency
        if len(self._history) < self.min_train_months:
            return None

        # Step 4: train Ridge on causal temporal features
        if self._model is None or self._rebalance_count % self.retrain_every == 0:
            train_df = pd.DataFrame(self._history)
            self._feature_cols = [c for c in train_df.columns if c != "_target"]
            X_train = train_df[self._feature_cols].values.astype(float)
            y_train = train_df["_target"].values.astype(float)

            # Impute non-finite values
            X_train = np.where(np.isfinite(X_train), X_train, 0.0)
            y_train = np.where(np.isfinite(y_train), y_train, 0.0)

            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X_train)

            self._model = Ridge(alpha=self.ridge_alpha)
            self._model.fit(X_scaled, y_train)

        # Step 5: predict
        predict_df = current_features.reindex(columns=self._feature_cols, fill_value=0.0)
        X_pred = predict_df.values.astype(float)
        X_pred = np.where(np.isfinite(X_pred), X_pred, 0.0)
        X_pred_scaled = self._scaler.transform(X_pred)

        predictions = self._model.predict(X_pred_scaled)
        pred_series = pd.Series(predictions, index=predict_df.index)

        # Step 6: go long the top predicted ETFs
        n_hold = max(1, int(len(pred_series) * self.top_frac))
        top_tickers = pred_series.nlargest(n_hold).index.tolist()

        return {t: 1.0 / n_hold for t in top_tickers}
