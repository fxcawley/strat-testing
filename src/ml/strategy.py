"""
ML return-prediction strategy.

At each monthly rebalance:
  1. Compute features for all ETFs using data up to today
  2. Compute realized next-month returns for all historical months
     (the training target)
  3. Train model on historical features -> returns
  4. Predict next-month return for each ETF
  5. Go long the top-predicted ETFs

Walk-forward: the model only ever sees data available at the prediction
date. Training targets are computed from realized returns in the past.
No look-ahead.

Two models:
  - Ridge regression (linear, L2 regularized)
  - HistGradientBoosting (nonlinear, handles missing features natively)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from src.ml.features import build_feature_matrix


@dataclass
class MLReturnPredictor:
    """Walk-forward ML strategy for ETF return prediction.

    Parameters
    ----------
    model_type : str
        "ridge" or "gbm".
    top_frac : float
        Fraction of universe to go long.
    min_train_months : int
        Minimum number of monthly observations before trading.
    retrain_every : int
        Retrain every N rebalances (1 = every month).
    ridge_alpha : float
        L2 regularization strength for ridge.
    gbm_max_depth : int
        Max tree depth for gradient boosting.
    gbm_n_estimators : int
        Number of trees for gradient boosting.
    """
    model_type: Literal["ridge", "gbm"] = "ridge"
    top_frac: float = 0.30
    min_train_months: int = 36
    retrain_every: int = 1
    ridge_alpha: float = 1.0
    gbm_max_depth: int = 3
    gbm_n_estimators: int = 100

    # Internal state
    _history: list[dict] = field(default_factory=list, repr=False)
    _model: object = field(default=None, repr=False)
    _scaler: object = field(default=None, repr=False)
    _feature_cols: list[str] = field(default_factory=list, repr=False)
    _rebalance_count: int = field(default=0, repr=False)
    _last_features: pd.DataFrame = field(default=None, repr=False)

    def _build_model(self):
        if self.model_type == "ridge":
            return Ridge(alpha=self.ridge_alpha)
        elif self.model_type == "gbm":
            return HistGradientBoostingRegressor(
                max_depth=self.gbm_max_depth,
                max_iter=self.gbm_n_estimators,
                learning_rate=0.05,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def generate_signals(self, date, universe, lookback):
        self._rebalance_count += 1

        # Step 1: compute current features
        current_features = build_feature_matrix(lookback, universe, date)
        if current_features.empty:
            return None

        # Step 2: if we had features from last rebalance, compute realized
        # returns and add to training history
        if self._last_features is not None:
            for ticker in self._last_features.index:
                if ticker in lookback and len(lookback[ticker]) > 22:
                    close = lookback[ticker]["Close"]
                    # Realized return over the last month
                    if len(close) >= 22:
                        realized_ret = close.iloc[-1] / close.iloc[-22] - 1
                        feat_row = self._last_features.loc[ticker].to_dict()
                        feat_row["_target"] = realized_ret
                        feat_row["_ticker"] = ticker
                        feat_row["_date"] = date
                        self._history.append(feat_row)

        # Save current features for next month's target computation
        self._last_features = current_features.copy()

        # Step 3: check if we have enough training data
        if len(self._history) < self.min_train_months:
            return None  # not enough history to train -- hold existing positions

        # Step 4: train model (or use cached)
        if (self._model is None or
                self._rebalance_count % self.retrain_every == 0):
            train_df = pd.DataFrame(self._history)

            # Identify feature columns (exclude metadata)
            self._feature_cols = [c for c in train_df.columns
                                  if c not in ("_target", "_ticker", "_date")]

            X_train = train_df[self._feature_cols].values
            y_train = train_df["_target"].values

            # Handle NaN/inf in features
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

            # Scale features for ridge
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X_train)

            self._model = self._build_model()
            self._model.fit(X_scaled, y_train)

        # Step 5: predict next-month return for each ETF
        # Align current features to training feature columns
        predict_df = current_features.reindex(columns=self._feature_cols, fill_value=0.0)
        X_pred = predict_df.values
        X_pred = np.nan_to_num(X_pred, nan=0.0, posinf=0.0, neginf=0.0)
        X_pred_scaled = self._scaler.transform(X_pred)

        predictions = self._model.predict(X_pred_scaled)
        pred_series = pd.Series(predictions, index=predict_df.index)

        # Step 6: go long the top-predicted ETFs
        n_hold = max(1, int(len(pred_series) * self.top_frac))
        top_tickers = pred_series.nlargest(n_hold).index.tolist()

        # Equal weight the top predictions
        weights = {t: 1.0 / n_hold for t in top_tickers}
        return weights
