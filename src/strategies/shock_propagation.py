"""
ETF Shock Propagation strategy.

Detects significant price shocks in ETFs and models how they propagate
through the network over subsequent periods using lead-lag relationships.

Key ideas:
  - Detect "shocks": daily returns exceeding a z-score threshold (tail events).
  - Build a lead-lag adjacency matrix from lagged cross-correlations:
    edge(A->B, lag=k) = corr(ret_A[t], ret_B[t+k]) for k in [1..max_lag].
  - When a shock hits ETF A, compute the expected propagated impact on
    every other ETF B using the lead-lag weights and historical response
    coefficients.
  - The trading signal combines: (a) direct momentum, (b) expected
    propagated impact from recent shocks in other ETFs.
  - Go long ETFs with the most favorable combined signal.

No look-ahead.  Lead-lag estimation and shock detection use only past data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ShockPropagationStrategy:
    """Trade on expected shock transmission between ETFs.

    Parameters
    ----------
    shock_threshold : float
        Z-score threshold for detecting a shock (e.g. 2.0 = 2 sigma).
    max_lag : int
        Maximum lag (in days) for lead-lag estimation.
    corr_lookback : int
        Days of history for computing lead-lag correlations.
    shock_lookback : int
        Days to look back for recent shocks.
    momentum_lookback : int
        Days for direct momentum signal component.
    propagation_weight : float
        Weight on propagation signal vs direct momentum (0-1).
    top_frac : float
        Fraction of universe to go long.
    decay : float
        Exponential decay factor for shock freshness (per day).
    """
    shock_threshold: float = 2.0
    max_lag: int = 5
    corr_lookback: int = 126
    shock_lookback: int = 21
    momentum_lookback: int = 63
    propagation_weight: float = 0.4
    top_frac: float = 0.30
    decay: float = 0.85

    def _detect_shocks(self, returns: pd.Series) -> pd.DataFrame:
        """Detect shock events in a return series.

        Returns a DataFrame with columns [date, return, z_score, direction]
        for each shock event in the lookback window.
        """
        if len(returns) < 42:
            return pd.DataFrame(columns=["date", "return", "z_score", "direction"])

        # Rolling z-score using expanding window (avoid look-ahead)
        roll_mean = returns.rolling(63, min_periods=21).mean()
        roll_std = returns.rolling(63, min_periods=21).std()
        z_scores = (returns - roll_mean) / roll_std.replace(0, np.nan)

        # Only look at recent shocks
        recent_z = z_scores.iloc[-self.shock_lookback:]
        shock_mask = recent_z.abs() > self.shock_threshold

        if not shock_mask.any():
            return pd.DataFrame(columns=["date", "return", "z_score", "direction"])

        shock_dates = recent_z[shock_mask]
        records = []
        for dt, z_val in shock_dates.items():
            records.append({
                "date": dt,
                "return": float(returns.loc[dt]),
                "z_score": float(z_val),
                "direction": 1.0 if z_val > 0 else -1.0,
            })
        return pd.DataFrame(records)

    def _build_leadlag_matrix(
        self, returns_df: pd.DataFrame,
    ) -> np.ndarray:
        """Build lead-lag adjacency matrix.

        entry[i, j] = max over lags k of corr(ret_i[t], ret_j[t+k])
        weighted by 1/k to prefer shorter lags.

        Positive entry means: a move in i tends to lead a same-direction
        move in j.
        """
        tickers = returns_df.columns.tolist()
        n = len(tickers)
        leadlag = np.zeros((n, n))

        rets = returns_df.values  # (T, n)
        T = rets.shape[0]

        for lag in range(1, self.max_lag + 1):
            if T <= lag + 10:
                continue
            # Correlation between ret_i[:-lag] and ret_j[lag:]
            leader = rets[:T - lag]
            follower = rets[lag:]

            # Standardize
            l_mean = leader.mean(axis=0, keepdims=True)
            l_std = leader.std(axis=0, keepdims=True)
            l_std = np.where(l_std > 1e-10, l_std, 1.0)
            f_mean = follower.mean(axis=0, keepdims=True)
            f_std = follower.std(axis=0, keepdims=True)
            f_std = np.where(f_std > 1e-10, f_std, 1.0)

            l_norm = (leader - l_mean) / l_std
            f_norm = (follower - f_mean) / f_std

            # Cross-correlation matrix
            corr_matrix = (l_norm.T @ f_norm) / (T - lag)

            # Weight by inverse lag (prefer shorter propagation paths)
            lag_weight = 1.0 / lag
            leadlag += corr_matrix * lag_weight

        # Normalize by number of lags
        n_lags = min(self.max_lag, max(T - 11, 0))
        if n_lags > 0:
            leadlag /= n_lags

        # Zero out self-connections and weak links
        np.fill_diagonal(leadlag, 0)
        leadlag[np.abs(leadlag) < 0.02] = 0

        return leadlag

    def _compute_propagated_signal(
        self,
        ticker_idx: int,
        shocks_by_ticker: dict[int, pd.DataFrame],
        leadlag: np.ndarray,
        current_date: pd.Timestamp,
    ) -> float:
        """Compute expected impact on ticker from shocks in other tickers.

        For each other ticker j that had a recent shock:
          impact += leadlag[j, ticker_idx] * shock_magnitude * decay^(days_since_shock)
        """
        n = leadlag.shape[0]
        total_impact = 0.0

        for j in range(n):
            if j == ticker_idx:
                continue
            shocks = shocks_by_ticker.get(j)
            if shocks is None or shocks.empty:
                continue

            # Lead-lag coefficient: how much j leads ticker_idx
            coef = leadlag[j, ticker_idx]
            if abs(coef) < 1e-6:
                continue

            for _, shock in shocks.iterrows():
                shock_date = shock["date"]
                days_ago = (current_date - shock_date).days
                if days_ago < 0 or days_ago > self.shock_lookback:
                    continue
                freshness = self.decay ** days_ago
                impact = coef * shock["z_score"] * freshness
                total_impact += impact

        return total_impact

    def generate_signals(self, date, universe, lookback):
        # Step 1: build return matrix
        returns_dict = {}
        for ticker in universe:
            df = lookback.get(ticker)
            if df is None or len(df) < self.corr_lookback + self.max_lag + 10:
                continue
            returns_dict[ticker] = df["Close"].pct_change().iloc[-self.corr_lookback:]

        if len(returns_dict) < 5:
            return None

        returns_df = pd.DataFrame(returns_dict).dropna(axis=1, how="any")
        valid_tickers = returns_df.columns.tolist()

        if len(valid_tickers) < 5:
            return None

        # Step 2: build lead-lag adjacency matrix
        leadlag = self._build_leadlag_matrix(returns_df)

        # Step 3: detect recent shocks for each ticker
        shocks_by_idx: dict[int, pd.DataFrame] = {}
        for i, ticker in enumerate(valid_tickers):
            df = lookback.get(ticker)
            if df is None:
                continue
            rets = df["Close"].pct_change().dropna()
            shocks = self._detect_shocks(rets)
            if not shocks.empty:
                shocks_by_idx[i] = shocks

        # Step 4: compute combined signal for each ticker
        scores: dict[str, float] = {}
        for i, ticker in enumerate(valid_tickers):
            # Direct momentum component
            df = lookback.get(ticker)
            if df is None:
                continue
            close = df["Close"]
            n = len(close)
            if n < self.momentum_lookback + 5:
                continue
            direct_mom = close.iloc[-1] / close.iloc[-self.momentum_lookback] - 1

            # Propagated shock component
            prop_signal = self._compute_propagated_signal(
                i, shocks_by_idx, leadlag, date
            )

            # Combine: weighted average of direct momentum and propagation signal
            combined = (
                (1 - self.propagation_weight) * direct_mom
                + self.propagation_weight * prop_signal * 0.01  # scale propagation
            )
            scores[ticker] = combined

        if len(scores) < 3:
            return None

        # Step 5: go long top fraction
        sorted_tickers = sorted(scores, key=scores.get, reverse=True)
        n_long = max(1, int(len(sorted_tickers) * self.top_frac))
        longs = sorted_tickers[:n_long]

        return {t: 1.0 / n_long for t in longs}
