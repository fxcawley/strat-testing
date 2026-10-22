"""
MiniWeekly: compact weekly-rebalanced strategy.

A tiny model designed for weekly retraining and rapid iteration.
15 features, Ridge regression, ~0.1s retrain time.  Simple enough
to validate and tweak week-to-week.

Feature set (per ETF):
  Price-based (7):
    1. ret_1w        5-day return
    2. ret_4w        21-day return
    3. ret_13w       63-day return (quarterly momentum)
    4. ret_12_1      252d return skipping last 21d (classic momentum)
    5. vol_21d       21-day realized vol (annualized)
    6. vol_ratio     vol_21d / vol_63d (vol regime shift)
    7. dd_52w        drawdown from 52-week high

  Network-derived (2):
    8. shock_score   propagated shock impact from lead-lag network
    9. diffuse_rank  percentile rank from graph-diffused momentum

  Market regime (4):
    10. mkt_ret_4w   SPY 4-week return
    11. mkt_vol      SPY 21-day realized vol
    12. breadth      frac of universe with positive 4w return
    13. dispersion   cross-sectional std of 4w returns

  Interaction (2):
    14. mom_x_vol    ret_13w * (1 / vol_21d)   -- vol-adjusted momentum
    15. dd_x_breadth dd_52w * breadth           -- contrarian in broad rally

Walk-forward training:
  - Expanding window, retrain every `retrain_every` weeks
  - Ridge with alpha=10 (strong regularization for small feature set)
  - Go long top_frac of universe by predicted return
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Compact feature extraction
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float) -> float:
    return a / b if abs(b) > 1e-10 else 0.0


def extract_mini_features(
    close: np.ndarray,
    ticker_idx: int,
    n_universe: int,
    leadlag: np.ndarray | None,
    shock_zscores: dict[int, float] | None,
    diffusion_ranks: dict[str, float] | None,
    ticker: str,
    mkt_feats: dict[str, float] | None,
) -> dict[str, float]:
    """Extract the 15-feature vector for one ETF.

    All inputs are numpy arrays of historical data ending at the
    current date.  No look-ahead.
    """
    n = len(close)
    if n < 260:
        return {}

    f = {}
    c = close

    # 1-7: price-based features
    f["ret_1w"] = c[-1] / c[-5] - 1 if n > 5 else 0.0
    f["ret_4w"] = c[-1] / c[-21] - 1 if n > 21 else 0.0
    f["ret_13w"] = c[-1] / c[-63] - 1 if n > 63 else 0.0
    f["ret_12_1"] = c[-22] / c[-252] - 1 if n > 252 else 0.0

    rets = np.diff(c) / c[:-1]
    f["vol_21d"] = float(np.std(rets[-21:]) * np.sqrt(252)) if len(rets) >= 21 else 0.15
    vol_63d = float(np.std(rets[-63:]) * np.sqrt(252)) if len(rets) >= 63 else 0.15
    f["vol_ratio"] = _safe_div(f["vol_21d"], vol_63d)

    f["dd_52w"] = c[-1] / np.max(c[-252:]) - 1 if n >= 252 else 0.0

    # 8: shock propagation score
    shock_score = 0.0
    if leadlag is not None and shock_zscores is not None:
        for j, z in shock_zscores.items():
            if j != ticker_idx and j < leadlag.shape[0] and ticker_idx < leadlag.shape[1]:
                shock_score += leadlag[j, ticker_idx] * z
    f["shock_score"] = shock_score

    # 9: graph diffusion rank
    f["diffuse_rank"] = diffusion_ranks.get(ticker, 0.5) if diffusion_ranks else 0.5

    # 10-13: market regime (same for all ETFs on a given date)
    if mkt_feats:
        f["mkt_ret_4w"] = mkt_feats.get("mkt_ret_4w", 0.0)
        f["mkt_vol"] = mkt_feats.get("mkt_vol", 0.15)
        f["breadth"] = mkt_feats.get("breadth", 0.5)
        f["dispersion"] = mkt_feats.get("dispersion", 0.02)
    else:
        f["mkt_ret_4w"] = 0.0
        f["mkt_vol"] = 0.15
        f["breadth"] = 0.5
        f["dispersion"] = 0.02

    # 14-15: interaction features
    f["mom_x_vol"] = f["ret_13w"] * _safe_div(1.0, f["vol_21d"])
    f["dd_x_breadth"] = f["dd_52w"] * f["breadth"]

    return f


# ---------------------------------------------------------------------------
# Network signal helpers (lightweight versions)
# ---------------------------------------------------------------------------

def _build_leadlag_fast(returns_df: pd.DataFrame, max_lag: int = 3) -> np.ndarray:
    """Lightweight lead-lag matrix (lags 1-3 only for speed)."""
    rets = returns_df.values
    T, n = rets.shape
    ll = np.zeros((n, n))

    for lag in range(1, max_lag + 1):
        if T <= lag + 10:
            continue
        leader = rets[:T - lag]
        follower = rets[lag:]
        l_std = leader.std(axis=0, keepdims=True)
        f_std = follower.std(axis=0, keepdims=True)
        l_std = np.where(l_std > 1e-10, l_std, 1.0)
        f_std = np.where(f_std > 1e-10, f_std, 1.0)
        l_norm = (leader - leader.mean(axis=0, keepdims=True)) / l_std
        f_norm = (follower - follower.mean(axis=0, keepdims=True)) / f_std
        ll += (l_norm.T @ f_norm) / (T - lag) / lag

    n_lags = min(max_lag, max(T - 11, 1))
    ll /= n_lags
    np.fill_diagonal(ll, 0)
    ll[np.abs(ll) < 0.03] = 0
    return ll


def _detect_recent_shocks(returns_df: pd.DataFrame, threshold: float = 2.0, window: int = 10):
    """Return dict of {col_idx: z_score} for ETFs with recent shocks."""
    shocks = {}
    for i, col in enumerate(returns_df.columns):
        rets = returns_df[col].values
        if len(rets) < 42:
            continue
        mu = np.mean(rets[-63:])
        sigma = np.std(rets[-63:])
        if sigma < 1e-10:
            continue
        # Check last `window` days for shocks
        recent = rets[-window:]
        z = (recent - mu) / sigma
        max_z = z[np.argmax(np.abs(z))]
        if abs(max_z) > threshold:
            shocks[i] = float(max_z) * (0.85 ** np.argmax(np.abs(z)))  # decay by recency
    return shocks


def _diffusion_ranks(returns_df: pd.DataFrame, mom_lookback: int = 252, skip: int = 21):
    """Compute graph-diffused momentum percentile ranks."""
    from scipy.linalg import expm

    tickers = returns_df.columns.tolist()
    n = len(tickers)

    # Correlation adjacency
    corr = returns_df.corr().values
    corr = 0.7 * corr + 0.3 * np.eye(n)  # shrink
    W = np.maximum(corr, 0)
    np.fill_diagonal(W, 0)
    W[W < 0.15] = 0

    # Raw momentum
    raw = np.zeros(n)
    for i, col in enumerate(tickers):
        vals = returns_df[col].values
        cumret = np.cumprod(1 + vals) - 1
        if len(cumret) > mom_lookback:
            end = len(cumret) - skip if skip > 0 else len(cumret)
            start = end - mom_lookback
            if start >= 0 and end > 0:
                raw[i] = (1 + cumret[end - 1]) / (1 + cumret[start]) - 1

    # Diffuse
    D = np.diag(W.sum(axis=1))
    L = D - W
    d_inv = np.zeros(n)
    diag_d = np.diag(D)
    nz = diag_d > 1e-10
    d_inv[nz] = 1.0 / np.sqrt(diag_d[nz])
    L_norm = np.diag(d_inv) @ L @ np.diag(d_inv)
    kernel = expm(-0.5 * L_norm)
    diffused = kernel @ raw

    # Percentile rank
    order = np.argsort(diffused)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0, 1, n)
    return {tickers[i]: float(ranks[i]) for i in range(n)}


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "ret_1w", "ret_4w", "ret_13w", "ret_12_1",
    "vol_21d", "vol_ratio", "dd_52w",
    "shock_score", "diffuse_rank",
    "mkt_ret_4w", "mkt_vol", "breadth", "dispersion",
    "mom_x_vol", "dd_x_breadth",
]


@dataclass
class MiniWeeklyStrategy:
    """Compact weekly model: 15 features, Ridge, fast retrain.

    Parameters
    ----------
    top_frac : float
        Fraction of universe to go long each week.
    min_train_weeks : int
        Minimum training samples before first trade.
    ridge_alpha : float
        L2 regularization strength.
    retrain_every : int
        Retrain model every N rebalances (1 = every week).
    target_horizon : str
        "1w" for next-week return target, "4w" for monthly.
    """
    top_frac: float = 0.30
    min_train_weeks: int = 52
    ridge_alpha: float = 10.0
    retrain_every: int = 4
    target_horizon: str = "1w"

    # Internal state
    _history: list[dict] = field(default_factory=list, repr=False)
    _model: Ridge | None = field(default=None, repr=False)
    _scaler: StandardScaler | None = field(default=None, repr=False)
    _rebalance_count: int = field(default=0, repr=False)
    _last_features: pd.DataFrame | None = field(default=None, repr=False)
    _last_date: pd.Timestamp | None = field(default=None, repr=False)

    def _compute_market_features(self, lookback, universe) -> dict[str, float]:
        """Market-level features (computed once per rebalance)."""
        mf = {}
        spy = lookback.get("SPY")
        if spy is not None and len(spy) > 25:
            c = spy["Close"].values
            mf["mkt_ret_4w"] = c[-1] / c[-21] - 1 if len(c) > 21 else 0.0
            rets = np.diff(c) / c[:-1]
            mf["mkt_vol"] = float(np.std(rets[-21:]) * np.sqrt(252)) if len(rets) >= 21 else 0.15

        # Breadth and dispersion
        pos, total = 0, 0
        rets_4w = []
        for t in universe:
            df = lookback.get(t)
            if df is None or len(df) < 25:
                continue
            r4w = df["Close"].iloc[-1] / df["Close"].iloc[-21] - 1
            rets_4w.append(r4w)
            if r4w > 0:
                pos += 1
            total += 1
        mf["breadth"] = pos / max(total, 1)
        mf["dispersion"] = float(np.std(rets_4w)) if len(rets_4w) > 3 else 0.02
        return mf

    def generate_signals(self, date, universe, lookback):
        self._rebalance_count += 1

        # Step 1: build return matrix for network signals (63d, fast)
        ret_dict = {}
        for t in universe:
            df = lookback.get(t)
            if df is None or len(df) < 70:
                continue
            ret_dict[t] = df["Close"].pct_change().iloc[-63:]
        ret_df = pd.DataFrame(ret_dict).dropna(axis=1, how="any") if ret_dict else pd.DataFrame()
        valid_net = ret_df.columns.tolist() if not ret_df.empty else []

        # Network signals (lightweight)
        leadlag = _build_leadlag_fast(ret_df, max_lag=3) if len(valid_net) >= 5 else None
        shock_z = _detect_recent_shocks(ret_df) if len(valid_net) >= 5 else None
        diff_ranks = _diffusion_ranks(ret_df) if len(valid_net) >= 5 else None

        # Market features
        mkt_feats = self._compute_market_features(lookback, universe)

        # Step 2: extract features for all ETFs
        rows = []
        ticker_to_netidx = {t: i for i, t in enumerate(valid_net)}
        for ticker in universe:
            df = lookback.get(ticker)
            if df is None or len(df) < 260:
                continue
            close = df["Close"].values.astype(float)
            net_idx = ticker_to_netidx.get(ticker, -1)
            feats = extract_mini_features(
                close, net_idx, len(valid_net),
                leadlag, shock_z, diff_ranks, ticker, mkt_feats,
            )
            if not feats:
                continue
            feats["_ticker"] = ticker
            rows.append(feats)

        if not rows:
            return None

        current_features = pd.DataFrame(rows).set_index("_ticker")

        # Step 3: compute realized returns from last rebalance -> training target
        if self._last_features is not None and self._last_date is not None:
            for ticker in self._last_features.index:
                if ticker in lookback:
                    close = lookback[ticker]["Close"]
                    prev_mask = close.index <= self._last_date
                    if prev_mask.any():
                        prev_price = close.loc[prev_mask].iloc[-1]
                        cur_price = close.iloc[-1]
                        if prev_price > 0:
                            feat_row = self._last_features.loc[ticker].to_dict()
                            feat_row["_target"] = cur_price / prev_price - 1
                            self._history.append(feat_row)

        self._last_features = current_features.copy()
        self._last_date = date

        # Step 4: check training data
        if len(self._history) < self.min_train_weeks:
            return None

        # Step 5: train/retrain
        if self._model is None or self._rebalance_count % self.retrain_every == 0:
            train_df = pd.DataFrame(self._history)
            feat_cols = [c for c in FEATURE_NAMES if c in train_df.columns]
            X = train_df[feat_cols].values.astype(float)
            y = train_df["_target"].values.astype(float)
            X = np.where(np.isfinite(X), X, 0.0)
            y = np.where(np.isfinite(y), y, 0.0)

            self._scaler = StandardScaler()
            X_s = self._scaler.fit_transform(X)
            self._model = Ridge(alpha=self.ridge_alpha)
            self._model.fit(X_s, y)
            self._feat_cols = feat_cols

        # Step 6: predict
        pred_df = current_features.reindex(columns=self._feat_cols, fill_value=0.0)
        X_p = pred_df.values.astype(float)
        X_p = np.where(np.isfinite(X_p), X_p, 0.0)
        X_p_s = self._scaler.transform(X_p)
        preds = self._model.predict(X_p_s)
        pred_s = pd.Series(preds, index=pred_df.index)

        # Step 7: go long top fraction
        n_hold = max(1, int(len(pred_s) * self.top_frac))
        top = pred_s.nlargest(n_hold).index.tolist()
        return {t: 1.0 / n_hold for t in top}

    def get_feature_importance(self) -> pd.Series | None:
        """Return Ridge coefficients as a feature importance measure."""
        if self._model is None:
            return None
        return pd.Series(
            self._model.coef_,
            index=self._feat_cols,
            name="ridge_coef",
        ).sort_values(key=abs, ascending=False)
