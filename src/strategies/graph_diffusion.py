"""
Graph Diffusion strategy for ETFs.

Models ETFs as nodes in a correlation graph.  Edges represent trailing
return co-movement (Pearson correlation).  A raw momentum signal is
assigned to each node, then diffused across the graph using the heat
kernel on the graph Laplacian.

Key ideas:
  - Build a dynamic correlation graph from trailing daily returns.
  - Threshold weak edges to create a sparse, meaningful topology.
  - Compute the graph Laplacian L = D - W (degree matrix minus adjacency).
  - Diffuse signals: s_diffused = exp(-t * L) @ s_raw
    where t controls diffusion time (how far signals spread).
  - The diffused signal captures "network momentum": an ETF may rank
    highly not because of its own momentum, but because its correlated
    peers are rallying.
  - Go long the top ETFs by diffused signal score.

No look-ahead.  Correlation matrix and signals use only past data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import expm


@dataclass
class GraphDiffusionStrategy:
    """Graph-based diffusion momentum on ETFs.

    Parameters
    ----------
    corr_lookback : int
        Days of trailing returns for correlation estimation.
    momentum_lookback : int
        Days for raw momentum signal.
    skip_days : int
        Skip most recent N days in momentum calc (Jegadeesh-Titman).
    diffusion_time : float
        Heat diffusion time parameter.  Larger = more spreading.
        0 = pure momentum (no diffusion).
    corr_threshold : float
        Minimum absolute correlation to form an edge.
    top_frac : float
        Fraction of universe to go long.
    use_partial_corr : bool
        If True, shrink the correlation matrix toward identity
        (Ledoit-Wolf-style) for stability.
    shrinkage : float
        Shrinkage intensity toward identity (0 = full sample, 1 = identity).
    """
    corr_lookback: int = 126
    momentum_lookback: int = 252
    skip_days: int = 21
    diffusion_time: float = 0.5
    corr_threshold: float = 0.15
    top_frac: float = 0.30
    use_partial_corr: bool = True
    shrinkage: float = 0.3

    def _build_correlation_graph(
        self, returns_matrix: pd.DataFrame,
    ) -> tuple[np.ndarray, list[str]]:
        """Build thresholded correlation adjacency matrix.

        Returns (adjacency_matrix, ticker_list).
        """
        corr = returns_matrix.corr()
        tickers = corr.columns.tolist()
        n = len(tickers)

        # Shrink toward identity for stability
        if self.use_partial_corr:
            C = corr.values
            C = (1 - self.shrinkage) * C + self.shrinkage * np.eye(n)
        else:
            C = corr.values

        # Threshold: zero out weak correlations, keep only positive edges
        W = np.maximum(C, 0)
        np.fill_diagonal(W, 0)
        W[W < self.corr_threshold] = 0

        return W, tickers

    def _heat_kernel_diffusion(self, W: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """Diffuse signal through graph using heat kernel.

        s_out = exp(-t * L) @ s_in
        where L = D - W is the graph Laplacian.
        """
        n = W.shape[0]
        if n == 0:
            return signal

        D = np.diag(W.sum(axis=1))
        L = D - W

        # Normalize Laplacian for numerical stability
        d_inv_sqrt = np.zeros(n)
        diag_D = np.diag(D)
        nonzero = diag_D > 1e-10
        d_inv_sqrt[nonzero] = 1.0 / np.sqrt(diag_D[nonzero])
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt

        # Heat kernel
        kernel = expm(-self.diffusion_time * L_norm)
        return kernel @ signal

    def generate_signals(self, date, universe, lookback):
        # Step 1: build return matrix for correlation estimation
        returns_dict = {}
        for ticker in universe:
            df = lookback.get(ticker)
            if df is None or len(df) < self.corr_lookback + 5:
                continue
            close = df["Close"]
            rets = close.pct_change().iloc[-self.corr_lookback:]
            if rets.isna().sum() > self.corr_lookback * 0.2:
                continue
            returns_dict[ticker] = rets

        if len(returns_dict) < 5:
            return None

        returns_df = pd.DataFrame(returns_dict).dropna(axis=1, how="any")
        valid_tickers = returns_df.columns.tolist()

        if len(valid_tickers) < 5:
            return None

        # Step 2: compute raw momentum signal for each valid ticker
        raw_signal = np.zeros(len(valid_tickers))
        for i, ticker in enumerate(valid_tickers):
            df = lookback.get(ticker)
            if df is None:
                continue
            close = df["Close"]
            n = len(close)
            end_pos = n - self.skip_days if self.skip_days > 0 else n
            start_pos = end_pos - self.momentum_lookback
            if start_pos < 0 or end_pos < 1:
                continue
            ret = close.iloc[end_pos - 1] / close.iloc[start_pos] - 1
            raw_signal[i] = float(ret)

        # Step 3: build correlation graph and diffuse
        W, graph_tickers = self._build_correlation_graph(returns_df)

        # Align raw signal to graph ordering
        ticker_to_idx = {t: i for i, t in enumerate(valid_tickers)}
        aligned_signal = np.array([
            raw_signal[ticker_to_idx[t]] if t in ticker_to_idx else 0.0
            for t in graph_tickers
        ])

        # Step 4: diffuse the momentum signal through the graph
        diffused = self._heat_kernel_diffusion(W, aligned_signal)

        # Step 5: rank by diffused signal, go long top fraction
        scores = {t: float(diffused[i]) for i, t in enumerate(graph_tickers)}

        sorted_tickers = sorted(scores, key=scores.get, reverse=True)
        n_long = max(1, int(len(sorted_tickers) * self.top_frac))
        longs = sorted_tickers[:n_long]

        return {t: 1.0 / n_long for t in longs}
