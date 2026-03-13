"""
Simple blend strategy: allocates a fixed fraction to each sub-strategy.

Used for combining XS Momentum (equity ETFs) with Trend Following
(multi-asset). The blend is static -- no regime detection, no
dynamic reweighting. This is the minimum-complexity combination.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class StaticBlend:
    """Blend multiple sub-strategies with fixed weights.

    Parameters
    ----------
    strategies : dict[str, tuple[object, float]]
        Named sub-strategies with their blend weights.
        {name: (strategy_instance, weight)}.  Weights should sum to 1.0.
    """
    strategies: dict[str, tuple[object, float]] = field(default_factory=dict)

    def generate_signals(self, date, universe, lookback):
        blended: dict[str, float] = {}

        active_weight = 0.0
        for name, (strat, weight) in self.strategies.items():
            sig = strat.generate_signals(date, universe, lookback)
            if sig is None or not sig:
                continue
            active_weight += weight
            for tkr, tw in sig.items():
                blended[tkr] = blended.get(tkr, 0.0) + weight * tw

        if not blended:
            return None

        # Re-normalize: if one sub-strategy returned None, scale up
        # the other's weights to fill the allocation
        if active_weight > 0 and active_weight < 0.99:
            scale = 1.0 / active_weight
            blended = {t: w * scale for t, w in blended.items()}

        # Cap total at 1.0
        total = sum(abs(w) for w in blended.values())
        if total > 1.0:
            blended = {t: w / total for t, w in blended.items()}

        return {t: w for t, w in blended.items() if abs(w) > 1e-6}
