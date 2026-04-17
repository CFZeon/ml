"""Slippage models for execution-aware backtests."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


def _resolve_index(*values):
    for value in values:
        if isinstance(value, (pd.Series, pd.DataFrame)):
            return value.index
    raise TypeError("at least one pandas object is required to resolve the slippage index")


def _coerce_series(values, index, fill_value=0.0):
    if isinstance(values, pd.Series):
        series = values.reindex(index)
    else:
        series = pd.Series(values, index=index)
    series = pd.to_numeric(series, errors="coerce")
    if fill_value is not None:
        series = series.fillna(fill_value)
    return series.astype(float)


@runtime_checkable
class SlippageModel(Protocol):
    def estimate(self, trade_notional: pd.Series, volume: pd.Series,
                 volatility: pd.Series, price: pd.Series,
                 orderbook_depth: pd.DataFrame | None = None) -> pd.Series:
        """Estimate one-way slippage rates for each bar."""


class FlatSlippageModel:
    def __init__(self, rate: float):
        self.rate = float(rate)

    def estimate(self, trade_notional: pd.Series, volume: pd.Series,
                 volatility: pd.Series, price: pd.Series,
                 orderbook_depth: pd.DataFrame | None = None) -> pd.Series:
        index = _resolve_index(trade_notional, volume, volatility, price)
        return pd.Series(self.rate, index=index, dtype=float)


class SquareRootImpactModel:
    def __init__(self, adv_window: int = 14, base_impact_bps: float = 5.0):
        self.adv_window = max(1, int(adv_window))
        self.base_impact_bps = float(base_impact_bps)

    def estimate(self, trade_notional: pd.Series, volume: pd.Series,
                 volatility: pd.Series, price: pd.Series,
                 orderbook_depth: pd.DataFrame | None = None) -> pd.Series:
        index = _resolve_index(trade_notional, volume, volatility, price)
        trade_notional = _coerce_series(trade_notional, index, fill_value=0.0).clip(lower=0.0)
        volume = _coerce_series(volume, index, fill_value=0.0).clip(lower=0.0)
        volatility = _coerce_series(volatility, index, fill_value=np.nan).abs()
        price = _coerce_series(price, index, fill_value=np.nan).abs().replace(0.0, np.nan)

        adv = volume.rolling(self.adv_window).mean()
        participation = trade_notional.divide((adv * price).replace(0.0, np.nan)).clip(lower=0.0, upper=1.0)
        impact = (self.base_impact_bps * volatility * np.sqrt(participation)) / 10000.0

        floor_rate = self.base_impact_bps / 10000.0
        has_turnover = trade_notional.gt(0.0)
        impact = impact.where(has_turnover, 0.0)
        impact.loc[has_turnover] = np.maximum(impact.loc[has_turnover].fillna(0.0), floor_rate)
        return impact.fillna(0.0)


class OrderBookImpactModel:
    def estimate(self, trade_notional: pd.Series, volume: pd.Series,
                 volatility: pd.Series, price: pd.Series,
                 orderbook_depth: pd.DataFrame | None = None) -> pd.Series:
        raise NotImplementedError("L2 data adapter not yet available")


__all__ = [
    "FlatSlippageModel",
    "OrderBookImpactModel",
    "SlippageModel",
    "SquareRootImpactModel",
]
