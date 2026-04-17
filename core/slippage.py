"""Slippage models for execution-aware backtests."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


_SLIPPAGE_MODEL_ALIASES = {
    "flat": "flat",
    "sqrt-impact": "sqrt_impact",
    "sqrt_impact": "sqrt_impact",
    "square-root-impact": "sqrt_impact",
    "square_root_impact": "sqrt_impact",
    "orderbook": "orderbook",
    "order_book": "orderbook",
}


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


def _align_numeric_series(values, index, fill_value=0.0):
    if values is None:
        return pd.Series(fill_value, index=index, dtype=float)
    return _coerce_series(values, index, fill_value=fill_value)


def _align_numeric_frame(frame, index):
    if frame is None:
        return None
    aligned = frame.reindex(index).copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame, index=index)
    for column in aligned.columns:
        aligned[column] = pd.to_numeric(aligned[column], errors="coerce")
    return aligned


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


def _resolve_slippage_model(slippage_model, slippage_rate):
    if slippage_model is None:
        return FlatSlippageModel(rate=slippage_rate)

    if isinstance(slippage_model, str):
        resolved_name = _SLIPPAGE_MODEL_ALIASES.get(slippage_model.strip().lower(), slippage_model.strip().lower())
        if resolved_name == "flat":
            return FlatSlippageModel(rate=slippage_rate)
        if resolved_name == "sqrt_impact":
            return SquareRootImpactModel()
        if resolved_name == "orderbook":
            return OrderBookImpactModel()
        raise ValueError("Unsupported slippage_model. Choose from ['flat', 'sqrt_impact', 'orderbook']")

    if not hasattr(slippage_model, "estimate"):
        raise TypeError("slippage_model must be None, a supported string alias, or implement estimate(...)")
    return slippage_model


def _estimate_trade_notional_slippage_rates(trade_notional, execution_series,
                                            slippage_rate, slippage_model=None,
                                            volume=None, orderbook_depth=None):
    execution_series = pd.Series(execution_series, copy=False).astype(float)
    model = _resolve_slippage_model(slippage_model, slippage_rate)
    if not isinstance(model, FlatSlippageModel) and volume is None:
        raise ValueError("volume is required when using a non-flat slippage model")

    trade_notional = _align_numeric_series(trade_notional, execution_series.index, fill_value=0.0).clip(lower=0.0)
    aligned_volume = _align_numeric_series(volume, execution_series.index, fill_value=0.0).clip(lower=0.0)
    volatility_window = max(1, int(getattr(model, "adv_window", 14)))
    volatility = execution_series.pct_change().rolling(volatility_window).std()

    slippage_rates = model.estimate(
        trade_notional=trade_notional,
        volume=aligned_volume,
        volatility=volatility,
        price=execution_series,
        orderbook_depth=_align_numeric_frame(orderbook_depth, execution_series.index),
    )
    return _align_numeric_series(slippage_rates, execution_series.index, fill_value=0.0).clip(lower=0.0)


def _estimate_reference_trade_slippage_rates(equity, execution_series,
                                             slippage_rate, slippage_model=None,
                                             volume=None, orderbook_depth=None):
    execution_series = pd.Series(execution_series, copy=False).astype(float)
    trade_notional = pd.Series(max(float(equity), 0.0), index=execution_series.index, dtype=float)
    return _estimate_trade_notional_slippage_rates(
        trade_notional=trade_notional,
        execution_series=execution_series,
        slippage_rate=slippage_rate,
        slippage_model=slippage_model,
        volume=volume,
        orderbook_depth=orderbook_depth,
    )


def _estimate_slippage_rates(position, equity, valuation_series, execution_series,
                             slippage_rate, slippage_model=None, volume=None,
                             funding_rates=None, orderbook_depth=None):
    position = pd.Series(position, index=valuation_series.index, copy=False).reindex(valuation_series.index).fillna(0.0).astype(float)
    turnover = position.diff().abs().fillna(position.abs()).astype(float)
    aligned_funding = _align_numeric_series(funding_rates, valuation_series.index, fill_value=0.0)
    gross_returns = position * valuation_series.pct_change().fillna(0.0) - position * aligned_funding
    gross_equity = float(equity) * (1.0 + gross_returns).cumprod()
    prev_equity = gross_equity.shift(1).fillna(float(equity))
    trade_notional = prev_equity * turnover
    slippage_rates = _estimate_trade_notional_slippage_rates(
        trade_notional=trade_notional,
        execution_series=execution_series,
        slippage_rate=slippage_rate,
        slippage_model=slippage_model,
        volume=volume,
        orderbook_depth=orderbook_depth,
    )
    return slippage_rates.where(turnover > 0.0, 0.0), turnover


__all__ = [
    "FlatSlippageModel",
    "OrderBookImpactModel",
    "SlippageModel",
    "SquareRootImpactModel",
]
