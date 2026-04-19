"""Execution cost models for execution-aware backtests."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


_SLIPPAGE_MODEL_ALIASES = {
    "flat": "flat",
    "proxy": "proxy",
    "proxy_impact": "proxy",
    "sqrt-impact": "sqrt_impact",
    "sqrt_impact": "sqrt_impact",
    "square-root-impact": "sqrt_impact",
    "square_root_impact": "sqrt_impact",
    "depth": "depth_curve",
    "depth_curve": "depth_curve",
    "l2": "depth_curve",
    "orderbook": "depth_curve",
    "order_book": "depth_curve",
    "fill_aware": "fill_aware",
}


def _resolve_index(*values):
    for value in values:
        if isinstance(value, (pd.Series, pd.DataFrame)):
            return value.index
    raise TypeError("at least one pandas object is required to resolve the execution-cost index")


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


def _first_matching_column(frame, candidates):
    if frame is None or frame.empty:
        return None
    lowered = {str(column).lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _sum_matching_columns(frame, candidates):
    if frame is None or frame.empty:
        return None
    matches = [column for column in frame.columns if any(token in str(column).lower() for token in candidates)]
    if not matches:
        return None
    return frame[matches].sum(axis=1).astype(float)


@runtime_checkable
class SlippageModel(Protocol):
    def estimate(self, trade_notional: pd.Series, volume: pd.Series,
                 volatility: pd.Series, price: pd.Series,
                 orderbook_depth: pd.DataFrame | None = None) -> pd.Series:
        """Estimate one-way execution-cost rates for each bar."""


@dataclass(frozen=True)
class FlatSlippageModel:
    rate: float

    def estimate(self, trade_notional: pd.Series, volume: pd.Series,
                 volatility: pd.Series, price: pd.Series,
                 orderbook_depth: pd.DataFrame | None = None) -> pd.Series:
        index = _resolve_index(trade_notional, volume, volatility, price)
        return pd.Series(float(self.rate), index=index, dtype=float)


@dataclass(frozen=True)
class ProxyImpactModel:
    adv_window: int = 14
    spread_bps: float = 2.0
    impact_bps: float = 5.0
    volatility_multiplier: float = 1.0
    adverse_selection_bps: float = 1.0
    stress_multiplier: float = 1.0

    def estimate(self, trade_notional: pd.Series, volume: pd.Series,
                 volatility: pd.Series, price: pd.Series,
                 orderbook_depth: pd.DataFrame | None = None) -> pd.Series:
        index = _resolve_index(trade_notional, volume, volatility, price)
        trade_notional = _coerce_series(trade_notional, index, fill_value=0.0).clip(lower=0.0)
        volume = _coerce_series(volume, index, fill_value=0.0).clip(lower=0.0)
        volatility = _coerce_series(volatility, index, fill_value=0.0).abs()
        price = _coerce_series(price, index, fill_value=np.nan).abs().replace(0.0, np.nan)

        adv = volume.rolling(max(1, int(self.adv_window)), min_periods=1).mean()
        participation = trade_notional.divide((adv * price).replace(0.0, np.nan)).clip(lower=0.0, upper=1.0)
        spread_component = pd.Series(self.spread_bps / 10000.0, index=index, dtype=float)
        impact_component = (self.impact_bps / 10000.0) * np.sqrt(participation.fillna(0.0))
        adverse_component = (self.adverse_selection_bps / 10000.0) * participation.fillna(0.0) * (1.0 + volatility.fillna(0.0))
        volatility_scale = 1.0 + self.volatility_multiplier * volatility.fillna(0.0)
        total = (spread_component + impact_component + adverse_component) * volatility_scale * self.stress_multiplier
        return total.where(trade_notional.gt(0.0), 0.0).fillna(0.0)


@dataclass(frozen=True)
class SquareRootImpactModel:
    adv_window: int = 14
    base_impact_bps: float = 5.0
    stress_multiplier: float = 1.0

    def estimate(self, trade_notional: pd.Series, volume: pd.Series,
                 volatility: pd.Series, price: pd.Series,
                 orderbook_depth: pd.DataFrame | None = None) -> pd.Series:
        index = _resolve_index(trade_notional, volume, volatility, price)
        trade_notional = _coerce_series(trade_notional, index, fill_value=0.0).clip(lower=0.0)
        volume = _coerce_series(volume, index, fill_value=0.0).clip(lower=0.0)
        volatility = _coerce_series(volatility, index, fill_value=np.nan).abs()
        price = _coerce_series(price, index, fill_value=np.nan).abs().replace(0.0, np.nan)

        adv = volume.rolling(max(1, int(self.adv_window)), min_periods=1).mean()
        participation = trade_notional.divide((adv * price).replace(0.0, np.nan)).clip(lower=0.0, upper=1.0)
        impact = (self.base_impact_bps * volatility * np.sqrt(participation)) / 10000.0

        floor_rate = (self.base_impact_bps / 10000.0) * self.stress_multiplier
        has_turnover = trade_notional.gt(0.0)
        impact = impact.where(has_turnover, 0.0)
        impact.loc[has_turnover] = np.maximum(impact.loc[has_turnover].fillna(0.0) * self.stress_multiplier, floor_rate)
        return impact.fillna(0.0)


@dataclass(frozen=True)
class DepthCurveImpactModel:
    spread_weight_bps: float = 1.0
    depth_penalty_bps: float = 7.5
    imbalance_penalty_bps: float = 3.0
    queue_penalty_bps: float = 1.0
    snapshot_age_penalty_bps: float = 0.5
    stress_multiplier: float = 1.0

    def estimate(self, trade_notional: pd.Series, volume: pd.Series,
                 volatility: pd.Series, price: pd.Series,
                 orderbook_depth: pd.DataFrame | None = None) -> pd.Series:
        index = _resolve_index(trade_notional, volume, volatility, price)
        if orderbook_depth is None:
            raise ValueError("orderbook_depth is required when using DepthCurveImpactModel")

        trade_notional = _coerce_series(trade_notional, index, fill_value=0.0).clip(lower=0.0)
        price = _coerce_series(price, index, fill_value=np.nan).abs().replace(0.0, np.nan)
        depth_frame = _align_numeric_frame(orderbook_depth, index)

        ask_depth = _sum_matching_columns(depth_frame, ["ask_size", "ask_qty", "ask_depth"])
        bid_depth = _sum_matching_columns(depth_frame, ["bid_size", "bid_qty", "bid_depth"])
        if ask_depth is None and bid_depth is None:
            raise ValueError("orderbook_depth must provide bid/ask depth columns")
        if ask_depth is None:
            ask_depth = _align_numeric_series(None, index, fill_value=0.0)
        if bid_depth is None:
            bid_depth = _align_numeric_series(None, index, fill_value=0.0)

        spread_bps_column = _first_matching_column(depth_frame, ["spread_bps", "quoted_spread_bps"])
        if spread_bps_column is not None:
            spread_component = _coerce_series(depth_frame[spread_bps_column], index, fill_value=0.0) / 10000.0
        else:
            ask_price_column = _first_matching_column(depth_frame, ["ask_price_1", "best_ask", "ask_price"])
            bid_price_column = _first_matching_column(depth_frame, ["bid_price_1", "best_bid", "bid_price"])
            if ask_price_column is not None and bid_price_column is not None:
                ask_price = _coerce_series(depth_frame[ask_price_column], index, fill_value=np.nan)
                bid_price = _coerce_series(depth_frame[bid_price_column], index, fill_value=np.nan)
                mid = ((ask_price + bid_price) / 2.0).replace(0.0, np.nan)
                spread_component = (ask_price - bid_price).clip(lower=0.0).divide(mid).fillna(0.0)
            else:
                spread_component = pd.Series(self.spread_weight_bps / 10000.0, index=index, dtype=float)

        total_depth_notional = ((ask_depth + bid_depth).replace(0.0, np.nan) * price).replace(0.0, np.nan)
        participation = trade_notional.divide(total_depth_notional).clip(lower=0.0)

        imbalance_column = _first_matching_column(depth_frame, ["imbalance", "book_imbalance"])
        if imbalance_column is not None:
            imbalance = _coerce_series(depth_frame[imbalance_column], index, fill_value=0.0).abs()
        else:
            total_depth = (ask_depth + bid_depth).replace(0.0, np.nan)
            imbalance = (bid_depth - ask_depth).divide(total_depth).abs().fillna(0.0)

        queue_column = _first_matching_column(depth_frame, ["queue_proxy", "queue", "queue_ahead"])
        queue_proxy = _coerce_series(depth_frame[queue_column], index, fill_value=0.0) if queue_column is not None else pd.Series(0.0, index=index, dtype=float)

        age_column = _first_matching_column(depth_frame, ["snapshot_age_seconds", "snapshot_age", "age_seconds"])
        snapshot_age = _coerce_series(depth_frame[age_column], index, fill_value=0.0) if age_column is not None else pd.Series(0.0, index=index, dtype=float)

        total = (
            spread_component
            + (self.depth_penalty_bps / 10000.0) * participation.fillna(0.0)
            + (self.imbalance_penalty_bps / 10000.0) * imbalance.fillna(0.0)
            + (self.queue_penalty_bps / 10000.0) * queue_proxy.fillna(0.0)
            + (self.snapshot_age_penalty_bps / 10000.0) * np.log1p(snapshot_age.clip(lower=0.0))
        ) * self.stress_multiplier
        return total.where(trade_notional.gt(0.0), 0.0).fillna(0.0)


@dataclass(frozen=True)
class OrderBookImpactModel(DepthCurveImpactModel):
    def estimate(self, trade_notional: pd.Series, volume: pd.Series,
                 volatility: pd.Series, price: pd.Series,
                 orderbook_depth: pd.DataFrame | None = None) -> pd.Series:
        if orderbook_depth is None:
            raise NotImplementedError("L2 data adapter not yet available")
        return super().estimate(
            trade_notional=trade_notional,
            volume=volume,
            volatility=volatility,
            price=price,
            orderbook_depth=orderbook_depth,
        )


@dataclass(frozen=True)
class FillAwareCostModel:
    base_model: object | None = None

    def estimate(self, trade_notional: pd.Series, volume: pd.Series,
                 volatility: pd.Series, price: pd.Series,
                 orderbook_depth: pd.DataFrame | None = None) -> pd.Series:
        model = self.base_model or ProxyImpactModel()
        return model.estimate(
            trade_notional=trade_notional,
            volume=volume,
            volatility=volatility,
            price=price,
            orderbook_depth=orderbook_depth,
        )


def _resolve_slippage_model(slippage_model, slippage_rate):
    if slippage_model is None:
        return FlatSlippageModel(rate=slippage_rate)

    if isinstance(slippage_model, str):
        resolved_name = _SLIPPAGE_MODEL_ALIASES.get(slippage_model.strip().lower(), slippage_model.strip().lower())
        if resolved_name == "flat":
            return FlatSlippageModel(rate=slippage_rate)
        if resolved_name == "proxy":
            return ProxyImpactModel()
        if resolved_name == "sqrt_impact":
            return SquareRootImpactModel()
        if resolved_name == "depth_curve":
            return DepthCurveImpactModel()
        if resolved_name == "fill_aware":
            return FillAwareCostModel(base_model=ProxyImpactModel())
        raise ValueError("Unsupported slippage_model. Choose from ['flat', 'proxy', 'sqrt_impact', 'depth_curve', 'fill_aware']")

    if not hasattr(slippage_model, "estimate"):
        raise TypeError("slippage_model must be None, a supported string alias, or implement estimate(...)")
    return slippage_model


def _estimate_trade_notional_slippage_rates(trade_notional, execution_series,
                                            slippage_rate, slippage_model=None,
                                            volume=None, orderbook_depth=None):
    execution_series = pd.Series(execution_series, copy=False).astype(float)
    model = _resolve_slippage_model(slippage_model, slippage_rate)
    if not isinstance(model, FlatSlippageModel) and volume is None and not isinstance(model, DepthCurveImpactModel):
        raise ValueError("volume is required when using a non-flat execution-cost model")

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


def _clone_model_with_stress(model, stress_multiplier):
    if isinstance(model, FlatSlippageModel):
        return FlatSlippageModel(rate=float(model.rate) * float(stress_multiplier))
    if isinstance(model, FillAwareCostModel):
        base_model = model.base_model or ProxyImpactModel()
        return FillAwareCostModel(base_model=_clone_model_with_stress(base_model, stress_multiplier))
    if hasattr(model, "stress_multiplier"):
        return replace(model, stress_multiplier=float(stress_multiplier))
    return model


def _estimate_fill_event_costs(order_ledger, execution_series,
                               slippage_rate, slippage_model=None,
                               volume=None, orderbook_depth=None,
                               stress_multipliers=(1.0, 1.25, 1.5)):
    execution_series = pd.Series(execution_series, copy=False).astype(float)
    ledger = pd.DataFrame(order_ledger).copy()
    model = _resolve_slippage_model(slippage_model, slippage_rate)
    if ledger.empty:
        zero_series = pd.Series(0.0, index=execution_series.index, dtype=float)
        return {
            "event_costs": pd.DataFrame(),
            "per_bar_cost": zero_series,
            "per_bar_rate": zero_series,
            "total_cost": 0.0,
            "stress_scenarios": {},
            "mode": type(model).__name__,
        }

    fill_events = ledger.loc[
        ledger.get("status").isin(["accepted", "adjusted", "partial_fill"])
        & pd.to_numeric(ledger.get("executed_notional"), errors="coerce").fillna(0.0).gt(0.0)
    ].copy()
    if fill_events.empty:
        zero_series = pd.Series(0.0, index=execution_series.index, dtype=float)
        return {
            "event_costs": fill_events,
            "per_bar_cost": zero_series,
            "per_bar_rate": zero_series,
            "total_cost": 0.0,
            "stress_scenarios": {},
            "mode": type(model).__name__,
        }

    event_index = pd.Index(fill_events["timestamp"])
    trade_notional = pd.Series(fill_events["executed_notional"].to_numpy(dtype=float), index=event_index, dtype=float)
    event_prices = pd.Series(fill_events.get("execution_price", np.nan).to_numpy(dtype=float), index=event_index, dtype=float)
    event_prices = event_prices.fillna(execution_series.reindex(event_index))

    aligned_volume = _align_numeric_series(volume, execution_series.index, fill_value=0.0).clip(lower=0.0)
    volume_events = aligned_volume.reindex(event_index).fillna(0.0)
    volatility_window = max(1, int(getattr(model, "adv_window", 14)))
    volatility = execution_series.pct_change().rolling(volatility_window).std().reindex(event_index).fillna(0.0)
    depth_events = _align_numeric_frame(orderbook_depth, execution_series.index)
    if depth_events is not None:
        depth_events = depth_events.reindex(event_index)

    effective_model = model.base_model if isinstance(model, FillAwareCostModel) and model.base_model is not None else model
    event_rates = effective_model.estimate(
        trade_notional=trade_notional,
        volume=volume_events,
        volatility=volatility,
        price=event_prices,
        orderbook_depth=depth_events,
    ).clip(lower=0.0)
    fill_events["cost_rate"] = event_rates.to_numpy(dtype=float)
    fill_events["cost"] = fill_events["executed_notional"].to_numpy(dtype=float) * fill_events["cost_rate"].to_numpy(dtype=float)

    per_bar_cost = fill_events.groupby("timestamp")["cost"].sum().reindex(execution_series.index).fillna(0.0)
    per_bar_notional = fill_events.groupby("timestamp")["executed_notional"].sum().reindex(execution_series.index).fillna(0.0)
    per_bar_rate = per_bar_cost.divide(per_bar_notional.replace(0.0, np.nan)).fillna(0.0)

    stress_scenarios = {}
    for stress_multiplier in stress_multipliers:
        stressed_model = _clone_model_with_stress(model, stress_multiplier)
        scenario_rates = (stressed_model.base_model if isinstance(stressed_model, FillAwareCostModel) and stressed_model.base_model is not None else stressed_model).estimate(
            trade_notional=trade_notional,
            volume=volume_events,
            volatility=volatility,
            price=event_prices,
            orderbook_depth=depth_events,
        ).clip(lower=0.0)
        stress_scenarios[str(float(stress_multiplier))] = float((trade_notional * scenario_rates).sum())

    return {
        "event_costs": fill_events,
        "per_bar_cost": per_bar_cost,
        "per_bar_rate": per_bar_rate,
        "total_cost": float(per_bar_cost.sum()),
        "stress_scenarios": stress_scenarios,
        "mode": type(model).__name__,
    }


__all__ = [
    "DepthCurveImpactModel",
    "FillAwareCostModel",
    "FlatSlippageModel",
    "OrderBookImpactModel",
    "ProxyImpactModel",
    "SlippageModel",
    "SquareRootImpactModel",
    "_estimate_fill_event_costs",
    "_estimate_reference_trade_slippage_rates",
    "_estimate_slippage_rates",
    "_estimate_trade_notional_slippage_rates",
    "_resolve_slippage_model",
]