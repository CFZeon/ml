"""Backward-compatible execution cost exports.

This module remains as a compatibility surface for existing imports while the
canonical execution cost implementations live under core.execution.costs.
"""

from .execution.costs import (
    DepthCurveImpactModel,
    FillAwareCostModel,
    FlatSlippageModel,
    OrderBookImpactModel,
    ProxyImpactModel,
    SlippageModel,
    SquareRootImpactModel,
    _estimate_fill_event_costs,
    _estimate_reference_trade_slippage_rates,
    _estimate_slippage_rates,
    _estimate_trade_notional_slippage_rates,
    _resolve_slippage_model,
)

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
