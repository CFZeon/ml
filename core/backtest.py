"""Bet sizing and backtesting."""

import numpy as np
import pandas as pd

from .execution import (
    NautilusExecutionAdapter,
    OrderIntent,
    resolve_execution_policy,
    resolve_liquidity_inputs,
)
from .slippage import (
    _estimate_fill_event_costs,
    _estimate_slippage_rates,
    _estimate_trade_notional_slippage_rates,
)
from .scenarios import (
    apply_execution_scenarios,
    apply_scenario_price_policy,
    build_scenario_schedule,
    merge_scenario_lifecycle,
)
from .universe import apply_symbol_lifecycle_policy, build_symbol_lifecycle_frame

try:  # pragma: no cover - optional dependency exercised in integration tests
    import vectorbt as vbt
    from vectorbt.portfolio.enums import Direction, SizeType
except ImportError:  # pragma: no cover
    vbt = None
    Direction = None
    SizeType = None


_SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60
_DEFAULT_SIGNIFICANCE_CONFIG = {
    "enabled": True,
    "method": "stationary_bootstrap",
    "bootstrap_samples": 500,
    "confidence_level": 0.95,
    "mean_block_length": None,
    "random_state": 42,
    "min_observations": 8,
}


# ───────────────────────────────────────────────────────────────────────────
# Kelly criterion
# ───────────────────────────────────────────────────────────────────────────

def kelly_fraction(prob_win, avg_win, avg_loss, fraction=0.5):
    """Fractional Kelly position size.

    Parameters
    ----------
    prob_win : float   – estimated probability of a winning trade
    avg_win  : float   – average win magnitude  (positive)
    avg_loss : float   – average loss magnitude  (positive)
    fraction : float   – Kelly fraction (0.5 = half-Kelly)

    Returns float in [0, 1].
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0
    b = avg_win / avg_loss
    q = 1 - prob_win
    k = (prob_win * b - q) / b
    return max(0.0, min(k, 1.0)) * fraction


def _infer_periods_per_year(index):
    if len(index) < 2:
        return 0.0

    deltas = index.to_series().diff().dropna()
    if deltas.empty:
        return 0.0

    seconds = deltas.median().total_seconds()
    if seconds <= 0:
        return 0.0

    return _SECONDS_PER_YEAR / seconds


def _round_metric(value, digits=4):
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return round(float(value), digits)
    return value


def _safe_ratio(numerator, denominator, default=0.0):
    if denominator == 0:
        if numerator > 0:
            return float("inf")
        return default
    return numerator / denominator


def _apply_action_latency(requested_position, action_latency_bars):
    requested = pd.Series(requested_position, copy=False).astype(float)
    request_timestamps = pd.Series(requested.index, index=requested.index, dtype=object)
    latency = max(0, int(action_latency_bars))
    if latency <= 0:
        return requested, request_timestamps
    return requested.shift(latency).fillna(0.0), request_timestamps.shift(latency)


def _summarize_execution_delay_metrics(order_intents, order_ledger, index):
    if order_intents is None or order_intents.empty:
        return {
            "average_action_delay_bars": 0.0,
            "average_fill_delay_bars": 0.0,
            "max_fill_delay_bars": 0,
        }

    index_lookup = {timestamp: loc for loc, timestamp in enumerate(pd.DatetimeIndex(index))}

    def _delay_bars(later, earlier):
        later_loc = index_lookup.get(pd.Timestamp(later)) if later is not None and pd.notna(later) else None
        earlier_loc = index_lookup.get(pd.Timestamp(earlier)) if earlier is not None and pd.notna(earlier) else None
        if later_loc is None or earlier_loc is None:
            return None
        return max(0, int(later_loc - earlier_loc))

    action_delays = [
        _delay_bars(row.get("timestamp"), row.get("request_timestamp"))
        for _, row in order_intents.iterrows()
    ]
    action_delays = [delay for delay in action_delays if delay is not None]

    fill_delays = []
    if order_ledger is not None and not order_ledger.empty:
        for _, row in order_ledger.iterrows():
            if float(row.get("executed_notional", 0.0) or 0.0) <= 0.0:
                continue
            delay = _delay_bars(row.get("timestamp"), row.get("request_timestamp"))
            if delay is not None:
                fill_delays.append(delay)

    return {
        "average_action_delay_bars": float(np.mean(action_delays)) if action_delays else 0.0,
        "average_fill_delay_bars": float(np.mean(fill_delays)) if fill_delays else 0.0,
        "max_fill_delay_bars": int(max(fill_delays)) if fill_delays else 0,
    }


def _round_down_to_step(values, step):
    if step is None or step <= 0:
        return values
    return np.floor(values / step) * step


def _normalize_price_series(price, tick_size=None, fill_policy="strict", fill_limit=None,
                            return_diagnostics=False, series_name="price"):
    series = pd.Series(price, copy=False).astype(float)
    if tick_size is not None and tick_size > 0:
        series = pd.Series(_round_down_to_step(series.to_numpy(), tick_size), index=series.index, dtype=float)
    series = series.replace(0.0, np.nan)
    leading_missing_rows = int(series.isna().cumprod().sum()) if len(series) > 0 else 0

    if fill_policy == "strict" or fill_policy == "drop_rows":
        normalized = series.copy()
    elif fill_policy == "ffill":
        normalized = series.ffill()
    elif fill_policy == "ffill_with_limit":
        limit = max(1, int(fill_limit)) if fill_limit is not None else 1
        normalized = series.ffill(limit=limit)
    else:
        raise ValueError(f"Unsupported price fill policy: {fill_policy!r}")

    diagnostics = {
        "series": series_name,
        "policy": fill_policy,
        "fill_limit": None if fill_limit is None else int(fill_limit),
        "leading_missing_rows": leading_missing_rows,
        "forward_filled_rows": int((series.isna() & normalized.notna()).sum()),
        "invalid_rows": int(normalized.isna().sum()),
        "dropped_rows": 0,
    }
    if return_diagnostics:
        return normalized, diagnostics
    return normalized


def _normalize_position_targets(signals, leverage=1.0, allow_short=True):
    target = pd.Series(signals, copy=False).astype(float).fillna(0.0) * float(leverage)
    if allow_short:
        return target.clip(-abs(float(leverage)), abs(float(leverage)))
    return target.clip(0.0, abs(float(leverage)))


def _enabled_limit(value):
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        return None
    return value


def _resolve_weighted_average_price(symbol_filters, execution_price):
    reference_price = symbol_filters.get("weighted_average_price")
    if reference_price is None:
        reference_price = symbol_filters.get("reference_price")
    if reference_price is None:
        return float(execution_price)
    reference_price = float(reference_price)
    if not np.isfinite(reference_price) or reference_price <= 0.0:
        return float(execution_price)
    return reference_price


def _validate_order_intent(side, order_type, price, quantity, current_quantity,
                           symbol_filters=None, market="spot",
                           weighted_average_price=None):
    symbol_filters = dict(symbol_filters or {})
    normalized_market = str(market or "spot").lower()
    price = float(price)
    quantity = max(float(quantity), 0.0)
    current_quantity = float(current_quantity)
    side = str(side or "BUY").upper()
    order_type = str(order_type or "market").lower()
    weighted_average_price = float(weighted_average_price if weighted_average_price is not None else price)
    adjustment_reasons = []

    def _reject(reason):
        return {
            "status": "rejected",
            "reason": reason,
            "quantity": 0.0,
            "adjustment_reasons": adjustment_reasons,
            "notional": 0.0,
        }

    def _adjust(new_quantity, reason):
        nonlocal quantity
        new_quantity = max(float(new_quantity), 0.0)
        if not np.isclose(new_quantity, quantity, rtol=1e-9, atol=1e-12):
            quantity = new_quantity
            adjustment_reasons.append(reason)

    min_price = _enabled_limit(symbol_filters.get("min_price"))
    max_price = _enabled_limit(symbol_filters.get("max_price"))
    if min_price is not None and price < min_price - 1e-12:
        return _reject("min_price")
    if max_price is not None and price > max_price + 1e-12:
        return _reject("max_price")

    percent_price_by_side = symbol_filters.get("percent_price_by_side")
    if percent_price_by_side:
        if side == "BUY":
            lower = weighted_average_price * float(percent_price_by_side.get("bid_multiplier_down", 0.0))
            upper = weighted_average_price * float(percent_price_by_side.get("bid_multiplier_up", np.inf))
        else:
            lower = weighted_average_price * float(percent_price_by_side.get("ask_multiplier_down", 0.0))
            upper = weighted_average_price * float(percent_price_by_side.get("ask_multiplier_up", np.inf))
        if price < lower - 1e-12 or price > upper + 1e-12:
            return _reject("percent_price_by_side")
    elif symbol_filters.get("percent_price"):
        percent_price = symbol_filters["percent_price"]
        lower = weighted_average_price * float(percent_price.get("multiplier_down", 0.0))
        upper = weighted_average_price * float(percent_price.get("multiplier_up", np.inf))
        if price < lower - 1e-12 or price > upper + 1e-12:
            return _reject("percent_price")

    market_filter_active = order_type == "market" and any(
        symbol_filters.get(key) is not None for key in ["market_min_qty", "market_max_qty", "market_step_size"]
    )
    min_qty = _enabled_limit(symbol_filters.get("market_min_qty" if market_filter_active else "min_qty"))
    max_qty = _enabled_limit(symbol_filters.get("market_max_qty" if market_filter_active else "max_qty"))
    step_size = _enabled_limit(symbol_filters.get("market_step_size" if market_filter_active else "step_size"))
    lot_reason = "market_lot_size" if market_filter_active else "lot_size"

    if max_qty is not None and quantity > max_qty:
        _adjust(max_qty, lot_reason)

    max_position = _enabled_limit(symbol_filters.get("max_position"))
    if max_position is not None and normalized_market == "spot" and side == "BUY":
        allowed_quantity = max_position - current_quantity
        if allowed_quantity <= 0.0:
            return _reject("max_position")
        if quantity > allowed_quantity:
            _adjust(allowed_quantity, "max_position")

    max_notional = _enabled_limit(symbol_filters.get("max_notional"))
    apply_max_to_market = bool(symbol_filters.get("notional_apply_max_to_market", True))
    if max_notional is not None and (order_type != "market" or apply_max_to_market):
        max_quantity_for_notional = max_notional / max(weighted_average_price, 1e-12)
        if quantity * weighted_average_price > max_notional + 1e-12:
            _adjust(max_quantity_for_notional, "max_notional")

    if step_size is not None and quantity > 0.0:
        _adjust(float(_round_down_to_step(np.asarray([quantity]), step_size)[0]), lot_reason)

    if quantity <= 0.0:
        return _reject(adjustment_reasons[-1] if adjustment_reasons else lot_reason)
    if min_qty is not None and quantity < min_qty - 1e-12:
        return _reject(lot_reason)

    min_notional = _enabled_limit(symbol_filters.get("min_notional"))
    min_notional_applies = bool(symbol_filters.get("min_notional_apply_to_market", True))
    if min_notional is not None and (order_type != "market" or min_notional_applies):
        if quantity * weighted_average_price < min_notional - 1e-12:
            return _reject("min_notional")

    min_notional_floor = _enabled_limit(symbol_filters.get("notional_min_notional"))
    apply_min_to_market = bool(symbol_filters.get("notional_apply_min_to_market", True))
    if min_notional_floor is not None and (order_type != "market" or apply_min_to_market):
        if quantity * weighted_average_price < min_notional_floor - 1e-12:
            return _reject("min_notional")

    return {
        "status": "adjusted" if adjustment_reasons else "accepted",
        "reason": adjustment_reasons[0] if adjustment_reasons else None,
        "quantity": quantity,
        "adjustment_reasons": adjustment_reasons,
        "notional": quantity * price,
    }


def _build_legacy_execution_contract(close, requested_position, equity, execution_prices=None,
                                     symbol_filters=None, market="spot"):
    symbol_filters = dict(symbol_filters or {})
    valuation_series = pd.Series(close, copy=False).astype(float)
    raw_execution = execution_prices if execution_prices is not None else close
    execution_series = pd.Series(raw_execution, index=valuation_series.index).reindex(valuation_series.index).astype(float)
    requested_position = pd.Series(requested_position, index=valuation_series.index).reindex(valuation_series.index).fillna(0.0).astype(float)

    executable_position = pd.Series(0.0, index=valuation_series.index, dtype=float)
    order_rows = []
    current_position = 0.0
    current_quantity = 0.0
    working_equity = float(equity)
    previous_valuation_price = None

    for timestamp in valuation_series.index:
        valuation_price = float(valuation_series.loc[timestamp])
        execution_price = float(execution_series.loc[timestamp])
        if previous_valuation_price is not None and previous_valuation_price > 0.0:
            valuation_return = (valuation_price / previous_valuation_price) - 1.0
            working_equity *= 1.0 + current_position * valuation_return

        desired_position = float(requested_position.loc[timestamp])
        executed_position = current_position
        status = "accepted"
        reason = None
        requested_order_quantity = 0.0
        executed_order_quantity = 0.0
        requested_notional = 0.0
        executed_notional = 0.0
        side = None
        adjustment_reasons = []

        if execution_price <= 0.0 or not np.isfinite(execution_price):
            executed_position = current_position
            status = "rejected"
            reason = "invalid_execution_price"
        elif np.isclose(desired_position, current_position, rtol=1e-9, atol=1e-12):
            status = "noop"
            executed_position = current_position
        else:
            desired_quantity = desired_position * max(working_equity, 0.0) / max(execution_price, 1e-12)
            delta_quantity = desired_quantity - current_quantity
            side = "BUY" if delta_quantity > 0.0 else "SELL"
            requested_order_quantity = abs(float(delta_quantity))
            requested_notional = requested_order_quantity * execution_price
            validation = _validate_order_intent(
                side=side,
                order_type="market",
                price=execution_price,
                quantity=requested_order_quantity,
                current_quantity=current_quantity,
                symbol_filters=symbol_filters,
                market=market,
                weighted_average_price=_resolve_weighted_average_price(symbol_filters, execution_price),
            )
            status = validation["status"]
            reason = validation["reason"]
            adjustment_reasons = validation.get("adjustment_reasons", [])
            if status == "rejected":
                executed_position = current_position
            else:
                executed_order_quantity = float(validation["quantity"])
                executed_notional = float(validation["notional"])
                signed_delta = executed_order_quantity if side == "BUY" else -executed_order_quantity
                current_quantity = current_quantity + signed_delta
                if np.isclose(current_quantity, 0.0, rtol=1e-9, atol=1e-12):
                    current_quantity = 0.0
                executed_position = current_quantity * execution_price / max(working_equity, 1e-12)
                if np.isclose(executed_position, 0.0, rtol=1e-9, atol=1e-12):
                    executed_position = 0.0

        executable_position.loc[timestamp] = executed_position
        if status != "noop":
            order_rows.append(
                {
                    "timestamp": timestamp,
                    "requested_position": desired_position,
                    "executed_position": float(executed_position),
                    "previous_position": float(current_position),
                    "side": side,
                    "requested_order_quantity": float(requested_order_quantity),
                    "executed_order_quantity": float(executed_order_quantity),
                    "requested_notional": float(requested_notional),
                    "executed_notional": float(executed_notional),
                    "execution_price": execution_price,
                    "status": status,
                    "reason": reason,
                    "adjustment_reasons": adjustment_reasons,
                }
            )
        current_position = float(executed_position)
        previous_valuation_price = valuation_price

    order_ledger = pd.DataFrame(order_rows)
    rejected_mask = order_ledger["status"] == "rejected" if not order_ledger.empty else pd.Series(dtype=bool)
    total_requested_notional = float(order_ledger.get("requested_notional", pd.Series(dtype=float)).sum()) if not order_ledger.empty else 0.0
    blocked_notional = float(order_ledger.loc[rejected_mask, "requested_notional"].sum()) if not order_ledger.empty else 0.0
    return {
        "valuation_series": valuation_series,
        "execution_series": execution_series,
        "requested_position": requested_position,
        "position": executable_position,
        "order_ledger": order_ledger,
        "blocked_orders": int((order_ledger.get("status") == "rejected").sum()) if not order_ledger.empty else 0,
        "adjusted_orders": int((order_ledger.get("status") == "adjusted").sum()) if not order_ledger.empty else 0,
        "accepted_orders": int((order_ledger.get("status") == "accepted").sum()) if not order_ledger.empty else 0,
        "blocked_notional_share": _round_metric(_safe_ratio(blocked_notional, total_requested_notional), 6),
        "order_rejection_reasons": (
            order_ledger.loc[order_ledger["status"] == "rejected", "reason"].value_counts().to_dict()
            if not order_ledger.empty
            else {}
        ),
    }


def _build_execution_contract(close, requested_position, equity, execution_prices=None,
                              symbol_filters=None, market="spot", volume=None,
                              execution_policy=None, scenario_schedule=None,
                              scenario_policy=None):
    policy = resolve_execution_policy(execution_policy)
    effective_order_type = "market" if policy.order_type in {"market", "aggressive"} else policy.order_type
    if effective_order_type != "market":
        raise ValueError(
            "Passive or limit order types are not yet supported in the default bar execution engine; "
            "Phase 5 is restricted to market/aggressive execution."
        )
    if policy.adapter == "legacy":
        report = _build_legacy_execution_contract(
            close,
            requested_position,
            equity,
            execution_prices=execution_prices,
            symbol_filters=symbol_filters,
            market=market,
        )
        requested_total = float(report["order_ledger"].get("requested_notional", pd.Series(dtype=float)).sum()) if not report["order_ledger"].empty else 0.0
        executed_total = float(report["order_ledger"].get("executed_notional", pd.Series(dtype=float)).sum()) if not report["order_ledger"].empty else 0.0
        report["order_intents"] = report["order_ledger"].copy()
        report["execution_adapter"] = "legacy"
        report["execution_backend"] = "legacy"
        report["execution_policy"] = policy.to_dict()
        report["partial_fill_orders"] = 0
        report["cancelled_orders"] = 0
        report["unfilled_notional"] = max(0.0, requested_total - executed_total)
        report["fill_ratio"] = _round_metric(_safe_ratio(executed_total, requested_total), 6)
        return report

    symbol_filters = dict(symbol_filters or {})
    valuation_series = pd.Series(close, copy=False).astype(float)
    raw_execution = execution_prices if execution_prices is not None else close
    execution_series = pd.Series(raw_execution, index=valuation_series.index).reindex(valuation_series.index).astype(float)
    requested_position = pd.Series(requested_position, index=valuation_series.index).reindex(valuation_series.index).fillna(0.0).astype(float)
    requested_position, request_timestamps = _apply_action_latency(
        requested_position,
        policy.action_latency_bars,
    )
    if volume is None:
        volume_series = pd.Series(np.nan, index=valuation_series.index, dtype=float)
    else:
        volume_series = (
            pd.Series(volume, index=valuation_series.index)
            .reindex(valuation_series.index)
            .fillna(0.0)
            .astype(float)
            .clip(lower=0.0)
        )

    executable_position = pd.Series(0.0, index=valuation_series.index, dtype=float)
    order_rows = []
    intent_rows = []
    current_position = 0.0
    current_quantity = 0.0
    working_equity = float(equity)
    previous_valuation_price = None
    pending_order = None
    adapter_boundary = (
        NautilusExecutionAdapter(
            scenario_schedule=scenario_schedule,
            scenario_policy=scenario_policy,
        )
        if policy.adapter == "nautilus"
        else None
    )
    tolerance = 1e-12

    for bar_loc, timestamp in enumerate(valuation_series.index):
        valuation_price = float(valuation_series.loc[timestamp])
        execution_price = float(execution_series.loc[timestamp])
        if previous_valuation_price is not None and previous_valuation_price > 0.0:
            valuation_return = (valuation_price / previous_valuation_price) - 1.0
            working_equity *= 1.0 + current_position * valuation_return

        if execution_price <= 0.0 or not np.isfinite(execution_price):
            if pending_order is not None:
                order_rows.append(
                    {
                        "timestamp": timestamp,
                        "requested_position": float(pending_order["requested_position"]),
                        "executed_position": float(current_position),
                        "previous_position": float(current_position),
                        "side": pending_order["side"],
                        "requested_order_quantity": float(pending_order["remaining_quantity"]),
                        "executed_order_quantity": 0.0,
                        "requested_notional": float(pending_order["remaining_quantity"] * max(execution_price, 0.0)),
                        "executed_notional": 0.0,
                        "execution_price": execution_price,
                        "status": "cancelled",
                        "reason": "invalid_execution_price",
                        "adjustment_reasons": [],
                    }
                )
                pending_order = None
            executable_position.loc[timestamp] = current_position
            previous_valuation_price = valuation_price
            continue

        available_quantity = float(policy.participation_cap * volume_series.loc[timestamp]) if np.isfinite(volume_series.loc[timestamp]) else np.inf
        if available_quantity < 0.0 or not np.isfinite(available_quantity):
            available_quantity = np.inf

        desired_position = float(requested_position.loc[timestamp])
        request_timestamp = request_timestamps.loc[timestamp]
        request_bar = bar_loc if pd.isna(request_timestamp) else max(0, valuation_series.index.get_loc(pd.Timestamp(request_timestamp)))

        if pending_order is not None:
            pending_age = bar_loc - int(pending_order["submit_bar"])
            if not np.isclose(desired_position, pending_order["requested_position"], rtol=1e-9, atol=1e-12) and pending_age >= policy.cancel_replace_bars - 1:
                remaining_notional = float(pending_order["remaining_quantity"] * execution_price)
                order_rows.append(
                    {
                        "timestamp": timestamp,
                        "requested_position": float(pending_order["requested_position"]),
                        "executed_position": float(current_position),
                        "previous_position": float(current_position),
                        "side": pending_order["side"],
                        "requested_order_quantity": float(pending_order["remaining_quantity"]),
                        "executed_order_quantity": 0.0,
                        "requested_notional": remaining_notional,
                        "executed_notional": 0.0,
                        "execution_price": execution_price,
                        "request_timestamp": pending_order.get("request_timestamp"),
                        "status": "cancelled",
                        "reason": "cancel_replace",
                        "adjustment_reasons": [],
                    }
                )
                pending_order = None

        if pending_order is not None and available_quantity > tolerance:
            previous_position = float(current_position)
            fill_quantity = min(float(pending_order["remaining_quantity"]), available_quantity)
            validation = _validate_order_intent(
                side=pending_order["side"],
                order_type=effective_order_type,
                price=execution_price,
                quantity=fill_quantity,
                current_quantity=current_quantity,
                symbol_filters=symbol_filters,
                market=market,
                weighted_average_price=_resolve_weighted_average_price(symbol_filters, execution_price),
            )
            if validation["status"] == "rejected":
                order_rows.append(
                    {
                        "timestamp": timestamp,
                        "requested_position": float(pending_order["requested_position"]),
                        "executed_position": float(current_position),
                        "previous_position": previous_position,
                        "side": pending_order["side"],
                        "requested_order_quantity": float(fill_quantity),
                        "executed_order_quantity": 0.0,
                        "requested_notional": float(fill_quantity * execution_price),
                        "executed_notional": 0.0,
                        "execution_price": execution_price,
                        "request_timestamp": pending_order.get("request_timestamp"),
                        "status": "rejected",
                        "reason": validation["reason"],
                        "adjustment_reasons": validation.get("adjustment_reasons", []),
                    }
                )
                pending_order = None
            else:
                executed_quantity = float(validation["quantity"])
                executed_notional = float(validation["notional"])
                signed_delta = executed_quantity if pending_order["side"] == "BUY" else -executed_quantity
                current_quantity = current_quantity + signed_delta
                if np.isclose(current_quantity, 0.0, rtol=1e-9, atol=tolerance):
                    current_quantity = 0.0
                current_position = current_quantity * execution_price / max(working_equity, 1e-12)
                if np.isclose(current_position, 0.0, rtol=1e-9, atol=tolerance):
                    current_position = 0.0
                pending_order["remaining_quantity"] = max(0.0, float(pending_order["remaining_quantity"]) - executed_quantity)
                available_quantity = max(0.0, available_quantity - executed_quantity)
                order_rows.append(
                    {
                        "timestamp": timestamp,
                        "requested_position": float(pending_order["requested_position"]),
                        "executed_position": float(current_position),
                        "previous_position": previous_position,
                        "side": pending_order["side"],
                        "requested_order_quantity": float(fill_quantity),
                        "executed_order_quantity": executed_quantity,
                        "requested_notional": float(fill_quantity * execution_price),
                        "executed_notional": executed_notional,
                        "execution_price": execution_price,
                        "request_timestamp": pending_order.get("request_timestamp"),
                        "status": "partial_fill" if pending_order["remaining_quantity"] > tolerance else "accepted",
                        "reason": validation["reason"],
                        "adjustment_reasons": validation.get("adjustment_reasons", []),
                    }
                )
                pending_age = bar_loc - int(pending_order["submit_bar"])
                if pending_order["remaining_quantity"] <= tolerance:
                    pending_order = None
                elif pending_age >= policy.max_order_age_bars - 1:
                    order_rows.append(
                        {
                            "timestamp": timestamp,
                            "requested_position": float(pending_order["requested_position"]),
                            "executed_position": float(current_position),
                            "previous_position": float(current_position),
                            "side": pending_order["side"],
                            "requested_order_quantity": float(pending_order["remaining_quantity"]),
                            "executed_order_quantity": 0.0,
                            "requested_notional": float(pending_order["remaining_quantity"] * execution_price),
                            "executed_notional": 0.0,
                            "execution_price": execution_price,
                            "request_timestamp": pending_order.get("request_timestamp"),
                            "status": "cancelled",
                            "reason": "max_order_age",
                            "adjustment_reasons": [],
                        }
                    )
                    pending_order = None

        if pending_order is not None:
            pending_age = bar_loc - int(pending_order["submit_bar"])
            if pending_age >= policy.max_order_age_bars - 1:
                order_rows.append(
                    {
                        "timestamp": timestamp,
                        "requested_position": float(pending_order["requested_position"]),
                        "executed_position": float(current_position),
                        "previous_position": float(current_position),
                        "side": pending_order["side"],
                        "requested_order_quantity": float(pending_order["remaining_quantity"]),
                        "executed_order_quantity": 0.0,
                        "requested_notional": float(pending_order["remaining_quantity"] * execution_price),
                        "executed_notional": 0.0,
                        "execution_price": execution_price,
                        "request_timestamp": pending_order.get("request_timestamp"),
                        "status": "cancelled",
                        "reason": "max_order_age",
                        "adjustment_reasons": [],
                    }
                )
                pending_order = None

        if pending_order is None and not np.isclose(desired_position, current_position, rtol=1e-9, atol=1e-12):
            desired_quantity = desired_position * max(working_equity, 0.0) / max(execution_price, 1e-12)
            delta_quantity = desired_quantity - current_quantity
            side = "BUY" if delta_quantity > 0.0 else "SELL"
            requested_order_quantity = abs(float(delta_quantity))
            requested_notional = requested_order_quantity * execution_price
            if requested_order_quantity > tolerance:
                intent = OrderIntent(
                    timestamp=timestamp,
                    request_timestamp=None if pd.isna(request_timestamp) else pd.Timestamp(request_timestamp),
                    side=side,
                    order_type=effective_order_type,
                    time_in_force=policy.time_in_force,
                    requested_position=desired_position,
                    previous_position=float(current_position),
                    requested_order_quantity=float(requested_order_quantity),
                    requested_notional=float(requested_notional),
                    execution_price=float(execution_price),
                    participation_cap=float(policy.participation_cap),
                    min_fill_ratio=float(policy.min_fill_ratio),
                    action_latency_bars=int(policy.action_latency_bars),
                    max_order_age_bars=int(policy.max_order_age_bars),
                    cancel_replace_bars=int(policy.cancel_replace_bars),
                )
                intent_rows.append(intent.to_dict())
                validation = _validate_order_intent(
                    side=side,
                    order_type=effective_order_type,
                    price=execution_price,
                    quantity=requested_order_quantity,
                    current_quantity=current_quantity,
                    symbol_filters=symbol_filters,
                    market=market,
                    weighted_average_price=_resolve_weighted_average_price(symbol_filters, execution_price),
                )
                base_status = validation["status"]
                if base_status == "rejected":
                    order_rows.append(
                        {
                            "timestamp": timestamp,
                            "requested_position": desired_position,
                            "executed_position": float(current_position),
                            "previous_position": float(current_position),
                            "side": side,
                            "requested_order_quantity": float(requested_order_quantity),
                            "executed_order_quantity": 0.0,
                            "requested_notional": float(requested_notional),
                            "executed_notional": 0.0,
                            "execution_price": execution_price,
                            "request_timestamp": None if pd.isna(request_timestamp) else pd.Timestamp(request_timestamp),
                            "status": "rejected",
                            "reason": validation["reason"],
                            "adjustment_reasons": validation.get("adjustment_reasons", []),
                        }
                    )
                else:
                    validated_quantity = float(validation["quantity"])
                    immediately_fillable = min(validated_quantity, available_quantity)
                    immediate_fill_ratio = _safe_ratio(immediately_fillable, validated_quantity)
                    if validated_quantity <= tolerance or immediate_fill_ratio < policy.min_fill_ratio - 1e-12:
                        order_rows.append(
                            {
                                "timestamp": timestamp,
                                "requested_position": desired_position,
                                "executed_position": float(current_position),
                                "previous_position": float(current_position),
                                "side": side,
                                "requested_order_quantity": float(validated_quantity),
                                "executed_order_quantity": 0.0,
                                "requested_notional": float(validated_quantity * execution_price),
                                "executed_notional": 0.0,
                                "execution_price": execution_price,
                                "request_timestamp": None if pd.isna(request_timestamp) else pd.Timestamp(request_timestamp),
                                "status": "cancelled",
                                "reason": "min_fill_ratio",
                                "adjustment_reasons": validation.get("adjustment_reasons", []),
                            }
                        )
                    else:
                        executed_quantity = immediately_fillable
                        executed_notional = executed_quantity * execution_price
                        previous_position = float(current_position)
                        if executed_quantity > tolerance:
                            signed_delta = executed_quantity if side == "BUY" else -executed_quantity
                            current_quantity = current_quantity + signed_delta
                            if np.isclose(current_quantity, 0.0, rtol=1e-9, atol=tolerance):
                                current_quantity = 0.0
                            current_position = current_quantity * execution_price / max(working_equity, 1e-12)
                            if np.isclose(current_position, 0.0, rtol=1e-9, atol=tolerance):
                                current_position = 0.0
                            available_quantity = max(0.0, available_quantity - executed_quantity)
                        remaining_quantity = max(0.0, validated_quantity - executed_quantity)
                        order_rows.append(
                            {
                                "timestamp": timestamp,
                                "requested_position": desired_position,
                                "executed_position": float(current_position),
                                "previous_position": previous_position,
                                "side": side,
                                "requested_order_quantity": float(validated_quantity),
                                "executed_order_quantity": float(executed_quantity),
                                "requested_notional": float(validated_quantity * execution_price),
                                "executed_notional": float(executed_notional),
                                "execution_price": execution_price,
                                "request_timestamp": None if pd.isna(request_timestamp) else pd.Timestamp(request_timestamp),
                                "status": "partial_fill" if remaining_quantity > tolerance else base_status,
                                "reason": validation["reason"],
                                "adjustment_reasons": validation.get("adjustment_reasons", []),
                            }
                        )
                        if remaining_quantity > tolerance:
                            if policy.time_in_force == "IOC":
                                order_rows.append(
                                    {
                                        "timestamp": timestamp,
                                        "requested_position": desired_position,
                                        "executed_position": float(current_position),
                                        "previous_position": float(current_position),
                                        "side": side,
                                        "requested_order_quantity": float(remaining_quantity),
                                        "executed_order_quantity": 0.0,
                                        "requested_notional": float(remaining_quantity * execution_price),
                                        "executed_notional": 0.0,
                                        "execution_price": execution_price,
                                        "request_timestamp": None if pd.isna(request_timestamp) else pd.Timestamp(request_timestamp),
                                        "status": "cancelled",
                                        "reason": "ioc_unfilled",
                                        "adjustment_reasons": [],
                                    }
                                )
                            else:
                                pending_order = {
                                    "requested_position": desired_position,
                                    "side": side,
                                    "remaining_quantity": float(remaining_quantity),
                                    "request_timestamp": None if pd.isna(request_timestamp) else pd.Timestamp(request_timestamp),
                                    "request_bar": int(request_bar),
                                    "submit_bar": int(bar_loc),
                                }

        executable_position.loc[timestamp] = float(current_position)
        previous_valuation_price = valuation_price

    if pending_order is not None and not valuation_series.empty:
        last_timestamp = valuation_series.index[-1]
        last_execution_price = float(execution_series.iloc[-1])
        order_rows.append(
            {
                "timestamp": last_timestamp,
                "requested_position": float(pending_order["requested_position"]),
                "executed_position": float(current_position),
                "previous_position": float(current_position),
                "side": pending_order["side"],
                "requested_order_quantity": float(pending_order["remaining_quantity"]),
                "executed_order_quantity": 0.0,
                "requested_notional": float(pending_order["remaining_quantity"] * last_execution_price),
                "executed_notional": 0.0,
                "execution_price": last_execution_price,
                "status": "cancelled",
                "reason": "end_of_backtest",
                "adjustment_reasons": [],
            }
        )

    order_ledger = pd.DataFrame(order_rows)
    order_intents = pd.DataFrame(intent_rows)
    rejected_mask = order_ledger["status"] == "rejected" if not order_ledger.empty else pd.Series(dtype=bool)
    total_requested_notional = float(order_intents.get("requested_notional", pd.Series(dtype=float)).sum()) if not order_intents.empty else 0.0
    blocked_notional = float(order_ledger.loc[rejected_mask, "requested_notional"].sum()) if not order_ledger.empty else 0.0
    executed_notional = float(order_ledger.get("executed_notional", pd.Series(dtype=float)).sum()) if not order_ledger.empty else 0.0
    delay_metrics = _summarize_execution_delay_metrics(order_intents, order_ledger, valuation_series.index)
    return {
        "valuation_series": valuation_series,
        "execution_series": execution_series,
        "requested_position": requested_position,
        "position": executable_position,
        "order_intents": order_intents,
        "order_ledger": order_ledger,
        "execution_adapter": policy.adapter,
        "execution_backend": adapter_boundary.backend if adapter_boundary is not None else policy.adapter,
        "execution_adapter_scenarios": adapter_boundary.describe_scenarios() if adapter_boundary is not None else {},
        "execution_policy": policy.to_dict(),
        "blocked_orders": int((order_ledger.get("status") == "rejected").sum()) if not order_ledger.empty else 0,
        "adjusted_orders": int(((order_ledger.get("status").isin(["adjusted", "partial_fill"])) & order_ledger.get("adjustment_reasons").map(bool)).sum()) if not order_ledger.empty and "adjustment_reasons" in order_ledger else 0,
        "accepted_orders": int((order_ledger.get("status") == "accepted").sum()) if not order_ledger.empty else 0,
        "partial_fill_orders": int((order_ledger.get("status") == "partial_fill").sum()) if not order_ledger.empty else 0,
        "cancelled_orders": int((order_ledger.get("status") == "cancelled").sum()) if not order_ledger.empty else 0,
        "unfilled_notional": max(0.0, total_requested_notional - executed_notional),
        "fill_ratio": _round_metric(_safe_ratio(executed_notional, total_requested_notional), 6),
        "average_action_delay_bars": _round_metric(delay_metrics["average_action_delay_bars"], 6),
        "average_fill_delay_bars": _round_metric(delay_metrics["average_fill_delay_bars"], 6),
        "max_fill_delay_bars": int(delay_metrics["max_fill_delay_bars"]),
        "blocked_notional_share": _round_metric(_safe_ratio(blocked_notional, total_requested_notional), 6),
        "order_rejection_reasons": (
            order_ledger.loc[order_ledger["status"] == "rejected", "reason"].value_counts().to_dict()
            if not order_ledger.empty
            else {}
        ),
    }


def _annualized_sharpe(returns, annualization):
    if annualization <= 0 or len(returns) < 2:
        return 0.0
    volatility = float(np.std(returns, ddof=1))
    if volatility <= 0:
        return 0.0
    return float(np.mean(returns) / volatility * annualization)


def _annualized_sortino(returns, annualization):
    if annualization <= 0 or len(returns) < 2:
        return 0.0
    downside = np.where(np.asarray(returns, dtype=float) < 0.0, returns, 0.0)
    downside_vol = float(np.std(downside, ddof=1))
    if downside_vol <= 0:
        return 0.0
    return float(np.mean(returns) / downside_vol * annualization)


def _compute_total_return(equity_curve, starting_equity):
    if len(equity_curve) == 0 or starting_equity <= 0:
        return 0.0
    return float(equity_curve[-1] / starting_equity - 1.0)


def _compute_max_drawdown_from_equity(equity_curve):
    if len(equity_curve) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    return float(np.min((equity_curve - peak) / peak))


def _compute_cagr_from_equity(equity_curve, starting_equity, elapsed_years):
    if len(equity_curve) == 0 or elapsed_years <= 0 or starting_equity <= 0 or equity_curve[-1] <= 0:
        return 0.0
    return float((equity_curve[-1] / starting_equity) ** (1.0 / elapsed_years) - 1.0)


def _compute_calmar(cagr, max_drawdown):
    return float(_safe_ratio(cagr, abs(max_drawdown)))


def _resolve_significance_config(significance):
    if significance is None:
        return dict(_DEFAULT_SIGNIFICANCE_CONFIG)
    if isinstance(significance, bool):
        return {**_DEFAULT_SIGNIFICANCE_CONFIG, "enabled": significance}
    if not isinstance(significance, dict):
        raise TypeError("significance must be None, a bool, or a dict")
    return {**_DEFAULT_SIGNIFICANCE_CONFIG, **dict(significance)}


def _default_mean_block_length(sample_size):
    if sample_size <= 1:
        return 1
    return max(2, min(sample_size, int(round(sample_size ** (1.0 / 3.0)))))


def _stationary_bootstrap_indices(sample_size, mean_block_length, rng):
    if sample_size <= 0:
        return np.array([], dtype=int)

    mean_block_length = max(1, int(mean_block_length))
    restart_probability = min(1.0, 1.0 / float(mean_block_length))
    indices = np.empty(sample_size, dtype=int)
    indices[0] = int(rng.integers(0, sample_size))

    for loc in range(1, sample_size):
        if rng.random() < restart_probability:
            indices[loc] = int(rng.integers(0, sample_size))
        else:
            indices[loc] = (indices[loc - 1] + 1) % sample_size

    return indices


def _build_bootstrap_interval(samples, confidence_level, digits=4):
    finite = np.asarray(samples, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None

    alpha = (1.0 - float(confidence_level)) / 2.0
    lower, upper = np.quantile(finite, [alpha, 1.0 - alpha])
    return {
        "lower": _round_metric(float(lower), digits),
        "upper": _round_metric(float(upper), digits),
        "confidence_level": float(confidence_level),
    }


def _centered_bootstrap_p_value(samples, observed_estimate, threshold):
    finite = np.asarray(samples, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0 or observed_estimate is None or threshold is None:
        return None

    observed_estimate = float(observed_estimate)
    threshold = float(threshold)
    deviations = finite - observed_estimate
    test_statistic = observed_estimate - threshold
    equal_mask = np.isclose(deviations, test_statistic, rtol=1e-9, atol=1e-12)
    tail_probability = float(np.mean(deviations > test_statistic) + 0.5 * np.mean(equal_mask))
    return max(0.0, min(1.0, tail_probability))


def _align_benchmark_returns(benchmark_returns, index):
    if benchmark_returns is None:
        return None

    if isinstance(benchmark_returns, pd.Series):
        aligned = pd.Series(benchmark_returns, copy=False).reindex(index)
    else:
        aligned = pd.Series(benchmark_returns, index=index)

    if aligned.isna().any():
        raise ValueError("benchmark_returns must cover every backtest timestamp")

    return aligned.astype(float)


def _compute_significance_metrics(strat_ret, equity, periods_per_year, elapsed_years,
                                  sharpe, sortino, calmar, total_ret, max_dd,
                                  significance=None, benchmark_returns=None,
                                  benchmark_sharpe=None):
    config = _resolve_significance_config(significance)
    payload = {
        "enabled": bool(config.get("enabled", True)),
        "method": str(config.get("method", "stationary_bootstrap")),
        "bootstrap_samples": int(config.get("bootstrap_samples", 500)),
        "confidence_level": float(config.get("confidence_level", 0.95)),
        "mean_block_length": None,
        "random_state": config.get("random_state"),
        "benchmark_sharpe_ratio": None,
        "metrics": {},
    }
    if not payload["enabled"]:
        payload["reason"] = "disabled"
        return payload

    if payload["bootstrap_samples"] <= 0:
        raise ValueError("bootstrap_samples must be greater than zero")
    if not 0.0 < payload["confidence_level"] < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    method = payload["method"].lower()
    if method not in {"stationary", "stationary_bootstrap"}:
        raise ValueError(f"unsupported significance method: {payload['method']}")
    payload["method"] = "stationary_bootstrap"

    strategy_returns = pd.Series(strat_ret, copy=False).astype(float)
    benchmark_series = _align_benchmark_returns(benchmark_returns, strategy_returns.index)

    finite_mask = np.isfinite(strategy_returns.to_numpy())
    if benchmark_series is not None:
        finite_mask &= np.isfinite(benchmark_series.to_numpy())

    strategy_returns = strategy_returns.loc[finite_mask]
    benchmark_series = benchmark_series.loc[finite_mask] if benchmark_series is not None else None

    min_observations = max(2, int(config.get("min_observations", 8)))
    if len(strategy_returns) < min_observations:
        payload["enabled"] = False
        payload["reason"] = "insufficient_observations"
        return payload

    mean_block_length = config.get("mean_block_length")
    if mean_block_length is None:
        mean_block_length = _default_mean_block_length(len(strategy_returns))
    mean_block_length = max(1, int(mean_block_length))
    payload["mean_block_length"] = mean_block_length

    annualization = np.sqrt(periods_per_year) if periods_per_year > 0 else 0.0
    rng = np.random.default_rng(config.get("random_state"))
    strategy_values = strategy_returns.to_numpy(dtype=float)
    benchmark_values = benchmark_series.to_numpy(dtype=float) if benchmark_series is not None else None

    bootstrap_metrics = {
        "sharpe_ratio": np.empty(payload["bootstrap_samples"], dtype=float),
        "sortino_ratio": np.empty(payload["bootstrap_samples"], dtype=float),
        "calmar_ratio": np.empty(payload["bootstrap_samples"], dtype=float),
        "net_profit_pct": np.empty(payload["bootstrap_samples"], dtype=float),
        "max_drawdown": np.empty(payload["bootstrap_samples"], dtype=float),
    }
    benchmark_sharpes = np.empty(payload["bootstrap_samples"], dtype=float) if benchmark_values is not None else None

    for sample_idx in range(payload["bootstrap_samples"]):
        sampled_idx = _stationary_bootstrap_indices(len(strategy_values), mean_block_length, rng)
        sampled_returns = strategy_values[sampled_idx]
        sampled_equity = equity * np.cumprod(1.0 + sampled_returns)
        sampled_total_ret = _compute_total_return(sampled_equity, equity)
        sampled_max_dd = _compute_max_drawdown_from_equity(sampled_equity)
        sampled_cagr = _compute_cagr_from_equity(sampled_equity, equity, elapsed_years)

        bootstrap_metrics["sharpe_ratio"][sample_idx] = _annualized_sharpe(sampled_returns, annualization)
        bootstrap_metrics["sortino_ratio"][sample_idx] = _annualized_sortino(sampled_returns, annualization)
        bootstrap_metrics["calmar_ratio"][sample_idx] = _compute_calmar(sampled_cagr, sampled_max_dd)
        bootstrap_metrics["net_profit_pct"][sample_idx] = sampled_total_ret
        bootstrap_metrics["max_drawdown"][sample_idx] = sampled_max_dd

        if benchmark_sharpes is not None:
            benchmark_sharpes[sample_idx] = _annualized_sharpe(benchmark_values[sampled_idx], annualization)

    observed_metrics = {
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "calmar_ratio": float(calmar),
        "net_profit_pct": float(total_ret),
        "max_drawdown": float(max_dd),
    }

    observed_benchmark_sharpe = None
    if benchmark_series is not None:
        observed_benchmark_sharpe = _annualized_sharpe(benchmark_values, annualization)
        payload["benchmark_sharpe_ratio"] = _round_metric(observed_benchmark_sharpe, 4)
    elif benchmark_sharpe is not None:
        observed_benchmark_sharpe = float(benchmark_sharpe)
        payload["benchmark_sharpe_ratio"] = _round_metric(observed_benchmark_sharpe, 4)

    for metric_name, point_estimate in observed_metrics.items():
        metric_payload = {
            "point_estimate": _round_metric(point_estimate, 4),
            "confidence_interval": _build_bootstrap_interval(
                bootstrap_metrics[metric_name],
                payload["confidence_level"],
                digits=4,
            ),
        }
        if metric_name == "sharpe_ratio":
            metric_payload["p_value_gt_zero"] = _round_metric(
                _centered_bootstrap_p_value(bootstrap_metrics[metric_name], point_estimate, 0.0),
                6,
            )
            if benchmark_sharpes is not None:
                observed_diff = point_estimate - float(observed_benchmark_sharpe)
                bootstrap_diffs = bootstrap_metrics[metric_name] - benchmark_sharpes
                metric_payload["p_value_gt_benchmark"] = _round_metric(
                    _centered_bootstrap_p_value(bootstrap_diffs, observed_diff, 0.0),
                    6,
                )
            elif payload["benchmark_sharpe_ratio"] is not None:
                metric_payload["p_value_gt_benchmark"] = _round_metric(
                    _centered_bootstrap_p_value(
                        bootstrap_metrics[metric_name],
                        point_estimate,
                        float(payload["benchmark_sharpe_ratio"]),
                    ),
                    6,
                )
        payload["metrics"][metric_name] = metric_payload

    return payload


def _vectorbt_trade_ledger(portfolio, index):
    readable = portfolio.trades.records_readable
    if readable.empty:
        return pd.DataFrame(columns=["entry_time", "exit_time", "direction", "bars", "entry_price", "exit_price", "return_pct"])

    entry_time = pd.to_datetime(readable["Entry Timestamp"], utc=True)
    exit_time = pd.to_datetime(readable["Exit Timestamp"], utc=True)
    entry_loc = index.get_indexer(entry_time)
    exit_loc = index.get_indexer(exit_time)
    bars = np.where((entry_loc >= 0) & (exit_loc >= 0), exit_loc - entry_loc + 1, np.nan)

    ledger = pd.DataFrame(
        {
            "entry_time": entry_time,
            "exit_time": exit_time,
            "direction": readable["Direction"].map({"Long": 1, "Short": -1}).fillna(0).astype(int),
            "bars": pd.Series(bars, dtype="float").fillna(0).astype(int),
            "entry_price": pd.to_numeric(readable["Avg Entry Price"], errors="coerce"),
            "exit_price": pd.to_numeric(readable["Avg Exit Price"], errors="coerce"),
            "return_pct": pd.to_numeric(readable["Return"], errors="coerce"),
        }
    )
    return ledger


def _compute_funding_cash(position, funding_rates, equity_curve):
    if funding_rates is None:
        return pd.Series(0.0, index=position.index, dtype=float)

    funding = pd.Series(funding_rates, index=position.index).reindex(position.index).fillna(0.0).astype(float)
    prev_equity = equity_curve.shift(1).fillna(equity_curve.iloc[0])
    return prev_equity * (-position.astype(float) * funding)


def _normalize_futures_account_config(futures_account=None, market="spot", leverage=1.0,
                                      contract_spec=None, leverage_brackets=None):
    normalized_market = str(market or "spot").lower()
    configured = dict(futures_account or {})
    if contract_spec is not None and configured.get("contract_spec") is None:
        configured["contract_spec"] = contract_spec
    if leverage_brackets is not None and configured.get("leverage_brackets") is None:
        configured["leverage_brackets"] = leverage_brackets

    enabled = configured.get("enabled")
    if enabled is None:
        enabled = normalized_market != "spot" and bool(configured)
    if not enabled:
        return None

    margin_mode = str(configured.get("margin_mode", "isolated")).lower()
    if margin_mode not in {"isolated", "cross"}:
        raise ValueError("futures_account.margin_mode must be 'isolated' or 'cross'")
    margin_safety_mode = str(configured.get("margin_safety_mode", "diagnostic")).lower()
    if margin_safety_mode not in {"diagnostic", "blocking"}:
        raise ValueError("futures_account.margin_safety_mode must be 'diagnostic' or 'blocking'")

    bracket_payload = configured.get("leverage_brackets") or {}
    if isinstance(bracket_payload, dict):
        bracket_rows = list(bracket_payload.get("brackets", []) or [])
        notional_coef = float(bracket_payload.get("notional_coef", bracket_payload.get("notionalCoef", 1.0)) or 1.0)
    else:
        bracket_rows = list(bracket_payload or [])
        notional_coef = 1.0

    contract_payload = dict(configured.get("contract_spec") or {})
    liquidation_fee_rate = configured.get("liquidation_fee_rate")
    if liquidation_fee_rate is None:
        liquidation_fee_rate = contract_payload.get("liquidation_fee_rate", 0.0)

    return {
        "enabled": True,
        "market": normalized_market,
        "margin_mode": margin_mode,
        "configured_leverage": max(1.0, float(configured.get("leverage", leverage))),
        "warning_margin_ratio": max(0.0, float(configured.get("warning_margin_ratio", 0.8))),
        "margin_safety_mode": margin_safety_mode,
        "maintenance_margin_ratio": max(0.0, float(configured.get("maintenance_margin_ratio", 0.0))),
        "maintenance_amount": max(0.0, float(configured.get("maintenance_amount", 0.0))),
        "liquidation_fee_rate": max(0.0, float(liquidation_fee_rate or 0.0)),
        "allow_reentry_after_liquidation": bool(configured.get("allow_reentry_after_liquidation", False)),
        "contract_spec": contract_payload,
        "leverage_brackets": bracket_rows,
        "notional_coef": notional_coef,
    }


def _resolve_futures_leverage_bracket(notional, futures_account):
    brackets = list((futures_account or {}).get("leverage_brackets") or [])
    if not brackets:
        return None

    scaled_notional = max(float(notional), 0.0)
    coef = float((futures_account or {}).get("notional_coef", 1.0) or 1.0)
    if coef > 0.0:
        scaled_notional /= coef

    fallback = brackets[-1]
    for bracket in brackets:
        floor = bracket.get("notional_floor")
        cap = bracket.get("notional_cap")
        if floor is not None and scaled_notional < float(floor) - 1e-12:
            continue
        fallback = bracket
        if cap is None or scaled_notional <= float(cap) + 1e-12:
            return bracket
    return fallback


def _resolve_futures_leverage_cap(notional, futures_account):
    configured_leverage = max(1.0, float((futures_account or {}).get("configured_leverage", 1.0)))
    bracket = _resolve_futures_leverage_bracket(notional, futures_account)
    if bracket is None:
        return configured_leverage, None

    bracket_leverage = bracket.get("initial_leverage")
    if bracket_leverage is None or bracket_leverage <= 0.0:
        return configured_leverage, bracket
    return min(configured_leverage, float(bracket_leverage)), bracket


def _compute_futures_maintenance_margin(notional, futures_account, bracket=None):
    notional = max(float(notional), 0.0)
    if notional <= 0.0:
        return 0.0

    if bracket is None:
        bracket = _resolve_futures_leverage_bracket(notional, futures_account)
    if bracket is not None:
        ratio = float(bracket.get("maint_margin_ratio") or 0.0)
        cum = float(bracket.get("cum") or 0.0)
    else:
        ratio = float((futures_account or {}).get("maintenance_margin_ratio", 0.0) or 0.0)
        cum = float((futures_account or {}).get("maintenance_amount", 0.0) or 0.0)
    return notional * ratio + cum


def _cap_futures_target_position(target_position, account_equity, futures_account):
    requested = float(target_position)
    if not np.isfinite(requested):
        requested = 0.0

    requested_abs = abs(requested)
    configured_leverage = max(1.0, float((futures_account or {}).get("configured_leverage", 1.0)))
    capped_abs = min(requested_abs, configured_leverage)
    bracket = None

    for _ in range(4):
        leverage_cap, bracket = _resolve_futures_leverage_cap(max(capped_abs * max(float(account_equity), 0.0), 0.0), futures_account)
        leverage_cap = max(1.0, float(leverage_cap))
        if capped_abs <= leverage_cap + 1e-12:
            break
        capped_abs = leverage_cap

    capped = float(np.sign(requested) * capped_abs)
    return capped, {
        "requested_position": requested,
        "capped_position": capped,
        "adjusted": capped_abs < requested_abs - 1e-12,
        "bracket": bracket,
        "leverage_cap": max(1.0, float(_resolve_futures_leverage_cap(capped_abs * max(float(account_equity), 0.0), futures_account)[0])),
    }


def _enforce_futures_margin_safety(target_position, account_equity, futures_account):
    requested = float(target_position)
    warning_ratio = max(0.0, float((futures_account or {}).get("warning_margin_ratio", 0.8) or 0.8))
    margin_mode = str((futures_account or {}).get("margin_mode", "isolated")).lower()
    mode = str((futures_account or {}).get("margin_safety_mode", "blocking")).lower()
    requested_abs = abs(requested)
    safe_abs = requested_abs
    bracket = None
    projected_margin_ratio = 0.0

    if requested_abs <= 1e-12 or account_equity <= 0.0:
        return requested, {
            "mode": mode,
            "requested_position": requested,
            "safe_position": requested,
            "adjusted": False,
            "projected_margin_ratio": 0.0,
            "warning_margin_ratio": warning_ratio,
            "reason": None,
            "bracket": None,
        }

    for _ in range(16):
        projected_notional = safe_abs * max(float(account_equity), 0.0)
        leverage_cap, bracket = _resolve_futures_leverage_cap(projected_notional, futures_account)
        leverage_cap = max(1.0, float(leverage_cap))
        projected_initial_margin = projected_notional / leverage_cap if projected_notional > 0.0 else 0.0
        if margin_mode == "isolated":
            projected_margin_balance = min(projected_initial_margin, max(float(account_equity), 0.0))
        else:
            projected_margin_balance = max(float(account_equity), 0.0)
        projected_maintenance_margin = _compute_futures_maintenance_margin(
            projected_notional,
            futures_account,
            bracket=bracket,
        )
        projected_margin_ratio = _safe_ratio(
            projected_maintenance_margin,
            max(projected_margin_balance, 1e-12),
            default=float("inf"),
        ) if projected_notional > 0.0 else 0.0
        if projected_margin_ratio < warning_ratio - 1e-12:
            break
        safe_abs *= 0.85

    safe_position = float(np.sign(requested) * safe_abs)
    adjusted = safe_abs < requested_abs - 1e-12
    reason = None
    if adjusted and projected_margin_ratio >= warning_ratio - 1e-12:
        reason = "margin_safety_blocked"
    elif adjusted:
        reason = "margin_safety_clipped"

    if mode != "blocking":
        return requested, {
            "mode": mode,
            "requested_position": requested,
            "safe_position": safe_position,
            "adjusted": adjusted,
            "projected_margin_ratio": float(projected_margin_ratio),
            "warning_margin_ratio": warning_ratio,
            "reason": reason,
            "bracket": bracket,
        }

    return safe_position, {
        "mode": mode,
        "requested_position": requested,
        "safe_position": safe_position,
        "adjusted": adjusted,
        "projected_margin_ratio": float(projected_margin_ratio),
        "warning_margin_ratio": warning_ratio,
        "reason": reason,
        "bracket": bracket,
    }


def _build_futures_trade_notional(position, equity_curve):
    position = pd.Series(position, copy=False).astype(float)
    prev_equity = pd.Series(equity_curve, index=position.index).shift(1).fillna(float(equity_curve.iloc[0]))
    turnover = position.diff().abs().fillna(position.abs())
    return prev_equity * turnover


def _run_futures_account_backtest(close, position, equity, fee_rate, slippage_rate,
                                  execution_prices, signal_delay_bars, market,
                                  futures_account, funding_rates=None,
                                  significance=None, benchmark_returns=None,
                                  benchmark_sharpe=None, volume=None,
                                  slippage_model=None, orderbook_depth=None,
                                  execution_report=None):
    valuation_series = pd.Series(close, copy=False).astype(float)
    execution_series = valuation_series if execution_prices is None else pd.Series(
        execution_prices, index=valuation_series.index
    ).reindex(valuation_series.index).astype(float)
    requested_position = pd.Series(position, index=valuation_series.index, copy=False).reindex(valuation_series.index).fillna(0.0).astype(float)
    aligned_funding = pd.Series(0.0, index=valuation_series.index, dtype=float)
    if funding_rates is not None:
        aligned_funding = pd.Series(funding_rates, index=valuation_series.index).reindex(valuation_series.index).fillna(0.0).astype(float)

    requested_trade_notional = _build_futures_trade_notional(
        requested_position,
        pd.Series(float(equity), index=valuation_series.index, dtype=float),
    )
    slippage_rates = _estimate_trade_notional_slippage_rates(
        trade_notional=requested_trade_notional,
        execution_series=execution_series,
        slippage_rate=slippage_rate,
        slippage_model=slippage_model,
        volume=volume,
        orderbook_depth=orderbook_depth,
    )

    actual_position = pd.Series(0.0, index=valuation_series.index, dtype=float)
    equity_curve = pd.Series(0.0, index=valuation_series.index, dtype=float)
    strat_ret = pd.Series(0.0, index=valuation_series.index, dtype=float)
    unrealized_pnl_series = pd.Series(0.0, index=valuation_series.index, dtype=float)
    position_notional_series = pd.Series(0.0, index=valuation_series.index, dtype=float)
    initial_margin_series = pd.Series(0.0, index=valuation_series.index, dtype=float)
    maintenance_margin_series = pd.Series(0.0, index=valuation_series.index, dtype=float)
    margin_balance_series = pd.Series(0.0, index=valuation_series.index, dtype=float)
    margin_ratio_series = pd.Series(0.0, index=valuation_series.index, dtype=float)
    leverage_cap_series = pd.Series(0.0, index=valuation_series.index, dtype=float)
    realized_leverage_series = pd.Series(0.0, index=valuation_series.index, dtype=float)

    liquidation_rows = []
    leverage_adjustment_rows = []
    margin_safety_rows = []
    warning_ratio = float(futures_account.get("warning_margin_ratio", 0.8))
    margin_mode = futures_account.get("margin_mode", "isolated")

    prev_equity = float(equity)
    previous_position = 0.0
    isolated_margin_posted = 0.0
    total_fees_paid = 0.0
    total_slippage_paid = 0.0
    total_funding_pnl = 0.0
    total_liquidation_fees = 0.0
    bars_above_warning = 0

    for loc, timestamp in enumerate(valuation_series.index):
        price_now = float(valuation_series.iloc[loc])
        total_equity_before_trade = prev_equity
        funding_cash = 0.0
        unrealized_pnl = 0.0
        observed_notional = 0.0
        observed_initial_margin = 0.0
        observed_maintenance_margin = 0.0
        observed_margin_balance = 0.0
        observed_margin_ratio = 0.0
        liquidation_triggered = False

        if loc > 0 and abs(previous_position) > 1e-12 and prev_equity > 0.0:
            price_prev = float(valuation_series.iloc[loc - 1])
            price_return = 0.0 if price_prev <= 0.0 else (price_now / price_prev) - 1.0
            observed_notional = abs(previous_position) * prev_equity * max(1.0 + price_return, 0.0)
            leverage_cap, active_bracket = _resolve_futures_leverage_cap(observed_notional, futures_account)
            leverage_cap = max(1.0, float(leverage_cap))
            observed_initial_margin = observed_notional / leverage_cap
            unrealized_pnl = prev_equity * previous_position * price_return
            funding_cash = prev_equity * (-previous_position * float(aligned_funding.iloc[loc]))
            total_funding_pnl += funding_cash

            if margin_mode == "isolated":
                free_collateral = max(prev_equity - isolated_margin_posted, 0.0)
                observed_margin_balance = isolated_margin_posted + unrealized_pnl + funding_cash
                total_equity_before_trade = free_collateral + observed_margin_balance
            else:
                total_equity_before_trade = prev_equity + unrealized_pnl + funding_cash
                observed_margin_balance = total_equity_before_trade

            observed_maintenance_margin = _compute_futures_maintenance_margin(
                observed_notional,
                futures_account,
                bracket=active_bracket,
            )
            observed_margin_ratio = _safe_ratio(
                observed_maintenance_margin,
                max(observed_margin_balance, 1e-12),
                default=float("inf"),
            ) if observed_notional > 0.0 else 0.0

            if observed_margin_ratio >= warning_ratio:
                bars_above_warning += 1

            if observed_notional > 0.0 and observed_margin_balance <= observed_maintenance_margin + 1e-12:
                liquidation_triggered = True
                liquidation_fee = observed_notional * float(futures_account.get("liquidation_fee_rate", 0.0))
                total_liquidation_fees += liquidation_fee
                if margin_mode == "isolated":
                    total_equity_before_trade = free_collateral + max(observed_margin_balance - liquidation_fee, 0.0)
                else:
                    total_equity_before_trade = max(total_equity_before_trade - liquidation_fee, 0.0)
                liquidation_rows.append(
                    {
                        "timestamp": timestamp,
                        "margin_mode": margin_mode,
                        "position_before": float(previous_position),
                        "mark_price": price_now,
                        "position_notional": observed_notional,
                        "margin_balance": observed_margin_balance,
                        "maintenance_margin": observed_maintenance_margin,
                        "margin_ratio": observed_margin_ratio,
                        "liquidation_fee": liquidation_fee,
                        "equity_after_liquidation": total_equity_before_trade,
                    }
                )
                previous_position = 0.0
                isolated_margin_posted = 0.0

        requested_target = float(requested_position.iloc[loc])
        if liquidation_triggered and not futures_account.get("allow_reentry_after_liquidation", False):
            requested_target = 0.0

        capped_target, cap_meta = _cap_futures_target_position(
            requested_target,
            max(total_equity_before_trade, 0.0),
            futures_account,
        )
        if cap_meta.get("adjusted"):
            leverage_adjustment_rows.append(
                {
                    "timestamp": timestamp,
                    "requested_position": cap_meta.get("requested_position"),
                    "capped_position": cap_meta.get("capped_position"),
                    "leverage_cap": cap_meta.get("leverage_cap"),
                }
            )

        safe_target, margin_safety_meta = _enforce_futures_margin_safety(
            capped_target,
            max(total_equity_before_trade, 0.0),
            futures_account,
        )
        if margin_safety_meta.get("adjusted"):
            margin_safety_rows.append(
                {
                    "timestamp": timestamp,
                    "requested_position": margin_safety_meta.get("requested_position"),
                    "safe_position": margin_safety_meta.get("safe_position"),
                    "projected_margin_ratio": margin_safety_meta.get("projected_margin_ratio"),
                    "warning_margin_ratio": margin_safety_meta.get("warning_margin_ratio"),
                    "reason": margin_safety_meta.get("reason"),
                }
            )
        capped_target = float(safe_target)

        turnover = abs(capped_target - previous_position)
        fee_cash = max(prev_equity, 0.0) * turnover * float(fee_rate)
        slippage_cash = max(prev_equity, 0.0) * turnover * float(slippage_rates.iloc[loc])
        total_fees_paid += fee_cash
        total_slippage_paid += slippage_cash
        ending_equity = max(total_equity_before_trade - fee_cash - slippage_cash, 0.0)

        post_notional = abs(capped_target) * ending_equity
        post_leverage_cap, _ = _resolve_futures_leverage_cap(post_notional, futures_account)
        post_leverage_cap = max(1.0, float(post_leverage_cap))
        if margin_mode == "isolated" and post_notional > 0.0 and ending_equity > 0.0:
            isolated_margin_posted = min(post_notional / post_leverage_cap, ending_equity)
        else:
            isolated_margin_posted = 0.0

        actual_position.iloc[loc] = float(capped_target)
        equity_curve.iloc[loc] = float(ending_equity)
        strat_ret.iloc[loc] = float((ending_equity / prev_equity) - 1.0) if prev_equity > 0.0 else 0.0
        unrealized_pnl_series.iloc[loc] = float(unrealized_pnl)
        position_notional_series.iloc[loc] = float(observed_notional if observed_notional > 0.0 else post_notional)
        initial_margin_series.iloc[loc] = float(observed_initial_margin if observed_initial_margin > 0.0 else (post_notional / post_leverage_cap if post_notional > 0.0 else 0.0))
        maintenance_margin_series.iloc[loc] = float(observed_maintenance_margin)
        margin_balance_series.iloc[loc] = float(observed_margin_balance if observed_margin_balance > 0.0 else (ending_equity if margin_mode == "cross" and post_notional > 0.0 else 0.0))
        margin_ratio_series.iloc[loc] = float(observed_margin_ratio)
        leverage_cap_series.iloc[loc] = float(post_leverage_cap if post_notional > 0.0 else futures_account.get("configured_leverage", 1.0))
        realized_leverage_series.iloc[loc] = float(abs(capped_target))

        prev_equity = float(ending_equity)
        previous_position = float(capped_target)

    trade_ledger = _build_trade_ledger(strat_ret, actual_position, execution_series)
    execution_summary = dict(execution_report or {})
    if leverage_adjustment_rows:
        execution_summary["adjusted_orders"] = int(execution_summary.get("adjusted_orders", 0)) + len(leverage_adjustment_rows)
    futures_account_report = {
        "account_model": "futures_margin",
        "futures_margin_mode": margin_mode,
        "futures_contract": futures_account.get("contract_spec") or {},
        "futures_bracket_count": int(len(futures_account.get("leverage_brackets") or [])),
        "liquidation_event_count": int(len(liquidation_rows)),
        "liquidation_fee_paid": _round_metric(total_liquidation_fees, 2),
        "liquidation_events": pd.DataFrame(liquidation_rows),
        "margin_ratio_series": margin_ratio_series,
        "margin_balance_series": margin_balance_series,
        "maintenance_margin_series": maintenance_margin_series,
        "initial_margin_series": initial_margin_series,
        "position_notional_series": position_notional_series,
        "unrealized_pnl_series": unrealized_pnl_series,
        "realized_leverage_series": realized_leverage_series,
        "max_margin_ratio": _round_metric(float(margin_ratio_series.max()) if len(margin_ratio_series) > 0 else 0.0, 6),
        "warning_margin_ratio": float(warning_ratio),
        "bars_above_margin_warning": int(bars_above_warning),
        "bars_above_margin_warning_rate": _round_metric(float((margin_ratio_series >= warning_ratio).mean()) if len(margin_ratio_series) > 0 else 0.0, 6),
        "max_realized_leverage": _round_metric(float(realized_leverage_series.max()) if len(realized_leverage_series) > 0 else 0.0, 6),
        "avg_realized_leverage": _round_metric(float(realized_leverage_series.mean()) if len(realized_leverage_series) > 0 else 0.0, 6),
        "leverage_cap_adjustments": int(len(leverage_adjustment_rows)),
        "leverage_cap_adjustment_log": pd.DataFrame(leverage_adjustment_rows),
        "margin_safety_mode": futures_account.get("margin_safety_mode"),
        "margin_safety_adjustments": int(len(margin_safety_rows)),
        "margin_safety_adjustment_log": pd.DataFrame(margin_safety_rows),
    }

    return _summarize_backtest(
        equity_curve=equity_curve,
        strat_ret=strat_ret,
        position=actual_position,
        execution_series=execution_series,
        equity=equity,
        fees_paid=total_fees_paid,
        slippage_paid=total_slippage_paid,
        signal_delay_bars=signal_delay_bars,
        trade_ledger=trade_ledger,
        funding_pnl=total_funding_pnl,
        engine="pandas",
        significance=significance,
        benchmark_returns=benchmark_returns,
        benchmark_sharpe=benchmark_sharpe,
        execution_report=execution_summary,
        futures_account_report=futures_account_report,
    )


def _max_drawdown_duration(equity_curve, peak):
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        return 0, pd.Timedelta(0)

    underwater = equity_curve < peak
    max_bars = 0
    max_duration = pd.Timedelta(0)
    current_start = None
    current_bars = 0

    for timestamp, is_underwater in underwater.items():
        if is_underwater:
            if current_start is None:
                current_start = timestamp
                current_bars = 1
            else:
                current_bars += 1
            continue

        if current_start is not None:
            current_duration = timestamp - current_start
            if current_bars > max_bars:
                max_bars = current_bars
                max_duration = current_duration
            current_start = None
            current_bars = 0

    if current_start is not None:
        current_duration = equity_curve.index[-1] - current_start
        if current_bars > max_bars:
            max_bars = current_bars
            max_duration = current_duration

    return max_bars, max_duration


def _build_trade_ledger(strat_ret, position, execution_series):
    trades = []
    current_sign = 0.0
    entry_time = None
    entry_price = None
    segment_returns = []

    for timestamp in strat_ret.index:
        sign_now = float(np.sign(position.loc[timestamp]))
        if current_sign == 0.0:
            if sign_now != 0.0:
                current_sign = sign_now
                entry_time = timestamp
                entry_price = float(execution_series.loc[timestamp])
                segment_returns = []
        elif sign_now == current_sign:
            segment_returns.append(float(strat_ret.loc[timestamp]))
        else:
            segment_returns.append(float(strat_ret.loc[timestamp]))
            if segment_returns:
                trade_return = float(np.prod(1.0 + np.asarray(segment_returns, dtype=float)) - 1.0)
                exit_time = timestamp
                exit_price = float(execution_series.loc[exit_time])
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": exit_time,
                        "direction": int(np.sign(current_sign)),
                        "bars": int(len(segment_returns)),
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "return_pct": trade_return,
                    }
                )

            if sign_now != 0.0:
                current_sign = sign_now
                entry_time = timestamp
                entry_price = float(execution_series.loc[timestamp])
                segment_returns = []
            else:
                current_sign = 0.0
                entry_time = None
                entry_price = None
                segment_returns = []

    if current_sign != 0.0 and segment_returns:
        trade_return = float(np.prod(1.0 + np.asarray(segment_returns, dtype=float)) - 1.0)
        exit_time = strat_ret.index[-1]
        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": exit_time,
                "direction": int(np.sign(current_sign)),
                "bars": int(len(segment_returns)),
                "entry_price": entry_price,
                "exit_price": float(execution_series.loc[exit_time]),
                "return_pct": trade_return,
            }
        )

    if not trades:
        return pd.DataFrame(columns=["entry_time", "exit_time", "direction", "bars", "entry_price", "exit_price", "return_pct"])
    return pd.DataFrame(trades)


def _summarize_backtest(equity_curve, strat_ret, position, execution_series, equity,
                        fees_paid, slippage_paid, signal_delay_bars, trade_ledger,
                        funding_pnl=0.0, engine="pandas", significance=None,
                        benchmark_returns=None, benchmark_sharpe=None,
                        execution_report=None, futures_account_report=None):
    prev_equity = equity_curve.shift(1).fillna(equity)
    pnl = equity_curve - prev_equity

    total_ret = equity_curve.iloc[-1] / equity - 1
    periods_per_year = _infer_periods_per_year(equity_curve.index)
    annualization = np.sqrt(periods_per_year) if periods_per_year > 0 else 0.0
    volatility = strat_ret.std()
    sharpe = _annualized_sharpe(strat_ret.to_numpy(dtype=float), annualization)
    sortino = _annualized_sortino(strat_ret.to_numpy(dtype=float), annualization)
    peak = equity_curve.cummax()
    max_dd = ((equity_curve - peak) / peak).min()
    max_dd_amount = abs((equity_curve - peak).min())
    max_dd_bars, max_dd_duration = _max_drawdown_duration(equity_curve, peak)

    sign_changed = np.sign(position) != np.sign(position.shift(1).fillna(0.0))
    opened_trade = position.ne(0.0) & (position.shift(1).fillna(0.0).eq(0.0) | sign_changed)
    n_trades = int(opened_trade.sum())
    active_mask = position.abs() > 1e-12
    active = strat_ret[active_mask]
    active_pnl = pnl[active_mask]
    winners = active_pnl[active_pnl > 0]
    losers = active_pnl[active_pnl < 0]
    gross_profit = winners.sum()
    gross_loss = abs(losers.sum())
    win_rate = float(active_pnl.gt(0).mean()) if len(active_pnl) > 0 else 0.0
    avg_win = winners.mean() if len(winners) > 0 else 0.0
    avg_loss = abs(losers.mean()) if len(losers) > 0 else 0.0
    expectancy = active_pnl.mean() if len(active_pnl) > 0 else 0.0
    expectancy_pct = active.mean() if len(active) > 0 else 0.0
    profit_factor = _safe_ratio(gross_profit, gross_loss)
    exposure_rate = float(position.ne(0).mean()) if len(position) > 0 else 0.0
    avg_position_size = float(position.abs().mean()) if len(position) > 0 else 0.0
    total_turnover = float(position.diff().abs().fillna(position.abs()).sum()) if len(position) > 0 else 0.0

    trade_returns = trade_ledger["return_pct"] if not trade_ledger.empty else pd.Series(dtype=float)
    trade_winners = trade_returns[trade_returns > 0]
    trade_losers = trade_returns[trade_returns < 0]
    trade_profit_factor = _safe_ratio(float(trade_winners.sum()), abs(float(trade_losers.sum())))
    trade_win_rate = float(trade_returns.gt(0).mean()) if len(trade_returns) > 0 else 0.0
    avg_trade_return_pct = float(trade_returns.mean()) if len(trade_returns) > 0 else 0.0
    avg_trade_bars = float(trade_ledger["bars"].mean()) if not trade_ledger.empty else 0.0

    elapsed_years = 0.0
    if len(equity_curve.index) > 1:
        elapsed_years = (equity_curve.index[-1] - equity_curve.index[0]).total_seconds() / _SECONDS_PER_YEAR
    cagr = _compute_cagr_from_equity(equity_curve.to_numpy(dtype=float), equity, elapsed_years)
    calmar = _compute_calmar(cagr, max_dd)

    significance_metrics = _compute_significance_metrics(
        strat_ret=strat_ret,
        equity=equity,
        periods_per_year=periods_per_year,
        elapsed_years=elapsed_years,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        total_ret=total_ret,
        max_dd=max_dd,
        significance=significance,
        benchmark_returns=benchmark_returns,
        benchmark_sharpe=benchmark_sharpe,
    )

    funding_paid = max(-float(funding_pnl), 0.0)
    funding_received = max(float(funding_pnl), 0.0)

    summary = {
        "engine": engine,
        "starting_equity": _round_metric(equity, 2),
        "ending_equity": _round_metric(equity_curve.iloc[-1], 2),
        "net_profit": _round_metric(equity_curve.iloc[-1] - equity, 2),
        "net_profit_pct": _round_metric(total_ret, 4),
        "gross_profit": _round_metric(gross_profit, 2),
        "gross_loss": _round_metric(gross_loss, 2),
        "fees_paid": _round_metric(fees_paid, 2),
        "slippage_paid": _round_metric(slippage_paid, 2),
        "funding_pnl": _round_metric(funding_pnl, 2),
        "funding_paid": _round_metric(funding_paid, 2),
        "funding_received": _round_metric(funding_received, 2),
        "total_return": _round_metric(total_ret, 4),
        "cagr": _round_metric(cagr, 4),
        "sharpe_ratio": _round_metric(sharpe, 2),
        "sortino_ratio": _round_metric(sortino, 2),
        "calmar_ratio": _round_metric(calmar, 2),
        "annualized_volatility": _round_metric(volatility * annualization, 4),
        "max_drawdown": _round_metric(max_dd, 4),
        "max_drawdown_amount": _round_metric(max_dd_amount, 2),
        "max_drawdown_duration": max_dd_duration,
        "max_drawdown_duration_bars": max_dd_bars,
        "exposure_rate": _round_metric(exposure_rate, 4),
        "average_position_size": _round_metric(avg_position_size, 4),
        "signal_delay_bars": signal_delay_bars,
        "total_turnover": _round_metric(total_turnover, 4),
        "profit_factor": _round_metric(profit_factor, 2),
        "avg_win": _round_metric(avg_win, 2),
        "avg_loss": _round_metric(avg_loss, 2),
        "expectancy": _round_metric(expectancy, 2),
        "expectancy_pct": _round_metric(expectancy_pct, 6),
        "total_trades": n_trades,
        "win_rate": _round_metric(win_rate, 4),
        "active_bar_win_rate": _round_metric(win_rate, 4),
        "closed_trades": int(len(trade_ledger)),
        "trade_win_rate": _round_metric(trade_win_rate, 4),
        "avg_trade_return_pct": _round_metric(avg_trade_return_pct, 6),
        "avg_trade_bars": _round_metric(avg_trade_bars, 2),
        "trade_profit_factor": _round_metric(trade_profit_factor, 2),
        "statistical_significance": significance_metrics,
        "trade_ledger": trade_ledger,
        "equity_curve": equity_curve,
    }
    if execution_report is not None:
        summary.update(
            {
                "blocked_orders": int(execution_report.get("blocked_orders", 0)),
                "adjusted_orders": int(execution_report.get("adjusted_orders", 0)),
                "accepted_orders": int(execution_report.get("accepted_orders", 0)),
                "partial_fill_orders": int(execution_report.get("partial_fill_orders", 0)),
                "cancelled_orders": int(execution_report.get("cancelled_orders", 0)),
                "blocked_notional_share": execution_report.get("blocked_notional_share", 0.0),
                "fill_ratio": execution_report.get("fill_ratio", 0.0),
                "unfilled_notional": execution_report.get("unfilled_notional", 0.0),
                "average_action_delay_bars": execution_report.get("average_action_delay_bars", 0.0),
                "average_fill_delay_bars": execution_report.get("average_fill_delay_bars", 0.0),
                "max_fill_delay_bars": execution_report.get("max_fill_delay_bars", 0),
                "order_rejection_reasons": execution_report.get("order_rejection_reasons", {}),
                "execution_adapter": execution_report.get("execution_adapter"),
                "execution_backend": execution_report.get("execution_backend"),
                "execution_policy": execution_report.get("execution_policy", {}),
                "execution_cost_report": execution_report.get("execution_cost_report", {}),
                "order_intents": execution_report.get("order_intents", pd.DataFrame()),
                "order_ledger": execution_report.get("order_ledger", pd.DataFrame()),
                "price_fill_actions": execution_report.get("price_fill_actions", {}),
                "liquidity_report": execution_report.get("liquidity_report", {}),
                "scenario_report": execution_report.get("scenario_report", {}),
                "symbol_lifecycle_report": execution_report.get("symbol_lifecycle_report", {}),
            }
        )
    if futures_account_report is not None:
        summary.update(dict(futures_account_report))
    return summary


def _resolve_realized_slippage_paid(cost_report, fallback_total, execution_report=None):
    lifecycle_report = execution_report.get("symbol_lifecycle_report", {}) if execution_report is not None else {}
    if lifecycle_report.get("forced_liquidations", 0) > 0 or lifecycle_report.get("dropped_rows", 0) > 0:
        return float(fallback_total)
    if isinstance(cost_report, dict) and cost_report.get("total_cost") is not None:
        return float(cost_report.get("total_cost", fallback_total))
    return float(fallback_total)


def _run_vectorbt_backtest(close, position, equity, fee_rate, slippage_rate,
                           execution_prices, signal_delay_bars, allow_short,
                           symbol_filters=None, funding_rates=None,
                           significance=None, benchmark_returns=None,
                           benchmark_sharpe=None, volume=None,
                           slippage_model=None, orderbook_depth=None,
                           execution_report=None):
    if vbt is None or Direction is None or SizeType is None:
        raise ImportError("vectorbt is not installed")

    valuation_series = pd.Series(close, copy=False).astype(float)
    execution_series = pd.Series(execution_prices if execution_prices is not None else close, index=valuation_series.index).reindex(valuation_series.index).astype(float)
    slippage_rates, turnover = _estimate_slippage_rates(
        position=position,
        equity=equity,
        valuation_series=valuation_series,
        execution_series=execution_series,
        slippage_rate=slippage_rate,
        slippage_model=slippage_model,
        volume=volume,
        funding_rates=funding_rates,
        orderbook_depth=orderbook_depth,
    )
    cost_report = execution_report.get("execution_cost_report", {}) if execution_report is not None else {}

    portfolio = vbt.Portfolio.from_orders(
        close=valuation_series,
        size=position,
        size_type=SizeType.TargetPercent,
        direction=Direction.Both if allow_short else Direction.LongOnly,
        price=execution_series,
        fees=fee_rate,
        slippage=slippage_rates,
        init_cash=equity,
        freq=valuation_series.index.to_series().diff().median(),
    )

    base_equity = pd.Series(portfolio.value(), index=valuation_series.index, dtype=float)
    funding_cash = _compute_funding_cash(position, funding_rates, base_equity)
    adjusted_equity = base_equity + funding_cash.cumsum()
    adjusted_returns = adjusted_equity.pct_change().fillna(0.0)
    trade_ledger = _vectorbt_trade_ledger(portfolio, valuation_series.index)
    fees_paid = float(portfolio.orders.records_readable["Fees"].sum()) if not portfolio.orders.records_readable.empty else 0.0
    slippage_paid = _resolve_realized_slippage_paid(
        cost_report,
        fallback_total=(adjusted_equity.shift(1).fillna(equity) * turnover * slippage_rates).sum(),
        execution_report=execution_report,
    )

    return _summarize_backtest(
        equity_curve=adjusted_equity,
        strat_ret=adjusted_returns,
        position=position,
        execution_series=execution_series,
        equity=equity,
        fees_paid=fees_paid,
        slippage_paid=slippage_paid,
        signal_delay_bars=signal_delay_bars,
        trade_ledger=trade_ledger,
        funding_pnl=float(funding_cash.sum()),
        engine="vectorbt",
        significance=significance,
        benchmark_returns=benchmark_returns,
        benchmark_sharpe=benchmark_sharpe,
        execution_report=execution_report,
    )


def _run_pandas_backtest(close, position, equity, fee_rate, slippage_rate,
                         execution_prices, signal_delay_bars, funding_rates=None,
                         significance=None, benchmark_returns=None,
                         benchmark_sharpe=None, volume=None,
                         slippage_model=None, orderbook_depth=None,
                         execution_report=None):
    valuation_series = pd.Series(close, copy=False).astype(float)
    execution_series = valuation_series if execution_prices is None else pd.Series(execution_prices, index=valuation_series.index).reindex(valuation_series.index).astype(float)
    returns = valuation_series.pct_change().fillna(0.0)
    held_position = position.shift(1).fillna(0.0)
    slippage_rates, turnover = _estimate_slippage_rates(
        position=position,
        equity=equity,
        valuation_series=valuation_series,
        execution_series=execution_series,
        slippage_rate=slippage_rate,
        slippage_model=slippage_model,
        volume=volume,
        funding_rates=funding_rates,
        orderbook_depth=orderbook_depth,
    )
    cost_report = execution_report.get("execution_cost_report", {}) if execution_report is not None else {}
    fees = turnover * fee_rate
    slippage = turnover * slippage_rates
    funding_returns = pd.Series(0.0, index=position.index, dtype=float)
    if funding_rates is not None:
        funding_returns = -held_position * pd.Series(funding_rates, index=position.index).reindex(position.index).fillna(0.0)

    strat_ret = held_position * returns + funding_returns - fees - slippage
    equity_curve = equity * (1.0 + strat_ret).cumprod()
    trade_ledger = _build_trade_ledger(strat_ret, position, execution_series)
    prev_equity = equity_curve.shift(1).fillna(equity)
    return _summarize_backtest(
        equity_curve=equity_curve,
        strat_ret=strat_ret,
        position=position,
        execution_series=execution_series,
        equity=equity,
        fees_paid=float((prev_equity * fees).sum()),
        slippage_paid=_resolve_realized_slippage_paid(
            cost_report,
            fallback_total=(prev_equity * slippage).sum(),
            execution_report=execution_report,
        ),
        signal_delay_bars=signal_delay_bars,
        trade_ledger=trade_ledger,
        funding_pnl=float((prev_equity * funding_returns).sum()),
        engine="pandas",
        significance=significance,
        benchmark_returns=benchmark_returns,
        benchmark_sharpe=benchmark_sharpe,
        execution_report=execution_report,
    )


# ───────────────────────────────────────────────────────────────────────────
# Backtest engine adapter  (VectorBT first, pandas fallback)
# ───────────────────────────────────────────────────────────────────────────

def run_backtest(close, signals, equity=10_000.0, fee_rate=0.001, slippage_rate=0.0,
                 execution_prices=None, signal_delay_bars=1, engine="vectorbt",
                 market="spot", leverage=1.0, allow_short=None, symbol_filters=None,
                 funding_rates=None, significance=None, benchmark_returns=None,
                 benchmark_sharpe=None, volume=None, slippage_model=None,
                 orderbook_depth=None, execution_price_policy="strict",
                 execution_price_fill_limit=None, valuation_price_policy="drop_rows",
                 valuation_price_fill_limit=None, futures_account=None,
                 liquidity_lag_bars=1, execution_policy=None,
                 futures_contract=None, futures_leverage_brackets=None,
                 symbol_lifecycle=None, symbol_lifecycle_policy=None,
                 scenario_schedule=None, scenario_policy=None):
    """Run a backtest through the configured execution adapter.

    Parameters
    ----------
    close            : pd.Series  – mark-to-market or valuation price series
    signals          : pd.Series  – target portfolio weights before execution delay
    equity           : float      – starting capital
    fee_rate         : float      – one-way fee
    slippage_rate    : float      – one-way flat slippage rate used by the legacy/default model
    execution_prices : pd.Series or None – optional execution price series, e.g. next-bar open
    signal_delay_bars: int        – bars to delay signal application before execution
    engine           : str        – "vectorbt" (default) or "pandas"
    market           : str        – "spot", "um_futures", or "cm_futures"
    leverage         : float      – exposure multiplier applied to target weights
    allow_short      : bool|None  – defaults to False for spot, True for futures
    symbol_filters   : dict|None  – Binance execution filters (tick size, lot size, min notional)
    funding_rates    : pd.Series|None – futures funding rates aligned to close index; applied on funding timestamps only
    significance     : bool|dict|None – stationary-bootstrap significance settings; enabled by default
    benchmark_returns: pd.Series|array|None – optional benchmark returns aligned to backtest index for Sharpe comparison
    benchmark_sharpe : float|None – optional benchmark Sharpe ratio threshold when no benchmark return series is supplied
    volume           : pd.Series|array|None – bar volume aligned to the backtest index; required for non-flat slippage models
    slippage_model   : str|object|None – one of {"flat", "sqrt_impact", "orderbook"} or a custom estimator implementing estimate(...)
    orderbook_depth  : pd.DataFrame|None – optional L2 depth frame for future order-book-aware slippage models
    liquidity_lag_bars : int – lag applied to bar-volume liquidity inputs before cost estimation
    execution_policy : dict|ExecutionPolicy|None – order submission and fill policy for the execution adapter
    symbol_lifecycle : pd.DataFrame|list[dict]|dict|None – optional symbol halt/delist lifecycle events aligned or alignable to the backtest index
    symbol_lifecycle_policy : dict|None – lifecycle actions such as {"halt_action": "freeze", "delist_action": "liquidate"}
    scenario_schedule : list[dict]|pd.DataFrame|None – venue-state events such as downtime, stale marks, halts, and deleveraging windows
    scenario_policy : dict|None – scenario responses such as {"downtime_action": "freeze", "stale_mark_action": "reject"}

    execution_price_policy : str – one of {"strict", "ffill", "ffill_with_limit", "drop_rows"}
    execution_price_fill_limit : int|None – max consecutive execution-price fills when using "ffill_with_limit"
    valuation_price_policy : str – one of {"strict", "ffill", "ffill_with_limit", "drop_rows"}
    valuation_price_fill_limit : int|None – max consecutive valuation-price fills when using "ffill_with_limit"

    Returns dict with metrics and equity curve.
    """
    close = pd.Series(close, copy=False).astype(float)
    signal_series = pd.Series(signals, index=close.index).reindex(close.index).fillna(0.0).astype(float)
    benchmark_returns = _align_benchmark_returns(benchmark_returns, close.index)
    signal_delay_bars = max(0, int(signal_delay_bars))
    allow_short = (market or "spot") != "spot" if allow_short is None else bool(allow_short)
    symbol_filters = dict(symbol_filters or {})
    tick_size = symbol_filters.get("tick_size")

    valuation_series, valuation_fill_actions = _normalize_price_series(
        close,
        tick_size=tick_size,
        fill_policy=valuation_price_policy,
        fill_limit=valuation_price_fill_limit,
        return_diagnostics=True,
        series_name="valuation",
    )
    execution_input = execution_prices if execution_prices is not None else close
    execution_series, execution_fill_actions = _normalize_price_series(
        execution_input,
        tick_size=tick_size,
        fill_policy=execution_price_policy,
        fill_limit=execution_price_fill_limit,
        return_diagnostics=True,
        series_name="execution",
    )
    scenario_frame = build_scenario_schedule(close.index, scenario_schedule)
    valuation_series, execution_series, scenario_report = apply_scenario_price_policy(
        valuation_series,
        execution_series,
        scenario_frame,
        policy=scenario_policy,
    )

    valid_mask = pd.Series(True, index=close.index, dtype=bool)
    if valuation_price_policy == "drop_rows":
        valid_mask &= valuation_series.notna()
    elif valuation_fill_actions["invalid_rows"] > 0:
        raise ValueError(
            f"valuation price policy {valuation_price_policy!r} left {valuation_fill_actions['invalid_rows']} invalid rows"
        )

    if execution_price_policy == "drop_rows":
        valid_mask &= execution_series.notna()
    elif execution_fill_actions["invalid_rows"] > 0:
        raise ValueError(
            f"execution price policy {execution_price_policy!r} left {execution_fill_actions['invalid_rows']} invalid rows"
        )

    dropped_rows = int((~valid_mask).sum())
    valuation_fill_actions["dropped_rows"] = dropped_rows
    execution_fill_actions["dropped_rows"] = dropped_rows

    if dropped_rows > 0:
        close = close.loc[valid_mask]
        signal_series = signal_series.loc[valid_mask]
        valuation_series = valuation_series.loc[valid_mask]
        execution_series = execution_series.loc[valid_mask]
        if benchmark_returns is not None:
            benchmark_returns = benchmark_returns.loc[valid_mask]
        if funding_rates is not None:
            funding_rates = pd.Series(funding_rates, copy=False).reindex(valid_mask.index).loc[valid_mask]
        if volume is not None:
            volume = pd.Series(volume, copy=False).reindex(valid_mask.index).loc[valid_mask]

    if valuation_series.empty or execution_series.empty:
        raise ValueError("price normalization removed all rows from the backtest")

    liquidity_inputs = resolve_liquidity_inputs(
        index=execution_series.index,
        volume=volume,
        orderbook_depth=orderbook_depth,
        slippage_model=slippage_model,
        liquidity_lag_bars=liquidity_lag_bars,
    )
    volume = liquidity_inputs["volume"]
    orderbook_depth = liquidity_inputs["orderbook_depth"]

    position = _normalize_position_targets(
        signal_series.shift(signal_delay_bars).fillna(0.0),
        leverage=leverage,
        allow_short=allow_short,
    )
    execution_report = _build_execution_contract(
        close=valuation_series,
        requested_position=position,
        equity=equity,
        execution_prices=execution_series,
        symbol_filters=symbol_filters,
        market=market,
        volume=volume,
        execution_policy=execution_policy,
        scenario_schedule=scenario_frame,
        scenario_policy=scenario_policy,
    )
    execution_report["price_fill_actions"] = {
        "execution": execution_fill_actions,
        "valuation": valuation_fill_actions,
    }
    execution_report["liquidity_report"] = liquidity_inputs["diagnostics"]
    execution_report = apply_execution_scenarios(
        execution_report,
        scenario_frame.reindex(execution_report["position"].index),
        policy=scenario_policy,
    )
    execution_report["scenario_report"].update(
        {
            "stale_mark_action": scenario_report.get("stale_mark_action"),
            "stale_mark_rejections": scenario_report.get("stale_mark_rejections", 0),
            "warnings": scenario_report.get("warnings", []),
        }
    )

    lifecycle_frame = merge_scenario_lifecycle(
        index=execution_report["position"].index,
        symbol_lifecycle=symbol_lifecycle,
        scenario_schedule=scenario_frame.reindex(execution_report["position"].index),
        symbol=symbol_filters.get("symbol"),
    )
    if lifecycle_frame is not None:
        executable_position, keep_mask, lifecycle_report = apply_symbol_lifecycle_policy(
            execution_report["position"],
            lifecycle_frame,
            policy=symbol_lifecycle_policy,
        )
        execution_report["position"] = executable_position
        execution_report["symbol_lifecycle_report"] = lifecycle_report
        execution_report["symbol_lifecycle"] = lifecycle_frame.loc[keep_mask]

        aligned_index = executable_position.index
        valuation_series = valuation_series.reindex(aligned_index)
        execution_series = execution_series.reindex(aligned_index)
        execution_report["valuation_series"] = valuation_series
        execution_report["execution_series"] = execution_series
        if benchmark_returns is not None:
            benchmark_returns = benchmark_returns.reindex(aligned_index)
        if funding_rates is not None:
            funding_rates = pd.Series(funding_rates, copy=False).reindex(aligned_index)
        if volume is not None:
            volume = pd.Series(volume, copy=False).reindex(aligned_index)
        if orderbook_depth is not None:
            orderbook_depth = pd.DataFrame(orderbook_depth).reindex(aligned_index)

        if isinstance(execution_report.get("requested_position"), pd.Series):
            execution_report["requested_position"] = execution_report["requested_position"].reindex(aligned_index)
        for key in ["order_intents", "order_ledger"]:
            frame = execution_report.get(key)
            if isinstance(frame, pd.DataFrame) and not frame.empty and "timestamp" in frame.columns:
                execution_report[key] = frame.loc[frame["timestamp"].isin(aligned_index)].copy()

    execution_report["execution_cost_report"] = _estimate_fill_event_costs(
        order_ledger=execution_report.get("order_ledger", pd.DataFrame()),
        execution_series=execution_series,
        slippage_rate=slippage_rate,
        slippage_model=slippage_model,
        volume=volume,
        orderbook_depth=orderbook_depth,
    )
    lifecycle_report = execution_report.get("symbol_lifecycle_report", {})
    if lifecycle_report.get("forced_liquidations", 0) > 0 or lifecycle_report.get("dropped_rows", 0) > 0:
        execution_report["execution_cost_report"]["coverage_warning"] = (
            "forced lifecycle actions are summarized via turnover-based slippage rather than fill-event attribution"
        )
    valuation_series = execution_report["valuation_series"]
    execution_series = execution_report["execution_series"]
    executable_position = execution_report["position"]

    resolved_futures_account = _normalize_futures_account_config(
        futures_account=futures_account,
        market=market,
        leverage=leverage,
        contract_spec=futures_contract,
        leverage_brackets=futures_leverage_brackets,
    )
    if resolved_futures_account is not None:
        return _run_futures_account_backtest(
            close=valuation_series,
            position=executable_position,
            equity=equity,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            execution_prices=execution_series,
            signal_delay_bars=signal_delay_bars,
            market=market,
            futures_account=resolved_futures_account,
            funding_rates=funding_rates,
            significance=significance,
            benchmark_returns=benchmark_returns,
            benchmark_sharpe=benchmark_sharpe,
            volume=volume,
            slippage_model=slippage_model,
            orderbook_depth=orderbook_depth,
            execution_report=execution_report,
        )

    selected_engine = (engine or "vectorbt").lower()
    if selected_engine == "vectorbt":
        try:
            return _run_vectorbt_backtest(
                close=valuation_series,
                position=executable_position,
                equity=equity,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                execution_prices=execution_series,
                signal_delay_bars=signal_delay_bars,
                allow_short=allow_short,
                symbol_filters=None,
                funding_rates=funding_rates,
                significance=significance,
                benchmark_returns=benchmark_returns,
                benchmark_sharpe=benchmark_sharpe,
                volume=volume,
                slippage_model=slippage_model,
                orderbook_depth=orderbook_depth,
                execution_report=execution_report,
            )
        except ImportError:
            selected_engine = "pandas"

    return _run_pandas_backtest(
        close=valuation_series,
        position=executable_position,
        equity=equity,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        execution_prices=execution_series,
        signal_delay_bars=signal_delay_bars,
        funding_rates=funding_rates,
        significance=significance,
        benchmark_returns=benchmark_returns,
        benchmark_sharpe=benchmark_sharpe,
        volume=volume,
        slippage_model=slippage_model,
        orderbook_depth=orderbook_depth,
        execution_report=execution_report,
    )
