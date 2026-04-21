"""Empirical signal-decay analysis tied to execution assumptions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .execution import resolve_execution_policy
from .models import build_execution_outcome_frame


_DEFAULT_SIGNAL_DECAY_POLICY = {
    "enabled": True,
    "max_horizon_bars": None,
    "min_realized_trade_count": 20,
    "min_net_edge_at_effective_delay": 0.0,
    "min_half_life_holding_ratio": 1.0,
    "regime_min_samples": 20,
}


def resolve_effective_delay_bars(signal_delay_bars, execution_policy=None):
    configured_delay = max(0, int(signal_delay_bars or 0))
    policy = resolve_execution_policy(execution_policy)
    extra_delay = max(0, int(policy.max_order_age_bars) - 1, int(policy.cancel_replace_bars) - 1)
    return int(configured_delay + extra_delay)


def resolve_signal_decay_policy(config=None, *, holding_bars=1, effective_delay_bars=0):
    resolved = {**_DEFAULT_SIGNAL_DECAY_POLICY, **dict(config or {})}
    max_horizon = resolved.get("max_horizon_bars")
    if max_horizon is None:
        max_horizon = max(int(holding_bars), int(effective_delay_bars), int(holding_bars) * 3)

    return {
        "enabled": bool(resolved.get("enabled", True)),
        "max_horizon_bars": max(1, int(max_horizon)),
        "min_realized_trade_count": max(1, int(resolved.get("min_realized_trade_count", 20))),
        "min_net_edge_at_effective_delay": float(resolved.get("min_net_edge_at_effective_delay", 0.0)),
        "min_half_life_holding_ratio": max(0.0, float(resolved.get("min_half_life_holding_ratio", 1.0))),
        "regime_min_samples": max(1, int(resolved.get("regime_min_samples", 20))),
    }


def _as_series(value, index=None, dtype=float):
    if value is None:
        return None
    series = pd.Series(value, copy=False)
    if index is not None:
        series = series.reindex(index)
    if dtype is not None:
        series = series.astype(dtype)
    return series


def _as_frame(value, index=None):
    if value is None:
        return pd.DataFrame(index=index)
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
    elif isinstance(value, pd.Series):
        frame = value.to_frame(name=value.name or "regime")
    else:
        frame = pd.DataFrame(value)
    if index is not None:
        frame = frame.reindex(index)
    return frame


def _normalize_direction(series):
    aligned = pd.Series(series, copy=False).fillna(0.0).astype(float)
    return aligned.apply(lambda value: 1.0 if value > 1e-12 else (-1.0 if value < -1e-12 else 0.0))


def _normalize_segment(segment):
    predictions = _as_series(segment.get("predictions"))
    event_signals = _as_series(segment.get("event_signals"))
    base_index = None
    for candidate in [event_signals, predictions]:
        if candidate is not None and len(candidate) > 0:
            base_index = candidate.index
            break

    if base_index is None:
        base_index = pd.DatetimeIndex([])

    predictions = _as_series(segment.get("predictions"), index=base_index)
    direction_edge = _as_series(segment.get("direction_edge"), index=base_index)
    event_signals = _as_series(segment.get("event_signals"), index=base_index)
    valuation_prices = _as_series(segment.get("valuation_prices"), dtype=float)
    execution_prices = _as_series(segment.get("execution_prices"), index=valuation_prices.index, dtype=float)
    if execution_prices is None:
        execution_prices = valuation_prices.copy()

    runtime_kwargs = dict(segment.get("runtime_kwargs") or {})
    regimes = _as_frame(segment.get("regimes"), index=base_index)
    cutoff_timestamp = segment.get("cutoff_timestamp")
    if cutoff_timestamp is None and len(valuation_prices.index) > 0:
        cutoff_timestamp = valuation_prices.index[-1]

    return {
        "predictions": predictions if predictions is not None else pd.Series(index=base_index, dtype=float),
        "direction_edge": direction_edge,
        "event_signals": event_signals if event_signals is not None else pd.Series(index=base_index, dtype=float),
        "valuation_prices": valuation_prices,
        "execution_prices": execution_prices,
        "runtime_kwargs": runtime_kwargs,
        "fee_rate": float(segment.get("fee_rate", 0.0)),
        "slippage_rate": float(segment.get("slippage_rate", 0.0)),
        "equity": float(segment.get("equity", 10_000.0)),
        "liquidity_lag_bars": int(segment.get("liquidity_lag_bars", runtime_kwargs.get("liquidity_lag_bars", 1))),
        "regimes": regimes,
        "cutoff_timestamp": cutoff_timestamp,
    }


def _resolve_score_direction(segment):
    if segment.get("direction_edge") is not None and not segment["direction_edge"].empty:
        score = segment["direction_edge"].clip(-1.0, 1.0).fillna(0.0)
    else:
        score = _normalize_direction(segment["predictions"]).astype(float)
    return _normalize_direction(score), score.abs().clip(0.0, 1.0)


def _resolve_trade_direction(segment):
    event_signals = segment.get("event_signals")
    if event_signals is None or event_signals.empty:
        return _normalize_direction(segment["predictions"]).astype(float)
    return _normalize_direction(event_signals).astype(float)


def _build_section_outcomes(segment, section, *, holding_bars, signal_delay_bars):
    valuation_prices = segment["valuation_prices"]
    if valuation_prices is None or valuation_prices.empty:
        return pd.DataFrame(), None

    if section == "raw_scores":
        direction, weights = _resolve_score_direction(segment)
        outcomes = build_execution_outcome_frame(
            direction,
            valuation_prices=valuation_prices,
            execution_prices=valuation_prices,
            holding_bars=holding_bars,
            signal_delay_bars=signal_delay_bars,
            fee_rate=0.0,
            slippage_rate=0.0,
            cutoff_timestamp=segment.get("cutoff_timestamp"),
        )
        return outcomes, weights.reindex(outcomes.index).fillna(0.0)

    direction = _resolve_trade_direction(segment)
    if section == "thresholded_signals":
        outcomes = build_execution_outcome_frame(
            direction,
            valuation_prices=valuation_prices,
            execution_prices=valuation_prices,
            holding_bars=holding_bars,
            signal_delay_bars=signal_delay_bars,
            fee_rate=0.0,
            slippage_rate=0.0,
            cutoff_timestamp=segment.get("cutoff_timestamp"),
        )
        return outcomes, None

    runtime_kwargs = dict(segment.get("runtime_kwargs") or {})
    outcomes = build_execution_outcome_frame(
        direction,
        valuation_prices=valuation_prices,
        execution_prices=segment.get("execution_prices"),
        holding_bars=holding_bars,
        signal_delay_bars=signal_delay_bars,
        fee_rate=float(segment.get("fee_rate", 0.0)),
        slippage_rate=float(segment.get("slippage_rate", 0.0)),
        funding_rates=runtime_kwargs.get("funding_rates"),
        cutoff_timestamp=segment.get("cutoff_timestamp"),
        equity=float(segment.get("equity", 10_000.0)),
        volume=runtime_kwargs.get("volume"),
        slippage_model=runtime_kwargs.get("slippage_model"),
        orderbook_depth=runtime_kwargs.get("orderbook_depth"),
        liquidity_lag_bars=int(segment.get("liquidity_lag_bars", runtime_kwargs.get("liquidity_lag_bars", 1))),
    )
    return outcomes, None


def _summarize_outcome_batches(batches):
    trade_count = 0
    available_count = 0
    weight_total = 0.0
    gross_total = 0.0
    net_total = 0.0
    profitable_total = 0.0

    for outcomes, weights in batches:
        if outcomes is None or outcomes.empty:
            continue

        available = pd.to_numeric(outcomes.get("outcome_available"), errors="coerce").fillna(0).astype(int)
        available_count += int(available.sum())

        trade_mask = pd.to_numeric(outcomes.get("trade_taken"), errors="coerce").fillna(0).astype(bool)
        if not trade_mask.any():
            continue

        gross = pd.to_numeric(outcomes.loc[trade_mask, "gross_trade_return"], errors="coerce").to_numpy(dtype=float)
        net = pd.to_numeric(outcomes.loc[trade_mask, "net_trade_return"], errors="coerce").to_numpy(dtype=float)
        profitable = pd.to_numeric(outcomes.loc[trade_mask, "profitable"], errors="coerce").to_numpy(dtype=float)
        batch_weights = (
            pd.to_numeric(weights.loc[trade_mask], errors="coerce").to_numpy(dtype=float)
            if weights is not None
            else np.ones(len(gross), dtype=float)
        )
        valid_mask = (
            np.isfinite(gross)
            & np.isfinite(net)
            & np.isfinite(profitable)
            & np.isfinite(batch_weights)
            & (batch_weights > 0.0)
        )
        if not valid_mask.any():
            continue

        gross = gross[valid_mask]
        net = net[valid_mask]
        profitable = profitable[valid_mask]
        batch_weights = batch_weights[valid_mask]
        trade_count += int(len(gross))
        gross_total += float(np.sum(batch_weights * gross))
        net_total += float(np.sum(batch_weights * net))
        profitable_total += float(np.sum(batch_weights * profitable))
        weight_total += float(np.sum(batch_weights))

    if weight_total <= 0.0:
        return {
            "trade_count": int(trade_count),
            "available_count": int(available_count),
            "gross_edge": None,
            "net_edge": None,
            "profitable_rate": None,
        }

    return {
        "trade_count": int(trade_count),
        "available_count": int(available_count),
        "gross_edge": float(gross_total / weight_total),
        "net_edge": float(net_total / weight_total),
        "profitable_rate": float(profitable_total / weight_total),
    }


def _build_curve(section, segments, values, *, curve_kind, holding_bars=None, signal_delay_bars=None):
    rows = []
    for value in values:
        batches = []
        for segment in segments:
            outcomes, weights = _build_section_outcomes(
                segment,
                section,
                holding_bars=int(value if curve_kind == "horizon" else holding_bars),
                signal_delay_bars=int(value if curve_kind == "delay" else signal_delay_bars),
            )
            batches.append((outcomes, weights))

        summary = _summarize_outcome_batches(batches)
        row = {
            f"{curve_kind}_bars": int(value),
            **summary,
        }
        if section == "raw_scores":
            row["score_edge"] = row["gross_edge"]
        rows.append(row)
    return rows


def _first_curve_row(rows, key, value):
    for row in rows:
        if int(row.get(key, -1)) == int(value):
            return row
    return None


def _empirical_half_life(curve_rows):
    if not curve_rows:
        return None

    baseline = curve_rows[0].get("net_edge")
    if baseline is None or not np.isfinite(float(baseline)) or float(baseline) <= 0.0:
        return None

    threshold = 0.5 * float(baseline)
    for row in curve_rows[1:]:
        edge = row.get("net_edge")
        if edge is None or not np.isfinite(float(edge)):
            continue
        if float(edge) <= threshold:
            return int(row["horizon_bars"])
    return int(curve_rows[-1]["horizon_bars"])


def _effective_decay_horizon(curve_rows):
    if not curve_rows:
        return None

    baseline = curve_rows[0].get("net_edge")
    if baseline is None or not np.isfinite(float(baseline)):
        return None
    if float(baseline) <= 0.0:
        return 0

    for row in curve_rows[1:]:
        edge = row.get("net_edge")
        if edge is None or not np.isfinite(float(edge)):
            continue
        if float(edge) <= 0.0:
            return int(row["horizon_bars"])
    return int(curve_rows[-1]["horizon_bars"])


def _summarize_section(section_name, segments, *, horizons, delays, configured_signal_delay_bars, holding_bars):
    horizon_curve = _build_curve(
        section_name,
        segments,
        horizons,
        curve_kind="horizon",
        signal_delay_bars=configured_signal_delay_bars,
    )
    delay_curve = _build_curve(
        section_name,
        segments,
        delays,
        curve_kind="delay",
        holding_bars=holding_bars,
    )
    return {
        "horizon_curve": horizon_curve,
        "delay_curve": delay_curve,
    }


def _filter_segments_for_regime(segments, column, value):
    filtered_segments = []
    for segment in segments:
        regimes = segment.get("regimes")
        if regimes is None or regimes.empty or column not in regimes.columns:
            continue
        mask = regimes[column].eq(value).fillna(False)
        selected_index = mask.index[mask]
        if len(selected_index) == 0:
            continue
        filtered_segments.append(
            {
                **segment,
                "predictions": segment["predictions"].reindex(selected_index),
                "direction_edge": (
                    segment["direction_edge"].reindex(selected_index)
                    if segment.get("direction_edge") is not None
                    else None
                ),
                "event_signals": segment["event_signals"].reindex(selected_index),
                "regimes": regimes.reindex(selected_index),
            }
        )
    return filtered_segments


def _build_regime_summary(segments, *, policy, holding_bars, signal_delay_bars, execution_policy):
    regime_columns = []
    for segment in segments:
        regimes = segment.get("regimes")
        if regimes is None or regimes.empty:
            continue
        for column in regimes.columns:
            if column not in regime_columns:
                regime_columns.append(column)

    column_reports = {}
    for column in regime_columns:
        values = []
        for segment in segments:
            regimes = segment.get("regimes")
            if regimes is None or regimes.empty or column not in regimes.columns:
                continue
            values.extend([value for value in regimes[column].dropna().unique().tolist() if value not in values])

        rows = []
        for value in values:
            filtered_segments = _filter_segments_for_regime(segments, column, value)
            if not filtered_segments:
                continue

            report = build_signal_decay_report(
                filtered_segments,
                holding_bars=holding_bars,
                signal_delay_bars=signal_delay_bars,
                execution_policy=execution_policy,
                config=policy,
                include_regime_summary=False,
            )
            if int(report.get("trade_count", 0)) < int(policy.get("regime_min_samples", 20)):
                continue

            rows.append(
                {
                    "regime": str(value),
                    "trade_count": int(report.get("trade_count", 0)),
                    "half_life_bars": report.get("half_life_bars"),
                    "effective_decay_horizon_bars": report.get("effective_decay_horizon_bars"),
                    "net_edge_at_effective_delay": report.get("net_edge_at_effective_delay"),
                    "edge_retention_at_effective_delay": report.get("edge_retention_at_effective_delay"),
                }
            )

        if rows:
            column_reports[column] = rows

    return {
        "columns": column_reports,
        "column_count": int(len(column_reports)),
    }


def build_signal_decay_report(
    segments,
    *,
    holding_bars,
    signal_delay_bars,
    execution_policy=None,
    config=None,
    include_regime_summary=True,
):
    normalized_segments = [
        _normalize_segment(segment)
        for segment in list(segments or [])
        if segment is not None
    ]
    effective_delay_bars = resolve_effective_delay_bars(signal_delay_bars, execution_policy)
    policy = resolve_signal_decay_policy(
        config,
        holding_bars=holding_bars,
        effective_delay_bars=effective_delay_bars,
    )

    if not policy["enabled"]:
        return {
            "enabled": False,
            "observed": False,
            "segment_count": int(len(normalized_segments)),
            "configured_signal_delay_bars": int(signal_delay_bars),
            "effective_delay_bars": int(effective_delay_bars),
            "holding_bars": int(holding_bars),
            "policy": policy,
            "promotion_pass": True,
            "gate_mode": "disabled",
            "reasons": [],
            "warnings": [],
        }

    horizons = list(range(1, int(policy["max_horizon_bars"]) + 1))
    delay_start = max(0, int(signal_delay_bars))
    delays = list(range(delay_start, int(effective_delay_bars) + 1))
    if not delays:
        delays = [int(effective_delay_bars)]

    raw_scores = _summarize_section(
        "raw_scores",
        normalized_segments,
        horizons=horizons,
        delays=delays,
        configured_signal_delay_bars=int(signal_delay_bars),
        holding_bars=int(holding_bars),
    )
    thresholded_signals = _summarize_section(
        "thresholded_signals",
        normalized_segments,
        horizons=horizons,
        delays=delays,
        configured_signal_delay_bars=int(signal_delay_bars),
        holding_bars=int(holding_bars),
    )
    realized_outcomes = _summarize_section(
        "realized_outcomes",
        normalized_segments,
        horizons=horizons,
        delays=delays,
        configured_signal_delay_bars=int(signal_delay_bars),
        holding_bars=int(holding_bars),
    )

    realized_horizon_curve = realized_outcomes["horizon_curve"]
    realized_delay_curve = realized_outcomes["delay_curve"]
    thresholded_delay_curve = thresholded_signals["delay_curve"]
    raw_delay_curve = raw_scores["delay_curve"]

    configured_delay_row = _first_curve_row(realized_delay_curve, "delay_bars", int(signal_delay_bars))
    effective_delay_row = _first_curve_row(realized_delay_curve, "delay_bars", int(effective_delay_bars))
    effective_thresholded_row = _first_curve_row(thresholded_delay_curve, "delay_bars", int(effective_delay_bars))
    effective_raw_row = _first_curve_row(raw_delay_curve, "delay_bars", int(effective_delay_bars))

    half_life_bars = _empirical_half_life(realized_horizon_curve)
    effective_decay_horizon_bars = _effective_decay_horizon(realized_horizon_curve)
    net_edge_at_configured_delay = None if configured_delay_row is None else configured_delay_row.get("net_edge")
    net_edge_at_effective_delay = None if effective_delay_row is None else effective_delay_row.get("net_edge")
    gross_edge_at_effective_delay = None if effective_thresholded_row is None else effective_thresholded_row.get("gross_edge")
    raw_score_edge_at_effective_delay = None if effective_raw_row is None else effective_raw_row.get("score_edge")

    edge_retention = None
    if (
        net_edge_at_configured_delay is not None
        and np.isfinite(float(net_edge_at_configured_delay))
        and float(net_edge_at_configured_delay) > 0.0
        and net_edge_at_effective_delay is not None
        and np.isfinite(float(net_edge_at_effective_delay))
    ):
        edge_retention = float(net_edge_at_effective_delay) / float(net_edge_at_configured_delay)

    effective_trade_count = int(effective_delay_row.get("trade_count", 0)) if effective_delay_row is not None else 0
    warnings = []
    reasons = []
    low_sample_advisory = effective_trade_count < int(policy["min_realized_trade_count"])
    required_half_life_bars = float(policy["min_half_life_holding_ratio"]) * float(holding_bars)
    gate_mode = "blocking"
    promotion_pass = True

    if low_sample_advisory:
        warnings.append("insufficient_realized_decay_trade_count")
        gate_mode = "advisory"
    else:
        if net_edge_at_effective_delay is None or not np.isfinite(float(net_edge_at_effective_delay)) or float(net_edge_at_effective_delay) <= 0.0:
            reasons.append("edge_decays_before_effective_delay")
        elif float(net_edge_at_effective_delay) < float(policy["min_net_edge_at_effective_delay"]):
            reasons.append("net_edge_at_effective_delay_below_minimum")

        if half_life_bars is None or float(half_life_bars) < float(required_half_life_bars):
            reasons.append("signal_half_life_below_requirement")
        promotion_pass = not reasons

    report = {
        "enabled": True,
        "observed": bool(normalized_segments),
        "method": "empirical_edge_decay",
        "segment_count": int(len(normalized_segments)),
        "configured_signal_delay_bars": int(signal_delay_bars),
        "effective_delay_bars": int(effective_delay_bars),
        "holding_bars": int(holding_bars),
        "policy": policy,
        "required_half_life_bars": float(required_half_life_bars),
        "promotion_pass": bool(promotion_pass),
        "gate_mode": gate_mode,
        "reasons": reasons,
        "warnings": warnings,
        "low_sample_advisory": bool(low_sample_advisory),
        "trade_count": int(effective_trade_count),
        "half_life_bars": None if half_life_bars is None else int(half_life_bars),
        "effective_decay_horizon_bars": (
            None if effective_decay_horizon_bars is None else int(effective_decay_horizon_bars)
        ),
        "net_edge_at_configured_delay": net_edge_at_configured_delay,
        "net_edge_at_effective_delay": net_edge_at_effective_delay,
        "gross_edge_at_effective_delay": gross_edge_at_effective_delay,
        "raw_score_edge_at_effective_delay": raw_score_edge_at_effective_delay,
        "edge_retention_at_effective_delay": edge_retention,
        "gross_edge_by_horizon": thresholded_signals["horizon_curve"],
        "net_edge_by_horizon": realized_horizon_curve,
        "edge_by_execution_delay": realized_delay_curve,
        "raw_scores": raw_scores,
        "thresholded_signals": thresholded_signals,
        "realized_outcomes": realized_outcomes,
    }
    if include_regime_summary:
        report["regime_summary"] = _build_regime_summary(
            normalized_segments,
            policy=policy,
            holding_bars=holding_bars,
            signal_delay_bars=signal_delay_bars,
            execution_policy=execution_policy,
        )
    return report


__all__ = [
    "build_signal_decay_report",
    "resolve_effective_delay_bars",
    "resolve_signal_decay_policy",
]