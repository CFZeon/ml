"""Venue failure and stress-scenario helpers for backtests."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .universe import build_symbol_lifecycle_frame


@dataclass
class ScenarioEvent:
    event_type: str
    start: pd.Timestamp
    end: pd.Timestamp | None = None
    value: float | None = None
    control_intent: str | None = None
    control_tags: tuple[str, ...] | None = None


def _coerce_scenario_frame(index, scenario_schedule=None):
    timeline = pd.DatetimeIndex(index)
    base = pd.DataFrame(
        {
            "venue_down": False,
            "stale_mark": False,
            "trading_halt": False,
            "forced_deleveraging": False,
            "leverage_cap": np.nan,
        },
        index=timeline,
    )
    if scenario_schedule is None:
        return base

    if isinstance(scenario_schedule, pd.DataFrame):
        schedule = scenario_schedule.copy()
    else:
        schedule = pd.DataFrame(scenario_schedule)

    if schedule.empty:
        return base

    schedule.index = pd.DatetimeIndex(schedule.index)
    schedule = schedule.reindex(timeline)
    for column in base.columns:
        if column not in schedule.columns:
            schedule[column] = base[column]
    for column in ["venue_down", "stale_mark", "trading_halt", "forced_deleveraging"]:
        schedule[column] = schedule[column].fillna(False).astype(bool)
    schedule["leverage_cap"] = pd.to_numeric(schedule["leverage_cap"], errors="coerce")
    return schedule[base.columns]


def build_scenario_schedule(index, events=None):
    timeline = pd.DatetimeIndex(index)
    if len(timeline) == 0:
        return pd.DataFrame(index=timeline)
    if isinstance(events, pd.DataFrame):
        return _coerce_scenario_frame(timeline, events)

    schedule = _coerce_scenario_frame(timeline)
    if not events:
        return schedule

    for raw_event in events:
        event = raw_event if isinstance(raw_event, dict) else vars(raw_event)
        event_type = str(event.get("event_type") or event.get("type") or "").lower()
        start = pd.Timestamp(event.get("start") or event.get("timestamp"))
        end = pd.Timestamp(event.get("end")) if event.get("end") is not None else start
        mask = (timeline >= start) & (timeline <= end)
        if not mask.any():
            continue

        if event_type in {"downtime", "venue_down", "outage"}:
            schedule.loc[mask, "venue_down"] = True
        elif event_type in {"stale_mark", "stale_marks"}:
            schedule.loc[mask, "stale_mark"] = True
        elif event_type in {"halt", "trading_halt"}:
            schedule.loc[mask, "trading_halt"] = True
        elif event_type in {"forced_deleveraging", "deleveraging"}:
            schedule.loc[mask, "forced_deleveraging"] = True
        elif event_type in {"leverage_change", "leverage_cap"}:
            schedule.loc[mask, "leverage_cap"] = float(event.get("value") or event.get("leverage_cap") or np.nan)

    return schedule


def summarize_scenario_schedule(index, scenario_schedule=None):
    schedule = _coerce_scenario_frame(index, scenario_schedule)
    event_columns = {
        "venue_down": "downtime",
        "stale_mark": "stale_mark",
        "trading_halt": "halt",
        "forced_deleveraging": "forced_deleveraging",
        "leverage_cap": "leverage_cap",
    }
    configured_event_types = []
    configured_event_count = 0
    for column, event_name in event_columns.items():
        series = schedule.get(column)
        if series is None:
            continue
        if column == "leverage_cap":
            active = pd.Series(series, index=schedule.index).notna()
        else:
            active = pd.Series(series, index=schedule.index).fillna(False).astype(bool)
        if active.any():
            configured_event_types.append(event_name)
            configured_event_count += int(active.sum())

    return {
        "configured": bool(configured_event_types),
        "configured_event_types": configured_event_types,
        "configured_event_count": int(configured_event_count),
    }


def apply_scenario_price_policy(valuation_series, execution_series, scenario_schedule, policy=None):
    policy = dict(policy or {})
    schedule = _coerce_scenario_frame(valuation_series.index, scenario_schedule)
    valuation = pd.Series(valuation_series, copy=False).astype(float).copy()
    execution = pd.Series(execution_series, copy=False).astype(float).copy()
    report = {
        "stale_mark_action": str(policy.get("stale_mark_action", "warn")).lower(),
        "stale_mark_rejections": 0,
        "warnings": [],
    }
    report.update(summarize_scenario_schedule(valuation.index, schedule))

    stale_mask = schedule.get("stale_mark", pd.Series(False, index=valuation.index)).astype(bool)
    if stale_mask.any():
        if report["stale_mark_action"] == "reject":
            valuation.loc[stale_mask] = np.nan
            report["stale_mark_rejections"] = int(stale_mask.sum())
        else:
            report["warnings"].append("stale_mark_warning")
    return valuation, execution, report


def merge_scenario_lifecycle(index, symbol_lifecycle=None, scenario_schedule=None, symbol=None):
    schedule = _coerce_scenario_frame(index, scenario_schedule)
    halted = schedule.get("trading_halt", pd.Series(False, index=pd.Index(index))).fillna(False).astype(bool)
    base_lifecycle = build_symbol_lifecycle_frame(index=index, symbol=symbol, events=symbol_lifecycle)

    if schedule.empty or "trading_halt" not in schedule.columns or not halted.any():
        return base_lifecycle

    if base_lifecycle is None:
        base_lifecycle = pd.DataFrame({"status": "TRADING"}, index=pd.Index(index))
    scenario_status = pd.Series("TRADING", index=halted.index, dtype=object)
    scenario_status.loc[halted] = "HALTED"
    combined = base_lifecycle.copy()
    combined["status"] = combined["status"].fillna("TRADING").astype(str).str.upper()
    combined.loc[scenario_status == "HALTED", "status"] = "HALTED"
    return combined[["status"]]


def apply_execution_scenarios(execution_report, scenario_schedule, policy=None):
    policy = dict(policy or {})
    report = {
        "downtime_action": str(policy.get("downtime_action", "freeze")).lower(),
        "suppressed_orders": 0,
        "forced_deleveraging_rows": 0,
    }
    schedule = _coerce_scenario_frame(execution_report["position"].index, scenario_schedule)
    report.update(summarize_scenario_schedule(execution_report["position"].index, schedule))
    if schedule.empty:
        execution_report["scenario_report"] = report
        return execution_report

    position = pd.Series(execution_report["position"], copy=False).astype(float).copy()
    previous_position = position.shift(1).fillna(0.0)
    order_ledger = copy.deepcopy(execution_report.get("order_ledger", pd.DataFrame()))
    order_intents = copy.deepcopy(execution_report.get("order_intents", pd.DataFrame()))

    venue_down = schedule.get("venue_down", pd.Series(False, index=position.index)).fillna(False).astype(bool)
    leverage_cap = pd.Series(schedule.get("leverage_cap"), index=position.index)
    has_cap = leverage_cap.notna()
    deleveraging = schedule.get("forced_deleveraging", pd.Series(False, index=position.index)).fillna(False).astype(bool)
    if not venue_down.any() and not has_cap.any() and not deleveraging.any():
        execution_report = dict(execution_report)
        execution_report["scenario_report"] = report
        return execution_report

    if venue_down.any():
        if report["downtime_action"] == "kill_switch":
            position.loc[venue_down] = 0.0
        else:
            position.loc[venue_down] = previous_position.loc[venue_down]

        if isinstance(order_ledger, pd.DataFrame) and not order_ledger.empty and "timestamp" in order_ledger.columns:
            suppressed_mask = order_ledger["timestamp"].isin(position.index[venue_down])
            report["suppressed_orders"] = int(suppressed_mask.sum())
            order_ledger = order_ledger.loc[~suppressed_mask].copy()

    if has_cap.any():
        for timestamp in leverage_cap.index[has_cap]:
            cap = abs(float(leverage_cap.loc[timestamp]))
            position.loc[timestamp] = float(np.clip(position.loc[timestamp], -cap, cap))

    if deleveraging.any():
        action = str(policy.get("forced_deleveraging_action", "liquidate")).lower()
        report["forced_deleveraging_rows"] = int(deleveraging.sum())
        if action == "liquidate":
            position.loc[deleveraging] = 0.0

    if isinstance(order_ledger, pd.DataFrame) and not order_ledger.empty and "timestamp" in order_ledger.columns:
        aligned_previous = position.shift(1).fillna(0.0)
        row_timestamps = pd.to_datetime(order_ledger["timestamp"], utc=True)
        valid_rows = pd.Series(True, index=order_ledger.index, dtype=bool)
        if "executed_position" in order_ledger.columns:
            expected_executed = row_timestamps.map(position).astype(float)
            valid_rows &= np.isclose(
                order_ledger["executed_position"].astype(float),
                expected_executed,
                rtol=1e-9,
                atol=1e-12,
            )
        if "previous_position" in order_ledger.columns:
            expected_previous = row_timestamps.map(aligned_previous).astype(float)
            valid_rows &= np.isclose(
                order_ledger["previous_position"].astype(float),
                expected_previous,
                rtol=1e-9,
                atol=1e-12,
            )
        order_ledger = order_ledger.loc[valid_rows].copy()

    execution_report = dict(execution_report)
    execution_report["position"] = position
    execution_report["order_ledger"] = order_ledger
    execution_report["order_intents"] = order_intents
    if isinstance(order_intents, pd.DataFrame) and not order_intents.empty:
        requested_total = float(order_intents.get("requested_notional", pd.Series(dtype=float)).sum())
    else:
        requested_total = 0.0
    if isinstance(order_ledger, pd.DataFrame) and not order_ledger.empty:
        executed_total = float(order_ledger.get("executed_notional", pd.Series(dtype=float)).sum())
        execution_report["accepted_orders"] = int((order_ledger.get("status") == "accepted").sum())
        execution_report["partial_fill_orders"] = int((order_ledger.get("status") == "partial_fill").sum())
        execution_report["cancelled_orders"] = int((order_ledger.get("status") == "cancelled").sum()) + int(report["suppressed_orders"])
        execution_report["fill_ratio"] = executed_total / requested_total if requested_total > 0.0 else 0.0
    else:
        execution_report["accepted_orders"] = 0
        execution_report["partial_fill_orders"] = 0
        execution_report["cancelled_orders"] = int(execution_report.get("cancelled_orders", 0)) + int(report["suppressed_orders"])
        execution_report["fill_ratio"] = 0.0
    execution_report["scenario_report"] = report
    return execution_report


def run_scenario_matrix(run_backtest_fn, base_kwargs, scenario_matrix):
    results = {"base": run_backtest_fn(**dict(base_kwargs))}
    for name, payload in dict(scenario_matrix or {}).items():
        kwargs = dict(base_kwargs)
        kwargs["scenario_schedule"] = payload.get("events") or payload.get("scenario_schedule")
        kwargs["scenario_policy"] = payload.get("policy") or {}
        scenario_result = run_backtest_fn(**kwargs)
        if isinstance(scenario_result, dict):
            control_intent = payload.get("control_intent")
            control_tags = [
                str(tag)
                for tag in list(payload.get("control_tags") or ([] if control_intent is None else [control_intent]))
                if str(tag).strip()
            ]
            scenario_report = dict(scenario_result.get("scenario_report") or {})
            scenario_report["scenario_name"] = str(name)
            if control_intent is not None:
                scenario_report["control_intent"] = str(control_intent)
            if control_tags:
                scenario_report["control_tags"] = control_tags
            scenario_result = dict(scenario_result)
            scenario_result["scenario_report"] = scenario_report
        results[str(name)] = scenario_result
    return results


__all__ = [
    "ScenarioEvent",
    "apply_execution_scenarios",
    "apply_scenario_price_policy",
    "build_scenario_schedule",
    "merge_scenario_lifecycle",
    "run_scenario_matrix",
]