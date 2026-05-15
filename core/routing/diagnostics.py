"""Router replay and diagnostics helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from ..regimes.contracts import RegimeStateContract
from .contracts import RouterStateSnapshot, RoutingDecisionContract


def _normalize_regime_sequence(regime_states: Sequence[Any] | None) -> list[RegimeStateContract]:
    normalized = []
    for item in list(regime_states or []):
        if isinstance(item, RegimeStateContract):
            normalized.append(RegimeStateContract.from_dict(item.to_dict()))
        elif isinstance(item, Mapping):
            normalized.append(RegimeStateContract.from_dict(item))
        else:
            raise TypeError("regime state trace entries must be RegimeStateContract values or mappings")
    return normalized


def _coerce_timestamp(value: Any) -> Any:
    if value is None:
        return None
    try:
        return pd.Timestamp(value)
    except Exception:
        return value


def _serialize_timestamp(value: Any) -> Any:
    if value is None:
        return None
    try:
        return pd.Timestamp(value).isoformat()
    except Exception:
        return value


def _timestamp_token(value: Any) -> tuple[int, Any]:
    if value is None:
        return (0, None)
    try:
        return (1, pd.Timestamp(value).value)
    except Exception:
        return (2, str(value))


def _timestamp_leq(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return True
    return _timestamp_token(left) <= _timestamp_token(right)


def _timestamp_eq(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return _timestamp_token(left) == _timestamp_token(right)


def _coerce_timedelta(value: Any) -> pd.Timedelta | None:
    if value in (None, "", {}):
        return None
    try:
        return pd.Timedelta(value)
    except Exception:
        return None


def _resolve_contract_expiry(contract: RegimeStateContract) -> Any:
    if contract.expires_at is not None:
        return contract.expires_at
    metadata = dict(contract.metadata or {})
    explicit_expiry = metadata.get("expires_at")
    if explicit_expiry is not None:
        return explicit_expiry
    max_age = contract.max_age if contract.max_age is not None else metadata.get("max_age")
    max_age_delta = _coerce_timedelta(max_age)
    if max_age_delta is None:
        return None
    anchor = contract.available_at if contract.available_at is not None else contract.as_of
    if anchor is None:
        return None
    try:
        return pd.Timestamp(anchor) + max_age_delta
    except Exception:
        return None


def build_admissible_router_regime_trace(
    regime_states: Sequence[Any] | None,
    decision_timestamps: Sequence[Any] | pd.Index,
) -> dict[str, Any]:
    normalized_states = _normalize_regime_sequence(regime_states)
    if not normalized_states:
        raise ValueError("regime_states must contain at least one entry")

    decision_index = pd.Index(decision_timestamps)
    sorted_states = sorted(
        normalized_states,
        key=lambda contract: (
            _timestamp_token(contract.as_of),
            _timestamp_token(contract.available_at if contract.available_at is not None else contract.as_of),
        ),
    )
    source_position_by_as_of = {
        _coerce_timestamp(contract.as_of): position
        for position, contract in enumerate(sorted_states)
    }

    aligned = []
    timing_blocked_count = 0
    stale_count = 0
    unavailable_count = 0
    warm_count = 0
    recognition_lags = []

    for decision_position, decision_time in enumerate(decision_index):
        same_row_contract = next(
            (
                contract
                for contract in sorted_states
                if _timestamp_eq(contract.as_of, decision_time)
            ),
            None,
        )
        latest_admissible = None
        for contract in sorted_states:
            contract_as_of = contract.as_of if contract.as_of is not None else contract.available_at
            contract_available_at = contract.available_at if contract.available_at is not None else contract.as_of
            if not _timestamp_leq(contract_as_of, decision_time):
                continue
            contract_expires_at = _resolve_contract_expiry(contract)
            if contract_expires_at is not None and not _timestamp_leq(decision_time, contract_expires_at):
                continue
            if _timestamp_leq(contract_available_at, decision_time):
                latest_admissible = contract

        if latest_admissible is not None:
            source_position = source_position_by_as_of.get(_coerce_timestamp(latest_admissible.as_of), decision_position)
            stale_bars = max(0, int(decision_position - int(source_position)))
            detector_outputs = dict(latest_admissible.detector_outputs or {})
            if stale_bars > 0:
                detector_outputs["stale"] = 1
            metadata = dict(latest_admissible.metadata or {})
            expires_at = _resolve_contract_expiry(latest_admissible)
            freshness_state = "stale" if stale_bars > 0 else str(latest_admissible.freshness_state or metadata.get("freshness_state") or "fresh")
            metadata.update(
                {
                    "availability_state": "stale" if stale_bars > 0 else str(metadata.get("availability_state") or "known"),
                    "decision_timestamp": _serialize_timestamp(decision_time),
                    "source_as_of": _serialize_timestamp(latest_admissible.as_of),
                    "source_available_at": _serialize_timestamp(latest_admissible.available_at),
                    "source_expires_at": _serialize_timestamp(expires_at),
                    "stale_bars": int(stale_bars),
                    "freshness_state": freshness_state,
                }
            )
            recognition_lag = latest_admissible.recognition_lag_bars
            if recognition_lag is not None:
                recognition_lags.append(float(recognition_lag))
            aligned.append(
                RegimeStateContract(
                    as_of=decision_time,
                    available_at=latest_admissible.available_at,
                    expires_at=expires_at,
                    source_available_at=latest_admissible.source_available_at,
                    label=latest_admissible.label,
                    probabilities=dict(latest_admissible.probabilities or {}),
                    confidence=latest_admissible.confidence,
                    confidence_kind=latest_admissible.confidence_kind,
                    detector_outputs=detector_outputs,
                    warm=bool(latest_admissible.warm),
                    recognition_lag_bars=latest_admissible.recognition_lag_bars,
                    max_age=latest_admissible.max_age,
                    freshness_state=freshness_state,
                    metadata=metadata,
                )
            )
            if stale_bars > 0:
                stale_count += 1
            continue

        detector_outputs = {}
        metadata = {
            "decision_timestamp": _serialize_timestamp(decision_time),
            "source_as_of": None,
            "source_available_at": None,
            "source_expires_at": None,
            "stale_bars": None,
        }
        availability_state = "unavailable"
        available_at = decision_time
        source_available_at = None
        recognition_lag_bars = None
        expires_at = None
        freshness_state = "unavailable"
        if same_row_contract is not None:
            detector_outputs.update(dict(same_row_contract.detector_outputs or {}))
            metadata.update(dict(same_row_contract.metadata or {}))
            metadata["source_as_of"] = _serialize_timestamp(same_row_contract.as_of)
            metadata["source_available_at"] = _serialize_timestamp(same_row_contract.available_at)
            expires_at = _resolve_contract_expiry(same_row_contract)
            metadata["source_expires_at"] = _serialize_timestamp(expires_at)
            available_at = same_row_contract.available_at if same_row_contract.available_at is not None else decision_time
            source_available_at = same_row_contract.source_available_at
            recognition_lag_bars = same_row_contract.recognition_lag_bars
            reason = str(metadata.get("reason") or "").strip().lower()
            if bool(int(detector_outputs.get("unavailable", 0) or 0)) or reason in {"unfitted", "missing_observation"}:
                availability_state = "unavailable"
                detector_outputs["unavailable"] = 1
                unavailable_count += 1
                freshness_state = "unavailable"
            elif not bool(same_row_contract.warm):
                availability_state = "warm"
                warm_count += 1
                freshness_state = "delayed"
            else:
                availability_state = "timing_blocked"
                detector_outputs["timing_blocked"] = 1
                timing_blocked_count += 1
                freshness_state = "delayed"
        else:
            detector_outputs["unavailable"] = 1
            metadata["reason"] = "missing_contract"
            unavailable_count += 1

        if recognition_lag_bars is not None:
            recognition_lags.append(float(recognition_lag_bars))
        metadata["availability_state"] = availability_state
        metadata["freshness_state"] = freshness_state
        aligned.append(
            RegimeStateContract(
                as_of=decision_time,
                available_at=available_at,
                expires_at=expires_at,
                source_available_at=source_available_at,
                label=None,
                probabilities={},
                confidence=None,
                confidence_kind=None,
                detector_outputs=detector_outputs,
                warm=False,
                recognition_lag_bars=recognition_lag_bars,
                freshness_state=freshness_state,
                metadata=metadata,
            )
        )

    return {
        "regime_states": aligned,
        "alignment": {
            "mode": "decision_time_admissible",
            "target_row_count": int(len(decision_index)),
            "source_row_count": int(len(sorted_states)),
            "timing_blocked_row_count": int(timing_blocked_count),
            "stale_row_count": int(stale_count),
            "warm_row_count": int(warm_count),
            "unavailable_row_count": int(unavailable_count),
            "mean_recognition_lag_bars": (
                None
                if not recognition_lags
                else float(sum(recognition_lags) / len(recognition_lags))
            ),
            "max_recognition_lag_bars": (
                None
                if not recognition_lags
                else float(max(recognition_lags))
            ),
        },
    }


def replay_router_trace(
    router,
    specialists,
    regime_states: Sequence[Any] | None,
    *,
    specialist_health_trace: Sequence[Any] | None = None,
    decision_timestamps: Sequence[Any] | None = None,
):
    normalized_states = _normalize_regime_sequence(regime_states)
    if not normalized_states:
        raise ValueError("regime_states must contain at least one entry")
    decision_index = None if decision_timestamps is None else pd.Index(decision_timestamps)
    if decision_index is not None and len(decision_index) != len(normalized_states):
        raise ValueError("decision_timestamps must match the regime_states length")

    state = router.initialize(specialists)
    decisions = []
    states = [RouterStateSnapshot.from_dict(state.to_dict()).to_dict()]
    selected_model_ids = []
    blocked_switch_reasons = {}
    route_reason_counts = {}
    regime_availability_counts = {}
    safe_mode_action_counts = {}
    allocation_control_reason_counts = {}
    switch_count = 0
    allocation_change_count = 0
    blocked_allocation_count = 0
    executed_weight_l1_change_total = 0.0
    executed_weight_turnover_total = 0.0
    effective_model_count_total = 0.0
    max_effective_model_count = 0.0
    previous_active_model_id = state.active_model_id

    for index, regime_state in enumerate(normalized_states):
        health_payload = None
        if specialist_health_trace is not None and index < len(specialist_health_trace):
            health_payload = specialist_health_trace[index]
        decision_time = regime_state.as_of
        if decision_index is not None:
            decision_time = decision_index[index]
        state, decision = router.select(
            state,
            regime_state,
            specialist_health=health_payload,
            timestamp=decision_time,
        )
        if not isinstance(decision, RoutingDecisionContract):
            decision = RoutingDecisionContract.from_dict(decision)
        decision_payload = decision.to_dict()
        decision_payload["decision_timestamp"] = _serialize_timestamp(decision_time)
        decisions.append(decision_payload)
        states.append(RouterStateSnapshot.from_dict(state.to_dict()).to_dict())
        selected_model_ids.append(decision.selected_model_id)
        route_reason_counts[str(decision.route_reason)] = int(route_reason_counts.get(str(decision.route_reason), 0)) + 1
        availability_state = str((decision.metadata or {}).get("regime_availability_state") or "known")
        regime_availability_counts[availability_state] = int(regime_availability_counts.get(availability_state, 0)) + 1
        safe_mode_action = (decision.metadata or {}).get("safe_mode_action")
        if safe_mode_action:
            safe_mode_action = str(safe_mode_action)
            safe_mode_action_counts[safe_mode_action] = int(safe_mode_action_counts.get(safe_mode_action, 0)) + 1
        allocation_control_reason = decision.allocation_control_reason
        if allocation_control_reason:
            allocation_control_reason = str(allocation_control_reason)
            allocation_control_reason_counts[allocation_control_reason] = int(
                allocation_control_reason_counts.get(allocation_control_reason, 0)
            ) + 1
        blocked_reason = (decision.metadata or {}).get("blocked_switch_reason")
        if blocked_reason:
            blocked_switch_reasons[str(blocked_reason)] = int(blocked_switch_reasons.get(str(blocked_reason), 0)) + 1
            selected_model_id = None if decision.selected_model_id is None else str(decision.selected_model_id)
            blocked_allocation = any(
                str(model_id) != selected_model_id and float(weight) != 0.0
                for model_id, weight in dict(decision.weights or {}).items()
            )
            if blocked_allocation:
                blocked_allocation_count += 1
        executed_weight_l1_change_total += float(decision.executed_weight_l1_change or 0.0)
        executed_weight_turnover_total += float(decision.executed_weight_turnover or 0.0)
        effective_model_count = float(decision.effective_model_count or 0.0)
        effective_model_count_total += effective_model_count
        max_effective_model_count = max(max_effective_model_count, effective_model_count)
        if float(decision.executed_weight_turnover or 0.0) > 0.0:
            allocation_change_count += 1
        if state.active_model_id != previous_active_model_id:
            switch_count += 1
            previous_active_model_id = state.active_model_id

    decision_count = len(decisions)
    return {
        "manifest": router.manifest().to_dict(),
        "initial_state": states[0],
        "state_trace": states[1:],
        "decision_trace": decisions,
        "summary": {
            "decision_count": decision_count,
            "switch_count": int(switch_count),
            "selected_model_ids": selected_model_ids,
            "route_reason_counts": route_reason_counts,
            "blocked_switch_reasons": blocked_switch_reasons,
            "allocation_change_count": int(allocation_change_count),
            "blocked_allocation_count": int(blocked_allocation_count),
            "executed_weight_l1_change_total": float(executed_weight_l1_change_total),
            "executed_weight_turnover_total": float(executed_weight_turnover_total),
            "mean_executed_weight_l1_change": (
                float(executed_weight_l1_change_total / decision_count) if decision_count > 0 else 0.0
            ),
            "mean_executed_weight_turnover": (
                float(executed_weight_turnover_total / decision_count) if decision_count > 0 else 0.0
            ),
            "mean_effective_model_count": (
                float(effective_model_count_total / decision_count) if decision_count > 0 else 0.0
            ),
            "max_effective_model_count": float(max_effective_model_count),
            "allocation_control_reason_counts": allocation_control_reason_counts,
            "regime_availability_counts": regime_availability_counts,
            "safe_mode_action_counts": safe_mode_action_counts,
        },
    }


__all__ = ["build_admissible_router_regime_trace", "replay_router_trace"]