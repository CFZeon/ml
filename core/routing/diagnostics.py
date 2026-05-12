"""Router replay and diagnostics helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

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


def replay_router_trace(router, specialists, regime_states: Sequence[Any] | None, *, specialist_health_trace: Sequence[Any] | None = None):
    normalized_states = _normalize_regime_sequence(regime_states)
    if not normalized_states:
        raise ValueError("regime_states must contain at least one entry")

    state = router.initialize(specialists)
    decisions = []
    states = [RouterStateSnapshot.from_dict(state.to_dict()).to_dict()]
    selected_model_ids = []
    blocked_switch_reasons = {}
    route_reason_counts = {}
    regime_availability_counts = {}
    safe_mode_action_counts = {}
    switch_count = 0
    previous_active_model_id = state.active_model_id

    for index, regime_state in enumerate(normalized_states):
        health_payload = None
        if specialist_health_trace is not None and index < len(specialist_health_trace):
            health_payload = specialist_health_trace[index]
        state, decision = router.select(
            state,
            regime_state,
            specialist_health=health_payload,
            timestamp=regime_state.available_at,
        )
        if not isinstance(decision, RoutingDecisionContract):
            decision = RoutingDecisionContract.from_dict(decision)
        decisions.append(decision.to_dict())
        states.append(RouterStateSnapshot.from_dict(state.to_dict()).to_dict())
        selected_model_ids.append(decision.selected_model_id)
        route_reason_counts[str(decision.route_reason)] = int(route_reason_counts.get(str(decision.route_reason), 0)) + 1
        availability_state = str((decision.metadata or {}).get("regime_availability_state") or "known")
        regime_availability_counts[availability_state] = int(regime_availability_counts.get(availability_state, 0)) + 1
        safe_mode_action = (decision.metadata or {}).get("safe_mode_action")
        if safe_mode_action:
            safe_mode_action = str(safe_mode_action)
            safe_mode_action_counts[safe_mode_action] = int(safe_mode_action_counts.get(safe_mode_action, 0)) + 1
        blocked_reason = (decision.metadata or {}).get("blocked_switch_reason")
        if blocked_reason:
            blocked_switch_reasons[str(blocked_reason)] = int(blocked_switch_reasons.get(str(blocked_reason), 0)) + 1
        if state.active_model_id != previous_active_model_id:
            switch_count += 1
            previous_active_model_id = state.active_model_id

    return {
        "manifest": router.manifest().to_dict(),
        "initial_state": states[0],
        "state_trace": states[1:],
        "decision_trace": decisions,
        "summary": {
            "decision_count": len(decisions),
            "switch_count": int(switch_count),
            "selected_model_ids": selected_model_ids,
            "route_reason_counts": route_reason_counts,
            "blocked_switch_reasons": blocked_switch_reasons,
            "regime_availability_counts": regime_availability_counts,
            "safe_mode_action_counts": safe_mode_action_counts,
        },
    }


__all__ = ["replay_router_trace"]