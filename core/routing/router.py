"""Deterministic specialist-library router implementations."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ..regimes.contracts import RegimeStateContract
from ..specialists import (
    SpecialistHealthContract,
    build_specialist_selection_contract,
    normalize_specialist_library_snapshot,
)
from .contracts import RouterManifest, RouterStateSnapshot, RoutingDecisionContract, RoutingScoreComponent


_VALID_SAFE_MODE_POLICIES = {"fallback_only", "no_trade"}
_VALID_ALLOCATION_MODES = {"selection_only", "mixture_allocation"}


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_state(state: Any) -> RouterStateSnapshot:
    if state is None:
        return RouterStateSnapshot()
    if isinstance(state, RouterStateSnapshot):
        return RouterStateSnapshot.from_dict(state.to_dict())
    if isinstance(state, Mapping):
        return RouterStateSnapshot.from_dict(state)
    raise TypeError("router state must be a RouterStateSnapshot or mapping")


def _normalize_regime_state(regime_state: Any, *, timestamp: Any) -> RegimeStateContract:
    if isinstance(regime_state, RegimeStateContract):
        return RegimeStateContract.from_dict(regime_state.to_dict())
    if isinstance(regime_state, Mapping):
        return RegimeStateContract.from_dict(regime_state)
    return RegimeStateContract(as_of=timestamp, available_at=timestamp)


def _normalize_safe_mode_policy(value: Any) -> str:
    policy = str(value or "fallback_only").strip().lower()
    if policy not in _VALID_SAFE_MODE_POLICIES:
        return "fallback_only"
    return policy


def _normalize_allocation_mode(value: Any) -> str:
    mode = str(value or "mixture_allocation").strip().lower()
    if mode not in _VALID_ALLOCATION_MODES:
        return "mixture_allocation"
    return mode


def _classify_regime_availability(regime_state: RegimeStateContract) -> str:
    detector_outputs = dict(regime_state.detector_outputs or {})
    metadata = dict(regime_state.metadata or {})
    reason = str(metadata.get("reason") or "").strip().lower()

    if bool(detector_outputs.get("unavailable", 0)) or reason in {"unfitted", "missing_observation"}:
        return "unavailable"
    if regime_state.label is None and not detector_outputs and regime_state.confidence is None:
        return "unavailable"
    if not bool(regime_state.warm):
        return "warm"
    return "known"


def _normalize_specialist_health_map(
    snapshot,
    specialist_health: Any = None,
) -> dict[str, SpecialistHealthContract]:
    base_map = {
        str(contract.model_id): SpecialistHealthContract.from_dict(contract.to_dict())
        for contract in list(snapshot.health or [])
    }
    if specialist_health is None:
        return base_map
    if isinstance(specialist_health, Mapping) and "health" in specialist_health:
        specialist_health = specialist_health.get("health")
    if isinstance(specialist_health, Mapping):
        items = specialist_health.values()
    else:
        items = list(specialist_health or [])
    for item in items:
        contract = item if isinstance(item, SpecialistHealthContract) else SpecialistHealthContract.from_dict(item)
        base_map[str(contract.model_id)] = contract
    return base_map


def _normalize_weight_map(weights: Mapping[str, float] | None) -> dict[str, float]:
    cleaned = {
        str(model_id): float(weight)
        for model_id, weight in dict(weights or {}).items()
        if abs(float(weight)) > 0.0
    }
    total = float(sum(cleaned.values()))
    if total <= 0.0:
        return {}
    return {model_id: float(weight) / total for model_id, weight in cleaned.items()}


def _effective_model_count(weights: Mapping[str, float] | None) -> float:
    normalized = _normalize_weight_map(weights)
    if not normalized:
        return 0.0
    squared_sum = float(sum(float(weight) ** 2 for weight in normalized.values()))
    if squared_sum <= 0.0:
        return 0.0
    return float(1.0 / squared_sum)


def _compute_weight_change(previous_weights: Mapping[str, float] | None, current_weights: Mapping[str, float] | None) -> tuple[float, float]:
    previous = _normalize_weight_map(previous_weights)
    current = _normalize_weight_map(current_weights)
    model_ids = set(previous.keys()) | set(current.keys())
    l1_change = float(sum(abs(float(current.get(model_id, 0.0)) - float(previous.get(model_id, 0.0))) for model_id in model_ids))
    turnover = float(0.5 * l1_change)
    return l1_change, turnover


def _previous_executed_weights(state: RouterStateSnapshot) -> dict[str, float]:
    metadata = dict(state.metadata or {})
    stored = metadata.get("executed_weights")
    if isinstance(stored, Mapping):
        return _normalize_weight_map(stored)
    if state.active_model_id is None:
        return {}
    return {str(state.active_model_id): 1.0}


def _attach_executed_weights(state: RouterStateSnapshot, weights: Mapping[str, float] | None) -> RouterStateSnapshot:
    metadata = dict(state.metadata or {})
    metadata["executed_weights"] = _normalize_weight_map(weights)
    return RouterStateSnapshot(
        active_model_id=state.active_model_id,
        last_switch_at=state.last_switch_at,
        cooldown_active=state.cooldown_active,
        pending_challenger_id=state.pending_challenger_id,
        pending_challenger_streak=state.pending_challenger_streak,
        metadata=metadata,
    )


class _BaseSpecialistRouter:
    score_component_names = ["regime_confidence", "stability_score", "decay_score", "failure_flags"]

    def __init__(
        self,
        *,
        policy_name: str | None = None,
        hysteresis_margin: float = 0.0,
        min_persistence_bars: int = 1,
        cooldown_bars: int = 0,
        fallback_bias: float = 0.45,
        stability_weight: float = 0.25,
        decay_weight: float = 0.15,
        failure_flag_penalty: float = 0.2,
        missing_health_score: float = 0.5,
        safe_mode_policy: str = "fallback_only",
    ):
        self.policy_name = None if policy_name is None else str(policy_name)
        self.hysteresis_margin = float(max(0.0, hysteresis_margin))
        self.min_persistence_bars = int(max(1, int(min_persistence_bars)))
        self.cooldown_bars = int(max(0, int(cooldown_bars)))
        self.fallback_bias = float(fallback_bias)
        self.stability_weight = float(stability_weight)
        self.decay_weight = float(decay_weight)
        self.failure_flag_penalty = float(failure_flag_penalty)
        self.missing_health_score = float(missing_health_score)
        self.safe_mode_policy = _normalize_safe_mode_policy(safe_mode_policy)
        self._snapshot = None

    @property
    def router_type(self) -> str:
        raise NotImplementedError

    def initialize(self, specialists: Any) -> RouterStateSnapshot:
        snapshot = normalize_specialist_library_snapshot(specialists)
        if snapshot is None:
            raise ValueError("specialist library is required to initialize router")
        self._snapshot = snapshot
        selection_contract = build_specialist_selection_contract(snapshot)
        active_model_ids = list(selection_contract.get("active_model_ids") or [])
        fallback_model_id = snapshot.fallback_model_id
        active_model_id = None
        if fallback_model_id and fallback_model_id in active_model_ids:
            active_model_id = str(fallback_model_id)
        elif active_model_ids:
            active_model_id = str(active_model_ids[0])
        return RouterStateSnapshot(
            active_model_id=active_model_id,
            cooldown_active=False,
            pending_challenger_id=None,
            pending_challenger_streak=0,
            metadata={
                "decision_count": 0,
                "last_switch_decision_index": None,
                "executed_weights": ({str(active_model_id): 1.0} if active_model_id is not None else {}),
            },
        )

    def manifest(self) -> RouterManifest:
        return RouterManifest(
            router_type=self.router_type,
            score_component_names=list(self.score_component_names),
            policy_name=self.policy_name,
            hysteresis_margin=self.hysteresis_margin,
            min_persistence_bars=self.min_persistence_bars,
            cooldown_bars=self.cooldown_bars,
            metadata={
                "fallback_bias": self.fallback_bias,
                "stability_weight": self.stability_weight,
                "decay_weight": self.decay_weight,
                "failure_flag_penalty": self.failure_flag_penalty,
                "safe_mode_policy": self.safe_mode_policy,
            },
        )

    def _selection_contract(self) -> dict[str, Any]:
        if self._snapshot is None:
            raise RuntimeError("router must be initialized before select()")
        return build_specialist_selection_contract(self._snapshot)

    def _specialist_spec_map(self) -> dict[str, Any]:
        if self._snapshot is None:
            return {}
        return {str(spec.model_id): spec for spec in list(self._snapshot.specialists or [])}

    def _candidate_model_ids(self) -> list[str]:
        selection_contract = self._selection_contract()
        candidate_model_ids = [str(item) for item in list(selection_contract.get("active_model_ids") or [])]
        if not candidate_model_ids:
            candidate_model_ids = [str(item) for item in list(selection_contract.get("certified_model_ids") or [])]
        if not candidate_model_ids:
            candidate_model_ids = [str(item) for item in list(selection_contract.get("candidate_model_ids") or [])]
        if not candidate_model_ids:
            candidate_model_ids = [str(item) for item in list(selection_contract.get("degraded_model_ids") or [])]

        if self._snapshot and self._snapshot.fallback_model_id is not None:
            fallback_model_id = str(self._snapshot.fallback_model_id)
            if fallback_model_id not in candidate_model_ids:
                candidate_model_ids.append(fallback_model_id)
        return candidate_model_ids

    def _score_model(
        self,
        model_id: str,
        *,
        regime_state: RegimeStateContract,
        health_map: Mapping[str, SpecialistHealthContract],
    ) -> tuple[float, list[RoutingScoreComponent]]:
        selection_contract = self._selection_contract()
        regime_label = None if regime_state.label is None else str(regime_state.label)
        confidence_kind = str(getattr(regime_state, "confidence_kind", None) or "").strip().lower()
        regime_confidence = 0.0
        if confidence_kind in {"", "posterior", "calibrated_score"}:
            regime_confidence = _coerce_float(regime_state.confidence, 0.0)
        compatible_regimes = list((selection_contract.get("compatible_regimes") or {}).get(str(model_id)) or [])
        if compatible_regimes:
            compatibility_score = 1.0 if regime_label in compatible_regimes else 0.0
        elif self._snapshot and str(model_id) == str(self._snapshot.fallback_model_id):
            compatibility_score = self.fallback_bias
        else:
            compatibility_score = 0.5

        health = health_map.get(str(model_id))
        stability_score = self.missing_health_score if health is None else _coerce_float(health.stability_score, self.missing_health_score)
        decay_score = 0.0 if health is None else _coerce_float(health.decay_score, 0.0)
        failure_flags = [] if health is None else list(health.failure_flags or [])
        failure_penalty = self.failure_flag_penalty if failure_flags else 0.0

        regime_component = regime_confidence * compatibility_score
        total_score = regime_component + (self.stability_weight * stability_score) - (self.decay_weight * decay_score) - failure_penalty
        components = [
            RoutingScoreComponent(
                name="regime_confidence",
                value=float(regime_confidence),
                weight=float(compatibility_score),
                penalized=False,
                metadata={
                    "regime_label": regime_label,
                    "compatible_regimes": compatible_regimes,
                    "confidence_kind": confidence_kind or None,
                },
            ),
            RoutingScoreComponent(
                name="stability_score",
                value=float(stability_score),
                weight=float(self.stability_weight),
                penalized=False,
            ),
            RoutingScoreComponent(
                name="decay_score",
                value=float(decay_score),
                weight=-float(self.decay_weight),
                penalized=bool(decay_score > 0.0),
            ),
            RoutingScoreComponent(
                name="failure_flags",
                value=float(len(failure_flags)),
                weight=-float(self.failure_flag_penalty),
                penalized=bool(failure_flags),
                metadata={"failure_flags": failure_flags},
            ),
        ]
        return float(total_score), components

    def _build_candidate_eligibility(
        self,
        *,
        regime_state: RegimeStateContract,
        health_map: Mapping[str, SpecialistHealthContract],
    ) -> dict[str, dict[str, Any]]:
        selection_contract = self._selection_contract()
        spec_map = self._specialist_spec_map()
        regime_label = None if regime_state.label is None else str(regime_state.label)
        lifecycle_state_by_model_id = dict(selection_contract.get("lifecycle_state_by_model_id") or {})
        compatible_regimes_by_model_id = dict(selection_contract.get("compatible_regimes") or {})
        fallback_model_id = None
        if self._snapshot and self._snapshot.fallback_model_id is not None:
            fallback_model_id = str(self._snapshot.fallback_model_id)

        eligibility = {}
        for model_id in self._candidate_model_ids():
            resolved_model_id = str(model_id)
            spec = spec_map.get(resolved_model_id)
            spec_metadata = dict((spec.metadata or {}) if spec is not None else {})
            health = health_map.get(resolved_model_id)
            health_metadata = dict((health.metadata or {}) if health is not None else {})
            lifecycle_state = str(
                lifecycle_state_by_model_id.get(resolved_model_id)
                or spec_metadata.get("lifecycle_state")
                or "candidate"
            )
            compatible_regimes = [
                str(item)
                for item in list(
                    compatible_regimes_by_model_id.get(resolved_model_id)
                    or (spec.compatible_regimes if spec is not None else [])
                    or []
                )
            ]
            incompatible_regimes = [
                str(item)
                for item in list((spec.incompatible_regimes if spec is not None else []) or [])
            ]
            failure_flags = [] if health is None else [str(item) for item in list(health.failure_flags or [])]
            health_state = str(health_metadata.get("health_state") or "").strip().lower()
            calibration_expired = bool(
                health_metadata.get("calibration_expired")
                or health_metadata.get("calibration_stale")
                or spec_metadata.get("calibration_expired")
            )
            is_fallback = bool(fallback_model_id is not None and resolved_model_id == fallback_model_id)

            reasons = []
            if lifecycle_state == "retired":
                reasons.append("lifecycle_retired")
            if not is_fallback and lifecycle_state == "degraded":
                reasons.append("lifecycle_degraded")
            if not is_fallback and compatible_regimes and regime_label not in compatible_regimes:
                reasons.append("regime_incompatible")
            if regime_label is not None and regime_label in incompatible_regimes:
                reasons.append("regime_explicitly_incompatible")
            if failure_flags:
                reasons.append("health_failure_flags")
            if health_state in {"failed", "degraded", "blocked"} or bool(health_metadata.get("eligibility_blocked", False)):
                reasons.append("health_blocked")
            if calibration_expired:
                reasons.append("calibration_expired")
            if not is_fallback and bool(getattr(health, "fallback_only", False)):
                reasons.append("fallback_only")

            eligibility[resolved_model_id] = {
                "model_id": resolved_model_id,
                "eligible": not reasons,
                "reasons": list(dict.fromkeys(reasons)),
                "fallback_model": is_fallback,
                "regime_label": regime_label,
                "compatible_regimes": compatible_regimes,
                "lifecycle_state": lifecycle_state,
                "health_failure_flags": failure_flags,
                "health_state": None if not health_state else health_state,
                "calibration_fresh": not calibration_expired,
            }
        return eligibility

    def _resolve_safe_mode_target(self) -> tuple[str | None, str]:
        fallback_model_id = None
        if self._snapshot and self._snapshot.fallback_model_id is not None:
            fallback_model_id = str(self._snapshot.fallback_model_id)
        if self.safe_mode_policy == "fallback_only" and fallback_model_id is not None:
            return fallback_model_id, "fallback_only"
        return None, "no_trade"

    def _resolve_no_candidate_state(
        self,
        state: RouterStateSnapshot,
        *,
        timestamp: Any,
    ) -> tuple[RouterStateSnapshot, None, str, str]:
        metadata = dict(state.metadata or {})
        decision_count = int(metadata.get("decision_count", 0)) + 1
        last_switch_index = metadata.get("last_switch_decision_index")
        if last_switch_index is not None:
            last_switch_index = int(last_switch_index)

        next_metadata = dict(metadata)
        next_metadata["decision_count"] = decision_count
        next_metadata["last_switch_decision_index"] = last_switch_index
        next_metadata["raw_best_model_id"] = None
        next_metadata["blocked_switch_reason"] = None
        next_state = RouterStateSnapshot(
            active_model_id=None,
            last_switch_at=state.last_switch_at,
            cooldown_active=False,
            pending_challenger_id=None,
            pending_challenger_streak=0,
            metadata=next_metadata,
        )
        return next_state, None, "no_eligible_candidates", "no_trade"

    def _build_decision_metrics(
        self,
        state: RouterStateSnapshot,
        *,
        weights: Mapping[str, float] | None,
        allocation_control_reason: str | None,
    ) -> tuple[dict[str, float], list[str], float, float, float, str | None]:
        normalized_weights = _normalize_weight_map(weights)
        executed_candidate_ids = [str(model_id) for model_id in normalized_weights.keys()]
        l1_change, turnover = _compute_weight_change(_previous_executed_weights(state), normalized_weights)
        effective_count = _effective_model_count(normalized_weights)
        return normalized_weights, executed_candidate_ids, l1_change, turnover, effective_count, allocation_control_reason

    def _resolve_safe_mode_state(
        self,
        state: RouterStateSnapshot,
        *,
        timestamp: Any,
        availability_state: str,
    ) -> tuple[RouterStateSnapshot, str | None, str, str]:
        metadata = dict(state.metadata or {})
        decision_count = int(metadata.get("decision_count", 0)) + 1
        last_switch_index = metadata.get("last_switch_decision_index")
        if last_switch_index is not None:
            last_switch_index = int(last_switch_index)

        current_model_id = None if state.active_model_id is None else str(state.active_model_id)
        target_model_id, safe_mode_action = self._resolve_safe_mode_target()
        switched = bool(target_model_id != current_model_id)

        next_metadata = dict(metadata)
        next_metadata["decision_count"] = decision_count
        next_metadata["last_switch_decision_index"] = decision_count if switched else last_switch_index
        next_metadata["raw_best_model_id"] = target_model_id
        next_metadata["blocked_switch_reason"] = None
        next_metadata["regime_availability_state"] = str(availability_state)
        next_metadata["safe_mode_policy"] = self.safe_mode_policy
        next_metadata["safe_mode_action"] = safe_mode_action

        next_state = RouterStateSnapshot(
            active_model_id=target_model_id,
            last_switch_at=(timestamp if switched else state.last_switch_at),
            cooldown_active=False,
            pending_challenger_id=None,
            pending_challenger_streak=0,
            metadata=next_metadata,
        )
        return next_state, target_model_id, f"{availability_state}_safe_mode", safe_mode_action

    def _resolve_routing_state(
        self,
        state: RouterStateSnapshot,
        *,
        candidate_scores: Mapping[str, float],
        timestamp: Any,
    ) -> tuple[RouterStateSnapshot, str, bool, bool, str | None, str | None]:
        metadata = dict(state.metadata or {})
        decision_count = int(metadata.get("decision_count", 0)) + 1
        last_switch_index = metadata.get("last_switch_decision_index")
        if last_switch_index is not None:
            last_switch_index = int(last_switch_index)
        current_model_id = None if state.active_model_id is None else str(state.active_model_id)
        raw_best_model_id = None
        if candidate_scores:
            raw_best_model_id = max(candidate_scores.items(), key=lambda item: (float(item[1]), item[0]))[0]
        raw_best_score = float(candidate_scores.get(raw_best_model_id, float("-inf"))) if raw_best_model_id is not None else float("-inf")
        current_score = float(candidate_scores.get(current_model_id, float("-inf"))) if current_model_id is not None else float("-inf")
        cooldown_active = bool(
            self.cooldown_bars > 0
            and current_model_id is not None
            and current_model_id in candidate_scores
            and last_switch_index is not None
            and (decision_count - int(last_switch_index)) <= int(self.cooldown_bars)
        )

        pending_challenger_id = state.pending_challenger_id
        pending_challenger_streak = int(state.pending_challenger_streak or 0)
        selected_model_id = current_model_id
        route_reason = "no_candidates"
        hysteresis_applied = False
        blocked_switch_reason = None
        switched = False

        if raw_best_model_id is None:
            selected_model_id = current_model_id
        elif current_model_id is None or current_model_id not in candidate_scores:
            selected_model_id = raw_best_model_id
            route_reason = "initial_selection"
            switched = bool(selected_model_id != current_model_id)
            pending_challenger_id = None
            pending_challenger_streak = 0
        elif raw_best_model_id == current_model_id:
            selected_model_id = current_model_id
            route_reason = "highest_score"
            pending_challenger_id = None
            pending_challenger_streak = 0
        else:
            score_gap = float(raw_best_score) - float(current_score)
            if cooldown_active:
                selected_model_id = current_model_id
                route_reason = "cooldown_hold"
                blocked_switch_reason = "cooldown_active"
                pending_challenger_id = raw_best_model_id
                pending_challenger_streak = 1 if pending_challenger_id != state.pending_challenger_id else pending_challenger_streak + 1
            elif score_gap <= float(self.hysteresis_margin):
                selected_model_id = current_model_id
                route_reason = "hysteresis_hold"
                blocked_switch_reason = "hysteresis_margin_not_met"
                hysteresis_applied = True
                pending_challenger_id = raw_best_model_id
                pending_challenger_streak = 1 if pending_challenger_id != state.pending_challenger_id else pending_challenger_streak + 1
            else:
                if state.pending_challenger_id == raw_best_model_id:
                    pending_challenger_streak = int(state.pending_challenger_streak or 0) + 1
                else:
                    pending_challenger_id = raw_best_model_id
                    pending_challenger_streak = 1

                if pending_challenger_streak < int(self.min_persistence_bars):
                    selected_model_id = current_model_id
                    route_reason = "persistence_hold"
                    blocked_switch_reason = "persistence_requirement_not_met"
                else:
                    selected_model_id = raw_best_model_id
                    route_reason = "highest_score"
                    switched = True
                    pending_challenger_id = None
                    pending_challenger_streak = 0

        next_metadata = dict(metadata)
        next_metadata["decision_count"] = decision_count
        next_metadata["last_switch_decision_index"] = decision_count if switched else last_switch_index
        next_metadata["raw_best_model_id"] = raw_best_model_id
        next_metadata["blocked_switch_reason"] = blocked_switch_reason

        next_state = RouterStateSnapshot(
            active_model_id=selected_model_id,
            last_switch_at=(timestamp if switched else state.last_switch_at),
            cooldown_active=cooldown_active,
            pending_challenger_id=pending_challenger_id,
            pending_challenger_streak=pending_challenger_streak,
            metadata=next_metadata,
        )
        return next_state, selected_model_id, route_reason, hysteresis_applied, cooldown_active, blocked_switch_reason


class HardSwitchRouter(_BaseSpecialistRouter):
    @property
    def router_type(self) -> str:
        return "hard_switch"

    def select(self, state: Any, regime_state: Any, specialist_health: Any = None, timestamp: Any = None):
        normalized_state = _normalize_state(state)
        resolved_timestamp = timestamp or normalized_state.last_switch_at or "now"
        resolved_regime_state = _normalize_regime_state(regime_state, timestamp=resolved_timestamp)
        availability_state = _classify_regime_availability(resolved_regime_state)
        if availability_state != "known":
            next_state, selected_model_id, route_reason, safe_mode_action = self._resolve_safe_mode_state(
                normalized_state,
                timestamp=resolved_timestamp,
                availability_state=availability_state,
            )
            weights, executed_candidate_ids, l1_change, turnover, effective_count, allocation_control_reason = self._build_decision_metrics(
                normalized_state,
                weights=({selected_model_id: 1.0} if selected_model_id is not None else {}),
                allocation_control_reason="safe_mode_selection_only",
            )
            next_state = _attach_executed_weights(next_state, weights)
            decision = RoutingDecisionContract(
                as_of=resolved_regime_state.as_of,
                available_at=resolved_regime_state.available_at,
                selected_model_id=selected_model_id,
                weights=weights,
                executed_candidate_ids=executed_candidate_ids,
                executed_weight_l1_change=l1_change,
                executed_weight_turnover=turnover,
                effective_model_count=effective_count,
                allocation_control_reason=allocation_control_reason,
                regime_label=(None if resolved_regime_state.label is None else str(resolved_regime_state.label)),
                regime_confidence=resolved_regime_state.confidence,
                route_reason=route_reason,
                hysteresis_applied=False,
                cooldown_active=False,
                candidate_scores=({selected_model_id: 1.0} if selected_model_id is not None else {}),
                components=[],
                metadata={
                    "router_type": self.router_type,
                    "candidate_components": {},
                    "blocked_switch_reason": None,
                    "regime_availability_state": availability_state,
                    "safe_mode_policy": self.safe_mode_policy,
                    "safe_mode_active": True,
                    "safe_mode_action": safe_mode_action,
                },
            )
            return next_state, decision
        health_map = _normalize_specialist_health_map(self._snapshot, specialist_health=specialist_health)
        candidate_eligibility = self._build_candidate_eligibility(
            regime_state=resolved_regime_state,
            health_map=health_map,
        )
        candidate_scores = {}
        candidate_components = {}
        for model_id in self._candidate_model_ids():
            if not bool((candidate_eligibility.get(str(model_id)) or {}).get("eligible", False)):
                continue
            score, components = self._score_model(
                model_id,
                regime_state=resolved_regime_state,
                health_map=health_map,
            )
            candidate_scores[str(model_id)] = float(score)
            candidate_components[str(model_id)] = components

        if not candidate_scores:
            next_state, selected_model_id, route_reason, safe_mode_action = self._resolve_no_candidate_state(
                normalized_state,
                timestamp=resolved_timestamp,
            )
            weights, executed_candidate_ids, l1_change, turnover, effective_count, allocation_control_reason = self._build_decision_metrics(
                normalized_state,
                weights={},
                allocation_control_reason="no_trade",
            )
            next_state = _attach_executed_weights(next_state, weights)
            decision = RoutingDecisionContract(
                as_of=resolved_regime_state.as_of,
                available_at=resolved_regime_state.available_at,
                selected_model_id=selected_model_id,
                weights=weights,
                executed_candidate_ids=executed_candidate_ids,
                executed_weight_l1_change=l1_change,
                executed_weight_turnover=turnover,
                effective_model_count=effective_count,
                allocation_control_reason=allocation_control_reason,
                regime_label=(None if resolved_regime_state.label is None else str(resolved_regime_state.label)),
                regime_confidence=resolved_regime_state.confidence,
                route_reason=route_reason,
                hysteresis_applied=False,
                cooldown_active=False,
                candidate_scores={},
                components=[],
                metadata={
                    "router_type": self.router_type,
                    "candidate_components": {},
                    "candidate_eligibility": candidate_eligibility,
                    "blocked_switch_reason": None,
                    "regime_availability_state": availability_state,
                    "safe_mode_policy": self.safe_mode_policy,
                    "safe_mode_active": True,
                    "safe_mode_action": safe_mode_action,
                },
            )
            return next_state, decision

        next_state, selected_model_id, route_reason, hysteresis_applied, cooldown_active, blocked_switch_reason = self._resolve_routing_state(
            normalized_state,
            candidate_scores=candidate_scores,
            timestamp=resolved_timestamp,
        )

        weights, executed_candidate_ids, l1_change, turnover, effective_count, allocation_control_reason = self._build_decision_metrics(
            normalized_state,
            weights=({selected_model_id: 1.0} if selected_model_id is not None else {}),
            allocation_control_reason="selection_only",
        )
        next_state = _attach_executed_weights(next_state, weights)

        decision = RoutingDecisionContract(
            as_of=resolved_regime_state.as_of,
            available_at=resolved_regime_state.available_at,
            selected_model_id=selected_model_id,
            weights=weights,
            executed_candidate_ids=executed_candidate_ids,
            executed_weight_l1_change=l1_change,
            executed_weight_turnover=turnover,
            effective_model_count=effective_count,
            allocation_control_reason=allocation_control_reason,
            regime_label=(None if resolved_regime_state.label is None else str(resolved_regime_state.label)),
            regime_confidence=resolved_regime_state.confidence,
            route_reason=route_reason,
            hysteresis_applied=hysteresis_applied,
            cooldown_active=cooldown_active,
            candidate_scores=candidate_scores,
            components=list(candidate_components.get(selected_model_id) or []),
            metadata={
                "router_type": self.router_type,
                "candidate_components": {
                    model_id: [component.to_dict() for component in components]
                    for model_id, components in candidate_components.items()
                },
                "candidate_eligibility": candidate_eligibility,
                "blocked_switch_reason": blocked_switch_reason,
                "regime_availability_state": availability_state,
                "safe_mode_policy": self.safe_mode_policy,
                "safe_mode_active": False,
                "safe_mode_action": None,
            },
        )
        return next_state, decision


class WeightedRouter(_BaseSpecialistRouter):
    def __init__(self, *, allocation_temperature: float = 1.0, allocation_mode: str = "mixture_allocation", **kwargs):
        super().__init__(**kwargs)
        self.allocation_temperature = float(max(0.01, allocation_temperature))
        self.allocation_mode = _normalize_allocation_mode(allocation_mode)

    @property
    def router_type(self) -> str:
        return "confidence_weighted"

    def manifest(self) -> RouterManifest:
        manifest = super().manifest()
        payload = manifest.to_dict()
        metadata = dict(payload.get("metadata") or {})
        metadata["allocation_temperature"] = self.allocation_temperature
        metadata["allocation_mode"] = self.allocation_mode
        payload["metadata"] = metadata
        return RouterManifest.from_dict(payload)

    def _resolve_executed_candidate_scores(
        self,
        *,
        candidate_scores: Mapping[str, float],
        selected_model_id: str | None,
        blocked_switch_reason: str | None,
    ) -> tuple[dict[str, float], str]:
        if selected_model_id is None:
            return {}, "no_trade"
        if self.allocation_mode == "selection_only":
            return ({str(selected_model_id): float(candidate_scores.get(str(selected_model_id), 0.0))}, "selection_only")
        if blocked_switch_reason is not None:
            return ({str(selected_model_id): float(candidate_scores.get(str(selected_model_id), 0.0))}, "blocked_selection_only")
        return ({str(model_id): float(score) for model_id, score in dict(candidate_scores or {}).items()}, "mixture_allocation")

    def select(self, state: Any, regime_state: Any, specialist_health: Any = None, timestamp: Any = None):
        normalized_state = _normalize_state(state)
        resolved_timestamp = timestamp or normalized_state.last_switch_at or "now"
        resolved_regime_state = _normalize_regime_state(regime_state, timestamp=resolved_timestamp)
        availability_state = _classify_regime_availability(resolved_regime_state)
        if availability_state != "known":
            next_state, selected_model_id, route_reason, safe_mode_action = self._resolve_safe_mode_state(
                normalized_state,
                timestamp=resolved_timestamp,
                availability_state=availability_state,
            )
            weights, executed_candidate_ids, l1_change, turnover, effective_count, allocation_control_reason = self._build_decision_metrics(
                normalized_state,
                weights=({selected_model_id: 1.0} if selected_model_id is not None else {}),
                allocation_control_reason="safe_mode_selection_only",
            )
            next_state = _attach_executed_weights(next_state, weights)
            decision = RoutingDecisionContract(
                as_of=resolved_regime_state.as_of,
                available_at=resolved_regime_state.available_at,
                selected_model_id=selected_model_id,
                weights=weights,
                executed_candidate_ids=executed_candidate_ids,
                executed_weight_l1_change=l1_change,
                executed_weight_turnover=turnover,
                effective_model_count=effective_count,
                allocation_control_reason=allocation_control_reason,
                regime_label=(None if resolved_regime_state.label is None else str(resolved_regime_state.label)),
                regime_confidence=resolved_regime_state.confidence,
                route_reason=route_reason,
                hysteresis_applied=False,
                cooldown_active=False,
                candidate_scores=({selected_model_id: 1.0} if selected_model_id is not None else {}),
                components=[],
                metadata={
                    "router_type": self.router_type,
                    "candidate_components": {},
                    "blocked_switch_reason": None,
                    "allocation_temperature": self.allocation_temperature,
                    "regime_availability_state": availability_state,
                    "safe_mode_policy": self.safe_mode_policy,
                    "safe_mode_active": True,
                    "safe_mode_action": safe_mode_action,
                },
            )
            return next_state, decision
        health_map = _normalize_specialist_health_map(self._snapshot, specialist_health=specialist_health)
        candidate_eligibility = self._build_candidate_eligibility(
            regime_state=resolved_regime_state,
            health_map=health_map,
        )
        candidate_scores = {}
        candidate_components = {}
        for model_id in self._candidate_model_ids():
            if not bool((candidate_eligibility.get(str(model_id)) or {}).get("eligible", False)):
                continue
            score, components = self._score_model(
                model_id,
                regime_state=resolved_regime_state,
                health_map=health_map,
            )
            candidate_scores[str(model_id)] = float(score)
            candidate_components[str(model_id)] = components

        if not candidate_scores:
            next_state, selected_model_id, route_reason, safe_mode_action = self._resolve_no_candidate_state(
                normalized_state,
                timestamp=resolved_timestamp,
            )
            weights, executed_candidate_ids, l1_change, turnover, effective_count, allocation_control_reason = self._build_decision_metrics(
                normalized_state,
                weights={},
                allocation_control_reason="no_trade",
            )
            next_state = _attach_executed_weights(next_state, weights)
            decision = RoutingDecisionContract(
                as_of=resolved_regime_state.as_of,
                available_at=resolved_regime_state.available_at,
                selected_model_id=selected_model_id,
                weights=weights,
                executed_candidate_ids=executed_candidate_ids,
                executed_weight_l1_change=l1_change,
                executed_weight_turnover=turnover,
                effective_model_count=effective_count,
                allocation_control_reason=allocation_control_reason,
                regime_label=(None if resolved_regime_state.label is None else str(resolved_regime_state.label)),
                regime_confidence=resolved_regime_state.confidence,
                route_reason=route_reason,
                hysteresis_applied=False,
                cooldown_active=False,
                candidate_scores={},
                components=[],
                metadata={
                    "router_type": self.router_type,
                    "candidate_components": {},
                    "candidate_eligibility": candidate_eligibility,
                    "blocked_switch_reason": None,
                    "allocation_temperature": self.allocation_temperature,
                    "regime_availability_state": availability_state,
                    "safe_mode_policy": self.safe_mode_policy,
                    "safe_mode_active": True,
                    "safe_mode_action": safe_mode_action,
                },
            )
            return next_state, decision

        next_state, selected_model_id, route_reason, hysteresis_applied, cooldown_active, blocked_switch_reason = self._resolve_routing_state(
            normalized_state,
            candidate_scores=candidate_scores,
            timestamp=resolved_timestamp,
        )

        executed_candidate_scores, allocation_control_reason = self._resolve_executed_candidate_scores(
            candidate_scores=candidate_scores,
            selected_model_id=selected_model_id,
            blocked_switch_reason=blocked_switch_reason,
        )

        weights = {}
        if executed_candidate_scores:
            ordered = list(executed_candidate_scores.items())
            maximum_score = max(score for _, score in ordered)
            exp_scores = {
                model_id: math.exp((float(score) - float(maximum_score)) / self.allocation_temperature)
                for model_id, score in ordered
            }
            total = float(sum(exp_scores.values()))
            if total > 0.0:
                weights = {model_id: float(value) / total for model_id, value in exp_scores.items()}

        weights, executed_candidate_ids, l1_change, turnover, effective_count, allocation_control_reason = self._build_decision_metrics(
            normalized_state,
            weights=weights,
            allocation_control_reason=allocation_control_reason,
        )
        next_state = _attach_executed_weights(next_state, weights)

        decision = RoutingDecisionContract(
            as_of=resolved_regime_state.as_of,
            available_at=resolved_regime_state.available_at,
            selected_model_id=selected_model_id,
            weights=weights,
            executed_candidate_ids=executed_candidate_ids,
            executed_weight_l1_change=l1_change,
            executed_weight_turnover=turnover,
            effective_model_count=effective_count,
            allocation_control_reason=allocation_control_reason,
            regime_label=(None if resolved_regime_state.label is None else str(resolved_regime_state.label)),
            regime_confidence=resolved_regime_state.confidence,
            route_reason=("weighted_allocation" if route_reason == "highest_score" else route_reason),
            hysteresis_applied=hysteresis_applied,
            cooldown_active=cooldown_active,
            candidate_scores=candidate_scores,
            components=list(candidate_components.get(selected_model_id) or []),
            metadata={
                "router_type": self.router_type,
                "candidate_components": {
                    model_id: [component.to_dict() for component in components]
                    for model_id, components in candidate_components.items()
                },
                "candidate_eligibility": candidate_eligibility,
                "blocked_switch_reason": blocked_switch_reason,
                "allocation_temperature": self.allocation_temperature,
                "regime_availability_state": availability_state,
                "safe_mode_policy": self.safe_mode_policy,
                "safe_mode_active": False,
                "safe_mode_action": None,
            },
        )
        return next_state, decision


def build_router(config: Mapping[str, Any] | None = None):
    config = dict(config or {})
    router_type = str(config.get("type", "hard_switch")).strip().lower()
    common_kwargs = {
        "policy_name": config.get("policy_name"),
        "hysteresis_margin": _coerce_float(config.get("hysteresis_margin"), 0.0),
        "min_persistence_bars": int(config.get("min_persistence_bars", 1) or 1),
        "cooldown_bars": int(config.get("cooldown_bars", 0) or 0),
        "fallback_bias": _coerce_float(config.get("fallback_bias"), 0.45),
        "stability_weight": _coerce_float(config.get("stability_weight"), 0.25),
        "decay_weight": _coerce_float(config.get("decay_weight"), 0.15),
        "failure_flag_penalty": _coerce_float(config.get("failure_flag_penalty"), 0.2),
        "missing_health_score": _coerce_float(config.get("missing_health_score"), 0.5),
        "safe_mode_policy": config.get("safe_mode_policy", "fallback_only"),
    }
    if router_type in {"hard_switch", "score_router"}:
        return HardSwitchRouter(**common_kwargs)
    if router_type in {"confidence_weighted", "weighted", "weighted_router"}:
        return WeightedRouter(
            allocation_temperature=_coerce_float(config.get("allocation_temperature"), 1.0),
            allocation_mode=config.get("allocation_mode", "mixture_allocation"),
            **common_kwargs,
        )
    raise ValueError(f"Unknown router type {config.get('type')!r}")


__all__ = [
    "HardSwitchRouter",
    "WeightedRouter",
    "build_router",
]