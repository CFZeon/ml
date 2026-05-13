"""Phase 0 router contracts and protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_value(item) for item in value]
    if value is None:
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return value


def _coerce_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return {str(key): _serialize_value(item) for key, item in dict(value or {}).items()}


@dataclass(frozen=True)
class RoutingScoreComponent:
    name: str
    value: float
    weight: float | None = None
    penalized: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "phase0.routing_score_component.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": str(self.name),
            "value": float(self.value),
            "weight": None if self.weight is None else float(self.weight),
            "penalized": bool(self.penalized),
            "metadata": _coerce_mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RoutingScoreComponent":
        data = dict(payload or {})
        weight = data.get("weight")
        return cls(
            schema_version=str(data.get("schema_version", "phase0.routing_score_component.v1")),
            name=str(data.get("name", "component")),
            value=float(data.get("value", 0.0)),
            weight=None if weight is None else float(weight),
            penalized=bool(data.get("penalized", False)),
            metadata=_coerce_mapping(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class RoutingDecisionContract:
    as_of: Any
    available_at: Any
    selected_model_id: str | None = None
    weights: dict[str, float] = field(default_factory=dict)
    executed_candidate_ids: list[str] = field(default_factory=list)
    executed_weight_l1_change: float = 0.0
    executed_weight_turnover: float = 0.0
    effective_model_count: float = 0.0
    allocation_control_reason: str | None = None
    regime_label: str | None = None
    regime_confidence: float | None = None
    route_reason: str = "uninitialized"
    hysteresis_applied: bool = False
    cooldown_active: bool = False
    candidate_scores: dict[str, float] = field(default_factory=dict)
    components: list[RoutingScoreComponent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "phase0.routing_decision.v2"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "as_of": _serialize_value(self.as_of),
            "available_at": _serialize_value(self.available_at),
            "selected_model_id": None if self.selected_model_id is None else str(self.selected_model_id),
            "weights": {str(key): float(value) for key, value in dict(self.weights).items()},
            "executed_candidate_ids": [str(item) for item in list(self.executed_candidate_ids or [])],
            "executed_weight_l1_change": float(self.executed_weight_l1_change),
            "executed_weight_turnover": float(self.executed_weight_turnover),
            "effective_model_count": float(self.effective_model_count),
            "allocation_control_reason": (
                None if self.allocation_control_reason is None else str(self.allocation_control_reason)
            ),
            "regime_label": None if self.regime_label is None else str(self.regime_label),
            "regime_confidence": None if self.regime_confidence is None else float(self.regime_confidence),
            "route_reason": str(self.route_reason),
            "hysteresis_applied": bool(self.hysteresis_applied),
            "cooldown_active": bool(self.cooldown_active),
            "candidate_scores": {str(key): float(value) for key, value in dict(self.candidate_scores).items()},
            "components": [item.to_dict() for item in list(self.components or [])],
            "metadata": _coerce_mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RoutingDecisionContract":
        data = dict(payload or {})
        confidence = data.get("regime_confidence")
        return cls(
            schema_version=str(data.get("schema_version", "phase0.routing_decision.v2")),
            as_of=data.get("as_of"),
            available_at=data.get("available_at"),
            selected_model_id=(None if data.get("selected_model_id") is None else str(data.get("selected_model_id"))),
            weights={str(key): float(value) for key, value in dict(data.get("weights") or {}).items()},
            executed_candidate_ids=[str(item) for item in list(data.get("executed_candidate_ids") or [])],
            executed_weight_l1_change=float(data.get("executed_weight_l1_change", 0.0) or 0.0),
            executed_weight_turnover=float(data.get("executed_weight_turnover", 0.0) or 0.0),
            effective_model_count=float(data.get("effective_model_count", 0.0) or 0.0),
            allocation_control_reason=(
                None if data.get("allocation_control_reason") is None else str(data.get("allocation_control_reason"))
            ),
            regime_label=(None if data.get("regime_label") is None else str(data.get("regime_label"))),
            regime_confidence=None if confidence is None else float(confidence),
            route_reason=str(data.get("route_reason", "uninitialized")),
            hysteresis_applied=bool(data.get("hysteresis_applied", False)),
            cooldown_active=bool(data.get("cooldown_active", False)),
            candidate_scores={str(key): float(value) for key, value in dict(data.get("candidate_scores") or {}).items()},
            components=[RoutingScoreComponent.from_dict(item) for item in list(data.get("components") or [])],
            metadata=_coerce_mapping(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class RouterStateSnapshot:
    active_model_id: str | None = None
    last_switch_at: Any = None
    cooldown_active: bool = False
    pending_challenger_id: str | None = None
    pending_challenger_streak: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "phase0.router_state_snapshot.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "active_model_id": None if self.active_model_id is None else str(self.active_model_id),
            "last_switch_at": _serialize_value(self.last_switch_at),
            "cooldown_active": bool(self.cooldown_active),
            "pending_challenger_id": None if self.pending_challenger_id is None else str(self.pending_challenger_id),
            "pending_challenger_streak": int(self.pending_challenger_streak),
            "metadata": _coerce_mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RouterStateSnapshot":
        data = dict(payload or {})
        return cls(
            schema_version=str(data.get("schema_version", "phase0.router_state_snapshot.v1")),
            active_model_id=(None if data.get("active_model_id") is None else str(data.get("active_model_id"))),
            last_switch_at=data.get("last_switch_at"),
            cooldown_active=bool(data.get("cooldown_active", False)),
            pending_challenger_id=(None if data.get("pending_challenger_id") is None else str(data.get("pending_challenger_id"))),
            pending_challenger_streak=int(data.get("pending_challenger_streak", 0)),
            metadata=_coerce_mapping(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class RouterManifest:
    router_type: str
    score_component_names: list[str] = field(default_factory=list)
    policy_name: str | None = None
    hysteresis_margin: float | None = None
    min_persistence_bars: int | None = None
    cooldown_bars: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "phase0.router_manifest.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "router_type": str(self.router_type),
            "score_component_names": [str(item) for item in list(self.score_component_names or [])],
            "policy_name": None if self.policy_name is None else str(self.policy_name),
            "hysteresis_margin": None if self.hysteresis_margin is None else float(self.hysteresis_margin),
            "min_persistence_bars": None if self.min_persistence_bars is None else int(self.min_persistence_bars),
            "cooldown_bars": None if self.cooldown_bars is None else int(self.cooldown_bars),
            "metadata": _coerce_mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RouterManifest":
        data = dict(payload or {})
        hysteresis_margin = data.get("hysteresis_margin")
        min_persistence_bars = data.get("min_persistence_bars")
        cooldown_bars = data.get("cooldown_bars")
        return cls(
            schema_version=str(data.get("schema_version", "phase0.router_manifest.v1")),
            router_type=str(data.get("router_type", "router")),
            score_component_names=[str(item) for item in list(data.get("score_component_names") or [])],
            policy_name=(None if data.get("policy_name") is None else str(data.get("policy_name"))),
            hysteresis_margin=None if hysteresis_margin is None else float(hysteresis_margin),
            min_persistence_bars=None if min_persistence_bars is None else int(min_persistence_bars),
            cooldown_bars=None if cooldown_bars is None else int(cooldown_bars),
            metadata=_coerce_mapping(data.get("metadata") or {}),
        )


class BaseRouter(Protocol):
    def initialize(self, specialists: Any) -> Any: ...

    def select(self, state: Any, regime_state: Any, specialist_health: Any, timestamp: Any) -> tuple[Any, RoutingDecisionContract]: ...

    def manifest(self) -> RouterManifest: ...


__all__ = [
    "BaseRouter",
    "RouterManifest",
    "RouterStateSnapshot",
    "RoutingDecisionContract",
    "RoutingScoreComponent",
]