"""Phase 0 regime contracts and protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence


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


def _coerce_sequence(value: Sequence[Any] | None) -> list[Any]:
    return [_serialize_value(item) for item in list(value or [])]


def _coerce_timestamp(value: Any) -> Any:
    return _serialize_value(value)


@dataclass(frozen=True)
class RegimeObservationContract:
    as_of: Any
    available_at: Any
    values: dict[str, Any]
    source_map: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "phase0.regime_observation.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "as_of": _coerce_timestamp(self.as_of),
            "available_at": _coerce_timestamp(self.available_at),
            "values": _coerce_mapping(self.values),
            "source_map": {str(key): str(value) for key, value in dict(self.source_map).items()},
            "metadata": _coerce_mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegimeObservationContract":
        data = dict(payload or {})
        return cls(
            schema_version=str(data.get("schema_version", "phase0.regime_observation.v1")),
            as_of=data.get("as_of"),
            available_at=data.get("available_at"),
            values=_coerce_mapping(data.get("values") or {}),
            source_map={str(key): str(value) for key, value in dict(data.get("source_map") or {}).items()},
            metadata=_coerce_mapping(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class RegimeStateContract:
    as_of: Any
    available_at: Any
    label: Any = None
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None
    confidence_kind: str | None = None
    recognition_lag_bars: int | None = None
    source_available_at: Any = None
    availability_reason: str | None = None
    detector_outputs: dict[str, Any] = field(default_factory=dict)
    warm: bool = True
    transition_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "phase0.regime_state.v2"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "as_of": _coerce_timestamp(self.as_of),
            "available_at": _coerce_timestamp(self.available_at),
            "label": _serialize_value(self.label),
            "probabilities": {str(key): float(value) for key, value in dict(self.probabilities).items()},
            "confidence": None if self.confidence is None else float(self.confidence),
            "confidence_kind": None if self.confidence_kind is None else str(self.confidence_kind),
            "recognition_lag_bars": None if self.recognition_lag_bars is None else int(self.recognition_lag_bars),
            "source_available_at": _coerce_timestamp(self.source_available_at),
            "availability_reason": None if self.availability_reason is None else str(self.availability_reason),
            "detector_outputs": _coerce_mapping(self.detector_outputs),
            "warm": bool(self.warm),
            "transition_id": None if self.transition_id is None else str(self.transition_id),
            "metadata": _coerce_mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegimeStateContract":
        data = dict(payload or {})
        probabilities = {
            str(key): float(value)
            for key, value in dict(data.get("probabilities") or {}).items()
            if value is not None
        }
        confidence = data.get("confidence")
        return cls(
            schema_version=str(data.get("schema_version", "phase0.regime_state.v2")),
            as_of=data.get("as_of"),
            available_at=data.get("available_at"),
            label=data.get("label"),
            probabilities=probabilities,
            confidence=None if confidence is None else float(confidence),
            confidence_kind=(None if data.get("confidence_kind") is None else str(data.get("confidence_kind"))),
            recognition_lag_bars=(
                None if data.get("recognition_lag_bars") is None else int(data.get("recognition_lag_bars"))
            ),
            source_available_at=data.get("source_available_at"),
            availability_reason=(
                None if data.get("availability_reason") is None else str(data.get("availability_reason"))
            ),
            detector_outputs=_coerce_mapping(data.get("detector_outputs") or {}),
            warm=bool(data.get("warm", True)),
            transition_id=(None if data.get("transition_id") is None else str(data.get("transition_id"))),
            metadata=_coerce_mapping(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class RegimeTransitionContract:
    as_of: Any
    available_at: Any
    from_label: Any = None
    to_label: Any = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "phase0.regime_transition.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "as_of": _coerce_timestamp(self.as_of),
            "available_at": _coerce_timestamp(self.available_at),
            "from_label": _serialize_value(self.from_label),
            "to_label": _serialize_value(self.to_label),
            "confidence": None if self.confidence is None else float(self.confidence),
            "metadata": _coerce_mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegimeTransitionContract":
        data = dict(payload or {})
        confidence = data.get("confidence")
        return cls(
            schema_version=str(data.get("schema_version", "phase0.regime_transition.v1")),
            as_of=data.get("as_of"),
            available_at=data.get("available_at"),
            from_label=data.get("from_label"),
            to_label=data.get("to_label"),
            confidence=None if confidence is None else float(confidence),
            metadata=_coerce_mapping(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class RegimeDetectorManifest:
    detector_name: str
    detector_type: str
    params: dict[str, Any] = field(default_factory=dict)
    warmup_bars: int | None = None
    fit_window: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "phase0.regime_detector_manifest.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "detector_name": str(self.detector_name),
            "detector_type": str(self.detector_type),
            "params": _coerce_mapping(self.params),
            "warmup_bars": None if self.warmup_bars is None else int(self.warmup_bars),
            "fit_window": _coerce_mapping(self.fit_window),
            "metadata": _coerce_mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegimeDetectorManifest":
        data = dict(payload or {})
        warmup_bars = data.get("warmup_bars")
        return cls(
            schema_version=str(data.get("schema_version", "phase0.regime_detector_manifest.v1")),
            detector_name=str(data.get("detector_name", "detector")),
            detector_type=str(data.get("detector_type", "unknown")),
            params=_coerce_mapping(data.get("params") or {}),
            warmup_bars=None if warmup_bars is None else int(warmup_bars),
            fit_window=_coerce_mapping(data.get("fit_window") or {}),
            metadata=_coerce_mapping(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class RegimeTraceSummary:
    mode: str
    row_count: int
    available_rows: int
    transition_count: int
    evidence_class: str = "preview_only"
    observation_columns: list[str] = field(default_factory=list)
    state_columns: list[str] = field(default_factory=list)
    label_distribution: dict[str, int] = field(default_factory=dict)
    dominant_label: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "phase0.regime_trace_summary.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": str(self.mode),
            "evidence_class": str(self.evidence_class),
            "row_count": int(self.row_count),
            "available_rows": int(self.available_rows),
            "transition_count": int(self.transition_count),
            "observation_columns": [str(item) for item in list(self.observation_columns or [])],
            "state_columns": [str(item) for item in list(self.state_columns or [])],
            "label_distribution": {str(key): int(value) for key, value in dict(self.label_distribution).items()},
            "dominant_label": None if self.dominant_label is None else str(self.dominant_label),
            "provenance": _coerce_mapping(self.provenance),
            "metadata": _coerce_mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegimeTraceSummary":
        data = dict(payload or {})
        return cls(
            schema_version=str(data.get("schema_version", "phase0.regime_trace_summary.v1")),
            mode=str(data.get("mode", "preview")),
            evidence_class=str(data.get("evidence_class", "preview_only")),
            row_count=int(data.get("row_count", 0)),
            available_rows=int(data.get("available_rows", 0)),
            transition_count=int(data.get("transition_count", 0)),
            observation_columns=[str(item) for item in list(data.get("observation_columns") or [])],
            state_columns=[str(item) for item in list(data.get("state_columns") or [])],
            label_distribution={str(key): int(value) for key, value in dict(data.get("label_distribution") or {}).items()},
            dominant_label=(None if data.get("dominant_label") is None else str(data.get("dominant_label"))),
            provenance=_coerce_mapping(data.get("provenance") or {}),
            metadata=_coerce_mapping(data.get("metadata") or {}),
        )


class BaseRegimeDetector(Protocol):
    def fit(self, observations: Any) -> "BaseRegimeDetector": ...

    def initialize(self, observations: Any | None = None) -> Any: ...

    def update(self, state: Any, observation: RegimeObservationContract) -> tuple[Any, RegimeStateContract]: ...

    def manifest(self) -> RegimeDetectorManifest: ...


__all__ = [
    "BaseRegimeDetector",
    "RegimeDetectorManifest",
    "RegimeObservationContract",
    "RegimeStateContract",
    "RegimeTraceSummary",
    "RegimeTransitionContract",
]