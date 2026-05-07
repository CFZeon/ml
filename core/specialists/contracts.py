"""Phase 0 specialist contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


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


def _coerce_string_list(value: Sequence[Any] | None) -> list[str]:
    return [str(item) for item in list(value or [])]


class SpecialistLifecycleState(str, Enum):
    CANDIDATE = "candidate"
    CERTIFIED = "certified"
    ACTIVE = "active"
    SHADOW_CHALLENGER = "shadow_challenger"
    DEGRADED = "degraded"
    RETIRED = "retired"


@dataclass(frozen=True)
class SpecialistSpec:
    model_id: str
    symbol: str
    timeframe: str
    compatible_regimes: list[str] = field(default_factory=list)
    incompatible_regimes: list[str] = field(default_factory=list)
    estimator_family: str = "unknown"
    feature_policy_id: str | None = None
    calibration_id: str | None = None
    training_window: dict[str, Any] = field(default_factory=dict)
    detector_bundle_id: str | None = None
    router_compatibility_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "phase0.specialist_spec.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": str(self.model_id),
            "symbol": str(self.symbol),
            "timeframe": str(self.timeframe),
            "compatible_regimes": _coerce_string_list(self.compatible_regimes),
            "incompatible_regimes": _coerce_string_list(self.incompatible_regimes),
            "estimator_family": str(self.estimator_family),
            "feature_policy_id": None if self.feature_policy_id is None else str(self.feature_policy_id),
            "calibration_id": None if self.calibration_id is None else str(self.calibration_id),
            "training_window": _coerce_mapping(self.training_window),
            "detector_bundle_id": None if self.detector_bundle_id is None else str(self.detector_bundle_id),
            "router_compatibility_version": (
                None if self.router_compatibility_version is None else str(self.router_compatibility_version)
            ),
            "metadata": _coerce_mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SpecialistSpec":
        data = dict(payload or {})
        return cls(
            schema_version=str(data.get("schema_version", "phase0.specialist_spec.v1")),
            model_id=str(data.get("model_id", "specialist")),
            symbol=str(data.get("symbol", "unknown")),
            timeframe=str(data.get("timeframe", "unknown")),
            compatible_regimes=_coerce_string_list(data.get("compatible_regimes")),
            incompatible_regimes=_coerce_string_list(data.get("incompatible_regimes")),
            estimator_family=str(data.get("estimator_family", "unknown")),
            feature_policy_id=(None if data.get("feature_policy_id") is None else str(data.get("feature_policy_id"))),
            calibration_id=(None if data.get("calibration_id") is None else str(data.get("calibration_id"))),
            training_window=_coerce_mapping(data.get("training_window") or {}),
            detector_bundle_id=(None if data.get("detector_bundle_id") is None else str(data.get("detector_bundle_id"))),
            router_compatibility_version=(
                None
                if data.get("router_compatibility_version") is None
                else str(data.get("router_compatibility_version"))
            ),
            metadata=_coerce_mapping(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class SpecialistArtifactRef:
    model_id: str
    artifact_uri: str
    meta_artifact_uri: str | None = None
    artifact_type: str = "model_bundle"
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "phase0.specialist_artifact_ref.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": str(self.model_id),
            "artifact_uri": str(self.artifact_uri),
            "meta_artifact_uri": None if self.meta_artifact_uri is None else str(self.meta_artifact_uri),
            "artifact_type": str(self.artifact_type),
            "metadata": _coerce_mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SpecialistArtifactRef":
        data = dict(payload or {})
        return cls(
            schema_version=str(data.get("schema_version", "phase0.specialist_artifact_ref.v1")),
            model_id=str(data.get("model_id", "specialist")),
            artifact_uri=str(data.get("artifact_uri", "")),
            meta_artifact_uri=(None if data.get("meta_artifact_uri") is None else str(data.get("meta_artifact_uri"))),
            artifact_type=str(data.get("artifact_type", "model_bundle")),
            metadata=_coerce_mapping(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class SpecialistHealthContract:
    model_id: str
    compatible_regimes: list[str] = field(default_factory=list)
    stability_score: float | None = None
    decay_score: float | None = None
    last_calibrated_at: Any = None
    failure_flags: list[str] = field(default_factory=list)
    fallback_only: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "phase0.specialist_health.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": str(self.model_id),
            "compatible_regimes": _coerce_string_list(self.compatible_regimes),
            "stability_score": None if self.stability_score is None else float(self.stability_score),
            "decay_score": None if self.decay_score is None else float(self.decay_score),
            "last_calibrated_at": _serialize_value(self.last_calibrated_at),
            "failure_flags": _coerce_string_list(self.failure_flags),
            "fallback_only": bool(self.fallback_only),
            "metadata": _coerce_mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SpecialistHealthContract":
        data = dict(payload or {})
        stability_score = data.get("stability_score")
        decay_score = data.get("decay_score")
        return cls(
            schema_version=str(data.get("schema_version", "phase0.specialist_health.v1")),
            model_id=str(data.get("model_id", "specialist")),
            compatible_regimes=_coerce_string_list(data.get("compatible_regimes")),
            stability_score=None if stability_score is None else float(stability_score),
            decay_score=None if decay_score is None else float(decay_score),
            last_calibrated_at=data.get("last_calibrated_at"),
            failure_flags=_coerce_string_list(data.get("failure_flags")),
            fallback_only=bool(data.get("fallback_only", False)),
            metadata=_coerce_mapping(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class SpecialistPerformanceSlice:
    model_id: str
    regime_label: str | None = None
    split_role: str = "training_slice"
    row_count: int = 0
    metric_summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "phase0.specialist_performance_slice.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": str(self.model_id),
            "regime_label": None if self.regime_label is None else str(self.regime_label),
            "split_role": str(self.split_role),
            "row_count": int(self.row_count),
            "metric_summary": _coerce_mapping(self.metric_summary),
            "metadata": _coerce_mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SpecialistPerformanceSlice":
        data = dict(payload or {})
        return cls(
            schema_version=str(data.get("schema_version", "phase0.specialist_performance_slice.v1")),
            model_id=str(data.get("model_id", "specialist")),
            regime_label=(None if data.get("regime_label") is None else str(data.get("regime_label"))),
            split_role=str(data.get("split_role", "training_slice")),
            row_count=int(data.get("row_count", 0)),
            metric_summary=_coerce_mapping(data.get("metric_summary") or {}),
            metadata=_coerce_mapping(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class SpecialistLibrarySnapshot:
    symbol: str
    timeframe: str
    fallback_model_id: str | None = None
    specialists: list[SpecialistSpec] = field(default_factory=list)
    health: list[SpecialistHealthContract] = field(default_factory=list)
    performance_slices: list[SpecialistPerformanceSlice] = field(default_factory=list)
    artifact_refs: list[SpecialistArtifactRef] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "phase0.specialist_library_snapshot.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "symbol": str(self.symbol),
            "timeframe": str(self.timeframe),
            "fallback_model_id": None if self.fallback_model_id is None else str(self.fallback_model_id),
            "specialists": [item.to_dict() for item in list(self.specialists or [])],
            "health": [item.to_dict() for item in list(self.health or [])],
            "performance_slices": [item.to_dict() for item in list(self.performance_slices or [])],
            "artifact_refs": [item.to_dict() for item in list(self.artifact_refs or [])],
            "metadata": _coerce_mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SpecialistLibrarySnapshot":
        data = dict(payload or {})
        return cls(
            schema_version=str(data.get("schema_version", "phase0.specialist_library_snapshot.v1")),
            symbol=str(data.get("symbol", "unknown")),
            timeframe=str(data.get("timeframe", "unknown")),
            fallback_model_id=(None if data.get("fallback_model_id") is None else str(data.get("fallback_model_id"))),
            specialists=[SpecialistSpec.from_dict(item) for item in list(data.get("specialists") or [])],
            health=[SpecialistHealthContract.from_dict(item) for item in list(data.get("health") or [])],
            performance_slices=[SpecialistPerformanceSlice.from_dict(item) for item in list(data.get("performance_slices") or [])],
            artifact_refs=[SpecialistArtifactRef.from_dict(item) for item in list(data.get("artifact_refs") or [])],
            metadata=_coerce_mapping(data.get("metadata") or {}),
        )


__all__ = [
    "SpecialistArtifactRef",
    "SpecialistHealthContract",
    "SpecialistLibrarySnapshot",
    "SpecialistLifecycleState",
    "SpecialistPerformanceSlice",
    "SpecialistSpec",
]