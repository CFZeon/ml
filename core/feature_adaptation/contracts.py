"""Phase 2 feature-adaptation contracts and protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


def _serialize_value(value: Any) -> Any:
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


def _coerce_string_list(value: list[Any] | tuple[Any, ...] | set[Any] | None) -> list[str]:
    return [str(item) for item in list(value or [])]


@dataclass(frozen=True)
class FeaturePolicyContract:
    policy_id: str
    feature_columns: list[str] = field(default_factory=list)
    disabled_columns: list[str] = field(default_factory=list)
    generated_columns: list[str] = field(default_factory=list)
    regime_column: str = "regime"
    scaling_mode: str = "identity"
    fallback_mode: str = "identity"
    sparse_regimes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "phase2.feature_policy.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_id": str(self.policy_id),
            "feature_columns": _coerce_string_list(self.feature_columns),
            "disabled_columns": _coerce_string_list(self.disabled_columns),
            "generated_columns": _coerce_string_list(self.generated_columns),
            "regime_column": str(self.regime_column),
            "scaling_mode": str(self.scaling_mode),
            "fallback_mode": str(self.fallback_mode),
            "sparse_regimes": _coerce_string_list(self.sparse_regimes),
            "metadata": _coerce_mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeaturePolicyContract":
        data = dict(payload or {})
        return cls(
            schema_version=str(data.get("schema_version", "phase2.feature_policy.v1")),
            policy_id=str(data.get("policy_id", "identity")),
            feature_columns=_coerce_string_list(data.get("feature_columns")),
            disabled_columns=_coerce_string_list(data.get("disabled_columns")),
            generated_columns=_coerce_string_list(data.get("generated_columns")),
            regime_column=str(data.get("regime_column", "regime")),
            scaling_mode=str(data.get("scaling_mode", "identity")),
            fallback_mode=str(data.get("fallback_mode", "identity")),
            sparse_regimes=_coerce_string_list(data.get("sparse_regimes")),
            metadata=_coerce_mapping(data.get("metadata") or {}),
        )


class BaseFeatureAdapter(Protocol):
    def fit(
        self,
        X: Any,
        regime_frame: Any,
        feature_metadata: Mapping[str, Any] | None = None,
    ) -> "BaseFeatureAdapter": ...

    def transform(
        self,
        X: Any,
        regime_frame: Any,
    ) -> tuple[Any, FeaturePolicyContract]: ...

    def manifest(self) -> dict[str, Any]: ...


__all__ = ["BaseFeatureAdapter", "FeaturePolicyContract"]