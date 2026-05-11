"""Specialist-library runtime helpers."""

from __future__ import annotations

from typing import Any, Mapping

from .contracts import (
    SpecialistArtifactRef,
    SpecialistLibrarySnapshot,
    SpecialistLifecycleState,
    SpecialistSpec,
)


_ALLOWED_LIFECYCLE_TRANSITIONS = {
    SpecialistLifecycleState.CANDIDATE: {
        SpecialistLifecycleState.CERTIFIED,
        SpecialistLifecycleState.RETIRED,
    },
    SpecialistLifecycleState.CERTIFIED: {
        SpecialistLifecycleState.ACTIVE,
        SpecialistLifecycleState.SHADOW_CHALLENGER,
        SpecialistLifecycleState.RETIRED,
    },
    SpecialistLifecycleState.ACTIVE: {
        SpecialistLifecycleState.DEGRADED,
        SpecialistLifecycleState.RETIRED,
    },
    SpecialistLifecycleState.SHADOW_CHALLENGER: {
        SpecialistLifecycleState.ACTIVE,
        SpecialistLifecycleState.RETIRED,
    },
    SpecialistLifecycleState.DEGRADED: {
        SpecialistLifecycleState.ACTIVE,
        SpecialistLifecycleState.RETIRED,
    },
    SpecialistLifecycleState.RETIRED: set(),
}

def _serialize_metadata_value(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _serialize_metadata_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_metadata_value(item) for item in value]
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


def _coerce_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    return {str(key): _serialize_metadata_value(item) for key, item in dict(metadata or {}).items()}


def _normalize_lifecycle_state(
    value: Any,
    *,
    default: SpecialistLifecycleState = SpecialistLifecycleState.CANDIDATE,
) -> SpecialistLifecycleState:
    if isinstance(value, SpecialistLifecycleState):
        return value
    if value is None:
        return default
    normalized = str(value).strip().lower()
    for state in SpecialistLifecycleState:
        if state.value == normalized:
            return state
    raise ValueError(f"Unknown specialist lifecycle state {value!r}")


def _replace_snapshot(
    snapshot: SpecialistLibrarySnapshot,
    *,
    specialists: list[SpecialistSpec] | None = None,
    artifact_refs: list[SpecialistArtifactRef] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SpecialistLibrarySnapshot:
    payload = snapshot.to_dict()
    if specialists is not None:
        payload["specialists"] = [item.to_dict() for item in specialists]
    if artifact_refs is not None:
        payload["artifact_refs"] = [item.to_dict() for item in artifact_refs]
    if metadata is not None:
        payload["metadata"] = _coerce_metadata(metadata)
    return SpecialistLibrarySnapshot.from_dict(payload)


def _replace_spec_state(
    spec: SpecialistSpec,
    lifecycle_state: SpecialistLifecycleState,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> SpecialistSpec:
    payload = spec.to_dict()
    spec_metadata = dict(payload.get("metadata") or {})
    spec_metadata.update(_coerce_metadata(metadata))
    spec_metadata["lifecycle_state"] = lifecycle_state.value
    payload["metadata"] = spec_metadata
    return SpecialistSpec.from_dict(payload)


def normalize_specialist_library_snapshot(snapshot: Any) -> SpecialistLibrarySnapshot | None:
    if snapshot is None:
        return None
    if isinstance(snapshot, SpecialistLibrarySnapshot):
        return SpecialistLibrarySnapshot.from_dict(snapshot.to_dict())
    if isinstance(snapshot, Mapping):
        return SpecialistLibrarySnapshot.from_dict(snapshot)
    raise TypeError("specialist_library must be a SpecialistLibrarySnapshot or mapping")


def resolve_specialist_lifecycle_transition(current_state: Any, target_state: Any) -> SpecialistLifecycleState:
    current = _normalize_lifecycle_state(current_state)
    target = _normalize_lifecycle_state(target_state, default=current)
    if target == current:
        return target
    allowed = _ALLOWED_LIFECYCLE_TRANSITIONS.get(current, set())
    if target not in allowed:
        raise ValueError(
            f"Illegal specialist lifecycle transition {current.value!r} -> {target.value!r}"
        )
    return target


def build_specialist_selection_contract(snapshot: Any) -> dict[str, Any]:
    resolved = normalize_specialist_library_snapshot(snapshot)
    if resolved is None:
        return {}

    lifecycle_state_by_model_id = {}
    compatible_regimes = {}
    buckets = {
        SpecialistLifecycleState.CANDIDATE.value: [],
        SpecialistLifecycleState.CERTIFIED.value: [],
        SpecialistLifecycleState.ACTIVE.value: [],
        SpecialistLifecycleState.SHADOW_CHALLENGER.value: [],
        SpecialistLifecycleState.DEGRADED.value: [],
        SpecialistLifecycleState.RETIRED.value: [],
    }

    for spec in list(resolved.specialists or []):
        lifecycle_state = _normalize_lifecycle_state((spec.metadata or {}).get("lifecycle_state"))
        lifecycle_state_by_model_id[str(spec.model_id)] = lifecycle_state.value
        compatible_regimes[str(spec.model_id)] = [str(item) for item in list(spec.compatible_regimes or [])]
        buckets[lifecycle_state.value].append(str(spec.model_id))

    return {
        "fallback_model_id": resolved.fallback_model_id,
        "candidate_model_ids": buckets[SpecialistLifecycleState.CANDIDATE.value],
        "certified_model_ids": buckets[SpecialistLifecycleState.CERTIFIED.value],
        "active_model_ids": buckets[SpecialistLifecycleState.ACTIVE.value],
        "shadow_model_ids": buckets[SpecialistLifecycleState.SHADOW_CHALLENGER.value],
        "degraded_model_ids": buckets[SpecialistLifecycleState.DEGRADED.value],
        "retired_model_ids": buckets[SpecialistLifecycleState.RETIRED.value],
        "compatible_regimes": compatible_regimes,
        "lifecycle_state_by_model_id": lifecycle_state_by_model_id,
    }


def _with_selection_contract(snapshot: SpecialistLibrarySnapshot) -> SpecialistLibrarySnapshot:
    metadata = dict(snapshot.metadata or {})
    metadata["selection_contract"] = build_specialist_selection_contract(snapshot)
    return _replace_snapshot(snapshot, metadata=metadata)


def project_specialist_library_snapshot(snapshot: Any, *, registry_status: str | None = None) -> SpecialistLibrarySnapshot:
    resolved = normalize_specialist_library_snapshot(snapshot)
    if resolved is None:
        raise ValueError("specialist_library snapshot is required")

    if registry_status is None:
        return _with_selection_contract(resolved)

    metadata = dict(resolved.metadata or {})
    metadata["projected_registry_status"] = str(registry_status)
    return _with_selection_contract(_replace_snapshot(resolved, metadata=metadata))


def attach_specialist_artifact_refs(
    snapshot: Any,
    *,
    artifact_uri: str,
    meta_artifact_uri: str | None = None,
    artifact_type: str = "registry_model_bundle",
    metadata: Mapping[str, Any] | None = None,
) -> SpecialistLibrarySnapshot:
    resolved = normalize_specialist_library_snapshot(snapshot)
    if resolved is None:
        raise ValueError("specialist_library snapshot is required")

    shared_metadata = _coerce_metadata(metadata)
    existing_refs = {str(ref.model_id): ref for ref in list(resolved.artifact_refs or [])}
    ordered_refs = []
    for spec in list(resolved.specialists or []):
        ref = existing_refs.get(str(spec.model_id))
        if ref is None:
            ref = SpecialistArtifactRef(
                model_id=str(spec.model_id),
                artifact_uri=str(artifact_uri),
                meta_artifact_uri=(None if meta_artifact_uri is None else str(meta_artifact_uri)),
                artifact_type=str(artifact_type),
                metadata={
                    **shared_metadata,
                    "artifact_scope": "specialist_library",
                    "bundle_model_id": str(spec.model_id),
                },
            )
        ordered_refs.append(ref)

    metadata_payload = dict(resolved.metadata or {})
    if shared_metadata:
        metadata_payload["artifact_ref_defaults"] = {
            **dict(metadata_payload.get("artifact_ref_defaults") or {}),
            **shared_metadata,
        }

    return _with_selection_contract(
        _replace_snapshot(
            resolved,
            artifact_refs=ordered_refs,
            metadata=metadata_payload,
        )
    )


def apply_specialist_lifecycle_transition(
    snapshot: Any,
    *,
    model_id: str,
    target_state: Any,
    metadata: Mapping[str, Any] | None = None,
) -> SpecialistLibrarySnapshot:
    resolved = normalize_specialist_library_snapshot(snapshot)
    if resolved is None:
        raise ValueError("specialist_library snapshot is required")

    matched = False
    specialists = []
    for spec in list(resolved.specialists or []):
        if str(spec.model_id) != str(model_id):
            specialists.append(spec)
            continue
        matched = True
        current_state = _normalize_lifecycle_state((spec.metadata or {}).get("lifecycle_state"))
        next_state = resolve_specialist_lifecycle_transition(current_state, target_state)
        specialists.append(
            _replace_spec_state(
                spec,
                next_state,
                metadata={
                    **_coerce_metadata(metadata),
                    "previous_lifecycle_state": current_state.value,
                },
            )
        )

    if not matched:
        raise KeyError(f"Unknown specialist model_id {model_id!r}")

    return _with_selection_contract(_replace_snapshot(resolved, specialists=specialists))


__all__ = [
    "apply_specialist_lifecycle_transition",
    "attach_specialist_artifact_refs",
    "build_specialist_selection_contract",
    "normalize_specialist_library_snapshot",
    "project_specialist_library_snapshot",
    "resolve_specialist_lifecycle_transition",
]