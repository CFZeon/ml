"""Specialist-library health and performance update helpers."""

from __future__ import annotations

from typing import Any, Mapping

from .contracts import (
    SpecialistHealthContract,
    SpecialistLibrarySnapshot,
    SpecialistPerformanceSlice,
)
from .library import build_specialist_selection_contract, normalize_specialist_library_snapshot


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


def _replace_snapshot(
    snapshot: SpecialistLibrarySnapshot,
    *,
    health: list[SpecialistHealthContract] | None = None,
    performance_slices: list[SpecialistPerformanceSlice] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SpecialistLibrarySnapshot:
    payload = snapshot.to_dict()
    if health is not None:
        payload["health"] = [item.to_dict() for item in health]
    if performance_slices is not None:
        payload["performance_slices"] = [item.to_dict() for item in performance_slices]
    if metadata is not None:
        payload["metadata"] = _coerce_metadata(metadata)
    updated = SpecialistLibrarySnapshot.from_dict(payload)
    updated_payload = updated.to_dict()
    updated_metadata = dict(updated_payload.get("metadata") or {})
    updated_metadata["selection_contract"] = build_specialist_selection_contract(updated)
    updated_payload["metadata"] = updated_metadata
    return SpecialistLibrarySnapshot.from_dict(updated_payload)


def _normalize_health_contract_payload(payload: Mapping[str, Any]) -> SpecialistHealthContract:
    data = dict(payload or {})
    model_id = str(data.get("model_id", "")).strip()
    if not model_id:
        raise ValueError("specialist health update entries must include model_id")
    data["model_id"] = model_id
    return SpecialistHealthContract.from_dict(data)


def _resolve_health_binding_metadata(payload: Mapping[str, Any], metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    data = dict(payload or {})
    resolved = _coerce_metadata(metadata)
    failure_flags = [str(item) for item in list(data.get("failure_flags") or [])]
    has_measured_evidence = bool(
        data.get("fallback_only")
        or data.get("stability_score") is not None
        or data.get("decay_score") is not None
        or data.get("last_calibrated_at") is not None
        or failure_flags
        or resolved.get("sample_count") is not None
        or resolved.get("metric_basis") is not None
    )
    if "health_binding_resolved" not in resolved:
        resolved["health_binding_resolved"] = bool(has_measured_evidence)
    if not str(resolved.get("health_state") or "").strip():
        if bool(data.get("fallback_only")):
            resolved["health_state"] = "fallback_only"
        elif failure_flags:
            resolved["health_state"] = "failed"
        elif bool(resolved.get("health_binding_resolved", False)):
            resolved["health_state"] = "measured"
        else:
            resolved["health_state"] = "unknown"
    resolved.setdefault("health_evidence_source", resolved.get("health_source") or resolved.get("source") or "specialist_health_update")
    if resolved.get("last_refresh_at") is None:
        last_refresh_at = resolved.get("health_recorded_at") or resolved.get("recorded_at")
        if last_refresh_at is not None:
            resolved["last_refresh_at"] = last_refresh_at
    return resolved


def _normalize_performance_slice_payload(payload: Mapping[str, Any]) -> SpecialistPerformanceSlice:
    data = dict(payload or {})
    model_id = str(data.get("model_id", "")).strip()
    if not model_id:
        raise ValueError("specialist performance updates must include model_id")
    data["model_id"] = model_id
    return SpecialistPerformanceSlice.from_dict(data)


def _performance_slice_key(performance_slice: SpecialistPerformanceSlice) -> tuple[str, str | None, str, Any, Any, Any, Any]:
    metadata = dict(performance_slice.metadata or {})
    return (
        str(performance_slice.model_id),
        None if performance_slice.regime_label is None else str(performance_slice.regime_label),
        str(performance_slice.split_role),
        metadata.get("window_start"),
        metadata.get("window_end"),
        metadata.get("recorded_at"),
        metadata.get("source"),
    )


def normalize_specialist_health_update(update: Mapping[str, Any] | None) -> dict[str, Any]:
    if update is None:
        raise ValueError("specialist health update payload is required")
    if not isinstance(update, Mapping):
        raise TypeError("specialist health update payload must be a mapping")

    payload = dict(update or {})
    recorded_at = payload.get("recorded_at")
    source = str(payload.get("source", "specialist_health_update") or "specialist_health_update")
    metadata = _coerce_metadata(payload.get("metadata") or {})
    if recorded_at is not None:
        metadata.setdefault("recorded_at", _serialize_metadata_value(recorded_at))
    metadata.setdefault("source", source)

    health_updates = []
    for item in list(payload.get("health") or []):
        if not isinstance(item, Mapping):
            raise TypeError("specialist health updates must be mappings")
        item_payload = dict(item)
        item_metadata = _coerce_metadata(item_payload.get("metadata") or {})
        if recorded_at is not None:
            item_metadata.setdefault("health_recorded_at", _serialize_metadata_value(recorded_at))
        item_metadata.setdefault("health_source", source)
        item_payload["metadata"] = _resolve_health_binding_metadata(item_payload, item_metadata)
        health_updates.append(_normalize_health_contract_payload(item_payload))

    performance_updates = []
    for item in list(payload.get("performance_slices") or []):
        if not isinstance(item, Mapping):
            raise TypeError("specialist performance updates must be mappings")
        item_payload = dict(item)
        item_metadata = _coerce_metadata(item_payload.get("metadata") or {})
        if recorded_at is not None:
            item_metadata.setdefault("recorded_at", _serialize_metadata_value(recorded_at))
        item_metadata.setdefault("source", source)
        item_payload["metadata"] = item_metadata
        performance_updates.append(_normalize_performance_slice_payload(item_payload))

    return {
        "recorded_at": None if recorded_at is None else _serialize_metadata_value(recorded_at),
        "source": source,
        "metadata": metadata,
        "health": [item.to_dict() for item in health_updates],
        "performance_slices": [item.to_dict() for item in performance_updates],
    }


def merge_specialist_health_contracts(snapshot: Any, health_contracts, *, recorded_at=None, source=None) -> SpecialistLibrarySnapshot:
    resolved = normalize_specialist_library_snapshot(snapshot)
    if resolved is None:
        raise ValueError("specialist_library snapshot is required")

    existing = {str(contract.model_id): contract for contract in list(resolved.health or [])}
    for item in list(health_contracts or []):
        contract = item if isinstance(item, SpecialistHealthContract) else _normalize_health_contract_payload(item)
        payload = contract.to_dict()
        metadata = dict(payload.get("metadata") or {})
        if recorded_at is not None:
            metadata["health_recorded_at"] = _serialize_metadata_value(recorded_at)
        if source is not None:
            metadata["health_source"] = str(source)
        payload["metadata"] = metadata
        existing[str(contract.model_id)] = SpecialistHealthContract.from_dict(payload)

    ordered_model_ids = []
    for spec in list(resolved.specialists or []):
        if str(spec.model_id) not in ordered_model_ids:
            ordered_model_ids.append(str(spec.model_id))
    for contract in list(resolved.health or []):
        if str(contract.model_id) not in ordered_model_ids:
            ordered_model_ids.append(str(contract.model_id))
    for model_id in existing:
        if model_id not in ordered_model_ids:
            ordered_model_ids.append(model_id)

    merged_health = [existing[model_id] for model_id in ordered_model_ids if model_id in existing]
    return _replace_snapshot(resolved, health=merged_health)


def upsert_specialist_performance_slices(snapshot: Any, performance_slices, *, recorded_at=None, source=None) -> SpecialistLibrarySnapshot:
    resolved = normalize_specialist_library_snapshot(snapshot)
    if resolved is None:
        raise ValueError("specialist_library snapshot is required")

    merged = {}
    ordered_keys = []
    for item in list(resolved.performance_slices or []):
        key = _performance_slice_key(item)
        merged[key] = item
        ordered_keys.append(key)

    for item in list(performance_slices or []):
        performance_slice = item if isinstance(item, SpecialistPerformanceSlice) else _normalize_performance_slice_payload(item)
        payload = performance_slice.to_dict()
        metadata = dict(payload.get("metadata") or {})
        if recorded_at is not None:
            metadata["recorded_at"] = _serialize_metadata_value(recorded_at)
        if source is not None:
            metadata["source"] = str(source)
        payload["metadata"] = metadata
        normalized_slice = SpecialistPerformanceSlice.from_dict(payload)
        key = _performance_slice_key(normalized_slice)
        if key not in merged:
            ordered_keys.append(key)
        merged[key] = normalized_slice

    merged_slices = [merged[key] for key in ordered_keys if key in merged]
    return _replace_snapshot(resolved, performance_slices=merged_slices)


def apply_specialist_health_update(snapshot: Any, update: Mapping[str, Any] | None) -> SpecialistLibrarySnapshot:
    resolved = normalize_specialist_library_snapshot(snapshot)
    if resolved is None:
        raise ValueError("specialist_library snapshot is required")

    normalized_update = normalize_specialist_health_update(update)
    recorded_at = normalized_update.get("recorded_at")
    source = normalized_update.get("source")
    updated = merge_specialist_health_contracts(
        resolved,
        normalized_update.get("health") or [],
        recorded_at=recorded_at,
        source=source,
    )
    updated = upsert_specialist_performance_slices(
        updated,
        normalized_update.get("performance_slices") or [],
        recorded_at=recorded_at,
        source=source,
    )

    metadata = dict(updated.metadata or {})
    history = dict(metadata.get("health_history") or {})
    history["update_count"] = int(history.get("update_count", 0)) + 1
    history["latest_update_at"] = recorded_at
    history["latest_source"] = source
    history["health_contract_count"] = int(len(updated.health or []))
    history["performance_slice_count"] = int(len(updated.performance_slices or []))
    history["failure_flagged_model_ids"] = sorted(
        str(contract.model_id)
        for contract in list(updated.health or [])
        if list(contract.failure_flags or [])
    )
    history["updated_model_ids"] = sorted(
        {
            str(item.get("model_id"))
            for item in list(normalized_update.get("health") or []) + list(normalized_update.get("performance_slices") or [])
            if item.get("model_id") is not None
        }
    )
    metadata["health_history"] = history
    return _replace_snapshot(updated, metadata=metadata)


__all__ = [
    "apply_specialist_health_update",
    "merge_specialist_health_contracts",
    "normalize_specialist_health_update",
    "upsert_specialist_performance_slices",
]