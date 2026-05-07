"""Immutable registry manifest helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any

import pandas as pd


def _serialize_manifest_value(value: Any):
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): _serialize_manifest_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_manifest_value(item) for item in value]
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


def build_feature_schema_hash(feature_columns):
    payload = json.dumps(list(feature_columns or []), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class RegistryVersionManifest:
    version_id: str
    symbol: str
    created_at: str
    initial_status: str
    artifact_manifest: str
    meta_artifact_manifest: str | None
    feature_schema: dict
    lineage: dict = field(default_factory=dict)
    training_summary: dict = field(default_factory=dict)
    validation_summary: dict = field(default_factory=dict)
    locked_holdout: dict = field(default_factory=dict)
    replication: dict = field(default_factory=dict)
    promotion_eligibility_report: dict = field(default_factory=dict)
    regime_contracts: dict = field(default_factory=dict)
    specialist_library: dict = field(default_factory=dict)
    router_manifest: dict = field(default_factory=dict)
    promotion_ready: bool | None = None

    def to_dict(self):
        return asdict(self)


def build_registry_manifest(
    *,
    version_id,
    symbol,
    initial_status,
    artifact_manifest,
    meta_artifact_manifest=None,
    feature_columns=None,
    lineage=None,
    training_summary=None,
    validation_summary=None,
    locked_holdout=None,
    replication=None,
    promotion_eligibility_report=None,
    regime_contracts=None,
    specialist_library=None,
    router_manifest=None,
    promotion_ready=None,
):
    columns = list(feature_columns or [])
    feature_schema = {
        "feature_order": columns,
        "required_columns": columns,
        "schema_hash": build_feature_schema_hash(columns),
    }
    return RegistryVersionManifest(
        version_id=str(version_id),
        symbol=str(symbol),
        created_at=pd.Timestamp.now(tz="UTC").isoformat(),
        initial_status=str(initial_status),
        artifact_manifest=str(artifact_manifest),
        meta_artifact_manifest=str(meta_artifact_manifest) if meta_artifact_manifest is not None else None,
        feature_schema=feature_schema,
        lineage=dict(lineage or {}),
        training_summary=dict(training_summary or {}),
        validation_summary=dict(validation_summary or {}),
        locked_holdout=dict(locked_holdout or {}),
        replication=dict(replication or {}),
        promotion_eligibility_report=dict(promotion_eligibility_report or {}),
        regime_contracts=dict(_serialize_manifest_value(regime_contracts) or {}),
        specialist_library=dict(_serialize_manifest_value(specialist_library) or {}),
        router_manifest=dict(_serialize_manifest_value(router_manifest) or {}),
        promotion_ready=promotion_ready,
    )


def flatten_registry_record(manifest, *, current_status, version_dir, latest_drift_report=None, latest_promotion_report=None):
    payload = manifest.to_dict() if hasattr(manifest, "to_dict") else dict(manifest or {})
    feature_schema = dict(payload.get("feature_schema") or {})
    training_summary = dict(payload.get("training_summary") or {})
    validation_summary = dict(payload.get("validation_summary") or {})
    locked_holdout = dict(payload.get("locked_holdout") or {})
    replication = dict(payload.get("replication") or {})
    promotion_eligibility_report = dict(payload.get("promotion_eligibility_report") or {})
    regime_contracts = dict(payload.get("regime_contracts") or {})
    specialist_library = dict(payload.get("specialist_library") or {})
    router_manifest = dict(payload.get("router_manifest") or {})
    promotion_score = dict(promotion_eligibility_report.get("score") or {})
    return {
        "version_id": payload.get("version_id"),
        "symbol": payload.get("symbol"),
        "created_at": payload.get("created_at"),
        "initial_status": payload.get("initial_status"),
        "current_status": current_status,
        "version_dir": str(version_dir),
        "artifact_manifest": payload.get("artifact_manifest"),
        "meta_artifact_manifest": payload.get("meta_artifact_manifest"),
        "feature_schema_hash": feature_schema.get("schema_hash"),
        "feature_count": int(len(feature_schema.get("feature_order") or [])),
        "selection_value": validation_summary.get("raw_objective_value") or training_summary.get("avg_f1_macro"),
        "promotion_score": promotion_score.get("value"),
        "promotion_score_basis": promotion_score.get("basis"),
        "eligibility_report_present": bool(promotion_eligibility_report),
        "promotion_ready": payload.get("promotion_ready"),
        "regime_contracts_present": bool(regime_contracts),
        "specialist_library_present": bool(specialist_library),
        "router_manifest_present": bool(router_manifest),
        "replication_present": bool(replication),
        "replication_coverage": replication.get("completed_cohort_count"),
        "replication_pass_rate": replication.get("pass_rate"),
        "latest_drift_report": str(latest_drift_report) if latest_drift_report is not None else None,
        "latest_promotion_report": str(latest_promotion_report) if latest_promotion_report is not None else None,
        "locked_holdout_score": locked_holdout.get("raw_objective_value"),
    }


__all__ = [
    "RegistryVersionManifest",
    "build_feature_schema_hash",
    "build_registry_manifest",
    "flatten_registry_record",
]