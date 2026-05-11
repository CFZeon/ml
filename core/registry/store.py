"""Filesystem-first local registry with immutable version manifests."""

from __future__ import annotations

from pathlib import Path
import uuid

import pandas as pd

from ..drift import evaluate_drift_guardrails
from ..models import load_model, save_model
from ..promotion import (
    create_promotion_eligibility_report,
    finalize_promotion_eligibility_report,
    get_promotion_score,
    resolve_promotion_gate_mode,
    upsert_promotion_gate,
)
from ..specialists.library import (
    apply_specialist_lifecycle_transition,
    attach_specialist_artifact_refs,
    normalize_specialist_library_snapshot,
    project_specialist_library_snapshot,
    resolve_specialist_lifecycle_transition,
)
from ..specialists.health import apply_specialist_health_update, normalize_specialist_health_update
from ..storage import read_json, read_parquet_frame, write_json, write_parquet_frame
from .manifest import build_feature_schema_hash, build_registry_manifest, flatten_registry_record


def evaluate_challenger_promotion(challenger_summary, champion_record=None, drift_report=None, monitoring_report=None, policy=None):
    policy = dict(policy or {})
    minimum_samples = int(policy.get("min_samples", 200))
    minimum_margin = float(policy.get("minimum_score_margin", 0.0))
    require_automl_ready = bool(policy.get("require_automl_promotion_ready", True))
    require_operational_health = bool(policy.get("require_operational_health", monitoring_report is not None))
    require_canonical_report = bool(policy.get("require_canonical_report", True))
    allow_legacy_versions = bool(policy.get("allow_legacy_versions", False))
    score_preference = str(policy.get("score_preference", "locked_holdout_first"))
    calibration_mode = bool(policy.get("calibration_mode", False))

    challenger = dict(challenger_summary or {})
    champion = dict(champion_record or {})

    challenger_report = dict(challenger.get("promotion_eligibility_report") or {})
    if challenger_report:
        eligibility_report = finalize_promotion_eligibility_report(challenger_report)
    else:
        eligibility_report = create_promotion_eligibility_report(calibration_mode=calibration_mode)
    if require_canonical_report and not challenger_report:
        eligibility_report = upsert_promotion_gate(
            eligibility_report,
            group="registry",
            name="canonical_report_present",
            passed=False,
            mode=resolve_promotion_gate_mode(policy, "canonical_report_present"),
            measured=False,
            threshold=True,
            reason="challenger_missing_canonical_eligibility_report",
            details={"legacy": True},
        )

    sample_count = int(challenger.get("sample_count") or 0)
    eligibility_report = upsert_promotion_gate(
        eligibility_report,
        group="registry",
        name="minimum_samples",
        passed=bool(sample_count >= minimum_samples),
        mode=resolve_promotion_gate_mode(policy, "minimum_samples"),
        measured=sample_count,
        threshold=minimum_samples,
        reason=None if sample_count >= minimum_samples else "minimum_samples_not_met",
        details={"sample_count": sample_count},
    )

    automl_ready = bool(eligibility_report.get("promotion_ready", challenger.get("promotion_ready", False)))
    eligibility_report = upsert_promotion_gate(
        eligibility_report,
        group="registry",
        name="automl_promotion_ready",
        passed=bool(automl_ready or not require_automl_ready),
        mode=resolve_promotion_gate_mode(policy, "automl_promotion_ready"),
        measured=automl_ready,
        threshold=True,
        reason=None if (automl_ready or not require_automl_ready) else "automl_promotion_not_ready",
        details={"require_automl_ready": require_automl_ready},
    )

    challenger_score = get_promotion_score(eligibility_report)
    champion_score = None
    champion_manifest = None
    if champion:
        champion_manifest = read_json(Path(champion["version_dir"]) / "version_manifest.json") or {}
        champion_report = dict(champion_manifest.get("promotion_eligibility_report") or {})
        if champion_report:
            champion_score = get_promotion_score(finalize_promotion_eligibility_report(champion_report))
        elif allow_legacy_versions:
            champion_score = {
                "basis": "locked_holdout_raw_objective" if champion.get("locked_holdout_score") is not None else "selection_value",
                "value": champion.get("locked_holdout_score") if champion.get("locked_holdout_score") is not None else champion.get("selection_value"),
            }
        else:
            eligibility_report = upsert_promotion_gate(
                eligibility_report,
                group="registry",
                name="champion_comparable",
                passed=False,
                mode=resolve_promotion_gate_mode(policy, "champion_comparable"),
                measured=False,
                threshold=True,
                reason="legacy_champion_missing_canonical_eligibility_report",
                details={"version_id": champion.get("version_id")},
            )

        challenger_score_value = challenger_score.get("value") if challenger_score else None
        challenger_score_basis = challenger_score.get("basis") if challenger_score else None
        champion_score_basis = champion_score.get("basis") if champion_score else None
        if challenger_score_value is None:
            eligibility_report = upsert_promotion_gate(
                eligibility_report,
                group="registry",
                name="promotion_score_present",
                passed=False,
                mode=resolve_promotion_gate_mode(policy, "promotion_score_present"),
                measured=None,
                threshold=score_preference,
                reason="promotion_score_unavailable",
                details={"score_preference": score_preference},
            )
        elif champion_score is not None and champion_score.get("value") is not None and challenger_score_basis != champion_score_basis:
            eligibility_report = upsert_promotion_gate(
                eligibility_report,
                group="registry",
                name="score_basis_aligned",
                passed=False,
                mode=resolve_promotion_gate_mode(policy, "score_basis_aligned"),
                measured={"challenger": challenger_score_basis, "champion": champion_score_basis},
                threshold="same_score_basis",
                reason="promotion_score_basis_mismatch",
                details={
                    "challenger_score": challenger_score,
                    "champion_score": champion_score,
                },
            )
        elif champion_score is not None and champion_score.get("value") is not None:
            eligibility_report = upsert_promotion_gate(
                eligibility_report,
                group="registry",
                name="score_basis_aligned",
                passed=True,
                mode=resolve_promotion_gate_mode(policy, "score_basis_aligned"),
                measured={"challenger": challenger_score_basis, "champion": champion_score_basis},
                threshold="same_score_basis",
                reason=None,
                details={
                    "challenger_score": challenger_score,
                    "champion_score": champion_score,
                },
            )
            margin = float(challenger_score_value) - float(champion_score["value"])
            eligibility_report = upsert_promotion_gate(
                eligibility_report,
                group="registry",
                name="score_margin",
                passed=bool(float(challenger_score_value) >= float(champion_score["value"]) + minimum_margin),
                mode=resolve_promotion_gate_mode(policy, "score_margin"),
                measured=margin,
                threshold=minimum_margin,
                reason=None if float(challenger_score_value) >= float(champion_score["value"]) + minimum_margin else "challenger_not_better_than_champion",
                details={
                    "challenger_score": challenger_score,
                    "champion_score": champion_score,
                    "score_preference": score_preference,
                },
            )
        elif champion and champion_score is None:
            eligibility_report = upsert_promotion_gate(
                eligibility_report,
                group="registry",
                name="promotion_score_present",
                passed=False,
                mode=resolve_promotion_gate_mode(policy, "promotion_score_present"),
                measured=None,
                threshold=score_preference,
                reason="promotion_score_unavailable",
                details={"score_preference": score_preference},
            )

    if drift_report is not None:
        drift_guardrails = evaluate_drift_guardrails(
            drift_report,
            policy={
                "min_samples": policy.get("drift_min_samples", minimum_samples),
                "cooldown_bars": policy.get("drift_cooldown_bars", 500),
                "min_drift_signals": policy.get("drift_min_signals", 2),
            },
        )
    else:
        drift_guardrails = None
    eligibility_report = upsert_promotion_gate(
        eligibility_report,
        group="registry",
        name="drift_guardrails",
        passed=bool(drift_guardrails is None or drift_guardrails.get("approved", False)),
        mode=resolve_promotion_gate_mode(policy, "drift_guardrails"),
        measured=(drift_guardrails or {}).get("signal_count"),
        threshold=policy.get("drift_min_signals", 2),
        reason=None if (drift_guardrails is None or drift_guardrails.get("approved", False)) else "drift_guardrails_not_met",
        details=drift_guardrails or {},
    )

    monitoring_health = None
    if monitoring_report is not None:
        monitoring_health = bool((monitoring_report or {}).get("healthy", False))
    eligibility_report = upsert_promotion_gate(
        eligibility_report,
        group="registry",
        name="registry_operational_health",
        passed=bool((not require_operational_health) or monitoring_health),
        mode=resolve_promotion_gate_mode(policy, "registry_operational_health"),
        measured=monitoring_health,
        threshold=True,
        reason=None if ((not require_operational_health) or monitoring_health) else "operational_monitoring_not_healthy",
        details=monitoring_report or {},
    )

    eligibility_report = finalize_promotion_eligibility_report(eligibility_report)
    approved = bool(eligibility_report.get("approved", False))
    return {
        "approved": approved,
        "reasons": list(eligibility_report.get("blocking_failures") or ["approved"]),
        "sample_count": sample_count,
        "selection_value": challenger.get("selection_value"),
        "score_basis": (challenger_score or {}).get("basis"),
        "score_value": (challenger_score or {}).get("value"),
        "drift_guardrails": drift_guardrails,
        "monitoring_health": monitoring_health,
        "promotion_eligibility_report": eligibility_report,
    }


class LocalRegistryStore:
    def __init__(self, root_dir=".cache/registry", max_versions_per_symbol=10):
        self.root_dir = Path(root_dir)
        self.max_versions_per_symbol = int(max(1, max_versions_per_symbol))
        self.index_path = self.root_dir / "index.parquet"
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _read_index(self):
        frame = read_parquet_frame(self.index_path)
        expected_columns = [
            "version_id",
            "symbol",
            "created_at",
            "promoted_at",
            "initial_status",
            "current_status",
            "version_dir",
            "artifact_manifest",
            "meta_artifact_manifest",
            "feature_schema_hash",
            "feature_count",
            "selection_value",
            "promotion_score",
            "promotion_score_basis",
            "eligibility_report_present",
            "promotion_ready",
            "regime_contracts_present",
            "specialist_library_present",
            "router_manifest_present",
            "replication_present",
            "replication_coverage",
            "replication_pass_rate",
            "latest_drift_report",
            "latest_paper_report",
            "latest_monitoring_report",
            "latest_promotion_report",
            "latest_specialist_lifecycle_report",
            "latest_specialist_health_report",
            "locked_holdout_score",
        ]
        if frame is None or frame.empty:
            return pd.DataFrame(columns=expected_columns)
        frame = frame.copy()
        for column in expected_columns:
            if column not in frame.columns:
                frame[column] = None
        return frame[expected_columns]

    def _write_index(self, frame):
        write_parquet_frame(self.index_path, frame.reset_index(drop=True))

    def _version_dir(self, symbol, version_id):
        return self.root_dir / str(symbol) / str(version_id)

    def _manifest_path(self, symbol, version_id):
        return self._version_dir(symbol, version_id) / "version_manifest.json"

    def _specialist_lifecycle_dir(self, symbol, version_id):
        return self._version_dir(symbol, version_id) / "specialist_lifecycle"

    def _specialist_health_dir(self, symbol, version_id):
        return self._version_dir(symbol, version_id) / "specialist_health"

    def _write_immutable_manifest(self, manifest_path, payload):
        if Path(manifest_path).exists():
            raise FileExistsError(f"Registry manifest already exists at {manifest_path}")
        write_json(manifest_path, payload)

    def read_version_manifest(self, version_id, symbol=None):
        row = self._find_row(version_id, symbol=symbol)
        if row is None:
            raise FileNotFoundError(f"Registry version {version_id!r} was not found")
        manifest_path = Path(row["version_dir"]) / "version_manifest.json"
        manifest = read_json(manifest_path)
        if manifest is None:
            raise FileNotFoundError(f"Registry manifest not found at {manifest_path}")
        return manifest

    def _find_row(self, version_id, symbol=None):
        index = self._read_index()
        mask = index["version_id"].astype(str) == str(version_id)
        if symbol is not None:
            mask &= index["symbol"].astype(str) == str(symbol)
        matches = index.loc[mask]
        if matches.empty:
            return None
        return matches.iloc[-1].to_dict()

    def list_versions(self, symbol=None):
        index = self._read_index()
        if symbol is not None:
            index = index.loc[index["symbol"].astype(str) == str(symbol)]
        if index.empty:
            return []
        index = index.sort_values(["symbol", "created_at"], ascending=[True, False])
        return index.to_dict(orient="records")

    def get_champion(self, symbol):
        versions = [row for row in self.list_versions(symbol=symbol) if row.get("current_status") == "champion"]
        return versions[0] if versions else None

    def _resolve_specialist_library_snapshot(
        self,
        specialist_library,
        training_summary,
        *,
        symbol,
        version_id,
        version_dir,
        status,
        primary_manifest_path,
        meta_manifest_path=None,
    ):
        candidate = specialist_library
        if not candidate:
            candidate = dict(training_summary or {}).get("specialist_library") or None
        snapshot = normalize_specialist_library_snapshot(candidate)
        if snapshot is None:
            return None

        projected = project_specialist_library_snapshot(snapshot, registry_status=status)
        return attach_specialist_artifact_refs(
            projected,
            artifact_uri="model",
            meta_artifact_uri=("meta_model" if meta_manifest_path is not None else None),
            artifact_type="registry_model_bundle",
            metadata={
                "version_id": str(version_id),
                "symbol": str(symbol),
                "initial_registry_status": str(status),
                "version_dir": str(version_dir),
                "artifact_manifest": str(primary_manifest_path.name),
                "meta_artifact_manifest": (
                    str(meta_manifest_path.name) if meta_manifest_path is not None else None
                ),
            },
        )

    def _read_specialist_lifecycle_events(self, version_id, *, symbol=None):
        row = self._find_row(version_id, symbol=symbol)
        if row is None:
            raise FileNotFoundError(f"Registry version {version_id!r} was not found")
        lifecycle_dir = self._specialist_lifecycle_dir(row.get("symbol"), version_id)
        if not lifecycle_dir.exists():
            return []
        ordered_events = []
        for path in lifecycle_dir.glob("*.json"):
            payload = read_json(path)
            if isinstance(payload, dict):
                ordered_events.append((str(payload.get("recorded_at") or ""), str(path.name), payload))
        ordered_events.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in ordered_events]

    def _read_specialist_health_updates(self, version_id, *, symbol=None):
        row = self._find_row(version_id, symbol=symbol)
        if row is None:
            raise FileNotFoundError(f"Registry version {version_id!r} was not found")
        health_dir = self._specialist_health_dir(row.get("symbol"), version_id)
        if not health_dir.exists():
            return []
        ordered_updates = []
        for path in health_dir.glob("*.json"):
            payload = read_json(path)
            if isinstance(payload, dict):
                ordered_updates.append((str(payload.get("recorded_at") or ""), str(path.name), payload))
        ordered_updates.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in ordered_updates]

    def register_version(
        self,
        model,
        *,
        symbol,
        feature_columns,
        metadata=None,
        training_summary=None,
        validation_summary=None,
        locked_holdout=None,
        replication=None,
        promotion_eligibility_report=None,
        lineage=None,
        regime_contracts=None,
        specialist_library=None,
        router_manifest=None,
        status="challenger",
        meta_model=None,
    ):
        version_id = f"{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        version_dir = self._version_dir(symbol, version_id)
        version_dir.mkdir(parents=True, exist_ok=False)

        primary_manifest_path = save_model(
            model,
            version_dir / "model",
            metadata={
                **dict(metadata or {}),
                "feature_columns": list(feature_columns or []),
                "symbol": symbol,
                "version_id": version_id,
            },
            feature_schema={
                "feature_order": list(feature_columns or []),
                "required_columns": list(feature_columns or []),
                "schema_hash": build_feature_schema_hash(feature_columns),
            },
        )
        meta_manifest_path = None
        if meta_model is not None:
            meta_manifest_path = save_model(
                meta_model,
                version_dir / "meta_model",
                metadata={"symbol": symbol, "version_id": version_id, "feature_columns": []},
                feature_schema={"feature_order": [], "required_columns": []},
            )

        resolved_specialist_library = self._resolve_specialist_library_snapshot(
            specialist_library,
            training_summary,
            symbol=symbol,
            version_id=version_id,
            version_dir=version_dir,
            status=status,
            primary_manifest_path=primary_manifest_path,
            meta_manifest_path=meta_manifest_path,
        )

        manifest = build_registry_manifest(
            version_id=version_id,
            symbol=symbol,
            initial_status=status,
            artifact_manifest=primary_manifest_path.name,
            meta_artifact_manifest=(meta_manifest_path.name if meta_manifest_path is not None else None),
            feature_columns=feature_columns,
            lineage=lineage,
            training_summary=training_summary,
            validation_summary=validation_summary,
            locked_holdout=locked_holdout,
            replication=replication,
            promotion_eligibility_report=promotion_eligibility_report,
            regime_contracts=regime_contracts,
            specialist_library=resolved_specialist_library,
            router_manifest=router_manifest,
            promotion_ready=(
                dict(promotion_eligibility_report or {}).get("promotion_ready")
                if promotion_eligibility_report is not None
                else (dict(validation_summary or {}).get("promotion_ready") if validation_summary else None)
            ),
        )
        manifest_path = self._manifest_path(symbol, version_id)
        self._write_immutable_manifest(manifest_path, manifest.to_dict())

        index = self._read_index()
        row = flatten_registry_record(
            manifest,
            current_status=status,
            version_dir=version_dir,
        )
        row["promoted_at"] = None
        if index.empty:
            index = pd.DataFrame([row])
        else:
            index = pd.concat([index, pd.DataFrame([row])], ignore_index=True)
        self._write_index(index)
        self._apply_retention(symbol)
        return version_id

    def _apply_retention(self, symbol):
        index = self._read_index()
        symbol_rows = index.loc[index["symbol"].astype(str) == str(symbol)].copy()
        if len(symbol_rows) <= self.max_versions_per_symbol:
            return

        protected_ids = set(symbol_rows.loc[symbol_rows["current_status"] == "champion", "version_id"].astype(str))
        excess = len(symbol_rows) - self.max_versions_per_symbol
        ordered = symbol_rows.sort_values("created_at", ascending=True)
        archived = 0
        for _, row in ordered.iterrows():
            version_id = str(row["version_id"])
            if version_id in protected_ids:
                continue
            mask = index["version_id"].astype(str) == version_id
            index.loc[mask, "current_status"] = "archived"
            archived += 1
            if archived >= excess:
                break
        self._write_index(index)

    def validate_schema(self, version_id, feature_columns, symbol=None):
        manifest = self.read_version_manifest(version_id, symbol=symbol)
        schema = dict(manifest.get("feature_schema") or {})
        expected = list(schema.get("feature_order") or [])
        return expected == list(feature_columns or [])

    def load(self, version_id, *, symbol=None, expected_feature_columns=None, load_meta_model=False):
        row = self._find_row(version_id, symbol=symbol)
        if row is None:
            raise FileNotFoundError(f"Registry version {version_id!r} was not found")
        version_dir = Path(row["version_dir"])
        model, metadata = load_model(version_dir / "model", expected_feature_columns=expected_feature_columns)
        payload = {
            "manifest": self.read_version_manifest(version_id, symbol=symbol),
            "metadata": metadata,
        }
        if load_meta_model and row.get("meta_artifact_manifest"):
            payload["meta_model"] = load_model(version_dir / "meta_model", expected_feature_columns=[])[0]
        return model, payload

    def attach_drift_report(self, version_id, drift_report, *, symbol=None):
        row = self._find_row(version_id, symbol=symbol)
        if row is None:
            raise FileNotFoundError(f"Registry version {version_id!r} was not found")
        version_dir = Path(row["version_dir"])
        drift_dir = version_dir / "drift"
        drift_dir.mkdir(parents=True, exist_ok=True)
        report_path = drift_dir / f"{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S')}.json"
        write_json(report_path, drift_report)

        index = self._read_index()
        mask = index["version_id"].astype(str) == str(version_id)
        index.loc[mask, "latest_drift_report"] = str(report_path)
        self._write_index(index)
        return report_path

    def attach_monitoring_report(self, version_id, monitoring_report, *, symbol=None):
        row = self._find_row(version_id, symbol=symbol)
        if row is None:
            raise FileNotFoundError(f"Registry version {version_id!r} was not found")
        version_dir = Path(row["version_dir"])
        monitoring_dir = version_dir / "monitoring"
        monitoring_dir.mkdir(parents=True, exist_ok=True)
        report_path = monitoring_dir / f"{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S')}.json"
        write_json(report_path, monitoring_report)

        index = self._read_index()
        mask = index["version_id"].astype(str) == str(version_id)
        index.loc[mask, "latest_monitoring_report"] = str(report_path)
        self._write_index(index)
        return report_path

    def attach_specialist_health_update(
        self,
        version_id,
        update=None,
        *,
        symbol=None,
        health=None,
        performance_slices=None,
        source="specialist_health_update",
        metadata=None,
        recorded_at=None,
    ):
        row = self._find_row(version_id, symbol=symbol)
        if row is None:
            raise FileNotFoundError(f"Registry version {version_id!r} was not found")

        specialist_library = self.read_specialist_library(version_id, symbol=symbol)
        if not specialist_library:
            raise ValueError(f"Registry version {version_id!r} does not contain a specialist_library")

        payload = normalize_specialist_health_update(
            update
            or {
                "recorded_at": recorded_at or pd.Timestamp.now(tz="UTC").isoformat(),
                "source": source,
                "metadata": dict(metadata or {}),
                "health": list(health or []),
                "performance_slices": list(performance_slices or []),
            }
        )

        health_dir = self._specialist_health_dir(row.get("symbol"), version_id)
        health_dir.mkdir(parents=True, exist_ok=True)
        report_path = health_dir / f"{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
        write_json(report_path, payload)

        index = self._read_index()
        mask = index["version_id"].astype(str) == str(version_id)
        index.loc[mask, "latest_specialist_health_report"] = str(report_path)
        self._write_index(index)
        return payload

    def attach_paper_report(self, version_id, paper_report, *, symbol=None):
        row = self._find_row(version_id, symbol=symbol)
        if row is None:
            raise FileNotFoundError(f"Registry version {version_id!r} was not found")
        version_dir = Path(row["version_dir"])
        paper_dir = version_dir / "paper"
        paper_dir.mkdir(parents=True, exist_ok=True)
        report_path = paper_dir / f"{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S')}.json"
        write_json(report_path, paper_report)

        index = self._read_index()
        mask = index["version_id"].astype(str) == str(version_id)
        index.loc[mask, "latest_paper_report"] = str(report_path)
        self._write_index(index)
        return report_path

    def _activate_specialists_for_champion(self, version_id, *, symbol=None):
        specialist_library = self.read_specialist_library(version_id, symbol=symbol)
        if not specialist_library:
            return []

        selection_contract = dict((specialist_library.get("metadata") or {}).get("selection_contract") or {})
        state_map = dict(selection_contract.get("lifecycle_state_by_model_id") or {})
        fallback_model_id = specialist_library.get("fallback_model_id")
        transitions = []

        if fallback_model_id and str(state_map.get(str(fallback_model_id))) == "candidate":
            transitions.append((str(fallback_model_id), "certified", "champion_fallback_certified"))
            transitions.append((str(fallback_model_id), "active", "champion_fallback_activated"))

        for model_id in list(selection_contract.get("certified_model_ids") or []):
            if str(state_map.get(str(model_id))) == "certified":
                transitions.append((str(model_id), "active", "champion_specialist_activated"))

        recorded = []
        for model_id, target_state, reason in transitions:
            result = self.record_specialist_lifecycle_transition(
                version_id,
                model_id,
                target_state,
                symbol=symbol,
                reason=reason,
            )
            if not result.get("skipped"):
                recorded.append(result)
        return recorded

    def record_promotion_decision(self, version_id, decision, *, symbol=None):
        row = self._find_row(version_id, symbol=symbol)
        if row is None:
            raise FileNotFoundError(f"Registry version {version_id!r} was not found")
        version_dir = Path(row["version_dir"])
        report_path = version_dir / "promotion_decision.json"
        write_json(report_path, decision)

        index = self._read_index()
        mask = index["version_id"].astype(str) == str(version_id)
        index.loc[mask, "latest_promotion_report"] = str(report_path)
        index.loc[mask, "promotion_ready"] = bool((decision or {}).get("approved", False))
        self._write_index(index)
        return report_path

    def read_specialist_library(self, version_id, *, symbol=None):
        row = self._find_row(version_id, symbol=symbol)
        if row is None:
            raise FileNotFoundError(f"Registry version {version_id!r} was not found")

        manifest = self.read_version_manifest(version_id, symbol=symbol)
        snapshot = normalize_specialist_library_snapshot(manifest.get("specialist_library") or {})
        if snapshot is None:
            return None

        runtime_snapshot = project_specialist_library_snapshot(
            snapshot,
            registry_status=row.get("current_status"),
        )
        for event in self._read_specialist_lifecycle_events(version_id, symbol=symbol):
            runtime_snapshot = apply_specialist_lifecycle_transition(
                runtime_snapshot,
                model_id=str(event.get("model_id")),
                target_state=event.get("target_state"),
                metadata={
                    "lifecycle_reason": event.get("reason"),
                    "lifecycle_recorded_at": event.get("recorded_at"),
                },
            )
        for update in self._read_specialist_health_updates(version_id, symbol=symbol):
            runtime_snapshot = apply_specialist_health_update(runtime_snapshot, update)
        return runtime_snapshot.to_dict()

    def get_active_specialist_library(self, symbol):
        champion = self.get_champion(symbol)
        if champion is None:
            return None
        return self.read_specialist_library(champion["version_id"], symbol=symbol)

    def record_specialist_lifecycle_transition(
        self,
        version_id,
        model_id,
        target_state,
        *,
        symbol=None,
        reason=None,
        metadata=None,
    ):
        row = self._find_row(version_id, symbol=symbol)
        if row is None:
            raise FileNotFoundError(f"Registry version {version_id!r} was not found")

        specialist_library = self.read_specialist_library(version_id, symbol=symbol)
        if not specialist_library:
            raise ValueError(f"Registry version {version_id!r} does not contain a specialist_library")

        state_map = dict(
            ((specialist_library.get("metadata") or {}).get("selection_contract") or {}).get(
                "lifecycle_state_by_model_id"
            )
            or {}
        )
        current_state = state_map.get(str(model_id))
        if current_state is None:
            raise KeyError(f"Unknown specialist model_id {model_id!r}")

        resolved_target_state = resolve_specialist_lifecycle_transition(current_state, target_state)
        if str(current_state) == resolved_target_state.value:
            return {
                "skipped": True,
                "model_id": str(model_id),
                "current_state": str(current_state),
                "target_state": resolved_target_state.value,
            }

        lifecycle_dir = self._specialist_lifecycle_dir(row.get("symbol"), version_id)
        lifecycle_dir.mkdir(parents=True, exist_ok=True)
        timestamp = pd.Timestamp.now(tz="UTC")
        payload = {
            "recorded_at": timestamp.isoformat(),
            "version_id": str(version_id),
            "symbol": str(row.get("symbol")),
            "model_id": str(model_id),
            "previous_state": str(current_state),
            "target_state": resolved_target_state.value,
            "reason": None if reason is None else str(reason),
            "metadata": dict(metadata or {}),
        }
        report_path = lifecycle_dir / f"{timestamp.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
        write_json(report_path, payload)

        index = self._read_index()
        mask = index["version_id"].astype(str) == str(version_id)
        index.loc[mask, "latest_specialist_lifecycle_report"] = str(report_path)
        self._write_index(index)
        return payload

    def promote(self, version_id, stage="champion", *, symbol=None, decision=None):
        row = self._find_row(version_id, symbol=symbol)
        if row is None:
            raise FileNotFoundError(f"Registry version {version_id!r} was not found")
        symbol = symbol or row.get("symbol")
        index = self._read_index()

        if stage == "champion":
            champion_mask = (
                index["symbol"].astype(str) == str(symbol)
            ) & (index["current_status"] == "champion")
            index.loc[champion_mask, "current_status"] = "archived"

        target_mask = index["version_id"].astype(str) == str(version_id)
        index.loc[target_mask, "current_status"] = stage
        if stage == "champion":
            index.loc[target_mask, "promoted_at"] = pd.Timestamp.now(tz="UTC").isoformat()
        self._write_index(index)
        if stage == "champion":
            self._activate_specialists_for_champion(version_id, symbol=symbol)
        if decision is not None:
            self.record_promotion_decision(version_id, decision, symbol=symbol)
        return self._find_row(version_id, symbol=symbol)

    def rollback(self, symbol):
        index = self._read_index()
        symbol_rows = index.loc[index["symbol"].astype(str) == str(symbol)].copy()
        if symbol_rows.empty:
            raise FileNotFoundError(f"No registry versions found for symbol {symbol!r}")

        current_champion = symbol_rows.loc[symbol_rows["current_status"] == "champion"]
        archived_rows = symbol_rows.loc[symbol_rows["current_status"] == "archived"].sort_values("created_at", ascending=False)
        if archived_rows.empty:
            raise RuntimeError(f"No archived versions available for rollback on {symbol!r}")

        target_version_id = str(archived_rows.iloc[0]["version_id"])
        if not current_champion.empty:
            current_id = str(current_champion.iloc[0]["version_id"])
            index.loc[index["version_id"].astype(str) == current_id, "current_status"] = "archived"
        target_mask = index["version_id"].astype(str) == target_version_id
        index.loc[target_mask, "current_status"] = "champion"
        index.loc[target_mask, "promoted_at"] = pd.Timestamp.now(tz="UTC").isoformat()
        self._write_index(index)
        return self._find_row(target_version_id, symbol=symbol)


__all__ = [
    "LocalRegistryStore",
    "evaluate_challenger_promotion",
]