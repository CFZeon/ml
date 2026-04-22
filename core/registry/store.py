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
            "replication_present",
            "replication_coverage",
            "replication_pass_rate",
            "latest_drift_report",
            "latest_monitoring_report",
            "latest_promotion_report",
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
        self._write_index(index)
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
        index.loc[index["version_id"].astype(str) == target_version_id, "current_status"] = "champion"
        self._write_index(index)
        return self._find_row(target_version_id, symbol=symbol)


__all__ = [
    "LocalRegistryStore",
    "evaluate_challenger_promotion",
]