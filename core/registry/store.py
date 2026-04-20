"""Filesystem-first local registry with immutable version manifests."""

from __future__ import annotations

from pathlib import Path
import uuid

import pandas as pd

from ..drift import evaluate_drift_guardrails
from ..models import load_model, save_model
from ..storage import read_json, read_parquet_frame, write_json, write_parquet_frame
from .manifest import build_feature_schema_hash, build_registry_manifest, flatten_registry_record


def evaluate_challenger_promotion(challenger_summary, champion_record=None, drift_report=None, monitoring_report=None, policy=None):
    policy = dict(policy or {})
    minimum_samples = int(policy.get("min_samples", 200))
    minimum_margin = float(policy.get("minimum_score_margin", 0.0))
    require_automl_ready = bool(policy.get("require_automl_promotion_ready", True))
    require_operational_health = bool(policy.get("require_operational_health", monitoring_report is not None))

    challenger = dict(challenger_summary or {})
    champion = dict(champion_record or {})
    reasons = []

    sample_count = int(challenger.get("sample_count") or 0)
    if sample_count < minimum_samples:
        reasons.append("minimum_samples_not_met")

    if require_automl_ready and not bool(challenger.get("promotion_ready", False)):
        reasons.append("automl_promotion_not_ready")

    challenger_score = challenger.get("selection_value")
    champion_score = champion.get("locked_holdout_score")
    if champion_score is not None and challenger_score is not None:
        if float(challenger_score) < float(champion_score) + minimum_margin:
            reasons.append("challenger_not_better_than_champion")

    if drift_report is not None:
        drift_guardrails = evaluate_drift_guardrails(
            drift_report,
            policy={
                "min_samples": policy.get("drift_min_samples", minimum_samples),
                "cooldown_bars": policy.get("drift_cooldown_bars", 500),
                "min_drift_signals": policy.get("drift_min_signals", 2),
            },
        )
        if not drift_guardrails.get("approved", False):
            reasons.append("drift_guardrails_not_met")
    else:
        drift_guardrails = None

    monitoring_health = None
    if monitoring_report is not None:
        monitoring_health = bool((monitoring_report or {}).get("healthy", False))
        if require_operational_health and not monitoring_health:
            reasons.append("operational_monitoring_not_healthy")

    approved = not reasons
    return {
        "approved": approved,
        "reasons": reasons or ["approved"],
        "sample_count": sample_count,
        "selection_value": challenger_score,
        "drift_guardrails": drift_guardrails,
        "monitoring_health": monitoring_health,
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
            "promotion_ready",
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
            promotion_ready=(dict(validation_summary or {}).get("promotion_ready") if validation_summary else None),
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