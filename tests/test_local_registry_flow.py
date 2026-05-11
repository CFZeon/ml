import tempfile
import unittest
from pathlib import Path
import json

import pandas as pd

from core import LocalRegistryStore, build_model
from core.specialists import SpecialistLibrarySnapshot, SpecialistSpec


def _fit_logistic_model():
    X = pd.DataFrame({"f1": [0.0, 1.0, 0.0, 1.0], "f2": [1.0, 1.0, 0.0, 0.0]})
    y = pd.Series([0, 1, 0, 1])
    model = build_model("logistic", {"c": 1.0})
    model.fit(X, y)
    return model, list(X.columns)


def _make_specialist_library():
    return SpecialistLibrarySnapshot(
        symbol="BTCUSDT",
        timeframe="1h",
        fallback_model_id="fallback_generalist",
        specialists=[
            SpecialistSpec(
                model_id="fallback_generalist",
                symbol="BTCUSDT",
                timeframe="1h",
                compatible_regimes=[],
                estimator_family="logisticregression",
                metadata={"fallback_only": True, "lifecycle_state": "candidate"},
            ),
            SpecialistSpec(
                model_id="specialist::bull",
                symbol="BTCUSDT",
                timeframe="1h",
                compatible_regimes=["bull"],
                estimator_family="logisticregression",
                metadata={"regime_label": "bull", "lifecycle_state": "candidate"},
            ),
        ],
    )


class LocalRegistryFlowTest(unittest.TestCase):
    def test_version_manifest_remains_immutable_after_status_and_drift_updates(self):
        model, feature_columns = _fit_logistic_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            version_id = store.register_version(
                model,
                symbol="BTCUSDT",
                feature_columns=feature_columns,
                training_summary={"avg_f1_macro": 0.75},
                validation_summary={"raw_objective_value": 0.12},
            )
            manifest_path = Path(temp_dir) / "BTCUSDT" / version_id / "version_manifest.json"
            original_manifest = manifest_path.read_text(encoding="utf-8")

            store.attach_drift_report(version_id, {"sample_count": 250, "evidence_count": 2}, symbol="BTCUSDT")
            store.promote(version_id, "champion", symbol="BTCUSDT", decision={"approved": True, "reasons": ["approved"]})

            updated_manifest = manifest_path.read_text(encoding="utf-8")
            self.assertEqual(original_manifest, updated_manifest)

    def test_challenger_rollback_restores_previous_champion(self):
        first_model, feature_columns = _fit_logistic_model()
        second_model, _ = _fit_logistic_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            v1 = store.register_version(
                first_model,
                symbol="BTCUSDT",
                feature_columns=feature_columns,
                training_summary={"avg_f1_macro": 0.70},
                validation_summary={"raw_objective_value": 0.10},
            )
            store.promote(v1, "champion", symbol="BTCUSDT", decision={"approved": True, "reasons": ["approved"]})

            v2 = store.register_version(
                second_model,
                symbol="BTCUSDT",
                feature_columns=feature_columns,
                training_summary={"avg_f1_macro": 0.82},
                validation_summary={"raw_objective_value": 0.14},
            )
            store.promote(v2, "champion", symbol="BTCUSDT", decision={"approved": True, "reasons": ["approved"]})

            restored = store.rollback("BTCUSDT")
            champion = store.get_champion("BTCUSDT")

            self.assertEqual(restored["version_id"], v1)
            self.assertEqual(champion["version_id"], v1)
            archived_v2 = next(row for row in store.list_versions("BTCUSDT") if row["version_id"] == v2)
            self.assertEqual(archived_v2["current_status"], "archived")

    def test_registry_persists_input_data_lineage(self):
        model, feature_columns = _fit_logistic_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            version_id = store.register_version(
                model,
                symbol="BTCUSDT",
                feature_columns=feature_columns,
                metadata={
                    "data_lineage": {
                        "source_datasets": [
                            {
                                "name": "binance_spot_bars",
                                "contract": {"contract_hash": "abc123"},
                                "source_fingerprint": "fingerprint-1",
                            }
                        ]
                    }
                },
                lineage={
                    "data_lineage": {
                        "source_datasets": [
                            {
                                "name": "binance_spot_bars",
                                "contract": {"contract_hash": "abc123"},
                                "source_fingerprint": "fingerprint-1",
                            }
                        ]
                    }
                },
                training_summary={"avg_f1_macro": 0.75},
                validation_summary={"raw_objective_value": 0.12},
            )

            version_dir = Path(temp_dir) / "BTCUSDT" / version_id
            version_manifest = json.loads((version_dir / "version_manifest.json").read_text(encoding="utf-8"))
            model_manifest = json.loads((version_dir / "model.json").read_text(encoding="utf-8"))

            self.assertEqual(
                version_manifest["lineage"]["data_lineage"]["source_datasets"][0]["contract"]["contract_hash"],
                "abc123",
            )
            self.assertEqual(
                model_manifest["metadata"]["data_lineage"]["source_datasets"][0]["source_fingerprint"],
                "fingerprint-1",
            )

    def test_registry_manifest_persists_signal_decay_summary(self):
        model, feature_columns = _fit_logistic_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            version_id = store.register_version(
                model,
                symbol="BTCUSDT",
                feature_columns=feature_columns,
                training_summary={
                    "avg_f1_macro": 0.75,
                    "signal_decay": {
                        "half_life_bars": 4,
                        "net_edge_at_effective_delay": 0.011,
                    },
                },
                validation_summary={"raw_objective_value": 0.12},
            )

            version_dir = Path(temp_dir) / "BTCUSDT" / version_id
            version_manifest = json.loads((version_dir / "version_manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(version_manifest["training_summary"]["signal_decay"]["half_life_bars"], 4)
            self.assertAlmostEqual(
                float(version_manifest["training_summary"]["signal_decay"]["net_edge_at_effective_delay"]),
                0.011,
            )

    def test_registry_manifest_persists_replication_summary(self):
        model, feature_columns = _fit_logistic_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            version_id = store.register_version(
                model,
                symbol="BTCUSDT",
                feature_columns=feature_columns,
                training_summary={"avg_f1_macro": 0.75},
                validation_summary={"raw_objective_value": 0.12},
                replication={
                    "enabled": True,
                    "completed_cohort_count": 3,
                    "pass_rate": 2.0 / 3.0,
                    "promotion_pass": True,
                },
            )

            version_dir = Path(temp_dir) / "BTCUSDT" / version_id
            version_manifest = json.loads((version_dir / "version_manifest.json").read_text(encoding="utf-8"))

            self.assertTrue(version_manifest["replication"]["enabled"])
            self.assertEqual(version_manifest["replication"]["completed_cohort_count"], 3)
            self.assertAlmostEqual(float(version_manifest["replication"]["pass_rate"]), 2.0 / 3.0)

    def test_registry_index_persists_promotion_score_basis(self):
        model, feature_columns = _fit_logistic_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            version_id = store.register_version(
                model,
                symbol="BTCUSDT",
                feature_columns=feature_columns,
                training_summary={"avg_f1_macro": 0.75},
                validation_summary={"raw_objective_value": 0.12},
                promotion_eligibility_report={
                    "score": {"basis": "selection_value", "value": 0.12},
                    "promotion_ready": True,
                },
            )

            row = next(record for record in store.list_versions("BTCUSDT") if record["version_id"] == version_id)

            self.assertEqual(row["promotion_score_basis"], "selection_value")
            self.assertAlmostEqual(float(row["promotion_score"]), 0.12)

    def test_registry_auto_persists_specialist_library_and_projects_active_runtime_state(self):
        model, feature_columns = _fit_logistic_model()
        specialist_library = _make_specialist_library()
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            version_id = store.register_version(
                model,
                symbol="BTCUSDT",
                feature_columns=feature_columns,
                training_summary={"avg_f1_macro": 0.75, "specialist_library": specialist_library.to_dict()},
                validation_summary={"raw_objective_value": 0.12},
            )

            manifest = store.read_version_manifest(version_id, symbol="BTCUSDT")
            runtime_before = store.read_specialist_library(version_id, symbol="BTCUSDT")
            store.promote(version_id, "champion", symbol="BTCUSDT", decision={"approved": True, "reasons": ["approved"]})
            runtime_after = store.read_specialist_library(version_id, symbol="BTCUSDT")

        manifest_states = {
            spec["model_id"]: spec["metadata"].get("lifecycle_state")
            for spec in manifest["specialist_library"]["specialists"]
        }
        runtime_before_states = runtime_before["metadata"]["selection_contract"]["lifecycle_state_by_model_id"]
        runtime_after_states = runtime_after["metadata"]["selection_contract"]["lifecycle_state_by_model_id"]

        self.assertEqual(manifest_states["specialist::bull"], "candidate")
        self.assertEqual(runtime_before_states["specialist::bull"], "candidate")
        self.assertEqual(runtime_after_states["specialist::bull"], "candidate")
        self.assertEqual(runtime_after_states["fallback_generalist"], "active")
        self.assertEqual(runtime_after["metadata"]["selection_contract"]["active_model_ids"], ["fallback_generalist"])
        self.assertEqual(len(manifest["specialist_library"]["artifact_refs"]), 2)

    def test_registry_promotes_certified_specialists_without_overwriting_runtime_state(self):
        model, feature_columns = _fit_logistic_model()
        specialist_library = _make_specialist_library()
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            version_id = store.register_version(
                model,
                symbol="BTCUSDT",
                feature_columns=feature_columns,
                specialist_library=specialist_library,
                training_summary={"avg_f1_macro": 0.75},
                validation_summary={"raw_objective_value": 0.12},
            )

            certified_runtime = None
            store.record_specialist_lifecycle_transition(
                version_id,
                "specialist::bull",
                "certified",
                symbol="BTCUSDT",
                reason="specialist_certification_passed",
            )
            certified_runtime = store.read_specialist_library(version_id, symbol="BTCUSDT")
            store.promote(version_id, "champion", symbol="BTCUSDT", decision={"approved": True, "reasons": ["approved"]})
            active_runtime = store.read_specialist_library(version_id, symbol="BTCUSDT")

        certified_states = certified_runtime["metadata"]["selection_contract"]["lifecycle_state_by_model_id"]
        active_states = active_runtime["metadata"]["selection_contract"]["lifecycle_state_by_model_id"]

        self.assertEqual(certified_states["specialist::bull"], "certified")
        self.assertEqual(active_states["specialist::bull"], "active")
        self.assertEqual(active_runtime["metadata"]["selection_contract"]["active_model_ids"], ["fallback_generalist", "specialist::bull"])

    def test_registry_records_specialist_lifecycle_events_without_mutating_manifest(self):
        model, feature_columns = _fit_logistic_model()
        specialist_library = _make_specialist_library()
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            version_id = store.register_version(
                model,
                symbol="BTCUSDT",
                feature_columns=feature_columns,
                specialist_library=specialist_library,
                training_summary={"avg_f1_macro": 0.75},
                validation_summary={"raw_objective_value": 0.12},
            )
            manifest_before = store.read_version_manifest(version_id, symbol="BTCUSDT")
            store.promote(version_id, "champion", symbol="BTCUSDT", decision={"approved": True, "reasons": ["approved"]})

            store.record_specialist_lifecycle_transition(
                version_id,
                "specialist::bull",
                "certified",
                symbol="BTCUSDT",
                reason="specialist_certification_passed",
            )
            store.record_specialist_lifecycle_transition(
                version_id,
                "specialist::bull",
                "active",
                symbol="BTCUSDT",
                reason="specialist_activated",
            )

            lifecycle_event = store.record_specialist_lifecycle_transition(
                version_id,
                "specialist::bull",
                "degraded",
                symbol="BTCUSDT",
                reason="performance_decay",
            )
            runtime = store.read_specialist_library(version_id, symbol="BTCUSDT")
            manifest_after = store.read_version_manifest(version_id, symbol="BTCUSDT")
            row = next(record for record in store.list_versions("BTCUSDT") if record["version_id"] == version_id)

        runtime_states = runtime["metadata"]["selection_contract"]["lifecycle_state_by_model_id"]

        self.assertEqual(lifecycle_event["target_state"], "degraded")
        self.assertEqual(runtime_states["specialist::bull"], "degraded")
        self.assertEqual(manifest_before["specialist_library"], manifest_after["specialist_library"])
        self.assertTrue(str(row["latest_specialist_lifecycle_report"]).endswith(".json"))

    def test_registry_replays_specialist_health_updates_without_mutating_manifest(self):
        model, feature_columns = _fit_logistic_model()
        specialist_library = _make_specialist_library()
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            version_id = store.register_version(
                model,
                symbol="BTCUSDT",
                feature_columns=feature_columns,
                specialist_library=specialist_library,
                training_summary={"avg_f1_macro": 0.75},
                validation_summary={"raw_objective_value": 0.12},
            )
            manifest_before = store.read_version_manifest(version_id, symbol="BTCUSDT")
            update = store.attach_specialist_health_update(
                version_id,
                symbol="BTCUSDT",
                update={
                    "recorded_at": "2026-05-09T00:00:00+00:00",
                    "source": "paper_monitoring",
                    "metadata": {
                        "window_start": "2026-05-01T00:00:00+00:00",
                        "window_end": "2026-05-08T00:00:00+00:00",
                    },
                    "health": [
                        {
                            "model_id": "specialist::bull",
                            "compatible_regimes": ["bull"],
                            "stability_score": 0.61,
                            "decay_score": 0.18,
                            "last_calibrated_at": "2026-05-08T00:00:00+00:00",
                            "failure_flags": ["drawdown_watch"],
                            "metadata": {"support_window": "2026Q2"},
                        }
                    ],
                    "performance_slices": [
                        {
                            "model_id": "specialist::bull",
                            "regime_label": "bull",
                            "split_role": "monitoring_window",
                            "row_count": 48,
                            "metric_summary": {"f1_macro": 0.55, "win_rate": 0.58},
                            "metadata": {
                                "window_start": "2026-05-01T00:00:00+00:00",
                                "window_end": "2026-05-08T00:00:00+00:00",
                            },
                        }
                    ],
                },
            )
            runtime = store.read_specialist_library(version_id, symbol="BTCUSDT")
            manifest_after = store.read_version_manifest(version_id, symbol="BTCUSDT")
            row = next(record for record in store.list_versions("BTCUSDT") if record["version_id"] == version_id)

        bull_health = next(item for item in runtime["health"] if item["model_id"] == "specialist::bull")
        bull_performance = [
            item
            for item in runtime["performance_slices"]
            if item["model_id"] == "specialist::bull" and item["split_role"] == "monitoring_window"
        ]
        history = runtime["metadata"]["health_history"]

        self.assertEqual(update["source"], "paper_monitoring")
        self.assertAlmostEqual(float(bull_health["stability_score"]), 0.61)
        self.assertEqual(bull_health["metadata"]["health_source"], "paper_monitoring")
        self.assertEqual(bull_health["metadata"]["health_recorded_at"], "2026-05-09T00:00:00+00:00")
        self.assertEqual(len(bull_performance), 1)
        self.assertAlmostEqual(float(bull_performance[0]["metric_summary"]["f1_macro"]), 0.55)
        self.assertEqual(bull_performance[0]["metadata"]["source"], "paper_monitoring")
        self.assertEqual(history["update_count"], 1)
        self.assertEqual(history["latest_update_at"], "2026-05-09T00:00:00+00:00")
        self.assertEqual(history["failure_flagged_model_ids"], ["specialist::bull"])
        self.assertEqual(history["updated_model_ids"], ["specialist::bull"])
        self.assertEqual(manifest_before["specialist_library"], manifest_after["specialist_library"])
        self.assertTrue(str(row["latest_specialist_health_report"]).endswith(".json"))

    def test_registry_rejects_malformed_specialist_health_updates(self):
        model, feature_columns = _fit_logistic_model()
        specialist_library = _make_specialist_library()
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            version_id = store.register_version(
                model,
                symbol="BTCUSDT",
                feature_columns=feature_columns,
                specialist_library=specialist_library,
                training_summary={"avg_f1_macro": 0.75},
                validation_summary={"raw_objective_value": 0.12},
            )

            with self.assertRaisesRegex(ValueError, "model_id"):
                store.attach_specialist_health_update(
                    version_id,
                    symbol="BTCUSDT",
                    health=[{"stability_score": 0.4}],
                )

            row = next(record for record in store.list_versions("BTCUSDT") if record["version_id"] == version_id)

        self.assertTrue(pd.isna(row["latest_specialist_health_report"]))


if __name__ == "__main__":
    unittest.main()