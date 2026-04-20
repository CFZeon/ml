import tempfile
import unittest
from pathlib import Path
import json

import pandas as pd

from core import LocalRegistryStore, build_model


def _fit_logistic_model():
    X = pd.DataFrame({"f1": [0.0, 1.0, 0.0, 1.0], "f2": [1.0, 1.0, 0.0, 0.0]})
    y = pd.Series([0, 1, 0, 1])
    model = build_model("logistic", {"c": 1.0})
    model.fit(X, y)
    return model, list(X.columns)


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


if __name__ == "__main__":
    unittest.main()