import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from sklearn.dummy import DummyClassifier

from core.registry import LocalRegistryStore, build_registry_manifest, flatten_registry_record
from core.routing import RouterManifest
from core.specialists import SpecialistLibrarySnapshot, SpecialistSpec


def _fit_dummy_classifier(label):
    features = pd.DataFrame({"feature": [0.0, 1.0, 2.0]})
    targets = pd.Series([label, label, label])
    model = DummyClassifier(strategy="prior")
    model.fit(features, targets)
    return model


class RegistryManifestContractCompatibilityTest(unittest.TestCase):
    def test_manifest_and_flattened_record_preserve_optional_phase_zero_sections(self):
        specialist_library = SpecialistLibrarySnapshot(
            symbol="BTCUSDT",
            timeframe="1h",
            fallback_model_id="fallback_generalist",
            specialists=[
                SpecialistSpec(
                    model_id="specialist::bull",
                    symbol="BTCUSDT",
                    timeframe="1h",
                    compatible_regimes=["bull"],
                    estimator_family="dummyclassifier",
                )
            ],
        )
        router_manifest = RouterManifest(router_type="score_router", score_component_names=["regime_confidence"])
        manifest = build_registry_manifest(
            version_id="v1",
            symbol="BTCUSDT",
            initial_status="challenger",
            artifact_manifest="model_manifest.json",
            feature_columns=["feature"],
            regime_contracts={"trace": {"transition_count": 1}},
            specialist_library=specialist_library,
            router_manifest=router_manifest,
        )
        flattened = flatten_registry_record(
            manifest,
            current_status="challenger",
            version_dir=Path("registry/BTCUSDT/v1"),
        )

        self.assertIn("regime_contracts", manifest.to_dict())
        self.assertTrue(flattened["regime_contracts_present"])
        self.assertTrue(flattened["specialist_library_present"])
        self.assertTrue(flattened["router_manifest_present"])

    def test_local_registry_store_accepts_optional_phase_zero_payloads(self):
        specialist_library = SpecialistLibrarySnapshot(
            symbol="BTCUSDT",
            timeframe="1h",
            fallback_model_id="fallback_generalist",
            specialists=[
                SpecialistSpec(
                    model_id="specialist::bull",
                    symbol="BTCUSDT",
                    timeframe="1h",
                    compatible_regimes=["bull"],
                    estimator_family="dummyclassifier",
                )
            ],
        )
        router_manifest = RouterManifest(router_type="score_router", score_component_names=["regime_confidence"])

        with TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=Path(temp_dir) / "registry")
            version_id = store.register_version(
                _fit_dummy_classifier(1),
                symbol="BTCUSDT",
                feature_columns=["feature"],
                regime_contracts={"trace": {"transition_count": 1}},
                specialist_library=specialist_library,
                router_manifest=router_manifest,
                metadata={"stage": "test"},
            )
            manifest = store.read_version_manifest(version_id, symbol="BTCUSDT")
            versions = store.list_versions(symbol="BTCUSDT")

        self.assertIn("regime_contracts", manifest)
        self.assertIn("specialist_library", manifest)
        self.assertIn("router_manifest", manifest)
        self.assertTrue(versions[0]["regime_contracts_present"])
        self.assertTrue(versions[0]["specialist_library_present"])
        self.assertTrue(versions[0]["router_manifest_present"])