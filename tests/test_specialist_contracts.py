import unittest

import pandas as pd
from sklearn.dummy import DummyClassifier

from core.regime_training import (
    RegimeAwareModelBundle,
    build_specialist_health_contracts,
    build_specialist_library_snapshot,
    build_specialist_specs_from_bundle,
)
from core.specialists import SpecialistLibrarySnapshot, SpecialistSpec, apply_specialist_health_update


def _fit_dummy_classifier(label):
    features = pd.DataFrame({"feature": [0.0, 1.0, 2.0]})
    targets = pd.Series([label, label, label])
    model = DummyClassifier(strategy="prior")
    model.fit(features, targets)
    return model


class SpecialistContractsTest(unittest.TestCase):
    def test_specialist_snapshot_helpers_capture_bundle_layout(self):
        bundle = RegimeAwareModelBundle(
            strategy="specialist",
            fallback_model=_fit_dummy_classifier(0),
            specialist_models={
                "bull": _fit_dummy_classifier(1),
                "risk_off": _fit_dummy_classifier(-1),
            },
            feature_columns=["feature"],
            regime_column="regime",
        )
        training_report = {
            "strategy": "specialist",
            "coverage_summary": {
                "status": "passed",
                "promotion_pass": True,
                "regime_distribution": {"bull": 12, "risk_off": 8},
            },
            "trained_rows_by_regime": {"bull": 12, "risk_off": 8},
            "trained_regimes": ["bull", "risk_off"],
            "skipped_regimes": {"flat": "insufficient_samples"},
        }

        specs = build_specialist_specs_from_bundle(
            bundle,
            training_report,
            symbol="BTCUSDT",
            timeframe="1h",
        )
        health = build_specialist_health_contracts(bundle, training_report)
        snapshot = build_specialist_library_snapshot(
            bundle,
            training_report,
            symbol="BTCUSDT",
            timeframe="1h",
        )
        roundtrip = SpecialistLibrarySnapshot.from_dict(snapshot.to_dict())

        self.assertEqual(len(specs), 3)
        self.assertEqual(specs[0].symbol, "BTCUSDT")
        self.assertTrue(any(spec.model_id == "specialist::bull" for spec in specs))
        self.assertTrue(any(item.model_id == "skipped::flat" for item in health))
        self.assertEqual(snapshot.fallback_model_id, "fallback_generalist")
        self.assertEqual(len(snapshot.performance_slices), 2)
        self.assertEqual(roundtrip.symbol, "BTCUSDT")
        self.assertEqual(roundtrip.timeframe, "1h")
        self.assertEqual(len(roundtrip.specialists), len(snapshot.specialists))
        specialist_health = next(item for item in health if item.model_id == "specialist::bull")
        fallback_health = next(item for item in health if item.model_id == "fallback_generalist")
        self.assertFalse(specialist_health.metadata["health_binding_resolved"])
        self.assertEqual(specialist_health.metadata["health_state"], "unknown")
        self.assertTrue(fallback_health.metadata["health_binding_resolved"])

    def test_specialist_health_updates_preserve_selection_contract(self):
        snapshot = SpecialistLibrarySnapshot(
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

        updated = apply_specialist_health_update(
            snapshot,
            {
                "recorded_at": "2026-05-09T00:00:00+00:00",
                "source": "specialist_monitoring",
                "health": [
                    {
                        "model_id": "specialist::bull",
                        "compatible_regimes": ["bull"],
                        "stability_score": 0.72,
                        "decay_score": 0.14,
                        "failure_flags": [],
                    }
                ],
                "performance_slices": [
                    {
                        "model_id": "specialist::bull",
                        "regime_label": "bull",
                        "split_role": "monitoring_window",
                        "row_count": 24,
                        "metric_summary": {"f1_macro": 0.57},
                    }
                ],
            },
        )

        self.assertEqual(updated.metadata["selection_contract"]["candidate_model_ids"], ["fallback_generalist", "specialist::bull"])
        self.assertEqual(updated.metadata["health_history"]["update_count"], 1)
        self.assertEqual(updated.health[0].model_id, "specialist::bull")
        self.assertEqual(len(updated.performance_slices), 1)
