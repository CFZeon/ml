import unittest

import pandas as pd
from sklearn.dummy import DummyClassifier

from core.regime_training import (
    RegimeAwareModelBundle,
    build_specialist_health_contracts,
    build_specialist_library_snapshot,
    build_specialist_specs_from_bundle,
)
from core.specialists import SpecialistLibrarySnapshot


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
