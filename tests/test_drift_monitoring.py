import unittest

import numpy as np
import pandas as pd

from core import ADWINDetector, DriftMonitor, evaluate_drift_guardrails


class DriftMonitoringTest(unittest.TestCase):
    def test_drift_alerts_do_not_approve_retrain_without_minimum_samples(self):
        reference_index = pd.date_range("2026-10-01", periods=300, freq="1h", tz="UTC")
        current_index = pd.date_range("2026-11-01", periods=120, freq="1h", tz="UTC")
        reference_features = pd.DataFrame(
            {
                "alpha": np.random.default_rng(1).normal(0.0, 1.0, len(reference_index)),
                "beta": np.random.default_rng(2).normal(0.0, 1.0, len(reference_index)),
            },
            index=reference_index,
        )
        current_features = pd.DataFrame(
            {
                "alpha": np.random.default_rng(3).normal(3.0, 1.0, len(current_index)),
                "beta": np.random.default_rng(4).normal(3.0, 1.0, len(current_index)),
            },
            index=current_index,
        )
        reference_predictions = pd.DataFrame({"p0": 0.8, "p1": 0.2}, index=reference_index)
        current_predictions = pd.DataFrame({"p0": 0.2, "p1": 0.8}, index=current_index)
        performance = pd.Series(np.r_[np.full(60, 0.6), np.full(60, -0.4)], index=current_index)

        monitor = DriftMonitor(
            reference_features,
            reference_predictions,
            config={"min_samples": 200, "min_drift_signals": 2, "psi_feature_share_threshold": 0.3},
        )
        report = monitor.check(
            current_features,
            current_predictions=current_predictions,
            current_performance=performance,
            bars_since_last_retrain=800,
        )
        guardrails = evaluate_drift_guardrails(report, {"min_samples": 200, "min_drift_signals": 2})

        self.assertTrue(report["feature_drift"])
        self.assertTrue(report["prediction_drift"])
        self.assertFalse(report["recommendation"]["should_retrain"])
        self.assertIn("minimum_samples_not_met", report["recommendation"]["reasons"])
        self.assertFalse(guardrails["approved"])

    def test_adwin_detector_or_fallback_flags_abrupt_mean_shift(self):
        detector = ADWINDetector(delta=0.002, fallback_window=20)
        updates = []
        for value in np.r_[np.full(80, 0.1), np.full(80, 2.5)]:
            updates.append(detector.update(value))

        self.assertTrue(any(update.get("drift_detected", False) for update in updates))


if __name__ == "__main__":
    unittest.main()