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
        self.assertTrue(guardrails["approved"])
        self.assertFalse(guardrails["structural_retrain_recommended"])
        self.assertEqual(guardrails["recommended_action"], "maintenance_refresh")

    def test_model_ttl_expiry_can_force_retrain_without_drift_signals(self):
        reference_index = pd.date_range("2026-10-01", periods=300, freq="1h", tz="UTC")
        current_index = pd.date_range("2026-11-01", periods=240, freq="1h", tz="UTC")
        reference_features = pd.DataFrame({"alpha": 0.0, "beta": 0.0}, index=reference_index)
        current_features = pd.DataFrame({"alpha": 0.0, "beta": 0.0}, index=current_index)
        reference_predictions = pd.DataFrame({"p0": 0.5, "p1": 0.5}, index=reference_index)
        current_predictions = pd.DataFrame({"p0": 0.5, "p1": 0.5}, index=current_index)
        performance = pd.Series(np.zeros(len(current_index)), index=current_index)

        monitor = DriftMonitor(
            reference_features,
            reference_predictions,
            config={"min_samples": 200, "min_drift_signals": 2, "max_bars_between_retrain": 672},
        )
        report = monitor.check(
            current_features,
            current_predictions=current_predictions,
            current_performance=performance,
            bars_since_last_retrain=800,
        )
        guardrails = evaluate_drift_guardrails(
            report,
            {"min_samples": 200, "min_drift_signals": 2, "max_bars_between_retrain": 672},
        )

        self.assertFalse(report["feature_drift"])
        self.assertFalse(report["prediction_drift"])
        self.assertFalse(report["performance_drift"])
        self.assertTrue(report["model_ttl_expired"])
        self.assertFalse(report["recommendation"]["should_retrain"])
        self.assertEqual(report["recommendation"]["recommended_action"], "maintenance_refresh")
        self.assertTrue(report["recommendation"]["maintenance_refresh_recommended"])
        self.assertFalse(report["recommendation"]["ttl_evidence_sufficient"])
        self.assertFalse(report["recommendation"]["structural_retrain_recommended"])
        self.assertIn("model_ttl_expired", report["recommendation"]["reasons"])
        self.assertTrue(guardrails["approved"])
        self.assertTrue(guardrails["maintenance_only_approved"])
        self.assertFalse(guardrails["adaptive_approved"])
        self.assertEqual(guardrails["recommended_action"], "maintenance_refresh")
        self.assertTrue(guardrails["maintenance_refresh_recommended"])
        self.assertFalse(guardrails["ttl_evidence_sufficient"])
        self.assertIn("model_ttl_expired", guardrails["reasons"])
        self.assertIn("insufficient_ttl_drift_evidence", guardrails["reasons"])

    def test_ttl_expiry_does_not_escalate_adaptive_action_without_ttl_signal_floor(self):
        reference_index = pd.date_range("2026-10-01", periods=320, freq="1h", tz="UTC")
        current_index = pd.date_range("2026-11-01", periods=240, freq="1h", tz="UTC")
        reference_features = pd.DataFrame({"alpha": 0.0, "beta": 0.0}, index=reference_index)
        current_features = pd.DataFrame({"alpha": 0.0, "beta": 0.0}, index=current_index)
        reference_predictions = pd.DataFrame({"p0": 0.62, "p1": 0.38}, index=reference_index)
        current_predictions = pd.DataFrame({"p0": 0.94, "p1": 0.06}, index=current_index)

        monitor = DriftMonitor(
            reference_features,
            reference_predictions,
            config={
                "min_samples": 200,
                "min_drift_signals": 1,
                "min_ttl_drift_signals": 2,
                "confidence_ks_threshold": 0.05,
                "max_bars_between_retrain": 672,
            },
        )
        report = monitor.check(current_features, current_predictions=current_predictions, bars_since_last_retrain=800)
        guardrails = evaluate_drift_guardrails(
            report,
            {
                "min_samples": 200,
                "min_drift_signals": 1,
                "min_ttl_drift_signals": 2,
                "confidence_ks_threshold": 0.05,
                "max_bars_between_retrain": 672,
            },
        )

        self.assertTrue(report["score_drift"])
        self.assertTrue(report["model_ttl_expired"])
        self.assertFalse(report["recommendation"]["ttl_evidence_sufficient"])
        self.assertEqual(report["recommendation"]["recommended_action"], "maintenance_refresh")
        self.assertFalse(report["recommendation"]["recalibration_recommended"])
        self.assertTrue(guardrails["maintenance_only_approved"])
        self.assertFalse(guardrails["adaptive_approved"])
        self.assertEqual(guardrails["recommended_action"], "maintenance_refresh")

    def test_adwin_detector_or_fallback_flags_abrupt_mean_shift(self):
        detector = ADWINDetector(delta=0.002, fallback_window=20)
        updates = []
        for value in np.r_[np.full(80, 0.1), np.full(80, 2.5)]:
            updates.append(detector.update(value))

        self.assertTrue(any(update.get("drift_detected", False) for update in updates))

    def test_drift_monitor_restores_performance_detector_state_across_cycles(self):
        index = pd.date_range("2026-10-01", periods=80, freq="1h", tz="UTC")
        reference_features = pd.DataFrame({"alpha": 0.0, "beta": 0.0}, index=index)
        reference_predictions = pd.DataFrame({"p0": 0.5, "p1": 0.5}, index=index)
        baseline_performance = pd.Series(np.full(80, 0.1), index=index)
        shifted_index = pd.date_range("2026-10-05", periods=80, freq="1h", tz="UTC")
        shifted_features = pd.DataFrame({"alpha": 0.0, "beta": 0.0}, index=shifted_index)
        shifted_predictions = pd.DataFrame({"p0": 0.5, "p1": 0.5}, index=shifted_index)
        shifted_performance = pd.Series(np.full(80, 2.5), index=shifted_index)

        monitor = DriftMonitor(
            reference_features,
            reference_predictions,
            config={"min_samples": 1, "min_drift_signals": 1},
        )
        first_report = monitor.check(
            reference_features,
            current_predictions=reference_predictions,
            current_performance=baseline_performance,
            bars_since_last_retrain=800,
        )

        restored_monitor = DriftMonitor(
            reference_features,
            reference_predictions,
            config={"min_samples": 1, "min_drift_signals": 1},
            state=first_report["drift_monitor_state"],
        )
        restored_report = restored_monitor.check(
            shifted_features,
            current_predictions=shifted_predictions,
            current_performance=shifted_performance,
            bars_since_last_retrain=800,
        )
        fresh_monitor = DriftMonitor(
            reference_features,
            reference_predictions,
            config={"min_samples": 1, "min_drift_signals": 1},
        )
        fresh_report = fresh_monitor.check(
            shifted_features,
            current_predictions=shifted_predictions,
            current_performance=shifted_performance,
            bars_since_last_retrain=800,
        )

        self.assertFalse(first_report["performance_drift"])
        self.assertTrue(restored_report["performance_drift"])
        self.assertFalse(fresh_report["performance_drift"])
        self.assertEqual(restored_report["drift_monitor_state"]["performance_detector"]["history_length"], 160)

    def test_regime_distribution_shift_is_counted_as_drift_evidence(self):
        reference_index = pd.date_range("2026-10-01", periods=320, freq="1h", tz="UTC")
        current_index = pd.date_range("2026-11-01", periods=240, freq="1h", tz="UTC")
        reference_features = pd.DataFrame({"alpha": 0.0, "beta": 0.0}, index=reference_index)
        current_features = pd.DataFrame({"alpha": 0.0, "beta": 0.0}, index=current_index)
        reference_predictions = pd.DataFrame({"p0": 0.5, "p1": 0.5}, index=reference_index)
        current_predictions = pd.DataFrame({"p0": 0.5, "p1": 0.5}, index=current_index)
        reference_regimes = pd.Series(np.where(np.arange(len(reference_index)) < 260, 0, 1), index=reference_index)
        current_regimes = pd.Series(np.where(np.arange(len(current_index)) < 40, 0, 3), index=current_index)

        monitor = DriftMonitor(
            reference_features,
            reference_predictions,
            reference_regimes=reference_regimes,
            config={
                "min_samples": 200,
                "min_drift_signals": 1,
                "regime_psi_threshold": 0.15,
                "regime_total_variation_threshold": 0.20,
                "regime_transition_threshold": 0.15,
            },
        )
        report = monitor.check(
            current_features,
            current_predictions=current_predictions,
            current_regimes=current_regimes,
            bars_since_last_retrain=900,
        )

        self.assertTrue(report["regime_drift"])
        self.assertFalse(report["feature_drift"])
        self.assertFalse(report["prediction_drift"])
        self.assertGreater(float(report["regime_report"]["distribution"]["psi"]), 0.15)
        self.assertFalse(report["recommendation"]["should_retrain"])
        self.assertEqual(report["recommendation"]["recommended_action"], "drift_investigation")
        self.assertTrue(report["recommendation"]["drift_investigation_recommended"])

    def test_score_drift_is_reported_separately_from_action_drift(self):
        reference_index = pd.date_range("2026-10-01", periods=320, freq="1h", tz="UTC")
        current_index = pd.date_range("2026-11-01", periods=240, freq="1h", tz="UTC")
        reference_features = pd.DataFrame({"alpha": 0.0, "beta": 0.0}, index=reference_index)
        current_features = pd.DataFrame({"alpha": 0.0, "beta": 0.0}, index=current_index)
        reference_predictions = pd.DataFrame({"p0": 0.62, "p1": 0.38}, index=reference_index)
        current_predictions = pd.DataFrame({"p0": 0.94, "p1": 0.06}, index=current_index)

        monitor = DriftMonitor(
            reference_features,
            reference_predictions,
            config={"min_samples": 200, "min_drift_signals": 1, "confidence_psi_threshold": 0.05},
        )
        report = monitor.check(current_features, current_predictions=current_predictions, bars_since_last_retrain=900)

        self.assertTrue(report["score_drift"])
        self.assertFalse(report["action_drift"])
        self.assertTrue(report["prediction_drift"])
        self.assertGreater(float(report["score_report"]["confidence_ks_statistic"]), 0.05)
        self.assertEqual(report["action_report"]["reference_action_rate"], 1.0)
        self.assertEqual(report["action_report"]["current_action_rate"], 1.0)

    def test_action_drift_tracks_abstain_shift(self):
        reference_index = pd.date_range("2026-10-01", periods=320, freq="1h", tz="UTC")
        current_index = pd.date_range("2026-11-01", periods=240, freq="1h", tz="UTC")
        reference_features = pd.DataFrame({"alpha": 0.0, "beta": 0.0}, index=reference_index)
        current_features = pd.DataFrame({"alpha": 0.0, "beta": 0.0}, index=current_index)
        reference_predictions = pd.DataFrame({-1: 0.1, 0: 0.1, 1: 0.8}, index=reference_index)
        current_predictions = pd.DataFrame(
            {
                -1: np.r_[np.full(120, 0.1), np.full(120, 0.05)],
                0: np.r_[np.full(120, 0.1), np.full(120, 0.9)],
                1: np.r_[np.full(120, 0.8), np.full(120, 0.05)],
            },
            index=current_index,
        )

        monitor = DriftMonitor(
            reference_features,
            reference_predictions,
            config={"min_samples": 200, "min_drift_signals": 1, "abstain_rate_delta_threshold": 0.2},
        )
        report = monitor.check(current_features, current_predictions=current_predictions, bars_since_last_retrain=900)

        self.assertTrue(report["action_drift"])
        self.assertGreater(abs(float(report["action_report"]["abstain_rate_delta"])), 0.2)

    def test_benign_probability_mean_move_without_action_or_performance_change_does_not_force_retraining(self):
        reference_index = pd.date_range("2026-10-01", periods=320, freq="1h", tz="UTC")
        current_index = pd.date_range("2026-11-01", periods=240, freq="1h", tz="UTC")
        reference_features = pd.DataFrame({"alpha": 0.0, "beta": 0.0}, index=reference_index)
        current_features = pd.DataFrame({"alpha": 0.0, "beta": 0.0}, index=current_index)
        reference_predictions = pd.DataFrame({"p0": 0.62, "p1": 0.38}, index=reference_index)
        current_predictions = pd.DataFrame({"p0": 0.58, "p1": 0.42}, index=current_index)
        performance = pd.Series(np.zeros(len(current_index)), index=current_index)

        monitor = DriftMonitor(
            reference_features,
            reference_predictions,
            config={
                "min_samples": 200,
                "min_drift_signals": 2,
                "prediction_kl_threshold": 0.2,
                "confidence_psi_threshold": 0.2,
                "confidence_ks_threshold": 1.1,
                "margin_psi_threshold": 0.2,
                "margin_ks_threshold": 1.1,
                "max_bars_between_retrain": 5000,
            },
        )
        report = monitor.check(
            current_features,
            current_predictions=current_predictions,
            current_performance=performance,
            bars_since_last_retrain=900,
        )

        self.assertFalse(report["score_drift"])
        self.assertFalse(report["action_drift"])
        self.assertFalse(report["performance_drift"])
        self.assertFalse(report["recommendation"]["should_retrain"])

    def test_interval_and_trade_rate_normalize_cooldown_and_sample_floors(self):
        reference_index = pd.date_range("2026-10-01", periods=240, freq="4h", tz="UTC")
        current_index = pd.date_range("2026-11-01", periods=60, freq="4h", tz="UTC")
        reference_features = pd.DataFrame({"alpha": 0.0, "beta": 0.0}, index=reference_index)
        current_features = pd.DataFrame({"alpha": 3.0, "beta": 3.0}, index=current_index)
        reference_predictions = pd.DataFrame({"p0": 0.8, "p1": 0.2}, index=reference_index)
        current_predictions = pd.DataFrame({"p0": 0.2, "p1": 0.8}, index=current_index)

        policy = {
            "interval": "4h",
            "expected_trades_per_day": 2.0,
            "min_expected_trades": 24,
            "min_drift_signals": 1,
        }
        monitor = DriftMonitor(reference_features, reference_predictions, config=policy)
        report = monitor.check(
            current_features,
            current_predictions=current_predictions,
            bars_since_last_retrain=100,
        )
        guardrails = evaluate_drift_guardrails(report, policy)

        self.assertEqual(int(monitor.config["cooldown_bars"]), 126)
        self.assertEqual(int(monitor.config["max_bars_between_retrain"]), 168)
        self.assertEqual(int(monitor.config["trade_rate_min_samples"]), 72)
        self.assertEqual(int(report["min_samples"]), 72)
        self.assertEqual(int(guardrails["cooldown_bars"]), 126)
        self.assertFalse(guardrails["approved"])
        self.assertIn("minimum_samples_not_met", guardrails["reasons"])


if __name__ == "__main__":
    unittest.main()