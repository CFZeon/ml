import json
import tempfile
import unittest
from pathlib import Path

from core import (
    build_deployment_candidate_id,
    build_live_calibration_report,
    persist_deployment_candidate_artifacts,
)


class LiveCalibrationReportTest(unittest.TestCase):
    def test_live_calibration_report_passes_when_metrics_are_within_tolerance(self):
        report = build_live_calibration_report(
            certified_expectations={"modeled_slippage_bps": 2.0, "modeled_fill_ratio": 0.95},
            paper_metrics={
                "mode": "shadow_live",
                "duration_days": 42,
                "modeled_slippage_bps": 2.0,
                "realized_slippage_bps": 2.12,
                "modeled_fill_ratio": 0.95,
                "realized_fill_ratio": 0.90,
                "data_breaches": 0,
                "funding_breaches": 0,
                "kill_switch_triggers": 0,
            },
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["reasons"], [])
        self.assertEqual(report["mode"], "shadow_live")

    def test_live_calibration_report_fails_when_window_is_underpowered_or_degraded(self):
        report = build_live_calibration_report(
            certified_expectations={"modeled_slippage_bps": 2.0, "modeled_fill_ratio": 0.95},
            paper_metrics={
                "mode": "paper",
                "duration_days": 14,
                "modeled_slippage_bps": 2.0,
                "realized_slippage_bps": 2.8,
                "modeled_fill_ratio": 0.95,
                "realized_fill_ratio": 0.70,
                "data_breaches": 1,
                "funding_breaches": 0,
                "kill_switch_triggers": 1,
            },
        )

        self.assertFalse(report["passed"])
        self.assertIn("paper_window_underpowered", report["reasons"])
        self.assertIn("paper_data_breaches_present", report["reasons"])
        self.assertIn("paper_kill_switch_triggered", report["reasons"])
        self.assertIn("slippage_error_above_tolerance", report["reasons"])
        self.assertIn("fill_ratio_degradation_above_tolerance", report["reasons"])

    def test_persist_deployment_candidate_artifacts_writes_expected_files(self):
        candidate_id = build_deployment_candidate_id(
            experiment_id="exp-001",
            frozen_candidate_hash="candidate-abc",
            execution_profile={"venue": "binance", "adapter": "nautilus", "mode": "paper"},
        )
        report = build_live_calibration_report(
            certified_expectations={"modeled_slippage_bps": 2.0, "modeled_fill_ratio": 0.95},
            paper_metrics={
                "mode": "paper",
                "duration_days": 35,
                "modeled_slippage_bps": 2.0,
                "realized_slippage_bps": 2.05,
                "modeled_fill_ratio": 0.95,
                "realized_fill_ratio": 0.91,
                "data_breaches": 0,
                "funding_breaches": 0,
                "kill_switch_triggers": 0,
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = persist_deployment_candidate_artifacts(
                root_dir=temp_dir,
                deployment_candidate_id=candidate_id,
                paper_metrics=report["paper_metrics"],
                live_calibration_report=report,
                fill_quality={"fill_ratio": 0.91},
                readiness={"capital_release_eligible": False},
            )

            for key in ("paper_metrics", "live_calibration_report", "fill_quality", "readiness"):
                self.assertTrue(Path(paths[key]).exists(), key)
            payload = json.loads(Path(paths["readiness"]).read_text(encoding="utf-8"))
            self.assertFalse(payload["capital_release_eligible"])


if __name__ == "__main__":
    unittest.main()