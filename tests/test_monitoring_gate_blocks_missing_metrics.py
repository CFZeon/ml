import unittest

from core import build_monitoring_report


class MonitoringGateBlocksMissingMetricsTest(unittest.TestCase):
    def test_local_certification_gate_blocks_missing_required_metrics(self):
        report = build_monitoring_report(
            policy={
                "policy_profile": "local_certification",
                "required_components": ["raw_data_freshness", "l2_snapshot_age", "inference"],
            }
        )

        gate = report["monitoring_gate_report"]
        self.assertFalse(report["healthy"])
        self.assertFalse(report["promotion_pass"])
        self.assertFalse(gate["promotion_pass"])
        self.assertEqual(gate["missing_metrics"], ["raw_data_freshness", "l2_snapshot_age", "inference"])
        self.assertIn("raw_data_freshness_missing", gate["blocking_reasons"])
        self.assertIn("l2_snapshot_age_missing", gate["blocking_reasons"])
        self.assertIn("inference_missing", gate["blocking_reasons"])
        self.assertIn("raw_data_freshness_missing", report["reasons"])
        self.assertIn("l2_snapshot_age_missing", report["reasons"])
        self.assertIn("inference_missing", report["reasons"])


if __name__ == "__main__":
    unittest.main()