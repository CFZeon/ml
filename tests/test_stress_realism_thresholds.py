import unittest

from core.promotion import evaluate_stress_realism_gate


class StressRealismThresholdsTest(unittest.TestCase):
    def test_missing_required_scenarios_fail_as_incomplete(self):
        result = evaluate_stress_realism_gate(
            {
                "evaluation_mode": "trade_ready",
                "stress_matrix": {
                    "configured": True,
                    "scenario_names": ["downtime"],
                    "worst_max_drawdown": -0.05,
                    "worst_fill_ratio": 0.9,
                    "worst_trade_count": 2,
                },
            },
            policy={"required_stress_scenarios": ["downtime", "halt"]},
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_class"], "scenarios_incomplete")
        self.assertEqual(result["reason"], "stress_scenarios_incomplete")

    def test_missing_stress_metrics_fail_as_incomplete(self):
        result = evaluate_stress_realism_gate(
            {
                "evaluation_mode": "trade_ready",
                "stress_matrix": {
                    "configured": True,
                    "scenario_names": ["downtime", "halt"],
                },
                "required_stress_scenarios": ["downtime", "halt"],
            }
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_class"], "scenarios_incomplete")
        self.assertEqual(result["reason"], "stress_metrics_incomplete")
        self.assertIn("worst_max_drawdown", result["missing_metrics"])


if __name__ == "__main__":
    unittest.main()