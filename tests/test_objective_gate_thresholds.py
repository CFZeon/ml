import unittest

from core.automl import _resolve_objective_gates


class ObjectiveGateThresholdsTest(unittest.TestCase):
    def test_default_thresholds_are_hardened(self):
        gates = _resolve_objective_gates({}, "risk_adjusted_after_costs")

        self.assertTrue(gates["enabled"])
        self.assertAlmostEqual(float(gates["min_directional_accuracy"]), 0.52, places=12)
        self.assertAlmostEqual(float(gates["max_log_loss"]), 0.78, places=12)
        self.assertAlmostEqual(float(gates["max_calibration_error"]), 0.15, places=12)
        self.assertEqual(int(gates["min_trade_count"]), 30)
        self.assertFalse(gates["require_statistical_significance"])
        self.assertIsNone(gates["min_significance_observations"])
        self.assertIsNone(gates["min_sharpe_ci_lower"])
        self.assertIsNone(gates["min_net_profit_pct_ci_lower"])


if __name__ == "__main__":
    unittest.main()