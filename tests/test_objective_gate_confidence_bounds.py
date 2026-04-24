import unittest

from core.automl import _evaluate_objective_gates


class ObjectiveGateConfidenceBoundsTest(unittest.TestCase):
    def test_sharpe_ci_lower_bound_gate_fails_when_evidence_is_weak(self):
        training = {
            "avg_directional_accuracy": 0.61,
            "avg_log_loss": 0.42,
            "avg_calibration_error": 0.08,
        }
        backtest = {
            "total_trades": 48,
            "statistical_significance": {
                "metrics": {
                    "sharpe_ratio": {"confidence_interval": {"lower": 0.12, "upper": 0.44}},
                }
            },
        }

        report = _evaluate_objective_gates(
            training,
            backtest,
            {"objective_gates": {"min_sharpe_ci_lower": 0.2}},
            "risk_adjusted_after_costs",
        )

        self.assertFalse(report["passed"])
        self.assertIn("sharpe_ci_lower", report["failed"])
        self.assertAlmostEqual(float(report["checks"]["sharpe_ci_lower"]["value"]), 0.12, places=12)

    def test_net_profit_ci_lower_bound_gate_fails_when_evidence_is_weak(self):
        training = {
            "avg_directional_accuracy": 0.61,
            "avg_log_loss": 0.42,
            "avg_calibration_error": 0.08,
        }
        backtest = {
            "total_trades": 48,
            "statistical_significance": {
                "metrics": {
                    "net_profit_pct": {"confidence_interval": {"lower": -0.01, "upper": 0.08}},
                }
            },
        }

        report = _evaluate_objective_gates(
            training,
            backtest,
            {"objective_gates": {"min_net_profit_pct_ci_lower": 0.0}},
            "risk_adjusted_after_costs",
        )

        self.assertFalse(report["passed"])
        self.assertIn("net_profit_pct_ci_lower", report["failed"])
        self.assertAlmostEqual(float(report["checks"]["net_profit_pct_ci_lower"]["value"]), -0.01, places=12)


if __name__ == "__main__":
    unittest.main()