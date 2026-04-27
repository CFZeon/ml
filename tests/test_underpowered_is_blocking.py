import unittest

from core.automl import _evaluate_objective_gates


class UnderpoweredCertificationBlockingTest(unittest.TestCase):
    def test_effective_bet_underpowering_blocks_certification_objective_gates(self):
        training = {
            "avg_directional_accuracy": 0.61,
            "avg_log_loss": 0.42,
            "avg_calibration_error": 0.08,
        }
        backtest = {
            "total_trades": 64,
            "effective_bet_count": 12,
            "statistical_significance": {
                "enabled": False,
                "reason": "insufficient_effective_bets",
                "underpowered_reason": "insufficient_effective_bets",
                "observation_count": 128,
                "min_observations": 32,
                "effective_bet_count": 12,
                "min_effective_bets": 40,
            },
        }

        report = _evaluate_objective_gates(
            training,
            backtest,
            {
                "objective_gates": {
                    "require_statistical_significance": True,
                    "min_significance_observations": 32,
                    "min_trade_count": 40,
                    "min_effective_bet_count": 40,
                }
            },
            "risk_adjusted_after_costs",
        )

        self.assertFalse(report["passed"])
        self.assertIn("statistical_significance", report["failed"])
        self.assertIn("effective_bet_count", report["failed"])
        self.assertIn("statistical_significance_insufficient_effective_bets", report["reasons"])
        self.assertIn("effective_bet_count_underpowered", report["reasons"])


if __name__ == "__main__":
    unittest.main()