import math
import unittest

from core.automl import _build_evaluation_record, compute_objective_value


class ObjectiveScoreSeparationTest(unittest.TestCase):
    def test_evaluation_record_preserves_raw_score_when_gate_fails(self):
        training = {
            "avg_directional_accuracy": 0.49,
            "avg_accuracy": 0.49,
            "avg_log_loss": 0.61,
            "avg_calibration_error": 0.08,
        }
        backtest = {
            "sharpe_ratio": 0.4,
            "net_profit_pct": 0.03,
            "max_drawdown": -0.02,
            "total_trades": 0,
            "bar_count": 100,
        }

        record = _build_evaluation_record(
            training,
            backtest,
            "risk_adjusted_after_costs",
            {},
            evidence_class="outer_replay",
        )

        diagnostics = record["objective_diagnostics"]
        self.assertTrue(math.isfinite(record["raw_objective_value"]))
        self.assertEqual(record["raw_objective_value"], diagnostics["raw_score"])
        self.assertEqual(record["gated_objective_value"], diagnostics["final_score"])
        self.assertEqual(diagnostics["final_score"], float("-inf"))
        self.assertEqual(
            compute_objective_value("risk_adjusted_after_costs", training, backtest, {}),
            float("-inf"),
        )


if __name__ == "__main__":
    unittest.main()