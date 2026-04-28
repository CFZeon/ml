import unittest

from core.automl import _evaluate_objective_gates
from core.pipeline import _resolve_backtest_significance_config


class ObjectiveGateConfidenceBoundsTest(unittest.TestCase):
    def test_trade_ready_runtime_significance_floor_is_enabled_by_default(self):
        config = _resolve_backtest_significance_config({"evaluation_mode": "trade_ready", "significance": False})

        self.assertTrue(config["enabled"])
        self.assertEqual(int(config["min_observations"]), 32)

    def test_trade_ready_research_override_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "backtest.research_only_override=true is only valid when backtest.evaluation_mode='research_only'",
        ):
            _resolve_backtest_significance_config(
                {"evaluation_mode": "trade_ready", "research_only_override": True, "significance": False}
            )

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

    def test_objective_gate_reports_explicit_low_sample_reason_when_significance_is_unavailable(self):
        training = {
            "avg_directional_accuracy": 0.61,
            "avg_log_loss": 0.42,
            "avg_calibration_error": 0.08,
        }
        backtest = {
            "total_trades": 48,
            "statistical_significance": {
                "enabled": False,
                "reason": "insufficient_observations",
                "observation_count": 24,
                "min_observations": 32,
            },
        }

        report = _evaluate_objective_gates(
            training,
            backtest,
            {
                "objective_gates": {
                    "require_statistical_significance": True,
                    "min_significance_observations": 32,
                    "min_trade_count": 20,
                }
            },
            "risk_adjusted_after_costs",
        )

        self.assertFalse(report["passed"])
        self.assertIn("statistical_significance", report["failed"])
        self.assertIn("significance_observation_count", report["failed"])
        self.assertIn("statistical_significance_insufficient_observations", report["reasons"])
        self.assertIn("statistical_significance_underpowered", report["reasons"])


if __name__ == "__main__":
    unittest.main()