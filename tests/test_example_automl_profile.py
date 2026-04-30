import unittest
from pathlib import Path

from example_automl import build_example_automl_config


class ExampleAutoMLProfileTests(unittest.TestCase):
    def test_default_example_keeps_locked_holdout_and_selection_gates_enabled(self):
        config = build_example_automl_config(
            automl_storage=Path(".cache") / "automl" / "example_automl_profile_test.db"
        )

        self.assertEqual(config["backtest"]["evaluation_mode"], "research_only")
        self.assertEqual(config["data"]["end"], "2024-07-01")
        significance = config["backtest"]["significance"]
        self.assertTrue(significance["enabled"])
        self.assertEqual(int(significance["min_observations"]), 64)
        self.assertEqual(int(significance["min_effective_bets"]), 64)
        self.assertEqual(float(config["backtest"]["slippage_rate"]), 0.0002)
        self.assertEqual(config["backtest"]["slippage_model"], "sqrt_impact")
        self.assertTrue(config["automl"]["locked_holdout_enabled"])
        self.assertEqual(float(config["automl"]["locked_holdout_fraction"]), 0.2)
        self.assertEqual(config["automl"]["policy_profile"], "legacy_permissive")
        self.assertTrue(config["automl"]["selection_policy"]["enabled"])
        self.assertEqual(
            config["automl"]["selection_policy"]["gate_modes"],
            {"locked_holdout": "blocking", "locked_holdout_gap": "blocking"},
        )
        self.assertTrue(config["automl"]["selection_policy"]["require_locked_holdout_pass"])
        self.assertEqual(config["example_runtime"]["mode"], "research_only_locked_holdout_power_gated")
        self.assertTrue(config["automl"]["overfitting_control"]["enabled"])
        self.assertTrue(config["automl"]["overfitting_control"]["post_selection"]["enabled"])
        self.assertTrue(config["automl"]["overfitting_control"]["post_selection"]["require_pass"])
        self.assertEqual(config["automl"]["overfitting_control"]["post_selection"]["pass_rule"], "spa")
        self.assertEqual(int(config["automl"]["overfitting_control"]["post_selection"]["bootstrap_samples"]), 300)
        objective_gates = config["automl"]["objective_gates"]
        self.assertTrue(objective_gates["enabled"])
        self.assertTrue(objective_gates["require_statistical_significance"])
        self.assertEqual(int(objective_gates["min_significance_observations"]), 64)
        self.assertEqual(int(objective_gates["min_effective_bet_count"]), 16)
        self.assertEqual(float(objective_gates["min_sharpe_ci_lower"]), 0.0)

    def test_research_demo_flag_restores_fast_unsafe_profile(self):
        config = build_example_automl_config(
            automl_storage=Path(".cache") / "automl" / "example_automl_research_demo_test.db",
            research_demo=True,
        )

        self.assertEqual(config["data"]["end"], "2024-05-01")
        self.assertNotIn("significance", config["backtest"])
        self.assertEqual(float(config["backtest"]["slippage_rate"]), 0.0002)
        self.assertEqual(config["backtest"]["slippage_model"], "sqrt_impact")
        self.assertFalse(config["automl"]["locked_holdout_enabled"])
        self.assertFalse(config["automl"]["selection_policy"]["enabled"])
        self.assertEqual(config["example_runtime"]["mode"], "research_demo")
        self.assertNotIn("objective_gates", config["automl"])
        self.assertFalse(config["automl"]["overfitting_control"]["post_selection"]["enabled"])