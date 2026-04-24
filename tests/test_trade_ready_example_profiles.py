import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from example_trade_ready_automl import build_trade_ready_example_config, prepare_trade_ready_runtime_config
from example_utils import (
    build_futures_research_config,
    build_spot_research_config,
    build_trade_ready_automl_overrides,
    print_automl_summary,
)


class TradeReadyExampleProfileTests(unittest.TestCase):
    def test_spot_builder_enables_strict_context_guardrails(self):
        config = build_spot_research_config(
            symbol="BTCUSDT",
            interval="1h",
            start="2024-01-01",
            end="2024-02-01",
            indicators=[],
            context_symbols=["ETHUSDT"],
        )

        features = config["features"]
        data = config["data"]
        backtest = config["backtest"]
        self.assertEqual(data["duplicate_policy"], "fail")
        self.assertEqual(data["futures_context"]["recent_stats_availability_lag"], "period_close")
        self.assertEqual(features["context_missing_policy"]["mode"], "preserve_missing")
        self.assertTrue(features["context_missing_policy"]["add_indicator"])
        self.assertEqual(float(features["futures_context_ttl"]["max_unknown_rate"]), 0.0)
        self.assertEqual(features["cross_asset_context_ttl"]["max_age"], "2h")
        self.assertEqual(backtest["evaluation_mode"], "research_only")

    def test_futures_builder_enables_strict_context_and_funding_guardrails(self):
        config = build_futures_research_config(
            symbol="BTCUSDT",
            interval="1h",
            start="2024-01-01",
            end="2024-02-01",
            indicators=[],
            context_symbols=["ETHUSDT"],
        )

        features = config["features"]
        data = config["data"]
        backtest = config["backtest"]
        self.assertEqual(data["duplicate_policy"], "fail")
        self.assertEqual(data["futures_context"]["recent_stats_availability_lag"], "period_close")
        self.assertEqual(features["context_missing_policy"]["mode"], "preserve_missing")
        self.assertEqual(float(features["futures_context_ttl"]["max_unknown_rate"]), 0.0)
        self.assertEqual(features["cross_asset_context_ttl"]["max_age"], "2h")
        self.assertEqual(backtest["evaluation_mode"], "research_only")
        self.assertEqual(backtest["funding_missing_policy"]["mode"], "strict")
        self.assertEqual(backtest["funding_missing_policy"]["expected_interval"], "8h")

    def test_trade_ready_automl_profile_enables_binding_controls(self):
        overrides = build_trade_ready_automl_overrides(
            storage_path=Path(".cache") / "automl" / "trade_ready_test.db",
            study_name="trade_ready_profile_test",
        )

        automl = overrides["automl"]
        self.assertTrue(automl["enabled"])
        self.assertEqual(automl["objective"], "risk_adjusted_after_costs")
        self.assertEqual(float(automl["minimum_dsr_threshold"]), 0.3)
        self.assertTrue(automl["locked_holdout_enabled"])
        self.assertGreater(float(automl["locked_holdout_fraction"]), 0.0)

        selection_policy = automl["selection_policy"]
        self.assertTrue(selection_policy["enabled"])
        self.assertTrue(selection_policy["require_locked_holdout_pass"])
        self.assertGreaterEqual(int(selection_policy["min_validation_trade_count"]), 20)
        self.assertEqual(selection_policy["required_execution_mode"], "event_driven")
        self.assertEqual(selection_policy["required_stress_scenarios"], ["downtime", "stale_mark", "halt"])

        overfitting_control = automl["overfitting_control"]
        self.assertTrue(overfitting_control["enabled"])
        self.assertTrue(overfitting_control["deflated_sharpe"]["enabled"])
        self.assertTrue(overfitting_control["pbo"]["enabled"])
        self.assertTrue(overfitting_control["post_selection"]["enabled"])
        self.assertTrue(overfitting_control["post_selection"]["require_pass"])

        replication = automl["replication"]
        self.assertTrue(replication["enabled"])
        self.assertTrue(replication["include_symbol_cohorts"])
        self.assertTrue(replication["include_window_cohorts"])
        self.assertEqual(int(replication["alternate_window_count"]), 1)
        self.assertEqual(int(replication["min_coverage"]), 2)
        self.assertEqual(float(replication["min_pass_rate"]), 1.0)

        search_space = automl["search_space"]
        self.assertIn("features", search_space)
        self.assertIn("labels", search_space)
        self.assertIn("model", search_space)

    def test_print_automl_summary_surfaces_replication_outcome(self):
        buffer = StringIO()
        with redirect_stdout(buffer):
            print_automl_summary(
                {
                    "study_name": "trade_ready_profile_test",
                    "objective": "risk_adjusted_after_costs",
                    "selection_metric": "selection_value",
                    "selection_mode": "penalized_ranking",
                    "trial_count": 2,
                    "best_value": 0.1,
                    "best_params": {},
                    "replication": {
                        "enabled": True,
                        "promotion_pass": False,
                        "completed_cohort_count": 1,
                        "requested_cohort_count": 2,
                        "pass_rate": 0.5,
                        "min_pass_rate": 1.0,
                        "reasons": ["replication_pass_rate_below_minimum"],
                    },
                }
            )

        rendered = buffer.getvalue()
        self.assertIn("replication  : passed=False", rendered)
        self.assertIn("replication why: ['replication_pass_rate_below_minimum']", rendered)

    def test_trade_ready_example_requires_real_nautilus_backend(self):
        config = build_trade_ready_example_config(automl_storage=Path(".cache") / "automl" / "trade_ready_exec_test.db")

        backtest = config["backtest"]
        automl = config["automl"]
        execution_policy = backtest["execution_policy"]
        self.assertEqual(backtest["evaluation_mode"], "trade_ready")
        self.assertEqual(execution_policy["adapter"], "nautilus")
        self.assertFalse(bool(execution_policy.get("force_simulation", False)))
        self.assertEqual(int(automl["n_trials"]), 2)
        self.assertEqual(automl["search_space"]["model"]["type"]["choices"], ["gbm"])
        self.assertEqual(automl["search_space"]["labels"]["barrier_tie_break"]["choices"], ["sl"])

    def test_trade_ready_example_runtime_falls_back_to_research_only_without_nautilus(self):
        config = build_trade_ready_example_config(automl_storage=Path(".cache") / "automl" / "trade_ready_exec_test.db")

        runtime_config, using_research_fallback = prepare_trade_ready_runtime_config(config, nautilus_available=False)

        self.assertTrue(using_research_fallback)
        self.assertEqual(runtime_config["backtest"]["evaluation_mode"], "research_only")
        self.assertTrue(runtime_config["backtest"]["execution_policy"]["force_simulation"])
        self.assertFalse(runtime_config["automl"]["locked_holdout_enabled"])
        self.assertIsNone(runtime_config["automl"]["minimum_dsr_threshold"])
        self.assertFalse(runtime_config["automl"]["selection_policy"]["enabled"])
        self.assertFalse(runtime_config["automl"]["overfitting_control"]["enabled"])


if __name__ == "__main__":
    unittest.main()