import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from example_trade_ready_automl import _build_trade_ready_example_config, build_trade_ready_example_config, prepare_trade_ready_runtime_config
from example_utils import (
    build_futures_research_config,
    build_spot_research_config,
    build_trade_ready_runtime_overrides,
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
        data_quality = config["data_quality"]
        self.assertEqual(data["duplicate_policy"], "fail")
        self.assertEqual(data["futures_context"]["recent_stats_availability_lag"], "period_close")
        self.assertEqual(features["context_missing_policy"]["mode"], "preserve_missing")
        self.assertTrue(features["context_missing_policy"]["add_indicator"])
        self.assertEqual(float(features["futures_context_ttl"]["max_unknown_rate"]), 0.0)
        self.assertEqual(features["cross_asset_context_ttl"]["max_age"], "2h")
        self.assertEqual(backtest["evaluation_mode"], "research_only")
        self.assertTrue(data_quality["exclude_flagged_quarantine_rows_from_modeling"])
        self.assertEqual(data_quality["actions"]["return_spike"], "null")

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
        data_quality = config["data_quality"]
        self.assertEqual(data["duplicate_policy"], "fail")
        self.assertEqual(data["futures_context"]["recent_stats_availability_lag"], "period_close")
        self.assertEqual(features["context_missing_policy"]["mode"], "preserve_missing")
        self.assertEqual(float(features["futures_context_ttl"]["max_unknown_rate"]), 0.0)
        self.assertEqual(features["cross_asset_context_ttl"]["max_age"], "2h")
        self.assertEqual(backtest["evaluation_mode"], "research_only")
        self.assertEqual(backtest["funding_missing_policy"]["mode"], "strict")
        self.assertEqual(backtest["funding_missing_policy"]["expected_interval"], "8h")
        self.assertTrue(data_quality["exclude_flagged_quarantine_rows_from_modeling"])
        self.assertEqual(data_quality["actions"]["quote_volume_inconsistency"], "null")

    def test_trade_ready_automl_profile_enables_binding_controls(self):
        overrides = build_trade_ready_automl_overrides(
            storage_path=Path(".cache") / "automl" / "trade_ready_test.db",
            study_name="trade_ready_profile_test",
        )

        automl = overrides["automl"]
        self.assertTrue(automl["enabled"])
        self.assertEqual(automl["objective"], "risk_adjusted_after_costs")
        self.assertEqual(float(automl["minimum_dsr_threshold"]), 0.4)
        self.assertTrue(automl["locked_holdout_enabled"])
        self.assertEqual(float(automl["locked_holdout_fraction"]), 0.25)
        self.assertEqual(automl["trade_ready_profile"]["name"], "certification")
        self.assertFalse(automl["trade_ready_profile"]["reduced_power"])

        selection_policy = automl["selection_policy"]
        self.assertTrue(selection_policy["enabled"])
        self.assertTrue(selection_policy["require_locked_holdout_pass"])
        self.assertGreaterEqual(int(selection_policy["min_validation_trade_count"]), 40)
        self.assertEqual(selection_policy["required_execution_mode"], "event_driven")
        self.assertEqual(selection_policy["required_stress_scenarios"], ["downtime", "stale_mark", "halt"])

        overfitting_control = automl["overfitting_control"]
        self.assertTrue(overfitting_control["enabled"])
        self.assertTrue(overfitting_control["deflated_sharpe"]["enabled"])
        self.assertEqual(int(overfitting_control["deflated_sharpe"]["min_track_record_length"]), 20)
        self.assertTrue(overfitting_control["pbo"]["enabled"])
        self.assertEqual(int(overfitting_control["pbo"]["min_block_size"]), 8)
        self.assertTrue(overfitting_control["post_selection"]["enabled"])
        self.assertTrue(overfitting_control["post_selection"]["require_pass"])
        self.assertEqual(int(overfitting_control["post_selection"]["bootstrap_samples"]), 1000)

        replication = automl["replication"]
        self.assertTrue(replication["enabled"])
        self.assertTrue(replication["include_symbol_cohorts"])
        self.assertTrue(replication["include_window_cohorts"])
        self.assertEqual(int(replication["alternate_window_count"]), 2)
        self.assertEqual(int(replication["min_coverage"]), 3)
        self.assertEqual(float(replication["min_pass_rate"]), 1.0)

        portability_contract = automl["portability_contract"]
        self.assertTrue(portability_contract["enabled"])
        self.assertEqual(portability_contract["accepted_kinds"], ["symbol", "period"])
        self.assertTrue(portability_contract["require_frozen_universe"])

        objective_gates = automl["objective_gates"]
        self.assertEqual(int(objective_gates["min_trade_count"]), 40)
        self.assertEqual(int(objective_gates["min_effective_bet_count"]), 40)
        self.assertTrue(objective_gates["require_statistical_significance"])
        self.assertEqual(int(objective_gates["min_significance_observations"]), 64)
        self.assertEqual(float(objective_gates["min_sharpe_ci_lower"]), 0.0)
        self.assertEqual(int(automl["trade_ready_profile"]["min_significance_observations"]), 64)

        search_space = automl["search_space"]
        self.assertEqual(search_space["features"]["lags"]["choices"], ["1,3,6"])
        self.assertEqual(search_space["labels"]["barrier_tie_break"]["choices"], ["sl"])
        self.assertEqual(search_space["regime"]["n_regimes"]["choices"], [2])
        self.assertEqual(search_space["model"]["type"]["choices"], ["gbm", "logistic"])
        self.assertEqual(automl["validation_contract"]["search_ranker"], "cpcv")
        self.assertEqual(automl["validation_contract"]["contiguous_validation"], "walk_forward_replay")

    def test_trade_ready_runtime_overrides_share_fail_closed_data_defaults(self):
        overrides = build_trade_ready_runtime_overrides(market="spot")

        self.assertEqual(overrides["data"]["gap_policy"], "fail")
        self.assertEqual(overrides["data"]["duplicate_policy"], "fail")
        self.assertTrue(overrides["data_quality"]["block_on_quarantine"])
        self.assertTrue(overrides["data_quality"]["exclude_flagged_quarantine_rows_from_modeling"])
        self.assertTrue(overrides["signals"]["require_paper_verification_for_kelly"])
        self.assertTrue(overrides["signals"]["require_live_calibration_for_kelly"])
        self.assertAlmostEqual(float(overrides["signals"]["uncalibrated_kelly_fraction_cap"]), 0.25, places=12)
        self.assertEqual(overrides["backtest"]["evaluation_mode"], "trade_ready")
        self.assertNotIn("funding_missing_policy", overrides["backtest"])

    def test_trade_ready_runtime_overrides_enable_strict_funding_for_futures(self):
        overrides = build_trade_ready_runtime_overrides(market="um_futures")

        funding_policy = overrides["backtest"]["funding_missing_policy"]
        self.assertTrue(overrides["backtest"]["apply_funding"])
        self.assertEqual(funding_policy["mode"], "strict")
        self.assertEqual(funding_policy["expected_interval"], "8h")
        self.assertEqual(float(funding_policy["max_gap_multiplier"]), 1.25)

    def test_trade_ready_smoke_profile_declares_reduced_power(self):
        overrides = build_trade_ready_automl_overrides(
            storage_path=Path(".cache") / "automl" / "trade_ready_smoke_test.db",
            study_name="trade_ready_smoke_profile_test",
            profile="smoke",
        )

        automl = overrides["automl"]
        self.assertEqual(automl["trade_ready_profile"]["name"], "smoke")
        self.assertTrue(automl["trade_ready_profile"]["reduced_power"])
        self.assertEqual(int(automl["n_trials"]), 4)
        self.assertEqual(int(automl["selection_policy"]["min_validation_trade_count"]), 20)
        self.assertEqual(int(automl["replication"]["alternate_window_count"]), 1)
        self.assertTrue(automl["portability_contract"]["enabled"])
        self.assertEqual(int(automl["objective_gates"]["min_trade_count"]), 20)
        self.assertEqual(int(automl["objective_gates"]["min_effective_bet_count"]), 20)
        self.assertTrue(automl["objective_gates"]["require_statistical_significance"])
        self.assertEqual(int(automl["objective_gates"]["min_significance_observations"]), 32)
        self.assertEqual(int(automl["trade_ready_profile"]["min_significance_observations"]), 32)

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
                    "oos_evidence": {
                        "class": "partial_oos",
                        "evidence_stack_complete": False,
                        "blocking_reasons": ["oos_control_incomplete:replication"],
                    },
                    "capital_evidence_contract": {
                        "requested_mode": "trade_ready",
                        "effective_mode": "trade_ready",
                        "capital_path_eligible": False,
                        "blocking_reasons": ["reduced_power_profile_not_capital_eligible"],
                    },
                    "replication": {
                        "enabled": True,
                        "promotion_pass": False,
                        "completed_cohort_count": 1,
                        "requested_cohort_count": 2,
                        "pass_rate": 0.5,
                        "min_pass_rate": 1.0,
                        "reasons": ["replication_pass_rate_below_minimum"],
                    },
                    "best_backtest": {
                        "funding_coverage_status": "fallback",
                        "funding_coverage_report": {
                            "coverage_status": "fallback",
                            "missing_event_count": 1,
                            "promotion_pass": True,
                            "coverage_reason": "missing_funding_events",
                            "fallback_assumption": "zero_fill_missing_funding_events",
                        },
                        "execution_evidence": {
                            "class": "research_surrogate",
                            "execution_mode": "conservative_bar_surrogate",
                            "promotion_execution_ready": False,
                            "blocking_reasons": ["execution_backend_not_event_driven", "bar_surrogate_only"],
                        },
                        "operational_monitoring": {
                            "healthy": False,
                            "policy": {"policy_profile": "local_certification"},
                            "reasons": ["inference_missing"],
                            "monitoring_gate_report": {
                                "promotion_pass": False,
                                "missing_metrics": ["inference"],
                                "blocking_reasons": ["inference_missing"],
                            },
                        },
                    },
                }
            )

        rendered = buffer.getvalue()
        self.assertIn("oos evidence : class=partial_oos  complete=False", rendered)
        self.assertIn("oos why      : ['oos_control_incomplete:replication']", rendered)
        self.assertIn("capital mode : requested=trade_ready  effective=trade_ready  eligible=False", rendered)
        self.assertIn("capital why  : ['reduced_power_profile_not_capital_eligible']", rendered)
        self.assertIn("execution    : class=research_surrogate  mode=conservative_bar_surrogate  ready=False", rendered)
        self.assertIn("execution why: ['execution_backend_not_event_driven', 'bar_surrogate_only']", rendered)
        self.assertIn("funding cov  : status=fallback  missing=1  pass=True", rendered)
        self.assertIn("funding why  : ['missing_funding_events', 'zero_fill_missing_funding_events']", rendered)
        self.assertIn("monitoring   : healthy=False  profile=local_certification", rendered)
        self.assertIn("op envelope : pass=False  missing=['inference']", rendered)
        self.assertIn("envelope why: ['inference_missing']", rendered)
        self.assertIn("replication  : passed=False", rendered)
        self.assertIn("replication why: ['replication_pass_rate_below_minimum']", rendered)

    def test_print_automl_summary_surfaces_objective_gate_reasons(self):
        buffer = StringIO()
        with redirect_stdout(buffer):
            print_automl_summary(
                {
                    "study_name": "trade_ready_profile_test",
                    "objective": "risk_adjusted_after_costs",
                    "selection_metric": "selection_value",
                    "selection_mode": "penalized_ranking",
                    "trial_count": 1,
                    "best_value": 0.1,
                    "best_params": {},
                    "best_objective_diagnostics": {
                        "raw_score": 0.1,
                        "classification_gates": {
                            "enabled": True,
                            "passed": False,
                            "failed": ["statistical_significance", "significance_observation_count"],
                            "reasons": [
                                "statistical_significance_insufficient_observations",
                                "statistical_significance_underpowered",
                            ],
                        },
                        "components": {},
                    },
                }
            )

        rendered = buffer.getvalue()
        self.assertIn("objective gate: passed=False", rendered)
        self.assertIn("objective why: ['statistical_significance_insufficient_observations', 'statistical_significance_underpowered']", rendered)

    def test_trade_ready_example_requires_real_nautilus_backend(self):
        config = build_trade_ready_example_config(automl_storage=Path(".cache") / "automl" / "trade_ready_exec_test.db")

        data = config["data"]
        data_quality = config["data_quality"]
        reference_data = config["reference_data"]
        data_certification = config["data_certification"]
        backtest = config["backtest"]
        automl = config["automl"]
        execution_policy = backtest["execution_policy"]
        self.assertEqual(data["gap_policy"], "fail")
        self.assertTrue(data_quality["block_on_quarantine"])
        self.assertTrue(reference_data["enabled"])
        self.assertEqual(reference_data["spot"]["partial_coverage_mode"], "blocking")
        self.assertEqual(reference_data["spot"]["divergence_mode"], "blocking")
        self.assertTrue(data_certification["enabled"])
        self.assertTrue(data_certification["require_reference_validation"])
        self.assertEqual(backtest["evaluation_mode"], "trade_ready")
        self.assertEqual(execution_policy["adapter"], "nautilus")
        self.assertFalse(bool(execution_policy.get("force_simulation", False)))
        self.assertEqual(automl["trade_ready_profile"]["name"], "certification")
        self.assertFalse(automl["trade_ready_profile"]["reduced_power"])
        self.assertEqual(int(automl["n_trials"]), 12)
        self.assertEqual(int(automl["trade_ready_profile"]["min_significance_observations"]), 64)
        self.assertEqual(int(backtest["significance"]["min_observations"]), 64)
        self.assertEqual(automl["search_space"]["model"]["type"]["choices"], ["gbm", "logistic"])
        self.assertEqual(automl["search_space"]["labels"]["barrier_tie_break"]["choices"], ["sl"])
        self.assertEqual(automl["search_space"]["features"]["lags"]["choices"], ["1,3,6"])

    def test_trade_ready_example_smoke_profile_declares_reduced_power(self):
        config = _build_trade_ready_example_config(
            automl_storage=Path(".cache") / "automl" / "trade_ready_smoke_profile.db",
            power_profile="smoke",
        )

        automl = config["automl"]
        self.assertEqual(automl["trade_ready_profile"]["name"], "smoke")
        self.assertTrue(automl["trade_ready_profile"]["reduced_power"])
        self.assertEqual(int(automl["n_trials"]), 4)
        self.assertEqual(int(automl["trade_ready_profile"]["min_significance_observations"]), 32)
        self.assertEqual(int(config["backtest"]["significance"]["min_observations"]), 32)
        self.assertEqual(config["data"]["end"], "2024-05-01")

    def test_print_automl_summary_surfaces_data_certification_outcome(self):
        buffer = StringIO()
        with redirect_stdout(buffer):
            print_automl_summary(
                {
                    "study_name": "trade_ready_profile_test",
                    "objective": "risk_adjusted_after_costs",
                    "selection_metric": "selection_value",
                    "selection_mode": "penalized_ranking",
                    "trial_count": 1,
                    "best_value": 0.1,
                    "best_params": {},
                    "best_training": {
                        "avg_directional_accuracy": 0.54,
                        "avg_accuracy": 0.52,
                        "avg_log_loss": 0.7,
                        "data_certification": {
                            "promotion_pass": False,
                            "mode": "blocking",
                            "reasons": ["reference_validation_unconfigured"],
                            "summary": {"failed_components": ["reference_integrity"]},
                            "components": {
                                "market_integrity": {"promotion_pass": True},
                                "data_quality": {"promotion_pass": True},
                                "context_ttl": {"promotion_pass": True},
                                "reference_integrity": {"promotion_pass": False},
                            },
                        },
                    },
                }
            )

        rendered = buffer.getvalue()
        self.assertIn("data cert", rendered)
        self.assertIn("reference_integrity", rendered)
        self.assertIn("reference_validation_unconfigured", rendered)

    def test_trade_ready_example_runtime_fails_closed_without_nautilus(self):
        config = build_trade_ready_example_config(automl_storage=Path(".cache") / "automl" / "trade_ready_exec_test.db")

        with self.assertRaisesRegex(RuntimeError, "Trade-ready certification requires a real Nautilus backend") as ctx:
            prepare_trade_ready_runtime_config(config, nautilus_available=False)

        self.assertIn("example_automl.py", str(ctx.exception))

    def test_trade_ready_example_smoke_profile_fails_closed_without_nautilus(self):
        config = _build_trade_ready_example_config(
            automl_storage=Path(".cache") / "automl" / "trade_ready_smoke_runtime_test.db",
            power_profile="smoke",
        )

        with self.assertRaisesRegex(RuntimeError, "Trade-ready certification requires a real Nautilus backend"):
            prepare_trade_ready_runtime_config(config, nautilus_available=False)


if __name__ == "__main__":
    unittest.main()