import copy
import os
import unittest

import core.automl as automl_module
from core.automl import run_automl_study
from tests.test_automl_holdout_objective import (
    _AutoMLDummyPipeline,
    _BasePipelineStub,
    _build_market_frame,
    _make_storage_path,
)


class _StressRealismBindingPipeline(_AutoMLDummyPipeline):
    def run_step(self, name):
        result = super().run_step(name)
        if name == "train_models":
            training = copy.deepcopy(result)
            training["regime"] = {
                "mode": "fold_local",
                "folds": [],
                "coverage_summary": {
                    "status": "passed",
                    "promotion_pass": True,
                    "reasons": [],
                    "configured_thresholds": {"min_distinct_regimes": 2, "max_dominant_share": 0.8},
                    "folds": [],
                },
                "regime_aware": {
                    "enabled": True,
                    "strategy": "specialist",
                    "coverage_summary": {"status": "passed", "promotion_pass": True, "reasons": []},
                    "fallback_rows": 40,
                    "fallback_evidence_rows": 100,
                    "fallback_share": 0.4,
                    "unseen_regimes": ["2"],
                    "trained_regimes": ["0", "1"],
                },
            }
            self.state["training"] = training
            self.step_results[name] = training
            return training

        if name == "run_backtest":
            backtest = copy.deepcopy(result)
            backtest.update(
                {
                    "evaluation_mode": "trade_ready",
                    "execution_mode": "event_driven",
                    "promotion_execution_ready": True,
                    "required_stress_scenarios": ["unseen_regime_fallback"],
                    "required_stress_control_intents": ["unseen_regime_fallback_pressure"],
                    "stress_matrix": {
                        "configured": True,
                        "scenario_count": 1,
                        "scenario_names": ["unseen_regime_fallback"],
                        "control_intents": ["unseen_regime_fallback_pressure"],
                        "worst_net_profit_pct": -0.02,
                        "worst_sharpe_ratio": 0.4,
                        "worst_max_drawdown": -0.08,
                        "worst_fill_ratio": 0.9,
                        "worst_trade_count": 2,
                        "results": {
                            "unseen_regime_fallback": {
                                "net_profit_pct": -0.02,
                                "sharpe_ratio": 0.4,
                                "max_drawdown": -0.08,
                                "fill_ratio": 0.9,
                                "total_trades": 2,
                                "control_tags": ["unseen_regime_fallback_pressure"],
                                "scenario_report": {
                                    "scenario_name": "unseen_regime_fallback",
                                    "control_tags": ["unseen_regime_fallback_pressure"],
                                },
                            }
                        },
                    },
                }
            )
            self.state["backtest"] = backtest
            self.step_results[name] = backtest
            return backtest

        return result


class _RouterStabilityBindingPipeline(_AutoMLDummyPipeline):
    def run_step(self, name):
        result = super().run_step(name)
        if name == "run_backtest":
            backtest = copy.deepcopy(result)
            backtest.update(
                {
                    "evaluation_mode": "trade_ready",
                    "execution_mode": "event_driven",
                    "promotion_execution_ready": True,
                    "stress_matrix": {
                        "configured": True,
                        "scenario_count": 1,
                        "scenario_names": ["downtime"],
                        "worst_net_profit_pct": -0.01,
                        "worst_sharpe_ratio": 0.5,
                        "worst_max_drawdown": -0.04,
                        "worst_fill_ratio": 0.95,
                        "worst_trade_count": 2,
                    },
                    "required_stress_scenarios": ["downtime"],
                    "router_stability_report": {
                        "enabled": True,
                        "applicable": True,
                        "decision_count": 120,
                        "switch_opportunities": 119,
                        "switch_count": 48,
                        "switch_rate": 0.4034,
                        "blocked_switch_count": 4,
                        "blocked_switch_rate": 0.0336,
                        "blocked_switch_reasons": {"persistence_requirement_not_met": 4},
                        "control_flags": {"hysteresis": True, "persistence": True, "cooldown": True},
                        "configured_control_count": 3,
                        "switching_cost_estimate": 24.0,
                        "switching_cost_share_of_starting_equity": 0.0024,
                    },
                }
            )
            self.state["backtest"] = backtest
            self.step_results[name] = backtest
            return backtest

        return result


class AutoMLStressRealismBindingTest(unittest.TestCase):
    def test_post_selection_stress_gate_blocks_unseen_regime_fallback_breach(self):
        if automl_module.optuna is None:
            self.skipTest("optuna is not installed")

        storage_path = _make_storage_path()
        raw = _build_market_frame(96)
        pipeline = _BasePipelineStub(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "features": {"schema_version": "stress_realism_binding_v1"},
                "model": {"type": "gbm"},
                "automl": {
                    "enabled": True,
                    "n_trials": 1,
                    "objective": "accuracy_first",
                    "seed": 7,
                    "validation_fraction": 0.2,
                    "locked_holdout_enabled": False,
                    "enable_pruning": False,
                    "overfitting_control": {"enabled": False},
                    "storage": storage_path,
                    "study_name": "automl_stress_realism_binding_test",
                    "selection_policy": {
                        "enabled": True,
                        "min_validation_trade_count": 1,
                        "require_locked_holdout_pass": False,
                        "required_stress_scenarios": ["unseen_regime_fallback"],
                        "required_stress_control_intents": ["unseen_regime_fallback_pressure"],
                        "require_unseen_regime_fallback_bound": True,
                        "max_unseen_regime_fallback_share": 0.2,
                        "max_stress_drawdown": 0.2,
                        "min_stress_fill_ratio": 0.5,
                        "min_stress_trade_count": 1,
                        "gate_modes": {
                            "feature_portability": "disabled",
                            "feature_admission": "disabled",
                            "regime_stability": "disabled",
                            "regime_coverage": "disabled",
                            "cross_venue_integrity": "disabled",
                            "data_certification": "disabled",
                            "signal_decay": "disabled",
                            "locked_holdout": "disabled",
                            "locked_holdout_gap": "disabled",
                            "replication": "disabled",
                            "param_fragility": "disabled",
                            "lookahead_guard": "disabled",
                        },
                    },
                },
            },
            raw_data=raw,
            data=raw.copy(),
        )

        try:
            summary = run_automl_study(
                pipeline,
                pipeline_class=_StressRealismBindingPipeline,
                trial_step_classes=[],
            )
        finally:
            if os.path.exists(storage_path):
                os.remove(storage_path)

        self.assertTrue(summary["best_selection_policy"]["selection_policy"]["eligible"])
        self.assertFalse(summary["promotion_ready"])
        gate_report = summary["promotion_eligibility_report"]["gate_status"]["stress_realism"]
        self.assertFalse(gate_report["passed"])
        self.assertEqual(gate_report["reason"], "unseen_regime_fallback_share_above_limit")
        self.assertEqual(gate_report["details"]["regime_fallback"]["fallback_share"], 0.4)

    def test_post_selection_router_stability_gate_blocks_over_switching(self):
        if automl_module.optuna is None:
            self.skipTest("optuna is not installed")

        storage_path = _make_storage_path()
        raw = _build_market_frame(96)
        pipeline = _BasePipelineStub(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "features": {"schema_version": "router_stability_binding_v1"},
                "model": {"type": "gbm"},
                "automl": {
                    "enabled": True,
                    "n_trials": 1,
                    "objective": "accuracy_first",
                    "seed": 7,
                    "validation_fraction": 0.2,
                    "locked_holdout_enabled": False,
                    "enable_pruning": False,
                    "overfitting_control": {"enabled": False},
                    "storage": storage_path,
                    "study_name": "automl_router_stability_binding_test",
                    "selection_policy": {
                        "enabled": True,
                        "min_validation_trade_count": 1,
                        "require_locked_holdout_pass": False,
                        "required_stress_scenarios": ["downtime"],
                        "max_stress_drawdown": 0.2,
                        "min_stress_fill_ratio": 0.5,
                        "min_stress_trade_count": 1,
                        "min_router_decision_count": 25,
                        "max_router_switch_rate": 0.20,
                        "require_router_stability_controls": True,
                        "gate_modes": {
                            "feature_portability": "disabled",
                            "feature_admission": "disabled",
                            "regime_stability": "disabled",
                            "regime_coverage": "disabled",
                            "cross_venue_integrity": "disabled",
                            "data_certification": "disabled",
                            "signal_decay": "disabled",
                            "locked_holdout": "disabled",
                            "locked_holdout_gap": "disabled",
                            "replication": "disabled",
                            "param_fragility": "disabled",
                            "lookahead_guard": "disabled",
                            "execution_realism": "disabled",
                            "stress_realism": "disabled",
                        },
                    },
                },
            },
            raw_data=raw,
            data=raw.copy(),
        )

        try:
            summary = run_automl_study(
                pipeline,
                pipeline_class=_RouterStabilityBindingPipeline,
                trial_step_classes=[],
            )
        finally:
            if os.path.exists(storage_path):
                os.remove(storage_path)

        self.assertFalse(summary["promotion_ready"])
        gate_report = summary["promotion_eligibility_report"]["gate_status"]["router_stability"]
        self.assertFalse(gate_report["passed"])
        self.assertEqual(gate_report["reason"], "router_switch_rate_above_limit")
        self.assertGreater(float(gate_report["details"]["switch_rate"]), 0.20)


if __name__ == "__main__":
    unittest.main()