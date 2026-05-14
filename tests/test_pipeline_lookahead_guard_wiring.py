import unittest

from core.automl import _resolve_lookahead_guard_gate, _resolve_selection_policy
from core.pipeline import ResearchPipeline, _resolve_lookahead_guard_config, _run_pipeline_lookahead_guard
from core.promotion import (
    create_promotion_eligibility_report,
    finalize_promotion_eligibility_report,
    resolve_promotion_gate_mode,
    upsert_promotion_gate,
)


class PipelineLookaheadGuardWiringTest(unittest.TestCase):
    def test_trade_ready_lookahead_defaults_cover_full_causal_surface(self):
        pipeline = ResearchPipeline(
            {
                "features": {},
                "backtest": {"evaluation_mode": "trade_ready"},
            }
        )

        guard_config = _resolve_lookahead_guard_config(pipeline)

        self.assertEqual(guard_config["audit_scope"], "full_causal_surface")
        self.assertEqual(
            guard_config["step_names"],
            [
                "build_features",
                "detect_regimes",
                "build_labels",
                "align_data",
                "train_models",
                "generate_signals",
            ],
        )
        self.assertIn("regimes", guard_config["artifact_names"])
        self.assertIn("admissible_regimes", guard_config["artifact_names"])
        self.assertIn("labels", guard_config["artifact_names"])
        self.assertIn("continuous_signals", guard_config["artifact_names"])
        self.assertIn("execution_prices", guard_config["artifact_names"])
        self.assertEqual(guard_config["required_evidence_class"], "capital_facing")

    def test_missing_lookahead_guard_is_not_treated_as_pass(self):
        outcome = _resolve_lookahead_guard_gate({})

        self.assertFalse(outcome["passed"])
        self.assertEqual(outcome["reason"], "lookahead_guard_missing")

    def test_disabled_lookahead_guard_is_not_treated_as_pass(self):
        outcome = _resolve_lookahead_guard_gate(
            {"lookahead_guard": {"enabled": False, "status": "disabled"}}
        )

        self.assertFalse(outcome["passed"])
        self.assertEqual(outcome["reason"], "lookahead_guard_disabled")

    def test_lookahead_guard_failure_is_blocking_under_default_selection_policy(self):
        selection_policy = _resolve_selection_policy({})
        lookahead_guard = {
            "enabled": True,
            "mode": "advisory",
            "promotion_pass": False,
            "reasons": ["lookahead_guard_failed"],
            "biased_columns": ["future_close"],
            "checked_timestamps": 8,
        }

        report = create_promotion_eligibility_report()
        report = upsert_promotion_gate(
            report,
            group="post_selection",
            name="lookahead_guard",
            passed=bool(lookahead_guard["promotion_pass"]),
            mode=resolve_promotion_gate_mode(selection_policy, "lookahead_guard"),
            measured=lookahead_guard["checked_timestamps"],
            threshold={"mode": lookahead_guard["mode"]},
            reason="lookahead_guard_failed",
            details=lookahead_guard,
        )
        report = finalize_promotion_eligibility_report(report)

        self.assertFalse(report["promotion_ready"])
        self.assertIn("lookahead_guard_failed", report["blocking_failures"])

    def test_unavailable_lookahead_guard_surface_is_not_treated_as_pass(self):
        outcome = _resolve_lookahead_guard_gate(
            {
                "lookahead_guard": {
                    "enabled": True,
                    "status": "unavailable",
                    "promotion_pass": False,
                    "reasons": ["lookahead_guard_stage_unavailable:train_models"],
                }
            }
        )

        self.assertFalse(outcome["passed"])
        self.assertEqual(outcome["reason"], "lookahead_guard_stage_unavailable:train_models")

    def test_router_enabled_research_pipeline_requests_router_audit_artifacts(self):
        pipeline = ResearchPipeline(
            {
                "features": {},
                "regime": {"method": "hmm", "n_regimes": 2},
                "model": {
                    "type": "logistic",
                    "regime_aware": {"enabled": True, "strategy": "specialist"},
                },
                "router": {"enabled": True},
                "backtest": {"evaluation_mode": "research_only"},
            }
        )

        guard_config = _resolve_lookahead_guard_config(pipeline)

        self.assertEqual(guard_config["audit_scope"], "full_causal_surface")
        self.assertIn("router_decisions", guard_config["artifact_names"])
        self.assertIn("routed_signals", guard_config["artifact_names"])

    def test_lookahead_guard_bypasses_nested_replay_runs(self):
        pipeline = ResearchPipeline(
            {
                "features": {},
                "regime": {"method": "hmm", "n_regimes": 2},
                "backtest": {"evaluation_mode": "research_only"},
            }
        )
        pipeline.state["_lookahead_replay_active"] = True

        report = _run_pipeline_lookahead_guard(pipeline)

        self.assertEqual(report["status"], "replay_bypass")
        self.assertTrue(report["promotion_pass"])
        self.assertEqual(report["reasons"], ["lookahead_guard_replay_bypass"])


if __name__ == "__main__":
    unittest.main()