import unittest

from core.automl import _resolve_lookahead_guard_gate, _resolve_selection_policy
from core.pipeline import ResearchPipeline, _resolve_lookahead_guard_config
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
        self.assertIn("labels", guard_config["artifact_names"])
        self.assertIn("continuous_signals", guard_config["artifact_names"])
        self.assertIn("execution_prices", guard_config["artifact_names"])

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


if __name__ == "__main__":
    unittest.main()