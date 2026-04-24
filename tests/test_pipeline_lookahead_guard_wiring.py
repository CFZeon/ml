import unittest

from core.automl import _resolve_selection_policy
from core.promotion import (
    create_promotion_eligibility_report,
    finalize_promotion_eligibility_report,
    resolve_promotion_gate_mode,
    upsert_promotion_gate,
)


class PipelineLookaheadGuardWiringTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()