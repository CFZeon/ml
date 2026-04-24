import unittest

from core import ResearchPipeline
from core.context import _resolve_context_missing_policy
from core.pipeline import _resolve_pipeline_context_missing_policy


class ContextMissingPolicyDefaultsTest(unittest.TestCase):
    def test_default_missing_policy_preserves_unknown_state(self):
        policy = _resolve_context_missing_policy()

        self.assertEqual(policy["mode"], "preserve_missing")
        self.assertTrue(policy["add_indicator"])
        self.assertEqual(float(policy["max_unknown_rate"]), 0.0)

    def test_legacy_compat_flag_restores_zero_fill_default(self):
        pipeline = ResearchPipeline({"features": {}, "compat": {"legacy_missing_semantics": True}})

        policy = _resolve_pipeline_context_missing_policy(pipeline)

        self.assertEqual(policy["mode"], "zero_fill")
        self.assertFalse(policy["add_indicator"])
        self.assertEqual(float(policy["max_unknown_rate"]), 1.0)


if __name__ == "__main__":
    unittest.main()