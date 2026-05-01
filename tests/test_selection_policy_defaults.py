import unittest

from core.automl import _resolve_overfitting_control, _resolve_selection_policy


class SelectionPolicyDefaultsTest(unittest.TestCase):
    def test_hardened_default_policy_profile_is_binding(self):
        policy = _resolve_selection_policy({})

        self.assertEqual(policy["policy_profile"], "hardened_default")
        self.assertTrue(policy["require_locked_holdout_pass"])
        self.assertTrue(policy["require_fold_stability_pass"])
        self.assertEqual(policy["gate_modes"]["locked_holdout"], "blocking")
        self.assertEqual(policy["gate_modes"]["locked_holdout_gap"], "blocking")
        self.assertEqual(policy["gate_modes"]["replication"], "blocking")
        self.assertEqual(policy["gate_modes"]["execution_realism"], "blocking")
        self.assertEqual(policy["gate_modes"]["stress_realism"], "blocking")
        self.assertEqual(policy["gate_modes"]["regime_coverage"], "blocking")
        self.assertEqual(policy["gate_modes"]["param_fragility"], "blocking")
        self.assertEqual(policy["gate_modes"]["lookahead_guard"], "blocking")

    def test_hardened_default_post_selection_requires_pass(self):
        control = _resolve_overfitting_control({})

        self.assertEqual(control["policy_profile"], "hardened_default")
        self.assertTrue(control["post_selection"]["require_pass"])

    def test_legacy_permissive_profile_preserves_opt_in_behavior(self):
        policy = _resolve_selection_policy({"policy_profile": "legacy_permissive"})
        control = _resolve_overfitting_control({"policy_profile": "legacy_permissive"})

        self.assertEqual(policy["policy_profile"], "legacy_permissive")
        self.assertFalse(policy["require_locked_holdout_pass"])
        self.assertEqual(policy["gate_modes"], {})
        self.assertEqual(policy["deprecation_warning"], "legacy_permissive_policy_profile_deprecated")
        self.assertFalse(control["post_selection"]["require_pass"])
        self.assertEqual(control["deprecation_warning"], "legacy_permissive_policy_profile_deprecated")


if __name__ == "__main__":
    unittest.main()