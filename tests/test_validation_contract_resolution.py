import unittest

from core.automl import _resolve_validation_contract
from example_utils import build_trade_ready_automl_overrides


class ValidationContractResolutionTest(unittest.TestCase):
    def test_trade_ready_profile_declares_multi_stage_validation_contract(self):
        profile = build_trade_ready_automl_overrides(
            study_name="validation_contract_test",
            storage_path=".cache/validation_contract_test.db",
        )

        contract = _resolve_validation_contract(
            {"model": {"cv_method": "cpcv"}},
            profile["automl"],
            holdout_enabled=True,
        )

        self.assertEqual(contract["search_ranker"], "cpcv")
        self.assertEqual(contract["contiguous_validation"], "walk_forward_replay")
        self.assertEqual(contract["locked_holdout"], "single_access_contiguous")
        self.assertEqual(contract["replication"], "required")

    def test_non_trade_ready_defaults_follow_declared_model_validation(self):
        contract = _resolve_validation_contract(
            {"model": {"cv_method": "walk_forward"}},
            {"locked_holdout_enabled": False, "replication": {"enabled": False}},
            holdout_enabled=False,
        )

        self.assertEqual(contract["search_ranker"], "walk_forward")
        self.assertEqual(contract["contiguous_validation"], "walk_forward")
        self.assertEqual(contract["locked_holdout"], "disabled")
        self.assertEqual(contract["replication"], "disabled")


if __name__ == "__main__":
    unittest.main()