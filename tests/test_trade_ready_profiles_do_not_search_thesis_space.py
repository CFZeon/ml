import unittest

from core.automl import _classify_search_space, _find_varying_thesis_paths, _validate_trade_ready_search_space
from example_utils import build_trade_ready_automl_overrides


class TradeReadyThesisFreezeTest(unittest.TestCase):
    def test_certification_profile_freezes_thesis_space(self):
        overrides = build_trade_ready_automl_overrides(
            storage_path=".cache/trade_ready_thesis_freeze.db",
            study_name="trade_ready_thesis_freeze_test",
        )

        search_space = overrides["automl"]["search_space"]
        varying_paths = _find_varying_thesis_paths(_classify_search_space(search_space))

        self.assertEqual(varying_paths, [])

    def test_certification_profile_rejects_reopened_thesis_space(self):
        overrides = build_trade_ready_automl_overrides(
            storage_path=".cache/trade_ready_thesis_reopen.db",
            study_name="trade_ready_thesis_reopen_test",
        )
        search_space = overrides["automl"]["search_space"]
        search_space["labels"]["barrier_tie_break"]["choices"] = ["sl", "pt"]

        with self.assertRaisesRegex(ValueError, "cannot search thesis_space parameters"):
            _validate_trade_ready_search_space(search_space, overrides["automl"])

    def test_smoke_profile_can_still_vary_thesis_space(self):
        overrides = build_trade_ready_automl_overrides(
            storage_path=".cache/trade_ready_smoke_thesis.db",
            study_name="trade_ready_smoke_thesis_test",
            profile="smoke",
        )

        search_space = overrides["automl"]["search_space"]
        varying_paths = _find_varying_thesis_paths(_classify_search_space(search_space))

        self.assertIn("labels.barrier_tie_break", varying_paths)
        self.assertIn("features.lags", varying_paths)

    def test_bundle_search_is_treated_as_thesis_space(self):
        search_space = {
            "orchestration": {
                "bundle": {
                    "type": "categorical",
                    "choices": [
                        {"name": "trend_native", "overrides": {"router": {"type": "hard_switch"}}},
                        {"name": "filtered_hmm", "overrides": {"router": {"type": "confidence_weighted"}}},
                    ],
                }
            }
        }

        varying_paths = _find_varying_thesis_paths(_classify_search_space(search_space))

        self.assertEqual(varying_paths, ["orchestration.bundle"])
        with self.assertRaisesRegex(ValueError, "cannot search thesis_space parameters"):
            _validate_trade_ready_search_space(search_space, {"trade_ready_profile": {"enabled": True}})


if __name__ == "__main__":
    unittest.main()