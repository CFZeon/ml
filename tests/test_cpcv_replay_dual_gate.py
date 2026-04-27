import unittest

from core.automl import _resolve_validation_sources


class CPCVReplayDualGateTest(unittest.TestCase):
    def test_cpcv_diagnostics_do_not_satisfy_replay_requirement_by_themselves(self):
        validation_contract = {
            "search_ranker": "cpcv",
            "contiguous_validation": "walk_forward_replay",
            "locked_holdout": "disabled",
            "replication": "disabled",
        }
        training = {
            "validation": {"method": "cpcv"},
        }
        backtest = {"validation_method": "cpcv"}

        sources = _resolve_validation_sources(
            training,
            backtest,
            validation_contract,
            holdout_enabled=False,
            replication_enabled=False,
        )

        self.assertTrue(sources["required_source_checks"]["search_ranker"])
        self.assertFalse(sources["required_source_checks"]["contiguous_validation"])
        self.assertFalse(sources["all_required_sources_passed"])


if __name__ == "__main__":
    unittest.main()