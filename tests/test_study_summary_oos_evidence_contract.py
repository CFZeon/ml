import unittest

from core.automl import run_automl_study
from tests.test_automl_holdout_objective import (
    _AutoMLDummyPipeline,
    _BasePipelineStub,
    _build_market_frame,
    _make_storage_path,
)


class StudySummaryOOSEvidenceContractTest(unittest.TestCase):
    def test_local_certification_summary_reports_adversarial_oos_evidence(self):
        raw = _build_market_frame(120)
        storage_path = _make_storage_path()
        base_pipeline = _BasePipelineStub(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "backtest": {"evaluation_mode": "local_certification"},
                "automl": {
                    "enabled": True,
                    "n_trials": 1,
                    "objective": "accuracy_first",
                    "policy_profile": "hardened_default",
                    "seed": 7,
                    "validation_fraction": 0.2,
                    "locked_holdout_bars": 24,
                    "locked_holdout_min_search_rows": 48,
                    "storage": storage_path,
                    "study_name": "study_summary_oos_evidence_local_certification",
                    "selection_policy": {"enabled": True},
                    "overfitting_control": {"enabled": True, "post_selection": {"enabled": True}},
                    "replication": {
                        "enabled": True,
                        "symbols": ["ETHUSDT"],
                        "include_window_cohorts": False,
                        "min_coverage": 1,
                        "min_pass_rate": 0.0,
                        "min_score": -1.0,
                    },
                },
                "features": {"schema_version": "test_v1"},
                "model": {"type": "gbm", "cv_method": "cpcv", "gap": 2},
            },
            raw_data=raw,
            data=raw.copy(),
        )

        summary = run_automl_study(
            base_pipeline,
            pipeline_class=_AutoMLDummyPipeline,
            trial_step_classes=[],
        )

        oos_evidence = summary["oos_evidence"]
        self.assertEqual(oos_evidence["class"], "adversarial_oos")
        self.assertTrue(oos_evidence["evidence_stack_complete"])
        self.assertTrue(oos_evidence["controls"]["cpcv_or_purged_temporal_search"]["complete"])
        self.assertTrue(oos_evidence["controls"]["search_stage_embargo"]["complete"])
        self.assertTrue(oos_evidence["controls"]["validation_holdout_gap"]["complete"])
        self.assertTrue(oos_evidence["controls"]["locked_holdout"]["complete"])
        self.assertTrue(oos_evidence["controls"]["post_selection_inference"]["complete"])
        self.assertTrue(oos_evidence["controls"]["replication"]["complete"])
        self.assertEqual(summary["summary_contract"]["oos_evidence"]["class"], "adversarial_oos")

    def test_research_summary_reports_partial_oos_when_only_search_control_exists(self):
        raw = _build_market_frame(120)
        storage_path = _make_storage_path()
        base_pipeline = _BasePipelineStub(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "backtest": {"evaluation_mode": "research_only"},
                "automl": {
                    "enabled": True,
                    "n_trials": 1,
                    "objective": "accuracy_first",
                    "policy_profile": "legacy_permissive",
                    "seed": 7,
                    "validation_fraction": 0.2,
                    "locked_holdout_enabled": False,
                    "storage": storage_path,
                    "study_name": "study_summary_oos_evidence_research_only",
                    "selection_policy": {"enabled": False},
                    "overfitting_control": {"enabled": False, "post_selection": {"enabled": False}},
                    "replication": {"enabled": False},
                },
                "features": {"schema_version": "test_v1"},
                "model": {"type": "gbm", "cv_method": "cpcv", "gap": 0},
            },
            raw_data=raw,
            data=raw.copy(),
        )

        summary = run_automl_study(
            base_pipeline,
            pipeline_class=_AutoMLDummyPipeline,
            trial_step_classes=[],
        )

        oos_evidence = summary["oos_evidence"]
        self.assertEqual(oos_evidence["class"], "partial_oos")
        self.assertFalse(oos_evidence["evidence_stack_complete"])
        self.assertTrue(oos_evidence["controls"]["cpcv_or_purged_temporal_search"]["complete"])
        self.assertFalse(oos_evidence["controls"]["locked_holdout"]["complete"])
        self.assertFalse(oos_evidence["controls"]["post_selection_inference"]["complete"])
        self.assertFalse(oos_evidence["controls"]["replication"]["complete"])
        self.assertIn("oos_control_incomplete:locked_holdout", oos_evidence["blocking_reasons"])


if __name__ == "__main__":
    unittest.main()