import unittest

from core.automl import run_automl_study
from core.automl_contracts import validate_summary_contract
from tests.test_automl_holdout_objective import (
    _AutoMLDummyPipeline,
    _BasePipelineStub,
    _build_market_frame,
    _make_storage_path,
)


class AutoMLEvidenceClassTest(unittest.TestCase):
    def test_run_automl_study_separates_selection_validation_and_holdout_evidence(self):
        raw = _build_market_frame(120)
        storage_path = _make_storage_path()

        base_pipeline = _BasePipelineStub(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "automl": {
                    "enabled": True,
                    "n_trials": 1,
                    "objective": "accuracy_first",
                    "policy_profile": "legacy_permissive",
                    "seed": 7,
                    "validation_fraction": 0.2,
                    "locked_holdout_bars": 24,
                    "locked_holdout_min_search_rows": 48,
                    "storage": storage_path,
                    "study_name": "automl_evidence_class_test",
                },
                "features": {"schema_version": "test_v1"},
                "model": {"type": "gbm"},
            },
            raw_data=raw,
            data=raw.copy(),
        )

        summary = run_automl_study(
            base_pipeline,
            pipeline_class=_AutoMLDummyPipeline,
            trial_step_classes=[],
        )

        self.assertEqual(summary["evidence_class"], "selection_evidence")
        self.assertEqual(summary["selection_evidence"]["evidence_class"], "selection_evidence")
        self.assertEqual(summary["validation_replay_evidence"]["evidence_class"], "outer_replay")
        self.assertEqual(summary["best_backtest"]["evidence_class"], "outer_replay")
        self.assertEqual(summary["validation_holdout"]["backtest"]["evidence_class"], "outer_replay")
        self.assertEqual(summary["locked_holdout"]["backtest"]["evidence_class"], "locked_holdout")
        self.assertEqual(summary["summary_contract"]["evidence_class"], "selection_evidence")

    def test_validate_summary_contract_preserves_gate_status_and_blocked_promotion(self):
        summary = validate_summary_contract(
            {
                "selection_outcome": {
                    "status": "abstain_no_eligible_trial",
                    "top_rejection_reasons": ["replication_failed"],
                },
                "best_overrides": {},
                "locked_holdout": {"enabled": False},
                "replication": {
                    "enabled": True,
                    "promotion_pass": False,
                    "reasons": ["replication_failed"],
                },
                "promotion_eligibility_report": {
                    "gate_status": {
                        "replication": {
                            "name": "replication",
                            "passed": False,
                            "mode": "blocking",
                        }
                    },
                    "blocking_failures": ["replication_failed"],
                    "promotion_ready": True,
                    "approved": True,
                },
            }
        )

        self.assertIn("replication", summary["promotion_eligibility_report"]["gate_status"])
        self.assertFalse(summary["promotion_eligibility_report"]["promotion_ready"])
        self.assertFalse(summary["promotion_eligibility_report"]["approved"])
        self.assertEqual(summary["promotion_eligibility_report"]["reasons"], ["replication_failed"])
