import unittest

from core.automl import run_automl_study
from tests.test_automl_holdout_objective import (
    _AutoMLDummyPipeline,
    _BasePipelineStub,
    _build_market_frame,
    _make_storage_path,
)


class CertificationRequiresCompleteOOSStackTest(unittest.TestCase):
    def test_trade_ready_aborts_before_optimization_when_oos_stack_is_incomplete(self):
        raw = _build_market_frame(120)
        storage_path = _make_storage_path()

        base_pipeline = _BasePipelineStub(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "backtest": {"evaluation_mode": "trade_ready"},
                "automl": {
                    "enabled": True,
                    "n_trials": 1,
                    "objective": "accuracy_first",
                    "policy_profile": "hardened_default",
                    "seed": 11,
                    "validation_fraction": 0.2,
                    "locked_holdout_bars": 24,
                    "locked_holdout_min_search_rows": 48,
                    "search_validation_gap_bars": 0,
                    "validation_holdout_gap_bars": 0,
                    "storage": storage_path,
                    "study_name": "automl_oos_stack_rejection_test",
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
                "model": {"type": "gbm", "cv_method": "walk_forward", "gap": 0},
            },
            raw_data=raw,
            data=raw.copy(),
        )

        with self.assertRaisesRegex(RuntimeError, "complete adversarial OOS evidence stack") as ctx:
            run_automl_study(
                base_pipeline,
                pipeline_class=_AutoMLDummyPipeline,
                trial_step_classes=[],
            )

        self.assertIn("oos_control_incomplete:cpcv_or_purged_temporal_search", str(ctx.exception))
        self.assertIn("oos_control_incomplete:search_stage_embargo", str(ctx.exception))
        self.assertIn("oos_control_incomplete:validation_holdout_gap", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()