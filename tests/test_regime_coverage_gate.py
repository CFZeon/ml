import copy
import os
import unittest

import core.automl as automl_module
from core.automl import run_automl_study
from tests.test_automl_holdout_objective import (
    _AutoMLDummyPipeline,
    _BasePipelineStub,
    _build_market_frame,
    _make_storage_path,
)


def _make_fold_coverage(status, *, dominant_share, distinct_regimes, reasons):
    return {
        "status": status,
        "promotion_pass": status == "passed",
        "distinct_regimes": distinct_regimes,
        "dominant_regime": "bull",
        "dominant_share": dominant_share,
        "regime_distribution": {"bull": dominant_share, "bear": round(1.0 - dominant_share, 4)},
        "reasons": list(reasons),
    }


def _make_coverage_summary(status, *, dominant_share, distinct_regimes, reasons):
    fold_summary = _make_fold_coverage(
        status,
        dominant_share=dominant_share,
        distinct_regimes=distinct_regimes,
        reasons=reasons,
    )
    return {
        "status": status,
        "promotion_pass": status == "passed",
        "reasons": list(reasons),
        "configured_thresholds": {"min_distinct_regimes": 2, "max_dominant_share": 0.8},
        "folds": [{"fold": 0, "split_id": "fold_0", "fit": fold_summary, "validation": fold_summary, "test": fold_summary, "specialist_shortfalls": {}}],
        "fit_ok_share": 1.0 if status == "passed" else 0.0,
        "validation_ok_share": 1.0 if status == "passed" else 0.0,
        "test_ok_share": 1.0 if status == "passed" else 0.0,
        "distinct_regimes_min_fit": distinct_regimes,
        "distinct_regimes_min_validation": distinct_regimes,
        "distinct_regimes_min_test": distinct_regimes,
        "max_dominant_share_fit": dominant_share,
        "max_dominant_share_validation": dominant_share,
        "max_dominant_share_test": dominant_share,
        "fallback_rows": 0,
        "unseen_regimes": [],
        "specialist_shortfalls": {},
        "strategy": None,
    }


def _make_unknown_coverage_summary():
    return {
        "status": "unknown",
        "promotion_pass": False,
        "reasons": ["regime_coverage_unavailable"],
        "configured_thresholds": {"min_distinct_regimes": 2, "max_dominant_share": 0.8},
        "folds": [],
        "fit_ok_share": None,
        "validation_ok_share": None,
        "test_ok_share": None,
        "distinct_regimes_min_fit": None,
        "distinct_regimes_min_validation": None,
        "distinct_regimes_min_test": None,
        "max_dominant_share_fit": None,
        "max_dominant_share_validation": None,
        "max_dominant_share_test": None,
        "fallback_rows": 0,
        "unseen_regimes": [],
        "specialist_shortfalls": {},
        "strategy": None,
    }


class _CoverageGatePipeline(_AutoMLDummyPipeline):
    coverage_summary = None

    @classmethod
    def set_coverage_summary(cls, summary):
        cls.coverage_summary = copy.deepcopy(summary)

    def run_step(self, name):
        result = super().run_step(name)
        if name != "train_models":
            return result

        training = copy.deepcopy(result)
        regime_summary = dict(training.get("regime") or {})
        regime_summary["coverage_summary"] = copy.deepcopy(type(self).coverage_summary)
        training["regime"] = regime_summary
        self.state["training"] = training
        self.step_results[name] = training
        return training


class RegimeCoverageGateTest(unittest.TestCase):
    def _run_study(self, coverage_summary):
        if automl_module.optuna is None:
            self.skipTest("optuna is not installed")

        storage_path = _make_storage_path()
        raw = _build_market_frame(96)
        _CoverageGatePipeline.set_coverage_summary(coverage_summary)
        pipeline = _BasePipelineStub(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "features": {"schema_version": "coverage_gate_v1"},
                "model": {"type": "gbm"},
                "automl": {
                    "enabled": True,
                    "n_trials": 1,
                    "objective": "accuracy_first",
                    "seed": 7,
                    "validation_fraction": 0.2,
                    "locked_holdout_enabled": False,
                    "enable_pruning": False,
                    "overfitting_control": {"enabled": False},
                    "storage": storage_path,
                    "study_name": "regime_coverage_gate_test",
                    "selection_policy": {
                        "enabled": True,
                        "min_validation_trade_count": 1,
                        "require_locked_holdout_pass": False,
                        "gate_modes": {
                            "feature_portability": "disabled",
                            "feature_admission": "disabled",
                            "regime_stability": "disabled",
                            "cross_venue_integrity": "disabled",
                            "data_certification": "disabled",
                            "signal_decay": "disabled",
                            "execution_realism": "disabled",
                            "stress_realism": "disabled",
                            "locked_holdout": "disabled",
                            "locked_holdout_gap": "disabled",
                            "replication": "disabled",
                            "param_fragility": "disabled",
                            "lookahead_guard": "disabled",
                        },
                    },
                },
            },
            raw_data=raw,
            data=raw.copy(),
        )

        try:
            return run_automl_study(
                pipeline,
                pipeline_class=_CoverageGatePipeline,
                trial_step_classes=[],
            )
        finally:
            if os.path.exists(storage_path):
                os.remove(storage_path)

    def test_dominant_regime_blocks_selection(self):
        summary = self._run_study(
            _make_coverage_summary(
                "failed",
                dominant_share=0.92,
                distinct_regimes=1,
                reasons=["dominant_regime_exceeds_threshold"],
            )
        )

        rejected_trial = summary["top_trials"][0]
        self.assertFalse(rejected_trial["selection_policy"]["eligible"])
        self.assertFalse(rejected_trial["selection_policy"]["eligibility_checks"]["regime_coverage"])
        self.assertIn("dominant_regime_exceeds_threshold", rejected_trial["selection_policy"]["eligibility_reasons"])
        gate_report = rejected_trial["selection_policy"]["promotion_eligibility_report"]
        self.assertIn("regime_coverage", gate_report["gate_status"])
        self.assertFalse(gate_report["gate_status"]["regime_coverage"]["passed"])

    def test_missing_regime_coverage_is_unknown_and_blocking_in_trade_ready(self):
        summary = self._run_study(_make_unknown_coverage_summary())

        rejected_trial = summary["top_trials"][0]
        self.assertFalse(rejected_trial["selection_policy"]["eligible"])
        self.assertIn("regime_coverage_unavailable", rejected_trial["selection_policy"]["eligibility_reasons"])
        gate_report = rejected_trial["selection_policy"]["promotion_eligibility_report"]
        self.assertEqual(gate_report["gate_status"]["regime_coverage"]["status"], "unknown")
        self.assertEqual(gate_report["gate_status"]["regime_coverage"]["mode"], "blocking")

    def test_balanced_regime_coverage_passes_gate(self):
        summary = self._run_study(
            _make_coverage_summary(
                "passed",
                dominant_share=0.55,
                distinct_regimes=2,
                reasons=[],
            )
        )

        self.assertTrue(summary["best_selection_policy"]["selection_policy"]["eligible"])
        gate_report = summary["promotion_eligibility_report"]
        self.assertIn("regime_coverage", gate_report["gate_status"])
        self.assertTrue(gate_report["gate_status"]["regime_coverage"]["passed"])


if __name__ == "__main__":
    unittest.main()