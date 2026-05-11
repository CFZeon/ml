import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import core.automl as automl_module
from core import ResearchPipeline
from core.automl import _summarize_training, run_automl_study
from core.pipeline import _summarize_regime_coverage_folds


def _make_market_frame(rows=280, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-05-01", periods=rows, freq="1h", tz="UTC")
    phase = np.linspace(0.0, 8.0 * np.pi, rows)
    regime = np.where(np.sin(np.linspace(0.0, 3.0 * np.pi, rows)) >= 0.0, 1.0, -1.0)
    drift = np.linspace(0.0, 9.0, rows)
    noise = rng.normal(0.0, 0.18, rows).cumsum()
    close = 100.0 + drift + 1.6 * np.sin(phase) + regime * 0.55 + noise
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    volume = 1_200.0 + 150.0 * (1.0 + np.cos(phase / 2.0))
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": close * volume,
            "trades": 120,
        },
        index=index,
    )


def _make_pipeline_config(*, automl=None, strategy="feature", feature_adaptation=None):
    config = {
        "data": {"symbol": "BTCUSDT", "interval": "1h"},
        "indicators": [],
        "features": {"lags": [1, 3, 6], "frac_diff_d": 0.4, "rolling_window": 20},
        "regime": {"enabled": True, "method": "explicit", "n_regimes": 2, "feature_lookback": 64},
        "labels": {"kind": "fixed_horizon", "horizon": 6, "threshold": 0.0001},
        "model": {
            "type": "logistic",
            "cv_method": "walk_forward",
            "gap": 6,
            "n_splits": 3,
            "validation_fraction": 0.2,
            "meta_n_splits": 2,
            "params": {"random_state": 7, "max_iter": 500},
            "meta_params": {"random_state": 11, "max_iter": 500},
            "regime_aware": {
                "enabled": True,
                "strategy": strategy,
                "min_samples_per_regime": 24,
                "coverage_config": {"max_dominant_share": 1.0, "min_distinct_regimes": 1},
            },
        },
        "feature_selection": {"enabled": True, "max_features": 12, "min_mi_threshold": 0.0},
        "signals": {
            "avg_win": 0.02,
            "avg_loss": 0.02,
            "threshold": 0.0,
            "edge_threshold": 0.0,
            "meta_threshold": 0.5,
        },
        "backtest": {
            "engine": "vectorbt",
            "use_open_execution": False,
            "signal_delay_bars": 1,
            "fee_rate": 0.0,
            "slippage_rate": 0.0,
        },
    }
    if automl is not None:
        config["automl"] = automl
    if feature_adaptation is not None:
        config["feature_adaptation"] = feature_adaptation
    return config


def _build_pipeline(raw, *, automl=None, strategy="feature", feature_adaptation=None):
    pipeline = ResearchPipeline(
        _make_pipeline_config(
            automl=automl,
            strategy=strategy,
            feature_adaptation=feature_adaptation,
        )
    )
    pipeline.state["raw_data"] = raw
    pipeline.state["data"] = raw.copy()
    return pipeline


class AutoMLRegimeAwareTrainingTest(unittest.TestCase):
    def test_regime_coverage_summary_surfaces_unseen_regime_degradation(self):
        summary = _summarize_regime_coverage_folds(
            [
                {
                    "fold": 0,
                    "split_id": "fold_0",
                    "fit": {"status": "passed"},
                    "validation": {"status": "passed"},
                    "test": {"status": "passed", "available_rows": 40},
                    "training_report": {"trained_regimes": ["bull", "bear"], "skipped_regimes": {}},
                    "inference_report": {"fallback_rows": 10, "unseen_regimes": ["crash"]},
                },
                {
                    "fold": 1,
                    "split_id": "fold_1",
                    "fit": {"status": "passed"},
                    "validation": {"status": "passed"},
                    "test": {"status": "passed", "available_rows": 20},
                    "training_report": {"trained_regimes": ["bull", "bear"], "skipped_regimes": {}},
                    "inference_report": {"fallback_rows": 0, "unseen_regimes": []},
                },
            ],
            coverage_config={"max_dominant_share": 1.0, "min_distinct_regimes": 1},
            strategy="specialist",
            fold_backtests=[
                {"fold": 0, "split_id": "fold_0", "net_profit_pct": -2.0, "sharpe_ratio": -0.5, "max_drawdown": 0.08, "total_trades": 4},
                {"fold": 1, "split_id": "fold_1", "net_profit_pct": 1.5, "sharpe_ratio": 0.4, "max_drawdown": 0.03, "total_trades": 3},
            ],
        )

        report = summary["unseen_regime_degradation_report"]
        self.assertTrue(report["enabled"])
        self.assertEqual(report["affected_fold_count"], 1)
        self.assertEqual(report["fallback_rows"], 10)
        self.assertEqual(report["fallback_evidence_rows"], 60)
        self.assertEqual(report["fallback_row_share"], 0.1667)
        self.assertEqual(report["unseen_regimes"], ["crash"])
        self.assertEqual(report["by_regime"]["crash"]["affected_fold_count"], 1)
        self.assertEqual(report["affected_folds"][0]["split_id"], "fold_0")
        self.assertEqual(report["affected_fold_metrics"]["mean_net_profit_pct"], -2.0)
        self.assertEqual(report["clean_fold_metrics"]["mean_net_profit_pct"], 1.5)

    def test_train_models_supports_regime_aware_primary_path(self):
        pipeline = _build_pipeline(_make_market_frame(seed=17), strategy="feature")

        pipeline.build_features()
        pipeline.build_labels()
        pipeline.align_data()
        training = pipeline.train_models()
        signals = pipeline.generate_signals()
        backtest = pipeline.run_backtest()

        regime_summary = training["regime"]["regime_aware"]
        self.assertTrue(regime_summary["enabled"])
        self.assertEqual(regime_summary["strategy"], "feature")
        self.assertGreater(len(regime_summary["folds"]), 0)
        self.assertGreater(len(training["oos_predictions"]), 0)
        coverage_summary = training["regime"]["coverage_summary"]
        self.assertIsNotNone(coverage_summary["fit_ok_share"])
        self.assertGreater(len(coverage_summary["folds"]), 0)
        self.assertIn("fit", coverage_summary["folds"][0])
        self.assertIn("validation", coverage_summary["folds"][0])
        self.assertIn("test", coverage_summary["folds"][0])
        self.assertTrue(coverage_summary["unseen_regime_degradation_report"]["enabled"])
        self.assertIn("unseen_regime_degradation_report", regime_summary)
        self.assertIn("unseen_regime_degradation_report", backtest)
        self.assertEqual(list(signals["continuous_signals"].index), list(training["oos_predictions"].index))
        self.assertGreaterEqual(backtest["total_trades"], 0)

    def test_train_models_reports_identity_feature_adaptation_metadata(self):
        pipeline = _build_pipeline(
            _make_market_frame(seed=23),
            strategy="feature",
            feature_adaptation={
                "scaling": {"mode": "identity", "fallback": "identity"},
                "interaction_budget": {"enabled": True, "max_features": 3, "max_regimes": 2},
            },
        )

        pipeline.build_features()
        pipeline.build_labels()
        pipeline.align_data()
        training = pipeline.train_models()

        adaptation = training["feature_adaptation"]
        self.assertTrue(adaptation["enabled"])
        self.assertEqual(adaptation["mode"], "fold_local")
        self.assertEqual(adaptation["requested_scaling_mode"], "identity")
        self.assertEqual(adaptation["requested_selection_mode"], "identity")
        self.assertTrue(adaptation["deferred_runtime"])
        self.assertFalse(adaptation["applied_in_any_fold"])
        self.assertGreater(len(adaptation["folds"]), 0)
        self.assertEqual(adaptation["folds"][0]["adapter_type"], "identity")
        self.assertEqual(adaptation["folds"][0]["input_features"], adaptation["folds"][0]["output_features"])
        self.assertEqual(adaptation["last_manifest"]["adapter_type"], "identity")
        self.assertEqual(adaptation["last_policy"]["scaling_mode"], "identity")
        self.assertEqual(pipeline.state["feature_adaptation"]["requested_scaling_mode"], "identity")
        json.dumps(adaptation)

    def test_train_models_applies_regime_conditioned_scaling_for_specialists(self):
        pipeline = _build_pipeline(
            _make_market_frame(seed=31),
            strategy="specialist",
            feature_adaptation={
                "scaling": {
                    "mode": "regime_conditioned",
                    "fallback": "global",
                    "min_regime_samples": 12,
                    "confidence_floor": 0.0,
                }
            },
        )

        pipeline.build_features()
        pipeline.build_labels()
        pipeline.align_data()
        training = pipeline.train_models()
        signals = pipeline.generate_signals()

        adaptation = training["feature_adaptation"]
        self.assertTrue(adaptation["enabled"])
        self.assertTrue(adaptation["applied_in_any_fold"])
        self.assertEqual(adaptation["requested_scaling_mode"], "regime_conditioned")
        self.assertEqual(adaptation["folds"][0]["adapter_type"], "regime_conditioned_scaling")
        self.assertGreater(adaptation["last_manifest"]["eligible_scaling_column_count"], 0)
        self.assertGreaterEqual(adaptation["last_manifest"]["regime_bank_count"], 1)
        self.assertIsNotNone(signals["continuous_signals"])

    def test_train_models_applies_regime_conditioned_masking_for_specialists(self):
        pipeline = _build_pipeline(
            _make_market_frame(seed=33),
            strategy="specialist",
            feature_adaptation={
                "selection": {
                    "mode": "per_regime_mask",
                    "fallback": "global",
                    "min_regime_samples": 12,
                    "min_feature_rows": 8,
                    "min_active_share": 0.1,
                },
                "disable_incompatible_features": True,
            },
        )

        pipeline.build_features()
        pipeline.build_labels()
        pipeline.align_data()
        training = pipeline.train_models()
        signals = pipeline.generate_signals()

        adaptation = training["feature_adaptation"]
        disabled_columns = set(adaptation["last_policy"].get("disabled_columns") or [])
        self.assertTrue(adaptation["enabled"])
        self.assertTrue(adaptation["applied_in_any_fold"])
        self.assertEqual(adaptation["requested_selection_mode"], "per_regime_mask")
        self.assertEqual(adaptation["last_manifest"]["adapter_type"], "composite_feature_adaptation")
        self.assertGreater(adaptation["last_manifest"]["mask_candidate_column_count"], 0)
        self.assertGreaterEqual(adaptation["last_manifest"]["regime_mask_count"], 1)
        self.assertTrue(disabled_columns.isdisjoint(set(training["last_selected_columns"])))
        self.assertIsNotNone(signals["continuous_signals"])

    def test_train_models_rejects_feature_strategy_scaling(self):
        pipeline = _build_pipeline(
            _make_market_frame(seed=37),
            strategy="feature",
            feature_adaptation={
                "scaling": {"mode": "regime_conditioned", "fallback": "global"},
            },
        )

        pipeline.build_features()
        pipeline.build_labels()
        pipeline.align_data()
        with self.assertRaisesRegex(ValueError, "strategy='feature'"):
            pipeline.train_models()

    def test_train_models_rejects_feature_strategy_masking(self):
        pipeline = _build_pipeline(
            _make_market_frame(seed=41),
            strategy="feature",
            feature_adaptation={
                "selection": {"mode": "per_regime_mask", "fallback": "global"},
                "disable_incompatible_features": True,
            },
        )

        pipeline.build_features()
        pipeline.build_labels()
        pipeline.align_data()
        with self.assertRaisesRegex(ValueError, "strategy='feature'"):
            pipeline.train_models()

    def test_train_models_feature_strategy_bundle_honors_interaction_budget(self):
        pipeline = _build_pipeline(
            _make_market_frame(seed=43),
            strategy="feature",
            feature_adaptation={
                "interaction_budget": {
                    "enabled": True,
                    "max_features": 1,
                    "max_regimes": 1,
                }
            },
        )

        pipeline.build_features()
        pipeline.build_labels()
        pipeline.align_data()
        training = pipeline.train_models()

        regime_fold = training["regime"]["regime_aware"]["folds"][0]
        feature_adaptation = regime_fold["training_report"]["feature_adaptation"]
        manifest = feature_adaptation["manifest"]

        self.assertEqual(manifest["adapter_type"], "regime_feature_strategy")
        self.assertEqual(manifest["max_interaction_features"], 1)
        self.assertEqual(manifest["max_interaction_regimes"], 1)
        self.assertLessEqual(manifest["interaction_column_count"], 1)
        self.assertGreater(manifest["generated_column_count"], 0)

    def test_automl_training_summary_preserves_feature_adaptation(self):
        pipeline = _build_pipeline(
            _make_market_frame(seed=47),
            strategy="specialist",
            feature_adaptation={
                "selection": {
                    "mode": "per_regime_mask",
                    "fallback": "global",
                    "min_regime_samples": 12,
                    "min_feature_rows": 8,
                    "min_active_share": 0.1,
                },
                "disable_incompatible_features": True,
            },
        )

        pipeline.build_features()
        pipeline.build_labels()
        pipeline.align_data()
        training = pipeline.train_models()
        summary = _summarize_training(training)

        self.assertIn("feature_adaptation", summary)
        self.assertIn("specialist_library", summary)
        self.assertEqual(summary["feature_adaptation"]["last_manifest"]["adapter_type"], "composite_feature_adaptation")
        self.assertEqual(summary["specialist_library"]["fallback_model_id"], "fallback_generalist")
        json.dumps(automl_module._json_ready(summary["feature_adaptation"]))
        json.dumps(automl_module._json_ready(summary["specialist_library"]))

    def test_train_models_surfaces_specialist_library_for_specialist_strategy(self):
        pipeline = _build_pipeline(
            _make_market_frame(seed=49),
            strategy="specialist",
            feature_adaptation={
                "selection": {
                    "mode": "per_regime_mask",
                    "fallback": "global",
                    "min_regime_samples": 12,
                    "min_feature_rows": 8,
                    "min_active_share": 0.1,
                },
                "disable_incompatible_features": True,
            },
        )

        pipeline.build_features()
        pipeline.build_labels()
        pipeline.align_data()
        training = pipeline.train_models()

        specialist_library = training["specialist_library"]
        self.assertEqual(specialist_library["fallback_model_id"], "fallback_generalist")
        selection_contract = specialist_library["metadata"]["selection_contract"]
        self.assertTrue(selection_contract["candidate_model_ids"])
        self.assertEqual(selection_contract["active_model_ids"], [])
        self.assertEqual(pipeline.state["specialist_library"], specialist_library)

    def test_refit_artifact_surfaces_feature_adaptation(self):
        pipeline = _build_pipeline(
            _make_market_frame(rows=220, seed=53),
            strategy="specialist",
            feature_adaptation={
                "selection": {
                    "mode": "per_regime_mask",
                    "fallback": "global",
                    "min_regime_samples": 10,
                    "min_feature_rows": 6,
                    "min_active_share": 0.1,
                },
                "disable_incompatible_features": True,
            },
        )

        pipeline.build_features()
        pipeline.build_labels()
        pipeline.align_data()
        pipeline.train_models()

        artifact = pipeline.refit_selected_candidate(
            {
                "best_overrides": {"model": {"type": "logistic"}},
                "selection_freeze": {"candidate_hash": "candidate-1", "trial_number": 0},
            }
        )

        self.assertIn("feature_adaptation", artifact)
        self.assertIn("specialist_library", artifact)
        self.assertEqual(
            artifact["feature_adaptation"].get("last_manifest", {}).get("adapter_type"),
            "composite_feature_adaptation",
        )
        self.assertEqual(artifact["feature_adaptation"], artifact["training"].get("feature_adaptation"))
        self.assertEqual(artifact["specialist_library"], artifact["training"].get("specialist_library"))

    def test_run_automl_study_preserves_regime_aware_trial_metadata(self):
        if automl_module.optuna is None:
            self.skipTest("optuna is not installed")

        raw = _make_market_frame(seed=29)
        fd, storage_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(storage_path)
        automl_config = {
            "enabled": True,
            "n_trials": 1,
            "seed": 13,
            "validation_fraction": 0.2,
            "locked_holdout_enabled": False,
            "enable_pruning": False,
            "objective": "directional_accuracy",
            "storage": str(storage_path),
            "study_name": "automl_regime_aware_training_test",
            "policy_profile": "legacy_permissive",
            "selection_policy": {"enabled": False},
            "overfitting_control": {"enabled": False},
            "search_space": {
                "features": {
                    "lags": {"type": "categorical", "choices": ["1,3,6"]},
                    "frac_diff_d": {"type": "categorical", "choices": [0.4]},
                    "rolling_window": {"type": "categorical", "choices": [20]},
                    "squeeze_quantile": {"type": "categorical", "choices": [0.2]},
                },
                "feature_selection": {
                    "enabled": {"type": "categorical", "choices": [True]},
                    "max_features": {"type": "categorical", "choices": [12]},
                    "min_mi_threshold": {"type": "categorical", "choices": [0.0]},
                },
                "regime": {
                    "n_regimes": {"type": "categorical", "choices": [2]},
                },
                "model": {
                    "type": {"type": "categorical", "choices": ["logistic"]},
                    "gap": {"type": "categorical", "choices": [6]},
                    "regime_aware": {
                        "enabled": {"type": "categorical", "choices": [True]},
                        "strategy": {"type": "categorical", "choices": ["feature"]},
                        "min_samples_per_regime": {"type": "categorical", "choices": [24]},
                    },
                    "params": {
                        "logistic": {
                            "c": {"type": "categorical", "choices": [1.0]},
                        }
                    },
                },
            },
        }
        pipeline = _build_pipeline(raw, automl=automl_config, strategy="feature")

        try:
            summary = run_automl_study(pipeline, pipeline_class=ResearchPipeline, trial_step_classes=[])
        finally:
            if os.path.exists(storage_path):
                os.remove(storage_path)

        self.assertTrue(summary["best_overrides"]["model"]["regime_aware"]["enabled"])
        self.assertEqual(summary["best_overrides"]["model"]["regime_aware"]["strategy"], "feature")
        self.assertTrue(summary["best_training"]["regime"]["regime_aware"]["enabled"])
        self.assertEqual(summary["best_training"]["regime"]["regime_aware"]["strategy"], "feature")
        self.assertIn("coverage_summary", summary["best_training"]["regime"])

    def test_run_automl_study_surfaces_orchestration_bundle_lineage(self):
        if automl_module.optuna is None:
            self.skipTest("optuna is not installed")

        raw = _make_market_frame(seed=71)
        fd, storage_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(storage_path)
        automl_config = {
            "enabled": True,
            "n_trials": 1,
            "objective": "directional_accuracy",
            "storage": str(storage_path),
            "study_name": "automl_orchestration_bundle_test",
            "policy_profile": "legacy_permissive",
            "selection_policy": {"enabled": False},
            "overfitting_control": {"enabled": False},
            "search_space": {
                "orchestration": {
                    "bundle": {
                        "type": "categorical",
                        "choices": [
                            {
                                "name": "trend_native_weighted",
                                "description": "Trend-state detector with weighted routing metadata.",
                                "regime": {
                                    "detectors": [
                                        {
                                            "name": "trend_native_primary",
                                            "type": "trend_state",
                                            "primary": True,
                                            "warmup_bars": 120,
                                        }
                                    ]
                                },
                                "feature_adaptation": {
                                    "scaling": {"mode": "identity", "fallback": "identity"},
                                },
                                "model_library": {
                                    "fallback_model": "fallback_generalist",
                                    "specialists": [
                                        {
                                            "model_id": "trend_model",
                                            "estimator": "logistic",
                                            "compatible_regimes": ["trend_up_low_vol"],
                                        }
                                    ],
                                },
                                "router": {
                                    "type": "confidence_weighted",
                                    "hysteresis_margin": 0.05,
                                    "min_persistence_bars": 2,
                                    "cooldown_bars": 4,
                                },
                                "model": {
                                    "regime_aware": {
                                        "enabled": True,
                                        "strategy": "feature",
                                        "min_samples_per_regime": 24,
                                    }
                                },
                            }
                        ],
                    }
                },
                "features": {
                    "lags": {"type": "categorical", "choices": ["1,3,6"]},
                    "frac_diff_d": {"type": "categorical", "choices": [0.4]},
                    "rolling_window": {"type": "categorical", "choices": [20]},
                    "squeeze_quantile": {"type": "categorical", "choices": [0.2]},
                },
                "feature_selection": {
                    "enabled": {"type": "categorical", "choices": [True]},
                    "max_features": {"type": "categorical", "choices": [12]},
                    "min_mi_threshold": {"type": "categorical", "choices": [0.0]},
                },
                "model": {
                    "type": {"type": "categorical", "choices": ["logistic"]},
                    "gap": {"type": "categorical", "choices": [6]},
                    "params": {
                        "logistic": {
                            "c": {"type": "categorical", "choices": [1.0]},
                        }
                    },
                },
            },
        }
        pipeline = _build_pipeline(raw, automl=automl_config, strategy="feature")

        try:
            summary = run_automl_study(pipeline, pipeline_class=ResearchPipeline, trial_step_classes=[])
        finally:
            if os.path.exists(storage_path):
                os.remove(storage_path)

        self.assertEqual(summary["best_bundle_lineage"]["bundle_name"], "trend_native_weighted")
        self.assertEqual(summary["best_bundle_lineage"]["primary_detector"]["type"], "trend_state")
        self.assertEqual(summary["best_bundle_lineage"]["specialist_model_ids"], ["trend_model"])
        self.assertEqual(summary["best_bundle_lineage"]["router"]["type"], "confidence_weighted")
        self.assertEqual(summary["best_overrides"]["experiment"]["bundle_name"], "trend_native_weighted")
        self.assertEqual(summary["best_overrides"]["model"]["regime_aware"]["router"]["type"], "confidence_weighted")


if __name__ == "__main__":
    unittest.main()