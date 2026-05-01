import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import core.automl as automl_module
from core import ResearchPipeline
from core.automl import run_automl_study


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


def _make_pipeline_config(*, automl=None, strategy="feature"):
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
    return config


def _build_pipeline(raw, *, automl=None, strategy="feature"):
    pipeline = ResearchPipeline(_make_pipeline_config(automl=automl, strategy=strategy))
    pipeline.state["raw_data"] = raw
    pipeline.state["data"] = raw.copy()
    return pipeline


class AutoMLRegimeAwareTrainingTest(unittest.TestCase):
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
        self.assertEqual(list(signals["continuous_signals"].index), list(training["oos_predictions"].index))
        self.assertGreaterEqual(backtest["total_trades"], 0)

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
                "labels": {
                    "pt_mult": {"type": "categorical", "choices": [1.5]},
                    "sl_mult": {"type": "categorical", "choices": [1.5]},
                    "max_holding": {"type": "categorical", "choices": [12]},
                    "min_return": {"type": "categorical", "choices": [0.0]},
                    "volatility_window": {"type": "categorical", "choices": [12]},
                    "barrier_tie_break": {"type": "categorical", "choices": ["sl"]},
                },
                "regime": {
                    "n_regimes": {"type": "categorical", "choices": [2]},
                },
                "model": {
                    "type": {"type": "categorical", "choices": ["logistic"]},
                    "gap": {"type": "categorical", "choices": [6]},
                    "validation_fraction": {"type": "categorical", "choices": [0.2]},
                    "meta_n_splits": {"type": "categorical", "choices": [2]},
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


if __name__ == "__main__":
    unittest.main()