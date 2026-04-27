import copy
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import core.automl as automl_module
from core.automl import _build_experiment_manifest, run_automl_study


def _build_market_frame(rows):
    index = pd.date_range("2026-01-01", periods=rows, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": np.linspace(100.0, 110.0, len(index)),
            "high": np.linspace(101.0, 111.0, len(index)),
            "low": np.linspace(99.0, 109.0, len(index)),
            "close": np.linspace(100.5, 110.5, len(index)),
            "volume": np.full(len(index), 1_000.0),
            "quote_volume": np.linspace(100_000.0, 110_000.0, len(index)),
            "trades": np.full(len(index), 100),
        },
        index=index,
    )


class _BasePipelineStub:
    def __init__(self, config, raw_data):
        self.config = copy.deepcopy(config)
        self.state = {
            "raw_data": raw_data.copy(),
            "data": raw_data.copy(),
            "data_lineage": {
                "source_datasets": [
                    {
                        "name": "binance_bars",
                        "source_fingerprint": f"rows:{len(raw_data)}",
                        "contract": {"contract_hash": f"contract:{len(raw_data)}"},
                    }
                ]
            },
            "symbol_filters": {"tick_size": 0.1, "step_size": 0.001, "min_notional": 10.0},
            "cross_asset_context": {},
            "futures_context": None,
        }

    def require(self, key):
        return self.state[key]


class _AutoMLManifestPipeline:
    def __init__(self, config, steps=None):
        self.config = copy.deepcopy(config)
        self.state = {}
        self.step_results = {}

    def run_step(self, name):
        if name == "build_features":
            self.step_results[name] = self.state["data"]
            return self.step_results[name]

        if name == "detect_regimes":
            self.step_results[name] = None
            return None

        if name == "build_labels":
            index = self.state["data"].index
            y = pd.Series(np.where(np.arange(len(index)) % 2 == 0, 1, -1), index=index)
            labels = pd.DataFrame({"label": y, "t1": index}, index=index)
            self.state["labels"] = labels
            self.step_results[name] = labels
            return labels

        if name == "align_data":
            index = self.state["data"].index
            X = pd.DataFrame({"feature": np.arange(len(index), dtype=float)}, index=index)
            y = pd.Series(np.where(np.arange(len(index)) % 2 == 0, 1, -1), index=index)
            labels = pd.DataFrame({"label": y, "t1": index}, index=index)
            aligned = {"X": X, "y": y, "labels": labels}
            self.state["X"] = X
            self.state["y"] = y
            self.state["labels_aligned"] = labels
            self.step_results[name] = aligned
            return aligned

        if name == "select_features":
            self.step_results[name] = None
            return None

        if name == "compute_sample_weights":
            weights = pd.Series(1.0, index=self.state["X"].index, name="sample_weight")
            self.state["sample_weights"] = weights
            self.step_results[name] = weights
            return weights

        if name == "train_models":
            oos_index = self.state["X"].index[-12:]
            probability_frame = pd.DataFrame({-1: 0.2, 0: 0.0, 1: 0.8}, index=oos_index)
            signals = pd.Series(1.0, index=oos_index)
            training = {
                "fold_metrics": [{"accuracy": 0.65, "f1_macro": 0.65, "directional_accuracy": 0.65, "directional_f1_macro": 0.65}],
                "avg_accuracy": 0.65,
                "avg_f1_macro": 0.65,
                "avg_directional_accuracy": 0.65,
                "avg_directional_f1_macro": 0.65,
                "avg_log_loss": 0.25,
                "avg_brier_score": 0.10,
                "avg_calibration_error": 0.05,
                "headline_metrics": {"directional_accuracy": 0.65},
                "last_model": object(),
                "last_meta": object(),
                "last_selected_columns": ["feature"],
                "last_avg_win": 0.02,
                "last_avg_loss": 0.01,
                "oos_predictions": pd.Series(1, index=oos_index),
                "oos_probabilities": probability_frame,
                "oos_meta_prob": pd.Series(0.6, index=oos_index),
                "oos_profitability_prob": pd.Series(0.6, index=oos_index),
                "oos_direction_edge": pd.Series(0.6, index=oos_index),
                "oos_confidence": pd.Series(0.8, index=oos_index),
                "oos_expected_trade_edge": pd.Series(0.02, index=oos_index),
                "oos_position_size": signals,
                "oos_kelly_size": signals,
                "oos_event_signals": signals,
                "oos_continuous_signals": signals,
                "oos_signals": pd.Series(1, index=oos_index),
                "feature_block_diagnostics": {"summary": [], "top_features": [], "folds": []},
                "feature_selection": {"enabled": False, "mode": "fold_local", "avg_input_features": 1.0, "avg_selected_features": 1.0, "folds": []},
                "bootstrap": {"model_type": "logistic", "used_in_any_fold": False, "warning_count": 0, "folds": []},
                "purging": [],
                "signal_tuning": [],
                "operational_monitoring": {},
            }
            self.state["training"] = training
            self.step_results[name] = training
            return training

        if name == "generate_signals":
            index = self.state["training"]["oos_predictions"].index
            continuous = pd.Series(1.0, index=index)
            payload = {"signals": continuous.astype(int), "continuous_signals": continuous}
            self.state["signals"] = payload
            self.step_results[name] = payload
            return payload

        if name == "run_backtest":
            index = self.state["signals"]["continuous_signals"].index
            returns = np.full(len(index), 0.001)
            equity_curve = pd.Series(10_000.0 * np.cumprod(1.0 + returns), index=index)
            backtest = {
                "net_profit": float(len(index)),
                "net_profit_pct": 0.05,
                "sharpe_ratio": 1.2,
                "sortino_ratio": 1.2,
                "calmar_ratio": 1.1,
                "profit_factor": 1.5,
                "max_drawdown": -0.02,
                "total_trades": len(index),
                "win_rate": 0.55,
                "ending_equity": float(equity_curve.iloc[-1]),
                "equity_curve": equity_curve,
                "returns": pd.Series(returns, index=index),
                "statistical_significance": {
                    "enabled": True,
                    "observation_count": len(index),
                    "metrics": {
                        "sharpe_ratio": {"confidence_interval": {"lower": 0.1, "upper": 2.0}},
                    },
                },
            }
            self.state["backtest"] = backtest
            self.step_results[name] = backtest
            return backtest

        raise KeyError(name)


def _build_base_pipeline(storage_path, *, schema_version="schema_v1", resume_mode="never"):
    raw_data = _build_market_frame(72)
    config = {
        "data": {"symbol": "BTCUSDT", "interval": "1h"},
        "features": {"schema_version": schema_version, "lookahead_guard": {"enabled": True, "mode": "blocking"}},
        "model": {"type": "logistic"},
        "automl": {
            "enabled": True,
            "n_trials": 1,
            "seed": 7,
            "validation_fraction": 0.2,
            "locked_holdout_enabled": False,
            "enable_pruning": False,
            "objective": "directional_accuracy",
            "storage": str(storage_path),
            "study_name": "automl_manifest_test",
            "resume_mode": resume_mode,
            "selection_policy": {"enabled": False},
            "overfitting_control": {"enabled": False},
        },
    }
    return _BasePipelineStub(config, raw_data)


class AutoMLExperimentManifestTest(unittest.TestCase):
    def test_manifest_hash_changes_when_search_space_changes(self):
        raw_data = _build_market_frame(48)
        state_bundle = {
            "raw_data": raw_data,
            "data_lineage": {"source_datasets": [{"name": "bars", "source_fingerprint": "v1"}]},
            "symbol_filters": {},
            "universe_policy": {},
            "universe_snapshot_meta": {},
            "eligible_symbols": ["BTCUSDT"],
        }
        base_config = {"data": {"symbol": "BTCUSDT", "interval": "1h"}, "features": {"schema_version": "schema_v1"}}
        automl_config = {"objective": "directional_accuracy"}

        manifest_a = _build_experiment_manifest(
            base_config,
            automl_config,
            state_bundle,
            {"model": {"type": {"type": "categorical", "choices": ["logistic"]}}},
        )
        manifest_b = _build_experiment_manifest(
            base_config,
            automl_config,
            state_bundle,
            {"model": {"type": {"type": "categorical", "choices": ["gbm"]}}},
        )

        self.assertNotEqual(manifest_a["search_space_hash"], manifest_b["search_space_hash"])
        self.assertNotEqual(manifest_a["experiment_id"], manifest_b["experiment_id"])

    def test_manifest_hash_changes_when_data_window_changes(self):
        base_config = {"data": {"symbol": "BTCUSDT", "interval": "1h"}, "features": {"schema_version": "schema_v1"}}
        automl_config = {"objective": "directional_accuracy"}
        search_space = {"model": {"type": {"type": "categorical", "choices": ["logistic"]}}}

        manifest_a = _build_experiment_manifest(
            base_config,
            automl_config,
            {
                "raw_data": _build_market_frame(48),
                "data_lineage": {},
                "symbol_filters": {},
                "universe_policy": {},
                "universe_snapshot_meta": {},
                "eligible_symbols": [],
            },
            search_space,
        )
        manifest_b = _build_experiment_manifest(
            base_config,
            automl_config,
            {
                "raw_data": _build_market_frame(64),
                "data_lineage": {},
                "symbol_filters": {},
                "universe_policy": {},
                "universe_snapshot_meta": {},
                "eligible_symbols": [],
            },
            search_space,
        )

        self.assertNotEqual(manifest_a["data_lineage_hash"], manifest_b["data_lineage_hash"])
        self.assertNotEqual(manifest_a["experiment_id"], manifest_b["experiment_id"])

    def test_manifest_hash_changes_when_schema_version_changes(self):
        raw_data = _build_market_frame(48)
        state_bundle = {
            "raw_data": raw_data,
            "data_lineage": {},
            "symbol_filters": {},
            "universe_policy": {},
            "universe_snapshot_meta": {},
            "eligible_symbols": [],
        }
        automl_config = {"objective": "directional_accuracy"}
        search_space = {"model": {"type": {"type": "categorical", "choices": ["logistic"]}}}

        manifest_a = _build_experiment_manifest(
            {"data": {"symbol": "BTCUSDT", "interval": "1h"}, "features": {"schema_version": "schema_v1"}},
            automl_config,
            state_bundle,
            search_space,
        )
        manifest_b = _build_experiment_manifest(
            {"data": {"symbol": "BTCUSDT", "interval": "1h"}, "features": {"schema_version": "schema_v2"}},
            automl_config,
            state_bundle,
            search_space,
        )

        self.assertNotEqual(manifest_a["feature_schema_hash"], manifest_b["feature_schema_hash"])
        self.assertNotEqual(manifest_a["experiment_id"], manifest_b["experiment_id"])

    def test_completed_trials_carry_experiment_lineage_attrs(self):
        if automl_module.optuna is None:
            self.skipTest("optuna is not installed")

        temp_dir = tempfile.mkdtemp()
        try:
            base_pipeline = _build_base_pipeline(Path(temp_dir) / "study.db")
            with mock.patch("core.automl._sample_trial_overrides", return_value={}):
                summary = run_automl_study(base_pipeline, pipeline_class=_AutoMLManifestPipeline, trial_step_classes=[])

            manifest_path = Path(summary["experiment_artifacts"]["manifest"])
            summary_path = Path(summary["experiment_artifacts"]["summary"])
            self.assertTrue(manifest_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertIn("experiment_id", summary)
            self.assertEqual(summary["resume_mode"], "never")

            study = automl_module.optuna.load_study(
                study_name=summary["study_name"],
                storage=f"sqlite:///{Path(summary['storage']).as_posix()}",
            )
            completed_trials = [trial for trial in study.trials if trial.state == automl_module.optuna.trial.TrialState.COMPLETE]
            self.assertEqual(len(completed_trials), 1)
            trial = completed_trials[0]
            self.assertEqual(trial.user_attrs["experiment_id"], summary["experiment_id"])
            self.assertEqual(trial.user_attrs["resume_mode"], "never")
            self.assertEqual(trial.user_attrs["data_lineage_hash"], summary["experiment_manifest"]["data_lineage_hash"])
            self.assertEqual(trial.user_attrs["feature_schema_hash"], summary["experiment_manifest"]["feature_schema_hash"])
            self.assertEqual(trial.user_attrs["objective_hash"], summary["experiment_manifest"]["objective_hash"])
            self.assertEqual(trial.user_attrs["search_space_hash"], summary["experiment_manifest"]["search_space_hash"])
            storage_backend = getattr(study, "_storage", None)
            if storage_backend is not None and hasattr(storage_backend, "remove_session"):
                storage_backend.remove_session()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()