import copy
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from core.automl import compute_objective_value, run_automl_study


class _BasePipelineStub:
    def __init__(self, config, raw_data, data):
        self.config = copy.deepcopy(config)
        self.state = {
            "raw_data": raw_data,
            "data": data,
            "futures_context": {"mark_price": raw_data[["close"]].rename(columns={"close": "mark_close"})},
            "cross_asset_context": {"ETHUSDT": raw_data.copy()},
            "symbol_filters": {"tick_size": 0.1, "step_size": 0.001, "min_notional": 10.0},
        }

    def require(self, key):
        return self.state[key]


class _AutoMLDummyPipeline:
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
            test_size = int(self.config.get("model", {}).get("test_size", max(1, len(self.state["X"]) // 4)))
            oos_index = self.state["X"].index[-test_size:]
            has_full_state = all(key in self.state for key in ("futures_context", "cross_asset_context", "symbol_filters"))
            directional_accuracy = 0.9 if has_full_state else 0.1
            probability_frame = pd.DataFrame({-1: 0.2, 0: 0.0, 1: 0.8}, index=oos_index)
            signal_strength = pd.Series(1.0, index=oos_index)
            training = {
                "fold_metrics": [
                    {
                        "accuracy": directional_accuracy,
                        "f1_macro": directional_accuracy,
                        "directional_accuracy": directional_accuracy,
                        "directional_f1_macro": directional_accuracy,
                        "log_loss": 0.2,
                        "brier_score": 0.1,
                        "calibration_error": 0.05,
                    }
                ],
                "avg_accuracy": directional_accuracy,
                "avg_f1_macro": directional_accuracy,
                "avg_directional_accuracy": directional_accuracy,
                "avg_directional_f1_macro": directional_accuracy,
                "avg_log_loss": 0.2,
                "avg_brier_score": 0.1,
                "avg_calibration_error": 0.05,
                "headline_metrics": {
                    "directional_accuracy": directional_accuracy,
                    "log_loss": 0.2,
                    "brier_score": 0.1,
                    "calibration_error": 0.05,
                },
                "last_model": object(),
                "last_meta": object(),
                "last_selected_columns": ["feature"],
                "last_signal_params": {},
                "last_avg_win": 0.02,
                "last_avg_loss": 0.01,
                "oos_predictions": pd.Series(1, index=oos_index),
                "oos_probabilities": probability_frame,
                "oos_meta_prob": pd.Series(0.6, index=oos_index),
                "oos_profitability_prob": pd.Series(0.6, index=oos_index),
                "oos_direction_edge": pd.Series(0.6, index=oos_index),
                "oos_confidence": pd.Series(0.8, index=oos_index),
                "oos_expected_trade_edge": pd.Series(0.02, index=oos_index),
                "oos_position_size": signal_strength,
                "oos_kelly_size": signal_strength,
                "oos_event_signals": signal_strength,
                "oos_continuous_signals": signal_strength,
                "oos_signals": pd.Series(1, index=oos_index),
                "feature_block_diagnostics": {"summary": [], "top_features": [], "folds": []},
                "regime": {"mode": "fold_local", "folds": []},
                "stationarity": {"mode": "fold_local", "folds": []},
                "feature_selection": {"enabled": False, "mode": "fold_local", "avg_selected_features": 1.0, "folds": []},
                "purging": [],
                "signal_tuning": [],
                "oos_avg_win": 0.02,
                "oos_avg_loss": 0.01,
            }
            self.state["training"] = training
            self.step_results[name] = training
            return training

        if name == "generate_signals":
            index = self.state["training"]["oos_predictions"].index
            continuous = pd.Series(1.0, index=index)
            signals = {"signals": continuous.astype(int), "continuous_signals": continuous}
            self.state["signals"] = signals
            self.step_results[name] = signals
            return signals

        if name == "run_backtest":
            rows = len(self.state["signals"]["continuous_signals"])
            backtest = {
                "net_profit": float(rows),
                "net_profit_pct": float(rows) / 100.0,
                "sharpe_ratio": 1.0,
                "sortino_ratio": 1.0,
                "calmar_ratio": 1.0,
                "profit_factor": 1.5,
                "max_drawdown": -0.02,
                "total_trades": rows,
                "win_rate": 0.5,
                "ending_equity": 10_000.0 + float(rows),
            }
            self.state["backtest"] = backtest
            self.step_results[name] = backtest
            return backtest

        raise KeyError(name)


class AutoMLHoldoutObjectiveTest(unittest.TestCase):
    def test_accuracy_first_objective_prefers_better_directional_model(self):
        accurate_training = {
            "avg_accuracy": 0.64,
            "avg_directional_accuracy": 0.64,
            "avg_log_loss": 0.45,
            "avg_brier_score": 0.22,
            "avg_calibration_error": 0.06,
        }
        profitable_training = {
            "avg_accuracy": 0.55,
            "avg_directional_accuracy": 0.55,
            "avg_log_loss": 0.30,
            "avg_brier_score": 0.18,
            "avg_calibration_error": 0.04,
        }
        weak_backtest = {"net_profit_pct": -0.10, "sharpe_ratio": -0.5, "max_drawdown": -0.20, "total_trades": 20}
        strong_backtest = {"net_profit_pct": 0.50, "sharpe_ratio": 5.0, "max_drawdown": -0.05, "total_trades": 40}

        accurate_score = compute_objective_value("accuracy_first", accurate_training, weak_backtest)
        profitable_score = compute_objective_value("accuracy_first", profitable_training, strong_backtest)

        self.assertGreater(accurate_score, profitable_score)

    def test_run_automl_study_propagates_state_and_reports_locked_holdout(self):
        index = pd.date_range("2026-01-01", periods=120, freq="1h", tz="UTC")
        raw = pd.DataFrame(
            {
                "open": np.linspace(100.0, 110.0, len(index)),
                "high": np.linspace(100.5, 110.5, len(index)),
                "low": np.linspace(99.5, 109.5, len(index)),
                "close": np.linspace(100.0, 110.0, len(index)),
                "volume": 1_000.0,
                "quote_volume": np.linspace(100_000.0, 110_000.0, len(index)),
                "trades": 100,
            },
            index=index,
        )

        fd, storage_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(storage_path)

        base_pipeline = _BasePipelineStub(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "automl": {
                    "enabled": True,
                    "n_trials": 1,
                    "objective": "accuracy_first",
                    "seed": 7,
                    "locked_holdout_bars": 24,
                    "storage": storage_path,
                    "study_name": "automl_holdout_state_test",
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

        self.assertEqual(summary["objective"], "accuracy_first")
        self.assertAlmostEqual(float(summary["best_training"]["avg_directional_accuracy"]), 0.9, places=6)
        self.assertTrue(summary["locked_holdout"]["enabled"])
        self.assertEqual(int(summary["locked_holdout"]["search_rows"]), 96)
        self.assertEqual(int(summary["locked_holdout"]["holdout_rows"]), 24)
        self.assertEqual(int(summary["locked_holdout"]["aligned_holdout_rows"]), 24)
        self.assertEqual(int(summary["locked_holdout"]["backtest"]["total_trades"]), 24)


if __name__ == "__main__":
    unittest.main()