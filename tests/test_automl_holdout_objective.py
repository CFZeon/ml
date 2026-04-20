import copy
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import core.automl as automl_module
from core.automl import _build_trial_return_frame, compute_cpcv_pbo, compute_objective_value, run_automl_study


def _build_market_frame(rows):
    index = pd.date_range("2026-01-01", periods=rows, freq="1h", tz="UTC")
    return pd.DataFrame(
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


def _make_storage_path():
    fd, storage_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(storage_path)
    return storage_path


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
                "feature_selection": {
                    "enabled": False,
                    "mode": "fold_local",
                    "avg_input_features": 1.0,
                    "avg_selected_features": 1.0,
                    "folds": [],
                },
                "bootstrap": {"model_type": self.config.get("model", {}).get("type", "gbm"), "used_in_any_fold": False, "warning_count": 0, "folds": []},
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
            index = self.state["signals"]["continuous_signals"].index
            returns = np.linspace(0.001, 0.002, rows, dtype=float) if rows > 1 else np.array([0.001], dtype=float)
            equity_curve = pd.Series(10_000.0 * np.cumprod(1.0 + returns), index=index)
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
                "equity_curve": equity_curve,
                "statistical_significance": {
                    "enabled": True,
                    "metrics": {
                        "sharpe_ratio": {
                            "confidence_interval": {"lower": 0.1, "upper": 2.0},
                        }
                    },
                },
            }
            self.state["backtest"] = backtest
            self.step_results[name] = backtest
            return backtest

        raise KeyError(name)


class _ScenarioAutoMLPipeline(_AutoMLDummyPipeline):
    metrics_by_variant = {}
    full_rows = None
    validation_runs = 0

    @classmethod
    def reset(cls):
        cls.metrics_by_variant = {}
        cls.full_rows = None
        cls.validation_runs = 0

    def _resolve_variant(self):
        threshold = self.config.get("signals", {}).get("threshold")
        if threshold is not None:
            threshold_key = str(threshold)
            if threshold_key in type(self).metrics_by_variant:
                return threshold_key

        max_holding = self.config.get("labels", {}).get("max_holding")
        if max_holding is not None:
            max_holding_key = str(max_holding)
            if max_holding_key in type(self).metrics_by_variant:
                return max_holding_key

        max_features = self.config.get("feature_selection", {}).get("max_features")
        if max_features is not None:
            max_features_key = str(max_features)
            if max_features_key in type(self).metrics_by_variant:
                return max_features_key

        model_type = self.config.get("model", {}).get("type")
        if model_type is not None:
            return str(model_type)
        return str(threshold if threshold is not None else 0.0)

    def _resolve_phase(self):
        if "test_size" not in self.config.get("model", {}):
            return "search"
        if self.full_rows is not None and len(self.state["data"]) >= int(self.full_rows):
            return "holdout"
        return "validation"

    def _resolve_scenario(self):
        variant = self._resolve_variant()
        phase = self._resolve_phase()
        return dict(type(self).metrics_by_variant[variant][phase])

    def run_step(self, name):
        if name == "train_models":
            scenario = self._resolve_scenario()
            self.state["_scenario"] = scenario

            test_size = int(self.config.get("model", {}).get("test_size", max(1, len(self.state["X"]) // 4)))
            oos_index = self.state["X"].index[-test_size:]
            directional_accuracy = float(scenario.get("directional_accuracy", 0.6))
            log_loss = float(scenario.get("log_loss", 0.2))
            brier_score = float(scenario.get("brier_score", 0.1))
            calibration_error = float(scenario.get("calibration_error", 0.05))
            fold_metrics = copy.deepcopy(scenario.get("fold_metrics") or [
                {
                    "accuracy": directional_accuracy,
                    "f1_macro": directional_accuracy,
                    "directional_accuracy": directional_accuracy,
                    "directional_f1_macro": directional_accuracy,
                    "log_loss": log_loss,
                    "brier_score": brier_score,
                    "calibration_error": calibration_error,
                }
            ])
            fold_backtests = copy.deepcopy(scenario.get("fold_backtests") or [])

            def _avg_metric(rows, key, fallback=None):
                values = [float(row[key]) for row in rows if row.get(key) is not None]
                if values:
                    return float(np.mean(values))
                return fallback

            probability_frame = pd.DataFrame({-1: 0.2, 0: 0.0, 1: 0.8}, index=oos_index)
            signal_strength = pd.Series(1.0, index=oos_index)
            training = {
                "fold_metrics": fold_metrics,
                "fold_backtests": fold_backtests,
                "avg_accuracy": _avg_metric(fold_metrics, "accuracy", directional_accuracy),
                "avg_f1_macro": _avg_metric(fold_metrics, "f1_macro", directional_accuracy),
                "avg_directional_accuracy": _avg_metric(fold_metrics, "directional_accuracy", directional_accuracy),
                "avg_directional_f1_macro": _avg_metric(fold_metrics, "directional_f1_macro", directional_accuracy),
                "avg_log_loss": _avg_metric(fold_metrics, "log_loss", log_loss),
                "avg_brier_score": _avg_metric(fold_metrics, "brier_score", brier_score),
                "avg_calibration_error": _avg_metric(fold_metrics, "calibration_error", calibration_error),
                "headline_metrics": {
                    "directional_accuracy": _avg_metric(fold_metrics, "directional_accuracy", directional_accuracy),
                    "log_loss": _avg_metric(fold_metrics, "log_loss", log_loss),
                    "brier_score": _avg_metric(fold_metrics, "brier_score", brier_score),
                    "calibration_error": _avg_metric(fold_metrics, "calibration_error", calibration_error),
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
                "fold_stability": copy.deepcopy(scenario.get("fold_stability")),
                "feature_selection": {
                    "enabled": False,
                    "mode": "fold_local",
                    "avg_input_features": 1.0,
                    "avg_selected_features": 1.0,
                    "folds": [],
                },
                "bootstrap": {"model_type": self.config.get("model", {}).get("type", "gbm"), "used_in_any_fold": False, "warning_count": 0, "folds": []},
                "purging": [],
                "signal_tuning": [],
                "oos_avg_win": 0.02,
                "oos_avg_loss": 0.01,
            }
            self.state["training"] = training
            self.step_results[name] = training
            return training

        if name == "run_backtest":
            scenario = self.state.get("_scenario") or self._resolve_scenario()
            phase = self._resolve_phase()
            if phase == "validation":
                type(self).validation_runs += 1

            index = self.state["signals"]["continuous_signals"].index
            returns = scenario.get("returns")
            if returns is None:
                if len(index) <= 1:
                    returns = np.array([float(scenario.get("return_start", 0.001))], dtype=float)
                else:
                    returns = np.linspace(
                        float(scenario.get("return_start", 0.001)),
                        float(scenario.get("return_end", 0.002)),
                        len(index),
                        dtype=float,
                    )
            else:
                returns = np.asarray(returns, dtype=float)
                if returns.size != len(index):
                    returns = np.resize(returns, len(index))

            equity_curve = pd.Series(10_000.0 * np.cumprod(1.0 + returns), index=index)
            backtest = {
                "net_profit": float(scenario.get("net_profit", len(index))),
                "net_profit_pct": float(scenario.get("net_profit_pct", np.sum(returns))),
                "sharpe_ratio": float(scenario.get("sharpe_ratio", 1.0)),
                "sortino_ratio": float(scenario.get("sortino_ratio", scenario.get("sharpe_ratio", 1.0))),
                "calmar_ratio": float(scenario.get("calmar_ratio", scenario.get("sharpe_ratio", 1.0))),
                "profit_factor": float(scenario.get("profit_factor", 1.5)),
                "max_drawdown": float(scenario.get("max_drawdown", -0.02)),
                "total_trades": len(index),
                "win_rate": float(scenario.get("win_rate", 0.5)),
                "ending_equity": float(equity_curve.iloc[-1]),
                "equity_curve": equity_curve,
                "statistical_significance": {
                    "enabled": True,
                    "metrics": {
                        "sharpe_ratio": {
                            "confidence_interval": {
                                "lower": float(scenario.get("ci_lower", 0.1)),
                                "upper": float(scenario.get("ci_upper", 2.0)),
                            },
                        }
                    },
                },
            }
            self.state["backtest"] = backtest
            self.step_results[name] = backtest
            return backtest

        return super().run_step(name)


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

    def test_risk_adjusted_after_costs_uses_backtest_and_gates_classification_quality(self):
        strong_backtest = {
            "net_profit_pct": 0.12,
            "sharpe_ratio": 1.8,
            "max_drawdown": -0.05,
            "total_trades": 18,
            "bar_count": 72,
            "statistical_significance": {
                "metrics": {
                    "sharpe_ratio": {"confidence_interval": {"lower": 0.7, "upper": 2.6}},
                }
            },
        }
        weak_classifier = {
            "avg_accuracy": 0.48,
            "avg_directional_accuracy": 0.48,
            "avg_log_loss": 0.22,
            "avg_brier_score": 0.12,
            "avg_calibration_error": 0.04,
        }
        acceptable_classifier = {
            "avg_accuracy": 0.58,
            "avg_directional_accuracy": 0.58,
            "avg_log_loss": 0.24,
            "avg_brier_score": 0.13,
            "avg_calibration_error": 0.05,
        }

        rejected_score = compute_objective_value(
            "risk_adjusted_after_costs",
            weak_classifier,
            strong_backtest,
            {"objective_gates": {"min_directional_accuracy": 0.55, "min_trade_count": 1}},
        )
        accepted_score = compute_objective_value(
            "risk_adjusted_after_costs",
            acceptable_classifier,
            strong_backtest,
            {"objective_gates": {"min_directional_accuracy": 0.55, "min_trade_count": 1}},
        )

        self.assertEqual(rejected_score, float("-inf"))
        self.assertGreater(accepted_score, 0.0)

    def test_risk_adjusted_after_costs_can_use_confidence_lower_bound(self):
        training = {
            "avg_accuracy": 0.60,
            "avg_directional_accuracy": 0.60,
            "avg_log_loss": 0.22,
            "avg_brier_score": 0.12,
            "avg_calibration_error": 0.05,
        }
        backtest = {
            "net_profit_pct": 0.10,
            "sharpe_ratio": 2.0,
            "max_drawdown": -0.05,
            "total_trades": 12,
            "bar_count": 120,
            "statistical_significance": {
                "metrics": {
                    "sharpe_ratio": {"confidence_interval": {"lower": 0.8, "upper": 2.9}},
                }
            },
        }

        point_estimate_score = compute_objective_value(
            "risk_adjusted_after_costs",
            training,
            backtest,
            {"objective_gates": {"min_trade_count": 1}, "objective_use_confidence_lower_bound": False},
        )
        lower_bound_score = compute_objective_value(
            "risk_adjusted_after_costs",
            training,
            backtest,
            {"objective_gates": {"min_trade_count": 1}, "objective_use_confidence_lower_bound": True},
        )

        self.assertGreater(point_estimate_score, lower_bound_score)

    def test_compute_objective_value_can_apply_deflated_sharpe_penalty(self):
        raw_score = compute_objective_value("sharpe_ratio", {}, {"sharpe_ratio": 2.5})
        penalized_score = compute_objective_value(
            "sharpe_ratio",
            {},
            {"sharpe_ratio": 2.5},
            overfitting_context={"apply_penalty": True, "deflated_sharpe_ratio": 0.37},
        )

        self.assertAlmostEqual(raw_score, 2.5, places=6)
        self.assertAlmostEqual(penalized_score, 0.37, places=6)

    def test_compute_cpcv_pbo_reports_consistent_winner(self):
        index = pd.date_range("2026-01-01", periods=24, freq="1h", tz="UTC")
        trial_returns = pd.DataFrame(
            {
                0: np.tile([0.012, 0.008, 0.011, 0.009], 6),
                1: np.tile([0.010, -0.010, 0.012, -0.012], 6),
                2: np.tile([-0.004, 0.002, -0.003, 0.001], 6),
            },
            index=index,
        )

        report = compute_cpcv_pbo(trial_returns, n_blocks=6, test_blocks=3, min_block_size=2)

        self.assertTrue(report["enabled"])
        self.assertGreaterEqual(int(report["split_count"]), 1)
        self.assertLess(float(report["probability_of_backtest_overfitting"]), 0.5)

    def test_compute_cpcv_pbo_requires_multiple_trials(self):
        index = pd.date_range("2026-01-01", periods=8, freq="1h", tz="UTC")
        trial_returns = pd.DataFrame({0: np.linspace(0.001, 0.008, len(index))}, index=index)

        report = compute_cpcv_pbo(trial_returns, n_blocks=4, test_blocks=2, min_block_size=2)

        self.assertFalse(report["enabled"])
        self.assertEqual(report["reason"], "insufficient_trials")

    def test_build_trial_return_frame_preserves_missing_periods(self):
        index = pd.date_range("2026-01-01", periods=8, freq="1h", tz="UTC")
        trial_records = {
            0: {"returns": pd.Series([0.012, 0.011, 0.013, 0.010, 0.014, 0.012, 0.011, 0.013], index=index)},
            1: {"returns": pd.Series([0.010, 0.009, 0.011, 0.010, 0.009, 0.010], index=index[[0, 1, 3, 4, 6, 7]])},
            2: {"returns": pd.Series([0.004, 0.006, 0.005, 0.004, 0.005, 0.004], index=index[[0, 2, 3, 4, 5, 7]])},
        }

        frame = _build_trial_return_frame(trial_records)

        self.assertEqual(frame.shape, (8, 3))
        self.assertTrue(pd.isna(frame.loc[index[2], 1]))
        self.assertTrue(pd.isna(frame.loc[index[1], 2]))
        self.assertGreater(int(frame.isna().sum().sum()), 0)

    def test_compute_cpcv_pbo_strict_intersection_reports_overlap_diagnostics(self):
        index = pd.date_range("2026-01-01", periods=8, freq="1h", tz="UTC")
        trial_returns = pd.DataFrame(
            {
                0: [0.012, 0.011, 0.013, 0.010, 0.014, 0.012, 0.011, 0.013],
                1: [0.010, 0.009, np.nan, 0.010, 0.011, np.nan, 0.009, 0.010],
                2: [0.004, np.nan, 0.006, 0.005, 0.004, 0.005, np.nan, 0.004],
            },
            index=index,
        )

        report = compute_cpcv_pbo(
            trial_returns,
            n_blocks=2,
            test_blocks=1,
            min_block_size=2,
            overlap_policy="strict_intersection",
            min_overlap_fraction=0.4,
        )

        self.assertTrue(report["enabled"])
        self.assertEqual(report["overlap_policy"], "strict_intersection")
        self.assertEqual(report["path_rows"], 8)
        self.assertEqual(report["strict_overlap_rows"], 4)
        self.assertAlmostEqual(float(report["strict_overlap_fraction"]), 0.5, places=6)
        self.assertEqual(int(report["excluded_low_overlap_split_count"]), 0)

    def test_compute_cpcv_pbo_pairwise_overlap_excludes_weak_comparisons(self):
        index = pd.date_range("2026-01-01", periods=12, freq="1h", tz="UTC")
        trial_returns = pd.DataFrame(
            {
                0: [0.012, 0.011, 0.013, 0.012, 0.011, 0.010, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                1: [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.008, 0.009, 0.007, 0.008, 0.009, 0.007],
                2: [0.005, 0.004, 0.006, 0.005, 0.004, 0.006, 0.005, 0.004, 0.006, 0.005, 0.004, 0.006],
            },
            index=index,
        )

        report = compute_cpcv_pbo(
            trial_returns,
            n_blocks=4,
            test_blocks=2,
            min_block_size=2,
            overlap_policy="pairwise_overlap",
            min_overlap_fraction=0.4,
            min_overlap_observations=2,
        )

        self.assertFalse(report["enabled"])
        self.assertEqual(report["reason"], "no_valid_splits")
        self.assertEqual(report["overlap_policy"], "pairwise_overlap")
        self.assertGreater(int(report["excluded_low_overlap_split_count"]), 0)
        self.assertGreater(int(report["excluded_low_overlap_trial_pairs"]), 0)
        self.assertLess(float(report["pairwise_overlap_min_fraction"]), 0.4)

    def test_run_automl_study_propagates_state_and_reports_locked_holdout(self):
        raw = _build_market_frame(120)
        storage_path = _make_storage_path()

        base_pipeline = _BasePipelineStub(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "automl": {
                    "enabled": True,
                    "n_trials": 1,
                    "objective": "accuracy_first",
                    "seed": 7,
                    "validation_fraction": 0.2,
                    "locked_holdout_bars": 24,
                    "locked_holdout_min_search_rows": 48,
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
        self.assertEqual(summary["selection_metric"], "accuracy_first")
        self.assertAlmostEqual(float(summary["best_training"]["avg_directional_accuracy"]), 0.9, places=6)
        self.assertIn("overfitting_diagnostics", summary)
        self.assertTrue(summary["overfitting_diagnostics"]["enabled"])
        self.assertFalse(summary["overfitting_diagnostics"]["pbo"]["enabled"])
        self.assertIn("bootstrap", summary["best_training"])
        self.assertFalse(summary["best_training"]["bootstrap"]["used_in_any_fold"])
        self.assertTrue(summary["validation_holdout"]["enabled"])
        self.assertEqual(int(summary["validation_holdout"]["search_rows"]), 68)
        self.assertEqual(int(summary["validation_holdout"]["validation_rows"]), 24)
        self.assertEqual(int(summary["validation_holdout"]["search_validation_gap_rows"]), 2)
        self.assertEqual(int(summary["validation_holdout"]["aligned_gap_rows"]), 2)
        self.assertEqual(int(summary["validation_holdout"]["aligned_validation_rows"]), 24)
        self.assertTrue(summary["locked_holdout"]["enabled"])
        self.assertEqual(int(summary["locked_holdout"]["search_rows"]), 68)
        self.assertEqual(int(summary["locked_holdout"]["validation_rows"]), 24)
        self.assertEqual(int(summary["locked_holdout"]["validation_holdout_gap_rows"]), 2)
        self.assertEqual(int(summary["locked_holdout"]["pre_holdout_rows"]), 92)
        self.assertEqual(int(summary["locked_holdout"]["holdout_rows"]), 24)
        self.assertEqual(int(summary["locked_holdout"]["aligned_gap_rows"]), 4)
        self.assertEqual(int(summary["locked_holdout"]["aligned_pre_holdout_rows"]), 92)
        self.assertEqual(int(summary["locked_holdout"]["aligned_holdout_rows"]), 24)
        self.assertEqual(int(summary["locked_holdout"]["backtest"]["total_trades"]), 24)
        self.assertFalse(summary["locked_holdout"]["holdout_warning"])

    def test_default_automl_search_space_excludes_signal_policy_knobs(self):
        self.assertNotIn("signals", automl_module.DEFAULT_AUTOML_SEARCH_SPACE)

    def test_run_automl_study_rejects_signal_policy_search_space(self):
        raw = _build_market_frame(80)
        storage_path = _make_storage_path()

        base_pipeline = _BasePipelineStub(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "automl": {
                    "enabled": True,
                    "n_trials": 1,
                    "objective": "accuracy_first",
                    "seed": 11,
                    "validation_fraction": 0.2,
                    "locked_holdout_fraction": 0.2,
                    "locked_holdout_min_search_rows": 32,
                    "storage": storage_path,
                    "study_name": "automl_signal_policy_rejection_test",
                    "search_space": {
                        "signals": {
                            "threshold": {"type": "categorical", "choices": [0.01, 0.02]},
                        }
                    },
                },
                "model": {"type": "gbm"},
            },
            raw_data=raw,
            data=raw.copy(),
        )

        with self.assertRaisesRegex(ValueError, "signal-policy search is disabled"):
            run_automl_study(
                base_pipeline,
                pipeline_class=_AutoMLDummyPipeline,
                trial_step_classes=[],
            )

    def test_default_objective_prefers_better_after_cost_backtest(self):
        raw = _build_market_frame(100)
        storage_path = _make_storage_path()

        _ScenarioAutoMLPipeline.reset()
        _ScenarioAutoMLPipeline.full_rows = len(raw)
        _ScenarioAutoMLPipeline.metrics_by_variant = {
            "rf": {
                "search": {
                    "directional_accuracy": 0.72,
                    "log_loss": 0.20,
                    "calibration_error": 0.04,
                    "sharpe_ratio": -0.4,
                    "returns": [-0.0010, 0.0003, -0.0011, 0.0002],
                },
                "validation": {
                    "directional_accuracy": 0.72,
                    "log_loss": 0.20,
                    "calibration_error": 0.04,
                    "sharpe_ratio": -0.5,
                    "returns": [-0.0012, 0.0002, -0.0011, 0.0001],
                },
                "holdout": {
                    "directional_accuracy": 0.72,
                    "log_loss": 0.20,
                    "calibration_error": 0.04,
                    "sharpe_ratio": -0.3,
                    "returns": [-0.0009, 0.0002, -0.0008, 0.0001],
                },
            },
            "gbm": {
                "search": {
                    "directional_accuracy": 0.58,
                    "log_loss": 0.28,
                    "calibration_error": 0.08,
                    "sharpe_ratio": 1.2,
                    "returns": [0.0012, 0.0008, 0.0013, 0.0009],
                },
                "validation": {
                    "directional_accuracy": 0.58,
                    "log_loss": 0.28,
                    "calibration_error": 0.08,
                    "sharpe_ratio": 1.4,
                    "returns": [0.0014, 0.0010, 0.0015, 0.0011],
                },
                "holdout": {
                    "directional_accuracy": 0.58,
                    "log_loss": 0.28,
                    "calibration_error": 0.08,
                    "sharpe_ratio": 1.1,
                    "returns": [0.0010, 0.0007, 0.0011, 0.0008],
                },
            },
        }

        base_pipeline = _BasePipelineStub(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "automl": {
                    "enabled": True,
                    "n_trials": 2,
                    "seed": 37,
                    "validation_fraction": 0.2,
                    "locked_holdout_fraction": 0.2,
                    "locked_holdout_min_search_rows": 40,
                    "enable_pruning": False,
                    "minimum_dsr_threshold": None,
                    "storage": storage_path,
                    "study_name": "automl_trading_first_default_test",
                    "selection_policy": {"min_validation_trade_count": 1, "require_locked_holdout_pass": False},
                },
                "model": {"type": "rf"},
            },
            raw_data=raw,
            data=raw.copy(),
        )

        variants = [{"model": {"type": "rf"}}, {"model": {"type": "gbm"}}]
        with mock.patch("core.automl._sample_trial_overrides", side_effect=lambda trial, _: variants[trial.number]):
            summary = run_automl_study(
                base_pipeline,
                pipeline_class=_ScenarioAutoMLPipeline,
                trial_step_classes=[],
            )

        self.assertEqual(summary["objective"], "risk_adjusted_after_costs")
        self.assertEqual(summary["best_overrides"]["model"]["type"], "gbm")
        self.assertTrue(summary["best_objective_diagnostics"]["classification_gates"]["passed"])

    def test_explicit_accuracy_first_override_still_works(self):
        raw = _build_market_frame(100)
        storage_path = _make_storage_path()

        _ScenarioAutoMLPipeline.reset()
        _ScenarioAutoMLPipeline.full_rows = len(raw)
        _ScenarioAutoMLPipeline.metrics_by_variant = {
            "rf": {
                "search": {
                    "directional_accuracy": 0.72,
                    "log_loss": 0.20,
                    "calibration_error": 0.04,
                    "sharpe_ratio": -0.4,
                    "returns": [-0.0010, 0.0003, -0.0011, 0.0002],
                },
                "validation": {
                    "directional_accuracy": 0.72,
                    "log_loss": 0.20,
                    "calibration_error": 0.04,
                    "sharpe_ratio": -0.5,
                    "returns": [-0.0012, 0.0002, -0.0011, 0.0001],
                },
                "holdout": {
                    "directional_accuracy": 0.72,
                    "log_loss": 0.20,
                    "calibration_error": 0.04,
                    "sharpe_ratio": -0.3,
                    "returns": [-0.0009, 0.0002, -0.0008, 0.0001],
                },
            },
            "gbm": {
                "search": {
                    "directional_accuracy": 0.58,
                    "log_loss": 0.28,
                    "calibration_error": 0.08,
                    "sharpe_ratio": 1.2,
                    "returns": [0.0012, 0.0008, 0.0013, 0.0009],
                },
                "validation": {
                    "directional_accuracy": 0.58,
                    "log_loss": 0.28,
                    "calibration_error": 0.08,
                    "sharpe_ratio": 1.4,
                    "returns": [0.0014, 0.0010, 0.0015, 0.0011],
                },
                "holdout": {
                    "directional_accuracy": 0.58,
                    "log_loss": 0.28,
                    "calibration_error": 0.08,
                    "sharpe_ratio": 1.1,
                    "returns": [0.0010, 0.0007, 0.0011, 0.0008],
                },
            },
        }

        base_pipeline = _BasePipelineStub(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "automl": {
                    "enabled": True,
                    "n_trials": 2,
                    "objective": "accuracy_first",
                    "seed": 41,
                    "validation_fraction": 0.2,
                    "locked_holdout_fraction": 0.2,
                    "locked_holdout_min_search_rows": 40,
                    "enable_pruning": False,
                    "minimum_dsr_threshold": None,
                    "storage": storage_path,
                    "study_name": "automl_accuracy_override_test",
                    "selection_policy": {"min_validation_trade_count": 1, "require_locked_holdout_pass": False},
                },
                "model": {"type": "rf"},
            },
            raw_data=raw,
            data=raw.copy(),
        )

        variants = [{"model": {"type": "rf"}}, {"model": {"type": "gbm"}}]
        with mock.patch("core.automl._sample_trial_overrides", side_effect=lambda trial, _: variants[trial.number]):
            summary = run_automl_study(
                base_pipeline,
                pipeline_class=_ScenarioAutoMLPipeline,
                trial_step_classes=[],
            )

        self.assertEqual(summary["objective"], "accuracy_first")
        self.assertEqual(summary["best_overrides"]["model"]["type"], "rf")

    def test_two_stage_holdout_validation_used_for_ranking(self):
        raw = _build_market_frame(100)
        storage_path = _make_storage_path()

        _ScenarioAutoMLPipeline.reset()
        _ScenarioAutoMLPipeline.full_rows = len(raw)
        _ScenarioAutoMLPipeline.metrics_by_variant = {
            "rf": {
                "search": {"sharpe_ratio": 2.0, "returns": [0.0016, 0.0012, 0.0017, 0.0013]},
                "validation": {"sharpe_ratio": 0.4, "returns": [0.0005, 0.0002, 0.0006, 0.0003]},
                "holdout": {"sharpe_ratio": 0.3, "returns": [0.0004, 0.0002, 0.0005, 0.0003]},
            },
            "gbm": {
                "search": {"sharpe_ratio": 0.8, "returns": [0.0007, 0.0003, 0.0008, 0.0004]},
                "validation": {"sharpe_ratio": 1.6, "returns": [0.0022, 0.0014, 0.0024, 0.0016]},
                "holdout": {"sharpe_ratio": 1.2, "returns": [0.0018, 0.0011, 0.0020, 0.0012]},
            },
        }

        base_pipeline = _BasePipelineStub(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "automl": {
                    "enabled": True,
                    "n_trials": 2,
                    "objective": "sharpe_ratio",
                    "seed": 11,
                    "validation_fraction": 0.2,
                    "locked_holdout_fraction": 0.2,
                    "locked_holdout_min_search_rows": 40,
                    "enable_pruning": False,
                    "storage": storage_path,
                    "study_name": "automl_validation_ranking_test",
                },
                "model": {"type": "rf"},
            },
            raw_data=raw,
            data=raw.copy(),
        )

        variants = ["rf", "gbm"]
        with mock.patch("core.automl._sample_trial_overrides", side_effect=lambda trial, _: {"model": {"type": variants[trial.number]}}):
            summary = run_automl_study(
                base_pipeline,
                pipeline_class=_ScenarioAutoMLPipeline,
                trial_step_classes=[],
            )

        self.assertEqual(summary["best_overrides"]["model"]["type"], "gbm")
        self.assertAlmostEqual(float(summary["best_backtest"]["sharpe_ratio"]), 1.6, places=6)
        self.assertAlmostEqual(float(summary["validation_holdout"]["backtest"]["sharpe_ratio"]), 1.6, places=6)

    def test_dsr_threshold_rejects_low_dsr_trials(self):
        raw = _build_market_frame(100)
        storage_path = _make_storage_path()

        _ScenarioAutoMLPipeline.reset()
        _ScenarioAutoMLPipeline.full_rows = len(raw)
        _ScenarioAutoMLPipeline.metrics_by_variant = {
            "rf": {
                "search": {"sharpe_ratio": 1.8, "returns": [0.0018, 0.0014, 0.0019, 0.0015]},
                "validation": {"sharpe_ratio": 2.5, "returns": [0.0]},
                "holdout": {"sharpe_ratio": 0.2, "returns": [0.0002, 0.0001, 0.0003, 0.0001]},
            },
            "gbm": {
                "search": {"sharpe_ratio": 1.0, "returns": [0.0010, 0.0006, 0.0011, 0.0007]},
                "validation": {"sharpe_ratio": 1.2, "returns": [0.0020, 0.0010, 0.0022, 0.0012]},
                "holdout": {"sharpe_ratio": 0.9, "returns": [0.0017, 0.0009, 0.0018, 0.0010]},
            },
        }

        base_pipeline = _BasePipelineStub(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "automl": {
                    "enabled": True,
                    "n_trials": 2,
                    "objective": "sharpe_ratio",
                    "seed": 13,
                    "validation_fraction": 0.2,
                    "locked_holdout_fraction": 0.2,
                    "locked_holdout_min_search_rows": 40,
                    "enable_pruning": False,
                    "minimum_dsr_threshold": 0.3,
                    "storage": storage_path,
                    "study_name": "automl_dsr_threshold_test",
                },
                "model": {"type": "rf"},
            },
            raw_data=raw,
            data=raw.copy(),
        )

        variants = ["rf", "gbm"]
        with mock.patch("core.automl._sample_trial_overrides", side_effect=lambda trial, _: {"model": {"type": variants[trial.number]}}):
            summary = run_automl_study(
                base_pipeline,
                pipeline_class=_ScenarioAutoMLPipeline,
                trial_step_classes=[],
            )

        self.assertEqual(summary["best_overrides"]["model"]["type"], "gbm")
        self.assertTrue(summary["top_trials"][0]["meets_minimum_dsr_threshold"])
        self.assertFalse(summary["top_trials"][1]["meets_minimum_dsr_threshold"])
        self.assertEqual(summary["top_trials"][1]["value"], float("-inf"))

    def test_optuna_pruner_reduces_trial_count(self):
        raw = _build_market_frame(100)
        storage_path = _make_storage_path()

        _ScenarioAutoMLPipeline.reset()
        _ScenarioAutoMLPipeline.full_rows = len(raw)
        _ScenarioAutoMLPipeline.metrics_by_variant = {
            "0.0": {
                "search": {"sharpe_ratio": 1.0, "returns": [0.0016, 0.0011, 0.0018, 0.0012]},
                "validation": {"sharpe_ratio": 1.0, "returns": [0.0016, 0.0011, 0.0018, 0.0012]},
                "holdout": {"sharpe_ratio": 0.8, "returns": [0.0013, 0.0008, 0.0014, 0.0009]},
            },
            "1.0": {
                "search": {"sharpe_ratio": 0.9, "returns": [0.0015, 0.0010, 0.0016, 0.0011]},
                "validation": {"sharpe_ratio": 0.9, "returns": [0.0015, 0.0010, 0.0016, 0.0011]},
                "holdout": {"sharpe_ratio": 0.7, "returns": [0.0012, 0.0007, 0.0013, 0.0008]},
            },
            "2.0": {
                "search": {"sharpe_ratio": 1.1, "returns": [0.0017, 0.0012, 0.0019, 0.0013]},
                "validation": {"sharpe_ratio": 1.1, "returns": [0.0017, 0.0012, 0.0019, 0.0013]},
                "holdout": {"sharpe_ratio": 0.9, "returns": [0.0014, 0.0009, 0.0015, 0.0010]},
            },
            "3.0": {
                "search": {"sharpe_ratio": 1.05, "returns": [0.00165, 0.00115, 0.00185, 0.00125]},
                "validation": {"sharpe_ratio": 1.05, "returns": [0.00165, 0.00115, 0.00185, 0.00125]},
                "holdout": {"sharpe_ratio": 0.85, "returns": [0.00135, 0.00085, 0.00145, 0.00095]},
            },
            "4.0": {
                "search": {"sharpe_ratio": 0.95, "returns": [0.00155, 0.00105, 0.00175, 0.00115]},
                "validation": {"sharpe_ratio": 0.95, "returns": [0.00155, 0.00105, 0.00175, 0.00115]},
                "holdout": {"sharpe_ratio": 0.75, "returns": [0.00125, 0.00075, 0.00135, 0.00085]},
            },
            "5.0": {
                "search": {"sharpe_ratio": -10.0, "returns": [-0.0020, -0.0015, -0.0021, -0.0016]},
                "validation": {"sharpe_ratio": -10.0, "returns": [-0.0020, -0.0015, -0.0021, -0.0016]},
                "holdout": {"sharpe_ratio": -10.0, "returns": [-0.0020, -0.0015, -0.0021, -0.0016]},
            },
        }

        base_pipeline = _BasePipelineStub(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "automl": {
                    "enabled": True,
                    "n_trials": 6,
                    "objective": "sharpe_ratio",
                    "seed": 17,
                    "validation_fraction": 0.2,
                    "locked_holdout_fraction": 0.2,
                    "locked_holdout_min_search_rows": 40,
                    "enable_pruning": True,
                    "minimum_dsr_threshold": None,
                    "storage": storage_path,
                    "study_name": "automl_pruner_test",
                },
                "model": {"type": "gbm"},
            },
            raw_data=raw,
            data=raw.copy(),
        )

        with mock.patch(
            "core.automl._sample_trial_overrides",
            side_effect=lambda trial, _: {"labels": {"pt_sl": (2.0, 2.0), "max_holding": float(trial.number)}},
        ):
            summary = run_automl_study(
                base_pipeline,
                pipeline_class=_ScenarioAutoMLPipeline,
                trial_step_classes=[],
            )

        self.assertEqual(summary["trial_count"], 5)
        self.assertEqual(_ScenarioAutoMLPipeline.validation_runs, 5)

    def test_locked_holdout_warns_when_sharpe_ci_lower_is_negative(self):
        raw = _build_market_frame(100)
        storage_path = _make_storage_path()

        _ScenarioAutoMLPipeline.reset()
        _ScenarioAutoMLPipeline.full_rows = len(raw)
        _ScenarioAutoMLPipeline.metrics_by_variant = {
            "gbm": {
                "search": {"sharpe_ratio": 1.0, "returns": [0.0015, 0.0010, 0.0017, 0.0011]},
                "validation": {"sharpe_ratio": 1.1, "returns": [0.0017, 0.0012, 0.0018, 0.0013]},
                "holdout": {
                    "sharpe_ratio": 0.2,
                    "returns": [0.0008, -0.0007, 0.0006, -0.0005],
                    "ci_lower": -0.1,
                    "ci_upper": 0.4,
                },
            },
        }

        base_pipeline = _BasePipelineStub(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "automl": {
                    "enabled": True,
                    "n_trials": 1,
                    "objective": "sharpe_ratio",
                    "seed": 19,
                    "validation_fraction": 0.2,
                    "locked_holdout_fraction": 0.2,
                    "locked_holdout_min_search_rows": 40,
                    "enable_pruning": False,
                    "storage": storage_path,
                    "study_name": "automl_holdout_warning_test",
                },
                "model": {"type": "gbm"},
            },
            raw_data=raw,
            data=raw.copy(),
        )

        with mock.patch("core.automl._sample_trial_overrides", side_effect=lambda trial, _: {"model": {"type": "gbm"}}):
            summary = run_automl_study(
                base_pipeline,
                pipeline_class=_ScenarioAutoMLPipeline,
                trial_step_classes=[],
            )

        self.assertTrue(summary["locked_holdout"]["holdout_warning"])

    def test_locked_holdout_is_evaluated_once_after_selection_freeze(self):
        raw = _build_market_frame(100)
        storage_path = _make_storage_path()

        _ScenarioAutoMLPipeline.reset()
        _ScenarioAutoMLPipeline.full_rows = len(raw)
        _ScenarioAutoMLPipeline.metrics_by_variant = {
            "rf": {
                "search": {"sharpe_ratio": 2.2, "returns": [0.0018, 0.0014, 0.0019, 0.0015]},
                "validation": {"sharpe_ratio": 2.0, "returns": [0.0017, 0.0013, 0.0018, 0.0014]},
                "holdout": {
                    "sharpe_ratio": 0.2,
                    "returns": [0.0008, -0.0007, 0.0006, -0.0005],
                    "ci_lower": -0.1,
                    "ci_upper": 0.4,
                },
            },
            "gbm": {
                "search": {"sharpe_ratio": 1.8, "returns": [0.0015, 0.0011, 0.0016, 0.0012]},
                "validation": {"sharpe_ratio": 1.7, "returns": [0.0014, 0.0010, 0.0015, 0.0011]},
                "holdout": {"sharpe_ratio": 1.6, "returns": [0.0013, 0.0009, 0.0014, 0.0010]},
            },
        }

        base_pipeline = _BasePipelineStub(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "automl": {
                    "enabled": True,
                    "n_trials": 2,
                    "objective": "sharpe_ratio",
                    "seed": 43,
                    "validation_fraction": 0.2,
                    "locked_holdout_fraction": 0.2,
                    "locked_holdout_min_search_rows": 40,
                    "enable_pruning": False,
                    "storage": storage_path,
                    "study_name": "automl_locked_holdout_freeze_test",
                    "selection_policy": {
                        "min_validation_trade_count": 1,
                        "require_locked_holdout_pass": True,
                        "min_locked_holdout_score": 0.5,
                    },
                },
                "model": {"type": "rf"},
            },
            raw_data=raw,
            data=raw.copy(),
        )

        variants = [{"model": {"type": "rf"}}, {"model": {"type": "gbm"}}]
        with mock.patch("core.automl._sample_trial_overrides", side_effect=lambda trial, _: variants[trial.number]):
            with mock.patch("core.automl._evaluate_locked_holdout", wraps=automl_module._evaluate_locked_holdout) as holdout_eval:
                summary = run_automl_study(
                    base_pipeline,
                    pipeline_class=_ScenarioAutoMLPipeline,
                    trial_step_classes=[],
                )

        self.assertEqual(holdout_eval.call_count, 1)
        self.assertEqual(summary["best_overrides"]["model"]["type"], "rf")
        self.assertFalse(summary["promotion_ready"])
        self.assertIn("locked_holdout_failed", summary["promotion_reasons"])
        self.assertEqual(int(summary["locked_holdout"]["access_count"]), 1)
        self.assertTrue(summary["locked_holdout"]["evaluated_once"])
        self.assertTrue(summary["locked_holdout"]["evaluated_after_freeze"])
        self.assertTrue(summary["best_selection_policy"]["selection_policy"]["frozen"])
        self.assertFalse(summary["best_selection_policy"]["selection_policy"]["holdout_consulted_for_selection"])

    def test_selection_freeze_summary_records_candidate_hash_and_holdout_diagnostics(self):
        raw = _build_market_frame(120)
        storage_path = _make_storage_path()

        _ScenarioAutoMLPipeline.reset()
        _ScenarioAutoMLPipeline.full_rows = len(raw)
        _ScenarioAutoMLPipeline.metrics_by_variant = {
            "gbm": {
                "search": {"sharpe_ratio": 1.3, "returns": [0.0014, 0.0010, 0.0015, 0.0011]},
                "validation": {"sharpe_ratio": 1.2, "returns": [0.0013, 0.0009, 0.0014, 0.0010]},
                "holdout": {"sharpe_ratio": 1.1, "returns": [0.0012, 0.0008, 0.0013, 0.0009]},
            },
        }

        base_pipeline = _BasePipelineStub(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "automl": {
                    "enabled": True,
                    "n_trials": 1,
                    "objective": "sharpe_ratio",
                    "seed": 47,
                    "validation_fraction": 0.2,
                    "locked_holdout_bars": 24,
                    "locked_holdout_min_search_rows": 48,
                    "enable_pruning": False,
                    "storage": storage_path,
                    "study_name": "automl_selection_snapshot_test",
                },
                "model": {"type": "gbm"},
            },
            raw_data=raw,
            data=raw.copy(),
        )

        with mock.patch("core.automl._sample_trial_overrides", side_effect=lambda trial, _: {"model": {"type": "gbm"}}):
            summary = run_automl_study(
                base_pipeline,
                pipeline_class=_ScenarioAutoMLPipeline,
                trial_step_classes=[],
            )

        self.assertEqual(summary["selection_freeze"]["trial_number"], int(summary["best_trial_number"]))
        self.assertEqual(
            summary["selection_freeze"]["candidate_hash"],
            summary["locked_holdout"]["frozen_candidate_hash"],
        )
        self.assertEqual(
            summary["overfitting_diagnostics"]["selection_freeze"]["candidate_hash"],
            summary["selection_freeze"]["candidate_hash"],
        )
        self.assertTrue(summary["overfitting_diagnostics"]["holdout_evaluated_once"])
        self.assertTrue(summary["overfitting_diagnostics"]["holdout_evaluated_after_freeze"])

    def test_selection_policy_rejects_excess_complexity(self):
        raw = _build_market_frame(100)
        storage_path = _make_storage_path()

        _ScenarioAutoMLPipeline.reset()
        _ScenarioAutoMLPipeline.full_rows = len(raw)
        _ScenarioAutoMLPipeline.metrics_by_variant = {
            "rf": {
                "search": {"sharpe_ratio": 2.2, "returns": [0.0018, 0.0014, 0.0019, 0.0015]},
                "validation": {"sharpe_ratio": 2.0, "returns": [0.0017, 0.0013, 0.0018, 0.0014]},
                "holdout": {"sharpe_ratio": 1.5, "returns": [0.0012, 0.0009, 0.0013, 0.0010]},
            },
            "logistic": {
                "search": {"sharpe_ratio": 1.5, "returns": [0.0012, 0.0008, 0.0013, 0.0009]},
                "validation": {"sharpe_ratio": 1.4, "returns": [0.0011, 0.0007, 0.0012, 0.0008]},
                "holdout": {"sharpe_ratio": 1.1, "returns": [0.0009, 0.0006, 0.0010, 0.0007]},
            },
        }

        base_pipeline = _BasePipelineStub(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "automl": {
                    "enabled": True,
                    "n_trials": 2,
                    "objective": "sharpe_ratio",
                    "seed": 23,
                    "validation_fraction": 0.2,
                    "locked_holdout_fraction": 0.2,
                    "locked_holdout_min_search_rows": 40,
                    "enable_pruning": False,
                    "storage": storage_path,
                    "study_name": "automl_complexity_gate_test",
                    "selection_policy": {
                        "min_validation_trade_count": 1,
                        "max_complexity_score": 2.0,
                        "require_locked_holdout_pass": False,
                    },
                },
                "model": {"type": "rf"},
            },
            raw_data=raw,
            data=raw.copy(),
        )

        overrides = [
            {
                "model": {"type": "rf", "params": {"n_estimators": 400, "max_depth": 8, "min_samples_leaf": 1}},
                "features": {"lags": "1,3,6"},
                "labels": {"max_holding": 48},
                "regime": {"n_regimes": 4},
            },
            {"model": {"type": "logistic", "params": {"c": 1.0}}},
        ]
        with mock.patch("core.automl._sample_trial_overrides", side_effect=lambda trial, _: overrides[trial.number]):
            summary = run_automl_study(
                base_pipeline,
                pipeline_class=_ScenarioAutoMLPipeline,
                trial_step_classes=[],
            )

        self.assertEqual(summary["best_overrides"]["model"]["type"], "logistic")
        rejected_trial = next(trial for trial in summary["top_trials"] if trial["model_family"] == "rf")
        self.assertFalse(rejected_trial["selection_policy"]["eligible"])
        self.assertIn("complexity_score_above_limit", rejected_trial["selection_policy"]["eligibility_reasons"])
        self.assertIn("trial_complexity_score", summary["best_selection_policy"])

    def test_selection_policy_rejects_fragile_top_candidate(self):
        raw = _build_market_frame(100)
        storage_path = _make_storage_path()

        _ScenarioAutoMLPipeline.reset()
        _ScenarioAutoMLPipeline.full_rows = len(raw)
        _ScenarioAutoMLPipeline.metrics_by_variant = {
            "rf": {
                "search": {"sharpe_ratio": 2.0, "returns": [0.0018, 0.0014, 0.0019, 0.0015]},
                "validation": {"sharpe_ratio": 1.9, "returns": [0.0017, 0.0013, 0.0018, 0.0014]},
                "holdout": {"sharpe_ratio": 1.6, "returns": [0.0014, 0.0010, 0.0015, 0.0011]},
            },
            "gbm": {
                "search": {"sharpe_ratio": 1.7, "returns": [0.0015, 0.0011, 0.0016, 0.0012]},
                "validation": {"sharpe_ratio": 1.6, "returns": [0.0014, 0.0010, 0.0015, 0.0011]},
                "holdout": {"sharpe_ratio": 1.4, "returns": [0.0012, 0.0009, 0.0013, 0.0010]},
            },
        }

        base_pipeline = _BasePipelineStub(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "automl": {
                    "enabled": True,
                    "n_trials": 2,
                    "objective": "sharpe_ratio",
                    "seed": 29,
                    "validation_fraction": 0.2,
                    "locked_holdout_fraction": 0.2,
                    "locked_holdout_min_search_rows": 40,
                    "enable_pruning": False,
                    "storage": storage_path,
                    "study_name": "automl_fragility_gate_test",
                    "selection_policy": {
                        "min_validation_trade_count": 1,
                        "max_param_fragility": 0.10,
                        "require_locked_holdout_pass": False,
                    },
                },
                "model": {"type": "rf"},
            },
            raw_data=raw,
            data=raw.copy(),
        )

        variants = [
            {"model": {"type": "rf"}},
            {"model": {"type": "gbm"}},
        ]

        def _fragility_side_effect(*args, **kwargs):
            overrides = kwargs["overrides"]
            model_type = overrides["model"]["type"]
            if model_type == "rf":
                return {
                    "enabled": True,
                    "baseline_value": 1.9,
                    "param_fragility_score": 0.55,
                    "dispersion": 0.30,
                    "max_downside": 0.55,
                    "evaluated_count": 4,
                    "perturbations": [],
                    "reason": None,
                    "passed": False,
                }
            return {
                "enabled": True,
                "baseline_value": 1.6,
                "param_fragility_score": 0.04,
                "dispersion": 0.02,
                "max_downside": 0.04,
                "evaluated_count": 4,
                "perturbations": [],
                "reason": None,
                "passed": True,
            }

        with mock.patch("core.automl._sample_trial_overrides", side_effect=lambda trial, _: variants[trial.number]):
            with mock.patch("core.automl._evaluate_candidate_fragility", side_effect=_fragility_side_effect):
                summary = run_automl_study(
                    base_pipeline,
                    pipeline_class=_ScenarioAutoMLPipeline,
                    trial_step_classes=[],
                )

        self.assertEqual(summary["best_overrides"]["model"]["type"], "gbm")
        rejected_trial = next(trial for trial in summary["top_trials"] if trial["model_family"] == "rf")
        self.assertFalse(rejected_trial["selection_policy"]["eligible"])
        self.assertIn("parameter_fragility_above_limit", rejected_trial["selection_policy"]["eligibility_reasons"])
        self.assertAlmostEqual(float(summary["best_selection_policy"]["param_fragility_score"]), 0.04, places=6)

    def test_selection_policy_rejects_unstable_top_candidate(self):
        raw = _build_market_frame(100)
        storage_path = _make_storage_path()

        _ScenarioAutoMLPipeline.reset()
        _ScenarioAutoMLPipeline.full_rows = len(raw)
        _ScenarioAutoMLPipeline.metrics_by_variant = {
            "rf": {
                "search": {"sharpe_ratio": 2.1, "returns": [0.0019, 0.0015, 0.0020, 0.0016]},
                "validation": {
                    "sharpe_ratio": 1.9,
                    "returns": [0.0017, 0.0013, 0.0018, 0.0014],
                    "fold_stability": {
                        "enabled": True,
                        "policy_enabled": True,
                        "primary_metric": "directional_accuracy",
                        "passed": False,
                        "reasons": ["directional_accuracy_cv_above_limit"],
                        "metrics": {
                            "directional_accuracy": {
                                "count": 3,
                                "mean": 0.62,
                                "std": 0.19,
                                "median": 0.66,
                                "min": 0.39,
                                "max": 0.81,
                                "cv": 0.3064516129,
                            }
                        },
                        "worst_fold_sharpe": 0.12,
                        "worst_fold_net_profit_pct": -0.018,
                        "max_drawdown_dispersion": 0.11,
                    },
                },
                "holdout": {"sharpe_ratio": 1.3, "returns": [0.0011, 0.0008, 0.0012, 0.0009]},
            },
            "gbm": {
                "search": {"sharpe_ratio": 1.7, "returns": [0.0015, 0.0011, 0.0016, 0.0012]},
                "validation": {
                    "sharpe_ratio": 1.6,
                    "returns": [0.0014, 0.0010, 0.0015, 0.0011],
                    "fold_stability": {
                        "enabled": True,
                        "policy_enabled": True,
                        "primary_metric": "directional_accuracy",
                        "passed": True,
                        "reasons": [],
                        "metrics": {
                            "directional_accuracy": {
                                "count": 3,
                                "mean": 0.58,
                                "std": 0.03,
                                "median": 0.58,
                                "min": 0.55,
                                "max": 0.61,
                                "cv": 0.0517241379,
                            }
                        },
                        "worst_fold_sharpe": 0.74,
                        "worst_fold_net_profit_pct": 0.012,
                        "max_drawdown_dispersion": 0.02,
                    },
                },
                "holdout": {"sharpe_ratio": 1.4, "returns": [0.0012, 0.0009, 0.0013, 0.0010]},
            },
        }

        base_pipeline = _BasePipelineStub(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "validation": {
                    "stability_policy": {
                        "enabled": True,
                        "cv_metric": "directional_accuracy",
                        "max_cv": 0.15,
                    }
                },
                "automl": {
                    "enabled": True,
                    "n_trials": 2,
                    "objective": "sharpe_ratio",
                    "seed": 37,
                    "validation_fraction": 0.2,
                    "locked_holdout_fraction": 0.2,
                    "locked_holdout_min_search_rows": 40,
                    "enable_pruning": False,
                    "storage": storage_path,
                    "study_name": "automl_stability_gate_test",
                    "selection_policy": {
                        "min_validation_trade_count": 1,
                        "require_locked_holdout_pass": False,
                        "require_fold_stability_pass": True,
                    },
                },
                "model": {"type": "rf"},
            },
            raw_data=raw,
            data=raw.copy(),
        )

        variants = [
            {"model": {"type": "rf"}},
            {"model": {"type": "gbm"}},
        ]
        with mock.patch("core.automl._sample_trial_overrides", side_effect=lambda trial, _: variants[trial.number]):
            summary = run_automl_study(
                base_pipeline,
                pipeline_class=_ScenarioAutoMLPipeline,
                trial_step_classes=[],
            )

        self.assertEqual(summary["best_overrides"]["model"]["type"], "gbm")
        rejected_trial = next(trial for trial in summary["top_trials"] if trial["model_family"] == "rf")
        self.assertFalse(rejected_trial["selection_policy"]["eligible"])
        self.assertIn("fold_stability_failed", rejected_trial["selection_policy"]["eligibility_reasons"])
        self.assertFalse(rejected_trial["fold_stability"]["passed"])
        self.assertTrue(summary["best_selection_policy"]["fold_stability"]["passed"])

    def test_selection_policy_summary_exposes_gap_complexity_and_fragility(self):
        raw = _build_market_frame(120)
        storage_path = _make_storage_path()

        _ScenarioAutoMLPipeline.reset()
        _ScenarioAutoMLPipeline.full_rows = len(raw)
        _ScenarioAutoMLPipeline.metrics_by_variant = {
            "gbm": {
                "search": {"sharpe_ratio": 1.4, "returns": [0.0014, 0.0010, 0.0015, 0.0011]},
                "validation": {"sharpe_ratio": 1.2, "returns": [0.0012, 0.0008, 0.0013, 0.0009]},
                "holdout": {"sharpe_ratio": 1.0, "returns": [0.0010, 0.0007, 0.0011, 0.0008]},
            },
        }

        base_pipeline = _BasePipelineStub(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "automl": {
                    "enabled": True,
                    "n_trials": 1,
                    "objective": "sharpe_ratio",
                    "seed": 31,
                    "validation_fraction": 0.2,
                    "locked_holdout_fraction": 0.2,
                    "locked_holdout_min_search_rows": 48,
                    "enable_pruning": False,
                    "storage": storage_path,
                    "study_name": "automl_selection_policy_summary_test",
                    "search_space": {
                        "feature_selection": {"max_features": {"type": "categorical", "choices": [32, 48, 64]}}
                    },
                    "selection_policy": {
                        "min_validation_trade_count": 1,
                        "require_locked_holdout_pass": False,
                    },
                },
                "model": {"type": "gbm"},
            },
            raw_data=raw,
            data=raw.copy(),
        )

        with mock.patch(
            "core.automl._sample_trial_overrides",
            side_effect=lambda trial, _: {
                "feature_selection": {"enabled": True, "max_features": 48},
                "model": {"type": "gbm"},
            },
        ):
            summary = run_automl_study(
                base_pipeline,
                pipeline_class=_ScenarioAutoMLPipeline,
                trial_step_classes=[],
            )

        self.assertIn("trial_complexity_score", summary["best_selection_policy"])
        self.assertIn("fold_stability", summary["best_selection_policy"])
        self.assertIn("generalization_gap", summary["best_selection_policy"])
        self.assertIn("param_fragility_score", summary["best_selection_policy"])
        self.assertIsNotNone(summary["best_selection_policy"]["generalization_gap"]["search_to_validation"])
        self.assertIsNotNone(summary["top_trials"][0]["trial_complexity_score"])
        self.assertIn("fold_stability", summary["top_trials"][0])
        self.assertIsNotNone(summary["top_trials"][0]["param_fragility_score"])

    def test_selection_policy_binds_feature_portability_and_regime_stability(self):
        raw = _build_market_frame(120)
        storage_path = _make_storage_path()

        class _GovernanceGatePipeline(_ScenarioAutoMLPipeline):
            def run_step(self, name):
                result = super().run_step(name)
                if name == "train_models":
                    model_type = str(self.config.get("model", {}).get("type", "gbm"))
                    training = self.state["training"]
                    bad_variant = model_type == "gbm"
                    training["feature_governance"] = {
                        "admission_summary": {"promotion_pass": True, "reasons": []}
                    }
                    training["feature_portability_diagnostics"] = {
                        "promotion_pass": not bad_variant,
                        "reasons": (["venue_specific_importance_dominates"] if bad_variant else []),
                    }
                    training["operational_monitoring"] = {"healthy": True, "reasons": []}
                    training["regime"] = {
                        "mode": "fold_local",
                        "folds": [],
                        "ablation_summary": {
                            "promotion_pass": not bad_variant,
                            "reasons": (["contextual_regime_less_stable"] if bad_variant else []),
                        },
                    }
                    training["promotion_gates"] = {
                        "feature_portability": not bad_variant,
                        "feature_admission": True,
                        "regime_stability": not bad_variant,
                        "operational_health": True,
                    }
                    self.state["training"] = training
                    self.step_results[name] = training
                    return training
                return result

        _GovernanceGatePipeline.reset()
        _GovernanceGatePipeline.full_rows = len(raw)
        _GovernanceGatePipeline.metrics_by_variant = {
            "gbm": {
                "search": {"sharpe_ratio": 1.4, "returns": [0.0014, 0.0011, 0.0015, 0.0012]},
                "validation": {"sharpe_ratio": 1.3, "returns": [0.0013, 0.0010, 0.0014, 0.0011]},
                "holdout": {"sharpe_ratio": 1.2, "returns": [0.0012, 0.0009, 0.0013, 0.0010]},
            },
            "rf": {
                "search": {"sharpe_ratio": 1.1, "returns": [0.0011, 0.0008, 0.0012, 0.0009]},
                "validation": {"sharpe_ratio": 1.0, "returns": [0.0010, 0.0007, 0.0011, 0.0008]},
                "holdout": {"sharpe_ratio": 0.9, "returns": [0.0009, 0.0006, 0.0010, 0.0007]},
            },
        }

        base_pipeline = _BasePipelineStub(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "automl": {
                    "enabled": True,
                    "n_trials": 2,
                    "objective": "sharpe_ratio",
                    "seed": 41,
                    "validation_fraction": 0.2,
                    "locked_holdout_fraction": 0.2,
                    "locked_holdout_min_search_rows": 48,
                    "enable_pruning": False,
                    "storage": storage_path,
                    "study_name": "automl_governance_gate_binding_test",
                    "selection_policy": {
                        "min_validation_trade_count": 1,
                        "require_locked_holdout_pass": False,
                    },
                },
                "model": {"type": "gbm"},
            },
            raw_data=raw,
            data=raw.copy(),
        )

        variants = [
            {"model": {"type": "gbm"}},
            {"model": {"type": "rf"}},
        ]
        with mock.patch("core.automl._sample_trial_overrides", side_effect=lambda trial, _: variants[trial.number]):
            summary = run_automl_study(
                base_pipeline,
                pipeline_class=_GovernanceGatePipeline,
                trial_step_classes=[],
            )

        self.assertEqual(summary["best_overrides"]["model"]["type"], "rf")
        rejected_trial = next(trial for trial in summary["top_trials"] if trial["model_family"] == "gbm")
        self.assertFalse(rejected_trial["selection_policy"]["eligible"])
        self.assertIn("feature_portability_failed", rejected_trial["selection_policy"]["eligibility_reasons"])
        self.assertIn("regime_stability_failed", rejected_trial["selection_policy"]["eligibility_reasons"])
        gate_report = rejected_trial["selection_policy"]["promotion_eligibility_report"]
        self.assertIn("feature_portability", gate_report["gate_status"])
        self.assertIn("regime_stability", gate_report["gate_status"])
        self.assertEqual(gate_report["gate_status"]["feature_portability"]["reason"], "feature_portability_failed")
        self.assertEqual(gate_report["gate_status"]["regime_stability"]["reason"], "regime_stability_failed")
        self.assertFalse(gate_report["eligible_before_post_checks"])

    def test_selection_policy_binds_cross_venue_integrity_gate(self):
        raw = _build_market_frame(120)
        storage_path = _make_storage_path()

        class _CrossVenueGatePipeline(_ScenarioAutoMLPipeline):
            def run_step(self, name):
                result = super().run_step(name)
                if name == "train_models":
                    model_type = str(self.config.get("model", {}).get("type", "gbm"))
                    training = self.state["training"]
                    bad_variant = model_type == "gbm"
                    training["feature_governance"] = {
                        "admission_summary": {"promotion_pass": True, "reasons": []}
                    }
                    training["feature_portability_diagnostics"] = {"promotion_pass": True, "reasons": []}
                    training["operational_monitoring"] = {"healthy": True, "reasons": []}
                    training["regime"] = {
                        "mode": "fold_local",
                        "folds": [],
                        "ablation_summary": {"promotion_pass": True, "reasons": []},
                    }
                    training["cross_venue_integrity"] = {
                        "promotion_pass": not bad_variant,
                        "gate_mode": "blocking",
                        "reasons": (["spot_reference_divergence"] if bad_variant else []),
                    }
                    training["promotion_gates"] = {
                        "feature_portability": True,
                        "feature_admission": True,
                        "regime_stability": True,
                        "operational_health": True,
                        "cross_venue_integrity": not bad_variant,
                    }
                    self.state["training"] = training
                    self.step_results[name] = training
                    return training
                return result

        _CrossVenueGatePipeline.reset()
        _CrossVenueGatePipeline.full_rows = len(raw)
        _CrossVenueGatePipeline.metrics_by_variant = {
            "gbm": {
                "search": {"sharpe_ratio": 1.3, "returns": [0.0014, 0.0011, 0.0015, 0.0012]},
                "validation": {"sharpe_ratio": 1.2, "returns": [0.0013, 0.0010, 0.0014, 0.0011]},
                "holdout": {"sharpe_ratio": 1.1, "returns": [0.0012, 0.0009, 0.0013, 0.0010]},
            },
            "rf": {
                "search": {"sharpe_ratio": 1.0, "returns": [0.0010, 0.0008, 0.0011, 0.0009]},
                "validation": {"sharpe_ratio": 0.9, "returns": [0.0009, 0.0007, 0.0010, 0.0008]},
                "holdout": {"sharpe_ratio": 0.8, "returns": [0.0008, 0.0006, 0.0009, 0.0007]},
            },
        }

        base_pipeline = _BasePipelineStub(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "automl": {
                    "enabled": True,
                    "n_trials": 2,
                    "objective": "sharpe_ratio",
                    "seed": 17,
                    "validation_fraction": 0.2,
                    "locked_holdout_fraction": 0.2,
                    "locked_holdout_min_search_rows": 48,
                    "enable_pruning": False,
                    "storage": storage_path,
                    "study_name": "automl_cross_venue_gate_binding_test",
                    "selection_policy": {
                        "min_validation_trade_count": 1,
                        "require_locked_holdout_pass": False,
                    },
                },
                "model": {"type": "gbm"},
            },
            raw_data=raw,
            data=raw.copy(),
        )

        variants = [
            {"model": {"type": "gbm"}},
            {"model": {"type": "rf"}},
        ]
        with mock.patch("core.automl._sample_trial_overrides", side_effect=lambda trial, _: variants[trial.number]):
            summary = run_automl_study(
                base_pipeline,
                pipeline_class=_CrossVenueGatePipeline,
                trial_step_classes=[],
            )

        self.assertEqual(summary["best_overrides"]["model"]["type"], "rf")
        rejected_trial = next(trial for trial in summary["top_trials"] if trial["model_family"] == "gbm")
        self.assertFalse(rejected_trial["selection_policy"]["eligible"])
        self.assertIn("spot_reference_divergence", rejected_trial["selection_policy"]["eligibility_reasons"])
        gate_report = rejected_trial["selection_policy"]["promotion_eligibility_report"]
        self.assertIn("cross_venue_integrity", gate_report["gate_status"])
        self.assertEqual(gate_report["gate_status"]["cross_venue_integrity"]["reason"], "spot_reference_divergence")


if __name__ == "__main__":
    unittest.main()