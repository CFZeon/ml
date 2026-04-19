import copy
import unittest

import numpy as np
import pandas as pd

from core.automl import _execute_temporal_split_candidate, _resolve_holdout_plan


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


class _TemporalGapPipeline:
    def __init__(self, config, steps=None):
        self.config = copy.deepcopy(config)
        self.state = {}
        self.step_results = {}

    def run_step(self, name):
        if name == "build_features":
            features = self.state["data"][["close"]].rename(columns={"close": "feature"}).astype(float)
            self.state["features"] = features
            self.step_results[name] = features
            return features

        if name == "detect_regimes":
            self.step_results[name] = None
            return None

        if name == "build_labels":
            index = self.state["data"].index
            labels = pd.DataFrame({"label": 1, "t1": index}, index=index)
            self.state["labels"] = labels
            self.step_results[name] = labels
            return labels

        if name == "align_data":
            X = self.state["features"].copy()
            y = pd.Series(1, index=X.index)
            labels = self.state["labels"].copy()
            self.state["X"] = X
            self.state["y"] = y
            self.state["labels_aligned"] = labels
            aligned = {"X": X, "y": y, "labels_aligned": labels}
            self.step_results[name] = aligned
            return aligned

        if name == "compute_sample_weights":
            weights = pd.Series(1.0, index=self.state["X"].index)
            self.state["sample_weights"] = weights
            self.step_results[name] = weights
            return weights

        if name == "train_models":
            train_size = int(self.config.get("model", {}).get("train_size", 0))
            test_size = int(self.config.get("model", {}).get("test_size", 0))
            index = self.state["X"].index
            train_index = index[:train_size]
            test_index = index[train_size:train_size + test_size]
            training = {
                "train_index": train_index,
                "test_index": test_index,
                "fold_metrics": [{"accuracy": 1.0, "f1_macro": 1.0, "directional_accuracy": 1.0}],
                "avg_accuracy": 1.0,
                "avg_f1_macro": 1.0,
                "avg_directional_accuracy": 1.0,
                "last_model": object(),
                "last_meta": object(),
                "last_selected_columns": ["feature"],
                "oos_predictions": pd.Series(1, index=test_index),
                "oos_probabilities": pd.DataFrame({-1: 0.0, 0: 0.0, 1: 1.0}, index=test_index),
                "oos_meta_prob": pd.Series(1.0, index=test_index),
                "oos_profitability_prob": pd.Series(1.0, index=test_index),
                "oos_direction_edge": pd.Series(1.0, index=test_index),
                "oos_confidence": pd.Series(1.0, index=test_index),
                "oos_expected_trade_edge": pd.Series(0.01, index=test_index),
                "oos_position_size": pd.Series(1.0, index=test_index),
                "oos_kelly_size": pd.Series(1.0, index=test_index),
                "oos_event_signals": pd.Series(1.0, index=test_index),
                "oos_continuous_signals": pd.Series(1.0, index=test_index),
                "oos_signals": pd.Series(1, index=test_index),
                "oos_avg_win": 0.02,
                "oos_avg_loss": 0.01,
            }
            self.state["training"] = training
            self.step_results[name] = training
            return training

        if name == "generate_signals":
            index = self.state["training"]["test_index"]
            signals = {
                "signals": pd.Series(1, index=index),
                "continuous_signals": pd.Series(1.0, index=index),
            }
            self.state["signals"] = signals
            self.step_results[name] = signals
            return signals

        if name == "run_backtest":
            index = self.state["signals"]["continuous_signals"].index
            equity_curve = pd.Series(10_000.0 + np.arange(len(index), dtype=float), index=index)
            backtest = {
                "net_profit_pct": 0.1,
                "sharpe_ratio": 1.0,
                "profit_factor": 1.2,
                "calmar_ratio": 1.0,
                "max_drawdown": -0.01,
                "total_trades": len(index),
                "equity_curve": equity_curve,
                "statistical_significance": {"enabled": False},
            }
            self.state["backtest"] = backtest
            self.step_results[name] = backtest
            return backtest

        raise KeyError(name)


class CrossStageEmbargoTest(unittest.TestCase):
    def test_validation_and_holdout_start_after_stage_gaps(self):
        raw_data = _build_market_frame(120)
        plan = _resolve_holdout_plan(
            raw_data,
            {
                "locked_holdout_enabled": True,
                "validation_fraction": 0.2,
                "locked_holdout_bars": 24,
                "locked_holdout_min_search_rows": 48,
                "search_validation_gap_bars": 5,
                "validation_holdout_gap_bars": 7,
            },
            base_config={"labels": {"kind": "fixed_horizon", "horizon": 1}, "backtest": {"signal_delay_bars": 0}},
        )

        self.assertTrue(plan["enabled"])
        self.assertEqual(int(plan["search_rows"]), 60)
        self.assertEqual(int(plan["search_validation_gap_bars"]), 5)
        self.assertEqual(int(plan["validation_holdout_gap_bars"]), 7)
        self.assertEqual(int(plan["dropped_gap_rows"]), 12)
        self.assertEqual(plan["validation_start_timestamp"], raw_data.index[65])
        self.assertEqual(plan["holdout_start_timestamp"], raw_data.index[96])

    def test_gap_rows_do_not_enter_train_or_test_slices(self):
        raw_data = _build_market_frame(32)
        state_bundle = {
            "raw_data": raw_data.copy(),
            "data": raw_data.copy(),
            "indicator_run": None,
            "futures_context": None,
            "cross_asset_context": None,
            "symbol_filters": {},
            "symbol_lifecycle": None,
            "universe_policy": None,
            "universe_snapshot": None,
            "universe_snapshot_meta": None,
            "eligible_symbols": None,
            "universe_report": None,
        }
        train_end_timestamp = raw_data.index[9]
        test_start_timestamp = raw_data.index[13]
        gap_timestamps = set(raw_data.index[10:13])

        training, _, split = _execute_temporal_split_candidate(
            base_config={},
            overrides={},
            pipeline_class=_TemporalGapPipeline,
            trial_step_classes=[],
            state_bundle=state_bundle,
            train_end_timestamp=train_end_timestamp,
            test_start_timestamp=test_start_timestamp,
        )

        train_index = set(training["train_index"])
        test_index = set(training["test_index"])
        self.assertEqual(int(split["aligned_gap_rows"]), 3)
        self.assertTrue(gap_timestamps.isdisjoint(train_index))
        self.assertTrue(gap_timestamps.isdisjoint(test_index))
        self.assertEqual(max(train_index), train_end_timestamp)
        self.assertEqual(min(test_index), test_start_timestamp)

    def test_default_gap_widens_when_label_horizon_increases(self):
        raw_data = _build_market_frame(160)
        automl_config = {
            "locked_holdout_enabled": True,
            "validation_fraction": 0.2,
            "locked_holdout_fraction": 0.2,
            "locked_holdout_min_search_rows": 20,
        }

        short_horizon_plan = _resolve_holdout_plan(
            raw_data,
            automl_config,
            base_config={
                "labels": {"kind": "fixed_horizon", "horizon": 1},
                "backtest": {"signal_delay_bars": 0, "use_open_execution": False},
            },
        )
        long_horizon_plan = _resolve_holdout_plan(
            raw_data,
            automl_config,
            base_config={
                "labels": {"kind": "fixed_horizon", "horizon": 12},
                "backtest": {"signal_delay_bars": 0, "use_open_execution": False},
            },
        )

        self.assertEqual(int(short_horizon_plan["search_validation_gap_bars"]), 1)
        self.assertEqual(int(long_horizon_plan["search_validation_gap_bars"]), 12)
        self.assertGreater(int(long_horizon_plan["search_validation_gap_bars"]), int(short_horizon_plan["search_validation_gap_bars"]))
        self.assertLess(int(long_horizon_plan["search_rows"]), int(short_horizon_plan["search_rows"]))


if __name__ == "__main__":
    unittest.main()