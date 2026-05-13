import io
import unittest
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from core import ResearchPipeline, cpcv_split
from core.pipeline import _summarize_path_backtests
from example_utils import print_training_summary


def _make_raw(n=320, seed=0, start="2026-01-01"):
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    trend = np.linspace(0.0, 12.0, n)
    cycle = 2.0 * np.sin(np.linspace(0.0, 10.0 * np.pi, n))
    noise = rng.normal(0.0, 0.35, n).cumsum() / 4.0
    close = 100.0 + trend + cycle + noise
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    volume = 1_000.0 + 100.0 * (1.0 + np.sin(np.linspace(0.0, 4.0 * np.pi, n)))
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": close * volume,
            "trades": 100,
        },
        index=index,
    )


def _make_pipeline(raw, model_config):
    pipeline = ResearchPipeline(
        {
            "data": {"symbol": "BTCUSDT", "interval": "1h"},
            "indicators": [],
            "features": {"lags": [1, 3], "frac_diff_d": 0.4, "rolling_window": 20},
            "labels": {"kind": "fixed_horizon", "horizon": 6, "threshold": 0.0001},
            "model": model_config,
            "feature_selection": {"enabled": True, "max_features": 12},
            "signals": {"avg_win": 0.02, "avg_loss": 0.02, "threshold": 0.0, "edge_threshold": 0.0, "meta_threshold": 0.5},
            "backtest": {
                "use_open_execution": False,
                "signal_delay_bars": 1,
                "fee_rate": 0.0,
                "slippage_rate": 0.0,
                "engine": "vectorbt",
            },
        }
    )
    pipeline.state["raw_data"] = raw
    pipeline.state["data"] = raw.copy()
    pipeline.build_features()
    pipeline.build_labels()
    pipeline.align_data()
    return pipeline


class CPCVValidationTest(unittest.TestCase):
    def test_path_summary_aggregates_router_stability_diagnostics(self):
        summary = _summarize_path_backtests(
            [
                {
                    "backtest": {
                        "router_stability_report": {
                            "enabled": True,
                            "applicable": True,
                            "decision_count": 100,
                            "switch_count": 12,
                            "switch_rate": 0.12,
                            "allocation_change_count": 16,
                            "allocation_change_rate": 0.16,
                            "executed_weight_turnover_total": 14.0,
                            "executed_weight_turnover_rate": 0.14,
                            "blocked_switch_count": 20,
                            "blocked_switch_rate": 0.2,
                            "blocked_allocation_count": 0,
                            "blocked_allocation_rate": 0.0,
                            "mean_effective_model_count": 1.05,
                            "blocked_switch_reasons": {"cooldown_active": 2},
                            "allocation_control_reason_counts": {"selection_only": 100},
                            "configured_control_count": 3,
                            "switching_cost_estimate": 18.0,
                            "switching_cost_share_of_starting_equity": 0.0018,
                        }
                    }
                },
                {
                    "backtest": {
                        "router_stability_report": {
                            "enabled": True,
                            "applicable": True,
                            "decision_count": 80,
                            "switch_count": 8,
                            "switch_rate": 0.1,
                            "allocation_change_count": 10,
                            "allocation_change_rate": 0.125,
                            "executed_weight_turnover_total": 9.0,
                            "executed_weight_turnover_rate": 0.1125,
                            "blocked_switch_count": 12,
                            "blocked_switch_rate": 0.15,
                            "blocked_allocation_count": 0,
                            "blocked_allocation_rate": 0.0,
                            "mean_effective_model_count": 1.0,
                            "blocked_switch_reasons": {"persistence_requirement_not_met": 3},
                            "allocation_control_reason_counts": {"selection_only": 80},
                            "configured_control_count": 2,
                            "switching_cost_estimate": 10.0,
                            "switching_cost_share_of_starting_equity": 0.001,
                        }
                    }
                },
            ]
        )

        report = summary["router_stability_report"]
        self.assertTrue(report["enabled"])
        self.assertEqual(report["path_count"], 2)
        self.assertEqual(report["applicable_path_count"], 2)
        self.assertAlmostEqual(float(report["mean_switch_rate"]), 0.11, places=6)
        self.assertAlmostEqual(float(report["mean_executed_weight_turnover_rate"]), 0.12625, places=6)
        self.assertEqual(report["blocked_switch_reasons"], {"cooldown_active": 2, "persistence_requirement_not_met": 3})

    def test_cpcv_split_generates_block_combinations_and_embargo(self):
        frame = pd.DataFrame({"feature": np.arange(12, dtype=float)})

        splits = list(cpcv_split(frame, n_blocks=4, test_blocks=2, embargo=1))

        self.assertEqual(len(splits), 6)
        train_idx, test_idx, metadata = splits[0]
        self.assertTupleEqual(metadata["test_blocks"], (0, 1))
        self.assertListEqual(test_idx.tolist(), [0, 1, 2, 3, 4, 5])
        self.assertListEqual(train_idx.tolist(), [7, 8, 9, 10, 11])
        self.assertEqual(int(metadata["embargo_rows"]), 1)

    def test_train_models_defaults_to_cpcv_and_emits_path_outputs(self):
        raw = _make_raw(n=320, seed=7)
        pipeline = _make_pipeline(
            raw,
            {
                "type": "gbm",
                "n_blocks": 4,
                "test_blocks": 2,
                "validation_fraction": 0.2,
                "meta_n_splits": 2,
            },
        )

        training = pipeline.train_models()
        signals = pipeline.generate_signals()
        backtest = pipeline.run_backtest()

        self.assertEqual(training["validation"]["method"], "cpcv")
        self.assertEqual(int(training["validation"]["feature_lag_bars"]), 3)
        self.assertEqual(int(training["validation"]["embargo_bars"]), 6)
        self.assertEqual(int(training["validation"]["effective_embargo_bars"]), 9)
        self.assertGreaterEqual(int(training["validation"]["split_count"]), 1)
        self.assertIsNone(training["oos_predictions"])
        self.assertEqual(len(training["oos_paths"]), int(training["validation"]["split_count"]))
        self.assertTrue(training["executable_validation"]["enabled"])
        self.assertEqual(training["executable_validation"]["training"]["validation"]["method"], "walk_forward")
        self.assertEqual(training["signal_policy"]["calibration_policy"]["policy_name"], "validation_only_or_defaults")
        self.assertFalse(training["signal_policy"]["last_policy_quality"]["cross_fold_borrowing_allowed"])
        self.assertEqual(signals["signal_source"], "cpcv_walk_forward_replay")
        self.assertEqual(signals["validation_method"], "cpcv")
        self.assertEqual(signals["primary_validation_method"], "walk_forward")
        self.assertEqual(int(signals["diagnostic_validation"]["path_count"]), len(training["oos_paths"]))
        self.assertEqual(backtest["validation_method"], "walk_forward")
        self.assertEqual(backtest["diagnostic_validation_method"], "cpcv")
        self.assertEqual(int(backtest["diagnostic_validation"]["path_count"]), len(training["oos_paths"]))
        self.assertEqual(backtest["diagnostic_validation"]["summary"]["aggregate_mode"], "diagnostic_distribution")
        self.assertTrue(backtest["diagnostic_validation"]["summary"]["regime_segment_report"]["enabled"])
        self.assertEqual(
            backtest["diagnostic_validation"]["summary"]["regime_segment_report"]["aggregate_mode"],
            "path_diagnostics_only",
        )
        self.assertEqual(
            int(backtest["diagnostic_validation"]["summary"]["regime_segment_report"]["path_count"]),
            len(training["oos_paths"]),
        )
        self.assertTrue(backtest["diagnostic_validation"]["summary"]["regime_segment_report"]["label_distribution"])
        self.assertTrue(backtest["diagnostic_validation"]["summary"]["transition_segment_report"]["enabled"])
        self.assertEqual(
            backtest["diagnostic_validation"]["summary"]["transition_segment_report"]["aggregate_mode"],
            "path_diagnostics_only",
        )
        first_transition = next(
            iter(backtest["diagnostic_validation"]["summary"]["transition_segment_report"]["by_transition"].values())
        )
        self.assertIn("mean_recognition_delay_bars", first_transition)
        self.assertIn("mean_delay_window_return", first_transition)
        self.assertIn("mean_positive_cumulative_onset_lag_bars", first_transition)
        self.assertIn("statistical_significance", backtest)
        self.assertTrue(backtest["statistical_significance"]["enabled"])
        self.assertFalse(backtest["diagnostic_validation"]["summary"]["statistical_significance"]["enabled"])

    def test_walk_forward_rejects_gap_shorter_than_feature_lag(self):
        raw = _make_raw(n=320, seed=13)
        pipeline = _make_pipeline(
            raw,
            {
                "type": "gbm",
                "cv_method": "walk_forward",
                "n_splits": 2,
                "gap": 2,
                "validation_fraction": 0.2,
                "meta_n_splits": 2,
            },
        )

        with self.assertRaisesRegex(ValueError, "feature_lag_bars=3"):
            pipeline.train_models()

    def test_explicit_walk_forward_preserves_flat_oos_outputs(self):
        raw = _make_raw(n=320, seed=11)
        pipeline = _make_pipeline(
            raw,
            {
                "type": "gbm",
                "cv_method": "walk_forward",
                "n_splits": 1,
                "gap": 3,
                "validation_fraction": 0.2,
                "meta_n_splits": 2,
            },
        )

        training = pipeline.train_models()
        signals = pipeline.generate_signals()

        self.assertEqual(training["validation"]["method"], "walk_forward")
        self.assertIsInstance(training["oos_predictions"], pd.Series)
        self.assertGreater(len(training["oos_predictions"]), 0)
        self.assertEqual(signals["signal_source"], "walk_forward_oos")

    def test_train_models_emits_fold_stability_diagnostics(self):
        raw = _make_raw(n=320, seed=19)
        pipeline = _make_pipeline(
            raw,
            {
                "type": "gbm",
                "n_blocks": 4,
                "test_blocks": 2,
                "validation_fraction": 0.2,
                "meta_n_splits": 2,
            },
        )

        training = pipeline.train_models()

        self.assertGreaterEqual(len(training["fold_backtests"]), 1)
        stability = training["fold_stability"]
        self.assertTrue(stability["enabled"])
        self.assertFalse(stability["policy_enabled"])
        self.assertIn("directional_accuracy", stability["metrics"])
        self.assertIn("sharpe_ratio", stability["metrics"])

        directional_values = np.asarray(
            [row["directional_accuracy"] for row in training["fold_metrics"] if row.get("directional_accuracy") is not None],
            dtype=float,
        )
        expected_std = float(np.std(directional_values, ddof=1)) if len(directional_values) > 1 else 0.0
        expected_cv = expected_std / abs(float(np.mean(directional_values))) if abs(float(np.mean(directional_values))) > 1e-12 else None
        self.assertAlmostEqual(float(stability["metrics"]["directional_accuracy"]["std"]), expected_std, places=6)
        if expected_cv is not None:
            self.assertAlmostEqual(float(stability["metrics"]["directional_accuracy"]["cv"]), expected_cv, places=6)

        worst_sharpe = min(float(row["sharpe_ratio"]) for row in training["fold_backtests"] if row.get("sharpe_ratio") is not None)
        self.assertAlmostEqual(float(stability["worst_fold_sharpe"]), worst_sharpe, places=6)

    def test_print_training_summary_displays_stability_diagnostics(self):
        raw = _make_raw(n=320, seed=23)
        pipeline = _make_pipeline(
            raw,
            {
                "type": "gbm",
                "n_blocks": 4,
                "test_blocks": 2,
                "validation_fraction": 0.2,
                "meta_n_splits": 2,
            },
        )

        training = pipeline.train_models()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            print_training_summary(training)

        output = buffer.getvalue()
        self.assertIn("stability", output)
        self.assertIn("worst sharpe", output)


if __name__ == "__main__":
    unittest.main()