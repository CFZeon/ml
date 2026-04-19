import io
import unittest
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from core import ResearchPipeline, cpcv_split
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
        self.assertEqual(int(training["validation"]["embargo_bars"]), 6)
        self.assertGreaterEqual(int(training["validation"]["split_count"]), 1)
        self.assertIsNone(training["oos_predictions"])
        self.assertEqual(len(training["oos_paths"]), int(training["validation"]["split_count"]))
        self.assertEqual(signals["signal_source"], "cpcv_oos_paths")
        self.assertEqual(int(signals["path_count"]), len(training["oos_paths"]))
        self.assertEqual(backtest["validation_method"], "cpcv")
        self.assertEqual(int(backtest["path_count"]), len(training["oos_paths"]))
        self.assertEqual(backtest["aggregate_mode"], "mean")
        self.assertIn("statistical_significance", backtest)
        self.assertTrue(backtest["statistical_significance"]["enabled"])
        self.assertEqual(backtest["statistical_significance"]["aggregate_mode"], "mean")

    def test_explicit_walk_forward_preserves_flat_oos_outputs(self):
        raw = _make_raw(n=320, seed=11)
        pipeline = _make_pipeline(
            raw,
            {
                "type": "gbm",
                "cv_method": "walk_forward",
                "n_splits": 1,
                "gap": 0,
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