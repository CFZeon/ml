import unittest

import numpy as np
import pandas as pd

from core import ResearchPipeline, run_lookahead_analysis


class LookaheadProvocationTest(unittest.TestCase):
    @staticmethod
    def _make_raw(n=260, seed=0, start="2026-06-01"):
        rng = np.random.default_rng(seed)
        index = pd.date_range(start, periods=n, freq="1h", tz="UTC")
        close = 100.0 + rng.normal(0.0, 1.0, n).cumsum()
        open_ = np.r_[close[0], close[:-1]]
        return pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) * 1.001,
                "low": np.minimum(open_, close) * 0.999,
                "close": close,
                "volume": 1_000.0 + rng.normal(0.0, 10.0, n),
                "quote_volume": close * 1_000.0,
                "trades": 100,
            },
            index=index,
        )

    def _make_pipeline(self, raw, builders=None):
        pipeline = ResearchPipeline(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "indicators": [],
                "features": {
                    "lags": [1, 3],
                    "frac_diff_d": None,
                    "rolling_window": 20,
                    "builders": list(builders or []),
                },
                "regime": {"method": "hmm", "n_regimes": 2},
                "labels": {"kind": "fixed_horizon", "horizon": 4, "threshold": 0.0001},
                "model": {"type": "gbm", "cv_method": "walk_forward", "n_splits": 1, "gap": 0},
                "feature_selection": {"enabled": False},
                "signals": {"avg_win": 0.02, "avg_loss": 0.02},
                "backtest": {"use_open_execution": False, "signal_delay_bars": 1},
            }
        )
        pipeline.state["raw_data"] = raw.copy()
        pipeline.state["data"] = raw.copy()
        return pipeline

    def test_detects_future_shifted_feature_via_prefix_replay(self):
        raw = self._make_raw(seed=12)

        def biased_builder(pipeline, features):
            built = features.copy()
            built["future_close"] = pipeline.require("data")["close"].shift(-1)
            return built

        pipeline = self._make_pipeline(raw, builders=[biased_builder])
        timestamps = raw.index[-24:-2]

        report = run_lookahead_analysis(
            pipeline,
            step_names=["build_features"],
            artifact_names=["features"],
            decision_timestamps=timestamps,
            min_prefix_rows=80,
        )

        self.assertTrue(report["has_bias"])
        self.assertIn("features", report["artifacts"])
        self.assertIn("future_close", report["artifacts"]["features"]["biased_columns"])
        self.assertEqual(report["artifacts"]["features"]["stage"], "build_features")
        self.assertGreater(report["artifacts"]["features"]["mismatch_count"], 0)

    def test_causal_feature_set_passes_prefix_replay(self):
        raw = self._make_raw(seed=21)
        pipeline = self._make_pipeline(raw)
        timestamps = raw.index[-24:-2]

        report = run_lookahead_analysis(
            pipeline,
            step_names=["build_features"],
            artifact_names=["features"],
            decision_timestamps=timestamps,
            min_prefix_rows=80,
        )

        self.assertFalse(report["has_bias"])
        self.assertEqual(report["artifacts"]["features"]["biased_columns"], [])
        self.assertEqual(report["artifacts"]["features"]["mismatch_count"], 0)

    def test_default_audit_surface_reports_probabilities_signals_and_execution_inputs(self):
        raw = self._make_raw(seed=7)
        pipeline = self._make_pipeline(raw)
        timestamps = raw.index[-4:-2]

        report = run_lookahead_analysis(
            pipeline,
            decision_timestamps=timestamps,
            min_prefix_rows=120,
        )

        self.assertIn("features", report["artifacts"])
        self.assertIn("regimes", report["artifacts"])
        self.assertIn("labels", report["artifacts"])
        self.assertIn("aligned_labels", report["artifacts"])
        self.assertIn("oos_probabilities", report["artifacts"])
        self.assertIn("signals", report["artifacts"])
        self.assertIn("continuous_signals", report["artifacts"])
        self.assertIn("execution_prices", report["artifacts"])
        self.assertIn("execution_volume", report["artifacts"])

    def test_blocking_lookahead_guard_rejects_future_shifted_builder_automatically(self):
        raw = self._make_raw(seed=31)

        def biased_builder(pipeline, features):
            built = features.copy()
            built["future_close"] = pipeline.require("data")["close"].shift(-1)
            return built

        pipeline = self._make_pipeline(raw, builders=[biased_builder])
        pipeline.config["features"]["lookahead_guard"] = {
            "mode": "blocking",
            "decision_sample_size": 12,
            "min_prefix_rows": 80,
        }

        pipeline.build_features()
        pipeline.detect_regimes()
        pipeline.build_labels()
        pipeline.align_data()

        with self.assertRaisesRegex(RuntimeError, "Lookahead guard failed"):
            pipeline.train_models()

        report = pipeline.state["lookahead_guard_report"]
        self.assertTrue(report["has_bias"])
        self.assertFalse(report["promotion_pass"])
        self.assertIn("future_close", report["biased_columns"])

    def test_advisory_lookahead_guard_records_failure_but_allows_training(self):
        raw = self._make_raw(seed=41)

        def biased_builder(pipeline, features):
            built = features.copy()
            built["future_close"] = pipeline.require("data")["close"].shift(-1)
            return built

        pipeline = self._make_pipeline(raw, builders=[biased_builder])
        pipeline.config["features"]["lookahead_guard"] = {
            "mode": "advisory",
            "decision_sample_size": 12,
            "min_prefix_rows": 80,
        }

        pipeline.build_features()
        pipeline.detect_regimes()
        pipeline.build_labels()
        pipeline.align_data()
        training = pipeline.train_models()

        self.assertIn("lookahead_guard", training)
        self.assertTrue(training["lookahead_guard"]["has_bias"])
        self.assertFalse(training["lookahead_guard"]["promotion_pass"])
        self.assertFalse(training["promotion_gates"]["lookahead_guard"])


if __name__ == "__main__":
    unittest.main()