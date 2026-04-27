import unittest

import numpy as np
import pandas as pd

from core import ResearchPipeline


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


def _make_pipeline(raw):
    pipeline = ResearchPipeline(
        {
            "data": {"symbol": "BTCUSDT", "interval": "1h"},
            "indicators": [],
            "features": {"lags": [1, 3], "frac_diff_d": 0.4, "rolling_window": 20},
            "labels": {"kind": "fixed_horizon", "horizon": 6, "threshold": 0.0001},
            "model": {
                "type": "gbm",
                "n_blocks": 4,
                "test_blocks": 2,
                "validation_fraction": 0.2,
                "meta_n_splits": 2,
            },
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


class MetricSourceReportingTest(unittest.TestCase):
    def test_cpcv_training_reports_diagnostic_and_tradable_sources(self):
        pipeline = _make_pipeline(_make_raw(seed=7))

        training = pipeline.train_models()

        self.assertEqual(training["validation_sources"]["selection_metric_source"], "walk_forward_replay")
        self.assertEqual(training["validation_sources"]["diagnostic_metric_source"], "cpcv")
        self.assertEqual(training["validation_sources"]["tradable_metric_source"], "walk_forward_replay")
        self.assertTrue(training["validation_sources"]["all_required_sources_passed"])

    def test_backtest_reports_same_validation_sources_as_tradable_summary(self):
        pipeline = _make_pipeline(_make_raw(seed=11))

        pipeline.train_models()
        pipeline.generate_signals()
        backtest = pipeline.run_backtest()

        self.assertEqual(backtest["validation_sources"]["selection_metric_source"], "walk_forward_replay")
        self.assertEqual(backtest["validation_sources"]["diagnostic_metric_source"], "cpcv")
        self.assertEqual(backtest["validation_sources"]["tradable_metric_source"], "walk_forward_replay")
        self.assertTrue(backtest["validation_sources"]["all_required_sources_passed"])


if __name__ == "__main__":
    unittest.main()