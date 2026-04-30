import unittest

import pandas as pd

from core import ResearchPipeline
from core.labeling import triple_barrier_labels


class LabelingFutureWindowIntegrityTest(unittest.TestCase):
    def test_triple_barrier_drops_rows_with_nan_in_forward_window(self):
        index = pd.date_range("2026-06-01", periods=6, freq="1h", tz="UTC")
        close = pd.Series([100.0, 100.5, 101.0, 101.5, 102.0, 102.5], index=index)
        volatility = pd.Series(0.01, index=index)
        high = close + 1.0
        low = close - 1.0
        high.iloc[1] = float("nan")

        labels = triple_barrier_labels(
            close=close,
            volatility=volatility,
            high=high,
            low=low,
            pt_sl=(1.0, 1.0),
            max_holding=2,
        )

        self.assertEqual(list(labels.index), list(index[2:4]))
        self.assertEqual(
            labels.attrs["integrity_report"]["dropped_incomplete_future_windows"],
            2,
        )

    def test_complete_forward_window_still_produces_expected_barrier(self):
        index = pd.date_range("2026-06-01", periods=5, freq="1h", tz="UTC")
        close = pd.Series([100.0, 100.5, 101.0, 101.5, 102.0], index=index)
        volatility = pd.Series(0.01, index=index)
        high = pd.Series([101.5, 101.6, 101.7, 101.8, 101.9], index=index)
        low = pd.Series([99.5, 100.0, 100.5, 101.0, 101.5], index=index)

        labels = triple_barrier_labels(
            close=close,
            volatility=volatility,
            high=high,
            low=low,
            pt_sl=(1.0, 1.0),
            max_holding=2,
        )

        self.assertEqual(labels.iloc[0]["barrier"], "pt")
        self.assertEqual(labels.attrs["integrity_report"]["dropped_incomplete_future_windows"], 0)

    def test_pipeline_build_labels_preserves_integrity_report(self):
        index = pd.date_range("2026-06-01", periods=6, freq="1h", tz="UTC")
        frame = pd.DataFrame(
            {
                "open": [100.0, 100.5, 101.0, 101.5, 102.0, 102.5],
                "high": [101.0, float("nan"), 102.0, 102.5, 103.0, 103.5],
                "low": [99.0, 99.5, 100.0, 100.5, 101.0, 101.5],
                "close": [100.0, 100.5, 101.0, 101.5, 102.0, 102.5],
                "volume": [1000.0] * 6,
                "quote_volume": [100000.0] * 6,
                "trades": [100] * 6,
            },
            index=index,
        )
        pipeline = ResearchPipeline(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "labels": {
                    "kind": "triple_barrier",
                    "pt_sl": (1.0, 1.0),
                    "max_holding": 2,
                    "missing_future_policy": "drop",
                    "volatility_builder": lambda _pipeline: pd.Series(0.01, index=index),
                },
                "backtest": {"use_open_execution": False, "signal_delay_bars": 0},
            }
        )
        pipeline.state["raw_data"] = frame.copy()
        pipeline.state["data"] = frame.copy()

        labels = pipeline.build_labels()

        self.assertGreater(
            labels.attrs["integrity_report"]["dropped_incomplete_future_windows"],
            0,
        )
        self.assertNotIn(index[0], labels.index)


if __name__ == "__main__":
    unittest.main()