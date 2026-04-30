import unittest
from unittest import mock

import pandas as pd

from core import ResearchPipeline
from core.data_quality import check_data_quality


class DataQualityQuarantineTest(unittest.TestCase):
    def test_synthetic_bad_prints_are_detected_and_quarantined(self):
        index = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
        frame = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0],
                "high": [101.0, 100.5, 103.0, 104.0],
                "low": [99.0, 101.5, 101.0, 102.0],
                "close": [100.5, 100.8, 102.5, 103.5],
                "volume": [10.0, 0.0, 12.0, 13.0],
                "quote_volume": [1005.0, 0.0, 1230.0, 1345.5],
                "trades": [100, 101, 102, 103],
            },
            index=index,
        )

        result = check_data_quality(frame)

        self.assertTrue(bool(result.quarantine_mask.iloc[1]))
        self.assertEqual(int(result.report["anomalies"]["ohlc_inconsistency"]["count"]), 1)
        self.assertEqual(int(result.report["anomalies"]["nonpositive_volume"]["count"]), 1)
        self.assertEqual(int(result.report["summary"]["quarantined_rows"]), 1)

    def test_quarantined_rows_are_removed_before_features_and_labels(self):
        index = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
        bad_timestamp = index[2]
        frame = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "high": [101.0, 102.0, 101.0, 104.0, 105.0],
                "low": [99.0, 100.0, 103.0, 102.0, 103.0],
                "close": [100.5, 101.5, 102.5, 103.5, 104.5],
                "volume": [10.0, 11.0, 0.0, 13.0, 14.0],
                "quote_volume": [1005.0, 1116.5, 0.0, 1345.5, 1463.0],
                "trades": [100, 101, 102, 103, 104],
                "taker_buy_base_vol": [4.0, 4.1, 0.0, 4.3, 4.4],
                "taker_buy_quote_vol": [400.0, 410.0, 0.0, 430.0, 440.0],
            },
            index=index,
        )
        report = {"status": "complete", "missing_rows": 0, "periods": []}
        pipeline = ResearchPipeline(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h", "start": "2024-01-01", "end": "2024-01-01 05:00:00"},
                "data_quality": {
                    "actions": {
                        "ohlc_inconsistency": "drop",
                        "nonpositive_volume": "drop",
                    }
                },
                "labels": {"kind": "fixed_horizon", "horizon": 1, "threshold": 0.0},
            }
        )

        with mock.patch("core.pipeline.fetch_binance_bars", return_value=(frame, report)), mock.patch("core.pipeline.fetch_binance_symbol_filters", return_value={}):
            pipeline.fetch_data()
            pipeline.check_data_quality()
            pipeline.run_indicators()
            pipeline.build_features()
            pipeline.build_labels()

        self.assertNotIn(bad_timestamp, pipeline.state["raw_data"].index)
        self.assertNotIn(bad_timestamp, pipeline.state["data"].index)
        self.assertNotIn(bad_timestamp, pipeline.state["features"].index)
        self.assertNotIn(bad_timestamp, pipeline.state["labels"].index)

    def test_quality_report_counts_anomalies_by_type_and_action(self):
        index = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
        frame = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0],
                "high": [101.0, 102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0, 102.0],
                "close": [100.5, 140.0, 102.5, 103.5],
                "volume": [10.0, 11.0, 12.0, 13.0],
                "quote_volume": [1005.0, 1116.5, 1230.0, 5000.0],
                "trades": [100, 101, 5000, 103],
            },
            index=index,
        )

        result = check_data_quality(
            frame,
            config={
                "actions": {
                    "return_spike": "null",
                    "quote_volume_inconsistency": "flag",
                    "trade_count_anomaly": "flag",
                },
                "return_spike_threshold": 0.5,
                "quote_volume_tolerance": 0.5,
                "trade_count_spike_threshold": 5.0,
            },
        )

        self.assertEqual(result.report["anomalies"]["return_spike"]["action"], "null")
        self.assertEqual(result.report["anomalies"]["quote_volume_inconsistency"]["action"], "flag")
        self.assertGreaterEqual(int(result.report["summary"]["action_counts"]["flag"]), 1)
        self.assertGreaterEqual(int(result.report["summary"]["action_counts"]["null"]), 1)
        self.assertIn("quarantine_disposition_counts", result.report["summary"])

    def test_flagged_rows_can_be_excluded_from_modeling_without_blocking(self):
        index = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
        frame = pd.DataFrame(
            {
                "open": [100.0, 100.2, 130.0, 100.4, 100.6, 100.8],
                "high": [100.5, 100.7, 130.5, 100.9, 101.1, 101.3],
                "low": [99.5, 99.7, 129.5, 99.9, 100.1, 100.3],
                "close": [100.1, 100.3, 130.2, 100.5, 100.7, 100.9],
                "volume": [10.0, 10.5, 11.0, 11.5, 12.0, 12.5],
                "quote_volume": [1001.0, 1053.15, 1432.2, 1155.75, 1208.4, 1261.25],
                "trades": [100, 101, 102, 103, 104, 105],
                "taker_buy_base_vol": [4.0, 4.1, 4.2, 4.3, 4.4, 4.5],
                "taker_buy_quote_vol": [400.0, 410.0, 420.0, 430.0, 440.0, 450.0],
            },
            index=index,
        )
        report = {"status": "complete", "missing_rows": 0, "periods": []}
        pipeline = ResearchPipeline(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h", "start": "2024-01-01", "end": "2024-01-01 06:00:00"},
                "data_quality": {
                    "exclude_flagged_quarantine_rows_from_modeling": True,
                    "rolling_window": 5,
                    "return_spike_threshold": 2.0,
                    "actions": {"return_spike": "flag"},
                },
                "labels": {"kind": "fixed_horizon", "horizon": 1, "threshold": 0.0},
            }
        )

        with mock.patch("core.pipeline.fetch_binance_bars", return_value=(frame, report)), mock.patch("core.pipeline.fetch_binance_symbol_filters", return_value={}):
            pipeline.fetch_data()
            pipeline.check_data_quality()
            pipeline.run_indicators()
            pipeline.build_features()
            pipeline.build_labels()

        spike_timestamp = index[2]
        self.assertNotIn(spike_timestamp, pipeline.state["raw_data"].index)
        self.assertNotIn(spike_timestamp, pipeline.state["data"].index)
        self.assertNotIn(spike_timestamp, pipeline.state["features"].index)
        self.assertNotIn(spike_timestamp, pipeline.state["labels"].index)
        summary = pipeline.state["data_quality_report"]["summary"]
        self.assertGreaterEqual(int(summary["modeling_excluded_rows"]), 1)
        self.assertEqual(
            int(summary["modeling_excluded_rows"]),
            int(summary["flagged_only_rows"]),
        )
        self.assertFalse(bool(pipeline.state["data_quality_report"]["blocking"]))
        self.assertEqual(
            pipeline.state["data_quality_report"]["quarantine_severity"],
            "flag_advisory",
        )

    def test_quarantine_can_block_run_when_configured(self):
        index = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
        frame = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 100.0, 103.0],
                "low": [99.0, 101.5, 101.0],
                "close": [100.5, 100.8, 102.5],
                "volume": [10.0, 0.0, 12.0],
                "quote_volume": [1005.0, 0.0, 1230.0],
                "trades": [100, 101, 102],
            },
            index=index,
        )

        result = check_data_quality(frame, config={"block_on_quarantine": True})

        self.assertEqual(result.report["status"], "quarantine")
        self.assertTrue(result.report["blocking"])


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()