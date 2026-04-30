import unittest
from unittest import mock

import pandas as pd

from core import (
    ResearchPipeline,
    load_historical_universe_snapshot,
    run_backtest,
)
from example_utils import build_example_universe_config


class HistoricalUniverseSelectionTest(unittest.TestCase):
    def test_delisted_symbols_disappear_from_later_snapshots(self):
        snapshots = [
            {
                "snapshot_timestamp": "2024-01-01T00:00:00Z",
                "symbols": [
                    {"symbol": "BTCUSDT", "status": "TRADING", "listing_start": "2020-01-01T00:00:00Z"},
                    {"symbol": "OLDUSDT", "status": "TRADING", "listing_start": "2020-06-01T00:00:00Z"},
                ],
            },
            {
                "snapshot_timestamp": "2024-02-01T00:00:00Z",
                "symbols": [
                    {"symbol": "BTCUSDT", "status": "TRADING", "listing_start": "2020-01-01T00:00:00Z"},
                ],
            },
        ]

        early_snapshot = load_historical_universe_snapshot("2024-01-15T00:00:00Z", snapshots=snapshots)
        later_snapshot = load_historical_universe_snapshot("2024-02-15T00:00:00Z", snapshots=snapshots)

        self.assertIn("OLDUSDT", early_snapshot.symbols["symbol"].tolist())
        self.assertNotIn("OLDUSDT", later_snapshot.symbols["symbol"].tolist())

    def test_cross_symbol_studies_reject_ineligible_symbols_at_study_start(self):
        index = pd.date_range("2024-02-01", periods=6, freq="h", tz="UTC")
        frame = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
                "high": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
                "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0],
                "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
                "volume": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                "quote_volume": [1005.0, 1116.5, 1230.0, 1345.5, 1463.0, 1582.5],
                "trades": [100, 101, 102, 103, 104, 105],
                "taker_buy_base_vol": [4.0, 4.1, 4.2, 4.3, 4.4, 4.5],
                "taker_buy_quote_vol": [400.0, 410.0, 420.0, 430.0, 440.0, 450.0],
            },
            index=index,
        )
        integrity_report = {"status": "complete", "missing_rows": 0, "periods": []}
        pipeline = ResearchPipeline(
            {
                "data": {
                    "symbol": "BTCUSDT",
                    "interval": "1h",
                    "start": "2024-02-01",
                    "end": "2024-02-01 06:00:00",
                    "cross_asset_context": {"symbols": ["ETHUSDT", "OLDUSDT"]},
                },
                "universe": {
                    "snapshots": [
                        {
                            "snapshot_timestamp": "2024-02-01T00:00:00Z",
                            "symbols": [
                                {
                                    "symbol": "BTCUSDT",
                                    "status": "TRADING",
                                    "listing_start": "2020-01-01T00:00:00Z",
                                    "avg_daily_quote_volume": 10_000_000.0,
                                },
                                {
                                    "symbol": "ETHUSDT",
                                    "status": "TRADING",
                                    "listing_start": "2020-01-01T00:00:00Z",
                                    "avg_daily_quote_volume": 8_000_000.0,
                                },
                                {
                                    "symbol": "OLDUSDT",
                                    "status": "DELISTED",
                                    "listing_start": "2020-01-01T00:00:00Z",
                                    "delisting_end": "2024-01-15T00:00:00Z",
                                    "avg_daily_quote_volume": 7_000_000.0,
                                },
                            ],
                        }
                    ],
                    "requested_symbol_policy": "error",
                    "min_history_days": 30,
                    "min_liquidity": 1_000_000.0,
                },
            }
        )

        with mock.patch("core.pipeline.fetch_binance_bars", return_value=(frame, integrity_report)), mock.patch(
            "core.pipeline.fetch_binance_symbol_filters", return_value={}
        ), mock.patch("core.pipeline.fetch_binance_futures_context", return_value={}):
            with self.assertRaisesRegex(ValueError, "OLDUSDT"):
                pipeline.fetch_data()

    def test_delisting_events_trigger_configured_backtest_handling(self):
        index = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
        close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], index=index)
        signals = pd.Series([0.0, 1.0, 1.0, 1.0, 1.0, 1.0], index=index)
        result = run_backtest(
            close=close,
            signals=signals,
            execution_prices=close,
            equity=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            signal_delay_bars=0,
            engine="pandas",
            volume=pd.Series(1_000.0, index=index),
            symbol_lifecycle=[
                {"symbol": "BTCUSDT", "timestamp": index[3], "status": "DELISTED"},
            ],
            symbol_lifecycle_policy={"delist_action": "liquidate"},
        )

        trade_ledger = result["trade_ledger"]
        lifecycle_report = result["symbol_lifecycle_report"]

        self.assertEqual(int(lifecycle_report["delist_events"]), 1)
        self.assertEqual(int(lifecycle_report["forced_liquidations"]), 1)
        self.assertEqual(len(trade_ledger), 1)
        self.assertEqual(pd.Timestamp(trade_ledger.iloc[0]["exit_time"]), index[3])

    def test_local_certification_rejects_synthetic_example_universe_snapshot(self):
        index = pd.date_range("2024-02-01", periods=6, freq="h", tz="UTC")
        frame = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
                "high": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
                "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0],
                "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
                "volume": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                "quote_volume": [1005.0, 1116.5, 1230.0, 1345.5, 1463.0, 1582.5],
                "trades": [100, 101, 102, 103, 104, 105],
                "taker_buy_base_vol": [4.0, 4.1, 4.2, 4.3, 4.4, 4.5],
                "taker_buy_quote_vol": [400.0, 410.0, 420.0, 430.0, 440.0, 450.0],
            },
            index=index,
        )
        integrity_report = {"status": "complete", "missing_rows": 0, "periods": []}
        pipeline = ResearchPipeline(
            {
                "data": {
                    "symbol": "BTCUSDT",
                    "interval": "1h",
                    "start": "2024-02-01",
                    "end": "2024-02-01 06:00:00",
                },
                "universe": build_example_universe_config(
                    "BTCUSDT",
                    market="spot",
                    snapshot_timestamp="2024-02-01T00:00:00Z",
                ),
                "backtest": {"evaluation_mode": "local_certification"},
            }
        )

        with mock.patch("core.pipeline.fetch_binance_bars", return_value=(frame, integrity_report)), mock.patch(
            "core.pipeline.fetch_binance_symbol_filters", return_value={}
        ):
            with self.assertRaisesRegex(ValueError, "Synthetic example universe snapshot is not allowed"):
                pipeline.fetch_data()

    def test_local_certification_accepts_explicit_frozen_universe_snapshot(self):
        index = pd.date_range("2024-02-01", periods=6, freq="h", tz="UTC")
        frame = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
                "high": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
                "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0],
                "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
                "volume": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                "quote_volume": [1005.0, 1116.5, 1230.0, 1345.5, 1463.0, 1582.5],
                "trades": [100, 101, 102, 103, 104, 105],
                "taker_buy_base_vol": [4.0, 4.1, 4.2, 4.3, 4.4, 4.5],
                "taker_buy_quote_vol": [400.0, 410.0, 420.0, 430.0, 440.0, 450.0],
            },
            index=index,
        )
        integrity_report = {"status": "complete", "missing_rows": 0, "periods": []}
        pipeline = ResearchPipeline(
            {
                "data": {
                    "symbol": "BTCUSDT",
                    "interval": "1h",
                    "start": "2024-02-01",
                    "end": "2024-02-01 06:00:00",
                },
                "universe": {
                    "market": "spot",
                    "snapshots": [
                        {
                            "snapshot_timestamp": "2024-02-01T00:00:00Z",
                            "market": "spot",
                            "source": "fixture_snapshot",
                            "symbols": [
                                {
                                    "symbol": "BTCUSDT",
                                    "market": "spot",
                                    "status": "TRADING",
                                    "listing_start": "2020-01-01T00:00:00Z",
                                    "avg_daily_quote_volume": 10_000_000.0,
                                }
                            ],
                        }
                    ],
                },
                "backtest": {"evaluation_mode": "local_certification"},
            }
        )

        with mock.patch("core.pipeline.fetch_binance_bars", return_value=(frame, integrity_report)), mock.patch(
            "core.pipeline.fetch_binance_symbol_filters", return_value={}
        ):
            data = pipeline.fetch_data()

        self.assertEqual(len(data), len(frame))
        self.assertEqual(pipeline.state["universe_snapshot_meta"]["source"], "fixture_snapshot")


if __name__ == "__main__":
    unittest.main()