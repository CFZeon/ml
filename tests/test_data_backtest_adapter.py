import unittest

import pandas as pd

from core.data import _COLUMNS, _prepare_frame
from core import join_custom_data, run_backtest


class DataBacktestAdapterTest(unittest.TestCase):
    def test_prepare_frame_accepts_header_row_archives(self):
        raw = pd.DataFrame(
            [
                _COLUMNS,
                [1704067200000, "100", "101", "99", "100.5", "10", 1704070799999, "1000", "12", "4", "400", "0"],
            ],
            columns=_COLUMNS,
        )

        prepared = _prepare_frame(raw)

        self.assertEqual(len(prepared), 1)
        self.assertAlmostEqual(float(prepared.iloc[0]["close"]), 100.5, places=6)

    def test_prepare_frame_backfills_missing_taker_columns(self):
        raw = pd.DataFrame(
            [
                [1704067200000, "100", "101", "99", "100.5", "10", 1704070799999, "1000", "12"],
            ],
            columns=_COLUMNS[:9],
        )

        prepared = _prepare_frame(raw)

        self.assertIn("taker_buy_base_vol", prepared.columns)
        self.assertIn("taker_buy_quote_vol", prepared.columns)
        self.assertAlmostEqual(float(prepared.iloc[0]["taker_buy_base_vol"]), 0.0, places=6)
        self.assertAlmostEqual(float(prepared.iloc[0]["taker_buy_quote_vol"]), 0.0, places=6)

    def test_point_in_time_custom_join_uses_availability_timestamp(self):
        index = pd.date_range("2026-03-10", periods=4, freq="1h", tz="UTC")
        base = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0],
                "high": [101.0, 102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0, 102.0],
                "close": [100.5, 101.5, 102.5, 103.5],
                "volume": [10.0, 11.0, 12.0, 13.0],
                "quote_volume": [1005.0, 1116.5, 1230.0, 1345.5],
                "trades": [100, 101, 102, 103],
            },
            index=index,
        )
        custom_frame = pd.DataFrame(
            {
                "timestamp": [index[0], index[1]],
                "available_at": [index[0] + pd.Timedelta(minutes=30), index[1] + pd.Timedelta(minutes=30)],
                "sentiment": [0.25, 0.75],
            }
        )

        joined, reports = join_custom_data(
            base,
            [
                {
                    "name": "sentiment_feed",
                    "frame": custom_frame,
                    "timestamp_column": "timestamp",
                    "availability_column": "available_at",
                    "prefix": "sent",
                }
            ],
        )

        self.assertTrue(pd.isna(joined.loc[index[0], "sent_sentiment"]))
        self.assertAlmostEqual(float(joined.loc[index[1], "sent_sentiment"]), 0.25, places=6)
        self.assertAlmostEqual(float(joined.loc[index[2], "sent_sentiment"]), 0.75, places=6)
        self.assertEqual(reports[0]["joined_columns"], ["sent_sentiment"])

    def test_vectorbt_backtest_adapter_supports_futures_execution_inputs(self):
        index = pd.date_range("2026-03-12", periods=6, freq="1h", tz="UTC")
        close = pd.Series([100.0, 101.0, 103.0, 102.0, 100.0, 99.0], index=index)
        signals = pd.Series([0.0, 0.5, 0.5, -0.5, -0.5, 0.0], index=index)
        funding_rates = pd.Series([0.0, 0.0, 0.0005, 0.0, -0.0003, 0.0], index=index)

        result = run_backtest(
            close=close,
            signals=signals,
            equity=10_000.0,
            fee_rate=0.001,
            slippage_rate=0.0005,
            execution_prices=close,
            signal_delay_bars=0,
            engine="vectorbt",
            market="um_futures",
            allow_short=True,
            leverage=1.0,
            funding_rates=funding_rates,
            symbol_filters={"tick_size": 0.1, "step_size": 0.001, "min_notional": 10.0},
        )

        self.assertEqual(result["engine"], "vectorbt")
        self.assertGreater(result["closed_trades"], 0)
        self.assertFalse(result["trade_ledger"].empty)
        self.assertIn("funding_pnl", result)
        self.assertIn("slippage_paid", result)
        self.assertIn("statistical_significance", result)
        self.assertFalse(result["statistical_significance"]["enabled"])

    def test_backtest_reports_stationary_bootstrap_confidence_intervals(self):
        index = pd.date_range("2026-03-12", periods=24, freq="1h", tz="UTC")
        close = pd.Series(
            [100.0, 100.8, 101.4, 101.1, 102.0, 102.8, 103.3, 103.0, 103.8, 104.5, 104.1, 105.0,
             105.7, 105.2, 106.0, 106.8, 106.5, 107.1, 107.9, 107.6, 108.4, 109.0, 108.7, 109.5],
            index=index,
        )
        signals = pd.Series(1.0, index=index)
        benchmark_close = pd.Series(
            [100.0, 100.4, 100.9, 100.7, 101.1, 101.4, 101.8, 101.7, 102.0, 102.4, 102.2, 102.6,
             102.9, 102.8, 103.1, 103.5, 103.4, 103.7, 104.0, 103.9, 104.2, 104.5, 104.4, 104.8],
            index=index,
        )

        result = run_backtest(
            close=close,
            signals=signals,
            equity=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            signal_delay_bars=0,
            engine="pandas",
            significance={"bootstrap_samples": 96, "mean_block_length": 4, "random_state": 7},
            benchmark_returns=benchmark_close.pct_change().fillna(0.0),
        )

        significance = result["statistical_significance"]
        self.assertTrue(significance["enabled"])
        self.assertEqual(significance["method"], "stationary_bootstrap")
        self.assertEqual(int(significance["bootstrap_samples"]), 96)
        self.assertEqual(int(significance["mean_block_length"]), 4)
        self.assertIn("benchmark_sharpe_ratio", significance)

        sharpe_stats = significance["metrics"]["sharpe_ratio"]
        self.assertIn("confidence_interval", sharpe_stats)
        self.assertIn("p_value_gt_zero", sharpe_stats)
        self.assertIn("p_value_gt_benchmark", sharpe_stats)
        self.assertLessEqual(sharpe_stats["confidence_interval"]["lower"], sharpe_stats["confidence_interval"]["upper"])
        self.assertGreaterEqual(float(sharpe_stats["p_value_gt_zero"]), 0.0)
        self.assertLessEqual(float(sharpe_stats["p_value_gt_zero"]), 1.0)
        self.assertGreaterEqual(float(sharpe_stats["p_value_gt_benchmark"]), 0.0)
        self.assertLessEqual(float(sharpe_stats["p_value_gt_benchmark"]), 1.0)

        for key in ["sortino_ratio", "calmar_ratio", "net_profit_pct", "max_drawdown"]:
            metric = significance["metrics"][key]
            self.assertIn("confidence_interval", metric)
            self.assertLessEqual(metric["confidence_interval"]["lower"], metric["confidence_interval"]["upper"])

    def test_backtest_significance_handles_short_samples(self):
        index = pd.date_range("2026-03-12", periods=5, freq="1h", tz="UTC")
        close = pd.Series([100.0, 100.5, 101.0, 100.8, 101.2], index=index)
        signals = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0], index=index)

        result = run_backtest(
            close=close,
            signals=signals,
            equity=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            signal_delay_bars=0,
            engine="pandas",
        )

        significance = result["statistical_significance"]
        self.assertFalse(significance["enabled"])
        self.assertEqual(significance["reason"], "insufficient_observations")


if __name__ == "__main__":
    unittest.main()