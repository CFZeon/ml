import unittest

import pandas as pd

from core import LiquidityInputResolver, OrderBookImpactModel, SquareRootImpactModel, run_backtest


class _RecordingSlippageModel:
    def __init__(self):
        self.last_volume = None
        self.last_orderbook_depth = None

    def estimate(self, trade_notional, volume, volatility, price, orderbook_depth=None):
        self.last_volume = pd.Series(volume, copy=True)
        self.last_orderbook_depth = None if orderbook_depth is None else orderbook_depth.copy()
        return pd.Series(0.0, index=price.index, dtype=float)


class CausalLiquidityInputsTest(unittest.TestCase):
    def test_open_execution_uses_lagged_bar_volume(self):
        index = pd.date_range("2026-06-10", periods=5, freq="1h", tz="UTC")
        close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=index)
        open_ = pd.Series([99.5, 100.5, 101.5, 102.5, 103.5], index=index)
        signals = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0], index=index)
        volume = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0], index=index)
        slippage_model = _RecordingSlippageModel()

        run_backtest(
            close=close,
            signals=signals,
            equity=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            execution_prices=open_,
            signal_delay_bars=0,
            engine="pandas",
            volume=volume,
            slippage_model=slippage_model,
        )

        expected = volume.shift(1).fillna(0.0)
        pd.testing.assert_series_equal(slippage_model.last_volume, expected, check_names=False)

    def test_lagged_adv_uses_only_prior_bars(self):
        index = pd.date_range("2026-06-11", periods=6, freq="1h", tz="UTC")
        volume = pd.Series([100.0, 200.0, 300.0, 600.0, 900.0, 1200.0], index=index)
        resolver = LiquidityInputResolver(liquidity_lag_bars=1)

        resolved = resolver.resolve(index=index, volume=volume, slippage_model=SquareRootImpactModel(adv_window=3))
        adv = resolved["volume"].rolling(3).mean()

        self.assertAlmostEqual(float(adv.iloc[3]), float(volume.iloc[0:3].mean()), places=6)
        self.assertNotAlmostEqual(float(adv.iloc[3]), float(volume.iloc[1:4].mean()), places=6)

    def test_post_timestamp_orderbook_snapshots_are_rejected(self):
        index = pd.date_range("2026-06-12", periods=3, freq="1h", tz="UTC")
        orderbook_depth = pd.DataFrame(
            {
                "snapshot_time": [index[0] + pd.Timedelta(minutes=1), index[1], index[2]],
                "bid_notional": [10_000.0, 11_000.0, 12_000.0],
                "ask_notional": [9_000.0, 10_000.0, 11_000.0],
            },
            index=index,
        )

        with self.assertRaisesRegex(ValueError, "ex-post rows"):
            LiquidityInputResolver(liquidity_lag_bars=1).resolve(
                index=index,
                orderbook_depth=orderbook_depth,
                slippage_model=OrderBookImpactModel(),
            )

    def test_backtest_reports_causal_liquidity_diagnostics(self):
        index = pd.date_range("2026-06-13", periods=5, freq="1h", tz="UTC")
        close = pd.Series([100.0, 101.0, 100.5, 102.0, 103.0], index=index)
        signals = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0], index=index)
        volume = pd.Series([150.0, 250.0, 350.0, 450.0, 550.0], index=index)

        result = run_backtest(
            close=close,
            signals=signals,
            equity=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            signal_delay_bars=0,
            engine="pandas",
            volume=volume,
            slippage_model=_RecordingSlippageModel(),
        )

        self.assertIn("liquidity_report", result)
        self.assertEqual(result["liquidity_report"]["liquidity_source"], "lagged_bar_volume")
        self.assertEqual(result["liquidity_report"]["liquidity_lag_bars"], 1)
        self.assertEqual(result["liquidity_report"]["ex_post_liquidity_rows"], 0)


if __name__ == "__main__":
    unittest.main()