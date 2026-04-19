import unittest

import pandas as pd

from core import run_backtest


class ExecutionPartialFillTest(unittest.TestCase):
    def test_oversized_orders_fill_partially_and_cancel_residual(self):
        index = pd.date_range("2024-01-01", periods=3, freq="h")
        close = pd.Series([100.0, 100.0, 100.0], index=index)
        signals = pd.Series([1.0, 1.0, 1.0], index=index)
        volume = pd.Series([5.0, 5.0, 5.0], index=index)

        result = run_backtest(
            close,
            signals,
            engine="pandas",
            signal_delay_bars=0,
            volume=volume,
            execution_policy={
                "adapter": "nautilus",
                "time_in_force": "IOC",
                "participation_cap": 1.0,
            },
        )

        self.assertEqual(result["execution_adapter"], "nautilus")
        self.assertGreater(int(result["partial_fill_orders"]), 0)
        self.assertGreater(int(result["cancelled_orders"]), 0)
        self.assertGreater(float(result["unfilled_notional"]), 0.0)
        self.assertLess(float(result["fill_ratio"]), 1.0)
        self.assertFalse(result["order_intents"].empty)

    def test_aged_gtc_orders_cancel_when_left_unfilled(self):
        index = pd.date_range("2024-01-01", periods=4, freq="h")
        close = pd.Series([100.0, 100.0, 100.0, 100.0], index=index)
        signals = pd.Series([0.0, 1.0, 1.0, 1.0], index=index)
        volume = pd.Series([0.0, 0.0, 0.0, 100.0], index=index)

        result = run_backtest(
            close,
            signals,
            engine="pandas",
            signal_delay_bars=0,
            volume=volume,
            execution_policy={
                "adapter": "nautilus",
                "time_in_force": "GTC",
                "participation_cap": 1.0,
                "max_order_age_bars": 2,
                "cancel_replace_bars": 2,
            },
        )

        self.assertGreaterEqual(int(result["cancelled_orders"]), 1)
        self.assertIn("max_order_age", set(result["order_ledger"].get("reason", pd.Series(dtype=object)).dropna()))
        self.assertAlmostEqual(float(result["order_ledger"].get("executed_notional", pd.Series(dtype=float)).sum()), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()