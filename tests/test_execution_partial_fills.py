import unittest

import pandas as pd

from core import run_backtest


class ExecutionPartialFillTest(unittest.TestCase):
    def test_unified_action_latency_delays_submission_and_reports_fill_delay(self):
        index = pd.date_range("2024-01-01", periods=5, freq="h")
        close = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=index)
        signals = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0], index=index)
        volume = pd.Series([1_000.0] * len(index), index=index)

        result = run_backtest(
            close,
            signals,
            engine="pandas",
            signal_delay_bars=0,
            volume=volume,
            execution_policy={
                "adapter": "nautilus",
                "force_simulation": True,
                "time_in_force": "IOC",
                "participation_cap": 1.0,
                "action_latency_bars": 1,
            },
        )

        self.assertFalse(result["order_intents"].empty)
        first_intent = result["order_intents"].iloc[0]
        self.assertEqual(pd.Timestamp(first_intent["request_timestamp"]), index[1])
        self.assertEqual(pd.Timestamp(first_intent["timestamp"]), index[2])
        self.assertAlmostEqual(float(result["average_action_delay_bars"]), 1.0, places=6)
        self.assertAlmostEqual(float(result["average_fill_delay_bars"]), 1.0, places=6)

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
                "force_simulation": True,
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
                "force_simulation": True,
                "time_in_force": "GTC",
                "participation_cap": 1.0,
                "max_order_age_bars": 2,
                "cancel_replace_bars": 2,
            },
        )

        self.assertGreaterEqual(int(result["cancelled_orders"]), 1)
        self.assertIn("max_order_age", set(result["order_ledger"].get("reason", pd.Series(dtype=object)).dropna()))
        self.assertAlmostEqual(float(result["order_ledger"].get("executed_notional", pd.Series(dtype=float)).sum()), 0.0, places=6)

    def test_bar_engine_rejects_passive_limit_orders(self):
        index = pd.date_range("2024-01-01", periods=3, freq="h")
        close = pd.Series([100.0, 100.0, 100.0], index=index)
        signals = pd.Series([0.0, 1.0, 0.0], index=index)

        with self.assertRaisesRegex(ValueError, "Passive or limit order types"):
            run_backtest(
                close,
                signals,
                engine="pandas",
                signal_delay_bars=0,
                volume=pd.Series([1_000.0] * len(index), index=index),
                execution_policy={
                    "adapter": "nautilus",
                    "force_simulation": True,
                    "order_type": "limit",
                    "time_in_force": "GTC",
                },
            )

    def test_blocking_futures_margin_safety_clips_unsafe_targets(self):
        index = pd.date_range("2024-01-01", periods=4, freq="h")
        close = pd.Series([100.0, 100.0, 100.0, 100.0], index=index)
        signals = pd.Series([0.0, 1.0, 1.0, 0.0], index=index)

        result = run_backtest(
            close,
            signals,
            engine="pandas",
            signal_delay_bars=0,
            market="um_futures",
            leverage=20.0,
            volume=pd.Series([1_000.0] * len(index), index=index),
            execution_policy={
                "adapter": "nautilus",
                "force_simulation": True,
                "time_in_force": "IOC",
                "participation_cap": 1.0,
            },
            futures_account={
                "enabled": True,
                "margin_mode": "cross",
                "leverage": 20.0,
                "warning_margin_ratio": 0.8,
                "maintenance_margin_ratio": 0.08,
                "margin_safety_mode": "blocking",
            },
        )

        self.assertGreater(int(result["margin_safety_adjustments"]), 0)
        self.assertEqual(result["margin_safety_mode"], "blocking")
        self.assertLess(float(result["max_realized_leverage"]), 20.0)
        self.assertFalse(result["margin_safety_adjustment_log"].empty)


if __name__ == "__main__":
    unittest.main()