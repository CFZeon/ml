import unittest

import pandas as pd

from core import build_signal_decay_report, resolve_effective_delay_bars


class SignalDecayMetricsTest(unittest.TestCase):
    def test_signal_decay_report_measures_half_life_and_effective_delay(self):
        index = pd.date_range("2026-03-11", periods=6, freq="1h", tz="UTC")
        valuation = pd.Series([100.0, 102.0, 100.0, 99.0, 99.0, 99.0], index=index)
        signals = pd.Series([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=index)
        predictions = pd.Series([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=index)
        direction_edge = pd.Series([0.85, 0.0, 0.0, 0.0, 0.0, 0.0], index=index)

        report = build_signal_decay_report(
            [
                {
                    "predictions": predictions,
                    "direction_edge": direction_edge,
                    "event_signals": signals,
                    "valuation_prices": valuation,
                    "execution_prices": valuation,
                    "fee_rate": 0.0,
                    "slippage_rate": 0.0,
                    "equity": 10_000.0,
                }
            ],
            holding_bars=1,
            signal_delay_bars=1,
            execution_policy={"max_order_age_bars": 2, "cancel_replace_bars": 1},
            config={
                "min_realized_trade_count": 1,
                "min_half_life_holding_ratio": 0.0,
            },
        )

        self.assertEqual(resolve_effective_delay_bars(1, {"max_order_age_bars": 2}), 2)
        self.assertEqual(report["effective_delay_bars"], 2)
        self.assertEqual(report["half_life_bars"], 2)
        self.assertLess(float(report["net_edge_at_effective_delay"]), 0.0)
        self.assertFalse(report["promotion_pass"])
        self.assertIn("edge_decays_before_effective_delay", report["reasons"])

    def test_signal_decay_gate_downgrades_to_advisory_for_low_trade_count(self):
        index = pd.date_range("2026-03-12", periods=6, freq="1h", tz="UTC")
        valuation = pd.Series([100.0, 101.0, 100.5, 100.0, 100.0, 100.0], index=index)
        signals = pd.Series([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=index)

        report = build_signal_decay_report(
            [
                {
                    "predictions": signals,
                    "direction_edge": pd.Series([0.6, 0.0, 0.0, 0.0, 0.0, 0.0], index=index),
                    "event_signals": signals,
                    "valuation_prices": valuation,
                    "execution_prices": valuation,
                }
            ],
            holding_bars=1,
            signal_delay_bars=1,
            execution_policy={"max_order_age_bars": 1, "cancel_replace_bars": 1},
            config={
                "min_realized_trade_count": 5,
                "min_half_life_holding_ratio": 0.0,
            },
        )

        self.assertTrue(report["promotion_pass"])
        self.assertEqual(report["gate_mode"], "advisory")
        self.assertTrue(report["low_sample_advisory"])
        self.assertIn("insufficient_realized_decay_trade_count", report["warnings"])


if __name__ == "__main__":
    unittest.main()