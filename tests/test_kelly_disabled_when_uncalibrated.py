import unittest

import pandas as pd

from core.pipeline import _build_signal_state


class KellyDisabledWhenUncalibratedTest(unittest.TestCase):
    def test_trade_ready_kelly_is_capped_until_paper_and_calibration_are_green(self):
        index = pd.date_range("2026-04-02", periods=1, freq="1h", tz="UTC")
        predictions = pd.Series([1], index=index)
        probability_frame = pd.DataFrame({-1: [0.05], 0: [0.0], 1: [0.95]}, index=index)
        profitability_prob = pd.Series([0.90], index=index)

        state = _build_signal_state(
            predictions,
            probability_frame,
            profitability_prob,
            {
                "threshold": 0.0,
                "edge_threshold": 0.0,
                "meta_threshold": 0.55,
                "fraction": 1.0,
                "sizing_mode": "kelly",
                "min_trades_for_kelly": 30,
                "max_kelly_fraction": 0.5,
                "trade_ready_mode": True,
                "uncalibrated_kelly_fraction_cap": 0.25,
            },
            avg_win=0.04,
            avg_loss=0.01,
            holding_bars=1,
            kelly_trade_count=50,
        )

        self.assertTrue(state["kelly_calibration_blocked"])
        self.assertIn("paper_verification_required", state["kelly_calibration_reasons"])
        self.assertIn("live_calibration_unavailable", state["kelly_calibration_reasons"])
        self.assertAlmostEqual(float(state["position_size"].iloc[0]), 0.25, places=6)

    def test_trade_ready_kelly_uses_configured_cap_after_green_paper_and_calibration(self):
        index = pd.date_range("2026-04-02", periods=1, freq="1h", tz="UTC")
        predictions = pd.Series([1], index=index)
        probability_frame = pd.DataFrame({-1: [0.05], 0: [0.0], 1: [0.95]}, index=index)
        profitability_prob = pd.Series([0.90], index=index)

        state = _build_signal_state(
            predictions,
            probability_frame,
            profitability_prob,
            {
                "threshold": 0.0,
                "edge_threshold": 0.0,
                "meta_threshold": 0.55,
                "fraction": 1.0,
                "sizing_mode": "kelly",
                "min_trades_for_kelly": 30,
                "max_kelly_fraction": 0.5,
                "trade_ready_mode": True,
                "paper_verified": True,
                "live_calibration_error": 0.05,
                "max_live_calibration_error": 0.25,
                "uncalibrated_kelly_fraction_cap": 0.25,
            },
            avg_win=0.04,
            avg_loss=0.01,
            holding_bars=1,
            kelly_trade_count=50,
        )

        self.assertFalse(state["kelly_calibration_blocked"])
        self.assertEqual(state["kelly_calibration_reasons"], [])
        self.assertAlmostEqual(float(state["position_size"].iloc[0]), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()