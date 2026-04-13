import unittest

import pandas as pd

from core import build_trade_outcome_frame
from core.pipeline import _build_signal_state


class SignalProfitabilitySizingTest(unittest.TestCase):
    def test_trade_outcome_frame_uses_profitability_not_raw_correctness(self):
        index = pd.date_range("2026-03-01", periods=4, freq="1h", tz="UTC")
        predictions = pd.Series([1, -1, 1, 0], index=index)
        labels = pd.DataFrame(
            {
                "label": [0, 0, 1, -1],
                "gross_return": [0.0200, -0.0150, 0.0005, -0.0100],
                "cost_rate": [0.0010, 0.0010, 0.0010, 0.0010],
            },
            index=index,
        )

        outcomes = build_trade_outcome_frame(predictions, labels)

        self.assertListEqual(outcomes["profitable"].tolist(), [1, 1, 0, 0])
        self.assertAlmostEqual(float(outcomes.loc[index[0], "net_trade_return"]), 0.0190, places=6)
        self.assertAlmostEqual(float(outcomes.loc[index[1], "net_trade_return"]), 0.0140, places=6)
        self.assertAlmostEqual(float(outcomes.loc[index[2], "net_trade_return"]), -0.0005, places=6)

    def test_expected_utility_sizing_uses_break_even_probability_gate(self):
        index = pd.date_range("2026-03-05", periods=2, freq="1h", tz="UTC")
        predictions = pd.Series([1, -1], index=index)
        probability_frame = pd.DataFrame(
            {
                -1: [0.35, 0.70],
                0: [0.0, 0.0],
                1: [0.65, 0.30],
            },
            index=index,
        )
        profitability_prob = pd.Series([0.48, 0.62], index=index)

        state = _build_signal_state(
            predictions,
            probability_frame,
            profitability_prob,
            {
                "threshold": 0.0,
                "edge_threshold": 0.0,
                "meta_threshold": 0.55,
                "fraction": 1.0,
                "sizing_mode": "expected_utility",
            },
            avg_win=0.015,
            avg_loss=0.010,
            holding_bars=1,
        )

        self.assertLess(state["break_even_profit_prob"], 0.48)
        self.assertGreater(float(state["position_size"].iloc[0]), 0.0)
        self.assertGreater(float(state["position_size"].iloc[1]), 0.0)
        self.assertNotEqual(int(state["signals"].iloc[0]), 0)
        self.assertNotEqual(int(state["signals"].iloc[1]), 0)


if __name__ == "__main__":
    unittest.main()