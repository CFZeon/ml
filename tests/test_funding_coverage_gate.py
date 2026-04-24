import unittest

import pandas as pd

from core import ResearchPipeline
from core.pipeline import _resolve_backtest_funding_missing_policy, _resolve_backtest_funding_rates


class FundingCoverageGateTest(unittest.TestCase):
    def test_trade_ready_defaults_to_strict_funding_policy(self):
        policy = _resolve_backtest_funding_missing_policy({"evaluation_mode": "trade_ready"})

        self.assertEqual(policy["mode"], "strict")

    def test_trade_ready_blocks_missing_funding_even_when_zero_fill_is_requested(self):
        index = pd.date_range("2026-02-01", periods=24, freq="1h", tz="UTC")
        pipeline = ResearchPipeline(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h", "market": "um_futures"},
                "backtest": {
                    "evaluation_mode": "trade_ready",
                    "apply_funding": True,
                    "funding_missing_policy": {"mode": "zero_fill", "expected_interval": "8h", "max_gap_multiplier": 1.25},
                },
            }
        )
        pipeline.state["futures_context"] = {}

        with self.assertRaisesRegex(RuntimeError, "Funding coverage gate failed: funding_frame_missing"):
            _resolve_backtest_funding_rates(pipeline, index)


if __name__ == "__main__":
    unittest.main()