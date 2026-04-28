import unittest

import pandas as pd

from core import ResearchPipeline
from core.pipeline import _resolve_backtest_funding_missing_policy, _resolve_backtest_funding_rates


class LocalCertificationFundingStrictTest(unittest.TestCase):
    def test_local_certification_defaults_to_strict_funding_policy(self):
        policy = _resolve_backtest_funding_missing_policy({"evaluation_mode": "local_certification"})

        self.assertEqual(policy["mode"], "strict")
        self.assertFalse(policy["allow_missing_events"])

    def test_local_certification_blocks_missing_funding_frame(self):
        index = pd.date_range("2026-02-01", periods=24, freq="1h", tz="UTC")
        pipeline = ResearchPipeline(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h", "market": "um_futures"},
                "backtest": {
                    "evaluation_mode": "local_certification",
                    "apply_funding": True,
                },
            }
        )
        pipeline.state["futures_context"] = {}

        with self.assertRaisesRegex(RuntimeError, "Funding coverage gate failed: funding_frame_missing"):
            _resolve_backtest_funding_rates(pipeline, index)

        report = pipeline.state["context_ttl_report"]["backtest_funding"]
        self.assertEqual(report["coverage_status"], "strict")
        self.assertFalse(report["promotion_pass"])


if __name__ == "__main__":
    unittest.main()