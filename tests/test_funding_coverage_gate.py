import unittest

import pandas as pd

from core import ResearchPipeline
from core.evaluation_modes import resolve_evaluation_mode
from core.pipeline import (
    _resolve_backtest_funding_missing_policy,
    _resolve_backtest_funding_rates,
    _resolve_pipeline_data_fetch_config,
    _resolve_pipeline_data_quality_config,
)


class FundingCoverageGateTest(unittest.TestCase):
    def test_research_demo_alias_resolves_to_research_only_profile(self):
        mode = resolve_evaluation_mode({"evaluation_mode": "research_demo"})

        self.assertEqual(mode.requested_mode, "research_only")
        self.assertEqual(mode.effective_mode, "research_demo")
        self.assertTrue(mode.is_research_only)
        self.assertFalse(mode.is_capital_facing)

    def test_research_defaults_to_preserve_missing_funding_policy(self):
        policy = _resolve_backtest_funding_missing_policy({"evaluation_mode": "research_only"})

        self.assertEqual(policy["mode"], "preserve_missing")
        self.assertFalse(policy["allow_missing_events"])

    def test_trade_ready_defaults_to_strict_funding_policy(self):
        policy = _resolve_backtest_funding_missing_policy({"evaluation_mode": "trade_ready"})

        self.assertEqual(policy["mode"], "strict")

    def test_trade_ready_research_override_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "backtest.research_only_override=true is only valid when backtest.evaluation_mode='research_only'",
        ):
            _resolve_backtest_funding_missing_policy(
                {
                    "evaluation_mode": "trade_ready",
                    "research_only_override": True,
                    "funding_missing_policy": {
                        "mode": "zero_fill",
                        "expected_interval": "8h",
                        "max_gap_multiplier": 1.5,
                    },
                }
            )

    def test_trade_ready_runtime_defaults_fail_closed_data_integrity_and_quarantine(self):
        pipeline = ResearchPipeline(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "backtest": {"evaluation_mode": "trade_ready"},
            }
        )

        data_config = _resolve_pipeline_data_fetch_config(pipeline)
        data_quality_config = _resolve_pipeline_data_quality_config(pipeline)

        self.assertEqual(data_config["gap_policy"], "fail")
        self.assertEqual(data_config["duplicate_policy"], "fail")
        self.assertTrue(data_quality_config["block_on_quarantine"])
        self.assertTrue(data_quality_config["exclude_flagged_quarantine_rows_from_modeling"])

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

    def test_research_preserve_missing_marks_funding_frame_as_incomplete(self):
        index = pd.date_range("2026-02-01", periods=24, freq="1h", tz="UTC")
        pipeline = ResearchPipeline(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h", "market": "um_futures"},
                "backtest": {
                    "evaluation_mode": "research_only",
                    "apply_funding": True,
                },
            }
        )
        pipeline.state["futures_context"] = {}

        funding = _resolve_backtest_funding_rates(pipeline, index)

        self.assertIsNone(funding)
        report = pipeline.state["context_ttl_report"]["backtest_funding"]
        self.assertEqual(report["coverage_status"], "incomplete")
        self.assertFalse(report["promotion_pass"])
        self.assertEqual(report["coverage_reason"], "funding_frame_missing")


if __name__ == "__main__":
    unittest.main()