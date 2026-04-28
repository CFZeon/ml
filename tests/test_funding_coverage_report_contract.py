import unittest

import numpy as np
import pandas as pd

from core import ResearchPipeline
from core.backtest import run_backtest
from core.pipeline import _resolve_backtest_funding_rates


def _make_funding_frame(index):
    funding_index = index[(index.hour % 8) == 0]
    funding_values = np.linspace(0.0001, 0.0001 * len(funding_index), len(funding_index))
    return pd.DataFrame({"funding_rate": funding_values}, index=funding_index)


class FundingCoverageReportContractTest(unittest.TestCase):
    def test_pipeline_funding_report_includes_expected_and_observed_timestamps(self):
        index = pd.date_range("2026-02-01", periods=48, freq="1h", tz="UTC")
        pipeline = ResearchPipeline(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h", "market": "um_futures"},
                "backtest": {
                    "evaluation_mode": "local_certification",
                    "apply_funding": True,
                    "funding_missing_policy": {"mode": "strict", "expected_interval": "8h", "max_gap_multiplier": 1.25},
                },
            }
        )
        pipeline.state["futures_context"] = {"funding": _make_funding_frame(index)}

        funding_rates = _resolve_backtest_funding_rates(pipeline, index)

        self.assertIsNotNone(funding_rates)
        report = pipeline.state["context_ttl_report"]["backtest_funding"]
        self.assertEqual(report["coverage_status"], "strict")
        self.assertFalse(report["policy"]["allow_missing_events"])
        self.assertEqual(report["missing_event_count"], 0)
        self.assertEqual(report["coverage_ratio"], 1.0)
        self.assertEqual(len(report["expected_timestamps"]), 6)
        self.assertEqual(len(report["observed_timestamps"]), 6)
        self.assertTrue(report["promotion_pass"])

    def test_backtest_summary_marks_not_applicable_without_funding(self):
        index = pd.date_range("2026-03-12", periods=6, freq="1h", tz="UTC")
        close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], index=index)
        signals = pd.Series([0.0, 0.5, 0.5, 0.0, -0.5, 0.0], index=index)

        result = run_backtest(
            close=close,
            signals=signals,
            equity=10_000.0,
            fee_rate=0.001,
            slippage_rate=0.0,
            execution_prices=close,
            signal_delay_bars=0,
            engine="pandas",
            market="spot",
            allow_short=False,
        )

        self.assertEqual(result["funding_coverage_status"], "not_applicable")
        self.assertFalse(result["funding_coverage_report"]["enabled"])


if __name__ == "__main__":
    unittest.main()