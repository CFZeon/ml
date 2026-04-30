import unittest

import pandas as pd

from core.backtest import run_backtest
from core.models import build_execution_outcome_frame


class FundingZeroFillForbiddenInCapitalModesTest(unittest.TestCase):
    def test_local_certification_backtest_rejects_missing_funding_events(self):
        index = pd.date_range("2026-03-12", periods=6, freq="1h", tz="UTC")
        close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], index=index)
        signals = pd.Series([0.0, 0.5, 0.5, 0.0, -0.5, 0.0], index=index)
        funding_rates = pd.Series([0.0, 0.0, float("nan"), 0.0, 0.0, 0.0], index=index)

        with self.assertRaisesRegex(RuntimeError, "Funding coverage breach: missing_funding_events"):
            run_backtest(
                close=close,
                signals=signals,
                equity=10_000.0,
                fee_rate=0.001,
                slippage_rate=0.0,
                execution_prices=close,
                signal_delay_bars=0,
                engine="pandas",
                market="um_futures",
                allow_short=True,
                leverage=1.0,
                funding_rates=funding_rates,
                evaluation_mode="local_certification",
            )

    def test_research_backtest_marks_missing_funding_as_incomplete(self):
        index = pd.date_range("2026-03-12", periods=6, freq="1h", tz="UTC")
        close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], index=index)
        signals = pd.Series([0.0, 0.5, 0.5, 0.0, -0.5, 0.0], index=index)
        funding_rates = pd.Series([0.0, 0.0, float("nan"), 0.0, 0.0, 0.0], index=index)

        result = run_backtest(
            close=close,
            signals=signals,
            equity=10_000.0,
            fee_rate=0.001,
            slippage_rate=0.0,
            execution_prices=close,
            signal_delay_bars=0,
            engine="pandas",
            market="um_futures",
            allow_short=True,
            leverage=1.0,
            funding_rates=funding_rates,
            evaluation_mode="research_only",
        )

        self.assertEqual(result["funding_coverage_status"], "incomplete")
        self.assertEqual(result["funding_coverage_report"]["missing_event_count"], 1)
        self.assertFalse(result["funding_coverage_report"]["promotion_pass"])
        self.assertNotIn("fallback_assumption", result["funding_coverage_report"])

    def test_explicit_debug_zero_fill_remains_visibly_labeled(self):
        index = pd.date_range("2026-03-12", periods=6, freq="1h", tz="UTC")
        close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], index=index)
        signals = pd.Series([0.0, 0.5, 0.5, 0.0, -0.5, 0.0], index=index)
        funding_rates = pd.Series([0.0, 0.0, float("nan"), 0.0, 0.0, 0.0], index=index)

        result = run_backtest(
            close=close,
            signals=signals,
            equity=10_000.0,
            fee_rate=0.001,
            slippage_rate=0.0,
            execution_prices=close,
            signal_delay_bars=0,
            engine="pandas",
            market="um_futures",
            allow_short=True,
            leverage=1.0,
            funding_rates=funding_rates,
            funding_missing_policy={"mode": "zero_fill_debug"},
            evaluation_mode="research_only",
        )

        self.assertEqual(result["funding_coverage_status"], "debug_fallback")
        self.assertEqual(
            result["funding_coverage_report"].get("fallback_assumption"),
            "zero_fill_missing_funding_events",
        )


    def test_local_certification_execution_outcomes_reject_missing_funding_events(self):
        index = pd.date_range("2026-03-12", periods=6, freq="1h", tz="UTC")
        valuation = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], index=index)
        predictions = pd.Series([1, -1], index=index[:2])
        funding_rates = pd.Series([0.0, 0.0, float("nan"), 0.0, 0.0, 0.0], index=index)

        with self.assertRaisesRegex(RuntimeError, "Execution outcome funding coverage breach: missing_funding_events"):
            build_execution_outcome_frame(
                predictions,
                valuation_prices=valuation,
                execution_prices=valuation,
                holding_bars=2,
                signal_delay_bars=1,
                funding_rates=funding_rates,
                evaluation_mode="local_certification",
            )


if __name__ == "__main__":
    unittest.main()