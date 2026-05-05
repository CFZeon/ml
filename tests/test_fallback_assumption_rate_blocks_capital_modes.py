import unittest

from core import build_monitoring_report


class FallbackAssumptionRateBlocksCapitalModesTest(unittest.TestCase):
    def test_trade_ready_blocks_surrogate_execution_and_funding_fallback(self):
        report = build_monitoring_report(
            backtest_reports=[
                {
                    "execution_evidence": {
                        "class": "research_surrogate",
                        "execution_mode": "conservative_bar_surrogate",
                        "promotion_execution_ready": False,
                    },
                    "funding_coverage_report": {
                        "coverage_status": "fallback",
                        "promotion_pass": True,
                    },
                }
            ],
            policy={
                "policy_profile": "trade_ready",
                "required_components": ["fallback_assumptions"],
            },
        )

        fallback = report["components"]["fallback_assumptions"]
        gate = report["monitoring_gate_report"]
        self.assertFalse(report["healthy"])
        self.assertFalse(gate["promotion_pass"])
        self.assertEqual(fallback["source_count"], 2)
        self.assertEqual(fallback["fallback_source_count"], 2)
        self.assertEqual(float(fallback["fallback_assumption_rate"]), 1.0)
        self.assertIn("fallback_assumption_rate_breach", gate["blocking_reasons"])
        self.assertIn("fallback_assumptions", report["reasons"])

    def test_research_profile_allows_fallback_assumptions(self):
        report = build_monitoring_report(
            backtest_reports=[
                {
                    "execution_evidence": {
                        "class": "research_surrogate",
                        "execution_mode": "conservative_bar_surrogate",
                        "promotion_execution_ready": False,
                    },
                    "funding_coverage_report": {
                        "coverage_status": "fallback",
                        "promotion_pass": True,
                    },
                }
            ],
            policy={
                "policy_profile": "research",
                "required_components": ["fallback_assumptions"],
            },
        )

        self.assertTrue(report["healthy"])
        self.assertTrue(report["promotion_pass"])

    def test_monitoring_counts_engine_and_pricing_fallback_metadata(self):
        report = build_monitoring_report(
            backtest_reports=[
                {
                    "engine": "pandas",
                    "requested_engine": "vectorbt",
                    "engine_fallback_used": True,
                    "engine_fallback_reason": "vectorbt_unavailable",
                    "execution_price_source": "close_fallback",
                    "same_bar_execution_fallback": True,
                    "backtest_warnings": ["same_bar_execution_fallback", "engine_fallback_to_pandas"],
                }
            ],
            policy={
                "policy_profile": "trade_ready",
                "required_components": ["fallback_assumptions"],
            },
        )

        fallback = report["components"]["fallback_assumptions"]
        self.assertFalse(report["healthy"])
        self.assertEqual(fallback["source_count"], 2)
        self.assertEqual(fallback["fallback_source_count"], 2)
        self.assertEqual(float(fallback["fallback_assumption_rate"]), 1.0)
        self.assertIn("fallback_assumption_rate_breach", report["monitoring_gate_report"]["blocking_reasons"])


if __name__ == "__main__":
    unittest.main()