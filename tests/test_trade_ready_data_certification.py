import unittest

from core import ResearchPipeline


class TradeReadyDataCertificationTests(unittest.TestCase):
    def test_trade_ready_certification_requires_reference_validation_when_configured(self):
        pipeline = ResearchPipeline(
            {
                "backtest": {"evaluation_mode": "trade_ready"},
                "data": {"futures_context": {"enabled": False}},
                "reference_data": {"enabled": True},
                "data_certification": {"enabled": True, "require_reference_validation": True},
            }
        )
        pipeline.state["data_integrity_report"] = {
            "status": "complete",
            "gap_policy": "fail",
            "duplicate_policy": "fail",
            "missing_rows": 0,
            "duplicate_report": {"conflicting_duplicate_timestamps": 0},
        }
        pipeline.state["data_quality_report"] = {
            "status": "pass",
            "blocking": False,
            "summary": {"quarantined_rows": 0},
        }

        report = pipeline.inspect_data_certification()

        self.assertTrue(report["enabled"])
        self.assertFalse(report["promotion_pass"])
        self.assertIn("reference_validation_unavailable", report["reasons"])
        self.assertIn("reference_integrity", report["summary"]["failed_components"])

    def test_trade_ready_certification_binds_context_ttl_breaches(self):
        pipeline = ResearchPipeline(
            {
                "backtest": {"evaluation_mode": "trade_ready"},
                "data": {
                    "futures_context": {"enabled": False},
                    "cross_asset_context": {"symbols": ["ETHUSDT"]},
                },
                "reference_data": {"enabled": True},
                "data_certification": {"enabled": True, "require_reference_validation": True},
            }
        )
        pipeline.state["data_integrity_report"] = {
            "status": "complete",
            "gap_policy": "fail",
            "duplicate_policy": "fail",
            "missing_rows": 0,
            "duplicate_report": {"conflicting_duplicate_timestamps": 0},
        }
        pipeline.state["data_quality_report"] = {
            "status": "pass",
            "blocking": False,
            "summary": {"quarantined_rows": 0},
        }
        pipeline.state["context_ttl_report"] = {
            "cross_asset_context": {
                "promotion_pass": False,
                "coverage_reason": "stale_context",
            }
        }
        pipeline.state["reference_integrity_report"] = {
            "promotion_pass": True,
            "gate_mode": "blocking",
            "available_venue_count": 2,
            "reasons": [],
            "warnings": [],
        }

        report = pipeline.inspect_data_certification()

        self.assertFalse(report["promotion_pass"])
        self.assertIn("context_ttl_breached", report["reasons"])
        self.assertIn("context_ttl", report["summary"]["failed_components"])
        self.assertTrue(report["components"]["reference_integrity"]["promotion_pass"])


if __name__ == "__main__":
    unittest.main()