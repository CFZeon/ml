import unittest
from pathlib import Path

from core import ResearchPipeline
from core.pipeline import _resolve_lookahead_guard_config
from example_trade_ready_automl import build_trade_ready_example_config


class GlobalLookaheadGuardDefaultTest(unittest.TestCase):
    def test_lookahead_guard_is_enabled_by_default_for_research_pipelines(self):
        pipeline = ResearchPipeline(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "features": {},
                "labels": {"kind": "fixed_horizon", "horizon": 2},
                "model": {"type": "logistic", "cv_method": "walk_forward"},
                "backtest": {"evaluation_mode": "research_only"},
            }
        )

        guard = _resolve_lookahead_guard_config(pipeline)

        self.assertTrue(guard["enabled"])
        self.assertEqual(guard["mode"], "advisory")
        self.assertFalse(guard["trade_ready_mode"])
        self.assertFalse(guard["automl_enabled"])

    def test_trade_ready_example_keeps_lookahead_guard_enabled(self):
        config = build_trade_ready_example_config(
            automl_storage=Path(".cache") / "test_trade_ready_lookahead_guard.db"
        )

        pipeline = ResearchPipeline(config)
        guard = _resolve_lookahead_guard_config(pipeline)

        self.assertTrue(guard["enabled"])
        self.assertEqual(guard["mode"], "blocking")
        self.assertTrue(guard["trade_ready_mode"])


if __name__ == "__main__":
    unittest.main()