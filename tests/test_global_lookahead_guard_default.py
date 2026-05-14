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

    def test_regime_aware_research_pipeline_defaults_to_full_causal_surface(self):
        pipeline = ResearchPipeline(
            {
                "features": {},
                "regime": {"method": "hmm", "n_regimes": 2},
                "feature_adaptation": {
                    "selection": {"mode": "per_regime_mask", "fallback": "global"},
                },
                "model": {
                    "type": "logistic",
                    "regime_aware": {"enabled": True, "strategy": "specialist"},
                },
                "router": {"enabled": True},
                "backtest": {"evaluation_mode": "research_only"},
            }
        )

        guard = _resolve_lookahead_guard_config(pipeline)

        self.assertTrue(guard["enabled"])
        self.assertEqual(guard["mode"], "advisory")
        self.assertEqual(guard["audit_scope"], "full_causal_surface")
        self.assertEqual(guard["required_evidence_class"], "causal_research")
        self.assertIn("admissible_regimes", guard["artifact_names"])
        self.assertIn("router_decisions", guard["artifact_names"])


if __name__ == "__main__":
    unittest.main()