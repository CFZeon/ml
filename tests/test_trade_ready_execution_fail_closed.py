import unittest

from core import ResearchPipeline
from core.pipeline import _resolve_backtest_execution_policy


class TradeReadyExecutionFailClosedTest(unittest.TestCase):
    def test_trade_ready_profile_rejects_surrogate_execution_without_override(self):
        pipeline = ResearchPipeline(
            {
                "backtest": {
                    "evaluation_mode": "trade_ready",
                    "execution_policy": {"adapter": "bar_surrogate"},
                }
            }
        )

        with self.assertRaisesRegex(RuntimeError, "Trade-ready evaluation requires a Nautilus execution adapter"):
            _resolve_backtest_execution_policy(pipeline)

    def test_trade_ready_profile_rejects_explicit_research_override(self):
        pipeline = ResearchPipeline(
            {
                "backtest": {
                    "evaluation_mode": "trade_ready",
                    "execution_profile": "research_surrogate",
                    "research_only_override": True,
                    "execution_policy": {"adapter": "bar_surrogate"},
                }
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "backtest.research_only_override=true is only valid when backtest.evaluation_mode='research_only'",
        ):
            _resolve_backtest_execution_policy(pipeline)


if __name__ == "__main__":
    unittest.main()