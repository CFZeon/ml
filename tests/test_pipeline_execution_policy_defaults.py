import unittest

from core.pipeline import ResearchPipeline
import core.pipeline as pipeline_module


class PipelineExecutionPolicyDefaultsTest(unittest.TestCase):
    def test_pipeline_inherits_central_execution_policy_defaults_when_unconfigured(self):
        pipeline = ResearchPipeline(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "backtest": {},
            }
        )

        policy = pipeline_module._resolve_backtest_execution_policy(pipeline)

        self.assertAlmostEqual(float(policy["participation_cap"]), 0.10, places=12)
        self.assertAlmostEqual(float(policy["min_fill_ratio"]), 0.25, places=12)

    def test_pipeline_still_honors_explicit_backtest_execution_overrides(self):
        pipeline = ResearchPipeline(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "backtest": {
                    "participation_cap": 0.5,
                    "min_fill_ratio": 0.6,
                },
            }
        )

        policy = pipeline_module._resolve_backtest_execution_policy(pipeline)

        self.assertAlmostEqual(float(policy["participation_cap"]), 0.5, places=12)
        self.assertAlmostEqual(float(policy["min_fill_ratio"]), 0.6, places=12)


if __name__ == "__main__":
    unittest.main()