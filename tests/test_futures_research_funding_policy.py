import unittest

from example_utils import build_research_demo_runtime_overrides, build_futures_research_config


class FuturesResearchFundingPolicyTest(unittest.TestCase):
    def test_futures_research_builder_keeps_nonzero_fill_default(self):
        config = build_futures_research_config(
            symbol="BTCUSDT",
            interval="1h",
            start="2024-01-01",
            end="2024-02-01",
            indicators=[],
        )

        self.assertEqual(config["backtest"]["funding_missing_policy"]["mode"], "strict")

    def test_research_demo_futures_override_makes_zero_fill_explicit_debug_only(self):
        overrides = build_research_demo_runtime_overrides(market="um_futures")

        self.assertEqual(overrides["backtest"]["funding_missing_policy"]["mode"], "zero_fill_debug")


if __name__ == "__main__":
    unittest.main()