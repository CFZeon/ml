import unittest

import pandas as pd

from core import run_backtest


class SurrogateExecutionBlocksCertificationTest(unittest.TestCase):
    def test_local_certification_surrogate_backtest_is_not_certification_evidence(self):
        index = pd.date_range("2024-01-01", periods=4, freq="h")
        close = pd.Series([100.0, 101.0, 102.0, 103.0], index=index)
        signals = pd.Series([0.0, 0.5, 0.5, 0.0], index=index)

        result = run_backtest(
            close,
            signals,
            execution_prices=close,
            engine="pandas",
            signal_delay_bars=0,
            volume=pd.Series(10_000.0, index=index),
            evaluation_mode="local_certification",
        )

        self.assertTrue(result["research_only"])
        self.assertIn("execution_backend_not_event_driven", result["trade_ready_blockers"])
        self.assertEqual(result["execution_evidence"]["class"], "research_surrogate")
        self.assertFalse(result["execution_evidence"]["promotion_execution_ready"])


if __name__ == "__main__":
    unittest.main()