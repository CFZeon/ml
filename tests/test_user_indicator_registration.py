import unittest

import numpy as np
import pandas as pd

from core import build_indicator, run_indicators


def _make_raw(index):
    close = 100.0 + np.linspace(0.0, 4.0, len(index))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    volume = np.full(len(index), 1_000.0)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": close * volume,
            "trades": 100,
        },
        index=index,
    )


class UserIndicatorRegistrationTest(unittest.TestCase):
    def test_function_indicator_modules_are_discovered_automatically(self):
        frame = _make_raw(pd.date_range("2026-01-01", periods=48, freq="1h", tz="UTC"))

        indicator = build_indicator({"kind": "returns", "params": {"periods": [1, 3]}})
        indicator_run = run_indicators(frame, [indicator, {"kind": "volatility", "params": {"window": 8}}])

        self.assertIn("returns_1", indicator_run.frame.columns)
        self.assertIn("returns_3", indicator_run.frame.columns)
        self.assertIn("volatility_8", indicator_run.frame.columns)
        self.assertIn("volatility_8_pct_rank", indicator_run.frame.columns)

    def test_function_indicator_outputs_are_renamed_by_instance_name(self):
        frame = _make_raw(pd.date_range("2026-02-01", periods=24, freq="1h", tz="UTC"))

        indicator_run = run_indicators(frame, [{"kind": "returns", "name": "fast_returns", "params": {"periods": [1]}}])

        self.assertIn("fast_returns_1", indicator_run.frame.columns)
        self.assertNotIn("returns_1", indicator_run.frame.columns)
        self.assertTrue(indicator_run.results[0].metadata["lookahead_safe"])


if __name__ == "__main__":
    unittest.main()