import unittest

import pandas as pd

from core import build_spot_reference_validation


def _make_market_frame(index, close_values):
    close = pd.Series(close_values, index=index, dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.25
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.25
    volume = pd.Series(1_000.0, index=index, dtype=float)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


class CrossVenueReferenceValidationTest(unittest.TestCase):
    def test_partial_spot_coverage_is_advisory(self):
        index = pd.date_range("2026-05-01", periods=6, freq="1h", tz="UTC")
        base = _make_market_frame(index, [100.0, 101.0, 102.0, 101.5, 102.5, 103.0])
        coinbase = _make_market_frame(index, [100.2, 101.1, 102.1, 101.7, 102.6, 103.2])

        result = build_spot_reference_validation(
            base,
            symbol="BTCUSDT",
            interval="1h",
            config={
                "fetch_live": False,
                "spot": {
                    "venues": ["coinbase", "kraken"],
                    "frames": {"coinbase": coinbase},
                    "partial_coverage_mode": "advisory",
                },
            },
        )

        self.assertTrue(result["report"]["promotion_pass"])
        self.assertEqual(result["report"]["gate_mode"], "advisory")
        self.assertIn("partial_reference_coverage", result["report"]["reasons"])
        self.assertIn("reference_price", result["overlay"].columns)

    def test_full_spot_cohort_blocks_on_severe_divergence(self):
        index = pd.date_range("2026-05-01", periods=6, freq="1h", tz="UTC")
        base = _make_market_frame(index, [100.0, 100.5, 101.0, 101.5, 102.0, 102.5])
        coinbase = _make_market_frame(index, [118.0, 118.5, 119.0, 119.5, 120.0, 120.5])
        kraken = _make_market_frame(index, [117.5, 118.2, 118.8, 119.2, 119.8, 120.1])

        result = build_spot_reference_validation(
            base,
            symbol="BTCUSDT",
            interval="1h",
            config={
                "fetch_live": False,
                "spot": {
                    "venues": ["coinbase", "kraken"],
                    "frames": {"coinbase": coinbase, "kraken": kraken},
                    "partial_coverage_mode": "advisory",
                    "min_coverage_ratio": 1.0,
                    "max_price_divergence_bps": 250.0,
                },
            },
        )

        self.assertFalse(result["report"]["promotion_pass"])
        self.assertEqual(result["report"]["gate_mode"], "blocking")
        self.assertIn("spot_reference_divergence", result["report"]["reasons"])
        self.assertTrue(result["report"]["full_cohort_available"])


if __name__ == "__main__":
    unittest.main()