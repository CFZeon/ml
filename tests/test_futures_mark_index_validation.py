import unittest

import pandas as pd

from core import build_futures_reference_validation


def _make_market_frame(index, close_values):
    close = pd.Series(close_values, index=index, dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.1
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.1
    volume = pd.Series(1_500.0, index=index, dtype=float)
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


class FuturesMarkIndexValidationTest(unittest.TestCase):
    def test_self_consistency_passes_when_mark_and_basis_align(self):
        index = pd.date_range("2026-05-01", periods=6, freq="1h", tz="UTC")
        base = _make_market_frame(index, [100.0, 100.3, 100.6, 100.9, 101.1, 101.4])
        futures_context = {
            "mark_price": pd.DataFrame(
                {
                    "mark_open": [100.0, 100.25, 100.55, 100.85, 101.05, 101.3],
                    "mark_high": [100.1, 100.4, 100.7, 101.0, 101.2, 101.45],
                    "mark_low": [99.9, 100.15, 100.45, 100.75, 100.95, 101.2],
                    "mark_close": [100.02, 100.34, 100.63, 100.95, 101.13, 101.42],
                },
                index=index,
            ),
            "basis": pd.DataFrame(
                {
                    "basisRate": [0.0002, 0.00025, 0.0003, 0.00028, 0.00027, 0.00026],
                    "basis": [0.02, 0.03, 0.03, 0.03, 0.03, 0.03],
                    "futuresPrice": [100.02, 100.34, 100.63, 100.95, 101.13, 101.42],
                    "indexPrice": [100.0, 100.31, 100.6, 100.92, 101.1, 101.39],
                },
                index=index,
            ),
            "funding": pd.DataFrame(
                {
                    "funding_rate": [0.0001, 0.0001, 0.0002],
                    "funding_mark_price": [100.0, 100.6, 101.1],
                },
                index=index[::2],
            ),
        }

        result = build_futures_reference_validation(
            base,
            futures_context=futures_context,
            symbol="BTCUSDT",
            interval="1h",
            config={"fetch_live": False},
        )

        self.assertTrue(result["report"]["promotion_pass"])
        self.assertEqual(result["report"]["gate_mode"], "blocking")
        self.assertIn("composite_basis", result["overlay"].columns)

    def test_self_consistency_fails_on_large_mark_index_and_basis_errors(self):
        index = pd.date_range("2026-05-01", periods=6, freq="1h", tz="UTC")
        base = _make_market_frame(index, [100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        futures_context = {
            "mark_price": pd.DataFrame(
                {
                    "mark_open": [120.0] * 6,
                    "mark_high": [121.0] * 6,
                    "mark_low": [119.0] * 6,
                    "mark_close": [120.0] * 6,
                },
                index=index,
            ),
            "basis": pd.DataFrame(
                {
                    "basisRate": [0.25] * 6,
                    "basis": [30.0] * 6,
                    "futuresPrice": [120.0] * 6,
                    "indexPrice": [80.0] * 6,
                },
                index=index,
            ),
        }

        result = build_futures_reference_validation(
            base,
            futures_context=futures_context,
            symbol="BTCUSDT",
            interval="1h",
            config={
                "fetch_live": False,
                "max_trade_mark_gap_bps": 100.0,
                "max_mark_index_gap_bps": 100.0,
                "max_basis_error_bps": 5.0,
            },
        )

        self.assertFalse(result["report"]["promotion_pass"])
        self.assertIn("futures_self_consistency_failed", result["report"]["reasons"])
        self.assertFalse(result["report"]["self_consistency"]["trade_mark_gap_bps"]["passed"])


if __name__ == "__main__":
    unittest.main()