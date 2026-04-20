import unittest

import pandas as pd

from core import load_custom_dataset, validate_market_frame_contract


class DataContractsTest(unittest.TestCase):
    def test_market_contract_allows_additive_columns(self):
        index = pd.date_range("2026-04-01", periods=2, freq="1h", tz="UTC")
        frame = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "volume": [10.0, 11.0],
                "quote_volume": [1005.0, 1116.5],
                "trades": [100, 101],
                "taker_buy_base_vol": [4.0, 4.5],
                "taker_buy_quote_vol": [400.0, 450.0],
                "vendor_debug_flag": [1, 0],
            },
            index=index,
        )

        validated, manifest = validate_market_frame_contract(frame, market="spot", dataset_name="spot_bars")

        self.assertIn("vendor_debug_flag", validated.columns)
        self.assertTrue(manifest["contract"]["allow_extra_columns"])

    def test_custom_contract_rejects_naive_timestamps(self):
        frame = pd.DataFrame(
            {
                "timestamp": ["2026-04-01 00:00:00"],
                "available_at": ["2026-04-01 01:00:00"],
                "sentiment": [0.25],
            }
        )

        with self.assertRaisesRegex(ValueError, "timezone-aware UTC|naive"):
            load_custom_dataset(
                frame=frame,
                name="sentiment_feed",
                timestamp_column="timestamp",
                availability_column="available_at",
                value_columns=["sentiment"],
            )


if __name__ == "__main__":
    unittest.main()