import unittest

import numpy as np
import pandas as pd

from core.features import build_feature_set


def _make_raw(n=64):
    index = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    close = np.linspace(100.0, 110.0, n)
    frame = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": np.full(n, 1_000.0),
            "quote_volume": close * 1_000.0,
            "trades": np.full(n, 100),
        },
        index=index,
    )
    return frame


class FeatureAvailabilityContractTest(unittest.TestCase):
    def test_built_feature_set_emits_availability_metadata_for_each_feature(self):
        feature_set = build_feature_set(_make_raw(), lags=[1, 3], frac_diff_d=0.4, rolling_window=10)

        self.assertTrue(feature_set.feature_availability)
        sample_column = "return_1"
        lagged_column = "return_1_lag1"

        for column in [sample_column, lagged_column, "close_fracdiff"]:
            availability = feature_set.feature_availability[column]
            self.assertEqual(sorted(availability.keys()), ["available_timestamp", "event_timestamp", "join_mode", "source"])
            self.assertTrue(availability["source"])

        self.assertEqual(feature_set.feature_availability[sample_column]["event_timestamp"], "index")
        self.assertEqual(feature_set.feature_availability[sample_column]["available_timestamp"], "index")
        self.assertEqual(feature_set.feature_availability[sample_column]["join_mode"], "same_index")
        self.assertEqual(feature_set.feature_availability[lagged_column]["join_mode"], "lagged")
        self.assertEqual(
            feature_set.feature_lineage[lagged_column]["availability"]["join_mode"],
            "lagged",
        )


if __name__ == "__main__":
    unittest.main()