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
        feature_set = build_feature_set(
            _make_raw(),
            lags=[1, 3],
            frac_diff_d=0.4,
            frac_diff_max_lag=8,
            rolling_window=10,
        )

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

    def test_short_sample_fracdiff_is_rejected_with_retention_report(self):
        feature_set = build_feature_set(_make_raw(), lags=[1, 3], frac_diff_d=0.4, rolling_window=10)

        self.assertNotIn("close_fracdiff", feature_set.frame.columns)
        retention = feature_set.transform_retention_reports["close_fracdiff"]
        self.assertFalse(retention["accepted"])
        self.assertIn("retained_rows_below_minimum", retention["reasons"])
        self.assertEqual(int(retention["retained_rows"]), 0)

    def test_fracdiff_max_lag_can_preserve_short_sample_feature(self):
        feature_set = build_feature_set(
            _make_raw(),
            lags=[1, 3],
            frac_diff_d=0.4,
            frac_diff_max_lag=8,
            rolling_window=10,
        )

        self.assertIn("close_fracdiff", feature_set.frame.columns)
        retention = feature_set.transform_retention_reports["close_fracdiff"]
        self.assertTrue(retention["accepted"])
        self.assertEqual(int(retention["configured_max_lag"]), 8)
        self.assertLessEqual(int(retention["applied_max_lag"]), 8)
        self.assertGreater(int(retention["retained_rows"]), 0)


if __name__ == "__main__":
    unittest.main()