import unittest

import numpy as np
import pandas as pd

from core.automl import _build_explicit_temporal_split


def _build_index(rows):
    return pd.date_range("2026-01-01", periods=rows, freq="1h", tz="UTC")


class TemporalSplitRowAccountingTest(unittest.TestCase):
    def test_explicit_temporal_split_accounts_for_all_rows(self):
        index = _build_index(32)
        split = _build_explicit_temporal_split(
            index,
            train_end_timestamp=index[9],
            test_start_timestamp=index[13],
        )

        self.assertEqual(len(split["train_index"]), 10)
        self.assertEqual(len(split["gap_index"]), 3)
        self.assertEqual(len(split["test_index"]), 19)
        self.assertEqual(len(split["train_index"]) + len(split["gap_index"]) + len(split["test_index"]), len(index))
        self.assertEqual(split["gap_bars"], 3)
        self.assertEqual(split["timestamp_bounds"]["train_end"], index[9].isoformat())
        self.assertEqual(split["timestamp_bounds"]["test_start"], index[13].isoformat())


if __name__ == "__main__":
    unittest.main()