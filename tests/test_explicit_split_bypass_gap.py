import unittest

import numpy as np
import pandas as pd

from core.automl import _build_explicit_temporal_split
from core.pipeline import _iter_validation_splits


class _PipelineStub:
    def __init__(self, model_config):
        self._model_config = dict(model_config)

    def section(self, key):
        if key == "model":
            return dict(self._model_config)
        return {}


class ExplicitSplitBypassGapTest(unittest.TestCase):
    def test_explicit_splits_do_not_reapply_configured_gap(self):
        index = pd.date_range("2026-01-01", periods=20, freq="1h", tz="UTC")
        X = pd.DataFrame({"feature": np.arange(len(index), dtype=float)}, index=index)
        explicit_split = _build_explicit_temporal_split(
            index,
            train_end_timestamp=index[9],
            test_start_timestamp=index[12],
        )
        pipeline = _PipelineStub(
            {
                "cv_method": "walk_forward",
                "gap": 5,
                "explicit_splits": [explicit_split],
            }
        )

        splits = list(_iter_validation_splits(pipeline, X))

        self.assertEqual(len(splits), 1)
        split = splits[0]
        np.testing.assert_array_equal(split["train_idx"], np.arange(10))
        np.testing.assert_array_equal(split["test_idx"], np.arange(12, 20))
        self.assertEqual(split["metadata"]["gap_rows"], 2)
        self.assertEqual(split["metadata"]["gap_bars"], 2)
        self.assertEqual(split["metadata"]["configured_gap"], 5)
        self.assertEqual(split["metadata"]["split_owner"], "explicit_splits")


if __name__ == "__main__":
    unittest.main()