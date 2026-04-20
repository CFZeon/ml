import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import core.context as context_module
import core.data as data_module
from core.models import build_model, load_model, save_model


class SafeArtifactStorageTest(unittest.TestCase):
    def test_tampered_artifacts_fail_hash_verification(self):
        X = pd.DataFrame({"f1": [0.0, 1.0, 0.0, 1.0], "f2": [1.0, 1.0, 0.0, 0.0]})
        y = pd.Series([0, 1, 0, 1])
        model = build_model("logistic", {"c": 1.0})
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "model_artifact"
            save_model(model, artifact_path, metadata={"feature_columns": list(X.columns)})

            model_path = artifact_path.with_suffix(".skops")
            with open(model_path, "ab") as handle:
                handle.write(b"tampered")

            with self.assertRaisesRegex(ValueError, "hash verification failed"):
                load_model(artifact_path, expected_feature_columns=list(X.columns))

    def test_feature_schema_mismatches_fail_closed(self):
        X = pd.DataFrame({"f1": [0.0, 1.0, 0.0, 1.0], "f2": [1.0, 1.0, 0.0, 0.0]})
        y = pd.Series([0, 1, 0, 1])
        model = build_model("logistic", {"c": 1.0})
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "model_artifact"
            save_model(model, artifact_path, metadata={"feature_columns": list(X.columns)})

            with self.assertRaisesRegex(ValueError, "Feature schema mismatch|Feature order mismatch"):
                load_model(artifact_path, expected_feature_columns=["f2", "f1"])

    def test_persistent_market_and_context_caches_do_not_write_pickle_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir)
            period = data_module.CachePeriod(
                kind="monthly",
                key="2024-01",
                start=pd.Timestamp("2024-01-01", tz="UTC"),
                end=pd.Timestamp("2024-02-01", tz="UTC"),
            )
            market_frame = pd.DataFrame(
                {
                    "open": [100.0],
                    "high": [101.0],
                    "low": [99.0],
                    "close": [100.5],
                    "volume": [10.0],
                    "quote_volume": [1005.0],
                    "trades": [100],
                    "taker_buy_base_vol": [4.0],
                    "taker_buy_quote_vol": [400.0],
                },
                index=pd.DatetimeIndex([pd.Timestamp("2024-01-01", tz="UTC")]),
            )
            market_cache_path = data_module._cache_path(cache_root, "BTCUSDT", "1h", period, market="spot")
            data_module._write_cache(market_cache_path, market_frame)

            object_cache_path = data_module._symbol_filters_cache_path(cache_root, "spot", "BTCUSDT")
            data_module._write_object_cache(object_cache_path, {"symbol": "BTCUSDT", "tick_size": 0.01})

            context_cache_path = context_module._cache_path(cache_root, "funding", {"symbol": "BTCUSDT", "interval": "1h"})
            context_module._write_cache(
                context_cache_path,
                pd.DataFrame(
                    {"funding_rate": [0.0001]},
                    index=pd.DatetimeIndex([pd.Timestamp("2024-01-01", tz="UTC")]),
                ),
            )

            self.assertTrue(market_cache_path.exists())
            self.assertTrue(object_cache_path.exists())
            self.assertTrue(context_cache_path.exists())
            self.assertEqual(market_cache_path.suffix, ".parquet")
            self.assertEqual(object_cache_path.suffix, ".json")
            self.assertEqual(context_cache_path.suffix, ".parquet")
            self.assertEqual(list(cache_root.rglob("*.pkl")), [])
            self.assertEqual(list(cache_root.rglob("*.pickle")), [])


if __name__ == "__main__":
    unittest.main()