import json
import unittest

import pandas as pd

from core import build_feature_strategy_adapter


class FeatureStrategyAdapterTest(unittest.TestCase):
    @staticmethod
    def _make_frames():
        index = pd.date_range("2026-05-01", periods=8, freq="1h", tz="UTC")
        X = pd.DataFrame(
            {
                "ret_signal": [0.2, 0.1, -0.3, 0.4, -0.1, 0.2, -0.2, 0.3],
                "momentum_signal": [1.0, 0.8, 0.7, 1.1, 0.9, 1.2, 0.6, 1.3],
                "level_feature": [10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.4],
            },
            index=index,
        )
        regime = pd.DataFrame(
            {
                "regime": [0, 0, 1, 1, 0, 1, 0, 1],
                "ewm_vol_20": [0.5, 0.55, 0.8, 0.82, 0.58, 0.78, 0.57, 0.8],
            },
            index=index,
        )
        config = {
            "regime_interactions": True,
            "feature_adaptation": {
                "interaction_budget": {
                    "enabled": True,
                    "max_features": 1,
                    "max_regimes": 1,
                    "max_dummy_cardinality": 4,
                }
            },
        }
        return X, regime, config

    def test_feature_strategy_adapter_is_prefix_invariant_and_budget_bounded(self):
        X, regime, config = self._make_frames()
        adapter = build_feature_strategy_adapter(config, regime_column="regime")
        adapter.fit(X, regime)

        transformed_full, policy_full = adapter.transform(X, regime)
        transformed_prefix, policy_prefix = adapter.transform(X.iloc[:4], regime.iloc[:4])
        manifest = adapter.manifest()

        pd.testing.assert_frame_equal(transformed_prefix, transformed_full.iloc[:4])
        self.assertEqual(list(transformed_full.columns), manifest["feature_columns"])
        self.assertLessEqual(manifest["interaction_column_count"], 1)
        self.assertLessEqual(len(policy_full.generated_columns), manifest["generated_column_count"])
        self.assertIn("vol_norm__ret_signal", transformed_full.columns)
        json.dumps(policy_full.to_dict())
        json.dumps(policy_prefix.to_dict())
        json.dumps(manifest)


if __name__ == "__main__":
    unittest.main()