import unittest

import pandas as pd

from core import apply_feature_adaptation_to_splits, build_feature_adapter


class FeatureAdaptationMaskingTest(unittest.TestCase):
    @staticmethod
    def _make_fit_frame():
        index = pd.date_range("2026-05-01", periods=6, freq="1h", tz="UTC")
        frame = pd.DataFrame(
            {
                "trend_feature": [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
                "reversion_feature": [0.0, 0.0, 0.0, 10.0, 11.0, 12.0],
                "shared_feature": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "constant_feature": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                "regime": [0, 0, 0, 1, 1, 1],
                "warm": [1, 1, 1, 1, 1, 1],
                "regime_confidence": [0.95, 0.96, 0.97, 0.98, 0.99, 0.99],
            },
            index=index,
        )
        regime = frame.loc[:, ["regime", "warm", "regime_confidence"]]
        return frame, regime

    @staticmethod
    def _make_mask_config(*, disable_incompatible_features=False, confidence_floor=None):
        selection = {
            "mode": "per_regime_mask",
            "fallback": "global",
            "min_regime_samples": 2,
            "min_feature_rows": 2,
            "min_active_share": 0.25,
            "min_variance": 1e-9,
            "activity_epsilon": 1e-9,
        }
        if confidence_floor is not None:
            selection["confidence_floor"] = confidence_floor
        return {
            "selection": selection,
            "disable_incompatible_features": disable_incompatible_features,
        }

    def test_per_regime_masking_is_prefix_invariant_and_preserves_schema(self):
        X_fit, fit_regime = self._make_fit_frame()
        test_index = pd.date_range("2026-05-02", periods=2, freq="1h", tz="UTC")
        X_test = pd.DataFrame(
            {
                "trend_feature": [4.0, 0.0],
                "reversion_feature": [0.0, 13.0],
                "shared_feature": [7.0, 8.0],
                "constant_feature": [5.0, 5.0],
                "regime": [0, 1],
                "warm": [1, 1],
                "regime_confidence": [0.93, 0.94],
            },
            index=test_index,
        )
        test_regime = X_test.loc[:, ["regime", "warm", "regime_confidence"]]
        adapter = build_feature_adapter(self._make_mask_config(), regime_column="regime")
        adapter.fit(X_fit, fit_regime)

        full_transformed, policy = adapter.transform(X_test, test_regime)
        prefix_transformed, _ = adapter.transform(X_test.iloc[:1], test_regime.iloc[:1])

        pd.testing.assert_frame_equal(prefix_transformed, full_transformed.iloc[:1])
        self.assertEqual(list(full_transformed.columns), list(X_test.columns))
        pd.testing.assert_series_equal(full_transformed["regime"], X_test["regime"], check_names=False)
        pd.testing.assert_series_equal(full_transformed["warm"], X_test["warm"], check_names=False)
        pd.testing.assert_series_equal(
            full_transformed["regime_confidence"],
            X_test["regime_confidence"],
            check_names=False,
        )
        self.assertEqual(float(full_transformed.loc[test_index[0], "trend_feature"]), 4.0)
        self.assertEqual(float(full_transformed.loc[test_index[0], "reversion_feature"]), 0.0)
        self.assertEqual(float(full_transformed.loc[test_index[1], "trend_feature"]), 0.0)
        self.assertEqual(float(full_transformed.loc[test_index[1], "reversion_feature"]), 13.0)
        self.assertEqual(float(full_transformed.loc[test_index[0], "constant_feature"]), 0.0)
        self.assertEqual(float(full_transformed.loc[test_index[1], "constant_feature"]), 0.0)
        self.assertEqual(policy.metadata["selection_mode"], "per_regime_mask")
        self.assertEqual(policy.metadata["regime_mask_count"], 2)

    def test_disable_incompatible_features_keeps_schema_and_marks_disabled_columns(self):
        X_fit, fit_regime = self._make_fit_frame()
        X_val = X_fit.iloc[:2].copy()
        X_test = X_fit.iloc[2:4].copy()
        batch = apply_feature_adaptation_to_splits(
            X_fit,
            X_val,
            X_test,
            fit_regime_frame=fit_regime,
            validation_regime_frame=X_val.loc[:, ["regime", "warm", "regime_confidence"]],
            test_regime_frame=X_test.loc[:, ["regime", "warm", "regime_confidence"]],
            config=self._make_mask_config(disable_incompatible_features=True),
            regime_column="regime",
        )

        self.assertEqual(batch.summary["adapter_type"], "composite_feature_adaptation")
        self.assertIn("constant_feature", batch.fit_policy.disabled_columns)
        self.assertIn("constant_feature", batch.fit_frame.columns)
        self.assertTrue((batch.fit_frame["constant_feature"] == 0.0).all())
        self.assertEqual(batch.summary["disabled_columns"], 1)
        self.assertEqual(
            batch.summary["fit_disabled_columns_by_reason"].get("constant_feature"),
            "globally_constant",
        )

    def test_masking_records_sparse_and_fallback_reasons(self):
        X_fit, fit_regime = self._make_fit_frame()
        test_index = pd.date_range("2026-05-03", periods=4, freq="1h", tz="UTC")
        X_test = pd.DataFrame(
            {
                "trend_feature": [9.0, 1.0, 2.0, 10.0],
                "reversion_feature": [1.0, 9.0, 2.0, 0.0],
                "shared_feature": [9.0, 10.0, 11.0, 12.0],
                "constant_feature": [5.0, 5.0, 5.0, 5.0],
                "regime": [0, 1, 2, 0],
                "warm": [0, 1, 1, 1],
                "regime_confidence": [0.95, 0.20, 0.95, 0.95],
            },
            index=test_index,
        )
        test_regime = X_test.loc[:, ["regime", "warm", "regime_confidence"]]
        adapter = build_feature_adapter(
            self._make_mask_config(confidence_floor=0.8),
            regime_column="regime",
        )
        adapter.fit(X_fit, fit_regime)

        transformed, policy = adapter.transform(X_test, test_regime)

        self.assertEqual(policy.metadata["fallback_rows_total"], 3)
        self.assertEqual(policy.metadata["fallback_rows_by_reason"].get("warm_or_unavailable"), 1)
        self.assertEqual(policy.metadata["fallback_rows_by_reason"].get("confidence_below_floor"), 1)
        self.assertEqual(policy.metadata["fallback_rows_by_reason"].get("missing_regime_bank"), 1)
        self.assertEqual(float(transformed.loc[test_index[0], "constant_feature"]), 0.0)
        self.assertEqual(float(transformed.loc[test_index[3], "reversion_feature"]), 0.0)
        self.assertGreater(policy.metadata["masked_cell_count"], 0)


if __name__ == "__main__":
    unittest.main()
