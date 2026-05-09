import unittest

import pandas as pd

from core.feature_adaptation import apply_feature_adaptation_to_splits


class FeatureAdaptationScalingTest(unittest.TestCase):
    @staticmethod
    def _fit_frames():
        index = pd.date_range("2026-05-01", periods=6, freq="1h", tz="UTC")
        fit = pd.DataFrame(
            {
                "ret_1": [1.0, 2.0, 1.5, 9.0, 10.0, 11.0],
                "vol_24": [100.0, 110.0, 105.0, 210.0, 220.0, 215.0],
                "constant_feature": [5.0] * 6,
                "regime": [0, 0, 0, 1, 1, 1],
                "warm": [1, 1, 1, 1, 1, 1],
                "regime_confidence": [0.95, 0.92, 0.91, 0.98, 0.96, 0.97],
                "score": [0.2, 0.25, 0.22, 0.85, 0.88, 0.9],
            },
            index=index,
        )
        regime = fit.loc[:, ["regime", "warm", "regime_confidence"]].copy()
        return fit, regime

    def test_regime_conditioned_scaling_is_prefix_invariant_and_preserves_schema(self):
        fit, fit_regime = self._fit_frames()
        prefix_index = pd.date_range("2026-05-02", periods=2, freq="1h", tz="UTC")
        prefix = pd.DataFrame(
            {
                "ret_1": [1.75, 10.5],
                "vol_24": [108.0, 218.0],
                "constant_feature": [5.0, 5.0],
                "regime": [0, 1],
                "warm": [1, 1],
                "regime_confidence": [0.94, 0.95],
                "score": [0.24, 0.89],
            },
            index=prefix_index,
        )
        extended = pd.concat(
            [
                prefix,
                pd.DataFrame(
                    {
                        "ret_1": [2.1, 9.5],
                        "vol_24": [109.5, 214.0],
                        "constant_feature": [5.0, 5.0],
                        "regime": [0, 1],
                        "warm": [1, 1],
                        "regime_confidence": [0.96, 0.97],
                        "score": [0.21, 0.84],
                    },
                    index=pd.date_range("2026-05-02 02:00", periods=2, freq="1h", tz="UTC"),
                ),
            ]
        )
        prefix_regime = prefix.loc[:, ["regime", "warm", "regime_confidence"]]
        extended_regime = extended.loc[:, ["regime", "warm", "regime_confidence"]]
        config = {
            "scaling": {
                "mode": "regime_conditioned",
                "fallback": "global",
                "min_regime_samples": 2,
                "confidence_floor": 0.8,
            }
        }

        prefix_result = apply_feature_adaptation_to_splits(
            fit,
            None,
            prefix,
            fit_regime_frame=fit_regime,
            test_regime_frame=prefix_regime,
            config=config,
        )
        extended_result = apply_feature_adaptation_to_splits(
            fit,
            None,
            extended,
            fit_regime_frame=fit_regime,
            test_regime_frame=extended_regime,
            config=config,
        )

        self.assertEqual(list(prefix_result.fit_frame.columns), list(fit.columns))
        self.assertEqual(list(prefix_result.test_frame.columns), list(prefix.columns))
        self.assertEqual(prefix_result.summary["adapter_type"], "regime_conditioned_scaling")
        self.assertFalse(prefix_result.summary["no_op"])
        pd.testing.assert_frame_equal(
            prefix_result.test_frame,
            extended_result.test_frame.loc[prefix.index],
        )
        pd.testing.assert_series_equal(prefix_result.test_frame["regime"], prefix["regime"], check_names=False)
        pd.testing.assert_series_equal(prefix_result.test_frame["warm"], prefix["warm"], check_names=False)
        self.assertTrue((prefix_result.test_frame["ret_1"] != prefix["ret_1"]).any())
        self.assertTrue((prefix_result.test_frame["constant_feature"] == 0.0).all())

    def test_regime_conditioned_scaling_records_fallback_reasons(self):
        fit, fit_regime = self._fit_frames()
        test = pd.DataFrame(
            {
                "ret_1": [1.2, 10.8, 4.0, 3.0],
                "vol_24": [101.0, 221.0, 150.0, 120.0],
                "constant_feature": [5.0, 5.0, 5.0, 5.0],
                "regime": [0, 1, 2, 0],
                "warm": [0, 1, 1, 1],
                "regime_confidence": [0.99, 0.2, 0.9, 0.4],
                "score": [0.18, 0.91, 0.5, 0.3],
            },
            index=pd.date_range("2026-05-03", periods=4, freq="1h", tz="UTC"),
        )
        test_regime = test.loc[:, ["regime", "warm", "regime_confidence"]]
        config = {
            "scaling": {
                "mode": "regime_conditioned",
                "fallback": "global",
                "min_regime_samples": 2,
                "confidence_floor": 0.8,
            }
        }

        result = apply_feature_adaptation_to_splits(
            fit,
            None,
            test,
            fit_regime_frame=fit_regime,
            test_regime_frame=test_regime,
            config=config,
        )

        manifest = result.manifest
        fallback_counts = result.test_policy.metadata["fallback_rows_by_reason"]
        self.assertEqual(manifest["regime_bank_count"], 2)
        self.assertIn("constant_feature", manifest["constant_columns"])
        self.assertEqual(result.summary["test_fallback_rows_total"], 4)
        self.assertEqual(fallback_counts["warm_or_unavailable"], 1)
        self.assertEqual(fallback_counts["confidence_below_floor"], 2)
        self.assertEqual(fallback_counts["missing_regime_bank"], 1)
        pd.testing.assert_series_equal(result.test_frame["regime"], test["regime"], check_names=False)
        pd.testing.assert_series_equal(result.test_frame["warm"], test["warm"], check_names=False)


if __name__ == "__main__":
    unittest.main()