import unittest

import numpy as np
import pandas as pd

from core import build_regime_aware_feature_frame, train_regime_aware_model, train_regime_aware_walk_forward


class RegimeAwareTrainingTest(unittest.TestCase):
    @staticmethod
    def _make_balanced_dataset(n=240, seed=0):
        rng = np.random.default_rng(seed)
        index = pd.date_range("2026-03-01", periods=n, freq="1h", tz="UTC")
        regime = np.tile([0, 1], n // 2)
        ret_signal = rng.normal(0.0, 1.0, n)
        momentum_signal = rng.normal(0.0, 0.7, n)
        vol_state = np.where(regime == 0, 0.8, 1.6) + rng.normal(0.0, 0.05, n)
        labels = np.where(regime == 0, np.where(ret_signal + 0.2 * momentum_signal > 0.0, 1, -1), np.where(-ret_signal + 0.2 * momentum_signal > 0.0, 1, -1))
        X = pd.DataFrame(
            {"ret_signal": ret_signal, "momentum_signal": momentum_signal},
            index=index,
        )
        regime_frame = pd.DataFrame({"regime": regime, "ewm_vol_20": vol_state}, index=index)
        y = pd.Series(labels, index=index)
        return X, y, regime_frame

    @staticmethod
    def _make_unseen_regime_dataset(seed=1):
        rng = np.random.default_rng(seed)
        index = pd.date_range("2026-04-01", periods=200, freq="1h", tz="UTC")
        regime = np.r_[np.tile([0, 1], 80), np.full(40, 2)]
        ret_signal = rng.normal(0.0, 1.0, len(index))
        momentum_signal = rng.normal(0.0, 0.6, len(index))
        labels = np.where(regime == 0, np.where(ret_signal > 0.0, 1, -1), np.where(regime == 1, np.where(-ret_signal > 0.0, 1, -1), np.where(momentum_signal > 0.0, 1, -1)))
        X = pd.DataFrame({"ret_signal": ret_signal, "momentum_signal": momentum_signal}, index=index)
        regime_frame = pd.DataFrame({"regime": regime}, index=index)
        y = pd.Series(labels, index=index)
        return X, y, regime_frame

    def test_feature_strategy_adds_regime_features_and_reports_coverage(self):
        X, y, regime_frame = self._make_balanced_dataset()

        result = train_regime_aware_walk_forward(
            X,
            y,
            regime_frame,
            strategy="feature",
            model_type="logistic",
            model_params={"random_state": 7, "max_iter": 400},
            feature_config={"regime_interactions": True},
            coverage_config={"max_dominant_share": 0.7, "min_distinct_regimes": 2},
            n_splits=3,
            train_size=120,
            test_size=30,
        )

        self.assertEqual(result["strategy"], "feature")
        self.assertGreater(len(result["oos_predictions"]), 0)
        self.assertTrue(all(fold["coverage"]["train"]["coverage_ok"] for fold in result["folds"]))
        self.assertTrue(any(column.startswith("regime__") for column in result["last_model"].feature_columns))
        self.assertTrue(any(column.startswith("vol_norm__") for column in result["last_model"].feature_columns))

    def test_specialist_strategy_falls_back_on_unseen_regime(self):
        X, y, regime_frame = self._make_unseen_regime_dataset()

        result = train_regime_aware_walk_forward(
            X,
            y,
            regime_frame,
            strategy="specialist",
            model_type="logistic",
            model_params={"random_state": 11, "max_iter": 400},
            coverage_config={"max_dominant_share": 1.0, "min_distinct_regimes": 1},
            min_samples_per_regime=40,
            n_splits=1,
            train_size=160,
            test_size=40,
        )

        fold = result["folds"][0]
        self.assertEqual(result["strategy"], "specialist")
        self.assertGreater(fold["inference_report"]["fallback_rows"], 0)
        self.assertIn("2", fold["inference_report"]["unseen_regimes"])
        self.assertIn("0", fold["training_report"]["trained_regimes"])
        self.assertIn("1", fold["training_report"]["trained_regimes"])

    def test_feature_strategy_rejects_non_identity_feature_adaptation_scaling(self):
        X, y, regime_frame = self._make_balanced_dataset()

        with self.assertRaisesRegex(ValueError, "strategy='feature'"):
            train_regime_aware_walk_forward(
                X,
                y,
                regime_frame,
                strategy="feature",
                model_type="logistic",
                model_params={"random_state": 7, "max_iter": 400},
                feature_config={
                    "regime_interactions": True,
                    "feature_adaptation": {
                        "scaling": {"mode": "regime_conditioned", "fallback": "global"}
                    },
                },
                coverage_config={"max_dominant_share": 0.7, "min_distinct_regimes": 2},
                n_splits=1,
                train_size=120,
                test_size=30,
            )

    def test_feature_strategy_rejects_non_identity_feature_adaptation_masking(self):
        X, y, regime_frame = self._make_balanced_dataset()

        with self.assertRaisesRegex(ValueError, "strategy='feature'"):
            train_regime_aware_walk_forward(
                X,
                y,
                regime_frame,
                strategy="feature",
                model_type="logistic",
                model_params={"random_state": 7, "max_iter": 400},
                feature_config={
                    "regime_interactions": True,
                    "feature_adaptation": {
                        "selection": {"mode": "per_regime_mask", "fallback": "global"},
                        "disable_incompatible_features": True,
                    },
                },
                coverage_config={"max_dominant_share": 0.7, "min_distinct_regimes": 2},
                n_splits=1,
                train_size=120,
                test_size=30,
            )

    def test_feature_strategy_bundle_reuses_frozen_adapter_at_inference(self):
        X, y, regime_frame = self._make_balanced_dataset(n=160, seed=4)
        train_X = X.iloc[:120]
        train_y = y.iloc[:120]
        train_regime = regime_frame.iloc[:120]
        test_X = X.iloc[120:140]
        test_regime = regime_frame.iloc[120:140].drop(columns=["ewm_vol_20"])

        bundle, report = train_regime_aware_model(
            train_X,
            train_y,
            train_regime,
            strategy="feature",
            model_type="logistic",
            model_params={"random_state": 7, "max_iter": 400},
            feature_config={
                "regime_interactions": True,
                "feature_adaptation": {
                    "interaction_budget": {
                        "enabled": True,
                        "max_features": 1,
                        "max_regimes": 1,
                    }
                },
            },
            coverage_config={"max_dominant_share": 0.7, "min_distinct_regimes": 2},
        )

        transformed = bundle._transform_feature_strategy(test_X, test_regime)
        recomputed = build_regime_aware_feature_frame(
            test_X,
            test_regime,
            config={
                "regime_interactions": True,
                "feature_adaptation": {
                    "interaction_budget": {
                        "enabled": True,
                        "max_features": 1,
                        "max_regimes": 1,
                    }
                },
            },
        ).frame.reindex(columns=bundle.feature_columns, fill_value=0.0)

        self.assertIsNotNone(bundle.feature_adapter)
        self.assertIn("feature_adaptation", report)
        self.assertIn("vol_norm__ret_signal", transformed.columns)
        self.assertFalse(transformed.equals(recomputed))


if __name__ == "__main__":
    unittest.main()