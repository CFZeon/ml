import unittest

import numpy as np
import pandas as pd

from core import (
    build_regime_aware_feature_frame,
    build_specialist_library_snapshot,
    train_regime_aware_model,
    train_regime_aware_walk_forward,
)
from core.regimes import RegimeStateContract


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

    @staticmethod
    def _make_single_episode_dataset(seed=2):
        rng = np.random.default_rng(seed)
        index = pd.date_range("2026-04-15", periods=160, freq="1h", tz="UTC")
        regime = np.r_[np.zeros(80, dtype=int), np.ones(80, dtype=int)]
        ret_signal = rng.normal(0.0, 1.0, len(index))
        momentum_signal = rng.normal(0.0, 0.5, len(index))
        labels = np.where(ret_signal + momentum_signal > 0.0, 1, -1)
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
        self.assertEqual(fold["inference_report"]["fallback_evidence_rows"], 40)
        self.assertEqual(fold["inference_report"]["fallback_row_share"], 1.0)
        self.assertEqual(fold["inference_report"]["candidate_classification"], "specialist_degraded_to_fallback")
        self.assertIn("2", fold["inference_report"]["unseen_regimes"])
        self.assertIn("0", fold["training_report"]["trained_regimes"])
        self.assertIn("1", fold["training_report"]["trained_regimes"])

    def test_specialist_inference_respects_delayed_regime_availability(self):
        X, y, regime_frame = self._make_balanced_dataset(n=180, seed=9)

        bundle, _ = train_regime_aware_model(
            X.iloc[:140],
            y.iloc[:140],
            regime_frame.iloc[:140],
            strategy="specialist",
            model_type="logistic",
            model_params={"random_state": 7, "max_iter": 400},
            min_samples_per_regime=20,
            coverage_config={"max_dominant_share": 1.0, "min_distinct_regimes": 1},
        )

        test_X = X.iloc[140:144]
        test_regime = regime_frame.iloc[140:144]
        state_contracts = [
            RegimeStateContract(
                as_of=test_X.index[0],
                available_at=test_X.index[1],
                label=int(test_regime.iloc[0]["regime"]),
                recognition_lag_bars=1,
                warm=True,
            ),
            *[
                RegimeStateContract(
                    as_of=timestamp,
                    available_at=timestamp,
                    label=int(test_regime.loc[timestamp, "regime"]),
                    recognition_lag_bars=0,
                    warm=True,
                )
                for timestamp in test_X.index[1:]
            ],
        ]

        _, _, inference_report = bundle.predict_with_probability_report(test_X, state_contracts)

        self.assertEqual(inference_report["fallback_rows"], 1)
        self.assertEqual(inference_report["timing_blocked_rows"], 1)
        self.assertEqual(inference_report["fallback_evidence_rows"], len(test_X))
        self.assertIn("missing", inference_report["unseen_regimes"])

    def test_specialist_strategy_prefers_canonical_regime_id_over_semantic_label(self):
        X, y, regime_frame = self._make_balanced_dataset(n=180, seed=13)
        semantic_frame = pd.DataFrame(
            {
                "regime": np.where(regime_frame["regime"].eq(0), "hmm__legacy_bull", "hmm__legacy_bear"),
                "canonical_regime_id": np.where(
                    regime_frame["regime"].eq(0),
                    "filtered_hmm__hmm_native__state_0",
                    "filtered_hmm__hmm_native__state_1",
                ),
            },
            index=regime_frame.index,
        )

        bundle, report = train_regime_aware_model(
            X.iloc[:140],
            y.iloc[:140],
            semantic_frame.iloc[:140],
            strategy="specialist",
            model_type="logistic",
            model_params={"random_state": 7, "max_iter": 400},
            min_samples_per_regime=20,
            coverage_config={"max_dominant_share": 1.0, "min_distinct_regimes": 1},
        )

        test_X = X.iloc[140:160]
        test_frame = semantic_frame.iloc[140:160]
        state_contracts = [
            RegimeStateContract(
                as_of=timestamp,
                available_at=timestamp,
                label=("hmm__renamed_bull" if test_frame.loc[timestamp, "canonical_regime_id"].endswith("state_0") else "hmm__renamed_bear"),
                warm=True,
                detector_outputs={
                    "canonical_regime_id": str(test_frame.loc[timestamp, "canonical_regime_id"]),
                },
            )
            for timestamp in test_X.index
        ]

        _, _, inference_report = bundle.predict_with_probability_report(test_X, state_contracts)

        self.assertEqual(report["regime_identity_column"], "canonical_regime_id")
        self.assertEqual(bundle.regime_column, "canonical_regime_id")
        self.assertEqual(inference_report["regime_identity_column"], "canonical_regime_id")
        self.assertEqual(inference_report["fallback_rows"], 0)
        self.assertEqual(inference_report["unseen_regimes"], [])

    def test_specialist_training_uses_admissible_regime_surface_for_delayed_contracts(self):
        X, y, regime_frame = self._make_balanced_dataset(n=120, seed=15)
        delayed_rows = 6
        state_contracts = []
        for position, timestamp in enumerate(regime_frame.index):
            available_at = timestamp if position >= delayed_rows else regime_frame.index[position + 1]
            state_contracts.append(
                RegimeStateContract(
                    as_of=timestamp,
                    available_at=available_at,
                    label=int(regime_frame.loc[timestamp, "regime"]),
                    recognition_lag_bars=(0 if position >= delayed_rows else 1),
                    warm=True,
                )
            )

        _, report = train_regime_aware_model(
            X,
            y,
            state_contracts,
            strategy="specialist",
            model_type="logistic",
            model_params={"random_state": 7, "max_iter": 400},
            min_samples_per_regime=10,
            coverage_config={"max_dominant_share": 1.0, "min_distinct_regimes": 1},
        )

        self.assertEqual(report["regime_surface_type"], "admissible_regime_surface")
        self.assertEqual(report["regime_admissibility_policy"], "available_at_decision_time")
        self.assertEqual(report["timing_blocked_training_rows"], delayed_rows)
        self.assertEqual(report["unknown_regime_rows"], delayed_rows)
        self.assertEqual(sum(report["trained_rows_by_regime"].values()), len(X) - delayed_rows)

    def test_specialist_walk_forward_emits_executable_routed_report(self):
        X, y, regime_frame = self._make_balanced_dataset(n=180, seed=21)

        result = train_regime_aware_walk_forward(
            X,
            y,
            regime_frame,
            strategy="specialist",
            model_type="logistic",
            model_params={"random_state": 7, "max_iter": 400},
            coverage_config={"max_dominant_share": 1.0, "min_distinct_regimes": 1},
            min_samples_per_regime=20,
            n_splits=1,
            train_size=140,
            test_size=40,
        )

        inference_report = result["folds"][0]["inference_report"]
        routed_report = inference_report["executable_routed_report"]

        self.assertEqual(inference_report["evidence_class"], "research_direct_model_skill")
        self.assertEqual(routed_report["evidence_class"], "executable_routed_skill")
        self.assertEqual(routed_report["fallback_evidence_rows"], 40)
        self.assertEqual(routed_report["routed_specialist_rows"], 0)
        self.assertEqual(routed_report["candidate_classification"], "specialist_degraded_to_fallback")
        self.assertEqual(routed_report["blocked_rows"], 40)
        self.assertIn("health_unbound", routed_report["eligibility_blocked_rows_by_reason"])
        self.assertEqual(len(routed_report["router_trace"]["decision_trace"]), 40)

    def test_single_episode_regime_stays_shadow_not_executable(self):
        X, y, regime_frame = self._make_single_episode_dataset()

        bundle, report = train_regime_aware_model(
            X,
            y,
            regime_frame,
            strategy="specialist",
            model_type="logistic",
            model_params={"random_state": 7, "max_iter": 400},
            min_samples_per_regime=40,
            coverage_config={"max_dominant_share": 1.0, "min_distinct_regimes": 1},
        )
        snapshot = build_specialist_library_snapshot(bundle, report, symbol="BTCUSDT", timeframe="1h")

        bull_sufficiency = report["sufficiency_by_regime"]["0"]
        specialist_bull = next(spec for spec in snapshot.specialists if spec.model_id == "specialist::0")
        fallback_spec = next(spec for spec in snapshot.specialists if spec.model_id == "fallback_generalist")

        self.assertEqual(bull_sufficiency["episode_count"], 1)
        self.assertFalse(bull_sufficiency["executable_pass"])
        self.assertEqual(bull_sufficiency["recommended_lifecycle_state"], "shadow")
        self.assertEqual(specialist_bull.metadata["lifecycle_state"], "shadow")
        self.assertEqual(fallback_spec.metadata["lifecycle_state"], "executable")

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