import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd

from core import ResearchPipeline, detect_regime
from core.features import FeatureSelectionResult
from core.pipeline import SignalsStep, _FoldScopedPipeline


class RegimeLeakageControlsTest(unittest.TestCase):
    # ------------------------------------------------------------------
    # Helper: minimal OHLCV frame
    # ------------------------------------------------------------------
    @staticmethod
    def _make_raw(n=260, seed=0, start="2026-02-01"):
        rng = np.random.default_rng(seed)
        index = pd.date_range(start, periods=n, freq="1h", tz="UTC")
        close = 100.0 + rng.normal(0, 1, n).cumsum()
        open_ = np.r_[close[0], close[:-1]]
        return pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) * 1.001,
                "low": np.minimum(open_, close) * 0.999,
                "close": close,
                "volume": 1_000.0,
                "quote_volume": close * 1_000.0,
                "trades": 100,
            },
            index=index,
        )

    @staticmethod
    def _make_signal_step_pipeline(index, train_end_position=3):
        X = pd.DataFrame({"feature": np.linspace(-1.0, 1.0, len(index))}, index=index)

        class DummyClassifier:
            def __init__(self, classes, probabilities):
                self.classes_ = np.array(classes)
                self._probabilities = np.array(probabilities, dtype=float)

            def predict(self, frame):
                return np.where(frame["feature"].to_numpy() >= 0.0, 1, -1)

            def predict_proba(self, frame):
                return np.tile(self._probabilities, (len(frame), 1))

        training = {
            "last_model": DummyClassifier(classes=[-1, 1], probabilities=[0.25, 0.75]),
            "last_meta": DummyClassifier(classes=[0, 1], probabilities=[0.2, 0.8]),
            "last_selected_columns": ["feature"],
            "last_regime_fit_index": index[: train_end_position + 1],
            "last_primary_calibrator": None,
            "last_meta_calibrator": None,
            "last_signal_params": {
                "threshold": 0.0,
                "edge_threshold": 0.0,
                "meta_threshold": 0.5,
                "fraction": 0.5,
            },
            "oos_predictions": None,
            "oos_probabilities": None,
            "oos_meta_prob": None,
            "oos_continuous_signals": None,
            "oos_avg_win": 0.02,
            "oos_avg_loss": 0.01,
            "fallback_inference": {
                "mode": "post_final_training_only",
                "last_fold_train_end": index[train_end_position],
                "aligned_safe_row_count": int(max(0, len(index) - (train_end_position + 1))),
            },
        }

        class FakePipeline:
            def __init__(self, features, training_state):
                self.state = {"X": features, "training": training_state}

            def require(self, key):
                return self.state[key]

            def section(self, key):
                if key == "signals":
                    return {
                        "threshold": 0.0,
                        "edge_threshold": 0.0,
                        "meta_threshold": 0.5,
                        "fraction": 0.5,
                        "avg_win": 0.02,
                        "avg_loss": 0.01,
                        "holding_bars": 1,
                    }
                return {}

        return FakePipeline(X, training)

    # ------------------------------------------------------------------
    # Existing: fit_features freeze test (now uses HMM)
    # ------------------------------------------------------------------
    def test_fit_features_freezes_past_regime_assignments(self):
        rng = np.random.default_rng(42)
        index = pd.date_range("2026-01-01", periods=160, freq="1h", tz="UTC")
        base = pd.DataFrame(
            {
                "trend_20": rng.normal(0.0, 0.4, len(index)).cumsum() / 10.0,
                "vol_20": np.abs(rng.normal(0.02, 0.01, len(index))),
                "range_20": np.abs(rng.normal(0.015, 0.004, len(index))),
            },
            index=index,
        )
        perturbed = base.copy()
        perturbed.iloc[110:, 0] += rng.normal(3.0, 0.5, len(index) - 110)
        perturbed.iloc[110:, 1] += 0.08
        perturbed.iloc[110:, 2] += 0.04

        fit_features = base.iloc[:80]
        # HMM fitted on the same fit_features → assignments on that window must match
        base_regimes = detect_regime(base, n_regimes=2, method="hmm", fit_features=fit_features)
        perturbed_regimes = detect_regime(perturbed, n_regimes=2, method="hmm", fit_features=fit_features)

        self.assertTrue(
            base_regimes.loc[fit_features.index].equals(perturbed_regimes.loc[fit_features.index]),
            "HMM fitted on the same reference window must assign identical labels to that window "
            "regardless of what happens in the test portion.",
        )

    # ------------------------------------------------------------------
    # NEW: stable cross-fold label ordering via norm-based sorting
    # ------------------------------------------------------------------
    def test_hmm_state_ordering_is_consistent_across_calls(self):
        """States sorted by ascending centroid L1-norm → state 0 always most neutral."""
        rng = np.random.default_rng(7)
        n = 120
        index = pd.date_range("2026-03-01", periods=n, freq="1h", tz="UTC")

        # Two clearly separable regimes: first half low vol, second half high vol
        vol = np.r_[np.full(n // 2, 0.01), np.full(n // 2, 0.05)]
        trend = np.r_[np.full(n // 2, 0.002), np.full(n // 2, -0.003)]
        features = pd.DataFrame({"vol_20": vol, "trend_20": trend}, index=index)

        regimes_a = detect_regime(features, n_regimes=2, method="hmm")
        regimes_b = detect_regime(features, n_regimes=2, method="hmm")

        # Deterministic: two independent calls on the same data must agree
        pd.testing.assert_series_equal(regimes_a, regimes_b)

        # State 0 should be the low-vol (neutral) period
        low_vol_labels = regimes_a.iloc[: n // 2].unique()
        high_vol_labels = regimes_a.iloc[n // 2 :].unique()
        self.assertIn(0, low_vol_labels, "State 0 (most neutral) should appear in the low-vol window")
        self.assertNotIn(0, high_vol_labels, "State 0 (most neutral) should not dominate the high-vol window")

    # ------------------------------------------------------------------
    # NEW: unsupported legacy regime methods are rejected
    # ------------------------------------------------------------------
    def test_kmeans_method_is_rejected(self):
        rng = np.random.default_rng(1)
        index = pd.date_range("2026-04-01", periods=80, freq="1h", tz="UTC")
        features = pd.DataFrame(
            {"vol_20": np.abs(rng.normal(0.02, 0.005, 80)), "trend_20": rng.normal(0, 0.01, 80)},
            index=index,
        )
        with self.assertRaisesRegex(ValueError, r"Unknown regime detection method='kmeans'"):
            detect_regime(features, n_regimes=2, method="kmeans")

    # ------------------------------------------------------------------
    # NEW: per-fold recomputation — builder sees only fold-windowed data
    # ------------------------------------------------------------------
    def test_fold_local_regime_builder_sees_only_buffered_window(self):
        """_build_fold_local_regime_frame must pass a fold-scoped view to the builder."""
        from core.pipeline import _build_fold_local_regime_frame

        raw = self._make_raw(n=200, seed=3)

        call_log = []

        def recording_builder(scoped_pl):
            data = scoped_pl.require("data")
            call_log.append(len(data))
            # Return minimal regime features so the rest of the function can proceed
            idx = data.index
            return pd.DataFrame(
                {"vol_20": data["close"].pct_change().rolling(5).std().fillna(0.0)},
                index=idx,
            )

        import types

        fake_pipeline = types.SimpleNamespace(
            state={"raw_data": raw},
            section=lambda key: (
                {"enabled": True, "method": "explicit", "n_regimes": 2, "builder": recording_builder}
                if key == "regime"
                else {}
            ),
            require=lambda key: raw,
        )

        # Fold starts at position 100 and ends at 180; lookback=80 (default).
        # Expected buffer: raw.iloc[20:180] = 160 rows  < 200 (total raw rows).
        fold_index = raw.index[100:180]
        fit_index = raw.index[100:140]
        _build_fold_local_regime_frame(fake_pipeline, fold_index, fit_index=fit_index)

        self.assertEqual(len(call_log), 1, "Builder should be called exactly once per fold")
        # Buffer = lookback(80) + fold_size(80) = 160 < 200 (total raw rows)
        self.assertLess(
            call_log[0],
            len(raw),
            "Builder should receive a windowed slice, not the entire raw dataset",
        )
        # Rows seen = min(fold_start=100, lookback=80) + fold_size(80) = 80 + 80 = 160
        expected_buffered_len = 80 + len(fold_index)  # lookback rows + fold window
        self.assertEqual(
            call_log[0],
            expected_buffered_len,
            f"Expected {expected_buffered_len} rows (lookback + fold), got {call_log[0]}",
        )
        # No future leakage: builder must NOT see rows beyond the fold end
        self.assertLessEqual(
            call_log[0],
            len(fold_index) + 80,
            "Buffered window must not exceed lookback + fold length",
        )

    # ------------------------------------------------------------------
    # NEW: _FoldScopedPipeline correctly scopes data
    # ------------------------------------------------------------------
    def test_fold_scoped_pipeline_overrides_data_keys(self):
        raw = self._make_raw(n=100, seed=5)
        windowed = raw.iloc[20:60]

        import types

        base_pipeline = types.SimpleNamespace(
            state={},
            section=lambda k: {},
            require=lambda k: raw,
        )
        scoped = _FoldScopedPipeline(base_pipeline, windowed)

        self.assertIs(scoped.require("data"), windowed)
        self.assertIs(scoped.require("raw_data"), windowed)
        # Non-data keys should pass through to the base pipeline
        self.assertIs(scoped.require("labels"), raw)

    # ------------------------------------------------------------------
    # Existing: fold-local regime column survives feature selection
    # ------------------------------------------------------------------
    def test_fold_local_regime_feature_survives_feature_selection(self):
        raw = self._make_raw(n=260, seed=0)

        pipeline = ResearchPipeline(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "indicators": [],
                "features": {"lags": [1, 3], "frac_diff_d": 0.4, "rolling_window": 20},
                "labels": {"kind": "fixed_horizon", "horizon": 4, "threshold": 0.0001},
                "model": {"type": "gbm", "n_splits": 1, "gap": 0},
                "feature_selection": {"enabled": True, "max_features": 1},
                "signals": {"avg_win": 0.02, "avg_loss": 0.02},
                "backtest": {"use_open_execution": False, "signal_delay_bars": 1},
            }
        )
        pipeline.state["raw_data"] = raw
        pipeline.state["data"] = raw.copy()
        pipeline.build_features()
        pipeline.build_labels()
        pipeline.align_data()

        def fake_regime_frame(_pipeline, fold_index, fit_index=None):
            split_point = max(1, len(fold_index) // 2)
            regime = pd.Series(0.0, index=fold_index)
            regime.iloc[split_point:] = 1.0
            frame = pd.DataFrame({"regime": regime}, index=fold_index)
            return frame, {"regime": "regime"}

        def fake_select_features(features, y, feature_blocks=None, config=None):
            return FeatureSelectionResult(
                frame=features.loc[:, ["regime"]].copy(),
                feature_blocks={"regime": "regime"},
                report={"top_mi_scores": {"regime": 1.0}},
            )

        with patch("core.pipeline._build_fold_local_regime_frame", side_effect=fake_regime_frame), patch(
            "core.pipeline.select_features",
            side_effect=fake_select_features,
        ):
            training = pipeline.train_models()

        self.assertIn("regime", training["last_selected_columns"])
        self.assertGreaterEqual(training["feature_selection"]["folds"][0]["selected_features"], 1)
        self.assertEqual(training["fallback_inference"]["mode"], "post_final_training_only")
        self.assertEqual(training["fallback_inference"]["feature_selection_fit_scope"], "last_fold_train_only")
        self.assertGreaterEqual(training["fallback_inference"]["aligned_safe_row_count"], 1)

    def test_signals_fallback_scores_only_post_training_rows(self):
        index = pd.date_range("2026-05-01", periods=6, freq="1h", tz="UTC")
        fake_pipeline = self._make_signal_step_pipeline(index, train_end_position=3)

        signal_result = SignalsStep().run(fake_pipeline)

        expected_index = index[index > index[3]]
        pd.testing.assert_index_equal(signal_result["signals"].index, expected_index)
        self.assertEqual(signal_result["signal_source"], "post_final_training_fallback")
        self.assertEqual(signal_result["fallback_scope"]["scored_row_count"], len(expected_index))
        self.assertEqual(signal_result["fallback_scope"]["excluded_row_count"], 4)

    def test_signals_fallback_warns_and_returns_empty_when_no_post_training_rows(self):
        index = pd.date_range("2026-05-01", periods=4, freq="1h", tz="UTC")
        fake_pipeline = self._make_signal_step_pipeline(index, train_end_position=3)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            signal_result = SignalsStep().run(fake_pipeline)

        self.assertEqual(len(signal_result["signals"]), 0)
        self.assertEqual(signal_result["signal_source"], "post_final_training_fallback_empty")
        self.assertTrue(
            any("post-final-training rows" in str(item.message) for item in caught),
            "Expected a warning when the restricted fallback window is empty.",
        )


if __name__ == "__main__":
    unittest.main()