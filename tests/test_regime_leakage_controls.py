import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from core import ResearchPipeline, detect_regime
from core.features import FeatureSelectionResult


class RegimeLeakageControlsTest(unittest.TestCase):
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
        base_regimes = detect_regime(base, n_regimes=3, fit_features=fit_features)
        perturbed_regimes = detect_regime(perturbed, n_regimes=3, fit_features=fit_features)

        self.assertTrue(base_regimes.loc[fit_features.index].equals(perturbed_regimes.loc[fit_features.index]))

    def test_fold_local_regime_feature_survives_feature_selection(self):
        index = pd.date_range("2026-02-01", periods=260, freq="1h", tz="UTC")
        close = 100.0 + 3.0 * np.sin(np.linspace(0.0, 10.0 * np.pi, len(index))) + np.linspace(0.0, 6.0, len(index))
        open_ = np.r_[close[0], close[:-1]]
        raw = pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) * 1.002,
                "low": np.minimum(open_, close) * 0.998,
                "close": close,
                "volume": 1_000.0,
                "quote_volume": close * 1_000.0,
                "trades": 100,
            },
            index=index,
        )

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


if __name__ == "__main__":
    unittest.main()