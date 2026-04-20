import json
import unittest

import numpy as np
import pandas as pd

from core import ResearchPipeline, check_stationarity, derive_feature_metadata, evaluate_feature_admission


def _make_raw(index):
    close = 100.0 + np.linspace(0.0, 12.0, len(index)) + 2.0 * np.sin(np.linspace(0.0, 8.0 * np.pi, len(index)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    volume = 1_000.0 + 120.0 * (1.0 + np.cos(np.linspace(0.0, 4.0 * np.pi, len(index))))
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": close * volume,
            "trades": 100,
        },
        index=index,
    )


class FeatureAdmissionPolicyTest(unittest.TestCase):
    def test_stationary_feature_can_still_fail_stability_admission(self):
        rng = np.random.default_rng(42)
        index = pd.date_range("2026-09-01", periods=160, freq="1h", tz="UTC")
        brittle = pd.Series(rng.normal(0.0, 1.0, len(index)), index=index, name="brittle_alpha")
        target = pd.Series(
            np.r_[brittle.iloc[:80].to_numpy(), -brittle.iloc[80:].to_numpy()],
            index=index,
            name="label",
        )

        stats = check_stationarity(brittle)
        self.assertTrue(stats["stationary"])

        metadata = derive_feature_metadata(
            feature_blocks={"brittle_alpha": "price_volume"},
            columns=["brittle_alpha"],
        )
        report = evaluate_feature_admission(
            pd.DataFrame({"brittle_alpha": brittle}, index=index),
            target,
            feature_metadata=metadata,
            config={
                "rolling_window": 32,
                "rolling_step": 16,
                "min_rolling_sign_stability": 0.75,
                "min_permutation_importance_gap": -1.0,
                "max_perturbation_sensitivity": 1.0,
            },
        )

        self.assertNotIn("brittle_alpha", report["admitted_columns"])
        self.assertIn("brittle_alpha", report["rejected_columns"])
        self.assertIn("rolling_sign_stability_failed", report["feature_reports"]["brittle_alpha"]["reasons"])

    def test_retired_features_are_excluded_unless_reenabled(self):
        index = pd.date_range("2026-09-01", periods=140, freq="1h", tz="UTC")
        raw_data = _make_raw(index)
        features = pd.DataFrame(
            {
                "stable_alpha": raw_data["close"].pct_change().fillna(0.0).rolling(3, min_periods=1).mean(),
                "retired_alpha": raw_data["close"].pct_change(2).fillna(0.0),
            },
            index=index,
        )
        metadata = derive_feature_metadata(
            feature_blocks={"stable_alpha": "price_volume", "retired_alpha": "price_volume"},
            columns=["stable_alpha", "retired_alpha"],
            retired_features={"retired_alpha": {"status": "retired", "reason": "manual_retire"}},
        )

        pipeline = ResearchPipeline(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "labels": {"kind": "fixed_horizon", "horizon": 3, "threshold": 0.0},
                "feature_governance": {
                    "retired_features": {"retired_alpha": {"status": "retired", "reason": "manual_retire"}},
                },
            }
        )
        pipeline.state["raw_data"] = raw_data
        pipeline.state["data"] = raw_data.copy()
        pipeline.state["features"] = features
        pipeline.state["feature_blocks"] = {"stable_alpha": "price_volume", "retired_alpha": "price_volume"}
        pipeline.state["feature_families"] = {"stable_alpha": "endogenous_price", "retired_alpha": "endogenous_price"}
        pipeline.state["feature_metadata"] = metadata
        pipeline.build_labels()
        aligned = pipeline.align_data()

        self.assertIn("stable_alpha", aligned["X"].columns)
        self.assertNotIn("retired_alpha", aligned["X"].columns)

        reenabled = ResearchPipeline(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "labels": {"kind": "fixed_horizon", "horizon": 3, "threshold": 0.0},
                "feature_governance": {
                    "retired_features": {"retired_alpha": {"status": "retired", "reason": "manual_retire"}},
                    "include_retired_features": True,
                },
            }
        )
        reenabled.state["raw_data"] = raw_data
        reenabled.state["data"] = raw_data.copy()
        reenabled.state["features"] = features
        reenabled.state["feature_blocks"] = {"stable_alpha": "price_volume", "retired_alpha": "price_volume"}
        reenabled.state["feature_families"] = {"stable_alpha": "endogenous_price", "retired_alpha": "endogenous_price"}
        reenabled.state["feature_metadata"] = metadata
        reenabled.build_labels()
        aligned_reenabled = reenabled.align_data()

        self.assertIn("retired_alpha", aligned_reenabled["X"].columns)

    def test_feature_governance_summary_is_serializable_in_training_output(self):
        index = pd.date_range("2026-09-01", periods=220, freq="1h", tz="UTC")
        raw_data = _make_raw(index)
        features = pd.DataFrame(
            {
                "stable_alpha": raw_data["close"].pct_change().fillna(0.0).rolling(4, min_periods=1).mean(),
                "support_alpha": raw_data["volume"].pct_change().fillna(0.0),
            },
            index=index,
        )
        pipeline = ResearchPipeline(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "labels": {"kind": "fixed_horizon", "horizon": 4, "threshold": 0.0},
                "model": {"type": "logistic", "cv_method": "walk_forward", "n_splits": 1, "gap": 0, "validation_fraction": 0.2, "meta_n_splits": 2},
                "feature_selection": {"enabled": False},
                "signals": {"avg_win": 0.02, "avg_loss": 0.02},
                "backtest": {"use_open_execution": False, "signal_delay_bars": 1},
            }
        )
        pipeline.state["raw_data"] = raw_data
        pipeline.state["data"] = raw_data.copy()
        pipeline.state["features"] = features
        pipeline.state["feature_blocks"] = {"stable_alpha": "price_volume", "support_alpha": "price_volume"}
        pipeline.state["feature_families"] = {"stable_alpha": "endogenous_price", "support_alpha": "endogenous_price"}
        pipeline.state["feature_metadata"] = derive_feature_metadata(
            feature_blocks=pipeline.state["feature_blocks"],
            feature_families=pipeline.state["feature_families"],
            columns=features.columns,
        )
        pipeline.build_labels()
        pipeline.align_data()

        training = pipeline.train_models()

        self.assertIn("feature_governance", training)
        self.assertIn("admission_summary", training["feature_governance"])
        self.assertIn("feature_admission", training["promotion_gates"])
        json.dumps(training["feature_governance"])


if __name__ == "__main__":
    unittest.main()