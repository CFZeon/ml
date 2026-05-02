import unittest

import numpy as np
import pandas as pd

from core import run_backtest
from core.automl import _validate_forbidden_search_space_paths
from core.features import fit_stationarity_transforms, screen_features_for_stationarity
from core.labeling import triple_barrier_labels
from core.models import build_model, cpcv_split


class RemediationBatch1Test(unittest.TestCase):
    def test_stationarity_fit_window_reports_leakage_risk(self):
        index = pd.date_range("2026-01-01", periods=200, freq="1h", tz="UTC")
        # Random walk: non-stationary baseline requiring transform search.
        values = np.cumsum(np.random.default_rng(42).normal(0, 1, size=len(index))) + 100.0
        features = pd.DataFrame({"x": values}, index=index)

        out = screen_features_for_stationarity(
            features,
            feature_blocks={"x": "price_volume"},
            config={"enabled": True, "drop_failed": False},
            fit_window=120,
        )
        self.assertFalse(bool(out.report.get("leakage_risk", True)))
        self.assertIn("transform_specs", out.report)

        out_full = screen_features_for_stationarity(
            features,
            feature_blocks={"x": "price_volume"},
            config={"enabled": True, "drop_failed": False},
        )
        self.assertTrue(bool(out_full.report.get("leakage_risk", False)))

    def test_fit_stationarity_transforms_returns_specs(self):
        index = pd.date_range("2026-02-01", periods=180, freq="1h", tz="UTC")
        values = np.cumsum(np.random.default_rng(7).normal(0, 0.5, size=len(index))) + 50.0
        features = pd.DataFrame({"close_like": values}, index=index)

        fitted = fit_stationarity_transforms(features, fit_window=120)
        self.assertIn("specs", fitted)
        self.assertIn("reports", fitted)
        self.assertEqual(int(fitted.get("fit_window")), 120)

    def test_triple_barrier_conservative_tie_break(self):
        index = pd.date_range("2026-03-01", periods=30, freq="1h", tz="UTC")
        close = pd.Series(100.0, index=index)
        # Force both barriers hit on each forward bar by making high/low cross both sides.
        high = pd.Series(110.0, index=index)
        low = pd.Series(90.0, index=index)
        vol = pd.Series(0.02, index=index)

        labeled = triple_barrier_labels(
            close=close,
            volatility=vol,
            high=high,
            low=low,
            pt_sl=(1.0, 1.0),
            max_holding=3,
            barrier_tie_break="conservative",
        )
        self.assertTrue((labeled["label"] == 0).all())
        self.assertTrue((labeled["barrier"] == "tie").all())
        integrity = dict(labeled.attrs.get("integrity_report") or {})
        self.assertGreater(int(integrity.get("tie_count", 0)), 0)

    def test_cpcv_embargo_extends_by_max_lag(self):
        X = pd.DataFrame({"x": np.arange(100)})
        splits = list(cpcv_split(X, n_blocks=5, test_blocks=2, embargo=3, max_lag=7))
        self.assertGreater(len(splits), 0)
        embargo_rows_seen = []
        for _, _, meta in splits:
            self.assertEqual(int(meta.get("lag_embargo_extension", -1)), 7)
            embargo_rows_seen.append(int(meta.get("embargo_rows", 0)))
        self.assertGreater(max(embargo_rows_seen), 0)

    def test_backtest_disallows_same_bar_when_disabled(self):
        index = pd.date_range("2026-04-01", periods=10, freq="1h", tz="UTC")
        close = pd.Series(np.linspace(100, 101, len(index)), index=index)
        signals = pd.Series(0.0, index=index)

        with self.assertRaises(ValueError):
            run_backtest(
                close=close,
                signals=signals,
                engine="pandas",
                allow_same_bar_fill_fallback=False,
                execution_prices=None,
            )

    def test_forbidden_search_paths_raise(self):
        bad_space = {
            "model": {
                "validation_fraction": {"type": "categorical", "choices": [0.2, 0.3]},
            }
        }
        with self.assertRaises(ValueError):
            _validate_forbidden_search_space_paths(bad_space)

    def test_rf_default_n_jobs_is_deterministic_single_worker(self):
        model = build_model("rf", model_params={"random_state": 42})
        self.assertEqual(int(model.n_jobs), 1)


if __name__ == "__main__":
    unittest.main()
