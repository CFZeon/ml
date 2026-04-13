import unittest

import numpy as np
import pandas as pd

from core import detect_regime


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


if __name__ == "__main__":
    unittest.main()