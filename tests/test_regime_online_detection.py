import unittest

import numpy as np
import pandas as pd

from core.regime import build_instrument_regime_state, detect_regime


class OnlineRegimeDetectionTest(unittest.TestCase):
    @staticmethod
    def _make_raw(n=220, seed=0):
        rng = np.random.default_rng(seed)
        index = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
        base_returns = rng.normal(0.0005, 0.004, n)
        base_returns[140:170] += rng.normal(-0.02, 0.03, 30)
        close = 100.0 * np.cumprod(1.0 + base_returns)
        open_ = np.r_[close[0], close[:-1]]
        volume = np.full(n, 2_000.0)
        volume[140:170] = 350.0
        return pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) * 1.002,
                "low": np.minimum(open_, close) * 0.998,
                "close": close,
                "volume": volume,
                "quote_volume": close * volume,
                "trades": np.where(volume > 1_000.0, 120, 25),
            },
            index=index,
        )

    def test_online_regime_prefix_is_invariant_to_appended_future(self):
        raw = self._make_raw()
        features = build_instrument_regime_state(raw, rolling_window=20)
        prefix = features.iloc[:140]

        prefix_regimes = detect_regime(
            prefix,
            method="online",
            config={"online_min_periods": 20, "online_lookback": 96},
        )
        full_regimes = detect_regime(
            features,
            method="online",
            config={"online_min_periods": 20, "online_lookback": 96},
        )

        pd.testing.assert_frame_equal(full_regimes.loc[prefix_regimes.index], prefix_regimes)

    def test_online_regime_surfaces_structural_break_and_volatility_shift(self):
        raw = self._make_raw(seed=7)
        features = build_instrument_regime_state(raw, rolling_window=20)
        regimes = detect_regime(
            features,
            method="online",
            config={"online_min_periods": 20, "online_lookback": 96},
        )

        self.assertIn("structural_break_regime", regimes.columns)
        self.assertIn("mean_reversion_regime", regimes.columns)

        calm_window = regimes.iloc[60:110]
        break_window = regimes.iloc[140:148]
        shock_window = regimes.iloc[145:165]
        self.assertLess(float(calm_window["structural_break_regime"].mean()), float(break_window["structural_break_regime"].mean()))
        self.assertLess(float(calm_window["volatility_regime"].mean()), float(shock_window["volatility_regime"].mean()))


if __name__ == "__main__":
    unittest.main()