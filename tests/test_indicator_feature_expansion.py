import unittest

import numpy as np
import pandas as pd

from core import (
    ADX,
    ATR,
    DonchianChannels,
    OnBalanceVolume,
    StochasticOscillator,
    build_feature_set,
    run_indicators,
)


def _make_raw(index):
    close = 100.0 + np.linspace(0.0, 15.0, len(index)) + 2.5 * np.sin(np.linspace(0.0, 10.0 * np.pi, len(index)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * (1.0015 + 0.0005 * np.sin(np.linspace(0.0, 6.0 * np.pi, len(index))))
    low = np.minimum(open_, close) * (0.9985 - 0.0003 * np.cos(np.linspace(0.0, 4.0 * np.pi, len(index))))
    volume = 1_000.0 + 150.0 * (1.0 + np.cos(np.linspace(0.0, 6.0 * np.pi, len(index))))
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


class IndicatorFeatureExpansionTest(unittest.TestCase):
    def test_new_indicator_extractors_emit_indicator_aware_features(self):
        index = pd.date_range("2026-01-01", periods=220, freq="1h", tz="UTC")
        raw = _make_raw(index)

        indicator_run = run_indicators(
            raw,
            [
                ADX(14),
                StochasticOscillator(14, 3, 3),
                OnBalanceVolume(),
                DonchianChannels(20),
                ATR(14),
            ],
        )
        feature_set = build_feature_set(
            indicator_run.frame,
            lags=[1, 3],
            frac_diff_d=0.4,
            indicator_run=indicator_run,
            rolling_window=20,
        )

        expected_columns = [
            "adx_14_strength",
            "adx_14_di_spread",
            "stoch_14_3_3_spread",
            "obv_pressure",
            "donchian_20_position",
            "breakout_trend_pressure",
            "trend_pullback_alignment",
        ]
        for column in expected_columns:
            self.assertIn(column, feature_set.frame.columns)
            self.assertEqual(feature_set.feature_families[column], "indicator")

        self.assertEqual(feature_set.feature_blocks["adx_14_strength"], "adx")
        self.assertEqual(feature_set.feature_blocks["stoch_14_3_3_spread"], "stochastic")
        self.assertEqual(feature_set.feature_blocks["obv_pressure"], "obv")
        self.assertEqual(feature_set.feature_blocks["donchian_20_position"], "donchian")

        self.assertNotIn("obv_diff", feature_set.frame.columns)
        self.assertNotIn("adx_14_diff", feature_set.frame.columns)


if __name__ == "__main__":
    unittest.main()