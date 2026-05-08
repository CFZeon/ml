import unittest

import numpy as np
import pandas as pd

from core.regimes.detectors import (
    BreakRegimeDetector,
    LiquidityRegimeDetector,
    TrendRegimeDetector,
    VolatilityRegimeDetector,
    build_regime_detector,
    is_native_regime_detector_spec,
    resolve_authoritative_regime_detector_spec,
)
from core.regimes.online_state import replay_regime_detector_trace


def _make_observations(index):
    angle = np.linspace(0.0, 4.0 * np.pi, len(index))
    return pd.DataFrame(
        {
            "trend_20": np.sin(angle),
            "trend_slope_5": np.cos(angle / 2.0),
            "vol_20": 0.2 + 0.1 * np.cos(angle),
            "atr_14": 1.0 + 0.5 * np.sin(angle / 3.0),
            "liquidity_20": np.linspace(-1.0, 1.0, len(index)),
            "trade_intensity": 10.0 + 2.0 * np.sin(angle),
            "amihud_20": np.linspace(0.8, -0.2, len(index)),
            "break_score": np.r_[np.zeros(len(index) // 2), np.linspace(0.0, 3.0, len(index) - len(index) // 2)],
        },
        index=index,
    )


class RegimeDetectorsTest(unittest.TestCase):
    def test_build_regime_detector_creates_native_detector_for_supported_types(self):
        config = {"column_name": "regime"}
        self.assertIsInstance(build_regime_detector({"type": "trend_state"}, config=config), TrendRegimeDetector)
        self.assertIsInstance(build_regime_detector({"type": "volatility_state"}, config=config), VolatilityRegimeDetector)
        self.assertIsInstance(build_regime_detector({"type": "liquidity_state"}, config=config), LiquidityRegimeDetector)
        self.assertIsInstance(build_regime_detector({"type": "break_state"}, config=config), BreakRegimeDetector)

    def test_native_detector_replay_is_prefix_invariant_for_frozen_fit_prefix(self):
        index = pd.date_range("2026-08-01", periods=96, freq="1h", tz="UTC")
        observations = _make_observations(index)
        fit_prefix = observations.iloc[:48]
        config = {"column_name": "regime", "lower_quantile": 0.3, "upper_quantile": 0.7}
        spec = {"name": "trend_native", "type": "trend_state"}

        prefix_only = replay_regime_detector_trace(
            observations.iloc[:72],
            detector=build_regime_detector(spec, config=config),
            source_map={column: "instrument_state" for column in observations.columns},
            provenance={"source_counts": {"instrument_state": len(observations.columns)}},
            fit_observations=fit_prefix,
        )
        extended = replay_regime_detector_trace(
            observations,
            detector=build_regime_detector(spec, config=config),
            source_map={column: "instrument_state" for column in observations.columns},
            provenance={"source_counts": {"instrument_state": len(observations.columns)}},
            fit_observations=fit_prefix,
        )

        pd.testing.assert_frame_equal(prefix_only["state_frame"], extended["state_frame"].iloc[:72])
        self.assertEqual(
            prefix_only["detector_manifests"][0].to_dict(),
            extended["detector_manifests"][0].to_dict(),
        )

    def test_native_detector_emits_neutral_unavailable_state_without_matching_columns(self):
        index = pd.date_range("2026-08-01", periods=12, freq="1h", tz="UTC")
        observations = pd.DataFrame({"feature_a": np.linspace(0.0, 1.0, len(index))}, index=index)
        replayed = replay_regime_detector_trace(
            observations,
            detector=build_regime_detector({"name": "trend_native", "type": "trend_state"}, config={"column_name": "regime"}),
            source_map={"feature_a": "instrument_state"},
            provenance={"source_counts": {"instrument_state": 1}},
            fit_observations=observations,
        )

        self.assertTrue((replayed["state_frame"]["trend_regime"] == 0).all())
        self.assertTrue((replayed["state_frame"]["warm"] == 0).all())
        self.assertEqual(replayed["detector_manifests"][0].metadata["selected_columns"], [])

    def test_resolve_authoritative_regime_detector_spec_prefers_primary_and_rejects_multiple_native(self):
        config = {
            "detectors": [
                {"name": "legacy", "type": "volatility_trend_hybrid"},
                {"name": "native_liquidity", "type": "liquidity_state", "primary": True},
            ]
        }
        selected = resolve_authoritative_regime_detector_spec(config)
        self.assertEqual(selected["name"], "native_liquidity")
        self.assertTrue(is_native_regime_detector_spec(selected))

        with self.assertRaisesRegex(ValueError, "at most one enabled native regime detector"):
            resolve_authoritative_regime_detector_spec(
                {
                    "detectors": [
                        {"name": "trend", "type": "trend_state"},
                        {"name": "vol", "type": "volatility_state"},
                    ]
                }
            )


if __name__ == "__main__":
    unittest.main()