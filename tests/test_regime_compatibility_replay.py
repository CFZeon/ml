import unittest

import numpy as np
import pandas as pd

from core.regime import detect_regime
from core.regimes.detectors import build_compatibility_regime_detector
from core.regimes.online_state import replay_regime_detector_trace


def _make_regime_observations(index):
    angle = np.linspace(0.0, 6.0 * np.pi, len(index))
    return pd.DataFrame(
        {
            "trend_20": np.sin(angle),
            "vol_20": 0.5 + 0.25 * np.cos(angle / 2.0),
            "liquidity_20": np.linspace(-1.0, 1.0, len(index)),
        },
        index=index,
    )


class RegimeCompatibilityReplayTest(unittest.TestCase):
    def test_compatibility_replay_matches_legacy_explicit_regime_frame(self):
        index = pd.date_range("2026-08-01", periods=96, freq="1h", tz="UTC")
        observations = _make_regime_observations(index)
        config = {"method": "explicit", "column_name": "regime"}

        replayed = replay_regime_detector_trace(
            observations,
            detector=build_compatibility_regime_detector(config),
            source_map={column: "instrument_state" for column in observations.columns},
            provenance={"source_counts": {"instrument_state": len(observations.columns)}},
            mode="global_preview_only",
        )
        legacy = detect_regime(observations, method="explicit", config=config)

        pd.testing.assert_frame_equal(replayed["state_frame"], legacy.reindex(observations.index))
        self.assertEqual(len(replayed["detector_manifests"]), 1)
        self.assertEqual(replayed["trace_summary"].available_rows, len(observations))

    def test_compatibility_replay_is_deterministic(self):
        index = pd.date_range("2026-08-01", periods=72, freq="1h", tz="UTC")
        observations = _make_regime_observations(index)
        config = {"method": "explicit", "column_name": "regime", "feature_lookback": 24}

        first = replay_regime_detector_trace(
            observations,
            detector=build_compatibility_regime_detector(config),
            source_map={column: "instrument_state" for column in observations.columns},
            provenance={"source_counts": {"instrument_state": len(observations.columns)}},
        )
        second = replay_regime_detector_trace(
            observations,
            detector=build_compatibility_regime_detector(config),
            source_map={column: "instrument_state" for column in observations.columns},
            provenance={"source_counts": {"instrument_state": len(observations.columns)}},
        )

        pd.testing.assert_frame_equal(first["state_frame"], second["state_frame"])
        self.assertEqual(first["trace_summary"].to_dict(), second["trace_summary"].to_dict())
        self.assertEqual(
            first["detector_manifests"][0].to_dict(),
            second["detector_manifests"][0].to_dict(),
        )


if __name__ == "__main__":
    unittest.main()