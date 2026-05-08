import types
import unittest

import numpy as np
import pandas as pd

from core import ResearchPipeline
from core.regime import RegimeFeatureSet
from core.regimes.observations import (
    build_fold_local_regime_observation_feature_set,
    resolve_pipeline_regime_observation_feature_set,
)


def _make_ohlcv(index, *, drift=10.0, amplitude=2.0, volume_base=1_000.0):
    steps = np.linspace(0.0, 1.0, len(index))
    cycle = np.sin(np.linspace(0.0, 6.0 * np.pi, len(index)))
    close = 100.0 + drift * steps + amplitude * cycle
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    volume = volume_base + 100.0 * (1.0 + np.cos(np.linspace(0.0, 4.0 * np.pi, len(index))))
    quote_volume = close * volume
    trades = 120 + (15.0 * (1.0 + np.sin(np.linspace(0.0, 4.0 * np.pi, len(index))))).astype(int)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": quote_volume,
            "trades": trades,
        },
        index=index,
    )


class RegimeObservationRuntimeTest(unittest.TestCase):
    def test_resolve_pipeline_regime_observation_feature_set_prefers_cached_observation_state(self):
        index = pd.date_range("2026-08-01", periods=6, freq="1h", tz="UTC")
        observation_frame = pd.DataFrame(
            {
                "trend_20": np.linspace(-1.0, 1.0, len(index)),
                "vol_20": np.linspace(0.1, 0.6, len(index)),
            },
            index=index,
        )
        legacy_frame = pd.DataFrame({"legacy_only": np.arange(len(index))}, index=index)

        pipeline = ResearchPipeline(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "indicators": [],
                "regime": {"method": "explicit"},
            }
        )
        pipeline.state["regime_observations"] = observation_frame
        pipeline.state["regime_observation_sources"] = {
            "trend_20": "instrument_state",
            "vol_20": "instrument_state",
        }
        pipeline.state["regime_observation_provenance"] = {
            "source_counts": {
                "instrument_state": 2,
                "market_state": 0,
                "cross_asset_state": 0,
            },
            "total_columns": 2,
        }
        pipeline.state["regime_features"] = legacy_frame
        pipeline.state["regime_feature_sources"] = {"legacy_only": "market_state"}

        feature_set = resolve_pipeline_regime_observation_feature_set(pipeline)

        pd.testing.assert_frame_equal(feature_set.frame, observation_frame)
        self.assertEqual(set(feature_set.frame.columns), {"trend_20", "vol_20"})
        self.assertEqual(feature_set.source_map["trend_20"], "instrument_state")

    def test_build_fold_local_regime_observation_feature_set_limits_builder_window_to_fold_boundary(self):
        index = pd.date_range("2026-08-01", periods=72, freq="1h", tz="UTC")
        raw_data = _make_ohlcv(index)
        builder_windows = []

        def builder(scoped_pipeline):
            scoped = scoped_pipeline.require("data")
            builder_windows.append(scoped.index)
            frame = pd.DataFrame(
                {
                    "trend_20": scoped["close"].pct_change(4).fillna(0.0),
                    "vol_20": scoped["close"].pct_change().rolling(5, min_periods=1).std().fillna(0.0),
                },
                index=scoped.index,
            )
            return RegimeFeatureSet(
                frame=frame,
                source_map={
                    "trend_20": "instrument_state",
                    "vol_20": "instrument_state",
                },
            )

        fake_pipeline = types.SimpleNamespace(
            state={"raw_data": raw_data, "data": raw_data.copy()},
            section=lambda key: (
                {"enabled": True, "method": "explicit", "builder": builder, "feature_lookback": 12}
                if key == "regime"
                else {}
            ),
            require=lambda key: raw_data,
        )

        fold_index = raw_data.index[24:48]
        feature_set = build_fold_local_regime_observation_feature_set(fake_pipeline, fold_index)

        self.assertEqual(len(builder_windows), 1)
        self.assertEqual(builder_windows[0][0], raw_data.index[12])
        self.assertEqual(builder_windows[0][-1], fold_index[-1])
        pd.testing.assert_index_equal(feature_set.frame.index, builder_windows[0])


if __name__ == "__main__":
    unittest.main()