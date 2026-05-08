"""Canonical regime observation builders for preview and fold-local replay."""

from __future__ import annotations

import pandas as pd

from ..regime import build_default_regime_feature_set, normalize_regime_feature_set


def _resolve_reference_overlay_data(pipeline):
    reference_overlay = pipeline.state.get("reference_overlay_data")
    if reference_overlay is not None:
        return reference_overlay
    return pipeline.state.get("reference_data")


class _ObservationScopedPipeline:
    _DATA_KEYS = frozenset({"raw_data", "data"})

    def __init__(self, pipeline, windowed_data):
        self._pipeline = pipeline
        self._windowed_data = windowed_data

    def section(self, key):
        return self._pipeline.section(key)

    def require(self, key):
        if key in self._DATA_KEYS:
            return self._windowed_data
        return self._pipeline.require(key)

    @property
    def state(self):
        return self._pipeline.state


def build_default_regime_observation_feature_set(pipeline):
    data = pipeline.require("data")
    features_config = pipeline.section("features")
    return build_default_regime_feature_set(
        data,
        base_interval=pipeline.section("data").get("interval", "1h"),
        rolling_window=features_config.get("rolling_window", 20),
        futures_context=pipeline.state.get("futures_context"),
        cross_asset_context=pipeline.state.get("cross_asset_context"),
        reference_data=_resolve_reference_overlay_data(pipeline),
        context_timeframes=features_config.get("context_timeframes"),
    )


def resolve_pipeline_regime_observation_feature_set(pipeline):
    observation_frame = pipeline.state.get("regime_observations")
    if observation_frame is None:
        observation_frame = pipeline.state.get("regime_features")

    if observation_frame is not None:
        return normalize_regime_feature_set(
            {
                "frame": observation_frame,
                "source_map": (
                    pipeline.state.get("regime_observation_sources")
                    or pipeline.state.get("regime_feature_sources")
                ),
                "provenance": (
                    pipeline.state.get("regime_observation_provenance")
                    or pipeline.state.get("regime_provenance")
                ),
            }
        )

    config = pipeline.section("regime")
    builder = config.get("builder") or build_default_regime_observation_feature_set
    return normalize_regime_feature_set(builder(pipeline))


def build_fold_local_regime_observation_feature_set(pipeline, index):
    if index is None or len(index) == 0:
        empty_index = pd.Index([]) if index is None else index
        return normalize_regime_feature_set(pd.DataFrame(index=empty_index))

    config = pipeline.section("regime")
    raw_data = pipeline.state.get("raw_data")
    if raw_data is None:
        raw_data = pipeline.state.get("data")

    lookback = int(config.get("feature_lookback", 80))
    if raw_data is not None and hasattr(raw_data, "index"):
        fold_start_pos = raw_data.index.searchsorted(index[0])
        fold_end_pos = raw_data.index.searchsorted(index[-1], side="right")
        buffer_start = max(0, fold_start_pos - lookback)
        buffered_data = raw_data.iloc[buffer_start:fold_end_pos]
    else:
        buffered_data = raw_data

    builder = config.get("builder") or build_default_regime_observation_feature_set
    scoped_pipeline = _ObservationScopedPipeline(pipeline, buffered_data)
    return normalize_regime_feature_set(builder(scoped_pipeline))


__all__ = [
    "build_default_regime_observation_feature_set",
    "build_fold_local_regime_observation_feature_set",
    "resolve_pipeline_regime_observation_feature_set",
]