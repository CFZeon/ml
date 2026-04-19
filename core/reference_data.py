"""Generic reference-overlay feature adapters for future multi-exchange data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FeatureBlock


def _empty_feature_block(index, block_name):
    return FeatureBlock(frame=pd.DataFrame(index=index), laggable_columns=[], block_name=block_name)


def _safe_divide(numerator, denominator):
    if isinstance(denominator, pd.Series):
        denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def _rolling_zscore(series, window):
    mean = series.rolling(window, min_periods=max(2, window // 2)).mean()
    std = series.rolling(window, min_periods=max(2, window // 2)).std().replace(0, np.nan)
    return (series - mean) / std


def _asof_reindex(base_index, frame):
    if frame is None or frame.empty:
        return pd.DataFrame(index=base_index)

    context = frame.sort_index().reset_index().rename(columns={frame.index.name or "index": "timestamp"})
    anchor = pd.DataFrame({"timestamp": pd.DatetimeIndex(base_index)})
    joined = pd.merge_asof(anchor, context, on="timestamp", direction="backward")
    return joined.set_index("timestamp").reindex(base_index)


def build_reference_overlay_feature_block(base_data, reference_data=None, rolling_window=20):
    base_frame = pd.DataFrame(base_data)
    if reference_data is None:
        return _empty_feature_block(base_frame.index, "reference_overlay")

    reference_frame = pd.DataFrame(reference_data)
    if reference_frame.empty:
        return _empty_feature_block(base_frame.index, "reference_overlay")

    aligned = _asof_reindex(base_frame.index, reference_frame)
    frame = pd.DataFrame(index=base_frame.index)

    reference_price = None
    for column in ["reference_price", "reference_close", "composite_price"]:
        if column in aligned.columns:
            reference_price = aligned[column].astype(float)
            break
    if reference_price is not None and "close" in base_frame.columns:
        base_close = base_frame["close"].astype(float)
        gap = _safe_divide(reference_price - base_close, base_close)
        frame["ref_price_gap"] = gap
        frame["ref_price_gap_z"] = _rolling_zscore(gap, rolling_window)

    if "reference_volume" in aligned.columns and "volume" in base_frame.columns:
        ref_volume = aligned["reference_volume"].astype(float)
        frame["ref_volume_ratio"] = _safe_divide(ref_volume, base_frame["volume"].astype(float))
        frame["ref_volume_ratio_z"] = _rolling_zscore(frame["ref_volume_ratio"], rolling_window)

    if "breadth" in aligned.columns:
        frame["ref_breadth"] = aligned["breadth"].astype(float)
        frame["ref_breadth_z"] = _rolling_zscore(frame["ref_breadth"], rolling_window)

    if "composite_funding_rate" in aligned.columns:
        frame["composite_funding_rate"] = aligned["composite_funding_rate"].astype(float)
        frame["composite_funding_z"] = _rolling_zscore(frame["composite_funding_rate"], rolling_window)

    if "composite_basis" in aligned.columns:
        frame["composite_basis"] = aligned["composite_basis"].astype(float)
        frame["composite_basis_z"] = _rolling_zscore(frame["composite_basis"], rolling_window)

    if frame.empty:
        return _empty_feature_block(base_frame.index, "reference_overlay")

    laggable_columns = list(frame.columns)
    return FeatureBlock(frame=frame, laggable_columns=laggable_columns, block_name="reference_overlay")


__all__ = ["build_reference_overlay_feature_block"]