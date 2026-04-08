"""Feature engineering: fractional differentiation, stationarity, derived features."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


BASE_COLUMNS = {"open", "high", "low", "close", "volume", "quote_volume", "trades"}


@dataclass
class FeatureBlock:
    frame: pd.DataFrame
    laggable_columns: list[str]


# ---------------------------------------------------------------------------
# Fractional differentiation
# ---------------------------------------------------------------------------

def fractional_diff(series, d, threshold=1e-5):
    """Fractionally differentiate *series* by order *d*.

    Uses expanding-window weights (cut off at *threshold*) applied as a
    convolution. Preserves long-memory information that integer differencing
    destroys.

    Returns a pd.Series (leading values are NaN until the weight window is
    filled).
    """
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1

    weights = np.array(weights[::-1])
    width = len(weights)
    values = series.values.astype(float)
    out = np.full(len(values), np.nan)

    for i in range(width - 1, len(values)):
        out[i] = weights @ values[i - width + 1: i + 1]

    return pd.Series(out, index=series.index, name="close_fracdiff")


# ---------------------------------------------------------------------------
# Stationarity check
# ---------------------------------------------------------------------------

def check_stationarity(series, significance=0.05):
    """Run Augmented Dickey-Fuller test.

    Returns dict with keys: stationary (bool), p_value, adf_stat.
    """
    clean = series.dropna()
    if len(clean) < 20:
        return {"stationary": False, "p_value": 1.0, "adf_stat": 0.0}
    result = adfuller(clean, maxlag=min(20, len(clean) // 4))
    return {
        "stationary": bool(result[1] < significance),
        "p_value": round(float(result[1]), 6),
        "adf_stat": round(float(result[0]), 4),
    }


def _safe_divide(numerator, denominator):
    if isinstance(denominator, pd.Series):
        denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def _rolling_zscore(series, window):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return (series - mean) / std


def _cross_up(series, threshold=0.0):
    return ((series > threshold) & (series.shift(1) <= threshold)).astype(float)


def _cross_down(series, threshold=0.0):
    return ((series < threshold) & (series.shift(1) >= threshold)).astype(float)


def _as_feature_block(frame, laggable_columns):
    laggable = [column for column in laggable_columns if column in frame.columns]
    return FeatureBlock(frame=frame, laggable_columns=laggable)


def _price_volume_features(df, rolling_window):
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    intrabar_range = high - low
    close_location = _safe_divide(close - low, intrabar_range)

    frame = pd.DataFrame(index=df.index)
    frame["return_1"] = close.pct_change(1)
    frame["return_5"] = close.pct_change(5)
    frame["return_10"] = close.pct_change(10)
    frame["log_return"] = np.log(close / close.shift(1))
    frame["intrabar_return"] = _safe_divide(close - open_, open_)
    frame["range_pct"] = _safe_divide(intrabar_range, close)
    frame["close_to_range_mid"] = close_location - 0.5
    frame["vol_change"] = volume.pct_change(1)
    frame["vol_zscore"] = _rolling_zscore(volume, rolling_window)

    if "quote_volume" in df.columns:
        frame["quote_vol_change"] = df["quote_volume"].astype(float).pct_change(1)
        frame["quote_vol_zscore"] = _rolling_zscore(df["quote_volume"].astype(float), rolling_window)
    if "trades" in df.columns:
        frame["trades_change"] = df["trades"].astype(float).pct_change(1)
        frame["trades_zscore"] = _rolling_zscore(df["trades"].astype(float), rolling_window)

    laggable = [
        "return_1",
        "return_5",
        "return_10",
        "log_return",
        "intrabar_return",
        "range_pct",
        "close_to_range_mid",
        "vol_change",
        "vol_zscore",
        "quote_vol_change",
        "quote_vol_zscore",
        "trades_change",
        "trades_zscore",
    ]
    return _as_feature_block(frame, laggable)


def _extract_rsi_features(result, df, rolling_window=20, **_):
    series = df[result.name].astype(float)
    centered = (series - 50.0) / 50.0
    overbought_dist = (series - 70.0).clip(lower=0.0) / 30.0
    oversold_dist = (30.0 - series).clip(lower=0.0) / 30.0

    frame = pd.DataFrame(
        {
            f"{result.name}_centered": centered,
            f"{result.name}_slope": series.diff() / 100.0,
            f"{result.name}_accel": series.diff().diff() / 100.0,
            f"{result.name}_bull_range": (series >= 60.0).astype(float),
            f"{result.name}_bear_range": (series <= 40.0).astype(float),
            f"{result.name}_overbought": (series >= 70.0).astype(float),
            f"{result.name}_oversold": (series <= 30.0).astype(float),
            f"{result.name}_overbought_dist": overbought_dist,
            f"{result.name}_oversold_dist": oversold_dist,
            f"{result.name}_cross_up_50": _cross_up(series, 50.0),
            f"{result.name}_cross_down_50": _cross_down(series, 50.0),
            f"{result.name}_zscore": _rolling_zscore(centered, rolling_window),
        },
        index=df.index,
    )
    laggable = [
        f"{result.name}_centered",
        f"{result.name}_slope",
        f"{result.name}_accel",
        f"{result.name}_overbought_dist",
        f"{result.name}_oversold_dist",
        f"{result.name}_zscore",
    ]
    return _as_feature_block(frame, laggable)


def _extract_macd_features(result, df, rolling_window=20, **_):
    line = df[f"{result.name}_line"].astype(float)
    signal = df[f"{result.name}_signal"].astype(float)
    hist = df[f"{result.name}_hist"].astype(float)
    close = df["close"].astype(float)

    line_pct = _safe_divide(line, close)
    signal_pct = _safe_divide(signal, close)
    hist_pct = _safe_divide(hist, close)

    frame = pd.DataFrame(
        {
            f"{result.name}_line_pct": line_pct,
            f"{result.name}_signal_pct": signal_pct,
            f"{result.name}_hist_pct": hist_pct,
            f"{result.name}_line_slope": line_pct.diff(),
            f"{result.name}_hist_slope": hist_pct.diff(),
            f"{result.name}_hist_zscore": _rolling_zscore(hist_pct, rolling_window),
            f"{result.name}_above_signal": (hist > 0).astype(float),
            f"{result.name}_above_zero": (line > 0).astype(float),
            f"{result.name}_cross_up_signal": _cross_up(hist, 0.0),
            f"{result.name}_cross_down_signal": _cross_down(hist, 0.0),
            f"{result.name}_cross_up_zero": _cross_up(line, 0.0),
            f"{result.name}_cross_down_zero": _cross_down(line, 0.0),
        },
        index=df.index,
    )
    laggable = [
        f"{result.name}_line_pct",
        f"{result.name}_signal_pct",
        f"{result.name}_hist_pct",
        f"{result.name}_line_slope",
        f"{result.name}_hist_slope",
        f"{result.name}_hist_zscore",
    ]
    return _as_feature_block(frame, laggable)


def _extract_bollinger_features(result, df, rolling_window=20, squeeze_quantile=0.2, **_):
    upper = df[f"{result.name}_upper"].astype(float)
    lower = df[f"{result.name}_lower"].astype(float)
    pctb = df[f"{result.name}_pctb"].astype(float)
    bandwidth = df[f"{result.name}_bw"].astype(float)
    close = df["close"].astype(float)
    band_width_abs = upper - lower
    squeeze_threshold = bandwidth.rolling(rolling_window).quantile(squeeze_quantile)

    frame = pd.DataFrame(
        {
            f"{result.name}_pctb_centered": pctb - 0.5,
            f"{result.name}_pctb_slope": pctb.diff(),
            f"{result.name}_bw": bandwidth,
            f"{result.name}_bw_zscore": _rolling_zscore(bandwidth, rolling_window),
            f"{result.name}_bw_change": bandwidth.diff(),
            f"{result.name}_squeeze": (bandwidth <= squeeze_threshold).astype(float),
            f"{result.name}_break_above_upper": (close > upper).astype(float),
            f"{result.name}_break_below_lower": (close < lower).astype(float),
            f"{result.name}_dist_upper_band": _safe_divide(upper - close, band_width_abs),
            f"{result.name}_dist_lower_band": _safe_divide(close - lower, band_width_abs),
            f"{result.name}_band_walk_up": (pctb >= 0.8).astype(float),
            f"{result.name}_band_walk_down": (pctb <= 0.2).astype(float),
        },
        index=df.index,
    )
    laggable = [
        f"{result.name}_pctb_centered",
        f"{result.name}_pctb_slope",
        f"{result.name}_bw",
        f"{result.name}_bw_zscore",
        f"{result.name}_bw_change",
        f"{result.name}_dist_upper_band",
        f"{result.name}_dist_lower_band",
    ]
    return _as_feature_block(frame, laggable)


def _extract_atr_features(result, df, rolling_window=20, **_):
    atr = df[result.name].astype(float)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    atr_pct = _safe_divide(atr, close)
    atr_pct_zscore = _rolling_zscore(atr_pct, rolling_window)

    frame = pd.DataFrame(
        {
            f"{result.name}_pct": atr_pct,
            f"{result.name}_pct_zscore": atr_pct_zscore,
            f"{result.name}_pct_change": atr_pct.diff(),
            f"{result.name}_range_to_atr": _safe_divide(high - low, atr),
            f"{result.name}_abs_return_to_atr": _safe_divide(close.diff().abs(), atr),
            f"{result.name}_high_vol_regime": (atr_pct_zscore >= 1.0).astype(float),
            f"{result.name}_low_vol_regime": (atr_pct_zscore <= -1.0).astype(float),
        },
        index=df.index,
    )
    laggable = [
        f"{result.name}_pct",
        f"{result.name}_pct_zscore",
        f"{result.name}_pct_change",
        f"{result.name}_range_to_atr",
        f"{result.name}_abs_return_to_atr",
    ]
    return _as_feature_block(frame, laggable)


def _extract_fvg_features(result, df, **_):
    prefix = result.name
    bull_size_pct = df[f"{prefix}_bull_size_pct"].astype(float).fillna(0.0)
    bull_age = df[f"{prefix}_bull_age"].astype(float).fillna(0.0)
    bull_distance = df[f"{prefix}_bull_distance_pct"].astype(float).fillna(0.0)
    bull_fill_state = df[f"{prefix}_bull_fill_state"].astype(float).fillna(0.0)
    bull_active_count = df[f"{prefix}_bull_active_count"].astype(float).fillna(0.0)

    bear_size_pct = df[f"{prefix}_bear_size_pct"].astype(float).fillna(0.0)
    bear_age = df[f"{prefix}_bear_age"].astype(float).fillna(0.0)
    bear_distance = df[f"{prefix}_bear_distance_pct"].astype(float).fillna(0.0)
    bear_fill_state = df[f"{prefix}_bear_fill_state"].astype(float).fillna(0.0)
    bear_active_count = df[f"{prefix}_bear_active_count"].astype(float).fillna(0.0)

    bull_open = (bull_fill_state > 0.0).astype(float)
    bear_open = (bear_fill_state > 0.0).astype(float)

    frame = pd.DataFrame(
        {
            f"{prefix}_bull_open": bull_open,
            f"{prefix}_bear_open": bear_open,
            f"{prefix}_bull_size_pct": bull_size_pct,
            f"{prefix}_bear_size_pct": bear_size_pct,
            f"{prefix}_bull_distance_pct": bull_distance,
            f"{prefix}_bear_distance_pct": bear_distance,
            f"{prefix}_bull_age_log": np.log1p(bull_age),
            f"{prefix}_bear_age_log": np.log1p(bear_age),
            f"{prefix}_bull_fill_state": bull_fill_state / 2.0,
            f"{prefix}_bear_fill_state": bear_fill_state / 2.0,
            f"{prefix}_bull_active_count": bull_active_count,
            f"{prefix}_bear_active_count": bear_active_count,
            f"{prefix}_gap_imbalance": bull_active_count - bear_active_count,
            f"{prefix}_fill_imbalance": bull_fill_state - bear_fill_state,
            f"{prefix}_distance_spread": bull_distance - bear_distance,
            f"{prefix}_size_spread": bull_size_pct - bear_size_pct,
            f"{prefix}_bull_new_gap": ((bull_open > 0.0) & (bull_age == 0.0)).astype(float),
            f"{prefix}_bear_new_gap": ((bear_open > 0.0) & (bear_age == 0.0)).astype(float),
            f"{prefix}_bull_partial_fill": (bull_fill_state == 2.0).astype(float),
            f"{prefix}_bear_partial_fill": (bear_fill_state == 2.0).astype(float),
            f"{prefix}_any_gap_open": ((bull_open + bear_open) > 0.0).astype(float),
        },
        index=df.index,
    )
    laggable = [
        f"{prefix}_bull_size_pct",
        f"{prefix}_bear_size_pct",
        f"{prefix}_bull_distance_pct",
        f"{prefix}_bear_distance_pct",
        f"{prefix}_bull_age_log",
        f"{prefix}_bear_age_log",
        f"{prefix}_bull_active_count",
        f"{prefix}_bear_active_count",
        f"{prefix}_gap_imbalance",
        f"{prefix}_fill_imbalance",
        f"{prefix}_distance_spread",
        f"{prefix}_size_spread",
    ]
    return _as_feature_block(frame, laggable)


def _extract_generic_indicator_features(result, df, rolling_window=20, **_):
    frame = pd.DataFrame(index=df.index)
    laggable = []

    for column in result.metadata.get("output_columns", []):
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            continue

        series = df[column].astype(float)
        diff_name = f"{column}_diff"
        zscore_name = f"{column}_zscore"
        frame[diff_name] = series.diff()
        frame[zscore_name] = _rolling_zscore(series, rolling_window)
        laggable.extend([diff_name, zscore_name])

    return _as_feature_block(frame, laggable)


INDICATOR_FEATURE_EXTRACTORS = {
    "rsi": _extract_rsi_features,
    "macd": _extract_macd_features,
    "bollinger": _extract_bollinger_features,
    "atr": _extract_atr_features,
    "fvg": _extract_fvg_features,
}


def _append_lags(features, laggable_columns, lags):
    if not lags:
        return features

    laggable_columns = [column for column in dict.fromkeys(laggable_columns) if column in features.columns]
    lagged_columns = {}
    for column in laggable_columns:
        for lag in lags:
            lagged_columns[f"{column}_lag{lag}"] = features[column].shift(lag)

    if not lagged_columns:
        return features

    return pd.concat([features, pd.DataFrame(lagged_columns, index=features.index)], axis=1)


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

def build_features(
    df,
    lags=None,
    frac_diff_d=None,
    indicator_run=None,
    rolling_window=20,
    squeeze_quantile=0.2,
):
    """Derive a feature matrix from an indicator-enriched OHLCV DataFrame.

    The default path is indicator-aware. Known indicators emit normalized state,
    momentum, crossover, and regime features instead of dumping every numeric
    indicator output into the model. Unknown indicators fall back to simple
    diff/z-score features.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV + indicator columns.
    lags : list[int] | None
        Optional lags applied only to selected continuous features.
    frac_diff_d : float | None
        Fractional-diff order for close.
    indicator_run : IndicatorRunResult | None
        Optional indicator metadata/results produced by run_indicators().
    rolling_window : int
        Rolling window for z-scores and squeeze detection.
    squeeze_quantile : float
        Rolling BandWidth quantile used to define a Bollinger squeeze.

    Returns
    -------
    pd.DataFrame
        Feature matrix aligned to *df* index.
    """
    base_block = _price_volume_features(df, rolling_window=rolling_window)
    features = base_block.frame.copy()
    laggable_columns = list(base_block.laggable_columns)

    if indicator_run is not None and getattr(indicator_run, "results", None):
        for result in indicator_run.results:
            extractor = INDICATOR_FEATURE_EXTRACTORS.get(result.kind, _extract_generic_indicator_features)
            block = extractor(
                result,
                df,
                rolling_window=rolling_window,
                squeeze_quantile=squeeze_quantile,
            )
            if not block.frame.empty:
                features = features.join(block.frame)
                laggable_columns.extend(block.laggable_columns)
    else:
        generic_result = type(
            "GenericIndicatorResult",
            (),
            {
                "metadata": {
                    "output_columns": [
                        column for column in df.columns
                        if column not in BASE_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
                    ]
                }
            },
        )()
        block = _extract_generic_indicator_features(
            generic_result,
            df,
            rolling_window=rolling_window,
        )
        if not block.frame.empty:
            features = features.join(block.frame)
            laggable_columns.extend(block.laggable_columns)

    if frac_diff_d is not None:
        features["close_fracdiff"] = fractional_diff(df["close"], d=frac_diff_d)
        laggable_columns.append("close_fracdiff")

    features = _append_lags(features, laggable_columns=laggable_columns, lags=lags)
    return features
