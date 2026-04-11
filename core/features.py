"""Feature engineering: fractional differentiation, stationarity, derived features."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


BASE_COLUMNS = {"open", "high", "low", "close", "volume", "quote_volume", "trades"}
DEFAULT_STATIONARITY_TRANSFORM_ORDER = ("log_diff", "pct_change", "diff", "zscore", "frac_diff")


@dataclass
class FeatureBlock:
    frame: pd.DataFrame
    laggable_columns: list[str]
    block_name: str


@dataclass
class BuiltFeatureSet:
    frame: pd.DataFrame
    feature_blocks: dict[str, str]


@dataclass
class FeatureScreeningResult:
    frame: pd.DataFrame
    feature_blocks: dict[str, str]
    report: dict


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

    series_name = getattr(series, "name", None)
    if series_name:
        output_name = f"{series_name}_fracdiff"
    else:
        output_name = "fracdiff"
    return pd.Series(out, index=series.index, name=output_name)


# ---------------------------------------------------------------------------
# Stationarity check
# ---------------------------------------------------------------------------

def check_stationarity(series, significance=0.05):
    """Run Augmented Dickey-Fuller test.

    Returns dict with keys: stationary (bool), p_value, adf_stat.
    """
    clean = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    unique_values = int(clean.nunique())
    if len(clean) < 20:
        return {
            "stationary": False,
            "p_value": 1.0,
            "adf_stat": 0.0,
            "observations": int(len(clean)),
            "unique_values": unique_values,
            "error": "too_short",
        }
    if unique_values <= 1:
        return {
            "stationary": False,
            "p_value": 1.0,
            "adf_stat": 0.0,
            "observations": int(len(clean)),
            "unique_values": unique_values,
            "error": "constant",
        }

    try:
        result = adfuller(clean, maxlag=min(20, len(clean) // 4))
    except Exception as exc:
        return {
            "stationary": False,
            "p_value": 1.0,
            "adf_stat": 0.0,
            "observations": int(len(clean)),
            "unique_values": unique_values,
            "error": type(exc).__name__,
        }

    return {
        "stationary": bool(result[1] < significance),
        "p_value": round(float(result[1]), 6),
        "adf_stat": round(float(result[0]), 4),
        "observations": int(len(clean)),
        "unique_values": unique_values,
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


def _as_feature_block(frame, laggable_columns, block_name):
    laggable = [column for column in laggable_columns if column in frame.columns]
    return FeatureBlock(frame=frame, laggable_columns=laggable, block_name=block_name)


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
    return _as_feature_block(frame, laggable, block_name="price_volume")


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
    return _as_feature_block(frame, laggable, block_name=result.kind)


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
    return _as_feature_block(frame, laggable, block_name=result.kind)


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
    return _as_feature_block(frame, laggable, block_name=result.kind)


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
    return _as_feature_block(frame, laggable, block_name=result.kind)


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
    return _as_feature_block(frame, laggable, block_name=result.kind)


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

    return _as_feature_block(frame, laggable, block_name=getattr(result, "kind", "generic_indicator"))


INDICATOR_FEATURE_EXTRACTORS = {
    "rsi": _extract_rsi_features,
    "macd": _extract_macd_features,
    "bollinger": _extract_bollinger_features,
    "atr": _extract_atr_features,
    "fvg": _extract_fvg_features,
}


def _indicator_interaction_features(df, indicator_run, rolling_window=20):
    if indicator_run is None or not getattr(indicator_run, "results", None):
        return _as_feature_block(pd.DataFrame(index=df.index), [], block_name="indicator_interactions")

    results_by_kind = {}
    for result in indicator_run.results:
        results_by_kind.setdefault(result.kind, []).append(result)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    frame = pd.DataFrame(index=df.index)
    laggable = []

    rsi_result = results_by_kind.get("rsi", [None])[0]
    macd_result = results_by_kind.get("macd", [None])[0]
    bollinger_result = results_by_kind.get("bollinger", [None])[0]
    atr_result = results_by_kind.get("atr", [None])[0]
    fvg_result = results_by_kind.get("fvg", [None])[0]

    rsi = None
    line_pct = None
    hist_pct = None
    pctb = None
    bandwidth = None
    atr_pct = None

    if rsi_result is not None:
        rsi = df[rsi_result.name].astype(float)
    if macd_result is not None:
        line_pct = _safe_divide(df[f"{macd_result.name}_line"].astype(float), close)
        hist_pct = _safe_divide(df[f"{macd_result.name}_hist"].astype(float), close)
    if bollinger_result is not None:
        pctb = df[f"{bollinger_result.name}_pctb"].astype(float)
        bandwidth = df[f"{bollinger_result.name}_bw"].astype(float)
    if atr_result is not None:
        atr_pct = _safe_divide(df[atr_result.name].astype(float), close)

    if rsi is not None and line_pct is not None and hist_pct is not None:
        trend_up = (line_pct > 0) & (hist_pct > 0)
        trend_down = (line_pct < 0) & (hist_pct < 0)
        frame["trend_rsi_alignment"] = np.sign(line_pct.fillna(0.0)) * ((rsi - 50.0) / 50.0)
        frame["bull_pullback_setup"] = (trend_up & rsi.between(40.0, 55.0)).astype(float)
        frame["bear_rebound_setup"] = (trend_down & rsi.between(45.0, 60.0)).astype(float)
        frame["momentum_exhaustion_score"] = hist_pct * ((rsi - 50.0) / 50.0)
        laggable.extend(["trend_rsi_alignment", "momentum_exhaustion_score"])

    if rsi is not None and pctb is not None and bandwidth is not None:
        squeeze_threshold = bandwidth.rolling(rolling_window).quantile(0.2)
        squeeze = bandwidth <= squeeze_threshold
        frame["squeeze_rsi_breakout_bias"] = (pctb - 0.5) * ((rsi - 50.0) / 50.0)
        frame["squeeze_oversold_long"] = (squeeze & (pctb < 0.2) & (rsi < 35.0)).astype(float)
        frame["squeeze_overbought_short"] = (squeeze & (pctb > 0.8) & (rsi > 65.0)).astype(float)
        laggable.append("squeeze_rsi_breakout_bias")

    if atr_pct is not None:
        atr_pct_safe = atr_pct.replace(0.0, np.nan)
        frame["return_1_to_atr"] = close.pct_change(1) / atr_pct_safe
        frame["return_5_to_atr"] = close.pct_change(5) / atr_pct_safe
        frame["range_to_atr_interaction"] = _safe_divide(high - low, close * atr_pct_safe)
        laggable.extend(["return_1_to_atr", "return_5_to_atr", "range_to_atr_interaction"])

    if atr_pct is not None and line_pct is not None:
        atr_pct_safe = atr_pct.replace(0.0, np.nan)
        frame["trend_strength_to_atr"] = _safe_divide(line_pct.abs(), atr_pct_safe)
        frame["hist_impulse_to_atr"] = _safe_divide(hist_pct, atr_pct_safe)
        laggable.extend(["trend_strength_to_atr", "hist_impulse_to_atr"])

    if fvg_result is not None:
        prefix = fvg_result.name
        bull_open = df[f"{prefix}_bull_fill_state"].astype(float).fillna(0.0) > 0.0
        bear_open = df[f"{prefix}_bear_fill_state"].astype(float).fillna(0.0) > 0.0
        gap_imbalance = (
            df[f"{prefix}_bull_active_count"].astype(float).fillna(0.0)
            - df[f"{prefix}_bear_active_count"].astype(float).fillna(0.0)
        )
        frame["fvg_gap_pressure"] = gap_imbalance
        laggable.append("fvg_gap_pressure")

        if line_pct is not None and hist_pct is not None:
            trend_up = (line_pct > 0) & (hist_pct > 0)
            trend_down = (line_pct < 0) & (hist_pct < 0)
            frame["fvg_bull_trend_confluence"] = (bull_open & trend_up).astype(float)
            frame["fvg_bear_trend_confluence"] = (bear_open & trend_down).astype(float)
            frame["fvg_momentum_alignment"] = gap_imbalance * hist_pct.fillna(0.0)
            laggable.append("fvg_momentum_alignment")

        if atr_pct is not None:
            atr_pct_safe = atr_pct.replace(0.0, np.nan)
            size_spread = (
                df[f"{prefix}_bull_size_pct"].astype(float).fillna(0.0)
                - df[f"{prefix}_bear_size_pct"].astype(float).fillna(0.0)
            )
            frame["fvg_size_to_atr"] = _safe_divide(size_spread, atr_pct_safe)
            laggable.append("fvg_size_to_atr")

    return _as_feature_block(frame, laggable, block_name="indicator_interactions")


def _append_lags(features, laggable_columns, lags, feature_blocks):
    if not lags:
        return features, dict(feature_blocks)

    laggable_columns = [column for column in dict.fromkeys(laggable_columns) if column in features.columns]
    lagged_columns = {}
    updated_blocks = dict(feature_blocks)
    for column in laggable_columns:
        for lag in lags:
            lagged_name = f"{column}_lag{lag}"
            lagged_columns[lagged_name] = features[column].shift(lag)
            updated_blocks[lagged_name] = feature_blocks.get(column, "unknown")

    if not lagged_columns:
        return features, updated_blocks

    lagged_frame = pd.DataFrame(lagged_columns, index=features.index)
    return pd.concat([features, lagged_frame], axis=1), updated_blocks


def _apply_stationarity_transform(series, transform_name, rolling_window, frac_diff_d, frac_diff_threshold):
    clean = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return None

    if transform_name == "log_diff":
        if (clean <= 0).any():
            return None
        return np.log(series).diff()

    if transform_name == "pct_change":
        if clean.abs().min() < 1e-12:
            return None
        return series.pct_change()

    if transform_name == "diff":
        return series.diff()

    if transform_name == "zscore":
        return _rolling_zscore(series, rolling_window)

    if transform_name == "frac_diff":
        if frac_diff_d is None:
            return None
        return fractional_diff(series, d=frac_diff_d, threshold=frac_diff_threshold)

    raise ValueError(f"Unsupported stationarity transform {transform_name!r}")


def screen_features_for_stationarity(features, feature_blocks=None, config=None, fit_features=None):
    """Screen each feature for stationarity and transform or drop when needed.

    When ``fit_features`` is supplied, transform selection is fitted on that
    reference slice and then applied causally to ``features``. This is useful
    for walk-forward validation where transform choice must be based on the
    training window only.
    """
    config = dict(config or {})
    reference_features = features if fit_features is None else fit_features.reindex(columns=features.columns)
    if not config.get("enabled", True):
        blocks = dict(feature_blocks or {})
        summary = {
            "enabled": False,
            "total_features": int(features.shape[1]),
            "screened_feature_count": int(features.shape[1]),
            "stationary_as_is": 0,
            "discrete_passthrough": 0,
            "transformed_features": 0,
            "dropped_features": 0,
            "dropped_constant": 0,
            "dropped_unrepaired": 0,
            "transform_usage": {},
            "block_feature_counts": {},
        }
        return FeatureScreeningResult(
            frame=features.copy(),
            feature_blocks=blocks,
            report={"summary": summary, "features": {}, "transformed_features": [], "dropped_features": []},
        )

    significance = config.get("significance", 0.05)
    transform_order = tuple(config.get("transform_order") or DEFAULT_STATIONARITY_TRANSFORM_ORDER)
    rolling_window = int(config.get("rolling_window", 20))
    frac_diff_d = config.get("frac_diff_d")
    frac_diff_threshold = float(config.get("frac_diff_threshold", 1e-5))
    discrete_max_unique = int(config.get("discrete_max_unique", 6))
    drop_failed = bool(config.get("drop_failed", True))

    screened_columns = {}
    kept_blocks = {}
    reports = {}
    transform_usage = {}
    transformed_features = []
    dropped_features = []

    summary = {
        "enabled": True,
        "total_features": int(features.shape[1]),
        "screened_feature_count": 0,
        "stationary_as_is": 0,
        "discrete_passthrough": 0,
        "transformed_features": 0,
        "dropped_features": 0,
        "dropped_constant": 0,
        "dropped_unrepaired": 0,
        "transform_usage": {},
        "block_feature_counts": {},
    }

    feature_blocks = dict(feature_blocks or {})

    for column in features.columns:
        series = pd.Series(features[column]).replace([np.inf, -np.inf], np.nan)
        reference_series = pd.Series(reference_features[column]).replace([np.inf, -np.inf], np.nan)
        clean = reference_series.dropna()
        unique_values = int(clean.nunique())
        block_name = feature_blocks.get(column, "unknown")
        report = {
            "block": block_name,
            "status": None,
            "selected_transform": None,
            "original": None,
            "final": None,
            "candidates": [],
        }

        if unique_values <= 1:
            report["status"] = "dropped_constant"
            report["original"] = {
                "stationary": False,
                "p_value": 1.0,
                "adf_stat": 0.0,
                "observations": int(len(clean)),
                "unique_values": unique_values,
                "error": "constant",
            }
            report["final"] = report["original"]
            reports[column] = report
            dropped_features.append(column)
            summary["dropped_features"] += 1
            summary["dropped_constant"] += 1
            continue

        if unique_values <= discrete_max_unique:
            screened_columns[column] = series
            kept_blocks[column] = block_name
            report["status"] = "discrete_passthrough"
            report["selected_transform"] = "passthrough"
            report["final"] = {
                "stationary": True,
                "p_value": 0.0,
                "adf_stat": 0.0,
                "observations": int(len(clean)),
                "unique_values": unique_values,
                "note": "discrete_passthrough",
            }
            reports[column] = report
            summary["discrete_passthrough"] += 1
            continue

        original = check_stationarity(reference_series, significance=significance)
        report["original"] = original
        if original["stationary"]:
            screened_columns[column] = series
            kept_blocks[column] = block_name
            report["status"] = "stationary"
            report["selected_transform"] = "passthrough"
            report["final"] = original
            reports[column] = report
            summary["stationary_as_is"] += 1
            continue

        selected_series = None
        selected_stats = None
        selected_transform = None

        for transform_name in transform_order:
            transformed = _apply_stationarity_transform(
                reference_series,
                transform_name=transform_name,
                rolling_window=rolling_window,
                frac_diff_d=frac_diff_d,
                frac_diff_threshold=frac_diff_threshold,
            )
            if transformed is None:
                report["candidates"].append({"transform": transform_name, "applicable": False})
                continue

            stats = check_stationarity(transformed, significance=significance)
            report["candidates"].append(
                {
                    "transform": transform_name,
                    "applicable": True,
                    "result": stats,
                }
            )
            if stats["stationary"]:
                selected_series = transformed
                selected_stats = stats
                selected_transform = transform_name
                break

        if selected_series is not None:
            screened_columns[column] = _apply_stationarity_transform(
                series,
                transform_name=selected_transform,
                rolling_window=rolling_window,
                frac_diff_d=frac_diff_d,
                frac_diff_threshold=frac_diff_threshold,
            )
            kept_blocks[column] = block_name
            report["status"] = "transformed"
            report["selected_transform"] = selected_transform
            report["final"] = selected_stats
            reports[column] = report
            transformed_features.append(column)
            summary["transformed_features"] += 1
            transform_usage[selected_transform] = transform_usage.get(selected_transform, 0) + 1
            continue

        report["status"] = "failed_kept" if not drop_failed else "dropped_unrepaired"
        report["selected_transform"] = None
        report["final"] = report["candidates"][-1]["result"] if report["candidates"] else original
        reports[column] = report

        if drop_failed:
            dropped_features.append(column)
            summary["dropped_features"] += 1
            summary["dropped_unrepaired"] += 1
            continue

        screened_columns[column] = series
        kept_blocks[column] = block_name

    screened = pd.DataFrame(screened_columns, index=features.index)
    summary["screened_feature_count"] = int(screened.shape[1])
    summary["transform_usage"] = dict(sorted(transform_usage.items()))
    block_counts = pd.Series(list(kept_blocks.values()), dtype="object")
    if not block_counts.empty:
        summary["block_feature_counts"] = {
            block: int(count)
            for block, count in block_counts.value_counts().sort_index().items()
        }

    report = {
        "summary": summary,
        "features": reports,
        "transformed_features": transformed_features,
        "dropped_features": dropped_features,
    }
    return FeatureScreeningResult(frame=screened, feature_blocks=kept_blocks, report=report)


# ---------------------------------------------------------------------------
# Feature selection (mutual information)
# ---------------------------------------------------------------------------

@dataclass
class FeatureSelectionResult:
    frame: pd.DataFrame
    feature_blocks: dict[str, str]
    report: dict


def select_features(features, y, feature_blocks=None, config=None):
    """Select top features by mutual information with the target.

    Parameters
    ----------
    features : pd.DataFrame - feature matrix
    y : pd.Series - target labels (aligned index)
    feature_blocks : dict - column -> block name mapping
    config : dict - selection config with keys:
        enabled (bool, default True)
        max_features (int or None) - hard cap on features to keep
        min_mi_threshold (float, default 0.0) - drop features below this MI score

    Returns FeatureSelectionResult with selected frame, blocks, and report.
    """
    from sklearn.feature_selection import mutual_info_classif

    config = dict(config or {})
    feature_blocks = dict(feature_blocks or {})

    if not config.get("enabled", True):
        return FeatureSelectionResult(
            frame=features.copy(),
            feature_blocks=dict(feature_blocks),
            report={"enabled": False, "input_features": features.shape[1], "selected_features": features.shape[1]},
        )

    common = features.index.intersection(y.index)
    X_aligned = features.loc[common].copy()
    y_aligned = y.loc[common].copy()

    clean_mask = X_aligned.notna().all(axis=1) & y_aligned.notna()
    X_clean = X_aligned.loc[clean_mask]
    y_clean = y_aligned.loc[clean_mask]

    if X_clean.empty or len(X_clean) < 50:
        return FeatureSelectionResult(
            frame=features.copy(),
            feature_blocks=dict(feature_blocks),
            report={"enabled": True, "input_features": features.shape[1], "selected_features": features.shape[1],
                    "error": "insufficient_clean_rows"},
        )

    X_filled = X_clean.fillna(0.0)
    mi_scores = mutual_info_classif(X_filled, y_clean, random_state=42, n_neighbors=5)
    mi_series = pd.Series(mi_scores, index=features.columns).sort_values(ascending=False)

    max_features = config.get("max_features")
    if max_features is None:
        n_samples = len(X_clean)
        max_features = max(10, min(n_samples // 10, features.shape[1]))

    min_mi = config.get("min_mi_threshold", 0.0)
    above_threshold = mi_series[mi_series > min_mi]
    selected_columns = list(above_threshold.head(max_features).index)

    if not selected_columns:
        selected_columns = list(mi_series.head(max(5, max_features // 2)).index)

    selected_blocks = {col: feature_blocks.get(col, "unknown") for col in selected_columns}
    selected_frame = features[selected_columns].copy()

    report = {
        "enabled": True,
        "input_features": int(features.shape[1]),
        "selected_features": len(selected_columns),
        "max_features_cap": max_features,
        "min_mi_threshold": min_mi,
        "top_mi_scores": {col: round(float(mi_series[col]), 6) for col in selected_columns[:20]},
        "dropped_columns": [col for col in features.columns if col not in selected_columns],
    }
    return FeatureSelectionResult(frame=selected_frame, feature_blocks=selected_blocks, report=report)


def build_feature_set(
    df,
    lags=None,
    frac_diff_d=None,
    indicator_run=None,
    rolling_window=20,
    squeeze_quantile=0.2,
):
    """Build a feature frame and feature-block mapping."""
    blocks = []
    blocks.append(_price_volume_features(df, rolling_window=rolling_window))

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
                blocks.append(block)

        interaction_block = _indicator_interaction_features(
            df,
            indicator_run=indicator_run,
            rolling_window=rolling_window,
        )
        if not interaction_block.frame.empty:
            blocks.append(interaction_block)
    else:
        generic_result = type(
            "GenericIndicatorResult",
            (),
            {
                "kind": "generic_indicator",
                "metadata": {
                    "output_columns": [
                        column for column in df.columns
                        if column not in BASE_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
                    ]
                },
            },
        )()
        block = _extract_generic_indicator_features(
            generic_result,
            df,
            rolling_window=rolling_window,
        )
        if not block.frame.empty:
            blocks.append(block)

    features = pd.DataFrame(index=df.index)
    feature_blocks = {}
    laggable_columns = []

    for block in blocks:
        if block.frame.empty:
            continue
        features = features.join(block.frame)
        laggable_columns.extend(block.laggable_columns)
        for column in block.frame.columns:
            feature_blocks[column] = block.block_name

    if frac_diff_d is not None:
        features["close_fracdiff"] = fractional_diff(df["close"], d=frac_diff_d)
        laggable_columns.append("close_fracdiff")
        feature_blocks["close_fracdiff"] = "price_volume"

    features, feature_blocks = _append_lags(
        features,
        laggable_columns=laggable_columns,
        lags=lags,
        feature_blocks=feature_blocks,
    )
    return BuiltFeatureSet(frame=features, feature_blocks=feature_blocks)


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
    feature_set = build_feature_set(
        df,
        lags=lags,
        frac_diff_d=frac_diff_d,
        indicator_run=indicator_run,
        rolling_window=rolling_window,
        squeeze_quantile=squeeze_quantile,
    )
    return feature_set.frame
