"""Feature engineering: fractional differentiation, stationarity, derived features."""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


# ---------------------------------------------------------------------------
# Fractional differentiation
# ---------------------------------------------------------------------------

def fractional_diff(series, d, threshold=1e-5):
    """Fractionally differentiate *series* by order *d*.

    Uses expanding-window weights (cut off at *threshold*) applied as a
    convolution.  Preserves long-memory information that integer differencing
    destroys.

    Returns a pd.Series (leading values are NaN until the weight window is
    filled).
    """
    # build weights (newest → oldest)
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1

    weights = np.array(weights[::-1])          # oldest-first
    width = len(weights)
    vals = series.values.astype(float)
    out = np.full(len(vals), np.nan)

    for i in range(width - 1, len(vals)):
        out[i] = weights @ vals[i - width + 1: i + 1]

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


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

def build_features(df, lags=None, frac_diff_d=None):
    """Derive a feature matrix from an indicator-enriched OHLCV DataFrame.

    Adds:
      - returns at 1/5/10 bars
      - log-return
      - volume z-score
      - z-scores and optional lags of every indicator column
      - fractionally differentiated close (if *frac_diff_d* is set)

    Parameters
    ----------
    df : pd.DataFrame  – OHLCV + indicator columns (output of attach_indicators)
    lags : list[int]    – e.g. [1, 3, 6]
    frac_diff_d : float – fractional-diff order (e.g. 0.4)

    Returns
    -------
    pd.DataFrame – feature matrix (same index as *df*, NaN where warm-up needed)
    """
    feat = pd.DataFrame(index=df.index)

    # Price returns
    feat["return_1"] = df["close"].pct_change(1)
    feat["return_5"] = df["close"].pct_change(5)
    feat["return_10"] = df["close"].pct_change(10)
    feat["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Volume
    feat["vol_change"] = df["volume"].pct_change(1)
    vol_mean = df["volume"].rolling(20).mean()
    vol_std = df["volume"].rolling(20).std()
    feat["vol_zscore"] = (df["volume"] - vol_mean) / vol_std

    # Indicator columns → z-scores (+ optional lags)
    base_cols = {"open", "high", "low", "close", "volume", "quote_volume", "trades"}
    ind_cols = [
        column for column in df.columns
        if column not in base_cols and pd.api.types.is_numeric_dtype(df[column])
    ]

    for col in ind_cols:
        feat[col] = df[col]
        rmean = df[col].rolling(20).mean()
        rstd = df[col].rolling(20).std()
        feat[f"{col}_z"] = (df[col] - rmean) / rstd

        if lags:
            for lag in lags:
                feat[f"{col}_lag{lag}"] = df[col].shift(lag)

    # Fractional differentiation
    if frac_diff_d is not None:
        feat["close_fracdiff"] = fractional_diff(df["close"], d=frac_diff_d)

    return feat
