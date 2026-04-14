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

    Uses fixed-window weights (cut off at *threshold*) applied as a 
    convolution. Preserves long-memory information while achieving stationarity.

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

    weights = np.array(weights, dtype=float)
    width = len(weights)
    values = series.values.astype(float)
    
    # We convolve the series with the weights. Note: we don't reverse weights
    # here because we're using the standard FIR filter definition.
    # 'valid' mode returns only the full-window results.
    out = np.full(len(values), np.nan)
    if len(values) >= width:
        # np.convolve(v, w) performs sum(v[i-k] * w[k]).
        # For fractional differentiation weights [w0, w1, w2...], 
        # the result is w0*x[t] + w1*x[t-1] + w2*x[t-2]...
        res = np.convolve(values, weights, mode='valid')
        out[width-1:] = res

    series_name = getattr(series, "name", None)
    output_name = f"{series_name}_fracdiff" if series_name else "fracdiff"
    return pd.Series(out, index=series.index, name=output_name)


# ---------------------------------------------------------------------------
# Stationarity guard
# ---------------------------------------------------------------------------

def check_stationarity(series, significance=0.05):
    """Run Augmented Dickey-Fuller (ADF) and KPSS tests.
    
    A series is considered stationary if ADF rejects the unit-root null 
    (p < sig) OR if KPSS fails to reject the stationarity null (p >= sig).
    """
    clean = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    unique_values = int(clean.nunique())
    if len(clean) < 20:
        return {"stationary": False, "adf_p": 1.0, "kpss_p": 0.0, "error": "too_short"}
    if unique_values <= 1:
        return {"stationary": False, "adf_p": 1.0, "kpss_p": 0.0, "error": "constant"}

    try:
        from statsmodels.tsa.stattools import adfuller, kpss
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_res = adfuller(clean, maxlag=min(20, len(clean) // 4))
            kpss_res = kpss(clean, regression='c', nlags='auto')
            
        adf_p = float(adf_res[1])
        kpss_p = float(kpss_res[1])
        
        adf_stationary = adf_p < significance
        kpss_stationary = kpss_p >= significance
        stationary = bool(adf_stationary or kpss_stationary)
        
        return {
            "stationary": stationary,
            "adf_p": round(adf_p, 6),
            "kpss_p": round(kpss_p, 6),
            "observations": int(len(clean)),
            "unique_values": unique_values,
        }
    except Exception as exc:
        return {"stationary": False, "adf_p": 1.0, "kpss_p": 0.0, "error": type(exc).__name__}


class StationarityGuard:
    """Stateful guard for fold-local stationarity screening with in-memory caching."""

    def __init__(self, significance=0.05, max_d=1.0, step_d=0.1):
        self.significance = significance
        self.max_d = max_d
        self.step_d = step_d
        self._cache = {}

    def _get_cache_key(self, column, series_fit):
        if series_fit.empty:
            return None
        return (column, len(series_fit), series_fit.index[-1])

    def _find_best_d(self, series_fit):
        """Find minimum d in [0, max_d] that achieves stationarity."""
        # 1. Check passthrough (d=0.0)
        stat = check_stationarity(series_fit, significance=self.significance)
        if stat["stationary"]:
            return 0.0, stat
            
        # 2. Search d space
        d_values = np.arange(self.step_d, self.max_d + self.step_d/2, self.step_d)
        for d in d_values:
            d = round(float(d), 2)
            if d >= 1.0:
                transformed = series_fit.diff()
                d = 1.0
            else:
                transformed = fractional_diff(series_fit, d=d)
                
            stat = check_stationarity(transformed, significance=self.significance)
            if stat["stationary"]:
                return d, stat
                
        # 3. Fallback to first-difference (1.0)
        return 1.0, stat

    def screen_and_transform(self, X_fit, X_apply=None, feature_blocks=None, config=None):
        """Screen features based on X_fit and apply selected transforms to X_apply."""
        if X_apply is None:
            X_apply = X_fit
            
        config = dict(config or {})
        feature_blocks = dict(feature_blocks or {})
        if not config.get("enabled", True):
            return FeatureScreeningResult(
                frame=X_apply.copy(), 
                feature_blocks=feature_blocks, 
                report={"summary": {"enabled": False}}
            )

        transformed_df = pd.DataFrame(index=X_apply.index)
        transforms_used = {}
        reports = {}
        transformed_features = []
        dropped_features = []
        
        summary = {
            "enabled": True,
            "total_features": int(X_fit.shape[1]),
            "stationary_as_is": 0,
            "transformed_features": 0,
            "dropped_features": 0,
            "transform_usage": {},
        }
        
        kept_blocks = {}
        discrete_max_unique = int(config.get("discrete_max_unique", 6))

        for column in X_fit.columns:
            series_fit = X_fit[column]
            series_apply = X_apply[column]
            block_name = feature_blocks.get(column, "unknown")
            
            clean = series_fit.replace([np.inf, -np.inf], np.nan).dropna()
            unique_values = int(clean.nunique())
            
            if unique_values <= 1:
                summary["dropped_features"] += 1
                dropped_features.append(column)
                continue
                
            if unique_values <= discrete_max_unique:
                transformed_df[column] = series_apply
                kept_blocks[column] = block_name
                continue
            
            cache_key = self._get_cache_key(column, series_fit)
            if cache_key and cache_key in self._cache:
                best_d, stat = self._cache[cache_key]
            else:
                best_d, stat = self._find_best_d(series_fit)
                if cache_key:
                    self._cache[cache_key] = (best_d, stat)
                
            transforms_used[column] = best_d
            if best_d == 0.0:
                transformed_df[column] = series_apply
                summary["stationary_as_is"] += 1
            elif best_d >= 1.0:
                transformed_df[column] = series_apply.diff()
                summary["transformed_features"] += 1
                transformed_features.append(column)
                summary["transform_usage"]["diff"] = summary["transform_usage"].get("diff", 0) + 1
            else:
                transformed_df[column] = fractional_diff(series_apply, d=best_d)
                summary["transformed_features"] += 1
                transformed_features.append(column)
                usage_key = f"frac_diff_{best_d}"
                summary["transform_usage"][usage_key] = summary["transform_usage"].get(usage_key, 0) + 1
                
            kept_blocks[column] = block_name
            reports[column] = {"block": block_name, "best_d": best_d, "stat": stat}
            
        return FeatureScreeningResult(
            frame=transformed_df,
            feature_blocks=kept_blocks,
            report={
                "summary": summary, 
                "features": reports, 
                "transforms_used": transforms_used,
                "transformed_features": transformed_features,
                "dropped_features": dropped_features,
            }
        )
