"""Regime feature construction, provenance, and detection helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .context import (
    build_cross_asset_context_feature_block,
    build_futures_context_feature_block,
    build_multi_timeframe_context_feature_block,
)
from .reference_data import build_reference_overlay_feature_block

REGIME_INSTRUMENT_SOURCE = "instrument_state"
REGIME_MARKET_SOURCE = "market_state"
REGIME_CROSS_ASSET_SOURCE = "cross_asset_state"
REGIME_SOURCE_ORDER = (
    REGIME_INSTRUMENT_SOURCE,
    REGIME_MARKET_SOURCE,
    REGIME_CROSS_ASSET_SOURCE,
)
CONTEXTUAL_REGIME_SOURCES = frozenset({REGIME_MARKET_SOURCE, REGIME_CROSS_ASSET_SOURCE})


@dataclass
class RegimeFeatureSet:
    frame: pd.DataFrame
    source_map: dict[str, str] = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)


def _sanitize_regime_frame(frame):
    clean = pd.DataFrame(frame).copy()
    if clean.empty:
        return clean
    return clean.replace([np.inf, -np.inf], np.nan)


def _infer_regime_source(column_name):
    normalized = str(column_name or "").lower()
    if normalized.startswith("ctx_"):
        return REGIME_CROSS_ASSET_SOURCE
    if normalized.startswith(("fut_", "ref_", "composite_")):
        return REGIME_MARKET_SOURCE
    if normalized.startswith("mtf_"):
        return REGIME_INSTRUMENT_SOURCE
    return REGIME_INSTRUMENT_SOURCE


def infer_regime_source_map(columns):
    return {column: _infer_regime_source(column) for column in columns}


def summarize_regime_provenance(source_map, columns=None):
    active_columns = list(columns if columns is not None else source_map.keys())
    filtered_source_map = {
        column: source_map.get(column, _infer_regime_source(column))
        for column in active_columns
    }
    counts = {
        source: int(sum(1 for column in active_columns if filtered_source_map.get(column) == source))
        for source in REGIME_SOURCE_ORDER
    }
    total = int(sum(counts.values()))
    shares = {
        source: (float(counts[source]) / float(total) if total > 0 else 0.0)
        for source in REGIME_SOURCE_ORDER
    }
    columns_by_source = {
        source: [column for column in active_columns if filtered_source_map.get(column) == source]
        for source in REGIME_SOURCE_ORDER
    }
    contextual_share = float(sum(shares[source] for source in CONTEXTUAL_REGIME_SOURCES))
    endogenous_share = float(shares[REGIME_INSTRUMENT_SOURCE])
    dominant_source = max(counts, key=counts.get) if total > 0 else None
    return {
        "source_counts": counts,
        "source_shares": shares,
        "columns_by_source": columns_by_source,
        "total_columns": total,
        "contextual_share": contextual_share,
        "endogenous_share": endogenous_share,
        "dominant_source": dominant_source,
    }


def normalize_regime_feature_set(value):
    if isinstance(value, RegimeFeatureSet):
        frame = _sanitize_regime_frame(value.frame)
        source_map = dict(value.source_map or {})
        provenance = dict(value.provenance or {})
    elif isinstance(value, dict) and "frame" in value:
        frame = _sanitize_regime_frame(value.get("frame"))
        source_map = dict(value.get("source_map") or {})
        provenance = dict(value.get("provenance") or {})
    elif isinstance(value, tuple) and len(value) == 2:
        frame = _sanitize_regime_frame(value[0])
        source_map = dict(value[1] or {})
        provenance = {}
    elif value is None:
        frame = pd.DataFrame()
        source_map = {}
        provenance = {}
    else:
        frame = _sanitize_regime_frame(value)
        source_map = {}
        provenance = {}

    inferred = infer_regime_source_map(frame.columns)
    merged_source_map = {column: source_map.get(column, inferred[column]) for column in frame.columns}
    if not provenance:
        provenance = summarize_regime_provenance(merged_source_map, columns=frame.columns)
    return RegimeFeatureSet(frame=frame, source_map=merged_source_map, provenance=provenance)


def _join_state_frame(base_frame, addition, source_name, source_map):
    if addition is None or addition.empty:
        return base_frame, source_map

    joined = base_frame.join(addition, how="outer") if not base_frame.empty else addition.copy()
    for column in addition.columns:
        source_map[column] = source_name
    return joined, source_map


def _resolve_regime_min_periods(window, floor=5):
    return int(max(floor, int(window) // 2))


def _online_cusum_score(series, window):
    values = pd.Series(series, copy=False).astype(float)
    min_periods = _resolve_regime_min_periods(window)
    rolling_mean = values.rolling(window, min_periods=min_periods).mean()
    rolling_std = values.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
    standardized = ((values - rolling_mean) / rolling_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    positive_cusum = standardized.clip(lower=0.0).rolling(window, min_periods=min_periods).sum()
    negative_cusum = (-standardized.clip(upper=0.0)).rolling(window, min_periods=min_periods).sum()
    return pd.concat([positive_cusum, negative_cusum], axis=1).max(axis=1)


def build_instrument_regime_state(base_data, rolling_window=20, base_interval="1h", context_timeframes=None):
    data = pd.DataFrame(base_data).copy()
    if data.empty:
        return pd.DataFrame(index=data.index)

    close = data["close"].astype(float)
    returns = close.pct_change()
    min_periods = _resolve_regime_min_periods(rolling_window)
    long_window = int(max(rolling_window * 3, rolling_window + 12))
    long_min_periods = _resolve_regime_min_periods(long_window, floor=min_periods)
    short_vol = returns.ewm(span=rolling_window, adjust=False, min_periods=min_periods).std()
    long_vol = returns.ewm(span=long_window, adjust=False, min_periods=long_min_periods).std()
    rolling_mean = close.rolling(rolling_window, min_periods=min_periods).mean()
    rolling_std = close.rolling(rolling_window, min_periods=min_periods).std().replace(0, np.nan)
    long_mean = close.rolling(long_window, min_periods=long_min_periods).mean()
    long_std = close.rolling(long_window, min_periods=long_min_periods).std().replace(0, np.nan)
    standardized_returns = (
        (returns - returns.rolling(rolling_window, min_periods=min_periods).mean())
        / returns.rolling(rolling_window, min_periods=min_periods).std().replace(0, np.nan)
    )
    quote_volume = (
        data["quote_volume"].astype(float)
        if "quote_volume" in data.columns
        else close * data["volume"].astype(float)
    )
    illiquidity = returns.abs() / quote_volume.replace(0, np.nan)

    frame = pd.DataFrame(
        {
            "ret_1": returns,
            "ret_6": close.pct_change(6),
            "trend_20": close.pct_change(20),
            "trend_60": close.pct_change(60),
            "vol_20": returns.rolling(20, min_periods=10).std(),
            "vol_60": returns.rolling(60, min_periods=20).std(),
            "ewm_vol_20": short_vol,
            "ewm_vol_60": long_vol,
            "vol_cluster_ratio_20_60": short_vol / long_vol.replace(0, np.nan),
            "vol_of_vol_20": short_vol.rolling(rolling_window, min_periods=min_periods).std(),
            "range_20": ((data["high"] - data["low"]) / close).rolling(20, min_periods=10).mean(),
            "trend_z_20": close.diff(rolling_window) / (rolling_std * np.sqrt(max(1, rolling_window))),
            "trend_z_60": close.diff(long_window) / (long_std * np.sqrt(max(1, long_window))),
            "mean_reversion_gap_20": (close - rolling_mean) / rolling_std,
            "mean_reversion_gap_60": (close - long_mean) / long_std,
            "drawdown_60": close / close.rolling(long_window, min_periods=long_min_periods).max() - 1.0,
            "break_score_20": _online_cusum_score(standardized_returns, rolling_window),
            "shock_score_20": standardized_returns.abs().ewm(span=rolling_window, adjust=False, min_periods=min_periods).mean(),
            "liquidity_20": np.log1p(quote_volume).rolling(20, min_periods=10).mean(),
            "illiquidity_20": illiquidity.rolling(20, min_periods=10).mean(),
            "trades_20": (
                data["trades"].astype(float).rolling(20, min_periods=10).mean()
                if "trades" in data.columns
                else np.nan
            ),
        },
        index=data.index,
    )

    multi_timeframe = build_multi_timeframe_context_feature_block(
        data,
        base_interval=base_interval,
        timeframes=context_timeframes,
        rolling_window=rolling_window,
    ).frame
    if not multi_timeframe.empty:
        frame = frame.join(multi_timeframe)

    return _sanitize_regime_frame(frame)


def build_market_regime_state(base_data, futures_context=None, reference_data=None, rolling_window=20):
    data = pd.DataFrame(base_data).copy()
    if data.empty:
        return pd.DataFrame(index=data.index)

    futures_frame = build_futures_context_feature_block(
        data,
        futures_context,
        rolling_window=rolling_window,
    ).frame
    reference_frame = build_reference_overlay_feature_block(
        data,
        reference_data=reference_data,
        rolling_window=rolling_window,
    ).frame
    frame = pd.DataFrame(index=data.index)
    if not futures_frame.empty:
        frame = frame.join(futures_frame)
    if not reference_frame.empty:
        frame = frame.join(reference_frame)
    return _sanitize_regime_frame(frame)


def build_cross_asset_regime_state(base_data, cross_asset_context=None, rolling_window=20):
    data = pd.DataFrame(base_data).copy()
    if data.empty:
        return pd.DataFrame(index=data.index)
    return _sanitize_regime_frame(
        build_cross_asset_context_feature_block(
            data,
            cross_asset_context,
            rolling_window=rolling_window,
        ).frame
    )


def build_default_regime_feature_set(
    base_data,
    *,
    base_interval="1h",
    rolling_window=20,
    futures_context=None,
    cross_asset_context=None,
    reference_data=None,
    context_timeframes=None,
):
    base_frame = pd.DataFrame(base_data).copy()
    frame = pd.DataFrame(index=base_frame.index)
    source_map = {}

    instrument_state = build_instrument_regime_state(
        base_frame,
        rolling_window=rolling_window,
        base_interval=base_interval,
        context_timeframes=context_timeframes,
    )
    frame, source_map = _join_state_frame(frame, instrument_state, REGIME_INSTRUMENT_SOURCE, source_map)

    market_state = build_market_regime_state(
        base_frame,
        futures_context=futures_context,
        reference_data=reference_data,
        rolling_window=rolling_window,
    )
    frame, source_map = _join_state_frame(frame, market_state, REGIME_MARKET_SOURCE, source_map)

    cross_asset_state = build_cross_asset_regime_state(
        base_frame,
        cross_asset_context=cross_asset_context,
        rolling_window=rolling_window,
    )
    frame, source_map = _join_state_frame(frame, cross_asset_state, REGIME_CROSS_ASSET_SOURCE, source_map)

    frame = _sanitize_regime_frame(frame)
    provenance = summarize_regime_provenance(source_map, columns=frame.columns)
    return RegimeFeatureSet(frame=frame, source_map=source_map, provenance=provenance)


def _coalesce_regime_signal(features, include_terms, exclude_terms=None,
                            reference_features=None, fallback_column=None):
    exclude_terms = tuple(exclude_terms or ())
    reference = features if reference_features is None else reference_features
    selected_columns = []
    for column in features.columns:
        normalized = column.lower()
        if any(term in normalized for term in include_terms) and not any(term in normalized for term in exclude_terms):
            selected_columns.append(column)

    if not selected_columns and fallback_column is not None and fallback_column in features.columns:
        selected_columns = [fallback_column]

    if not selected_columns:
        return pd.Series(0.0, index=features.index, dtype=float)

    selected = features[selected_columns].apply(pd.to_numeric, errors="coerce")
    reference_selected = reference[selected_columns].apply(pd.to_numeric, errors="coerce")
    reference_mean = reference_selected.mean()
    reference_std = reference_selected.std().replace(0, 1)
    standardized = (selected - reference_mean) / reference_std
    return standardized.mean(axis=1).fillna(0.0)


def _coalesce_online_regime_signal(features, include_terms, exclude_terms=None,
                                   fallback_column=None, lookback=None, min_periods=20):
    exclude_terms = tuple(exclude_terms or ())
    selected_columns = []
    for column in features.columns:
        normalized = column.lower()
        if any(term in normalized for term in include_terms) and not any(term in normalized for term in exclude_terms):
            selected_columns.append(column)

    if not selected_columns and fallback_column is not None and fallback_column in features.columns:
        selected_columns = [fallback_column]

    if not selected_columns:
        return pd.Series(0.0, index=features.index, dtype=float)

    selected = features[selected_columns].apply(pd.to_numeric, errors="coerce")
    history = selected.shift(1)
    if lookback is not None:
        history_mean = history.rolling(int(lookback), min_periods=int(min_periods)).mean()
        history_std = history.rolling(int(lookback), min_periods=int(min_periods)).std()
    else:
        history_mean = history.expanding(min_periods=int(min_periods)).mean()
        history_std = history.expanding(min_periods=int(min_periods)).std()
    history_std = history_std.replace(0, np.nan)
    standardized = ((selected - history_mean) / history_std).replace([np.inf, -np.inf], np.nan)
    return standardized.mean(axis=1).fillna(0.0)


def _bucket_regime_signal(series, lower_quantile=0.33, upper_quantile=0.67,
                          invert=False, reference_series=None):
    values = -pd.Series(series, copy=False) if invert else pd.Series(series, copy=False)
    reference = values if reference_series is None else (-pd.Series(reference_series, copy=False) if invert else pd.Series(reference_series, copy=False))
    clean = reference.dropna()
    if clean.empty:
        return pd.Series(0, index=values.index, dtype=int)

    lower = float(clean.quantile(lower_quantile))
    upper = float(clean.quantile(upper_quantile))
    bucket = pd.Series(0, index=values.index, dtype=int)
    bucket[values <= lower] = -1
    bucket[values >= upper] = 1
    return bucket


def _online_bucket_regime_signal(series, lower_quantile=0.33, upper_quantile=0.67,
                                 invert=False, lookback=None, min_periods=20):
    values = -pd.Series(series, copy=False) if invert else pd.Series(series, copy=False)
    history = values.shift(1)
    if lookback is not None:
        lower = history.rolling(int(lookback), min_periods=int(min_periods)).quantile(lower_quantile)
        upper = history.rolling(int(lookback), min_periods=int(min_periods)).quantile(upper_quantile)
    else:
        lower = history.expanding(min_periods=int(min_periods)).quantile(lower_quantile)
        upper = history.expanding(min_periods=int(min_periods)).quantile(upper_quantile)

    bucket = pd.Series(0, index=values.index, dtype=int)
    bucket[values <= lower] = -1
    bucket[values >= upper] = 1
    return bucket.fillna(0).astype(int)


def _online_upper_tail_regime_signal(series, upper_quantile=0.67, lookback=None, min_periods=20):
    values = pd.Series(series, copy=False)
    history = values.shift(1)
    if lookback is not None:
        upper = history.rolling(int(lookback), min_periods=int(min_periods)).quantile(upper_quantile)
    else:
        upper = history.expanding(min_periods=int(min_periods)).quantile(upper_quantile)
    regime = pd.Series(0, index=values.index, dtype=int)
    regime[values >= upper] = 1
    return regime.fillna(0).astype(int)


def _detect_explicit_regime(features, config=None, fit_features=None):
    config = dict(config or {})
    clean = features.dropna()
    if clean.empty:
        return pd.DataFrame(columns=["trend_regime", "volatility_regime", "liquidity_regime", "regime"])

    reference = clean if fit_features is None else fit_features.reindex(columns=features.columns).dropna()
    if reference.empty:
        reference = clean

    trend_score = _coalesce_regime_signal(
        clean,
        include_terms=("trend", "ret_", "return", "momentum"),
        exclude_terms=("vol", "volume", "liquid"),
        reference_features=reference,
    )
    volatility_score = _coalesce_regime_signal(
        clean,
        include_terms=("vol", "range", "atr", "dispersion"),
        exclude_terms=("volume", "liquid"),
        reference_features=reference,
    )
    liquidity_score = _coalesce_regime_signal(
        clean,
        include_terms=("liquid", "volume", "turnover", "trade"),
        exclude_terms=("illiquid",),
        reference_features=reference,
    )
    illiquidity_score = _coalesce_regime_signal(
        clean,
        include_terms=("illiquid", "amihud"),
        reference_features=reference,
    )
    liquidity_score = liquidity_score - illiquidity_score

    trend_reference = _coalesce_regime_signal(
        reference,
        include_terms=("trend", "ret_", "return", "momentum"),
        exclude_terms=("vol", "volume", "liquid"),
        reference_features=reference,
    )
    volatility_reference = _coalesce_regime_signal(
        reference,
        include_terms=("vol", "range", "atr", "dispersion"),
        exclude_terms=("volume", "liquid"),
        reference_features=reference,
    )
    liquidity_reference = _coalesce_regime_signal(
        reference,
        include_terms=("liquid", "volume", "turnover", "trade"),
        exclude_terms=("illiquid",),
        reference_features=reference,
    )
    illiquidity_reference = _coalesce_regime_signal(
        reference,
        include_terms=("illiquid", "amihud"),
        reference_features=reference,
    )
    liquidity_reference = liquidity_reference - illiquidity_reference

    lower_quantile = float(config.get("lower_quantile", 0.33))
    upper_quantile = float(config.get("upper_quantile", 0.67))
    liquidity_invert = bool(config.get("liquidity_invert", False))

    trend_regime = _bucket_regime_signal(
        trend_score,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        reference_series=trend_reference,
    )
    volatility_regime = _bucket_regime_signal(
        volatility_score,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        reference_series=volatility_reference,
    )
    liquidity_regime = _bucket_regime_signal(
        liquidity_score,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        invert=liquidity_invert,
        reference_series=liquidity_reference,
    )

    composite = (
        (trend_regime + 1) * 9
        + (volatility_regime + 1) * 3
        + (liquidity_regime + 1)
    ).astype(int)

    return pd.DataFrame(
        {
            "trend_regime": trend_regime.astype(int),
            "volatility_regime": volatility_regime.astype(int),
            "liquidity_regime": liquidity_regime.astype(int),
            "regime": composite,
        },
        index=clean.index,
    )


def _detect_online_regime(features, config=None, fit_features=None):
    del fit_features
    config = dict(config or {})
    clean = features.dropna(how="all")
    if clean.empty:
        return pd.DataFrame(
            columns=[
                "trend_regime",
                "mean_reversion_regime",
                "volatility_regime",
                "liquidity_regime",
                "structural_break_regime",
                "regime",
            ]
        )

    lookback = config.get("online_lookback")
    min_periods = int(config.get("online_min_periods", max(20, config.get("rolling_window", 20))))

    trend_score = _coalesce_online_regime_signal(
        clean,
        include_terms=("trend", "ret_", "return", "momentum", "slope"),
        exclude_terms=("vol", "volume", "liquid", "reversion", "break", "shock"),
        lookback=lookback,
        min_periods=min_periods,
    )
    mean_reversion_score = _coalesce_online_regime_signal(
        clean,
        include_terms=("reversion", "gap", "zscore", "distance"),
        exclude_terms=("vol", "volume", "liquid", "break", "shock"),
        lookback=lookback,
        min_periods=min_periods,
    ).abs()
    volatility_score = _coalesce_online_regime_signal(
        clean,
        include_terms=("vol", "range", "atr", "dispersion", "cluster", "drawdown", "shock"),
        exclude_terms=("volume", "liquid"),
        lookback=lookback,
        min_periods=min_periods,
    )
    liquidity_score = _coalesce_online_regime_signal(
        clean,
        include_terms=("liquid", "volume", "turnover", "trade"),
        exclude_terms=("illiquid",),
        lookback=lookback,
        min_periods=min_periods,
    )
    illiquidity_score = _coalesce_online_regime_signal(
        clean,
        include_terms=("illiquid", "amihud"),
        lookback=lookback,
        min_periods=min_periods,
    )
    liquidity_score = liquidity_score - illiquidity_score

    break_score = _coalesce_online_regime_signal(
        clean,
        include_terms=("break", "shock", "jump", "crash", "drawdown"),
        exclude_terms=("volume", "liquid"),
        lookback=lookback,
        min_periods=min_periods,
    )
    break_score = break_score + volatility_score.diff().abs().fillna(0.0)

    lower_quantile = float(config.get("lower_quantile", 0.33))
    upper_quantile = float(config.get("upper_quantile", 0.67))
    liquidity_invert = bool(config.get("liquidity_invert", False))

    trend_regime = _online_bucket_regime_signal(
        trend_score,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        lookback=lookback,
        min_periods=min_periods,
    )
    mean_reversion_regime = _online_bucket_regime_signal(
        mean_reversion_score,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        lookback=lookback,
        min_periods=min_periods,
    )
    volatility_regime = _online_bucket_regime_signal(
        volatility_score,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        lookback=lookback,
        min_periods=min_periods,
    )
    liquidity_regime = _online_bucket_regime_signal(
        liquidity_score,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        invert=liquidity_invert,
        lookback=lookback,
        min_periods=min_periods,
    )
    structural_break_regime = _online_upper_tail_regime_signal(
        break_score,
        upper_quantile=upper_quantile,
        lookback=lookback,
        min_periods=min_periods,
    )

    composite = (
        (trend_regime + 1) * 54
        + (mean_reversion_regime + 1) * 18
        + (volatility_regime + 1) * 6
        + (liquidity_regime + 1) * 2
        + structural_break_regime
    ).astype(int)

    return pd.DataFrame(
        {
            "trend_regime": trend_regime.astype(int),
            "mean_reversion_regime": mean_reversion_regime.astype(int),
            "volatility_regime": volatility_regime.astype(int),
            "liquidity_regime": liquidity_regime.astype(int),
            "structural_break_regime": structural_break_regime.astype(int),
            "regime": composite,
        },
        index=clean.index,
    )


def _detect_hmm_regime(features, n_regimes=2, config=None, fit_features=None):
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "hmmlearn is required for method='hmm'. "
            "Install it with: pip install hmmlearn>=0.3"
        ) from exc

    config = dict(config or {})
    clean = features.dropna()
    if clean.empty:
        return pd.Series(dtype=int, name="regime")

    reference = clean if fit_features is None else fit_features.reindex(columns=features.columns).dropna()
    if reference.empty:
        reference = clean

    n_states = max(1, min(int(n_regimes), len(reference)))
    if n_states == 1:
        return pd.Series(0, index=clean.index, name="regime", dtype=int)

    covariance_type = config.get("covariance_type", "diag")
    n_iter = int(config.get("n_iter", 100))
    tol = float(config.get("tol", 1e-3))
    random_state = int(config.get("random_state", 42))

    scaler = StandardScaler()
    scaler.fit(reference)
    normed_reference = scaler.transform(reference)
    normed = scaler.transform(clean)

    hmm_model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
    )
    try:
        hmm_model.fit(normed_reference)
        raw_labels = hmm_model.predict(normed)
    except Exception:  # noqa: BLE001
        return pd.Series(0, index=clean.index, name="regime", dtype=int)

    norms = np.linalg.norm(hmm_model.means_, ord=1, axis=1)
    sort_order = np.argsort(norms)
    remap = np.empty(n_states, dtype=int)
    for new_label, old_label in enumerate(sort_order):
        remap[old_label] = new_label

    return pd.Series(remap[raw_labels], index=clean.index, name="regime", dtype=int)


def detect_regime(features, n_regimes=2, method="hmm", config=None, fit_features=None):
    method = (method or "hmm").lower()
    if method == "explicit":
        return _detect_explicit_regime(features, config=config, fit_features=fit_features)
    if method == "online":
        return _detect_online_regime(features, config=config, fit_features=fit_features)
    if method == "hmm":
        return _detect_hmm_regime(features, n_regimes=n_regimes, config=config, fit_features=fit_features)

    raise ValueError(
        f"Unknown regime detection method={method!r}. Choose from ['hmm', 'explicit', 'online']"
    )


def coerce_regime_label_series(regimes, column_name="regime"):
    if isinstance(regimes, pd.DataFrame):
        if regimes.empty:
            return pd.Series(dtype=float, name=column_name)
        target_column = column_name if column_name in regimes.columns else regimes.columns[0]
        return pd.Series(regimes[target_column], copy=False)
    return pd.Series(regimes, copy=False)


def compute_regime_path_stability(regimes):
    labels = coerce_regime_label_series(regimes).dropna()
    if labels.empty:
        return {
            "available_rows": 0,
            "distinct_states": 0,
            "switch_count": 0,
            "switch_rate": None,
            "persistence": None,
        }

    if len(labels) == 1:
        return {
            "available_rows": 1,
            "distinct_states": int(labels.nunique()),
            "switch_count": 0,
            "switch_rate": 0.0,
            "persistence": 1.0,
        }

    switches = labels.ne(labels.shift()).iloc[1:]
    switch_count = int(switches.sum())
    switch_rate = float(switch_count / max(1, len(labels) - 1))
    return {
        "available_rows": int(len(labels)),
        "distinct_states": int(labels.nunique()),
        "switch_count": switch_count,
        "switch_rate": switch_rate,
        "persistence": float(1.0 - switch_rate),
    }


def _regime_agreement_rate(left, right):
    left_labels = coerce_regime_label_series(left).dropna()
    right_labels = coerce_regime_label_series(right).dropna()
    common_index = left_labels.index.intersection(right_labels.index)
    if len(common_index) == 0:
        return None
    return float((left_labels.loc[common_index] == right_labels.loc[common_index]).mean())


def _subset_regime_feature_set(feature_set, allowed_sources):
    normalized = normalize_regime_feature_set(feature_set)
    allowed = set(allowed_sources)
    selected_columns = [
        column for column in normalized.frame.columns
        if normalized.source_map.get(column) in allowed
    ]
    subset_frame = normalized.frame.loc[:, selected_columns].copy() if selected_columns else pd.DataFrame(index=normalized.frame.index)
    subset_map = {column: normalized.source_map[column] for column in selected_columns}
    return RegimeFeatureSet(
        frame=subset_frame,
        source_map=subset_map,
        provenance=summarize_regime_provenance(subset_map, columns=subset_frame.columns),
    )


def build_regime_ablation_report(feature_set, *, n_regimes=2, method="hmm", config=None, fit_features=None, full_regimes=None):
    normalized = normalize_regime_feature_set(feature_set)
    endogenous_only = _subset_regime_feature_set(normalized, {REGIME_INSTRUMENT_SOURCE})
    contextual_only = _subset_regime_feature_set(normalized, CONTEXTUAL_REGIME_SOURCES)

    report = {
        "full_provenance": normalized.provenance,
        "endogenous_provenance": endogenous_only.provenance,
        "contextual_provenance": contextual_only.provenance,
        "contextual_sources_present": bool(contextual_only.frame.shape[1] > 0),
        "contextual_column_count": int(contextual_only.frame.shape[1]),
        "agreement_rate": None,
        "full_stability": {},
        "endogenous_stability": {},
        "stability_improvement": None,
        "stability_gate": {
            "required": False,
            "passed": True,
            "min_persistence_improvement": float(dict(config or {}).get("min_persistence_improvement", 0.0)),
        },
    }

    if endogenous_only.frame.empty:
        report["reason"] = "endogenous_baseline_unavailable"
        return report

    fit_reference = None if fit_features is None else normalize_regime_feature_set(
        {
            "frame": fit_features,
            "source_map": normalized.source_map,
        }
    )
    endogenous_fit = None
    if fit_reference is not None:
        endogenous_fit = fit_reference.frame.reindex(columns=endogenous_only.frame.columns)

    baseline_regimes = detect_regime(
        endogenous_only.frame,
        n_regimes=n_regimes,
        method=method,
        config=config,
        fit_features=endogenous_fit,
    )
    enriched_regimes = full_regimes if full_regimes is not None else detect_regime(
        normalized.frame,
        n_regimes=n_regimes,
        method=method,
        config=config,
        fit_features=None if fit_reference is None else fit_reference.frame,
    )

    baseline_stability = compute_regime_path_stability(baseline_regimes)
    enriched_stability = compute_regime_path_stability(enriched_regimes)
    improvement = None
    if baseline_stability.get("persistence") is not None and enriched_stability.get("persistence") is not None:
        improvement = float(enriched_stability["persistence"] - baseline_stability["persistence"])

    gating_config = dict(config or {})
    min_improvement = float(gating_config.get("min_persistence_improvement", 0.0))
    require_improvement = bool(gating_config.get("require_stability_improvement", True)) and report["contextual_sources_present"]
    passed = True
    if require_improvement:
        passed = improvement is not None and improvement > min_improvement

    report.update(
        {
            "agreement_rate": _regime_agreement_rate(baseline_regimes, enriched_regimes),
            "full_stability": enriched_stability,
            "endogenous_stability": baseline_stability,
            "stability_improvement": improvement,
            "stability_gate": {
                "required": require_improvement,
                "passed": bool(passed),
                "min_persistence_improvement": min_improvement,
            },
        }
    )
    return report


def summarize_regime_ablation_reports(reports):
    rows = [row for row in (reports or []) if row]
    required_rows = [row for row in rows if row.get("stability_gate", {}).get("required")]
    failed_rows = [row for row in required_rows if not row.get("stability_gate", {}).get("passed", True)]
    agreement_rates = [row.get("agreement_rate") for row in rows if row.get("agreement_rate") is not None]
    improvements = [row.get("stability_improvement") for row in required_rows if row.get("stability_improvement") is not None]
    contextual_shares = [
        row.get("full_provenance", {}).get("contextual_share")
        for row in rows
        if row.get("full_provenance", {}).get("contextual_share") is not None
    ]

    if not required_rows:
        status = "unknown"
        reasons = ["regime_ablation_evidence_missing"]
    elif failed_rows:
        status = "failed"
        reasons = ["regime_stability_failed"]
    else:
        status = "passed"
        reasons = []

    return {
        "fold_count": int(len(rows)),
        "required_fold_count": int(len(required_rows)),
        "failed_fold_count": int(len(failed_rows)),
        "avg_contextual_share": float(np.mean(contextual_shares)) if contextual_shares else 0.0,
        "avg_agreement_rate": float(np.mean(agreement_rates)) if agreement_rates else None,
        "avg_persistence_improvement": float(np.mean(improvements)) if improvements else None,
        "status": status,
        "promotion_pass": status == "passed",
        "reasons": reasons,
    }


__all__ = [
    "CONTEXTUAL_REGIME_SOURCES",
    "REGIME_CROSS_ASSET_SOURCE",
    "REGIME_INSTRUMENT_SOURCE",
    "REGIME_MARKET_SOURCE",
    "REGIME_SOURCE_ORDER",
    "RegimeFeatureSet",
    "build_cross_asset_regime_state",
    "build_default_regime_feature_set",
    "build_instrument_regime_state",
    "build_market_regime_state",
    "build_regime_ablation_report",
    "coerce_regime_label_series",
    "compute_regime_path_stability",
    "detect_regime",
    "infer_regime_source_map",
    "normalize_regime_feature_set",
    "summarize_regime_ablation_reports",
    "summarize_regime_provenance",
]