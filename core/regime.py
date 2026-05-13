"""Regime feature construction, provenance, and detection helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .regimes.contracts import (
    BaseRegimeDetector,
    RegimeDetectorManifest,
    RegimeObservationContract,
    RegimeStateContract,
    RegimeTraceSummary,
    RegimeTransitionContract,
)

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


def _coerce_contract_scalar(value: Any):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _resolve_row_value(value, index, position):
    if value is None:
        return None
    if isinstance(value, pd.Series):
        series = value.reindex(index)
        return _coerce_contract_scalar(series.iloc[position])
    if isinstance(value, dict):
        return _coerce_contract_scalar(value.get(index[position]))
    if isinstance(value, (list, tuple, np.ndarray, pd.Index)):
        values = list(value)
        if position < len(values):
            return _coerce_contract_scalar(values[position])
        return None
    return _coerce_contract_scalar(value)


def _shift_contract_available_at(index, position, lag_bars):
    timestamp = index[position]
    lag = int(max(0, lag_bars or 0))
    if lag <= 0:
        return timestamp
    target_position = position + lag
    if target_position < len(index):
        return index[target_position]
    if len(index) >= 2:
        step = index[-1] - index[-2]
        overflow = target_position - (len(index) - 1)
        try:
            return index[-1] + (step * overflow)
        except Exception:
            return timestamp
    return timestamp


def _normalize_confidence_kind(value, *, has_probabilities, has_confidence):
    if value is not None and str(value).strip():
        return str(value).strip().lower()
    if has_probabilities:
        return "posterior"
    if has_confidence:
        return "heuristic"
    return "unsupported"


def build_regime_observation_contracts(observations, *, source_map=None, available_at=None, metadata=None):
    feature_set = normalize_regime_feature_set({"frame": observations, "source_map": source_map or {}})
    frame = feature_set.frame
    shared_metadata = dict(metadata or {})
    contracts = []
    for timestamp, row in frame.iterrows():
        values = {
            str(column): _coerce_contract_scalar(value)
            for column, value in row.items()
            if not pd.isna(value)
        }
        contracts.append(
            RegimeObservationContract(
                as_of=timestamp,
                available_at=(timestamp if available_at is None else available_at),
                values=values,
                source_map={str(key): str(value) for key, value in feature_set.source_map.items()},
                metadata=shared_metadata,
            )
        )
    return contracts


def build_regime_state_contracts(regimes, *, available_at=None, metadata=None):
    frame = regimes.copy() if isinstance(regimes, pd.DataFrame) else pd.DataFrame({"regime": pd.Series(regimes, copy=False)})
    if frame.empty:
        return []

    shared_metadata = dict(metadata or {})
    contracts = []
    index = pd.Index(frame.index)
    for position, (timestamp, row) in enumerate(frame.iterrows()):
        detector_outputs = {
            str(column): _coerce_contract_scalar(value)
            for column, value in row.items()
            if not pd.isna(value)
        }
        label = detector_outputs.get("regime")
        if label is None and detector_outputs:
            label = detector_outputs[next(iter(detector_outputs))]

        probabilities = {
            str(column): float(value)
            for column, value in detector_outputs.items()
            if (column.startswith("prob_") or column.endswith("_prob")) and value is not None
        }

        confidence = detector_outputs.get("regime_confidence")
        if confidence is None and probabilities:
            confidence = max(float(value) for value in probabilities.values())

        confidence_kind = _normalize_confidence_kind(
            detector_outputs.get("confidence_kind") or shared_metadata.get("confidence_kind"),
            has_probabilities=bool(probabilities),
            has_confidence=confidence is not None,
        )
        if confidence_kind == "unsupported":
            confidence = None

        explicit_source_available_at = (
            detector_outputs.get("source_available_at")
            or detector_outputs.get("available_at")
            or _resolve_row_value(available_at, index, position)
        )
        explicit_lag = detector_outputs.get("recognition_lag_bars")
        explicit_same_bar = detector_outputs.get("same_bar_available")
        if explicit_lag is None:
            if explicit_source_available_at is not None or explicit_same_bar is True:
                recognition_lag_bars = 0
            else:
                recognition_lag_bars = 1
        else:
            recognition_lag_bars = int(max(0, int(explicit_lag)))
        source_available_at = timestamp if explicit_source_available_at is None else explicit_source_available_at
        resolved_available_at = (
            explicit_source_available_at
            if explicit_source_available_at is not None
            else _shift_contract_available_at(index, position, recognition_lag_bars)
        )
        availability_reason = detector_outputs.get("availability_reason")
        if availability_reason is None:
            if explicit_source_available_at is not None:
                availability_reason = "declared_available_at"
            elif recognition_lag_bars > 0:
                availability_reason = "deferred_by_default_recognition_lag"
            else:
                availability_reason = "same_bar_declared"

        contracts.append(
            RegimeStateContract(
                as_of=timestamp,
                available_at=resolved_available_at,
                label=label,
                probabilities=probabilities,
                confidence=(None if confidence is None else float(confidence)),
                confidence_kind=confidence_kind,
                recognition_lag_bars=recognition_lag_bars,
                source_available_at=source_available_at,
                availability_reason=(None if availability_reason is None else str(availability_reason)),
                detector_outputs=detector_outputs,
                warm=bool(detector_outputs.get("warm", True)),
                metadata=shared_metadata,
            )
        )
    return contracts


def build_regime_transition_contracts(regimes, *, available_at=None, metadata=None):
    states = build_regime_state_contracts(regimes, available_at=available_at, metadata=metadata)
    if not states:
        return []

    transitions = []
    previous = states[0]
    for current in states[1:]:
        if current.label != previous.label:
            confidence = current.confidence if current.confidence is not None else previous.confidence
            transitions.append(
                RegimeTransitionContract(
                    as_of=current.as_of,
                    available_at=current.available_at,
                    from_label=previous.label,
                    to_label=current.label,
                    confidence=confidence,
                    metadata=dict(metadata or {}),
                )
            )
        previous = current
    return transitions


def build_regime_trace_summary(regimes, *, mode="preview", observation_columns=None, provenance=None, metadata=None):
    frame = regimes.copy() if isinstance(regimes, pd.DataFrame) else pd.DataFrame({"regime": pd.Series(regimes, copy=False)})
    evidence_class = resolve_regime_evidence_class(mode)
    if frame.empty:
        return RegimeTraceSummary(
            mode=mode,
            evidence_class=evidence_class,
            row_count=0,
            available_rows=0,
            transition_count=0,
            observation_columns=[str(item) for item in list(observation_columns or [])],
            state_columns=[str(item) for item in list(frame.columns)],
            provenance=dict(provenance or {}),
            metadata=dict(metadata or {}),
        )

    labels = coerce_regime_label_series(frame).dropna()
    distribution = labels.value_counts(dropna=True)
    transitions = build_regime_transition_contracts(frame, metadata=metadata)
    return RegimeTraceSummary(
        mode=mode,
        evidence_class=evidence_class,
        row_count=int(len(frame)),
        available_rows=int(frame.dropna(how="all").shape[0]),
        transition_count=int(len(transitions)),
        observation_columns=[str(item) for item in list(observation_columns or [])],
        state_columns=[str(item) for item in list(frame.columns)],
        label_distribution={str(key): int(value) for key, value in distribution.items()},
        dominant_label=(None if distribution.empty else str(distribution.index[0])),
        provenance=dict(provenance or {}),
        metadata=dict(metadata or {}),
    )


def summarize_regime_detection_result(result):
    payload = dict(result or {})
    observation_frame = payload.get("regime_observations")
    if observation_frame is None:
        observation_frame = payload.get("regime_features")
    observation_columns = list(pd.DataFrame(observation_frame).columns) if observation_frame is not None else []
    metadata = {
        str(key): value
        for key, value in payload.items()
        if key not in {"regimes", "regime_observations", "regime_features", "provenance"}
    }
    return build_regime_trace_summary(
        payload.get("regimes"),
        mode=str(payload.get("mode", "preview")),
        observation_columns=observation_columns,
        provenance=dict(payload.get("provenance") or {}),
        metadata=metadata,
    )


def _join_state_frame(base_frame, addition, source_name, source_map):
    if addition is None or addition.empty:
        return base_frame, source_map

    joined = base_frame.join(addition, how="outer") if not base_frame.empty else addition.copy()
    for column in addition.columns:
        source_map[column] = source_name
    return joined, source_map


def _resolve_regime_min_periods(window, floor=5):
    return int(max(floor, int(window) // 2))


def resolve_regime_evidence_class(mode):
    normalized = str(mode or "preview").strip().lower()
    if normalized == "global_preview_only":
        return "preview_only"
    if normalized in {"fold_local", "fold_local_replay", "walk_forward_replay"}:
        return "fold_local_oos"
    if normalized in {"locked_holdout", "locked_holdout_replay"}:
        return "locked_holdout"
    if normalized in {"live", "live_monitoring"}:
        return "live_monitoring"
    return normalized or "preview_only"


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
            "transition_count": 0,
            "mean_dwell_bars": None,
            "min_dwell_bars": None,
            "label_entropy": None,
            "dominant_share": None,
        }

    if len(labels) == 1:
        return {
            "available_rows": 1,
            "distinct_states": int(labels.nunique()),
            "switch_count": 0,
            "switch_rate": 0.0,
            "persistence": 1.0,
            "transition_count": 0,
            "mean_dwell_bars": 1.0,
            "min_dwell_bars": 1,
            "label_entropy": 0.0,
            "dominant_share": 1.0,
        }

    switches = labels.ne(labels.shift()).iloc[1:]
    switch_count = int(switches.sum())
    switch_rate = float(switch_count / max(1, len(labels) - 1))
    transition_count = switch_count
    dwell_lengths = labels.groupby(labels.ne(labels.shift()).cumsum()).size()
    distribution = labels.value_counts(normalize=True, dropna=True)
    label_entropy = float(-(distribution * np.log2(distribution.clip(lower=1e-12))).sum()) if not distribution.empty else None
    return {
        "available_rows": int(len(labels)),
        "distinct_states": int(labels.nunique()),
        "switch_count": switch_count,
        "switch_rate": switch_rate,
        "persistence": float(1.0 - switch_rate),
        "transition_count": int(transition_count),
        "mean_dwell_bars": float(dwell_lengths.mean()) if not dwell_lengths.empty else None,
        "min_dwell_bars": int(dwell_lengths.min()) if not dwell_lengths.empty else None,
        "label_entropy": label_entropy,
        "dominant_share": float(distribution.iloc[0]) if not distribution.empty else None,
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


def _replay_regime_detector_state(feature_set, *, detector_spec, config=None, fit_features=None):
    from .regimes.detectors import build_regime_detector
    from .regimes.online_state import replay_regime_detector_trace

    normalized = normalize_regime_feature_set(feature_set)
    fit_frame = None
    if fit_features is not None:
        fit_reference = normalize_regime_feature_set(
            {
                "frame": fit_features,
                "source_map": normalized.source_map,
            }
        )
        fit_frame = fit_reference.frame.reindex(columns=normalized.frame.columns)

    replay = replay_regime_detector_trace(
        normalized.frame,
        detector=build_regime_detector(
            detector_spec,
            config=config,
            source_map=normalized.source_map,
        ),
        source_map=normalized.source_map,
        provenance=normalized.provenance,
        fit_observations=fit_frame,
        mode="ablation_replay",
        metadata={"ablation": True},
    )
    return replay["state_frame"], replay


def build_regime_ablation_report(
    feature_set,
    *,
    n_regimes=2,
    method="hmm",
    config=None,
    fit_features=None,
    full_regimes=None,
    detector_spec=None,
):
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
        "incremental_evidence_gate": {
            "required": bool(dict(config or {}).get("require_incremental_evidence", False)),
            "min_accuracy_lift": float(dict(config or {}).get("min_accuracy_lift", 0.0)),
            "min_directional_accuracy_lift": float(dict(config or {}).get("min_directional_accuracy_lift", 0.0)),
            "max_fallback_row_share": dict(config or {}).get("max_fallback_row_share"),
            "min_label_entropy": float(dict(config or {}).get("min_label_entropy", 0.0)),
            "min_mean_dwell_bars": float(dict(config or {}).get("min_mean_dwell_bars", 1.0)),
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

    if detector_spec is not None:
        baseline_regimes, _ = _replay_regime_detector_state(
            endogenous_only,
            detector_spec=detector_spec,
            config=config,
            fit_features=endogenous_fit,
        )
        enriched_regimes = full_regimes if full_regimes is not None else _replay_regime_detector_state(
            normalized,
            detector_spec=detector_spec,
            config=config,
            fit_features=None if fit_reference is None else fit_reference.frame,
        )[0]
    else:
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


def summarize_regime_ablation_reports(reports, regime_aware_reports=None):
    rows = [row for row in (reports or []) if row]
    required_rows = [row for row in rows if row.get("stability_gate", {}).get("required")]
    failed_rows = [row for row in required_rows if not row.get("stability_gate", {}).get("passed", True)]
    agreement_rates = [row.get("agreement_rate") for row in rows if row.get("agreement_rate") is not None]
    improvements = [row.get("stability_improvement") for row in required_rows if row.get("stability_improvement") is not None]
    label_entropies = [
        row.get("full_stability", {}).get("label_entropy")
        for row in rows
        if row.get("full_stability", {}).get("label_entropy") is not None
    ]
    mean_dwell_bars = [
        row.get("full_stability", {}).get("mean_dwell_bars")
        for row in rows
        if row.get("full_stability", {}).get("mean_dwell_bars") is not None
    ]
    contextual_shares = [
        row.get("full_provenance", {}).get("contextual_share")
        for row in rows
        if row.get("full_provenance", {}).get("contextual_share") is not None
    ]
    regime_aware_rows = [row for row in (regime_aware_reports or []) if row]
    incremental_rows = [row.get("incremental_evidence") or {} for row in regime_aware_rows if row.get("incremental_evidence")]
    accuracy_lifts = [row.get("accuracy_lift") for row in incremental_rows if row.get("accuracy_lift") is not None]
    directional_accuracy_lifts = [
        row.get("directional_accuracy_lift")
        for row in incremental_rows
        if row.get("directional_accuracy_lift") is not None
    ]
    fallback_shares = [
        row.get("fallback_row_share")
        for row in incremental_rows
        if row.get("fallback_row_share") is not None
    ]
    max_fallback_row_share = max(fallback_shares) if fallback_shares else None

    incremental_required = any(bool(row.get("incremental_evidence_gate", {}).get("required", False)) for row in rows)
    min_accuracy_lift = max(
        [float(row.get("incremental_evidence_gate", {}).get("min_accuracy_lift", 0.0)) for row in rows] or [0.0]
    )
    min_directional_accuracy_lift = max(
        [float(row.get("incremental_evidence_gate", {}).get("min_directional_accuracy_lift", 0.0)) for row in rows] or [0.0]
    )
    min_label_entropy = max(
        [float(row.get("incremental_evidence_gate", {}).get("min_label_entropy", 0.0)) for row in rows] or [0.0]
    )
    min_mean_dwell = max(
        [float(row.get("incremental_evidence_gate", {}).get("min_mean_dwell_bars", 1.0)) for row in rows] or [1.0]
    )
    configured_fallback_limits = [
        row.get("incremental_evidence_gate", {}).get("max_fallback_row_share")
        for row in rows
        if row.get("incremental_evidence_gate", {}).get("max_fallback_row_share") is not None
    ]
    max_fallback_limit = min(float(value) for value in configured_fallback_limits) if configured_fallback_limits else None
    incremental_reasons = []
    if incremental_required and not incremental_rows:
        incremental_reasons.append("regime_incremental_evidence_missing")
    if incremental_rows and accuracy_lifts and float(np.mean(accuracy_lifts)) < float(min_accuracy_lift):
        incremental_reasons.append("regime_accuracy_lift_below_minimum")
    if incremental_rows and directional_accuracy_lifts and float(np.mean(directional_accuracy_lifts)) < float(min_directional_accuracy_lift):
        incremental_reasons.append("regime_directional_accuracy_lift_below_minimum")
    if max_fallback_limit is not None and max_fallback_row_share is not None and float(max_fallback_row_share) > float(max_fallback_limit):
        incremental_reasons.append("regime_fallback_share_above_limit")
    if label_entropies and float(np.mean(label_entropies)) < float(min_label_entropy):
        incremental_reasons.append("regime_label_entropy_below_minimum")
    if mean_dwell_bars and float(np.mean(mean_dwell_bars)) < float(min_mean_dwell):
        incremental_reasons.append("regime_mean_dwell_below_minimum")

    if not required_rows:
        status = "unknown"
        reasons = ["regime_ablation_evidence_missing"]
    elif failed_rows:
        status = "failed"
        reasons = ["regime_stability_failed"]
    elif incremental_reasons:
        status = "failed" if incremental_rows else "unknown"
        reasons = ["regime_incremental_evidence_failed"] + incremental_reasons
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
        "avg_label_entropy": float(np.mean(label_entropies)) if label_entropies else None,
        "avg_mean_dwell_bars": float(np.mean(mean_dwell_bars)) if mean_dwell_bars else None,
        "avg_accuracy_lift": float(np.mean(accuracy_lifts)) if accuracy_lifts else None,
        "avg_directional_accuracy_lift": (
            float(np.mean(directional_accuracy_lifts)) if directional_accuracy_lifts else None
        ),
        "avg_fallback_row_share": float(np.mean(fallback_shares)) if fallback_shares else None,
        "max_fallback_row_share": (None if max_fallback_row_share is None else float(max_fallback_row_share)),
        "incremental_evidence_required": incremental_required,
        "incremental_evidence_status": (
            "missing" if incremental_required and not incremental_rows else ("failed" if incremental_reasons else "passed")
        ),
        "incremental_evidence_reasons": incremental_reasons,
        "status": status,
        "promotion_pass": status == "passed",
        "reasons": reasons,
    }


__all__ = [
    "BaseRegimeDetector",
    "CONTEXTUAL_REGIME_SOURCES",
    "REGIME_CROSS_ASSET_SOURCE",
    "REGIME_INSTRUMENT_SOURCE",
    "REGIME_MARKET_SOURCE",
    "REGIME_SOURCE_ORDER",
    "RegimeDetectorManifest",
    "RegimeFeatureSet",
    "RegimeObservationContract",
    "RegimeStateContract",
    "RegimeTraceSummary",
    "RegimeTransitionContract",
    "build_cross_asset_regime_state",
    "build_default_regime_feature_set",
    "build_instrument_regime_state",
    "build_regime_observation_contracts",
    "build_market_regime_state",
    "build_regime_ablation_report",
    "build_regime_state_contracts",
    "build_regime_trace_summary",
    "build_regime_transition_contracts",
    "coerce_regime_label_series",
    "compute_regime_path_stability",
    "detect_regime",
    "infer_regime_source_map",
    "normalize_regime_feature_set",
    "resolve_regime_evidence_class",
    "summarize_regime_detection_result",
    "summarize_regime_ablation_reports",
    "summarize_regime_provenance",
]