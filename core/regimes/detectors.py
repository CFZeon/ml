"""Regime detectors for the Phase 1 replay runtime."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..regime import detect_regime
from .contracts import RegimeDetectorManifest, RegimeObservationContract, RegimeStateContract


_DETECTOR_TYPE_ALIASES = {
    "break": "break",
    "break_state": "break",
    "compatibility": "compatibility_explicit",
    "compatibility_explicit": "compatibility_explicit",
    "explicit": "compatibility_explicit",
    "filtered_hmm": "filtered_hmm",
    "hmm": "filtered_hmm",
    "liquidity": "liquidity",
    "liquidity_regime": "liquidity",
    "liquidity_state": "liquidity",
    "structural_break": "break",
    "structural_break_state": "break",
    "trend": "trend",
    "trend_regime": "trend",
    "trend_state": "trend",
    "vol": "volatility",
    "volatility": "volatility",
    "volatility_regime": "volatility",
    "volatility_state": "volatility",
    "volatility_trend_hybrid": "compatibility_explicit",
}
_NATIVE_DETECTOR_TYPES = {"break", "filtered_hmm", "liquidity", "trend", "volatility"}


def canonicalize_regime_detector_type(detector_type: Any) -> str:
    normalized = str(detector_type or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _DETECTOR_TYPE_ALIASES.get(normalized, normalized)


def is_native_regime_detector_type(detector_type: Any) -> bool:
    return canonicalize_regime_detector_type(detector_type) in _NATIVE_DETECTOR_TYPES


def is_native_regime_detector_spec(spec: Mapping[str, Any] | None) -> bool:
    return is_native_regime_detector_type(dict(spec or {}).get("type"))


def can_replay_regime_detector_type(detector_type: Any) -> bool:
    canonical = canonicalize_regime_detector_type(detector_type)
    return canonical == "compatibility_explicit" or canonical in _NATIVE_DETECTOR_TYPES


def can_replay_regime_detector_spec(spec: Mapping[str, Any] | None) -> bool:
    return can_replay_regime_detector_type(dict(spec or {}).get("type"))


def _normalize_frame(observations: Any) -> pd.DataFrame:
    if observations is None:
        return pd.DataFrame()
    if isinstance(observations, pd.DataFrame):
        return observations.copy()
    return pd.DataFrame(observations).copy()


def _coerce_scalar(value: Any) -> Any:
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


def _fit_window_payload(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"row_count": 0, "start": None, "end": None}
    return {
        "row_count": int(len(frame)),
        "start": frame.index[0],
        "end": frame.index[-1],
    }


def _coerce_sequence(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value if str(item))
    return (str(value),)


def _to_numeric_series(row: Mapping[str, Any], columns: Sequence[str]) -> pd.Series:
    return pd.to_numeric(pd.Series({column: row.get(column) for column in columns}), errors="coerce")


def _freeze_numeric_stats(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[pd.Series, pd.Series]:
    if not columns:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    numeric = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    means = numeric.mean().astype(float)
    stds = numeric.std().replace(0, 1).fillna(1.0).astype(float)
    return means, stds


def _score_with_frozen_stats(
    frame: pd.DataFrame,
    columns: Sequence[str],
    means: pd.Series,
    stds: pd.Series,
) -> pd.Series:
    if not columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    numeric = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    standardized = numeric.sub(means, axis=1).div(stds, axis=1)
    return standardized.mean(axis=1, skipna=True).fillna(0.0).astype(float)


def _score_row_with_frozen_stats(
    row: Mapping[str, Any],
    columns: Sequence[str],
    means: pd.Series,
    stds: pd.Series,
) -> tuple[float, bool]:
    if not columns:
        return 0.0, False
    numeric = _to_numeric_series(row, columns)
    standardized = numeric.sub(means.reindex(numeric.index)).div(stds.reindex(numeric.index).replace(0, 1).fillna(1.0))
    has_data = bool(standardized.notna().any())
    if not has_data:
        return 0.0, False
    return float(standardized.mean(skipna=True)), True


def _select_detector_columns(
    frame: pd.DataFrame,
    *,
    include_terms: Sequence[str] = (),
    exclude_terms: Sequence[str] = (),
    explicit_columns: Sequence[str] = (),
    fallback_column: str | None = None,
    allowed_sources: Sequence[str] = (),
    source_map: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    source_map = dict(source_map or {})
    allowed_source_set = {str(item) for item in allowed_sources if str(item)}
    if explicit_columns:
        columns = [column for column in explicit_columns if column in frame.columns]
    elif not include_terms:
        columns = []
    else:
        columns = []
        for column in frame.columns:
            normalized = str(column).lower()
            if include_terms and not any(term in normalized for term in include_terms):
                continue
            if exclude_terms and any(term in normalized for term in exclude_terms):
                continue
            if allowed_source_set and source_map.get(column) not in allowed_source_set:
                continue
            columns.append(column)

    if not columns and fallback_column is not None and fallback_column in frame.columns:
        if not allowed_source_set or source_map.get(fallback_column) in allowed_source_set:
            columns = [fallback_column]
    return tuple(columns)


def _summarize_selected_sources(columns: Sequence[str], source_map: Mapping[str, str] | None) -> dict[str, int]:
    counter = Counter(str((source_map or {}).get(column, "unknown")) for column in columns)
    return {key: int(value) for key, value in sorted(counter.items())}


def _logsumexp(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return float("-inf")
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("-inf")
    max_value = float(finite.max())
    return float(max_value + np.log(np.exp(array - max_value).sum()))


def _safe_log_probabilities(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    safe = np.full(array.shape, float("-inf"), dtype=float)
    positive = array > 0
    safe[positive] = np.log(array[positive])
    return safe


def _select_hmm_columns(
    frame: pd.DataFrame,
    *,
    explicit_columns: Sequence[str] = (),
    exclude_columns: Sequence[str] = (),
    allowed_sources: Sequence[str] = (),
    source_map: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    source_map = dict(source_map or {})
    excluded = {str(item) for item in exclude_columns if str(item)}
    allowed_source_set = {str(item) for item in allowed_sources if str(item)}

    if explicit_columns:
        selected = []
        for column in explicit_columns:
            if column not in frame.columns or column in excluded:
                continue
            if allowed_source_set and source_map.get(column) not in allowed_source_set:
                continue
            selected.append(column)
        return tuple(selected)

    selected = []
    for column in frame.columns:
        if column in excluded:
            continue
        if allowed_source_set and source_map.get(column) not in allowed_source_set:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().any():
            selected.append(column)
    return tuple(selected)


def _coerce_numeric_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(index=frame.index)
    return frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")


def _diag_gaussian_log_likelihood(
    observation: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
) -> np.ndarray:
    raw_covars = np.asarray(covars, dtype=float)
    if raw_covars.ndim == 3:
        variances = np.diagonal(raw_covars, axis1=1, axis2=2)
    else:
        variances = raw_covars
    variances = np.maximum(variances, 1e-12)
    centered = observation[None, :] - np.asarray(means, dtype=float)
    quadratic = ((centered ** 2) / variances).sum(axis=1)
    log_det = np.log(variances).sum(axis=1)
    dimension = float(observation.shape[0])
    return -0.5 * (dimension * np.log(2.0 * np.pi) + log_det + quadratic)


def _deterministic_hmm_state_order(means: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(np.asarray(means, dtype=float), ord=1, axis=1)
    ordered_state_indices = np.argsort(norms)
    raw_to_ordered = np.empty(len(ordered_state_indices), dtype=int)
    for ordered_label, raw_label in enumerate(ordered_state_indices):
        raw_to_ordered[int(raw_label)] = int(ordered_label)
    return ordered_state_indices.astype(int), raw_to_ordered.astype(int)


def _sanitize_semantic_feature_name(name: Any) -> str:
    text = str(name or "feature").strip().lower()
    cleaned = "".join(character if character.isalnum() else "_" for character in text).strip("_")
    return cleaned or "feature"


def _sanitize_regime_token(value: Any) -> str:
    text = str(value or "state").strip().lower()
    cleaned = "".join(character if character.isalnum() else "_" for character in text).strip("_")
    return cleaned or "state"


def _explicit_bucket_token(value: Any) -> str:
    try:
        numeric_value = int(value)
    except (TypeError, ValueError):
        return _sanitize_regime_token(value)
    return {
        -1: "low",
        0: "mid",
        1: "high",
    }.get(numeric_value, _sanitize_regime_token(f"bucket_{numeric_value}"))


def _default_explicit_canonical_regime_id(trend_regime: Any, volatility_regime: Any, liquidity_regime: Any) -> str:
    return (
        "compatibility_explicit"
        f"__trend_{_explicit_bucket_token(trend_regime)}"
        f"__volatility_{_explicit_bucket_token(volatility_regime)}"
        f"__liquidity_{_explicit_bucket_token(liquidity_regime)}"
    )


def _build_explicit_taxonomy_registry(state_frame: pd.DataFrame) -> dict[str, Any]:
    if state_frame.empty:
        return {
            "version": "compatibility_explicit.taxonomy.v1",
            "identity_basis": "explicit_bucket_state",
            "state_map": {},
        }

    required_columns = ["trend_regime", "volatility_regime", "liquidity_regime", "regime"]
    available_columns = [column for column in required_columns if column in state_frame.columns]
    if len(available_columns) != len(required_columns):
        return {
            "version": "compatibility_explicit.taxonomy.v1",
            "identity_basis": "explicit_bucket_state",
            "state_map": {},
        }

    taxonomy_frame = state_frame.loc[:, required_columns].dropna().drop_duplicates().sort_values(by="regime")
    state_map = {}
    for row in taxonomy_frame.itertuples(index=False):
        canonical_regime_id = _default_explicit_canonical_regime_id(
            row.trend_regime,
            row.volatility_regime,
            row.liquidity_regime,
        )
        state_map[str(int(row.regime))] = {
            "legacy_regime_label": int(row.regime),
            "canonical_regime_id": canonical_regime_id,
            "routing_regime_id": canonical_regime_id,
            "semantic_label": (
                f"trend_{_explicit_bucket_token(row.trend_regime)}"
                f"__volatility_{_explicit_bucket_token(row.volatility_regime)}"
                f"__liquidity_{_explicit_bucket_token(row.liquidity_regime)}"
            ),
            "state_signature_summary": {
                "trend_regime": int(row.trend_regime),
                "volatility_regime": int(row.volatility_regime),
                "liquidity_regime": int(row.liquidity_regime),
            },
            "mapping_confidence": 1.0,
            "remap_reason": "deterministic_bucket_identity",
        }

    return {
        "version": "compatibility_explicit.taxonomy.v1",
        "identity_basis": "explicit_bucket_state",
        "state_map": state_map,
    }


def _attach_explicit_taxonomy_columns(state_frame: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(state_frame).copy()
    if frame.empty:
        frame["canonical_regime_id"] = pd.Series(dtype=object)
        frame["routing_regime_id"] = pd.Series(dtype=object)
        return frame

    if not {"trend_regime", "volatility_regime", "liquidity_regime"}.issubset(frame.columns):
        frame["canonical_regime_id"] = pd.Series([None] * len(frame), index=frame.index, dtype=object)
        frame["routing_regime_id"] = pd.Series([None] * len(frame), index=frame.index, dtype=object)
        return frame

    canonical_regime_ids = []
    for row in frame.itertuples():
        if pd.isna(getattr(row, "trend_regime")) or pd.isna(getattr(row, "volatility_regime")) or pd.isna(getattr(row, "liquidity_regime")):
            canonical_regime_ids.append(None)
            continue
        canonical_regime_ids.append(
            _default_explicit_canonical_regime_id(
                getattr(row, "trend_regime"),
                getattr(row, "volatility_regime"),
                getattr(row, "liquidity_regime"),
            )
        )
    frame["canonical_regime_id"] = pd.Series(canonical_regime_ids, index=frame.index, dtype=object)
    frame["routing_regime_id"] = pd.Series(canonical_regime_ids, index=frame.index, dtype=object)
    return frame


def _default_hmm_canonical_regime_id(detector_name: Any, ordered_state_id: int) -> str:
    return f"filtered_hmm__{_sanitize_regime_token(detector_name)}__state_{int(ordered_state_id)}"


def _new_hmm_regime_family_id(detector_name: Any, ordered_state_id: int) -> str:
    return f"filtered_hmm__{_sanitize_regime_token(detector_name)}__new_family__state_{int(ordered_state_id)}"


def _bucket_semantic_state_value(value: float, all_values: np.ndarray) -> str:
    values = np.asarray(all_values, dtype=float)
    if values.size <= 1 or np.allclose(values, values[0]):
        return "neutral"
    lower = float(np.quantile(values, 1.0 / 3.0))
    upper = float(np.quantile(values, 2.0 / 3.0))
    if value <= lower:
        return "low"
    if value >= upper:
        return "high"
    return "mid"


def _build_hmm_semantic_state_map(ordered_means: np.ndarray, feature_names: Sequence[str]) -> dict[int, dict[str, Any]]:
    means = np.asarray(ordered_means, dtype=float)
    if means.ndim != 2 or means.size == 0:
        return {}

    resolved_names = [
        _sanitize_semantic_feature_name(feature_names[index] if index < len(feature_names) else f"feature_{index}")
        for index in range(means.shape[1])
    ]
    state_map = {}
    seen_labels = {}
    for state_index in range(means.shape[0]):
        feature_signature = {}
        state_signature_summary = {}
        label_parts = []
        for feature_index, feature_name in enumerate(resolved_names):
            value = float(means[state_index, feature_index])
            bucket = _bucket_semantic_state_value(value, means[:, feature_index])
            feature_signature[feature_name] = bucket
            state_signature_summary[feature_name] = round(value, 6)
            label_parts.append(f"{feature_name}_{bucket}")
        semantic_label = "hmm__" + "__".join(label_parts)
        duplicate_count = seen_labels.get(semantic_label, 0)
        seen_labels[semantic_label] = duplicate_count + 1
        if duplicate_count > 0:
            semantic_label = f"{semantic_label}__variant_{duplicate_count}"
        state_map[int(state_index)] = {
            "semantic_label": semantic_label,
            "feature_signature": feature_signature,
            "state_signature_summary": state_signature_summary,
        }
    return state_map


def _feature_signature_similarity(left: Mapping[str, Any] | None, right: Mapping[str, Any] | None) -> float:
    left_map = {str(key): str(value) for key, value in dict(left or {}).items()}
    right_map = {str(key): str(value) for key, value in dict(right or {}).items()}
    keys = set(left_map.keys()) | set(right_map.keys())
    if not keys:
        return 0.0
    matches = sum(1 for key in keys if left_map.get(key) == right_map.get(key))
    return float(matches / len(keys))


def _coerce_hmm_reference_state_map(config: Mapping[str, Any] | None, detector_name: Any) -> dict[int, dict[str, Any]]:
    params = dict(config or {})
    reference_manifest = dict(
        params.get("reference_detector_manifest")
        or params.get("reference_manifest")
        or {}
    )
    reference_metadata = dict(reference_manifest.get("metadata") or {})
    raw_state_map = (
        params.get("reference_canonical_state_map")
        or reference_metadata.get("canonical_state_map")
        or {}
    )
    normalized = {}
    for raw_state_index, payload in dict(raw_state_map or {}).items():
        try:
            state_index = int(raw_state_index)
        except (TypeError, ValueError):
            continue
        entry = dict(payload or {})
        normalized[state_index] = {
            "canonical_regime_id": str(
                entry.get("canonical_regime_id") or _default_hmm_canonical_regime_id(detector_name, state_index)
            ),
            "semantic_label": (
                None if entry.get("semantic_label") is None else str(entry.get("semantic_label"))
            ),
            "feature_signature": {
                str(key): str(value) for key, value in dict(entry.get("feature_signature") or {}).items()
            },
            "state_signature_summary": {
                str(key): float(value)
                for key, value in dict(entry.get("state_signature_summary") or {}).items()
                if value is not None
            },
        }
    return normalized


def _build_hmm_canonical_state_map(
    ordered_means: np.ndarray,
    feature_names: Sequence[str],
    *,
    detector_name: Any,
    semantic_state_map: Mapping[int, Mapping[str, Any]] | None,
    reference_state_map: Mapping[int, Mapping[str, Any]] | None = None,
    similarity_threshold: float = 0.75,
) -> dict[int, dict[str, Any]]:
    del ordered_means, feature_names
    semantic_map = {int(key): dict(value or {}) for key, value in dict(semantic_state_map or {}).items()}
    reference_map = {int(key): dict(value or {}) for key, value in dict(reference_state_map or {}).items()}
    used_reference_states: set[int] = set()
    canonical_state_map = {}

    for state_index, semantic_state in sorted(semantic_map.items()):
        feature_signature = {
            str(key): str(value) for key, value in dict(semantic_state.get("feature_signature") or {}).items()
        }
        state_signature_summary = {
            str(key): float(value)
            for key, value in dict(semantic_state.get("state_signature_summary") or {}).items()
            if value is not None
        }
        semantic_label = str(semantic_state.get("semantic_label") or f"hmm__state_{state_index}")

        best_reference_index = None
        best_similarity = -1.0
        for reference_index, reference_state in reference_map.items():
            if reference_index in used_reference_states:
                continue
            similarity = _feature_signature_similarity(feature_signature, reference_state.get("feature_signature"))
            if similarity > best_similarity:
                best_similarity = similarity
                best_reference_index = int(reference_index)

        if best_reference_index is not None and best_similarity >= float(similarity_threshold):
            reference_state = dict(reference_map.get(best_reference_index) or {})
            canonical_regime_id = str(
                reference_state.get("canonical_regime_id")
                or _default_hmm_canonical_regime_id(detector_name, best_reference_index)
            )
            remap_reason = "matched_prior_signature"
            mapping_confidence = float(best_similarity)
            used_reference_states.add(best_reference_index)
        elif reference_map:
            canonical_regime_id = _new_hmm_regime_family_id(detector_name, state_index)
            remap_reason = "new_regime_family"
            mapping_confidence = float(max(best_similarity, 0.0))
        else:
            canonical_regime_id = _default_hmm_canonical_regime_id(detector_name, state_index)
            remap_reason = "initial_fit"
            mapping_confidence = 1.0

        canonical_state_map[int(state_index)] = {
            "canonical_regime_id": canonical_regime_id,
            "semantic_label": semantic_label,
            "feature_signature": feature_signature,
            "state_signature_summary": state_signature_summary,
            "similarity_score": (
                None if best_similarity < 0.0 else round(float(best_similarity), 6)
            ),
            "mapping_confidence": round(float(mapping_confidence), 6),
            "remap_reason": remap_reason,
        }
    return canonical_state_map


def _summarize_hmm_taxonomy_stability(
    canonical_state_map: Mapping[int, Mapping[str, Any]] | None,
    *,
    reference_available: bool,
) -> dict[str, Any]:
    entries = [dict(value or {}) for value in dict(canonical_state_map or {}).values()]
    total_states = int(len(entries))
    if total_states <= 0:
        return {
            "reference_available": bool(reference_available),
            "neighboring_refit_agreement": None,
            "mean_state_signature_similarity": None,
            "remap_rate": None,
            "unresolved_new_state_rate": None,
            "compatibility_break_count": 0,
        }

    matched_count = sum(1 for entry in entries if entry.get("remap_reason") == "matched_prior_signature")
    unresolved_count = sum(1 for entry in entries if entry.get("remap_reason") == "new_regime_family")
    similarity_values = [
        float(entry.get("similarity_score"))
        for entry in entries
        if entry.get("similarity_score") is not None
    ]
    return {
        "reference_available": bool(reference_available),
        "neighboring_refit_agreement": (
            None if not reference_available else round(float(matched_count / total_states), 6)
        ),
        "mean_state_signature_similarity": (
            None if not similarity_values else round(float(np.mean(similarity_values)), 6)
        ),
        "remap_rate": (
            None if not reference_available else round(float(matched_count / total_states), 6)
        ),
        "unresolved_new_state_rate": (
            None if not reference_available else round(float(unresolved_count / total_states), 6)
        ),
        "compatibility_break_count": int(unresolved_count),
    }


def _forward_filter_step(
    emission_log_likelihood: np.ndarray,
    *,
    log_startprob: np.ndarray,
    log_transmat: np.ndarray,
    previous_filtered_log_prob: np.ndarray | None,
) -> np.ndarray:
    emission = np.asarray(emission_log_likelihood, dtype=float)
    if previous_filtered_log_prob is None:
        unnormalized = log_startprob + emission
    else:
        transition_terms = previous_filtered_log_prob[:, None] + log_transmat
        predicted = np.asarray([_logsumexp(transition_terms[:, column]) for column in range(transition_terms.shape[1])])
        unnormalized = predicted + emission

    normalizer = _logsumexp(unnormalized)
    if not np.isfinite(normalizer):
        raise ValueError("Filtered HMM forward step produced a non-finite normalizer")
    return unnormalized - normalizer


def _select_primary_regime_detector(regime_config: Mapping[str, Any]) -> dict[str, Any] | None:
    detectors = [dict(item) for item in list(regime_config.get("detectors") or []) if isinstance(item, Mapping)]
    if not detectors:
        return None

    enabled = [detector for detector in detectors if detector.get("enabled", True) is not False]
    if not enabled:
        return None

    ensemble = dict(regime_config.get("ensemble") or {})
    primary_name = str(ensemble.get("primary_detector", "") or "").strip()
    if primary_name:
        for detector in enabled:
            if str(detector.get("name", "") or "").strip() == primary_name:
                return detector

    for detector in enabled:
        if bool(detector.get("primary", False)):
            return detector
    return enabled[0]


def resolve_authoritative_regime_detector_spec(regime_config: Mapping[str, Any]) -> dict[str, Any] | None:
    detectors = [dict(item) for item in list(regime_config.get("detectors") or []) if isinstance(item, Mapping)]
    if not detectors:
        return None

    enabled = [detector for detector in detectors if detector.get("enabled", True) is not False]
    native_enabled = [detector for detector in enabled if is_native_regime_detector_spec(detector)]
    if len(native_enabled) > 1:
        names = [str(detector.get("name") or detector.get("type") or "detector") for detector in native_enabled]
        raise ValueError(
            "Phase 1 Slice 3 supports at most one enabled native regime detector before detector fusion lands: "
            f"{names}"
        )
    return _select_primary_regime_detector(regime_config)


@dataclass
class _NativeScoreRegimeDetector:
    config: Mapping[str, Any] | None = None
    detector_name: str = "detector"
    detector_type: str = "native"
    column_name: str = "regime"
    state_column: str = "regime"
    include_terms: tuple[str, ...] = ()
    exclude_terms: tuple[str, ...] = ()
    negative_terms: tuple[str, ...] = ()
    negative_exclude_terms: tuple[str, ...] = ()
    fallback_column: str | None = None
    upper_tail_only: bool = False
    source_map: Mapping[str, str] | None = None
    _fit_frame: pd.DataFrame = field(default_factory=pd.DataFrame, init=False, repr=False)
    _fit_window: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _selected_columns: tuple[str, ...] = field(default_factory=tuple, init=False, repr=False)
    _negative_columns: tuple[str, ...] = field(default_factory=tuple, init=False, repr=False)
    _column_means: pd.Series = field(default_factory=lambda: pd.Series(dtype=float), init=False, repr=False)
    _column_stds: pd.Series = field(default_factory=lambda: pd.Series(dtype=float), init=False, repr=False)
    _negative_means: pd.Series = field(default_factory=lambda: pd.Series(dtype=float), init=False, repr=False)
    _negative_stds: pd.Series = field(default_factory=lambda: pd.Series(dtype=float), init=False, repr=False)
    _thresholds: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _selected_source_counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.config or {})

    def _resolve_positive_columns(self, frame: pd.DataFrame) -> tuple[str, ...]:
        params = self.params
        return _select_detector_columns(
            frame,
            include_terms=_coerce_sequence(params.get("include_terms") or self.include_terms),
            exclude_terms=_coerce_sequence(params.get("exclude_terms") or self.exclude_terms),
            explicit_columns=_coerce_sequence(params.get("columns") or params.get("positive_columns")),
            fallback_column=params.get("fallback_column", self.fallback_column),
            allowed_sources=_coerce_sequence(params.get("allowed_sources")),
            source_map=self.source_map,
        )

    def _resolve_negative_columns(self, frame: pd.DataFrame) -> tuple[str, ...]:
        params = self.params
        return _select_detector_columns(
            frame,
            include_terms=_coerce_sequence(params.get("negative_terms") or self.negative_terms),
            exclude_terms=_coerce_sequence(params.get("negative_exclude_terms") or self.negative_exclude_terms),
            explicit_columns=_coerce_sequence(params.get("negative_columns")),
            allowed_sources=_coerce_sequence(params.get("allowed_sources")),
            source_map=self.source_map,
        )

    def _score_frame(self, frame: pd.DataFrame) -> pd.Series:
        positive = _score_with_frozen_stats(frame, self._selected_columns, self._column_means, self._column_stds)
        negative = _score_with_frozen_stats(frame, self._negative_columns, self._negative_means, self._negative_stds)
        score = positive - negative
        if bool(self.params.get("invert", False)):
            score = -score
        return score.astype(float)

    def _score_observation(self, observation: RegimeObservationContract) -> tuple[float, bool]:
        positive_score, positive_available = _score_row_with_frozen_stats(
            observation.values,
            self._selected_columns,
            self._column_means,
            self._column_stds,
        )
        negative_score, negative_available = _score_row_with_frozen_stats(
            observation.values,
            self._negative_columns,
            self._negative_means,
            self._negative_stds,
        )
        score = positive_score - negative_score
        if bool(self.params.get("invert", False)):
            score = -score
        return float(score), bool(positive_available or negative_available)

    def _fit_thresholds(self, fit_scores: pd.Series) -> dict[str, float]:
        clean = fit_scores.dropna()
        if clean.empty:
            return {}
        if self.upper_tail_only:
            upper_quantile = float(self.params.get("upper_quantile", 0.67))
            return {"trigger_threshold": float(clean.quantile(upper_quantile))}
        lower_quantile = float(self.params.get("lower_quantile", 0.33))
        upper_quantile = float(self.params.get("upper_quantile", 0.67))
        return {
            "lower_threshold": float(clean.quantile(lower_quantile)),
            "upper_threshold": float(clean.quantile(upper_quantile)),
        }

    def _emit_label(self, score: float, *, warm: bool) -> int:
        if not warm or not self._thresholds:
            return 0
        if self.upper_tail_only:
            return int(score >= float(self._thresholds.get("trigger_threshold", float("inf"))))
        lower = float(self._thresholds.get("lower_threshold", float("-inf")))
        upper = float(self._thresholds.get("upper_threshold", float("inf")))
        if score <= lower:
            return -1
        if score >= upper:
            return 1
        return 0

    def fit(self, observations: Any) -> "_NativeScoreRegimeDetector":
        self._fit_frame = _normalize_frame(observations)
        self._fit_window = _fit_window_payload(self._fit_frame)
        self._selected_columns = self._resolve_positive_columns(self._fit_frame)
        self._negative_columns = self._resolve_negative_columns(self._fit_frame)
        self._column_means, self._column_stds = _freeze_numeric_stats(self._fit_frame, self._selected_columns)
        self._negative_means, self._negative_stds = _freeze_numeric_stats(self._fit_frame, self._negative_columns)
        fit_scores = self._score_frame(self._fit_frame)
        self._thresholds = self._fit_thresholds(fit_scores)
        selected_for_summary = tuple(dict.fromkeys([*self._selected_columns, *self._negative_columns]))
        self._selected_source_counts = _summarize_selected_sources(selected_for_summary, self.source_map)
        return self

    def initialize(self, observations: Any | None = None) -> dict[str, Any]:
        return {"position": 0}

    def update(
        self,
        state: Any,
        observation: RegimeObservationContract,
    ) -> tuple[Any, RegimeStateContract]:
        runtime_state = dict(state or {})
        score, has_data = self._score_observation(observation)
        selected_column_count = len(self._selected_columns) + len(self._negative_columns)
        warm = bool(selected_column_count > 0 and has_data and self._thresholds)
        label = self._emit_label(score, warm=warm)
        detector_outputs = {
            "score": float(score),
            "selected_column_count": int(selected_column_count),
            "warm": int(warm),
            self.state_column: int(label),
            self.column_name: int(label),
        }
        for key, value in self._thresholds.items():
            detector_outputs[str(key)] = float(value)

        availability_state = "known" if warm else "warm"
        availability_reason = None
        confidence = None if not warm else 1.0
        probabilities = {} if not warm else {str(int(label)): 1.0}
        if selected_column_count <= 0:
            availability_state = "unavailable"
            availability_reason = "no_selected_columns"
            detector_outputs["unavailable"] = 1
            confidence = 0.0
        elif not has_data:
            availability_reason = "missing_observation"
            detector_outputs["missing_observation"] = 1
        elif not self._thresholds:
            availability_reason = "thresholds_unavailable"
            detector_outputs["thresholds_unavailable"] = 1

        runtime_state["position"] = int(runtime_state.get("position", 0)) + 1
        return runtime_state, RegimeStateContract(
            as_of=observation.as_of,
            available_at=observation.available_at,
            label=int(label),
            probabilities=probabilities,
            confidence=confidence,
            detector_outputs=detector_outputs,
            warm=bool(warm),
            metadata={
                "detector_name": self.detector_name,
                "detector_type": self.detector_type,
                "selected_columns": list(self._selected_columns),
                "negative_columns": list(self._negative_columns),
                "availability_state": availability_state,
                **({"reason": availability_reason} if availability_reason else {}),
            },
        )

    def manifest(self) -> RegimeDetectorManifest:
        params = dict(self.params)
        return RegimeDetectorManifest(
            detector_name=self.detector_name,
            detector_type=self.detector_type,
            params={
                key: value
                for key, value in {
                    "lower_quantile": params.get("lower_quantile", 0.33),
                    "upper_quantile": params.get("upper_quantile", 0.67),
                    "invert": bool(params.get("invert", False)),
                    "allowed_sources": list(_coerce_sequence(params.get("allowed_sources"))),
                }.items()
            },
            warmup_bars=(
                None
                if params.get("warmup_bars") is None and params.get("feature_lookback") is None
                else int(params.get("warmup_bars", params.get("feature_lookback")))
            ),
            fit_window=dict(self._fit_window),
            metadata={
                "selected_columns": list(self._selected_columns),
                "negative_columns": list(self._negative_columns),
                "selected_source_counts": dict(self._selected_source_counts),
                "state_column": self.state_column,
            },
        )


@dataclass
class TrendRegimeDetector(_NativeScoreRegimeDetector):
    detector_type: str = "trend"
    state_column: str = "trend_regime"
    include_terms: tuple[str, ...] = ("trend", "ret_", "return", "momentum", "slope")
    exclude_terms: tuple[str, ...] = ("vol", "volume", "liquid", "break", "shock")


@dataclass
class VolatilityRegimeDetector(_NativeScoreRegimeDetector):
    detector_type: str = "volatility"
    state_column: str = "volatility_regime"
    include_terms: tuple[str, ...] = ("vol", "range", "atr", "dispersion", "cluster", "drawdown", "shock")
    exclude_terms: tuple[str, ...] = ("volume", "liquid")


@dataclass
class LiquidityRegimeDetector(_NativeScoreRegimeDetector):
    detector_type: str = "liquidity"
    state_column: str = "liquidity_regime"
    include_terms: tuple[str, ...] = ("liquid", "volume", "turnover", "trade")
    exclude_terms: tuple[str, ...] = ("illiquid",)
    negative_terms: tuple[str, ...] = ("illiquid", "amihud")


@dataclass
class BreakRegimeDetector(_NativeScoreRegimeDetector):
    detector_type: str = "break"
    state_column: str = "structural_break_regime"
    include_terms: tuple[str, ...] = ("break", "shock", "jump", "crash", "drawdown")
    exclude_terms: tuple[str, ...] = ("volume", "liquid")
    upper_tail_only: bool = True


@dataclass
class FilteredHMMDetector:
    config: Mapping[str, Any] | None = None
    detector_name: str = "filtered_hmm"
    column_name: str = "regime"
    source_map: Mapping[str, str] | None = None
    _fit_frame: pd.DataFrame = field(default_factory=pd.DataFrame, init=False, repr=False)
    _fit_window: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _selected_columns: tuple[str, ...] = field(default_factory=tuple, init=False, repr=False)
    _selected_source_counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _scaler_mean: np.ndarray = field(default_factory=lambda: np.array([], dtype=float), init=False, repr=False)
    _scaler_scale: np.ndarray = field(default_factory=lambda: np.array([], dtype=float), init=False, repr=False)
    _startprob: np.ndarray = field(default_factory=lambda: np.array([], dtype=float), init=False, repr=False)
    _transmat: np.ndarray = field(default_factory=lambda: np.array([[]], dtype=float), init=False, repr=False)
    _means: np.ndarray = field(default_factory=lambda: np.array([[]], dtype=float), init=False, repr=False)
    _covars: np.ndarray = field(default_factory=lambda: np.array([[]], dtype=float), init=False, repr=False)
    _ordered_state_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int), init=False, repr=False)
    _raw_to_ordered: np.ndarray = field(default_factory=lambda: np.array([], dtype=int), init=False, repr=False)
    _fit_status: str = field(default="unfitted", init=False, repr=False)
    _fallback_reason: str | None = field(default=None, init=False, repr=False)
    _posterior_mode: str = field(default="filtered", init=False, repr=False)
    _state_count: int = field(default=0, init=False, repr=False)
    _clean_fit_rows: int = field(default=0, init=False, repr=False)
    _covariance_type: str = field(default="diag", init=False, repr=False)
    _semantic_state_map: dict[int, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _semantic_schema_version: str = field(default="filtered_hmm.semantic.v1", init=False, repr=False)
    _canonical_state_map: dict[int, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _canonical_schema_version: str = field(default="filtered_hmm.canonical.v1", init=False, repr=False)
    _taxonomy_stability_report: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.config or {})

    def _fallback_to_one_state(self, reason: str) -> "FilteredHMMDetector":
        self._fit_status = "fallback"
        self._fallback_reason = reason
        self._state_count = 1
        self._startprob = np.array([1.0], dtype=float)
        self._transmat = np.array([[1.0]], dtype=float)
        self._means = np.zeros((1, len(self._selected_columns)), dtype=float)
        self._covars = np.ones((1, len(self._selected_columns)), dtype=float)
        self._ordered_state_indices = np.array([0], dtype=int)
        self._raw_to_ordered = np.array([0], dtype=int)
        self._semantic_state_map = _build_hmm_semantic_state_map(
            self._means[self._ordered_state_indices],
            self._selected_columns,
        )
        reference_state_map = _coerce_hmm_reference_state_map(self.params, self.detector_name)
        self._canonical_state_map = _build_hmm_canonical_state_map(
            self._means[self._ordered_state_indices],
            self._selected_columns,
            detector_name=self.detector_name,
            semantic_state_map=self._semantic_state_map,
            reference_state_map=reference_state_map,
            similarity_threshold=float(self.params.get("canonical_similarity_threshold", 0.75)),
        )
        self._taxonomy_stability_report = _summarize_hmm_taxonomy_stability(
            self._canonical_state_map,
            reference_available=bool(reference_state_map),
        )
        return self

    def fit(self, observations: Any) -> "FilteredHMMDetector":
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "hmmlearn is required for FilteredHMMDetector. Install it with: pip install hmmlearn>=0.3"
            ) from exc

        params = self.params
        self._fit_frame = _normalize_frame(observations)
        self._fit_window = _fit_window_payload(self._fit_frame)
        self._covariance_type = str(params.get("covariance_type", "diag") or "diag").strip().lower()
        if self._covariance_type != "diag":
            raise ValueError(
                "FilteredHMMDetector currently supports covariance_type='diag' only. "
                f"Got {self._covariance_type!r}."
            )

        self._selected_columns = _select_hmm_columns(
            self._fit_frame,
            explicit_columns=_coerce_sequence(params.get("columns")),
            exclude_columns=_coerce_sequence(params.get("exclude_columns")),
            allowed_sources=_coerce_sequence(params.get("allowed_sources")),
            source_map=self.source_map,
        )
        self._selected_source_counts = _summarize_selected_sources(self._selected_columns, self.source_map)
        if not self._selected_columns:
            allow_fallback = bool(params.get("allow_empty_schema_fallback", True))
            if not allow_fallback:
                raise ValueError("FilteredHMMDetector resolved an empty observation schema")
            return self._fallback_to_one_state("empty_schema")

        numeric_fit = _coerce_numeric_frame(self._fit_frame, self._selected_columns)
        clean_fit = numeric_fit.dropna()
        self._clean_fit_rows = int(len(clean_fit))

        requested_state_count = int(params.get("state_count") or params.get("n_regimes") or 2)
        requested_state_count = max(1, requested_state_count)
        if clean_fit.empty:
            return self._fallback_to_one_state("empty_fit_window")

        state_count = max(1, min(requested_state_count, len(clean_fit)))
        if state_count <= 1:
            return self._fallback_to_one_state("insufficient_fit_rows")

        scaler = StandardScaler()
        scaler.fit(clean_fit)
        self._scaler_mean = np.asarray(scaler.mean_, dtype=float)
        self._scaler_scale = np.asarray(scaler.scale_, dtype=float)
        self._scaler_scale = np.where(self._scaler_scale == 0, 1.0, self._scaler_scale)

        normed_fit = scaler.transform(clean_fit)
        hmm_model = GaussianHMM(
            n_components=state_count,
            covariance_type=self._covariance_type,
            n_iter=int(params.get("n_iter", 100)),
            tol=float(params.get("tol", 1e-3)),
            random_state=int(params.get("random_state", 42)),
        )
        try:
            hmm_model.fit(normed_fit)
        except Exception:  # noqa: BLE001
            return self._fallback_to_one_state("fit_failure")

        self._fit_status = "fit"
        self._fallback_reason = None
        self._state_count = int(state_count)
        self._startprob = np.asarray(hmm_model.startprob_, dtype=float)
        self._transmat = np.asarray(hmm_model.transmat_, dtype=float)
        self._means = np.asarray(hmm_model.means_, dtype=float)
        self._covars = np.asarray(hmm_model.covars_, dtype=float)
        self._ordered_state_indices, self._raw_to_ordered = _deterministic_hmm_state_order(self._means)
        self._semantic_state_map = _build_hmm_semantic_state_map(
            self._means[self._ordered_state_indices],
            self._selected_columns,
        )
        reference_state_map = _coerce_hmm_reference_state_map(self.params, self.detector_name)
        self._canonical_state_map = _build_hmm_canonical_state_map(
            self._means[self._ordered_state_indices],
            self._selected_columns,
            detector_name=self.detector_name,
            semantic_state_map=self._semantic_state_map,
            reference_state_map=reference_state_map,
            similarity_threshold=float(self.params.get("canonical_similarity_threshold", 0.75)),
        )
        self._taxonomy_stability_report = _summarize_hmm_taxonomy_stability(
            self._canonical_state_map,
            reference_available=bool(reference_state_map),
        )
        return self

    def initialize(self, observations: Any | None = None) -> dict[str, Any]:
        del observations
        return {"position": 0, "filtered_log_prob": None}

    def _emit_unavailable_contract(
        self,
        observation: RegimeObservationContract,
        *,
        runtime_state: dict[str, Any],
        reason: str,
    ) -> tuple[dict[str, Any], RegimeStateContract]:
        detector_outputs = {
            "regime_confidence": 0.0,
            "selected_column_count": int(len(self._selected_columns)),
            "warm": 0,
            "degenerate_fallback": int(self._fit_status == "fallback"),
            "unavailable": 1,
        }
        for state_label in range(max(1, self._state_count)):
            detector_outputs[f"prob_state_{state_label}"] = 0.0

        runtime_state["position"] = int(runtime_state.get("position", 0)) + 1
        return runtime_state, RegimeStateContract(
            as_of=observation.as_of,
            available_at=observation.available_at,
            label=None,
            probabilities={},
            confidence=0.0,
            detector_outputs=detector_outputs,
            warm=False,
            metadata={
                "detector_name": self.detector_name,
                "detector_type": "filtered_hmm",
                "posterior_mode": self._posterior_mode,
                "semantic_schema_version": self._semantic_schema_version,
                "reason": reason,
                "selected_columns": list(self._selected_columns),
            },
        )

    def update(
        self,
        state: Any,
        observation: RegimeObservationContract,
    ) -> tuple[Any, RegimeStateContract]:
        runtime_state = dict(state or {})
        selected_column_count = len(self._selected_columns)
        warmup_bars = self.params.get("warmup_bars")

        if self._fit_status == "unfitted":
            return self._emit_unavailable_contract(observation, runtime_state=runtime_state, reason="unfitted")

        if self._fit_status == "fallback":
            position = int(runtime_state.get("position", 0))
            warm = warmup_bars is None or position + 1 >= int(warmup_bars)
            semantic_state = dict(self._semantic_state_map.get(0) or {})
            canonical_state = dict(self._canonical_state_map.get(0) or {})
            semantic_label = semantic_state.get("semantic_label", "hmm__fallback_state")
            canonical_regime_id = str(
                canonical_state.get("canonical_regime_id")
                or _default_hmm_canonical_regime_id(self.detector_name, 0)
            )
            detector_outputs = {
                "regime_confidence": 1.0,
                "selected_column_count": int(selected_column_count),
                "warm": int(warm),
                "degenerate_fallback": 1,
                "latent_regime_id": 0,
                "semantic_regime": semantic_label,
                "canonical_regime_id": canonical_regime_id,
                "prob_state_0": 1.0,
            }
            if warm:
                detector_outputs[self.column_name] = semantic_label
            runtime_state["position"] = position + 1
            runtime_state["filtered_log_prob"] = np.array([0.0], dtype=float)
            return runtime_state, RegimeStateContract(
                as_of=observation.as_of,
                available_at=observation.available_at,
                label=(semantic_label if warm else None),
                probabilities=({semantic_label: 1.0} if warm else {}),
                confidence=(1.0 if warm else None),
                detector_outputs=detector_outputs,
                warm=bool(warm),
                metadata={
                    "detector_name": self.detector_name,
                    "detector_type": "filtered_hmm",
                    "posterior_mode": self._posterior_mode,
                    "semantic_schema_version": self._semantic_schema_version,
                    "canonical_schema_version": self._canonical_schema_version,
                    "semantic_label": semantic_label,
                    "canonical_regime_id": canonical_regime_id,
                    "semantic_signature": dict(semantic_state.get("feature_signature") or {}),
                    "state_signature_summary": dict(semantic_state.get("state_signature_summary") or {}),
                    "mapping_confidence": canonical_state.get("mapping_confidence"),
                    "similarity_score": canonical_state.get("similarity_score"),
                    "remap_reason": canonical_state.get("remap_reason"),
                    "reason": self._fallback_reason,
                    "selected_columns": list(self._selected_columns),
                },
            )

        row = _to_numeric_series(observation.values, self._selected_columns)
        if row.isna().any():
            return self._emit_unavailable_contract(observation, runtime_state=runtime_state, reason="missing_observation")

        scaled = (row.to_numpy(dtype=float) - self._scaler_mean) / self._scaler_scale
        emission_log_likelihood = _diag_gaussian_log_likelihood(scaled, self._means, self._covars)
        filtered_log_prob = _forward_filter_step(
            emission_log_likelihood,
            log_startprob=_safe_log_probabilities(self._startprob),
            log_transmat=_safe_log_probabilities(self._transmat),
            previous_filtered_log_prob=runtime_state.get("filtered_log_prob"),
        )
        runtime_state["filtered_log_prob"] = filtered_log_prob
        runtime_state["position"] = int(runtime_state.get("position", 0)) + 1

        raw_probabilities = np.exp(filtered_log_prob)
        ordered_probabilities = raw_probabilities[self._ordered_state_indices]
        ordered_state_id = int(np.argmax(ordered_probabilities)) if ordered_probabilities.size else 0
        confidence = float(ordered_probabilities[ordered_state_id]) if ordered_probabilities.size else 0.0
        warm = warmup_bars is None or int(runtime_state["position"]) >= int(warmup_bars)
        semantic_state = dict(self._semantic_state_map.get(ordered_state_id) or {})
        canonical_state = dict(self._canonical_state_map.get(ordered_state_id) or {})
        semantic_label = semantic_state.get("semantic_label", f"hmm__state_{ordered_state_id}")
        canonical_regime_id = str(
            canonical_state.get("canonical_regime_id")
            or _default_hmm_canonical_regime_id(self.detector_name, ordered_state_id)
        )
        detector_outputs = {
            "regime_confidence": float(confidence),
            "selected_column_count": int(selected_column_count),
            "warm": int(warm),
            "log_evidence": float(_logsumexp(emission_log_likelihood)),
            "degenerate_fallback": 0,
            "latent_regime_id": int(ordered_state_id),
            "semantic_regime": semantic_label,
            "canonical_regime_id": canonical_regime_id,
        }
        if warm:
            detector_outputs[self.column_name] = semantic_label
        probability_payload = {}
        for state_label, probability in enumerate(ordered_probabilities):
            semantic_entry = dict(self._semantic_state_map.get(int(state_label)) or {})
            semantic_key = str(semantic_entry.get("semantic_label", f"hmm__state_{int(state_label)}"))
            detector_outputs[f"prob_state_{state_label}"] = float(probability)
            probability_payload[semantic_key] = float(probability)

        return runtime_state, RegimeStateContract(
            as_of=observation.as_of,
            available_at=observation.available_at,
            label=(semantic_label if warm else None),
            probabilities=probability_payload,
            confidence=(float(confidence) if warm else None),
            detector_outputs=detector_outputs,
            warm=bool(warm),
            metadata={
                "detector_name": self.detector_name,
                "detector_type": "filtered_hmm",
                "posterior_mode": self._posterior_mode,
                "semantic_schema_version": self._semantic_schema_version,
                "canonical_schema_version": self._canonical_schema_version,
                "semantic_label": semantic_label,
                "canonical_regime_id": canonical_regime_id,
                "semantic_signature": dict(semantic_state.get("feature_signature") or {}),
                "state_signature_summary": dict(semantic_state.get("state_signature_summary") or {}),
                "mapping_confidence": canonical_state.get("mapping_confidence"),
                "similarity_score": canonical_state.get("similarity_score"),
                "remap_reason": canonical_state.get("remap_reason"),
                "selected_columns": list(self._selected_columns),
            },
        )

    def manifest(self) -> RegimeDetectorManifest:
        params = dict(self.params)
        metadata = {
            "posterior_mode": self._posterior_mode,
            "fit_status": self._fit_status,
            "fallback_reason": self._fallback_reason,
            "selected_columns": list(self._selected_columns),
            "selected_source_counts": dict(self._selected_source_counts),
            "semantic_schema_version": self._semantic_schema_version,
            "semantic_state_map": {
                str(int(state_index)): {
                    "semantic_label": str(payload.get("semantic_label")),
                    "feature_signature": dict(payload.get("feature_signature") or {}),
                    "state_signature_summary": dict(payload.get("state_signature_summary") or {}),
                }
                for state_index, payload in dict(self._semantic_state_map or {}).items()
            },
            "canonical_schema_version": self._canonical_schema_version,
            "canonical_state_map": {
                str(int(state_index)): {
                    "canonical_regime_id": str(payload.get("canonical_regime_id")),
                    "semantic_label": str(payload.get("semantic_label")),
                    "feature_signature": dict(payload.get("feature_signature") or {}),
                    "state_signature_summary": dict(payload.get("state_signature_summary") or {}),
                    "similarity_score": payload.get("similarity_score"),
                    "mapping_confidence": payload.get("mapping_confidence"),
                    "remap_reason": payload.get("remap_reason"),
                }
                for state_index, payload in dict(self._canonical_state_map or {}).items()
            },
            "taxonomy_stability_report": dict(self._taxonomy_stability_report or {}),
            "state_remap": {
                str(int(raw_state)): int(ordered_state)
                for raw_state, ordered_state in enumerate(self._raw_to_ordered.tolist() if self._raw_to_ordered.size else [])
            },
            "clean_fit_rows": int(self._clean_fit_rows),
        }
        return RegimeDetectorManifest(
            detector_name=self.detector_name,
            detector_type="filtered_hmm",
            params={
                "n_regimes": int(self.params.get("state_count") or self.params.get("n_regimes") or max(1, self._state_count or 1)),
                "covariance_type": self._covariance_type,
                "n_iter": int(params.get("n_iter", 100)),
                "tol": float(params.get("tol", 1e-3)),
                "random_state": int(params.get("random_state", 42)),
            },
            warmup_bars=(
                None
                if params.get("warmup_bars") is None and params.get("feature_lookback") is None
                else int(params.get("warmup_bars", params.get("feature_lookback")))
            ),
            fit_window=dict(self._fit_window),
            metadata=metadata,
        )


@dataclass
class ExplicitCompatibilityRegimeDetector:
    config: Mapping[str, Any] | None = None
    detector_name: str = "compatibility_explicit"
    column_name: str = "regime"
    _fit_frame: pd.DataFrame = field(default_factory=pd.DataFrame, init=False, repr=False)
    _fit_window: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _taxonomy_registry: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def fit(self, observations: Any) -> "ExplicitCompatibilityRegimeDetector":
        self._fit_frame = _normalize_frame(observations)
        self._fit_window = _fit_window_payload(self._fit_frame)
        fitted_state_frame = detect_regime(
            self._fit_frame,
            n_regimes=int(dict(self.config or {}).get("n_regimes", 2)),
            method="explicit",
            config=dict(self.config or {}),
            fit_features=self._fit_frame,
        )
        self._taxonomy_registry = _build_explicit_taxonomy_registry(fitted_state_frame)
        return self

    def initialize(self, observations: Any | None = None) -> dict[str, Any]:
        replay_frame = _normalize_frame(observations)
        reference = self._fit_frame if not self._fit_frame.empty else replay_frame
        state_frame = detect_regime(
            replay_frame,
            n_regimes=int(dict(self.config or {}).get("n_regimes", 2)),
            method="explicit",
            config=dict(self.config or {}),
            fit_features=reference,
        )
        state_frame = _attach_explicit_taxonomy_columns(state_frame)
        if not self._taxonomy_registry:
            self._taxonomy_registry = _build_explicit_taxonomy_registry(state_frame)
        return {
            "position": 0,
            "state_frame": state_frame.reindex(replay_frame.index),
        }

    def update(
        self,
        state: Any,
        observation: RegimeObservationContract,
    ) -> tuple[Any, RegimeStateContract]:
        runtime_state = dict(state or {})
        position = int(runtime_state.get("position", 0))
        state_frame = runtime_state.get("state_frame")
        if not isinstance(state_frame, pd.DataFrame):
            state_frame = pd.DataFrame()

        if observation.as_of in state_frame.index:
            row = state_frame.loc[observation.as_of]
        else:
            row = pd.Series(dtype=float)

        detector_outputs = {
            str(column): _coerce_scalar(value)
            for column, value in row.items()
            if _coerce_scalar(value) is not None
        }
        label = detector_outputs.get(self.column_name)
        if label is None:
            label = detector_outputs.get("regime")
        if label is None:
            label = detector_outputs.get("canonical_regime_id")
        if label is None and detector_outputs:
            label = detector_outputs[next(iter(detector_outputs))]

        canonical_regime_id = detector_outputs.get("canonical_regime_id")

        runtime_state["position"] = position + 1
        return runtime_state, RegimeStateContract(
            as_of=observation.as_of,
            available_at=observation.available_at,
            label=label,
            probabilities=({} if label is None else {str(label): 1.0}),
            confidence=(None if label is None else 1.0),
            detector_outputs=detector_outputs,
            warm=bool(detector_outputs),
            metadata={
                "detector_name": self.detector_name,
                "compatibility_mode": "canonical_explicit",
                "canonical_regime_id": canonical_regime_id,
                "routing_regime_id": detector_outputs.get("routing_regime_id"),
            },
        )

    def manifest(self) -> RegimeDetectorManifest:
        config = dict(self.config or {})
        return RegimeDetectorManifest(
            detector_name=self.detector_name,
            detector_type="compatibility_explicit",
            params={
                "method": "explicit",
                "n_regimes": int(config.get("n_regimes", 2)),
                "lower_quantile": float(config.get("lower_quantile", 0.33)),
                "upper_quantile": float(config.get("upper_quantile", 0.67)),
                "liquidity_invert": bool(config.get("liquidity_invert", False)),
            },
            warmup_bars=(None if config.get("feature_lookback") is None else int(config.get("feature_lookback"))),
            fit_window=dict(self._fit_window),
            metadata={
                "compatibility_mode": "canonical_explicit",
                "taxonomy_registry": dict(self._taxonomy_registry or {}),
            },
        )


def build_compatibility_regime_detector(
    config: Mapping[str, Any] | None = None,
    *,
    detector_name: str | None = None,
) -> ExplicitCompatibilityRegimeDetector:
    resolved_config = dict(config or {})
    compatibility = dict(resolved_config.get("compatibility_adapter") or {})
    resolved_name = (
        detector_name
        or compatibility.get("primary_detector")
        or compatibility.get("detector_name")
        or "compatibility_explicit"
    )
    return ExplicitCompatibilityRegimeDetector(
        config=resolved_config,
        detector_name=str(resolved_name),
        column_name=str(resolved_config.get("column_name", "regime")),
    )


def build_regime_detector(
    spec: Mapping[str, Any] | None,
    *,
    config: Mapping[str, Any] | None = None,
    source_map: Mapping[str, str] | None = None,
):
    resolved_config = dict(config or {})
    resolved_spec = dict(spec or {})
    detector_name = str(
        resolved_spec.get("name")
        or resolved_spec.get("type")
        or resolved_config.get("compatibility_adapter", {}).get("primary_detector")
        or "detector"
    )
    canonical_type = canonicalize_regime_detector_type(resolved_spec.get("type") or resolved_config.get("method") or "explicit")

    if canonical_type == "compatibility_explicit":
        return build_compatibility_regime_detector(resolved_config, detector_name=detector_name)

    merged_params = dict(resolved_config)
    merged_params.update(dict(resolved_spec.get("params") or {}))
    if resolved_spec.get("warmup_bars") is not None:
        merged_params["warmup_bars"] = int(resolved_spec.get("warmup_bars"))
    detector_kwargs = {
        "config": merged_params,
        "detector_name": detector_name,
        "column_name": str(resolved_config.get("column_name", "regime")),
        "source_map": source_map,
    }

    if canonical_type == "trend":
        return TrendRegimeDetector(**detector_kwargs)
    if canonical_type == "volatility":
        return VolatilityRegimeDetector(**detector_kwargs)
    if canonical_type == "liquidity":
        return LiquidityRegimeDetector(**detector_kwargs)
    if canonical_type == "break":
        return BreakRegimeDetector(**detector_kwargs)
    if canonical_type == "filtered_hmm":
        return FilteredHMMDetector(**detector_kwargs)
    raise ValueError(
        f"Unsupported regime detector type={resolved_spec.get('type')!r}. "
        "Supported replay detectors: ['explicit', 'trend', 'volatility', 'liquidity', 'break', 'filtered_hmm']."
    )


__all__ = [
    "BreakRegimeDetector",
    "ExplicitCompatibilityRegimeDetector",
    "FilteredHMMDetector",
    "LiquidityRegimeDetector",
    "TrendRegimeDetector",
    "VolatilityRegimeDetector",
    "build_compatibility_regime_detector",
    "build_regime_detector",
    "can_replay_regime_detector_spec",
    "can_replay_regime_detector_type",
    "canonicalize_regime_detector_type",
    "is_native_regime_detector_spec",
    "is_native_regime_detector_type",
    "resolve_authoritative_regime_detector_spec",
]