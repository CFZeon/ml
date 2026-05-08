"""Regime detectors for the Phase 1 replay runtime."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

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
_NATIVE_DETECTOR_TYPES = {"break", "liquidity", "trend", "volatility"}


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
    standardized = numeric.sub(means, fill_value=0.0).div(stds.replace(0, 1), fill_value=1.0)
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

        runtime_state["position"] = int(runtime_state.get("position", 0)) + 1
        return runtime_state, RegimeStateContract(
            as_of=observation.as_of,
            available_at=observation.available_at,
            label=int(label),
            probabilities=({} if not warm else {str(int(label)): 1.0}),
            confidence=(None if not warm else 1.0),
            detector_outputs=detector_outputs,
            warm=bool(warm),
            metadata={
                "detector_name": self.detector_name,
                "detector_type": self.detector_type,
                "selected_columns": list(self._selected_columns),
                "negative_columns": list(self._negative_columns),
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
class ExplicitCompatibilityRegimeDetector:
    config: Mapping[str, Any] | None = None
    detector_name: str = "compatibility_explicit"
    column_name: str = "regime"
    _fit_frame: pd.DataFrame = field(default_factory=pd.DataFrame, init=False, repr=False)
    _fit_window: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def fit(self, observations: Any) -> "ExplicitCompatibilityRegimeDetector":
        self._fit_frame = _normalize_frame(observations)
        self._fit_window = _fit_window_payload(self._fit_frame)
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
        if label is None and detector_outputs:
            label = detector_outputs[next(iter(detector_outputs))]

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
                "compatibility_mode": "legacy_explicit",
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
            metadata={"compatibility_mode": "legacy_explicit"},
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
    raise ValueError(
        f"Unsupported regime detector type={resolved_spec.get('type')!r}. "
        "Supported Slice 3 replay detectors: ['explicit', 'trend', 'volatility', 'liquidity', 'break']."
    )


__all__ = [
    "BreakRegimeDetector",
    "ExplicitCompatibilityRegimeDetector",
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