"""Frozen global and per-regime z-score scaling helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .contracts import FeaturePolicyContract


_SCALING_MODE_ALIASES = {
    "identity": "identity",
    "none": "identity",
    "off": "identity",
    "global": "global",
    "standard": "global",
    "zscore": "global",
    "regime": "regime_conditioned",
    "regime_conditioned": "regime_conditioned",
    "regime_conditioned_zscore": "regime_conditioned",
}
_FALLBACK_MODE_ALIASES = {
    "global": "global",
    "identity": "identity",
    "none": "identity",
}
_PASSTHROUGH_COLUMNS = {
    "regime",
    "warm",
    "unavailable",
    "degenerate_fallback",
    "selected_column_count",
}


def _clone_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return {str(key): item for key, item in dict(value or {}).items()}


def _coerce_feature_frame(frame: Any) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    return pd.DataFrame(frame).copy()


def _coerce_regime_frame(frame: Any, index: pd.Index) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame(index=index)
    if isinstance(frame, pd.DataFrame):
        return frame.reindex(index).copy()
    if isinstance(frame, pd.Series):
        return frame.to_frame(name=frame.name or "regime").reindex(index).copy()
    return pd.DataFrame(frame).reindex(index).copy()


def canonicalize_scaling_mode(mode: Any) -> str:
    normalized = str(mode or "identity").strip().lower().replace("-", "_").replace(" ", "_")
    return _SCALING_MODE_ALIASES.get(normalized, normalized)


def canonicalize_fallback_mode(mode: Any) -> str:
    normalized = str(mode or "identity").strip().lower().replace("-", "_").replace(" ", "_")
    return _FALLBACK_MODE_ALIASES.get(normalized, normalized)


def _resolve_scaling_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    eligible = []
    passthrough = []
    for column in frame.columns:
        column_name = str(column)
        normalized = column_name.lower()
        if normalized in _PASSTHROUGH_COLUMNS or normalized.endswith("_regime"):
            passthrough.append(column_name)
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if numeric.notna().any():
            eligible.append(column_name)
        else:
            passthrough.append(column_name)
    return eligible, passthrough


def _resolve_regime_labels(regime_frame: pd.DataFrame, regime_column: str, index: pd.Index) -> pd.Series:
    if regime_frame.empty:
        return pd.Series(index=index, dtype=object)
    target_column = regime_column if regime_column in regime_frame.columns else regime_frame.columns[0]
    labels = pd.Series(regime_frame[target_column], index=index, dtype=object)
    return labels.where(~labels.isna(), None)


def _resolve_bool_series(regime_frame: pd.DataFrame, column: str, index: pd.Index, default: bool) -> pd.Series:
    if column not in regime_frame.columns:
        return pd.Series(default, index=index, dtype=bool)
    series = pd.Series(regime_frame[column], index=index)
    return series.fillna(default).astype(bool)


def _resolve_confidence_series(regime_frame: pd.DataFrame, index: pd.Index) -> pd.Series | None:
    if "regime_confidence" not in regime_frame.columns:
        return None
    return pd.to_numeric(regime_frame["regime_confidence"], errors="coerce").reindex(index)


def _normalize_regime_key(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except Exception:
            pass
    return str(value)


def _resolve_min_regime_samples(config: Mapping[str, Any]) -> int:
    scaling = _clone_mapping(config.get("scaling") or {})
    selection = _clone_mapping(config.get("selection") or {})
    candidates = [
        scaling.get("min_regime_samples"),
        selection.get("min_regime_samples"),
        config.get("default_min_regime_samples"),
        40,
    ]
    for value in candidates:
        if value is None:
            continue
        return max(1, int(value))
    return 40


def _resolve_confidence_floor(config: Mapping[str, Any]) -> float | None:
    scaling = _clone_mapping(config.get("scaling") or {})
    for value in [scaling.get("confidence_floor"), scaling.get("min_confidence")]:
        if value is not None:
            return float(value)
    return None


@dataclass
class FrozenScalerStats:
    columns: tuple[str, ...]
    means: pd.Series
    scales: pd.Series
    sample_count: int
    constant_columns: tuple[str, ...] = ()

    def transform_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.columns:
            return pd.DataFrame(index=frame.index)
        numeric = (
            frame.reindex(columns=list(self.columns))
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        transformed = numeric.sub(self.means, axis=1).div(self.scales, axis=1)
        return transformed.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    def manifest(self) -> dict[str, Any]:
        return {
            "sample_count": int(self.sample_count),
            "column_count": int(len(self.columns)),
            "constant_columns": list(self.constant_columns),
            "constant_column_count": int(len(self.constant_columns)),
        }


def _fit_frozen_scaler(frame: pd.DataFrame, columns: list[str]) -> FrozenScalerStats:
    if not columns:
        empty = pd.Series(dtype=float)
        return FrozenScalerStats(columns=(), means=empty, scales=empty, sample_count=int(len(frame)), constant_columns=())
    numeric = (
        frame.reindex(columns=columns)
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    means = numeric.mean(axis=0, skipna=True).fillna(0.0).astype(float)
    raw_scales = numeric.std(axis=0, ddof=0, skipna=True).fillna(0.0).astype(float)
    constant_mask = raw_scales.abs() <= 1e-12
    scales = raw_scales.where(~constant_mask, 1.0).astype(float)
    constant_columns = tuple(str(column) for column in raw_scales.index[constant_mask])
    return FrozenScalerStats(
        columns=tuple(str(column) for column in columns),
        means=means,
        scales=scales,
        sample_count=int(len(frame)),
        constant_columns=constant_columns,
    )


@dataclass
class RegimeConditionedScalingAdapter:
    config: Mapping[str, Any] | None = None
    regime_column: str = "regime"
    _feature_columns: list[str] = field(default_factory=list, init=False)
    _regime_columns: list[str] = field(default_factory=list, init=False)
    _feature_metadata_columns: int = field(default=0, init=False)
    _fit_row_count: int = field(default=0, init=False)
    _eligible_columns: tuple[str, ...] = field(default_factory=tuple, init=False)
    _passthrough_columns: tuple[str, ...] = field(default_factory=tuple, init=False)
    _global_scaler: FrozenScalerStats | None = field(default=None, init=False)
    _regime_scalers: dict[str, FrozenScalerStats] = field(default_factory=dict, init=False)
    _skipped_regimes: dict[str, str] = field(default_factory=dict, init=False)
    _regime_sample_counts: dict[str, int] = field(default_factory=dict, init=False)
    _fit_regime_labels: list[str] = field(default_factory=list, init=False)
    _fit_ready_rows: int = field(default=0, init=False)
    _fit_confident_rows: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.config = _clone_mapping(self.config)
        self.regime_column = str(self.regime_column or "regime")
        self._scaling_mode = canonicalize_scaling_mode(self.config.get("requested_scaling_mode"))
        self._fallback_mode = canonicalize_fallback_mode(self.config.get("fallback_mode"))
        self._min_regime_samples = _resolve_min_regime_samples(self.config)
        self._confidence_floor = _resolve_confidence_floor(self.config)
        if self._scaling_mode not in {"global", "regime_conditioned"}:
            raise ValueError(f"Unsupported feature_adaptation scaling mode={self._scaling_mode!r}")
        if self._fallback_mode not in {"global", "identity"}:
            raise ValueError(f"Unsupported feature_adaptation fallback mode={self._fallback_mode!r}")

    def fit(
        self,
        X: Any,
        regime_frame: Any,
        feature_metadata: Mapping[str, Any] | None = None,
    ) -> "RegimeConditionedScalingAdapter":
        X_frame = _coerce_feature_frame(X)
        regime = _coerce_regime_frame(regime_frame, X_frame.index)
        self._feature_columns = [str(column) for column in X_frame.columns]
        self._regime_columns = [str(column) for column in regime.columns]
        self._feature_metadata_columns = int(len(dict(feature_metadata or {})))
        self._fit_row_count = int(len(X_frame))
        eligible_columns, passthrough_columns = _resolve_scaling_columns(X_frame)
        self._eligible_columns = tuple(eligible_columns)
        self._passthrough_columns = tuple(passthrough_columns)
        self._global_scaler = _fit_frozen_scaler(X_frame, eligible_columns)
        self._regime_scalers = {}
        self._skipped_regimes = {}
        self._regime_sample_counts = {}

        labels = _resolve_regime_labels(regime, self.regime_column, X_frame.index)
        normalized_labels = labels.map(_normalize_regime_key)
        unique_labels = sorted({label for label in normalized_labels.dropna().tolist() if label is not None})
        self._fit_regime_labels = unique_labels
        warm = _resolve_bool_series(regime, "warm", X_frame.index, True)
        unavailable = _resolve_bool_series(regime, "unavailable", X_frame.index, False)
        ready_mask = labels.notna() & warm & ~unavailable
        self._fit_ready_rows = int(ready_mask.sum())
        confidence = _resolve_confidence_series(regime, X_frame.index)
        confident_mask = ready_mask.copy()
        if confidence is not None and self._confidence_floor is not None:
            confident_mask &= confidence.ge(self._confidence_floor).fillna(False)
        self._fit_confident_rows = int(confident_mask.sum())

        for regime_key in unique_labels:
            label_mask = normalized_labels.eq(regime_key)
            ready_label_mask = label_mask & ready_mask
            confident_label_mask = label_mask & confident_mask
            sample_count = int(confident_label_mask.sum())
            self._regime_sample_counts[regime_key] = sample_count
            if not self._eligible_columns:
                self._skipped_regimes[regime_key] = "no_eligible_scaling_columns"
                continue
            if sample_count >= self._min_regime_samples:
                self._regime_scalers[regime_key] = _fit_frozen_scaler(X_frame.loc[confident_label_mask], eligible_columns)
                continue
            if confidence is not None and self._confidence_floor is not None and int(ready_label_mask.sum()) > 0 and sample_count == 0:
                self._skipped_regimes[regime_key] = "confidence_below_floor"
                continue
            self._skipped_regimes[regime_key] = "insufficient_rows"
        return self

    def _row_reason(
        self,
        row_label: Any,
        *,
        warm: bool,
        unavailable: bool,
        confidence: float | None,
    ) -> tuple[str | None, str | None]:
        regime_key = _normalize_regime_key(row_label)
        if regime_key is None:
            return None, "missing_regime"
        if not warm or unavailable:
            return regime_key, "warm_or_unavailable"
        if self._confidence_floor is not None and confidence is not None and np.isfinite(confidence):
            if float(confidence) < float(self._confidence_floor):
                return regime_key, "confidence_below_floor"
        if self._confidence_floor is not None and confidence is not None and not np.isfinite(confidence):
            return regime_key, "confidence_below_floor"
        if regime_key in self._regime_scalers:
            return regime_key, None
        skipped_reason = self._skipped_regimes.get(regime_key)
        if skipped_reason == "insufficient_rows":
            return regime_key, "sparse_regime"
        if skipped_reason == "confidence_below_floor":
            return regime_key, "confidence_below_floor"
        if skipped_reason == "missing_regime_labels":
            return regime_key, "missing_regime"
        return regime_key, "missing_regime_bank"

    def transform(
        self,
        X: Any,
        regime_frame: Any,
    ) -> tuple[pd.DataFrame, FeaturePolicyContract]:
        X_frame = _coerce_feature_frame(X)
        regime = _coerce_regime_frame(regime_frame, X_frame.index)
        base_frame = X_frame.reindex(columns=self._feature_columns, fill_value=0.0).copy()
        transformed = base_frame.copy()
        labels = _resolve_regime_labels(regime, self.regime_column, transformed.index)
        warm = _resolve_bool_series(regime, "warm", transformed.index, True)
        unavailable = _resolve_bool_series(regime, "unavailable", transformed.index, False)
        confidence = _resolve_confidence_series(regime, transformed.index)

        regime_assignment_counts: Counter[str] = Counter()
        fallback_counts: Counter[str] = Counter()
        assigned_rows_by_regime: dict[str, list[Any]] = {}
        fallback_rows = 0

        if self._eligible_columns and self._global_scaler is not None:
            if self._scaling_mode == "global" or self._fallback_mode == "global":
                global_scaled = self._global_scaler.transform_frame(base_frame)
                transformed.loc[:, list(self._eligible_columns)] = global_scaled.loc[:, list(self._eligible_columns)]

            if self._scaling_mode == "regime_conditioned" and self._regime_scalers:
                for row_index in transformed.index:
                    regime_key, reason = self._row_reason(
                        labels.loc[row_index],
                        warm=bool(warm.loc[row_index]),
                        unavailable=bool(unavailable.loc[row_index]),
                        confidence=(None if confidence is None else confidence.loc[row_index]),
                    )
                    if reason is None and regime_key is not None:
                        assigned_rows_by_regime.setdefault(regime_key, []).append(row_index)
                        regime_assignment_counts[regime_key] += 1
                        continue
                    if reason is not None:
                        fallback_rows += 1
                        fallback_counts[reason] += 1
                        if self._fallback_mode == "identity":
                            fallback_counts["identity_fallback_configured"] += 1
                        else:
                            regime_assignment_counts["global_fallback"] += 1

                for regime_key, row_index in assigned_rows_by_regime.items():
                    if not row_index:
                        continue
                    scaler = self._regime_scalers[regime_key]
                    regime_scaled = scaler.transform_frame(base_frame.loc[row_index])
                    transformed.loc[row_index, list(self._eligible_columns)] = regime_scaled.loc[row_index, list(self._eligible_columns)]

        no_op = bool(not self._eligible_columns)
        if self._eligible_columns and self._scaling_mode == "regime_conditioned":
            no_op = bool(sum(regime_assignment_counts.values()) == 0 and self._fallback_mode == "identity")
        if self._eligible_columns and self._scaling_mode == "global":
            no_op = False
        if self._eligible_columns and self._scaling_mode == "regime_conditioned" and self._fallback_mode == "global":
            no_op = False

        manifest = self.manifest()
        sparse_regimes = sorted(
            regime_key for regime_key, reason in self._skipped_regimes.items() if reason == "insufficient_rows"
        )
        policy = FeaturePolicyContract(
            policy_id=str(manifest.get("policy_id", "regime_conditioned_scaling")),
            feature_columns=[str(column) for column in transformed.columns],
            disabled_columns=[],
            generated_columns=[],
            regime_column=self.regime_column,
            scaling_mode=self._scaling_mode,
            fallback_mode=self._fallback_mode,
            sparse_regimes=sparse_regimes,
            metadata={
                "adapter_type": manifest.get("adapter_type"),
                "no_op": no_op,
                "requested_enabled": bool(self.config.get("enabled", False)),
                "requested_scaling_mode": str(self.config.get("requested_scaling_mode", self._scaling_mode)),
                "requested_sections": list(self.config.get("requested_sections") or []),
                "deferred_runtime": bool(self.config.get("deferred_runtime", False)),
                "fit_row_count": int(self._fit_row_count),
                "transform_row_count": int(len(transformed)),
                "regime_row_count": int(len(regime)),
                "eligible_scaling_columns": list(self._eligible_columns),
                "eligible_scaling_column_count": int(len(self._eligible_columns)),
                "passthrough_columns": list(self._passthrough_columns),
                "passthrough_column_count": int(len(self._passthrough_columns)),
                "global_bank": manifest.get("global_bank"),
                "regime_bank_count": int(manifest.get("regime_bank_count", 0)),
                "regime_assignment_counts": dict(sorted(regime_assignment_counts.items())),
                "fallback_rows_total": int(fallback_rows),
                "fallback_rows_by_reason": dict(sorted(fallback_counts.items())),
                "skipped_regimes": dict(self._skipped_regimes),
                "constant_columns": list(manifest.get("constant_columns") or []),
                "constant_column_count": int(manifest.get("constant_column_count", 0)),
                "confidence_floor": self._confidence_floor,
                "min_regime_samples": int(self._min_regime_samples),
            },
        )
        return transformed, policy

    def manifest(self) -> dict[str, Any]:
        global_bank_manifest = self._global_scaler.manifest() if self._global_scaler is not None else {
            "sample_count": 0,
            "column_count": 0,
            "constant_columns": [],
            "constant_column_count": 0,
        }
        regime_banks = {
            str(regime_key): scaler.manifest()
            for regime_key, scaler in sorted(self._regime_scalers.items())
        }
        constant_columns = sorted(
            {
                *list(global_bank_manifest.get("constant_columns") or []),
                *(column for bank in regime_banks.values() for column in list(bank.get("constant_columns") or [])),
            }
        )
        return {
            "adapter_type": "regime_conditioned_scaling",
            "policy_id": "regime_conditioned_scaling",
            "regime_column": self.regime_column,
            "fit_row_count": int(self._fit_row_count),
            "feature_columns": list(self._feature_columns),
            "feature_count": int(len(self._feature_columns)),
            "regime_columns": list(self._regime_columns),
            "regime_column_count": int(len(self._regime_columns)),
            "feature_metadata_columns": int(self._feature_metadata_columns),
            "requested_enabled": bool(self.config.get("enabled", False)),
            "requested_scaling_mode": str(self.config.get("requested_scaling_mode", self._scaling_mode)),
            "fallback_mode": self._fallback_mode,
            "requested_sections": list(self.config.get("requested_sections") or []),
            "deferred_runtime": bool(self.config.get("deferred_runtime", False)),
            "no_op": bool(not self._eligible_columns),
            "scaling_mode": self._scaling_mode,
            "eligible_scaling_columns": list(self._eligible_columns),
            "eligible_scaling_column_count": int(len(self._eligible_columns)),
            "passthrough_columns": list(self._passthrough_columns),
            "passthrough_column_count": int(len(self._passthrough_columns)),
            "global_bank": global_bank_manifest,
            "regime_banks": regime_banks,
            "regime_bank_count": int(len(regime_banks)),
            "skipped_regimes": dict(self._skipped_regimes),
            "skipped_regime_count": int(len(self._skipped_regimes)),
            "regime_sample_counts": dict(sorted(self._regime_sample_counts.items())),
            "fit_regime_labels": list(self._fit_regime_labels),
            "fit_ready_rows": int(self._fit_ready_rows),
            "fit_confident_rows": int(self._fit_confident_rows),
            "min_regime_samples": int(self._min_regime_samples),
            "confidence_floor": self._confidence_floor,
            "constant_columns": constant_columns,
            "constant_column_count": int(len(constant_columns)),
        }


__all__ = [
    "RegimeConditionedScalingAdapter",
    "canonicalize_fallback_mode",
    "canonicalize_scaling_mode",
]