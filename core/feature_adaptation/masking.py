"""Frozen global and per-regime masking helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .contracts import FeaturePolicyContract


_SELECTION_MODE_ALIASES = {
    "identity": "identity",
    "mask": "per_regime_mask",
    "none": "identity",
    "off": "identity",
    "per_regime_mask": "per_regime_mask",
}
_SELECTION_FALLBACK_MODE_ALIASES = {
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


def canonicalize_selection_mode(mode: Any) -> str:
    normalized = str(mode or "identity").strip().lower().replace("-", "_").replace(" ", "_")
    return _SELECTION_MODE_ALIASES.get(normalized, normalized)


def canonicalize_selection_fallback_mode(mode: Any) -> str:
    normalized = str(mode or "identity").strip().lower().replace("-", "_").replace(" ", "_")
    return _SELECTION_FALLBACK_MODE_ALIASES.get(normalized, normalized)


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
        selection.get("min_regime_samples"),
        scaling.get("min_regime_samples"),
        config.get("default_min_regime_samples"),
        40,
    ]
    for value in candidates:
        if value is not None:
            return max(1, int(value))
    return 40


def _resolve_confidence_floor(config: Mapping[str, Any]) -> float | None:
    selection = _clone_mapping(config.get("selection") or {})
    scaling = _clone_mapping(config.get("scaling") or {})
    for value in [selection.get("confidence_floor"), selection.get("min_confidence"), scaling.get("confidence_floor")]:
        if value is not None:
            return float(value)
    return None


def _resolve_min_feature_rows(config: Mapping[str, Any]) -> int:
    selection = _clone_mapping(config.get("selection") or {})
    return max(1, int(selection.get("min_feature_rows", 24)))


def _resolve_min_active_share(config: Mapping[str, Any]) -> float:
    selection = _clone_mapping(config.get("selection") or {})
    return max(0.0, float(selection.get("min_active_share", 0.05)))


def _resolve_min_variance(config: Mapping[str, Any]) -> float:
    selection = _clone_mapping(config.get("selection") or {})
    return max(0.0, float(selection.get("min_variance", 1e-12)))


def _resolve_activity_epsilon(config: Mapping[str, Any]) -> float:
    selection = _clone_mapping(config.get("selection") or {})
    return max(0.0, float(selection.get("activity_epsilon", 1e-12)))


def _resolve_mask_columns(frame: pd.DataFrame, regime_frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    mask_candidates = []
    passthrough = []
    regime_columns = {str(column) for column in regime_frame.columns}
    for column in frame.columns:
        column_name = str(column)
        normalized = column_name.lower()
        if (
            column_name in regime_columns
            or normalized in _PASSTHROUGH_COLUMNS
            or normalized.endswith("_regime")
        ):
            passthrough.append(column_name)
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if numeric.notna().any():
            mask_candidates.append(column_name)
        else:
            passthrough.append(column_name)
    return mask_candidates, passthrough


@dataclass(frozen=True)
class ColumnActivityStats:
    finite_count: int
    active_count: int
    active_share: float
    std: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "finite_count": int(self.finite_count),
            "active_count": int(self.active_count),
            "active_share": float(self.active_share),
            "std": float(self.std),
        }


def _compute_column_activity(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    activity_epsilon: float,
) -> dict[str, ColumnActivityStats]:
    if not columns:
        return {}
    numeric = (
        frame.reindex(columns=columns)
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    stats = {}
    for column in columns:
        series = numeric[column]
        finite = series.dropna()
        finite_count = int(finite.shape[0])
        active_count = int((finite.abs() > float(activity_epsilon)).sum())
        active_share = float(active_count / finite_count) if finite_count > 0 else 0.0
        std = float(finite.std(ddof=0)) if finite_count > 0 else 0.0
        stats[str(column)] = ColumnActivityStats(
            finite_count=finite_count,
            active_count=active_count,
            active_share=active_share,
            std=std,
        )
    return stats


def _is_active(
    stats: ColumnActivityStats,
    *,
    min_feature_rows: int,
    min_active_share: float,
    min_variance: float,
) -> bool:
    return bool(
        stats.finite_count >= int(min_feature_rows)
        and stats.active_share >= float(min_active_share)
        and stats.std > float(min_variance)
    )


@dataclass
class FrozenMaskBank:
    active_columns: tuple[str, ...]
    sample_count: int

    def manifest(self) -> dict[str, Any]:
        return {
            "sample_count": int(self.sample_count),
            "active_columns": list(self.active_columns),
            "active_column_count": int(len(self.active_columns)),
        }


@dataclass
class RegimeMaskingAdapter:
    config: Mapping[str, Any] | None = None
    regime_column: str = "regime"
    _feature_columns: list[str] = field(default_factory=list, init=False)
    _regime_columns: list[str] = field(default_factory=list, init=False)
    _feature_metadata_columns: int = field(default=0, init=False)
    _fit_row_count: int = field(default=0, init=False)
    _mask_candidate_columns: tuple[str, ...] = field(default_factory=tuple, init=False)
    _passthrough_columns: tuple[str, ...] = field(default_factory=tuple, init=False)
    _global_activity_stats: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    _global_active_columns: tuple[str, ...] = field(default_factory=tuple, init=False)
    _regime_masks: dict[str, FrozenMaskBank] = field(default_factory=dict, init=False)
    _regime_sample_counts: dict[str, int] = field(default_factory=dict, init=False)
    _skipped_regimes: dict[str, str] = field(default_factory=dict, init=False)
    _fit_regime_labels: list[str] = field(default_factory=list, init=False)
    _fit_ready_rows: int = field(default=0, init=False)
    _fit_confident_rows: int = field(default=0, init=False)
    _disabled_columns: tuple[str, ...] = field(default_factory=tuple, init=False)
    _disabled_reasons: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.config = _clone_mapping(self.config)
        self.regime_column = str(self.regime_column or "regime")
        selection = _clone_mapping(self.config.get("selection") or {})
        self._selection_mode = canonicalize_selection_mode(
            self.config.get("requested_selection_mode", selection.get("mode") or "identity")
        )
        self._selection_fallback_mode = canonicalize_selection_fallback_mode(
            self.config.get("selection_fallback_mode")
            or selection.get("fallback")
            or ("global" if self._selection_mode == "per_regime_mask" else "identity")
        )
        self._disable_incompatible_features = bool(self.config.get("disable_incompatible_features", False))
        self._min_regime_samples = _resolve_min_regime_samples(self.config)
        self._confidence_floor = _resolve_confidence_floor(self.config)
        self._min_feature_rows = _resolve_min_feature_rows(self.config)
        self._min_active_share = _resolve_min_active_share(self.config)
        self._min_variance = _resolve_min_variance(self.config)
        self._activity_epsilon = _resolve_activity_epsilon(self.config)
        if self._selection_mode not in {"identity", "per_regime_mask"}:
            raise ValueError(f"Unsupported feature_adaptation selection mode={self._selection_mode!r}")
        if self._selection_fallback_mode not in {"global", "identity"}:
            raise ValueError(
                f"Unsupported feature_adaptation selection fallback={self._selection_fallback_mode!r}"
            )

    def fit(
        self,
        X: Any,
        regime_frame: Any,
        feature_metadata: Mapping[str, Any] | None = None,
    ) -> "RegimeMaskingAdapter":
        X_frame = _coerce_feature_frame(X)
        regime = _coerce_regime_frame(regime_frame, X_frame.index)
        self._feature_columns = [str(column) for column in X_frame.columns]
        self._regime_columns = [str(column) for column in regime.columns]
        self._feature_metadata_columns = int(len(dict(feature_metadata or {})))
        self._fit_row_count = int(len(X_frame))
        mask_candidates, passthrough_columns = _resolve_mask_columns(X_frame, regime)
        self._mask_candidate_columns = tuple(mask_candidates)
        self._passthrough_columns = tuple(passthrough_columns)
        global_stats = _compute_column_activity(
            X_frame,
            mask_candidates,
            activity_epsilon=self._activity_epsilon,
        )
        self._global_activity_stats = {
            column: stats.to_dict()
            for column, stats in sorted(global_stats.items())
        }
        self._global_active_columns = tuple(
            column
            for column in mask_candidates
            if _is_active(
                global_stats[column],
                min_feature_rows=self._min_feature_rows,
                min_active_share=self._min_active_share,
                min_variance=self._min_variance,
            )
        )

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

        self._regime_masks = {}
        self._regime_sample_counts = {}
        self._skipped_regimes = {}
        for regime_key in unique_labels:
            label_mask = normalized_labels.eq(regime_key)
            ready_label_mask = label_mask & ready_mask
            confident_label_mask = label_mask & confident_mask
            sample_count = int(confident_label_mask.sum())
            self._regime_sample_counts[regime_key] = sample_count
            if not self._global_active_columns:
                self._skipped_regimes[regime_key] = "no_global_active_columns"
                continue
            if sample_count >= self._min_regime_samples:
                regime_stats = _compute_column_activity(
                    X_frame.loc[confident_label_mask],
                    list(self._global_active_columns),
                    activity_epsilon=self._activity_epsilon,
                )
                active_columns = [
                    column
                    for column in self._global_active_columns
                    if _is_active(
                        regime_stats[column],
                        min_feature_rows=min(self._min_feature_rows, sample_count),
                        min_active_share=self._min_active_share,
                        min_variance=self._min_variance,
                    )
                ]
                self._regime_masks[regime_key] = FrozenMaskBank(
                    active_columns=tuple(active_columns),
                    sample_count=sample_count,
                )
                continue
            if confidence is not None and self._confidence_floor is not None and int(ready_label_mask.sum()) > 0 and sample_count == 0:
                self._skipped_regimes[regime_key] = "confidence_below_floor"
                continue
            self._skipped_regimes[regime_key] = "insufficient_rows"

        disabled_reasons = {}
        if self._disable_incompatible_features:
            for column in self._mask_candidate_columns:
                if column in self._global_active_columns:
                    continue
                column_stats = ColumnActivityStats(**self._global_activity_stats.get(column, {}))
                if float(column_stats.std) <= self._min_variance:
                    disabled_reasons[column] = "globally_constant"
                else:
                    disabled_reasons[column] = "below_global_support_threshold"
            if self._regime_masks:
                supported_union = set().union(*(set(bank.active_columns) for bank in self._regime_masks.values()))
                for column in self._global_active_columns:
                    if column not in supported_union:
                        disabled_reasons.setdefault(column, "inactive_in_all_supported_regimes")
        self._disabled_reasons = dict(sorted(disabled_reasons.items()))
        self._disabled_columns = tuple(self._disabled_reasons.keys())
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
        if regime_key in self._regime_masks:
            return regime_key, None
        skipped_reason = self._skipped_regimes.get(regime_key)
        if skipped_reason == "insufficient_rows":
            return regime_key, "sparse_regime"
        if skipped_reason == "confidence_below_floor":
            return regime_key, "confidence_below_floor"
        return regime_key, "missing_regime_bank"

    def transform(
        self,
        X: Any,
        regime_frame: Any,
    ) -> tuple[pd.DataFrame, FeaturePolicyContract]:
        X_frame = _coerce_feature_frame(X)
        regime = _coerce_regime_frame(regime_frame, X_frame.index)
        transformed = X_frame.reindex(columns=self._feature_columns, fill_value=0.0).copy()
        labels = _resolve_regime_labels(regime, self.regime_column, transformed.index)
        warm = _resolve_bool_series(regime, "warm", transformed.index, True)
        unavailable = _resolve_bool_series(regime, "unavailable", transformed.index, False)
        confidence = _resolve_confidence_series(regime, transformed.index)

        disabled_columns = list(self._disabled_columns)
        disabled_set = set(disabled_columns)
        if disabled_columns:
            transformed.loc[:, disabled_columns] = 0.0

        mask_assignment_counts: Counter[str] = Counter()
        fallback_counts: Counter[str] = Counter()
        assigned_rows_by_regime: dict[str, list[Any]] = {}
        global_fallback_rows: list[Any] = []
        identity_fallback_rows: list[Any] = []
        fallback_rows = 0
        masked_cell_count = 0
        disabled_cell_count = int(len(disabled_columns) * len(transformed))
        active_mask_candidates = [
            column for column in self._mask_candidate_columns
            if column not in disabled_set
        ]
        if self._selection_mode == "per_regime_mask" and active_mask_candidates:
            for row_index in transformed.index:
                regime_key, reason = self._row_reason(
                    labels.loc[row_index],
                    warm=bool(warm.loc[row_index]),
                    unavailable=bool(unavailable.loc[row_index]),
                    confidence=(None if confidence is None else confidence.loc[row_index]),
                )
                if reason is None and regime_key is not None:
                    assigned_rows_by_regime.setdefault(regime_key, []).append(row_index)
                    mask_assignment_counts[regime_key] += 1
                    continue
                fallback_rows += 1
                fallback_counts[reason] += 1
                if self._selection_fallback_mode == "global":
                    global_fallback_rows.append(row_index)
                    mask_assignment_counts["global_fallback"] += 1
                else:
                    identity_fallback_rows.append(row_index)
                    mask_assignment_counts["identity_fallback"] += 1

            for regime_key, row_index in assigned_rows_by_regime.items():
                active_columns = set(self._regime_masks[regime_key].active_columns) - disabled_set
                inactive_columns = [
                    column for column in active_mask_candidates
                    if column not in active_columns
                ]
                if inactive_columns:
                    transformed.loc[row_index, inactive_columns] = 0.0
                    masked_cell_count += int(len(inactive_columns) * len(row_index))

            if global_fallback_rows:
                global_active_columns = set(self._global_active_columns) - disabled_set
                inactive_columns = [
                    column for column in active_mask_candidates
                    if column not in global_active_columns
                ]
                if inactive_columns:
                    transformed.loc[global_fallback_rows, inactive_columns] = 0.0
                    masked_cell_count += int(len(inactive_columns) * len(global_fallback_rows))

        no_op = bool(
            self._selection_mode == "identity"
            and not disabled_columns
        )
        if self._selection_mode == "per_regime_mask":
            no_op = bool(not disabled_columns and masked_cell_count == 0)

        manifest = self.manifest()
        sparse_regimes = sorted(
            regime_key for regime_key, reason in self._skipped_regimes.items() if reason == "insufficient_rows"
        )
        policy = FeaturePolicyContract(
            policy_id=str(manifest.get("policy_id", "regime_masking")),
            feature_columns=[str(column) for column in transformed.columns],
            disabled_columns=disabled_columns,
            generated_columns=[],
            regime_column=self.regime_column,
            scaling_mode="identity",
            fallback_mode=self._selection_fallback_mode,
            sparse_regimes=sparse_regimes,
            metadata={
                "adapter_type": manifest.get("adapter_type"),
                "no_op": no_op,
                "requested_enabled": bool(self.config.get("enabled", False)),
                "requested_scaling_mode": str(self.config.get("requested_scaling_mode", "identity")),
                "requested_selection_mode": str(self.config.get("requested_selection_mode", self._selection_mode)),
                "requested_sections": list(self.config.get("requested_sections") or []),
                "deferred_runtime": bool(self.config.get("deferred_runtime", False)),
                "fit_row_count": int(self._fit_row_count),
                "transform_row_count": int(len(transformed)),
                "regime_row_count": int(len(regime)),
                "selection_mode": self._selection_mode,
                "selection_fallback_mode": self._selection_fallback_mode,
                "mask_candidate_columns": list(self._mask_candidate_columns),
                "mask_candidate_column_count": int(len(self._mask_candidate_columns)),
                "passthrough_columns": list(self._passthrough_columns),
                "passthrough_column_count": int(len(self._passthrough_columns)),
                "global_active_columns": list(self._global_active_columns),
                "global_active_column_count": int(len(self._global_active_columns)),
                "regime_mask_count": int(len(self._regime_masks)),
                "mask_assignment_counts": dict(sorted(mask_assignment_counts.items())),
                "fallback_rows_total": int(fallback_rows),
                "fallback_rows_by_reason": dict(sorted(fallback_counts.items())),
                "masked_cell_count": int(masked_cell_count),
                "disabled_columns_by_reason": dict(self._disabled_reasons),
                "disabled_cell_count": int(disabled_cell_count),
                "skipped_regimes": dict(self._skipped_regimes),
                "min_regime_samples": int(self._min_regime_samples),
                "confidence_floor": self._confidence_floor,
                "min_feature_rows": int(self._min_feature_rows),
                "min_active_share": float(self._min_active_share),
                "min_variance": float(self._min_variance),
                "activity_epsilon": float(self._activity_epsilon),
            },
        )
        return transformed, policy

    def manifest(self) -> dict[str, Any]:
        regime_masks = {
            str(regime_key): bank.manifest()
            for regime_key, bank in sorted(self._regime_masks.items())
        }
        return {
            "adapter_type": "regime_masking",
            "policy_id": "regime_masking",
            "regime_column": self.regime_column,
            "fit_row_count": int(self._fit_row_count),
            "feature_columns": list(self._feature_columns),
            "feature_count": int(len(self._feature_columns)),
            "regime_columns": list(self._regime_columns),
            "regime_column_count": int(len(self._regime_columns)),
            "feature_metadata_columns": int(self._feature_metadata_columns),
            "requested_enabled": bool(self.config.get("enabled", False)),
            "requested_scaling_mode": str(self.config.get("requested_scaling_mode", "identity")),
            "requested_selection_mode": str(self.config.get("requested_selection_mode", self._selection_mode)),
            "selection_mode": self._selection_mode,
            "selection_fallback_mode": self._selection_fallback_mode,
            "requested_sections": list(self.config.get("requested_sections") or []),
            "deferred_runtime": bool(self.config.get("deferred_runtime", False)),
            "no_op": bool(self._selection_mode == "identity" and not self._disabled_columns),
            "mask_candidate_columns": list(self._mask_candidate_columns),
            "mask_candidate_column_count": int(len(self._mask_candidate_columns)),
            "passthrough_columns": list(self._passthrough_columns),
            "passthrough_column_count": int(len(self._passthrough_columns)),
            "global_activity_stats": dict(self._global_activity_stats),
            "global_active_columns": list(self._global_active_columns),
            "global_active_column_count": int(len(self._global_active_columns)),
            "regime_masks": regime_masks,
            "regime_mask_count": int(len(regime_masks)),
            "regime_sample_counts": dict(sorted(self._regime_sample_counts.items())),
            "fit_regime_labels": list(self._fit_regime_labels),
            "fit_ready_rows": int(self._fit_ready_rows),
            "fit_confident_rows": int(self._fit_confident_rows),
            "skipped_regimes": dict(self._skipped_regimes),
            "skipped_regime_count": int(len(self._skipped_regimes)),
            "disabled_columns": list(self._disabled_columns),
            "disabled_column_count": int(len(self._disabled_columns)),
            "disabled_columns_by_reason": dict(self._disabled_reasons),
            "min_regime_samples": int(self._min_regime_samples),
            "confidence_floor": self._confidence_floor,
            "min_feature_rows": int(self._min_feature_rows),
            "min_active_share": float(self._min_active_share),
            "min_variance": float(self._min_variance),
            "activity_epsilon": float(self._activity_epsilon),
        }


__all__ = [
    "RegimeMaskingAdapter",
    "canonicalize_selection_fallback_mode",
    "canonicalize_selection_mode",
]