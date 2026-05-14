"""Regime-aware feature engineering and walk-forward training helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score

from .feature_adaptation import build_feature_strategy_adapter, validate_feature_adaptation_runtime_support
from .models import (
    _evaluate_directional_probability_quality,
    predict_probability_frame,
    train_model,
    walk_forward_split,
)
from .regimes.online_state import (
    build_admissible_regime_view,
    build_regime_frame_from_state_contracts,
    normalize_regime_state_contracts,
    slice_regime_state_contracts,
)
from .specialists.library import project_specialist_library_snapshot
from .specialists.contracts import (
    SpecialistHealthContract,
    SpecialistLibrarySnapshot,
    SpecialistLifecycleState,
    SpecialistPerformanceSlice,
    SpecialistSpec,
)


def _coerce_regime_frame(regime_data, index=None):
    if regime_data is None:
        frame = pd.DataFrame()
    elif isinstance(regime_data, pd.Series):
        column_name = regime_data.name or "regime"
        frame = regime_data.to_frame(name=column_name)
    else:
        try:
            frame = build_regime_frame_from_state_contracts(regime_data)
        except Exception:
            frame = pd.DataFrame(regime_data).copy()
    if index is not None:
        frame = frame.reindex(index)
    return frame


def _coerce_regime_state_contracts(regime_data, index=None):
    try:
        contracts = normalize_regime_state_contracts(regime_data)
    except Exception:
        return None
    if not contracts:
        return []
    return slice_regime_state_contracts(contracts, index) if index is not None else contracts


def _coerce_inference_regime_frame(regime_data, index=None, column_name="regime"):
    state_contracts = _coerce_regime_state_contracts(regime_data, index=index)
    if state_contracts is not None:
        return build_admissible_regime_view(state_contracts, index=index, column_name=column_name)
    return _coerce_regime_frame(regime_data, index=index)


def _diagnostic_flag_count(regime_frame, column_name, index):
    if regime_frame is None or regime_frame.empty or column_name not in regime_frame.columns:
        return 0
    series = pd.to_numeric(pd.Series(regime_frame[column_name], index=index, copy=False), errors="coerce")
    return int(series.fillna(0).gt(0).sum())


def _resolve_regime_identity_column(regime_frame, preferred_column="regime"):
    if regime_frame is None or regime_frame.empty:
        return str(preferred_column or "regime")

    ordered_candidates = []
    for candidate in ("canonical_regime_id", preferred_column, "regime"):
        candidate_name = str(candidate or "").strip()
        if candidate_name and candidate_name not in ordered_candidates:
            ordered_candidates.append(candidate_name)
    ordered_candidates.extend(
        str(column)
        for column in regime_frame.columns
        if str(column) not in ordered_candidates
    )

    for candidate_name in ordered_candidates:
        if candidate_name not in regime_frame.columns:
            continue
        series = pd.Series(regime_frame[candidate_name], copy=False)
        if series.notna().any():
            return candidate_name
    return str(regime_frame.columns[0])


def _build_regime_alias_map(regime_frame, identity_column, semantic_column="regime"):
    if (
        regime_frame is None
        or regime_frame.empty
        or identity_column == semantic_column
        or identity_column not in regime_frame.columns
        or semantic_column not in regime_frame.columns
    ):
        return {}

    alias_frame = regime_frame.loc[:, [identity_column, semantic_column]].dropna()
    if alias_frame.empty:
        return {}

    alias_map = {}
    for regime_id, rows in alias_frame.groupby(identity_column):
        aliases = sorted({str(value) for value in rows[semantic_column] if value is not None})
        if aliases:
            alias_map[str(regime_id)] = aliases
    return alias_map


def _subset_sampling_metadata(sampling_metadata, index):
    metadata = dict(sampling_metadata or {})
    if not metadata:
        return None

    subset = dict(metadata)
    labels = metadata.get("labels")
    if labels is not None:
        subset["labels"] = pd.DataFrame(labels).reindex(index)
    close = metadata.get("close")
    if close is not None:
        subset["close"] = pd.Series(close, copy=False)
    return subset


def _train_constant_safe_model(
    X,
    y,
    *,
    sample_weight=None,
    model_type="gbm",
    model_params=None,
    sampling_metadata=None,
):
    X_frame = pd.DataFrame(X).copy()
    y_series = pd.Series(y, index=X_frame.index)
    if y_series.nunique() < 2:
        model = DummyClassifier(strategy="prior")
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = pd.Series(sample_weight, index=y_series.index, dtype=float).to_numpy()
        model.fit(X_frame, y_series, **fit_kwargs)
        return model, {
            "sequential_bootstrap_enabled": False,
            "sequential_bootstrap_used": False,
            "reason": "single_class_training_fallback",
            "warning": None,
            "mean_uniqueness": None,
            "uniqueness_threshold": None,
            "high_concurrency": False,
            "random_state": dict(model_params or {}).get("random_state", 42),
            "bootstrap_sample_size": None,
            "constant_class": y_series.iloc[0] if len(y_series) else None,
        }

    return train_model(
        X_frame,
        y_series,
        sample_weight=sample_weight,
        model_type=model_type,
        model_params=model_params,
        sampling_metadata=sampling_metadata,
        return_report=True,
    )


def summarize_regime_coverage(regime_data, regime_column="regime", config=None):
    regime_frame = _coerce_regime_frame(regime_data)
    config = dict(config or {})
    max_dominant_share = float(config.get("max_dominant_share", 0.8))
    min_distinct_regimes = int(config.get("min_distinct_regimes", 2))

    if regime_frame.empty:
        return {
            "status": "unknown",
            "promotion_pass": False,
            "available_rows": 0,
            "distinct_regimes": 0,
            "dominant_regime": None,
            "dominant_share": None,
            "regime_distribution": {},
            "coverage_ok": False,
            "reasons": ["regime_data_unavailable"],
        }

    target_column = _resolve_regime_identity_column(regime_frame, preferred_column=regime_column)
    labels = pd.Series(regime_frame[target_column], copy=False).dropna()
    if labels.empty:
        return {
            "status": "unknown",
            "promotion_pass": False,
            "available_rows": 0,
            "distinct_regimes": 0,
            "dominant_regime": None,
            "dominant_share": None,
            "regime_distribution": {},
            "regime_identity_column": target_column,
            "coverage_ok": False,
            "reasons": ["regime_labels_missing"],
        }

    distribution = labels.value_counts(normalize=True, dropna=True)
    dominant_regime = distribution.index[0]
    dominant_share = float(distribution.iloc[0])
    distinct_regimes = int(labels.nunique())
    reasons = []
    if distinct_regimes < min_distinct_regimes:
        reasons.append("insufficient_regime_diversity")
    if dominant_share > max_dominant_share:
        reasons.append("dominant_regime_exceeds_threshold")
    status = "passed" if not reasons else "failed"
    semantic_distribution = {}
    if target_column != regime_column and regime_column in regime_frame.columns:
        semantic_labels = pd.Series(regime_frame[regime_column], copy=False).dropna()
        if not semantic_labels.empty:
            semantic_distribution = {
                str(key): float(value)
                for key, value in semantic_labels.value_counts(normalize=True, dropna=True).items()
            }
    return {
        "status": status,
        "promotion_pass": status == "passed",
        "available_rows": int(len(labels)),
        "distinct_regimes": distinct_regimes,
        "dominant_regime": dominant_regime,
        "dominant_share": dominant_share,
        "regime_distribution": {str(key): float(value) for key, value in distribution.items()},
        "regime_identity_column": target_column,
        "semantic_regime_distribution": semantic_distribution,
        "coverage_ok": not reasons,
        "reasons": reasons,
    }


@dataclass
class RegimeAwareFeatureFrame:
    frame: pd.DataFrame
    regime_columns: list[str] = field(default_factory=list)
    normalized_columns: list[str] = field(default_factory=list)
    interaction_columns: list[str] = field(default_factory=list)
    adapter: Any | None = None
    policy: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)


def build_regime_aware_feature_frame(X, regime_data, config=None):
    resolved_config = dict(config or {})
    regime_column = str(resolved_config.get("regime_column", "regime"))
    base = pd.DataFrame(X).copy()
    regime_frame = _coerce_regime_frame(regime_data, index=base.index)
    adapter = build_feature_strategy_adapter(resolved_config, regime_column=regime_column)
    adapter.fit(base, regime_frame)
    transformed, policy = adapter.transform(base, regime_frame)
    policy_dict = policy.to_dict()
    metadata = dict(policy.metadata or {})
    return RegimeAwareFeatureFrame(
        frame=transformed,
        regime_columns=[str(column) for column in list(metadata.get("regime_columns") or [])],
        normalized_columns=[str(column) for column in list(metadata.get("normalized_columns") or [])],
        interaction_columns=[str(column) for column in list(metadata.get("interaction_columns") or [])],
        adapter=adapter,
        policy=policy_dict,
        manifest=dict(adapter.manifest() or {}),
    )


def evaluate_regime_aware_predictions(y_true, predictions, probability_frame=None):
    y_series = pd.Series(y_true, index=predictions.index if isinstance(predictions, pd.Series) else None)
    pred_series = pd.Series(predictions, index=y_series.index)
    metrics = {
        "accuracy": round(float(accuracy_score(y_series, pred_series)), 4),
        "f1_macro": round(float(f1_score(y_series, pred_series, average="macro", zero_division=0)), 4),
        "label_abstain_rate": round(float(y_series.eq(0).mean()), 4),
        "prediction_coverage": round(float(pred_series.ne(0).mean()), 4),
    }
    directional_mask = y_series.ne(0)
    if directional_mask.any():
        metrics["directional_accuracy"] = round(
            float(accuracy_score(y_series.loc[directional_mask], pred_series.loc[directional_mask])),
            4,
        )
        metrics["directional_f1_macro"] = round(
            float(f1_score(y_series.loc[directional_mask], pred_series.loc[directional_mask], average="macro", zero_division=0)),
            4,
        )
    if probability_frame is not None:
        metrics.update(_evaluate_directional_probability_quality(probability_frame, y_series))
    return metrics


class RegimeAwareModelBundle:
    def __init__(
        self,
        *,
        strategy,
        model=None,
        fallback_model=None,
        specialist_models=None,
        feature_config=None,
        feature_columns=None,
        feature_adapter=None,
        feature_policy=None,
        feature_manifest=None,
        regime_column="regime",
        ordered_classes=(-1, 0, 1),
    ):
        self.strategy = str(strategy)
        self.model = model
        self.fallback_model = fallback_model
        self.specialist_models = dict(specialist_models or {})
        self.feature_config = dict(feature_config or {})
        self.feature_columns = list(feature_columns or [])
        self.feature_adapter = feature_adapter
        self.feature_policy = dict(feature_policy or {})
        self.feature_manifest = dict(feature_manifest or {})
        self.regime_column = regime_column
        self.ordered_classes = tuple(ordered_classes)

    def _transform_feature_strategy(self, X, regime_data):
        if self.feature_adapter is not None:
            X_frame = pd.DataFrame(X).copy()
            regime_frame = _coerce_regime_frame(regime_data, index=X_frame.index)
            transformed, _ = self.feature_adapter.transform(X_frame, regime_frame)
            return transformed.reindex(columns=self.feature_columns, fill_value=0.0)
        feature_result = build_regime_aware_feature_frame(X, regime_data, config=self.feature_config)
        return feature_result.frame.reindex(columns=self.feature_columns, fill_value=0.0)

    def predict(self, X, regime_data):
        predictions, _ = self.predict_with_report(X, regime_data)
        return predictions.to_numpy()

    def predict_proba(self, X, regime_data):
        _, probability_frame, _ = self.predict_with_probability_report(X, regime_data)
        return probability_frame

    def predict_with_report(self, X, regime_data):
        predictions, _, report = self.predict_with_probability_report(X, regime_data)
        return predictions, report

    def predict_with_probability_report(self, X, regime_data):
        X_frame = pd.DataFrame(X).copy()
        regime_frame = _coerce_inference_regime_frame(regime_data, index=X_frame.index, column_name=self.regime_column)
        if self.strategy == "feature":
            model = self.model
            if model is None:
                raise RuntimeError("feature strategy bundle is missing primary model")
            transformed = self._transform_feature_strategy(X_frame, regime_frame)
            predictions = pd.Series(model.predict(transformed), index=X_frame.index)
            probabilities = predict_probability_frame(model, transformed, ordered_classes=self.ordered_classes)
            report = {
                "strategy": self.strategy,
                "fallback_rows": 0,
                "fallback_evidence_rows": int(len(X_frame)),
                "fallback_row_share": 0.0 if len(X_frame) > 0 else None,
                "unseen_regimes": [],
                "candidate_classification": "generalist_only",
            }
            return predictions, probabilities, report

        if self.strategy != "specialist":
            raise ValueError(f"Unknown regime-aware strategy={self.strategy!r}")

        identity_column = _resolve_regime_identity_column(regime_frame, preferred_column=self.regime_column)
        labels = pd.Series(
            regime_frame[identity_column] if identity_column in regime_frame.columns else regime_frame.iloc[:, 0],
            index=X_frame.index,
        ) if not regime_frame.empty else pd.Series(np.nan, index=X_frame.index)
        alternate_labels = None
        for candidate_column in (self.regime_column, "regime"):
            if candidate_column != identity_column and candidate_column in regime_frame.columns:
                alternate_labels = pd.Series(regime_frame[candidate_column], index=X_frame.index)
                break
        alternate_lookup = {}
        if alternate_labels is not None:
            alias_frame = pd.DataFrame({"primary": labels, "alternate": alternate_labels}).dropna()
            for primary_value, rows in alias_frame.groupby("primary"):
                aliases = [value for value in pd.unique(rows["alternate"]) if value is not None]
                if aliases:
                    alternate_lookup[primary_value] = aliases
        fallback_model = self.fallback_model
        if fallback_model is None:
            raise RuntimeError("specialist strategy bundle is missing fallback model")
        predictions = pd.Series(0, index=X_frame.index, dtype=int)
        probabilities = pd.DataFrame(0.0, index=X_frame.index, columns=list(self.ordered_classes), dtype=float)
        fallback_rows = 0
        unseen_regimes = []
        timing_blocked_rows = _diagnostic_flag_count(regime_frame, "timing_blocked", X_frame.index)
        unavailable_rows = _diagnostic_flag_count(regime_frame, "unavailable", X_frame.index)

        for regime_value, row_index in labels.groupby(labels).groups.items():
            model = self.specialist_models.get(regime_value)
            if model is None:
                for alternate_value in list(alternate_lookup.get(regime_value) or []):
                    model = self.specialist_models.get(alternate_value)
                    if model is not None:
                        break
            if model is None:
                model = fallback_model
                fallback_rows += int(len(row_index))
                unseen_regimes.append(regime_value)
            group_X = X_frame.loc[row_index, self.feature_columns] if self.feature_columns else X_frame.loc[row_index]
            predictions.loc[row_index] = model.predict(group_X)
            probabilities.loc[row_index, :] = predict_probability_frame(
                model,
                group_X,
                ordered_classes=self.ordered_classes,
            ).to_numpy()

        missing_rows = labels.index[labels.isna()]
        if len(missing_rows) > 0:
            fallback_rows += int(len(missing_rows))
            fallback_X = X_frame.loc[missing_rows, self.feature_columns] if self.feature_columns else X_frame.loc[missing_rows]
            fallback_predictions = fallback_model.predict(fallback_X)
            predictions.loc[missing_rows] = fallback_predictions
            probabilities.loc[missing_rows, :] = predict_probability_frame(
                fallback_model,
                fallback_X,
                ordered_classes=self.ordered_classes,
            ).to_numpy()
            unseen_regimes.append("missing")

        report = {
            "strategy": self.strategy,
            "fallback_rows": int(fallback_rows),
            "fallback_evidence_rows": int(len(X_frame)),
            "fallback_row_share": (None if len(X_frame) <= 0 else round(float(fallback_rows / len(X_frame)), 4)),
            "regime_identity_column": identity_column,
            "timing_blocked_rows": int(timing_blocked_rows),
            "unavailable_rows": int(unavailable_rows),
            "unseen_regimes": sorted({str(value) for value in unseen_regimes}),
            "candidate_classification": (
                "generalist_only"
                if not self.specialist_models
                else ("specialist_degraded_to_fallback" if fallback_rows > 0 else "specialist_effective")
            ),
        }
        return predictions, probabilities, report


def _resolve_estimator_family(model):
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model") or next(reversed(model.named_steps.values()))
    return str(estimator.__class__.__name__).lower()


def _normalize_training_window(training_report):
    coverage_summary = dict((training_report or {}).get("coverage_summary") or {})
    return {
        "strategy": str((training_report or {}).get("strategy") or "unknown"),
        "coverage_status": coverage_summary.get("status"),
        "promotion_pass": coverage_summary.get("promotion_pass"),
    }


def _resolve_initial_specialist_lifecycle_state(metadata=None):
    configured = dict(metadata or {}).get("lifecycle_state")
    if configured is None:
        return SpecialistLifecycleState.CANDIDATE
    normalized = str(configured).strip().lower()
    for state in SpecialistLifecycleState:
        if state.value == normalized:
            return state
    raise ValueError(f"Unknown specialist lifecycle_state {configured!r}")


def _resolve_regime_binding_metadata(training_report=None, regime_name=None):
    resolved_report = dict(training_report or {})
    regime_key = str(regime_name)
    aliases_by_id = {
        str(key): [str(item) for item in list(value or [])]
        for key, value in dict(resolved_report.get("regime_aliases_by_id") or {}).items()
    }
    aliases = list(aliases_by_id.get(regime_key) or [])
    identity_kind = str(resolved_report.get("regime_identity_column") or "regime")
    metadata = {
        "routing_regime_id": regime_key,
        "routing_identity_kind": identity_kind,
        "regime_label": (aliases[0] if aliases else regime_key),
    }
    if aliases:
        metadata["semantic_labels"] = aliases
    return metadata


def build_specialist_specs_from_bundle(bundle, training_report=None, *, symbol="unknown", timeframe="unknown", metadata=None):
    training_report = dict(training_report or {})
    shared_metadata = dict(metadata or {})
    training_window = _normalize_training_window(training_report)
    lifecycle_state = _resolve_initial_specialist_lifecycle_state(shared_metadata)
    specs = []

    if bundle.strategy == "feature":
        compatible_regimes = list(dict((training_report.get("coverage_summary") or {}).get("regime_distribution") or {}).keys())
        specs.append(
            SpecialistSpec(
                model_id=str(shared_metadata.get("model_id", "global_regime_feature_model")),
                symbol=str(symbol),
                timeframe=str(timeframe),
                compatible_regimes=[str(item) for item in compatible_regimes],
                estimator_family=_resolve_estimator_family(bundle.model),
                feature_policy_id="regime_feature_strategy",
                training_window=training_window,
                metadata={
                    **shared_metadata,
                    "bundle_strategy": bundle.strategy,
                    "feature_column_count": int(len(bundle.feature_columns)),
                    "lifecycle_state": lifecycle_state.value,
                },
            )
        )
        return specs

    if bundle.fallback_model is not None:
        specs.append(
            SpecialistSpec(
                model_id=str(shared_metadata.get("fallback_model_id", "fallback_generalist")),
                symbol=str(symbol),
                timeframe=str(timeframe),
                compatible_regimes=[],
                estimator_family=_resolve_estimator_family(bundle.fallback_model),
                feature_policy_id="global_generalist",
                training_window=training_window,
                metadata={
                    **shared_metadata,
                    "bundle_strategy": bundle.strategy,
                    "fallback_only": True,
                    "lifecycle_state": lifecycle_state.value,
                },
            )
        )

    for regime_value, model in dict(bundle.specialist_models or {}).items():
        regime_name = str(regime_value)
        specs.append(
            SpecialistSpec(
                model_id=f"specialist::{regime_name}",
                symbol=str(symbol),
                timeframe=str(timeframe),
                compatible_regimes=[regime_name],
                estimator_family=_resolve_estimator_family(model),
                feature_policy_id="regime_specialist",
                training_window=training_window,
                metadata={
                    **shared_metadata,
                    **_resolve_regime_binding_metadata(training_report, regime_name),
                    "bundle_strategy": bundle.strategy,
                    "lifecycle_state": lifecycle_state.value,
                },
            )
        )
    return specs


def build_specialist_health_contracts(bundle, training_report=None, *, metadata=None):
    training_report = dict(training_report or {})
    shared_metadata = dict(metadata or {})
    skipped_regimes = {str(key): str(value) for key, value in dict(training_report.get("skipped_regimes") or {}).items()}
    health = []

    if bundle.strategy == "feature":
        health.append(
            SpecialistHealthContract(
                model_id=str(shared_metadata.get("model_id", "global_regime_feature_model")),
                compatible_regimes=list(dict((training_report.get("coverage_summary") or {}).get("regime_distribution") or {}).keys()),
                stability_score=None,
                decay_score=None,
                failure_flags=[],
                fallback_only=False,
                metadata={
                    **shared_metadata,
                    "bundle_strategy": bundle.strategy,
                    "health_binding_resolved": False,
                    "health_state": "unknown",
                    "health_evidence_source": "placeholder_training_summary",
                },
            )
        )
        return health

    if bundle.fallback_model is not None:
        health.append(
            SpecialistHealthContract(
                model_id=str(shared_metadata.get("fallback_model_id", "fallback_generalist")),
                compatible_regimes=[],
                stability_score=None,
                decay_score=None,
                failure_flags=[],
                fallback_only=True,
                metadata={
                    **shared_metadata,
                    "bundle_strategy": bundle.strategy,
                    "health_binding_resolved": True,
                    "health_state": "fallback_only",
                    "health_evidence_source": "fallback_safe_mode",
                },
            )
        )

    for regime_value in dict(bundle.specialist_models or {}).keys():
        regime_name = str(regime_value)
        health.append(
            SpecialistHealthContract(
                model_id=f"specialist::{regime_name}",
                compatible_regimes=[regime_name],
                stability_score=None,
                decay_score=None,
                failure_flags=[],
                fallback_only=False,
                metadata={
                    **shared_metadata,
                    **_resolve_regime_binding_metadata(training_report, regime_name),
                    "bundle_strategy": bundle.strategy,
                    "health_binding_resolved": False,
                    "health_state": "unknown",
                    "health_evidence_source": "placeholder_training_summary",
                },
            )
        )

    for regime_name, reason in skipped_regimes.items():
        health.append(
            SpecialistHealthContract(
                model_id=f"skipped::{regime_name}",
                compatible_regimes=[regime_name],
                stability_score=None,
                decay_score=None,
                failure_flags=[str(reason)],
                fallback_only=False,
                metadata={
                    **shared_metadata,
                    **_resolve_regime_binding_metadata(training_report, regime_name),
                    "bundle_strategy": bundle.strategy,
                    "skipped": True,
                    "health_binding_resolved": False,
                    "health_state": "unknown",
                    "health_evidence_source": "placeholder_training_summary",
                },
            )
        )
    return health


def build_specialist_library_snapshot(bundle, training_report=None, *, symbol="unknown", timeframe="unknown", metadata=None):
    training_report = dict(training_report or {})
    shared_metadata = dict(metadata or {})
    specs = build_specialist_specs_from_bundle(
        bundle,
        training_report,
        symbol=symbol,
        timeframe=timeframe,
        metadata=shared_metadata,
    )
    health = build_specialist_health_contracts(bundle, training_report, metadata=shared_metadata)

    performance_slices = []
    trained_rows = {str(key): int(value) for key, value in dict(training_report.get("trained_rows_by_regime") or {}).items()}
    for regime_name, row_count in trained_rows.items():
        performance_slices.append(
            SpecialistPerformanceSlice(
                model_id=f"specialist::{regime_name}",
                regime_label=str(_resolve_regime_binding_metadata(training_report, regime_name).get("regime_label")),
                split_role="training_slice",
                row_count=int(row_count),
                metric_summary={"trained_rows": int(row_count)},
                metadata={
                    **shared_metadata,
                    **_resolve_regime_binding_metadata(training_report, regime_name),
                    "bundle_strategy": bundle.strategy,
                },
            )
        )

    snapshot = SpecialistLibrarySnapshot(
        symbol=str(symbol),
        timeframe=str(timeframe),
        fallback_model_id=("fallback_generalist" if bundle.fallback_model is not None and bundle.strategy == "specialist" else None),
        specialists=specs,
        health=health,
        performance_slices=performance_slices,
        metadata={
            **shared_metadata,
            "bundle_strategy": bundle.strategy,
            "feature_column_count": int(len(bundle.feature_columns)),
            "trained_regimes": [str(item) for item in list(training_report.get("trained_regimes") or [])],
            "skipped_regimes": {str(key): str(value) for key, value in dict(training_report.get("skipped_regimes") or {}).items()},
        },
    )
    return project_specialist_library_snapshot(snapshot)


def train_regime_aware_model(
    X,
    y,
    regime_data,
    *,
    strategy="feature",
    model_type="gbm",
    model_params=None,
    feature_config=None,
    coverage_config=None,
    regime_column="regime",
    min_samples_per_regime=40,
    sample_weight=None,
    sampling_metadata=None,
):
    X_frame = pd.DataFrame(X).copy()
    y_series = pd.Series(y, index=X_frame.index)
    regime_frame = _coerce_regime_frame(regime_data, index=X_frame.index)
    regime_state_contracts = _coerce_regime_state_contracts(regime_data, index=X_frame.index)
    strategy = str(strategy).lower()
    validate_feature_adaptation_runtime_support(
        dict(feature_config or {}).get("feature_adaptation") or {},
        regime_aware_strategy=strategy,
    )
    coverage_summary = summarize_regime_coverage(regime_frame, regime_column=regime_column, config=coverage_config)

    if strategy == "feature":
        feature_result = build_regime_aware_feature_frame(X_frame, regime_frame, config=feature_config)
        model, sampling_report = _train_constant_safe_model(
            feature_result.frame,
            y_series,
            sample_weight=sample_weight,
            model_type=model_type,
            model_params=model_params,
            sampling_metadata=sampling_metadata,
        )
        bundle = RegimeAwareModelBundle(
            strategy="feature",
            model=model,
            feature_config=feature_config,
            feature_columns=list(feature_result.frame.columns),
            feature_adapter=feature_result.adapter,
            feature_policy=feature_result.policy,
            feature_manifest=feature_result.manifest,
            regime_column=regime_column,
        )
        report = {
            "strategy": "feature",
            "feature_columns": list(feature_result.frame.columns),
            "regime_columns": list(feature_result.regime_columns),
            "normalized_columns": list(feature_result.normalized_columns),
            "interaction_columns": list(feature_result.interaction_columns),
            "regime_alignment": "inference_aligned_input",
            "feature_adaptation": {
                "policy": dict(feature_result.policy or {}),
                "manifest": dict(feature_result.manifest or {}),
            },
            "coverage_summary": coverage_summary,
            "sampling_report": sampling_report,
            "candidate_classification": "generalist_only",
        }
        return bundle, report

    if strategy != "specialist":
        raise ValueError("strategy must be 'feature' or 'specialist'")

    if regime_frame.empty:
        raise ValueError("specialist strategy requires regime_data")

    target_column = _resolve_regime_identity_column(regime_frame, preferred_column=regime_column)
    labels = pd.Series(regime_frame[target_column], index=X_frame.index)
    regime_aliases_by_id = _build_regime_alias_map(regime_frame, target_column, semantic_column=regime_column)
    fallback_model, fallback_sampling_report = _train_constant_safe_model(
        X_frame,
        y_series,
        sample_weight=sample_weight,
        model_type=model_type,
        model_params=model_params,
        sampling_metadata=sampling_metadata,
    )
    specialist_models = {}
    specialist_rows = {}
    skipped_regimes = {}
    specialist_sampling_reports = {}
    for regime_value, row_index in labels.groupby(labels).groups.items():
        regime_X = X_frame.loc[row_index]
        regime_y = y_series.loc[row_index]
        if len(regime_X) < int(min_samples_per_regime):
            skipped_regimes[str(regime_value)] = "minimum_samples_not_met"
            continue
        if regime_y.nunique() < 2:
            skipped_regimes[str(regime_value)] = "single_class_regime"
            continue
        regime_weight = None
        if sample_weight is not None:
            regime_weight = pd.Series(sample_weight, index=X_frame.index).loc[row_index]
        regime_sampling_metadata = _subset_sampling_metadata(sampling_metadata, row_index)
        specialist_model, specialist_sampling_report = train_model(
            regime_X,
            regime_y,
            sample_weight=regime_weight,
            model_type=model_type,
            model_params=model_params,
            sampling_metadata=regime_sampling_metadata,
            return_report=True,
        )
        specialist_models[regime_value] = specialist_model
        specialist_rows[str(regime_value)] = int(len(regime_X))
        specialist_sampling_reports[str(regime_value)] = specialist_sampling_report

    bundle = RegimeAwareModelBundle(
        strategy="specialist",
        fallback_model=fallback_model,
        specialist_models=specialist_models,
        feature_columns=list(X_frame.columns),
        regime_column=target_column,
    )
    report = {
        "strategy": "specialist",
        "feature_columns": list(X_frame.columns),
        "trained_regimes": [str(value) for value in specialist_models],
        "trained_rows_by_regime": specialist_rows,
        "skipped_regimes": skipped_regimes,
        "fallback_enabled": True,
        "regime_alignment": "inference_aligned_input",
        "regime_identity_column": target_column,
        "regime_aliases_by_id": regime_aliases_by_id,
        "coverage_summary": coverage_summary,
        "fallback_sampling_report": fallback_sampling_report,
        "specialist_sampling_reports": specialist_sampling_reports,
        "candidate_classification": (
            "generalist_only" if not specialist_models else "specialist_effective"
        ),
    }
    return bundle, report


def train_regime_aware_walk_forward(
    X,
    y,
    regime_data,
    *,
    strategy="feature",
    model_type="gbm",
    model_params=None,
    feature_config=None,
    regime_column="regime",
    min_samples_per_regime=40,
    coverage_config=None,
    sample_weight=None,
    n_splits=3,
    train_size=None,
    test_size=None,
    gap=0,
    expanding=False,
):
    X_frame = pd.DataFrame(X).copy()
    y_series = pd.Series(y, index=X_frame.index)
    regime_frame = _coerce_regime_frame(regime_data, index=X_frame.index)
    regime_state_contracts = _coerce_regime_state_contracts(regime_data, index=X_frame.index)

    folds = []
    oos_predictions = []
    oos_probabilities = []
    last_model = None

    for fold_number, (train_idx, test_idx) in enumerate(
        walk_forward_split(
            X_frame,
            n_splits=n_splits,
            train_size=train_size,
            test_size=test_size,
            gap=gap,
            expanding=expanding,
        )
    ):
        X_train = X_frame.iloc[train_idx]
        X_test = X_frame.iloc[test_idx]
        y_train = y_series.iloc[train_idx]
        y_test = y_series.iloc[test_idx]
        regime_train = regime_frame.iloc[train_idx]
        regime_test = regime_frame.iloc[test_idx]
        regime_test_inference = regime_test
        if regime_state_contracts is not None:
            regime_test_inference = slice_regime_state_contracts(regime_state_contracts, X_test.index)
        weight_train = None
        if sample_weight is not None:
            weight_train = pd.Series(sample_weight, index=X_frame.index).iloc[train_idx]

        train_coverage = summarize_regime_coverage(regime_train, regime_column=regime_column, config=coverage_config)
        test_coverage = summarize_regime_coverage(regime_test, regime_column=regime_column, config=coverage_config)

        bundle, training_report = train_regime_aware_model(
            X_train,
            y_train,
            regime_train,
            strategy=strategy,
            model_type=model_type,
            model_params=model_params,
            feature_config=feature_config,
            coverage_config=coverage_config,
            regime_column=regime_column,
            min_samples_per_regime=min_samples_per_regime,
            sample_weight=weight_train,
        )
        predictions, probabilities, inference_report = bundle.predict_with_probability_report(X_test, regime_test_inference)
        metrics = evaluate_regime_aware_predictions(y_test, predictions, probabilities)

        folds.append(
            {
                "fold": fold_number,
                "train_index": X_train.index,
                "test_index": X_test.index,
                "coverage": {"train": train_coverage, "test": test_coverage},
                "training_report": training_report,
                "inference_report": inference_report,
                "metrics": metrics,
            }
        )
        oos_predictions.append(predictions)
        oos_probabilities.append(probabilities)
        last_model = bundle

    if last_model is None:
        raise RuntimeError("No walk-forward splits were generated for regime-aware training")

    combined_predictions = pd.concat(oos_predictions).sort_index()
    combined_probabilities = pd.concat(oos_probabilities).sort_index().reindex(combined_predictions.index)
    return {
        "strategy": strategy,
        "folds": folds,
        "last_model": last_model,
        "oos_predictions": combined_predictions,
        "oos_probabilities": combined_probabilities,
        "coverage_summary": {
            "train_ok_share": float(np.mean([fold["coverage"]["train"]["coverage_ok"] for fold in folds])),
            "test_ok_share": float(np.mean([fold["coverage"]["test"]["coverage_ok"] for fold in folds])),
        },
    }


__all__ = [
    "RegimeAwareFeatureFrame",
    "RegimeAwareModelBundle",
    "build_regime_aware_feature_frame",
    "build_specialist_health_contracts",
    "build_specialist_library_snapshot",
    "build_specialist_specs_from_bundle",
    "evaluate_regime_aware_predictions",
    "summarize_regime_coverage",
    "train_regime_aware_model",
    "train_regime_aware_walk_forward",
]