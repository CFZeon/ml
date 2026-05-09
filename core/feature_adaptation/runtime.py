"""Feature-adaptation runtime helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from .contracts import FeaturePolicyContract
from .masking import (
    RegimeMaskingAdapter,
    canonicalize_selection_fallback_mode,
    canonicalize_selection_mode,
)
from .scaling import RegimeConditionedScalingAdapter, canonicalize_fallback_mode, canonicalize_scaling_mode


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


def _requested_sections(config: Mapping[str, Any] | None) -> list[str]:
    configured = dict(config or {})
    sections = []
    if configured.get("scaling"):
        sections.append("scaling")
    if configured.get("selection"):
        sections.append("selection")
    if configured.get("interaction_budget"):
        sections.append("interaction_budget")
    if configured.get("disable_incompatible_features") is not None:
        sections.append("disable_incompatible_features")
    return sections


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def validate_feature_adaptation_config_contract(
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    configured = _clone_mapping(config)
    for section in ["scaling", "selection", "interaction_budget"]:
        value = configured.get(section)
        if value is not None and not isinstance(value, Mapping):
            raise ValueError(f"feature_adaptation.{section} must be a mapping when provided")

    interaction_budget = _clone_mapping(configured.get("interaction_budget") or {})
    integer_bounds = {
        "max_features": 0,
        "max_regimes": 0,
        "max_dummy_cardinality": 1,
    }
    for key, minimum in integer_bounds.items():
        if interaction_budget.get(key) is None:
            continue
        value = int(interaction_budget[key])
        if value < minimum:
            raise ValueError(f"feature_adaptation.interaction_budget.{key} must be >= {minimum}")

    scaling = _clone_mapping(configured.get("scaling") or {})
    selection = _clone_mapping(configured.get("selection") or {})
    positive_integer_bounds = {
        "scaling": {"min_regime_samples": 1},
        "selection": {"min_regime_samples": 1, "min_feature_rows": 1},
    }
    for section_name, bounds in positive_integer_bounds.items():
        section = scaling if section_name == "scaling" else selection
        for key, minimum in bounds.items():
            if section.get(key) is None:
                continue
            value = int(section[key])
            if value < minimum:
                raise ValueError(f"feature_adaptation.{section_name}.{key} must be >= {minimum}")
    return configured


def resolve_feature_adaptation_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    configured = validate_feature_adaptation_config_contract(config)
    scaling = _clone_mapping(configured.get("scaling") or {})
    selection = _clone_mapping(configured.get("selection") or {})
    interaction_budget = _clone_mapping(configured.get("interaction_budget") or {})
    explicit_enabled = configured.get("enabled")
    enabled = bool(configured) if explicit_enabled is None else bool(explicit_enabled)
    requested_scaling_mode = canonicalize_scaling_mode(scaling.get("mode") or "identity")
    fallback_mode = canonicalize_fallback_mode(
        scaling.get("fallback")
        or ("global" if requested_scaling_mode == "regime_conditioned" else "identity")
    )
    requested_selection_mode = canonicalize_selection_mode(selection.get("mode") or "identity")
    selection_fallback_mode = canonicalize_selection_fallback_mode(
        selection.get("fallback")
        or ("global" if requested_selection_mode == "per_regime_mask" else "identity")
    )
    requested_sections = _requested_sections(configured)
    deferred_runtime = bool(enabled and bool(interaction_budget))
    return {
        **configured,
        "enabled": enabled,
        "scaling": scaling,
        "selection": selection,
        "interaction_budget": interaction_budget,
        "requested_scaling_mode": requested_scaling_mode,
        "requested_selection_mode": requested_selection_mode,
        "selection_fallback_mode": selection_fallback_mode,
        "fallback_mode": fallback_mode,
        "requested_sections": requested_sections,
        "deferred_runtime": deferred_runtime,
    }


def validate_feature_adaptation_runtime_support(
    config: Mapping[str, Any] | None = None,
    *,
    regime_aware_strategy: str | None = None,
) -> dict[str, Any]:
    resolved = resolve_feature_adaptation_config(config)
    strategy = str(regime_aware_strategy or "").strip().lower()
    selection_mode = str(resolved.get("requested_selection_mode", "identity"))
    selection_fallback_mode = str(resolved.get("selection_fallback_mode", "identity"))
    disable_incompatible_features = bool(resolved.get("disable_incompatible_features", False))
    if selection_mode not in {"identity", "per_regime_mask"}:
        raise ValueError(f"Unsupported feature_adaptation selection mode={selection_mode!r}")
    if selection_fallback_mode not in {"global", "identity"}:
        raise ValueError(
            f"Unsupported feature_adaptation selection fallback={selection_fallback_mode!r}"
        )
    if (
        strategy == "feature"
        and bool(resolved.get("enabled", False))
        and (
            str(resolved.get("requested_scaling_mode", "identity")) != "identity"
            or selection_mode != "identity"
            or disable_incompatible_features
        )
    ):
        raise ValueError(
            "feature_adaptation non-identity scaling, masking, or incompatible-feature disabling is not supported "
            "with model.regime_aware.strategy='feature'; runtime parity remains deferred until the feature-strategy "
            "inference path is unified"
        )
    return resolved


class IdentityFeatureAdapter:
    def __init__(self, *, config: Mapping[str, Any] | None = None, regime_column: str = "regime"):
        self.config = resolve_feature_adaptation_config(config)
        self.regime_column = str(regime_column or "regime")
        self._feature_columns: list[str] = []
        self._regime_columns: list[str] = []
        self._fit_row_count = 0
        self._feature_metadata_columns = 0

    def fit(
        self,
        X: Any,
        regime_frame: Any,
        feature_metadata: Mapping[str, Any] | None = None,
    ) -> "IdentityFeatureAdapter":
        X_frame = _coerce_feature_frame(X)
        regime = _coerce_regime_frame(regime_frame, X_frame.index)
        self._feature_columns = [str(column) for column in X_frame.columns]
        self._regime_columns = [str(column) for column in regime.columns]
        self._fit_row_count = int(len(X_frame))
        self._feature_metadata_columns = int(len(dict(feature_metadata or {})))
        return self

    def transform(
        self,
        X: Any,
        regime_frame: Any,
    ) -> tuple[pd.DataFrame, FeaturePolicyContract]:
        X_frame = _coerce_feature_frame(X)
        regime = _coerce_regime_frame(regime_frame, X_frame.index)
        transformed = X_frame.reindex(columns=self._feature_columns, fill_value=0.0).copy()
        policy = FeaturePolicyContract(
            policy_id="identity",
            feature_columns=[str(column) for column in transformed.columns],
            disabled_columns=[],
            generated_columns=[],
            regime_column=self.regime_column,
            scaling_mode="identity",
            fallback_mode=str(self.config.get("fallback_mode", "identity")),
            sparse_regimes=[],
            metadata={
                "adapter_type": "identity",
                "no_op": True,
                "requested_enabled": bool(self.config.get("enabled", False)),
                "requested_scaling_mode": str(self.config.get("requested_scaling_mode", "identity")),
                "requested_selection_mode": str(self.config.get("requested_selection_mode", "identity")),
                "selection_fallback_mode": str(self.config.get("selection_fallback_mode", "identity")),
                "requested_sections": list(self.config.get("requested_sections") or []),
                "deferred_runtime": bool(self.config.get("deferred_runtime", False)),
                "fit_row_count": int(self._fit_row_count),
                "transform_row_count": int(len(transformed)),
                "regime_row_count": int(len(regime)),
            },
        )
        return transformed, policy

    def manifest(self) -> dict[str, Any]:
        return {
            "adapter_type": "identity",
            "policy_id": "identity",
            "regime_column": self.regime_column,
            "fit_row_count": int(self._fit_row_count),
            "feature_columns": list(self._feature_columns),
            "feature_count": int(len(self._feature_columns)),
            "regime_columns": list(self._regime_columns),
            "regime_column_count": int(len(self._regime_columns)),
            "feature_metadata_columns": int(self._feature_metadata_columns),
            "requested_enabled": bool(self.config.get("enabled", False)),
            "requested_scaling_mode": str(self.config.get("requested_scaling_mode", "identity")),
            "requested_selection_mode": str(self.config.get("requested_selection_mode", "identity")),
            "selection_fallback_mode": str(self.config.get("selection_fallback_mode", "identity")),
            "fallback_mode": str(self.config.get("fallback_mode", "identity")),
            "requested_sections": list(self.config.get("requested_sections") or []),
            "deferred_runtime": bool(self.config.get("deferred_runtime", False)),
            "no_op": True,
        }


class CompositeFeatureAdapter:
    def __init__(
        self,
        *,
        stages: list[tuple[str, Any]],
        config: Mapping[str, Any] | None = None,
        regime_column: str = "regime",
    ):
        self.stages = list(stages)
        self.config = resolve_feature_adaptation_config(config)
        self.regime_column = str(regime_column or "regime")
        self._feature_columns: list[str] = []
        self._regime_columns: list[str] = []
        self._feature_metadata_columns = 0
        self._fit_row_count = 0

    def fit(
        self,
        X: Any,
        regime_frame: Any,
        feature_metadata: Mapping[str, Any] | None = None,
    ) -> "CompositeFeatureAdapter":
        current = _coerce_feature_frame(X)
        regime = _coerce_regime_frame(regime_frame, current.index)
        self._feature_columns = [str(column) for column in current.columns]
        self._regime_columns = [str(column) for column in regime.columns]
        self._feature_metadata_columns = int(len(dict(feature_metadata or {})))
        self._fit_row_count = int(len(current))
        for _, stage in self.stages:
            stage.fit(current, regime, feature_metadata=feature_metadata)
            current, _ = stage.transform(current, regime)
        self._feature_columns = [str(column) for column in current.columns]
        return self

    def transform(
        self,
        X: Any,
        regime_frame: Any,
    ) -> tuple[pd.DataFrame, FeaturePolicyContract]:
        current = _coerce_feature_frame(X)
        regime = _coerce_regime_frame(regime_frame, current.index)
        stage_policies: dict[str, FeaturePolicyContract] = {}
        for stage_name, stage in self.stages:
            current, stage_policy = stage.transform(current, regime)
            stage_policies[stage_name] = stage_policy
        manifest = self.manifest()
        return current, _merge_stage_policies(
            stage_policies,
            manifest=manifest,
            config=self.config,
            regime_column=self.regime_column,
            transformed_frame=current,
            regime_frame=regime,
        )

    def manifest(self) -> dict[str, Any]:
        stage_manifests = {
            stage_name: dict(stage.manifest() or {})
            for stage_name, stage in self.stages
        }
        scaling_manifest = dict(stage_manifests.get("scaling") or {})
        masking_manifest = dict(stage_manifests.get("masking") or {})
        policy_ids = [
            str(stage_manifests.get(stage_name, {}).get("policy_id", stage_name))
            for stage_name, _ in self.stages
        ]
        return {
            "adapter_type": "composite_feature_adaptation",
            "policy_id": "__then__".join(policy_ids) if policy_ids else "composite_feature_adaptation",
            "regime_column": self.regime_column,
            "fit_row_count": int(self._fit_row_count),
            "feature_columns": list(self._feature_columns),
            "feature_count": int(len(self._feature_columns)),
            "regime_columns": list(self._regime_columns),
            "regime_column_count": int(len(self._regime_columns)),
            "feature_metadata_columns": int(self._feature_metadata_columns),
            "requested_enabled": bool(self.config.get("enabled", False)),
            "requested_scaling_mode": str(self.config.get("requested_scaling_mode", "identity")),
            "requested_selection_mode": str(self.config.get("requested_selection_mode", "identity")),
            "selection_fallback_mode": str(self.config.get("selection_fallback_mode", "identity")),
            "fallback_mode": str(
                masking_manifest.get("selection_fallback_mode")
                or scaling_manifest.get("fallback_mode")
                or self.config.get("fallback_mode", "identity")
            ),
            "requested_sections": list(self.config.get("requested_sections") or []),
            "deferred_runtime": bool(self.config.get("deferred_runtime", False)),
            "no_op": bool(all(stage_manifests.get(stage_name, {}).get("no_op", False) for stage_name, _ in self.stages)),
            "stages": stage_manifests,
            "eligible_scaling_column_count": int(scaling_manifest.get("eligible_scaling_column_count", 0)),
            "passthrough_column_count": int(scaling_manifest.get("passthrough_column_count", masking_manifest.get("passthrough_column_count", 0))),
            "regime_bank_count": int(scaling_manifest.get("regime_bank_count", 0)),
            "skipped_regime_count": int(scaling_manifest.get("skipped_regime_count", masking_manifest.get("skipped_regime_count", 0))),
            "constant_column_count": int(scaling_manifest.get("constant_column_count", 0)),
            "mask_candidate_column_count": int(masking_manifest.get("mask_candidate_column_count", 0)),
            "global_active_column_count": int(masking_manifest.get("global_active_column_count", 0)),
            "regime_mask_count": int(masking_manifest.get("regime_mask_count", 0)),
            "disabled_columns": list(masking_manifest.get("disabled_columns") or []),
            "disabled_column_count": int(masking_manifest.get("disabled_column_count", 0)),
            "disabled_columns_by_reason": dict(masking_manifest.get("disabled_columns_by_reason") or {}),
        }


def _merge_stage_policies(
    stage_policies: Mapping[str, FeaturePolicyContract],
    *,
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    regime_column: str,
    transformed_frame: pd.DataFrame,
    regime_frame: pd.DataFrame,
) -> FeaturePolicyContract:
    scaling_policy = stage_policies.get("scaling")
    masking_policy = stage_policies.get("masking")
    disabled_columns = _dedupe_preserve_order(
        [
            str(column)
            for policy in stage_policies.values()
            for column in policy.disabled_columns
        ]
    )
    generated_columns = _dedupe_preserve_order(
        [
            str(column)
            for policy in stage_policies.values()
            for column in policy.generated_columns
        ]
    )
    sparse_regimes = sorted(
        {
            str(regime_key)
            for policy in stage_policies.values()
            for regime_key in policy.sparse_regimes
        }
    )
    scaling_metadata = dict((scaling_policy.metadata if scaling_policy is not None else {}) or {})
    masking_metadata = dict((masking_policy.metadata if masking_policy is not None else {}) or {})
    fallback_rows_total = int(
        masking_metadata.get("fallback_rows_total", scaling_metadata.get("fallback_rows_total", 0))
    )
    fallback_rows_by_reason = dict(
        masking_metadata.get("fallback_rows_by_reason")
        or scaling_metadata.get("fallback_rows_by_reason")
        or {}
    )
    metadata = {
        "adapter_type": str(manifest.get("adapter_type", "identity")),
        "no_op": bool(manifest.get("no_op", True)),
        "requested_enabled": bool(config.get("enabled", False)),
        "requested_scaling_mode": str(config.get("requested_scaling_mode", "identity")),
        "requested_selection_mode": str(config.get("requested_selection_mode", "identity")),
        "selection_fallback_mode": str(config.get("selection_fallback_mode", "identity")),
        "requested_sections": list(config.get("requested_sections") or []),
        "deferred_runtime": bool(config.get("deferred_runtime", False)),
        "fit_row_count": int(manifest.get("fit_row_count", 0)),
        "transform_row_count": int(len(transformed_frame)),
        "regime_row_count": int(len(regime_frame)),
        "fallback_rows_total": fallback_rows_total,
        "fallback_rows_by_reason": fallback_rows_by_reason,
        "scaling_stage": scaling_policy.to_dict() if scaling_policy is not None else None,
        "masking_stage": masking_policy.to_dict() if masking_policy is not None else None,
        "scaling_fallback_rows_total": int(scaling_metadata.get("fallback_rows_total", 0)),
        "scaling_fallback_rows_by_reason": dict(scaling_metadata.get("fallback_rows_by_reason") or {}),
        "selection_mode": str(masking_metadata.get("selection_mode", manifest.get("requested_selection_mode", "identity"))),
        "selection_fallback_rows_total": int(masking_metadata.get("fallback_rows_total", 0)),
        "selection_fallback_rows_by_reason": dict(masking_metadata.get("fallback_rows_by_reason") or {}),
        "mask_candidate_column_count": int(masking_metadata.get("mask_candidate_column_count", manifest.get("mask_candidate_column_count", 0))),
        "global_active_column_count": int(masking_metadata.get("global_active_column_count", manifest.get("global_active_column_count", 0))),
        "regime_mask_count": int(masking_metadata.get("regime_mask_count", manifest.get("regime_mask_count", 0))),
        "masked_cell_count": int(masking_metadata.get("masked_cell_count", 0)),
        "disabled_columns_by_reason": dict(masking_metadata.get("disabled_columns_by_reason") or {}),
        "disabled_cell_count": int(masking_metadata.get("disabled_cell_count", 0)),
        "mask_assignment_counts": dict(masking_metadata.get("mask_assignment_counts") or {}),
    }
    return FeaturePolicyContract(
        policy_id=str(manifest.get("policy_id", "composite_feature_adaptation")),
        feature_columns=[str(column) for column in transformed_frame.columns],
        disabled_columns=disabled_columns,
        generated_columns=generated_columns,
        regime_column=str(regime_column or "regime"),
        scaling_mode=str(
            scaling_policy.scaling_mode if scaling_policy is not None else "identity"
        ),
        fallback_mode=str(
            masking_policy.fallback_mode
            if masking_policy is not None
            else (scaling_policy.fallback_mode if scaling_policy is not None else config.get("fallback_mode", "identity"))
        ),
        sparse_regimes=sparse_regimes,
        metadata=metadata,
    )


def build_feature_adapter(
    config: Mapping[str, Any] | None = None,
    *,
    regime_column: str = "regime",
):
    resolved = resolve_feature_adaptation_config(config)
    scaling_mode = str(resolved.get("requested_scaling_mode", "identity"))
    selection_mode = str(resolved.get("requested_selection_mode", "identity"))
    disable_incompatible_features = bool(resolved.get("disable_incompatible_features", False))
    if bool(resolved.get("enabled", False)) and (
        selection_mode != "identity" or disable_incompatible_features
    ):
        scaling_stage = (
            RegimeConditionedScalingAdapter(config=resolved, regime_column=regime_column)
            if scaling_mode in {"global", "regime_conditioned"}
            else IdentityFeatureAdapter(config=resolved, regime_column=regime_column)
        )
        masking_stage = RegimeMaskingAdapter(config=resolved, regime_column=regime_column)
        return CompositeFeatureAdapter(
            stages=[("scaling", scaling_stage), ("masking", masking_stage)],
            config=resolved,
            regime_column=regime_column,
        )
    if bool(resolved.get("enabled", False)) and scaling_mode in {"global", "regime_conditioned"}:
        return RegimeConditionedScalingAdapter(config=resolved, regime_column=regime_column)
    return IdentityFeatureAdapter(config=resolved, regime_column=regime_column)


@dataclass
class FeatureAdaptationBatchResult:
    adapter: Any
    fit_frame: pd.DataFrame
    validation_frame: pd.DataFrame | None
    test_frame: pd.DataFrame
    fit_policy: FeaturePolicyContract
    validation_policy: FeaturePolicyContract | None
    test_policy: FeaturePolicyContract
    manifest: dict[str, Any]
    summary: dict[str, Any]


def _policy_metadata(policy: FeaturePolicyContract | None) -> dict[str, Any]:
    return dict((policy.metadata if policy is not None else {}) or {})


def apply_feature_adaptation_to_splits(
    fit_frame: Any,
    validation_frame: Any,
    test_frame: Any,
    *,
    fit_regime_frame: Any = None,
    validation_regime_frame: Any = None,
    test_regime_frame: Any = None,
    feature_metadata: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    regime_column: str = "regime",
) -> FeatureAdaptationBatchResult:
    fit_features = _coerce_feature_frame(fit_frame)
    validation_features = None if validation_frame is None else _coerce_feature_frame(validation_frame)
    test_features = _coerce_feature_frame(test_frame)
    fit_regime = _coerce_regime_frame(fit_regime_frame, fit_features.index)
    validation_regime = (
        pd.DataFrame(index=pd.Index([], dtype=object))
        if validation_features is None
        else _coerce_regime_frame(validation_regime_frame, validation_features.index)
    )
    test_regime = _coerce_regime_frame(test_regime_frame, test_features.index)

    adapter = build_feature_adapter(config, regime_column=regime_column)
    adapter.fit(fit_features, fit_regime, feature_metadata=feature_metadata)

    transformed_fit, fit_policy = adapter.transform(fit_features, fit_regime)
    transformed_validation = None
    validation_policy = None
    if validation_features is not None:
        transformed_validation, validation_policy = adapter.transform(validation_features, validation_regime)
    transformed_test, test_policy = adapter.transform(test_features, test_regime)
    manifest = dict(adapter.manifest() or {})
    fit_metadata = _policy_metadata(fit_policy)
    validation_metadata = _policy_metadata(validation_policy)
    test_metadata = _policy_metadata(test_policy)
    summary = {
        "requested_enabled": bool(manifest.get("requested_enabled", False)),
        "requested_scaling_mode": str(manifest.get("requested_scaling_mode", "identity")),
        "requested_selection_mode": str(manifest.get("requested_selection_mode", "identity")),
        "selection_fallback_mode": str(manifest.get("selection_fallback_mode", "identity")),
        "fallback_mode": str(manifest.get("fallback_mode", "identity")),
        "requested_sections": list(manifest.get("requested_sections") or []),
        "adapter_type": str(manifest.get("adapter_type", "identity")),
        "policy_id": str(manifest.get("policy_id", fit_policy.policy_id)),
        "no_op": bool(fit_metadata.get("no_op", manifest.get("no_op", True))),
        "deferred_runtime": bool(manifest.get("deferred_runtime", False)),
        "input_features": int(fit_features.shape[1]),
        "output_features": int(transformed_fit.shape[1]),
        "feature_count_delta": int(transformed_fit.shape[1] - fit_features.shape[1]),
        "disabled_columns": int(len(fit_policy.disabled_columns)),
        "generated_columns": int(len(fit_policy.generated_columns)),
        "fit_rows": int(len(transformed_fit)),
        "validation_rows": int(len(transformed_validation)) if transformed_validation is not None else 0,
        "test_rows": int(len(transformed_test)),
        "scaling_mode": str(fit_policy.scaling_mode),
        "eligible_scaling_column_count": int(manifest.get("eligible_scaling_column_count", 0)),
        "passthrough_column_count": int(manifest.get("passthrough_column_count", 0)),
        "regime_bank_count": int(manifest.get("regime_bank_count", 0)),
        "skipped_regime_count": int(manifest.get("skipped_regime_count", 0)),
        "constant_column_count": int(manifest.get("constant_column_count", 0)),
        "mask_candidate_column_count": int(manifest.get("mask_candidate_column_count", 0)),
        "global_active_column_count": int(manifest.get("global_active_column_count", 0)),
        "regime_mask_count": int(manifest.get("regime_mask_count", 0)),
        "fit_fallback_rows_total": int(fit_metadata.get("fallback_rows_total", 0)),
        "validation_fallback_rows_total": int(validation_metadata.get("fallback_rows_total", 0)),
        "test_fallback_rows_total": int(test_metadata.get("fallback_rows_total", 0)),
        "fit_scaling_fallback_rows_total": int(fit_metadata.get("scaling_fallback_rows_total", 0)),
        "validation_scaling_fallback_rows_total": int(validation_metadata.get("scaling_fallback_rows_total", 0)),
        "test_scaling_fallback_rows_total": int(test_metadata.get("scaling_fallback_rows_total", 0)),
        "fit_selection_fallback_rows_total": int(fit_metadata.get("selection_fallback_rows_total", 0)),
        "validation_selection_fallback_rows_total": int(validation_metadata.get("selection_fallback_rows_total", 0)),
        "test_selection_fallback_rows_total": int(test_metadata.get("selection_fallback_rows_total", 0)),
        "fit_fallback_rows_by_reason": dict(fit_metadata.get("fallback_rows_by_reason") or {}),
        "validation_fallback_rows_by_reason": dict(validation_metadata.get("fallback_rows_by_reason") or {}),
        "test_fallback_rows_by_reason": dict(test_metadata.get("fallback_rows_by_reason") or {}),
        "fit_masked_cell_count": int(fit_metadata.get("masked_cell_count", 0)),
        "validation_masked_cell_count": int(validation_metadata.get("masked_cell_count", 0)),
        "test_masked_cell_count": int(test_metadata.get("masked_cell_count", 0)),
        "fit_disabled_columns_by_reason": dict(fit_metadata.get("disabled_columns_by_reason") or {}),
        "validation_disabled_columns_by_reason": dict(validation_metadata.get("disabled_columns_by_reason") or {}),
        "test_disabled_columns_by_reason": dict(test_metadata.get("disabled_columns_by_reason") or {}),
    }
    return FeatureAdaptationBatchResult(
        adapter=adapter,
        fit_frame=transformed_fit,
        validation_frame=transformed_validation,
        test_frame=transformed_test,
        fit_policy=fit_policy,
        validation_policy=validation_policy,
        test_policy=test_policy,
        manifest=manifest,
        summary=summary,
    )


__all__ = [
    "CompositeFeatureAdapter",
    "FeatureAdaptationBatchResult",
    "IdentityFeatureAdapter",
    "apply_feature_adaptation_to_splits",
    "build_feature_adapter",
    "resolve_feature_adaptation_config",
    "validate_feature_adaptation_config_contract",
    "validate_feature_adaptation_runtime_support",
]