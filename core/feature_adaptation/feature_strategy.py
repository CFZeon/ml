"""Feature-strategy regime-aware adapter helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .contracts import FeaturePolicyContract
from .runtime import resolve_feature_adaptation_config


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


def _sanitize_base_frame(frame: Any) -> pd.DataFrame:
    return (
        _coerce_feature_frame(frame)
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


def resolve_feature_strategy_adapter_config(
    feature_config: Mapping[str, Any] | None = None,
    *,
    regime_column: str = "regime",
) -> dict[str, Any]:
    configured = _clone_mapping(feature_config)
    feature_adaptation = resolve_feature_adaptation_config(configured.get("feature_adaptation") or {})
    interaction_budget = _clone_mapping(feature_adaptation.get("interaction_budget") or {})
    legacy_interactions_enabled = bool(configured.get("regime_interactions", True))
    interaction_enabled = bool(interaction_budget.get("enabled", legacy_interactions_enabled))
    max_dummy_cardinality = int(
        interaction_budget.get("max_dummy_cardinality", configured.get("max_dummy_cardinality", 16))
    )
    max_interaction_features = int(
        interaction_budget.get("max_features", configured.get("max_interaction_features", 4))
    )
    max_interaction_regimes = int(
        interaction_budget.get("max_regimes", configured.get("max_interaction_regimes", 6))
    )
    return {
        **configured,
        "feature_adaptation": feature_adaptation,
        "interaction_budget": interaction_budget,
        "enabled": True,
        "regime_column": str(configured.get("regime_column", regime_column or "regime")),
        "interaction_enabled": interaction_enabled,
        "max_dummy_cardinality": max(1, max_dummy_cardinality),
        "max_interaction_features": max(0, max_interaction_features),
        "max_interaction_regimes": max(0, max_interaction_regimes),
        "include_regime_state_columns": bool(interaction_budget.get("include_regime_state_columns", True)),
        "include_dummy_columns": bool(interaction_budget.get("include_dummy_columns", True)),
        "include_volatility_normalization": bool(
            interaction_budget.get("include_volatility_normalization", True)
        ),
        "requested_sections": list(feature_adaptation.get("requested_sections") or []),
    }


@dataclass(frozen=True)
class FrozenDummySpec:
    source_column: str
    category: str
    output_column: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_column": str(self.source_column),
            "category": str(self.category),
            "output_column": str(self.output_column),
        }


@dataclass(frozen=True)
class FrozenInteractionSpec:
    feature_column: str
    dummy_column: str
    output_column: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_column": str(self.feature_column),
            "dummy_column": str(self.dummy_column),
            "output_column": str(self.output_column),
        }


@dataclass
class RegimeFeatureStrategyAdapter:
    config: Mapping[str, Any] | None = None
    regime_column: str = "regime"
    _resolved_config: dict[str, Any] = field(default_factory=dict, init=False)
    _base_columns: list[str] = field(default_factory=list, init=False)
    _regime_columns: list[str] = field(default_factory=list, init=False)
    _fit_row_count: int = field(default=0, init=False)
    _feature_metadata_columns: int = field(default=0, init=False)
    _numeric_regime_columns: tuple[str, ...] = field(default_factory=tuple, init=False)
    _regime_state_columns: tuple[str, ...] = field(default_factory=tuple, init=False)
    _dummy_specs: tuple[FrozenDummySpec, ...] = field(default_factory=tuple, init=False)
    _normalized_source_columns: tuple[str, ...] = field(default_factory=tuple, init=False)
    _normalized_columns: tuple[str, ...] = field(default_factory=tuple, init=False)
    _volatility_source_column: str | None = field(default=None, init=False)
    _interaction_specs: tuple[FrozenInteractionSpec, ...] = field(default_factory=tuple, init=False)
    _generated_columns: tuple[str, ...] = field(default_factory=tuple, init=False)
    _feature_columns: tuple[str, ...] = field(default_factory=tuple, init=False)
    _candidate_interaction_features: tuple[str, ...] = field(default_factory=tuple, init=False)
    _candidate_interaction_regimes: tuple[str, ...] = field(default_factory=tuple, init=False)

    def __post_init__(self) -> None:
        self._resolved_config = resolve_feature_strategy_adapter_config(
            self.config,
            regime_column=self.regime_column,
        )
        self.regime_column = str(self._resolved_config.get("regime_column", self.regime_column or "regime"))

    def fit(
        self,
        X: Any,
        regime_frame: Any,
        feature_metadata: Mapping[str, Any] | None = None,
    ) -> "RegimeFeatureStrategyAdapter":
        base = _sanitize_base_frame(X)
        regime = _coerce_regime_frame(regime_frame, base.index)
        self._base_columns = [str(column) for column in base.columns]
        self._regime_columns = [str(column) for column in regime.columns]
        self._fit_row_count = int(len(base))
        self._feature_metadata_columns = int(len(dict(feature_metadata or {})))

        numeric_regime = regime.select_dtypes(include=[np.number]).copy()
        self._numeric_regime_columns = tuple(str(column) for column in numeric_regime.columns)
        if bool(self._resolved_config.get("include_regime_state_columns", True)):
            self._regime_state_columns = tuple(
                f"regime_state__{column}" for column in self._numeric_regime_columns
            )
        else:
            self._regime_state_columns = tuple()

        dummy_specs: list[FrozenDummySpec] = []
        if bool(self._resolved_config.get("include_dummy_columns", True)):
            max_dummy_cardinality = int(self._resolved_config.get("max_dummy_cardinality", 16))
            for column in regime.columns:
                series = regime[column]
                non_null = series.dropna()
                if non_null.empty or int(non_null.nunique()) > max_dummy_cardinality:
                    continue
                categories = sorted({str(value) for value in series.fillna("missing").astype(str).tolist()})
                for category in categories:
                    dummy_specs.append(
                        FrozenDummySpec(
                            source_column=str(column),
                            category=str(category),
                            output_column=f"regime__{column}_{category}",
                        )
                    )
        self._dummy_specs = tuple(dummy_specs)

        volatility_candidates = [
            column
            for column in self._numeric_regime_columns
            if any(term in column.lower() for term in ("vol", "atr", "range", "dispersion"))
        ]
        self._volatility_source_column = volatility_candidates[0] if volatility_candidates else None
        if bool(self._resolved_config.get("include_volatility_normalization", True)) and self._volatility_source_column:
            normalized_sources = [
                str(column)
                for column in self._base_columns
                if any(term in str(column).lower() for term in ("ret", "return", "momentum", "slope"))
                and not str(column).startswith(("regime_state__", "regime__", "vol_norm__", "cond__"))
            ]
            self._normalized_source_columns = tuple(normalized_sources)
            self._normalized_columns = tuple(f"vol_norm__{column}" for column in normalized_sources)
        else:
            self._normalized_source_columns = tuple()
            self._normalized_columns = tuple()

        dummy_columns = [spec.output_column for spec in self._dummy_specs]
        if bool(self._resolved_config.get("interaction_enabled", True)) and dummy_columns:
            candidate_features = [
                str(column)
                for column in self._base_columns
                if str(column) not in dummy_columns
                and not str(column).startswith(("regime_state__", "vol_norm__", "cond__"))
            ][: int(self._resolved_config.get("max_interaction_features", 4))]
            candidate_regimes = dummy_columns[: int(self._resolved_config.get("max_interaction_regimes", 6))]
            interaction_specs = []
            for feature_name in candidate_features:
                for dummy_name in candidate_regimes:
                    interaction_specs.append(
                        FrozenInteractionSpec(
                            feature_column=feature_name,
                            dummy_column=dummy_name,
                            output_column=f"cond__{feature_name}__{dummy_name}",
                        )
                    )
            self._candidate_interaction_features = tuple(candidate_features)
            self._candidate_interaction_regimes = tuple(candidate_regimes)
            self._interaction_specs = tuple(interaction_specs)
        else:
            self._candidate_interaction_features = tuple()
            self._candidate_interaction_regimes = tuple()
            self._interaction_specs = tuple()

        self._generated_columns = tuple(
            list(self._regime_state_columns)
            + [spec.output_column for spec in self._dummy_specs]
            + list(self._normalized_columns)
            + [spec.output_column for spec in self._interaction_specs]
        )
        self._feature_columns = tuple(self._base_columns + list(self._generated_columns))
        return self

    def transform(
        self,
        X: Any,
        regime_frame: Any,
    ) -> tuple[pd.DataFrame, FeaturePolicyContract]:
        base = _sanitize_base_frame(X).reindex(columns=self._base_columns, fill_value=0.0)
        regime = _coerce_regime_frame(regime_frame, base.index)
        transformed = base.copy()

        if self._regime_state_columns and self._numeric_regime_columns:
            numeric_regime = regime.select_dtypes(include=[np.number]).copy()
            numeric_regime = numeric_regime.reindex(columns=list(self._numeric_regime_columns), fill_value=0.0)
            for source_column, output_column in zip(self._numeric_regime_columns, self._regime_state_columns):
                transformed[output_column] = pd.to_numeric(numeric_regime[source_column], errors="coerce").fillna(0.0)

        for spec in self._dummy_specs:
            if spec.source_column in regime.columns:
                encoded = regime[spec.source_column].fillna("missing").astype(str)
            else:
                encoded = pd.Series("missing", index=regime.index, dtype=object)
            transformed[spec.output_column] = encoded.eq(spec.category).astype(float)

        if self._normalized_columns and self._volatility_source_column:
            if self._volatility_source_column in regime.columns:
                volatility_scale = pd.to_numeric(
                    regime[self._volatility_source_column], errors="coerce"
                ).abs()
                volatility_scale = volatility_scale.replace(0.0, np.nan).fillna(1.0).clip(lower=1e-6)
            else:
                volatility_scale = pd.Series(1.0, index=regime.index, dtype=float)
            for source_column, output_column in zip(self._normalized_source_columns, self._normalized_columns):
                transformed[output_column] = pd.to_numeric(
                    transformed[source_column], errors="coerce"
                ).fillna(0.0) / volatility_scale

        for spec in self._interaction_specs:
            feature_values = pd.to_numeric(transformed[spec.feature_column], errors="coerce").fillna(0.0)
            dummy_values = pd.to_numeric(transformed[spec.dummy_column], errors="coerce").fillna(0.0)
            transformed[spec.output_column] = feature_values * dummy_values

        transformed = transformed.reindex(columns=list(self._feature_columns), fill_value=0.0)
        manifest = self.manifest()
        policy = FeaturePolicyContract(
            policy_id=str(manifest.get("policy_id", "regime_feature_strategy")),
            feature_columns=[str(column) for column in transformed.columns],
            disabled_columns=[],
            generated_columns=list(self._generated_columns),
            regime_column=self.regime_column,
            scaling_mode="identity",
            fallback_mode="identity",
            sparse_regimes=[],
            metadata={
                "adapter_type": str(manifest.get("adapter_type", "regime_feature_strategy")),
                "no_op": bool(len(self._generated_columns) == 0),
                "requested_enabled": True,
                "requested_sections": list(self._resolved_config.get("requested_sections") or []),
                "fit_row_count": int(self._fit_row_count),
                "transform_row_count": int(len(transformed)),
                "regime_row_count": int(len(regime)),
                "regime_columns": list(self._regime_state_columns) + [spec.output_column for spec in self._dummy_specs],
                "normalized_columns": list(self._normalized_columns),
                "interaction_columns": [spec.output_column for spec in self._interaction_specs],
                "interaction_enabled": bool(self._resolved_config.get("interaction_enabled", True)),
                "interaction_budget": {
                    "max_features": int(self._resolved_config.get("max_interaction_features", 4)),
                    "max_regimes": int(self._resolved_config.get("max_interaction_regimes", 6)),
                    "max_dummy_cardinality": int(self._resolved_config.get("max_dummy_cardinality", 16)),
                },
                "selected_interaction_features": list(self._candidate_interaction_features),
                "selected_interaction_regimes": list(self._candidate_interaction_regimes),
                "volatility_source_column": self._volatility_source_column,
            },
        )
        return transformed, policy

    def manifest(self) -> dict[str, Any]:
        return {
            "adapter_type": "regime_feature_strategy",
            "policy_id": "regime_feature_strategy",
            "regime_column": self.regime_column,
            "fit_row_count": int(self._fit_row_count),
            "feature_columns": list(self._feature_columns),
            "feature_count": int(len(self._feature_columns)),
            "base_columns": list(self._base_columns),
            "base_column_count": int(len(self._base_columns)),
            "generated_columns": list(self._generated_columns),
            "generated_column_count": int(len(self._generated_columns)),
            "regime_columns": list(self._regime_columns),
            "regime_column_count": int(len(self._regime_columns)),
            "feature_metadata_columns": int(self._feature_metadata_columns),
            "requested_sections": list(self._resolved_config.get("requested_sections") or []),
            "interaction_enabled": bool(self._resolved_config.get("interaction_enabled", True)),
            "max_dummy_cardinality": int(self._resolved_config.get("max_dummy_cardinality", 16)),
            "max_interaction_features": int(self._resolved_config.get("max_interaction_features", 4)),
            "max_interaction_regimes": int(self._resolved_config.get("max_interaction_regimes", 6)),
            "regime_state_columns": list(self._regime_state_columns),
            "regime_state_column_count": int(len(self._regime_state_columns)),
            "dummy_columns": [spec.output_column for spec in self._dummy_specs],
            "dummy_column_count": int(len(self._dummy_specs)),
            "dummy_specs": [spec.to_dict() for spec in self._dummy_specs],
            "normalized_columns": list(self._normalized_columns),
            "normalized_column_count": int(len(self._normalized_columns)),
            "normalized_source_columns": list(self._normalized_source_columns),
            "volatility_source_column": self._volatility_source_column,
            "interaction_columns": [spec.output_column for spec in self._interaction_specs],
            "interaction_column_count": int(len(self._interaction_specs)),
            "interaction_specs": [spec.to_dict() for spec in self._interaction_specs],
            "selected_interaction_features": list(self._candidate_interaction_features),
            "selected_interaction_regimes": list(self._candidate_interaction_regimes),
            "no_op": bool(len(self._generated_columns) == 0),
        }


def build_feature_strategy_adapter(
    feature_config: Mapping[str, Any] | None = None,
    *,
    regime_column: str = "regime",
) -> RegimeFeatureStrategyAdapter:
    return RegimeFeatureStrategyAdapter(config=feature_config, regime_column=regime_column)


__all__ = [
    "RegimeFeatureStrategyAdapter",
    "build_feature_strategy_adapter",
    "resolve_feature_strategy_adapter_config",
]