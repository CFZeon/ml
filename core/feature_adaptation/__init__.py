"""Feature-adaptation runtime contracts and adapter helpers."""

from .contracts import BaseFeatureAdapter, FeaturePolicyContract
from .feature_strategy import (
    RegimeFeatureStrategyAdapter,
    build_feature_strategy_adapter,
    resolve_feature_strategy_adapter_config,
)
from .runtime import (
    FeatureAdaptationBatchResult,
    IdentityFeatureAdapter,
    apply_feature_adaptation_to_splits,
    build_feature_adapter,
    resolve_feature_adaptation_config,
    validate_feature_adaptation_config_contract,
    validate_feature_adaptation_runtime_support,
)
from .scaling import RegimeConditionedScalingAdapter

__all__ = [
    "BaseFeatureAdapter",
    "FeatureAdaptationBatchResult",
    "FeaturePolicyContract",
    "IdentityFeatureAdapter",
    "RegimeFeatureStrategyAdapter",
    "RegimeConditionedScalingAdapter",
    "apply_feature_adaptation_to_splits",
    "build_feature_adapter",
    "build_feature_strategy_adapter",
    "resolve_feature_adaptation_config",
    "resolve_feature_strategy_adapter_config",
    "validate_feature_adaptation_config_contract",
    "validate_feature_adaptation_runtime_support",
]