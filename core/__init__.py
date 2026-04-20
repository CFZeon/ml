"""Core trading system – import everything from here.

    from core import fetch_binance_vision, RSI, train_model, run_backtest
"""

from .data import (
    fetch_binance_bars,
    fetch_binance_exchange_info,
    fetch_binance_futures_contract_spec,
    fetch_binance_symbol_filters,
    fetch_binance_vision,
    join_custom_data,
    join_custom_dataset,
    load_futures_leverage_brackets,
    load_custom_dataset,
)
from .data_quality import DataQualityResult, check_data_quality
from .drift import ADWINDetector, DriftMonitor, evaluate_drift_guardrails
from .context import (
    fetch_binance_futures_context,
    fetch_context_symbol_bars,
    build_futures_context_feature_block,
    build_cross_asset_context_feature_block,
    build_multi_timeframe_context_feature_block,
)
from .execution import (
    ExecutionPolicy,
    LiquidityInputResolver,
    NautilusExecutionAdapter,
    OrderIntent,
    resolve_execution_policy,
    resolve_liquidity_inputs,
)
from .indicators import (
    Indicator,
    IndicatorResult,
    IndicatorRunResult,
    RSI,
    MACD,
    BollingerBands,
    ATR,
    FairValueGap,
    INDICATOR_REGISTRY,
    build_indicator,
    build_indicators,
    run_indicators,
    attach_indicators,
)
from .pipeline import (
    AlignDataStep,
    AutoMLStep,
    BacktestStep,
    DataQualityStep,
    DEFAULT_STEPS,
    FeatureSelectionStep,
    FetchDataStep,
    FeaturesStep,
    IndicatorsStep,
    LabelsStep,
    PipelineStep,
    RegimeStep,
    ResearchPipeline,
    SampleWeightsStep,
    SignalsStep,
    StationarityStep,
    TrainModelsStep,
)
from .regime import (
    RegimeFeatureSet,
    build_default_regime_feature_set,
    build_regime_ablation_report,
    compute_regime_path_stability,
    detect_regime,
    summarize_regime_ablation_reports,
    summarize_regime_provenance,
)
from .features import (
    build_feature_set,
    build_features,
    check_stationarity,
    derive_feature_lineage,
    fractional_diff,
    screen_features_for_stationarity,
    select_features,
)
from .feature_governance import (
    apply_feature_retirement,
    derive_feature_metadata,
    evaluate_feature_admission,
    evaluate_feature_portability,
    filter_feature_metadata,
    summarize_feature_admission_reports,
    summarize_feature_portability,
)
from .labeling import (
    triple_barrier_labels,
    fixed_horizon_labels,
    trend_scanning_labels,
    sample_weights_by_uniqueness,
    sequential_bootstrap,
)
from .lookahead import run_lookahead_analysis
from .slippage import (
    DepthCurveImpactModel,
    FillAwareCostModel,
    FlatSlippageModel,
    OrderBookImpactModel,
    ProxyImpactModel,
    SlippageModel,
    SquareRootImpactModel,
)
from .models import (
    build_execution_outcome_frame,
    build_trade_outcome_frame,
    build_model,
    cpcv_split,
    walk_forward_split,
    train_model,
    train_meta_model,
    evaluate_model,
    save_model,
    load_model,
)
from .stat_tests import (
    align_post_selection_return_matrix,
    compute_hansen_spa,
    compute_post_selection_inference,
    compute_white_reality_check,
    select_post_selection_candidates,
)
from .backtest import kelly_fraction, run_backtest
from .reference_data import build_reference_overlay_feature_block
from .registry import (
    LocalRegistryStore,
    RegistryVersionManifest,
    build_feature_schema_hash,
    build_registry_manifest,
    evaluate_challenger_promotion,
)
from .universe import (
    HistoricalUniverseSnapshot,
    apply_symbol_lifecycle_policy,
    build_symbol_lifecycle_frame,
    evaluate_universe_eligibility,
    load_historical_universe_snapshot,
    normalize_universe_snapshot,
    persist_historical_universe_snapshot,
)
