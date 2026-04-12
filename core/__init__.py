"""Core trading system – import everything from here.

    from core import fetch_binance_vision, RSI, train_model, run_backtest
"""

from .data import fetch_binance_vision
from .context import (
    fetch_binance_futures_context,
    fetch_context_symbol_bars,
    build_futures_context_feature_block,
    build_cross_asset_context_feature_block,
    build_multi_timeframe_context_feature_block,
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
from .features import (
    build_feature_set,
    build_features,
    check_stationarity,
    fractional_diff,
    screen_features_for_stationarity,
    select_features,
)
from .labeling import (
    triple_barrier_labels,
    fixed_horizon_labels,
    sample_weights_by_uniqueness,
    sequential_bootstrap,
)
from .models import (
    build_model,
    walk_forward_split,
    train_model,
    train_meta_model,
    evaluate_model,
    detect_regime,
    save_model,
    load_model,
)
from .backtest import kelly_fraction, run_backtest
