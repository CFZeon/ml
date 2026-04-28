"""Active-trading spot demo with cross-asset context.

Usage
-----
    python example_active_spot.py
    python example_active_spot.py --local-certification
"""

from core import ATR, BollingerBands, MACD, RSI, ResearchPipeline
from core.execution import NAUTILUS_AVAILABLE
from example_utils import (
    build_spot_research_config,
    clone_config_with_overrides,
    parse_local_certification_args,
    prepare_example_runtime_config,
    print_alignment_summary,
    print_backtest_summary,
    print_feature_selection_summary,
    print_label_summary,
    print_regime_summary,
    print_section,
    print_signal_summary,
    print_stationarity_summary,
    print_training_summary,
    print_weight_summary,
)


def main():
    args = parse_local_certification_args("Run the active spot demo.")
    sep = "=" * 60
    symbol = "BTCUSDT"
    interval = "1h"
    start = "2024-01-01"
    end = "2024-06-01"
    context_symbols = ["ETHUSDT", "SOLUSDT", "BNBUSDT"]
    indicators = [RSI(14), MACD(), BollingerBands(20), ATR(14)]

    config = build_spot_research_config(
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
        indicators=indicators,
        context_symbols=context_symbols,
    )
    config = clone_config_with_overrides(
        config,
        {
            "model": {
                "cv_method": "walk_forward",
                "n_splits": 3,
                "train_size": 720,
                "test_size": 168,
                "gap": 6,
                "validation_fraction": 0.2,
                "meta_n_splits": 2,
            },
            "signals": {
                "threshold": 0.0,
                "edge_threshold": 0.0,
                "fraction": 0.75,
                "meta_threshold": 0.5,
                "profitability_threshold": 0.5,
                "expected_edge_threshold": 0.0,
                "sizing_mode": "expected_utility",
            },
            "backtest": {
                "signal_delay_bars": 1,
            },
        },
    )
    try:
        config = prepare_example_runtime_config(
            config,
            market="spot",
            local_certification=args.local_certification,
            nautilus_available=NAUTILUS_AVAILABLE,
            example_name="example_active_spot.py",
        )
    except RuntimeError as exc:
        print(str(exc))
        raise SystemExit(2) from exc

    pipeline = ResearchPipeline(config)

    print_section(sep, 1, "Fetching active spot dataset")
    data = pipeline.fetch_data()
    example_runtime = dict(config.get("example_runtime") or {})
    print(f"  rows         : {len(data)}")
    print(f"  range        : {data.index[0]} -> {data.index[-1]}")
    if example_runtime:
        print(f"  runtime mode : {example_runtime.get('mode')}")
        print(f"  runtime note : {example_runtime.get('note')}")

    print_section(sep, 2, "Running indicators")
    indicator_run = pipeline.run_indicators()
    print(f"  indicators   : {[result.kind for result in indicator_run.results]}")
    print(f"  columns      : {len(indicator_run.frame.columns)}")

    print_section(sep, 3, "Building features and screening stationarity")
    features = pipeline.build_features()
    print(f"  feature count: {features.shape[1]}")
    stationarity = pipeline.check_stationarity()
    print_stationarity_summary(stationarity)

    print_section(sep, 4, "Previewing regime features")
    regimes = pipeline.detect_regimes()["regimes"]
    print_regime_summary(regimes)

    print_section(sep, 5, "Building triple-barrier labels")
    labels = pipeline.build_labels()
    print_label_summary(labels)

    print_section(sep, 6, "Aligning research matrix")
    aligned = pipeline.align_data()
    print_alignment_summary(aligned)

    print_section(sep, 7, "Previewing feature-selection and weighting")
    selection = pipeline.select_features()
    print_feature_selection_summary(selection)
    weights = pipeline.compute_sample_weights()
    print_weight_summary(weights)

    print_section(sep, 8, "Walk-forward training")
    training = pipeline.train_models()
    print_training_summary(training)

    print_section(sep, 9, "Generating active long-only signals")
    signal_result = pipeline.generate_signals()
    print_signal_summary(signal_result, allow_short=False)

    print_section(sep, 10, "Backtesting")
    backtest = pipeline.run_backtest()
    print_backtest_summary(backtest)
    total_trades = float(backtest.get("total_trades") or 0.0)
    if total_trades <= 0.0:
        print(
            "  note         : zero trades are unexpected for this active demo on the reference window; "
            "re-check the cached data and signal settings if this occurs."
        )

    print(f"\n{sep}\nActive spot example complete.\n{sep}")


if __name__ == "__main__":
    main()