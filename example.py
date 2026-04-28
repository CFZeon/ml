"""End-to-end pipeline: fetch → indicators → features → labels → train → backtest.

Usage
-----
    python example.py
    python example.py --local-certification
"""

from core import RSI, ATR, BollingerBands, MACD, ResearchPipeline
from core.execution import NAUTILUS_AVAILABLE
from example_utils import (
    build_spot_research_config,
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
    args = parse_local_certification_args("Run the baseline spot research example.")
    SEP = "=" * 60
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
    try:
        config = prepare_example_runtime_config(
            config,
            market="spot",
            local_certification=args.local_certification,
            nautilus_available=NAUTILUS_AVAILABLE,
            example_name="example.py",
        )
    except RuntimeError as exc:
        print(str(exc))
        raise SystemExit(2) from exc

    pipeline = ResearchPipeline(config)

    print_section(SEP, 1, "Fetching BTCUSDT spot data")
    df = pipeline.fetch_data()
    example_runtime = dict(config.get("example_runtime") or {})
    print(f"  rows         : {len(df)}")
    print(f"  range        : {df.index[0]} -> {df.index[-1]}")
    if example_runtime:
        print(f"  runtime mode : {example_runtime.get('mode')}")
        print(f"  runtime note : {example_runtime.get('note')}")

    print_section(SEP, 2, "Running indicators")
    indicator_run = pipeline.run_indicators()
    df = indicator_run.frame
    print(f"  indicators   : {[result.kind for result in indicator_run.results]}")
    print(f"  columns      : {len(df.columns)}")

    print_section(SEP, 3, "Building features and screening stationarity")
    features = pipeline.build_features()
    print(f"  feature count: {features.shape[1]}")
    stationarity = pipeline.check_stationarity()
    print_stationarity_summary(stationarity)

    print_section(SEP, 4, "Previewing regime features")
    regime_result = pipeline.detect_regimes()
    regimes = regime_result["regimes"]
    print_regime_summary(regimes)

    print_section(SEP, 5, "Building triple-barrier labels")
    labels = pipeline.build_labels()
    print_label_summary(labels)

    print_section(SEP, 6, "Aligning research matrix")
    aligned = pipeline.align_data()
    print_alignment_summary(aligned)

    print_section(SEP, 7, "Previewing feature-selection and weighting")
    selection = pipeline.select_features()
    print_feature_selection_summary(selection)
    weights = pipeline.compute_sample_weights()
    print_weight_summary(weights)

    print_section(SEP, 8, "CPCV training")
    training = pipeline.train_models()
    print_training_summary(training)

    print_section(SEP, 9, "Generating signals")
    signal_result = pipeline.generate_signals()
    print_signal_summary(signal_result, allow_short=False)

    print_section(SEP, 10, "Backtesting")
    bt = pipeline.run_backtest()
    print_backtest_summary(bt)

    print(f"\n{SEP}\nPipeline complete.\n{SEP}")


if __name__ == "__main__":
    main()
