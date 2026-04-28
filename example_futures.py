"""USDT-M futures example using real Binance data and the liquidation-aware pandas adapter.

Recent Binance futures aggregate stats are shifted to publication-safe timestamps
by default so the context block does not treat summarized periods as instantly known.

Usage
-----
    python example_futures.py
    python example_futures.py --local-certification
"""

from core import ATR, MACD, RSI, ResearchPipeline
from core.execution import NAUTILUS_AVAILABLE
from example_utils import (
    build_futures_research_config,
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
    args = parse_local_certification_args("Run the baseline futures research example.")
    sep = "=" * 60
    symbol = "BTCUSDT"
    interval = "1h"
    start = "2024-01-01"
    end = "2024-04-01"
    context_symbols = ["ETHUSDT", "SOLUSDT"]
    indicators = [RSI(14), MACD(), ATR(14)]

    config = clone_config_with_overrides(
        build_futures_research_config(
            symbol=symbol,
            interval=interval,
            start=start,
            end=end,
            indicators=indicators,
            context_symbols=context_symbols,
        ),
        {
            "model": {
                "cv_method": "walk_forward",
                "n_splits": 3,
                "train_size": 360,
                "test_size": 120,
                "gap": 3,
                "validation_fraction": 0.2,
                "meta_n_splits": 2,
            }
        },
    )
    try:
        config = prepare_example_runtime_config(
            config,
            market="um_futures",
            local_certification=args.local_certification,
            nautilus_available=NAUTILUS_AVAILABLE,
            example_name="example_futures.py",
        )
    except RuntimeError as exc:
        print(str(exc))
        raise SystemExit(2) from exc

    pipeline = ResearchPipeline(config)

    print_section(sep, 1, "Fetching BTCUSDT futures data")
    data = pipeline.fetch_data()
    example_runtime = dict(config.get("example_runtime") or {})
    filters = pipeline.state.get("symbol_filters", {})
    futures_context = pipeline.state.get("futures_context", {})
    contract_spec = pipeline.state.get("futures_contract_spec", {})
    print(f"  rows         : {len(data)}")
    print(f"  range        : {data.index[0]} -> {data.index[-1]}")
    if example_runtime:
        print(f"  runtime mode : {example_runtime.get('mode')}")
        print(f"  runtime note : {example_runtime.get('note')}")
    print(
        "  symbol filters: "
        f"tick={filters.get('tick_size')}  step={filters.get('step_size')}  min_notional={filters.get('min_notional')}"
    )
    if contract_spec:
        print(
            "  contract spec : "
            f"margin={contract_spec.get('margin_asset')}  "
            f"liq_fee={contract_spec.get('liquidation_fee_rate')}  "
            f"take_bound={contract_spec.get('market_take_bound')}"
        )
    print(f"  futures context tables: {sorted(futures_context)}")

    print_section(sep, 2, "Running indicators")
    indicator_run = pipeline.run_indicators()
    print(f"  indicators   : {[result.kind for result in indicator_run.results]}")

    print_section(sep, 3, "Building features and screening stationarity")
    features = pipeline.build_features()
    family_summary = pipeline.state.get("feature_family_summary", {})
    stationarity = pipeline.check_stationarity()
    print(f"  feature count : {features.shape[1]}")
    print(f"  families      : {family_summary.get('selected_family_counts', {})}")
    print_stationarity_summary(stationarity)

    print_section(sep, 4, "Previewing regime features")
    regimes = pipeline.detect_regimes()["regimes"]
    print_regime_summary(regimes)

    print_section(sep, 5, "Building labels and aligning research matrix")
    labels = pipeline.build_labels()
    aligned = pipeline.align_data()
    print_label_summary(labels)
    print_alignment_summary(aligned)

    print_section(sep, 6, "Previewing feature-selection and weighting")
    selection = pipeline.select_features()
    print_feature_selection_summary(selection)
    weights = pipeline.compute_sample_weights()
    print_weight_summary(weights)

    print_section(sep, 7, "Walk-forward training")
    training = pipeline.train_models()
    print_training_summary(training)

    print_section(sep, 8, "Generating signals")
    signals = pipeline.generate_signals()
    print_signal_summary(signals)

    print_section(sep, 9, "Backtesting")
    backtest = pipeline.run_backtest()
    print_backtest_summary(backtest)

    print(f"\n{sep}\nFutures example complete.\n{sep}")


if __name__ == "__main__":
    main()