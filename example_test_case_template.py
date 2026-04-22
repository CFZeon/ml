"""Copy-and-edit template for building a new research case.

Usage
-----
    python example_test_case_template.py
"""

from core import ATR, BollingerBands, MACD, RSI, ResearchPipeline
from example_utils import (
    build_spot_research_config,
    clone_config_with_overrides,
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


def build_case_config():
    # Edit this block first when creating a new scenario.
    base_config = build_spot_research_config(
        symbol="BTCUSDT",
        interval="1h",
        start="2024-01-01",
        end="2024-06-01",
        indicators=[RSI(14), MACD(), BollingerBands(20), ATR(14)],
        context_symbols=["ETHUSDT", "SOLUSDT"],
    )

    return clone_config_with_overrides(
        base_config,
        {
            "labels": {
                "max_holding": 18,
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


def main():
    sep = "=" * 60
    pipeline = ResearchPipeline(build_case_config())

    print_section(sep, 1, "Fetching data")
    data = pipeline.fetch_data()
    print(f"  rows         : {len(data)}")
    print(f"  range        : {data.index[0]} -> {data.index[-1]}")

    print_section(sep, 2, "Running indicators")
    indicator_run = pipeline.run_indicators()
    print(f"  indicators   : {[result.kind for result in indicator_run.results]}")

    print_section(sep, 3, "Building features and screening stationarity")
    features = pipeline.build_features()
    print(f"  feature count: {features.shape[1]}")
    stationarity = pipeline.check_stationarity()
    print_stationarity_summary(stationarity)

    print_section(sep, 4, "Previewing regime features")
    regimes = pipeline.detect_regimes()["regimes"]
    print_regime_summary(regimes)

    print_section(sep, 5, "Building labels")
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

    print_section(sep, 8, "Training")
    training = pipeline.train_models()
    print_training_summary(training)

    print_section(sep, 9, "Generating signals")
    signals = pipeline.generate_signals()
    print_signal_summary(signals, allow_short=False)

    print_section(sep, 10, "Backtesting")
    backtest = pipeline.run_backtest()
    print_backtest_summary(backtest)

    print(f"\n{sep}\nTemplate example complete.\n{sep}")


if __name__ == "__main__":
    main()