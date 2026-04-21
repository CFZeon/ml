"""Active-trading spot demo with cross-asset context.

Usage
-----
    python example_active_spot.py
"""

from core import ATR, BollingerBands, MACD, RSI, ResearchPipeline
from example_utils import (
    build_example_universe_config,
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
    sep = "=" * 60
    symbol = "BTCUSDT"
    interval = "1h"
    start = "2024-01-01"
    end = "2024-06-01"
    context_symbols = ["ETHUSDT", "SOLUSDT", "BNBUSDT"]

    pipeline = ResearchPipeline(
        {
            "data": {
                "symbol": symbol,
                "interval": interval,
                "start": start,
                "end": end,
                "futures_context": {"enabled": True, "include_recent_stats": True},
                "cross_asset_context": {"symbols": context_symbols},
            },
            "universe": build_example_universe_config(
                symbol,
                context_symbols=context_symbols,
                market="spot",
                snapshot_timestamp=start,
            ),
            "indicators": [RSI(14), MACD(), BollingerBands(20), ATR(14)],
            "features": {
                "lags": [1, 3, 6],
                "frac_diff_d": 0.4,
                "rolling_window": 20,
                "squeeze_quantile": 0.2,
                "context_timeframes": ["4h", "1d"],
            },
            "feature_selection": {"enabled": True, "max_features": 96, "min_mi_threshold": 0.0005},
            "regime": {"method": "hmm"},
            "labels": {
                "kind": "triple_barrier",
                "pt_sl": (2.0, 2.0),
                "max_holding": 24,
                "min_return": 0.001,
                "volatility_window": 24,
                "barrier_tie_break": "sl",
            },
            "model": {
                "type": "gbm",
                "cv_method": "cpcv",
                "n_blocks": 4,
                "test_blocks": 2,
                "validation_fraction": 0.2,
                "meta_n_splits": 2,
            },
            "signals": {
                "threshold": 0.0,
                "edge_threshold": 0.0,
                "shrinkage_alpha": 0.5,
                "fraction": 0.75,
                "min_trades_for_kelly": 30,
                "max_kelly_fraction": 0.5,
                "meta_threshold": 0.5,
                "profitability_threshold": 0.5,
                "expected_edge_threshold": 0.0,
                "sizing_mode": "expected_utility",
                "tuning_min_trades": 5,
            },
            "backtest": {
                "equity": 10_000,
                "fee_rate": 0.001,
                "slippage_rate": 0.0002,
                "slippage_model": "sqrt_impact",
                "engine": "vectorbt",
                "use_open_execution": True,
                "signal_delay_bars": 1,
            },
        }
    )

    print_section(sep, 1, "Fetching active spot dataset")
    data = pipeline.fetch_data()
    print(f"  rows         : {len(data)}")
    print(f"  range        : {data.index[0]} -> {data.index[-1]}")

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

    print_section(sep, 8, "CPCV training")
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