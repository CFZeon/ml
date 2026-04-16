"""End-to-end pipeline: fetch → indicators → features → labels → train → backtest.

Usage
-----
    python example.py
"""

from core import RSI, ATR, BollingerBands, MACD, ResearchPipeline
from example_utils import (
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
    SEP = "=" * 60
    pipeline = ResearchPipeline(
        {
            "data": {
                "symbol": "BTCUSDT",
                "interval": "1h",
                "start": "2024-01-01",
                "end": "2024-06-01",
                "futures_context": {"enabled": True, "include_recent_stats": True},
                "cross_asset_context": {"symbols": ["ETHUSDT", "SOLUSDT", "BNBUSDT"]},
            },
            "indicators": [RSI(14), MACD(), BollingerBands(20), ATR(14)],
            "features": {
                "lags": [1, 3, 6],
                "frac_diff_d": 0.4,
                "rolling_window": 20,
                "squeeze_quantile": 0.2,
                "context_timeframes": ["4h", "1d"],
            },
            "feature_selection": {"enabled": True, "max_features": 96, "min_mi_threshold": 0.0005},
            "regime": {"method": "explicit"},
            "labels": {
                "kind": "triple_barrier",
                "pt_sl": (2.0, 2.0),
                "max_holding": 24,
                "min_return": 0.001,
                "volatility_window": 24,
                "barrier_tie_break": "sl",
            },
            "model": {"type": "gbm", "n_splits": 3, "gap": 24, "validation_fraction": 0.2, "meta_n_splits": 2},
            "signals": {
                "avg_win": 0.02,
                "avg_loss": 0.02,
                "fraction": 0.5,
                "threshold": 0.01,
                "edge_threshold": 0.05,
                "meta_threshold": 0.55,
                "tuning_min_trades": 5,
            },
            "backtest": {
                "equity": 10_000,
                "fee_rate": 0.001,
                "slippage_rate": 0.0002,
                "engine": "vectorbt",
                "use_open_execution": True,
                "signal_delay_bars": 2,
            },
        }
    )

    print_section(SEP, 1, "Fetching BTCUSDT spot data")
    df = pipeline.fetch_data()
    print(f"  rows         : {len(df)}")
    print(f"  range        : {df.index[0]} -> {df.index[-1]}")

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

    print_section(SEP, 8, "Walk-forward training")
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
