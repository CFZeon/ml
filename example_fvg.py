"""End-to-end pipeline using a config-driven Fair Value Gap indicator setup.

Usage
-----
    python example_fvg.py
"""

import pandas as pd

from core import (
    ResearchPipeline,
)
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


def build_fvg_regime_features(pipeline):
    data = pipeline.require("data")
    features = pipeline.require("features")
    return pd.DataFrame(
        {
            "vol_20": data["close"].pct_change().rolling(20).std(),
            "fvg_gap_imbalance": features["fvg_main_gap_imbalance"],
            "fvg_distance_spread": features["fvg_main_distance_spread"],
        }
    ).dropna()


def main():
    sep = "=" * 60
    pipeline = ResearchPipeline(
        {
            "data": {
                "symbol": "BTCUSDT",
                "interval": "1h",
                "start": "2024-01-01",
                "end": "2024-06-01",
            },
            "indicators": [
                {"kind": "rsi", "params": {"period": 14}},
                {"kind": "atr", "params": {"period": 14}},
                {"kind": "fvg", "params": {"name": "fvg_main", "min_gap_pct": 0.0005}},
            ],
            "features": {
                "lags": [1, 3, 6],
                "frac_diff_d": 0.4,
                "rolling_window": 20,
                "squeeze_quantile": 0.2,
            },
            "feature_selection": {"enabled": True, "max_features": 96, "min_mi_threshold": 0.0005},
            "regime": {"method": "hmm", "builder": build_fvg_regime_features},  # HMM with stable norm-sorted state ordering
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

    print_section(sep, 1, "Fetching BTCUSDT spot data")
    df = pipeline.fetch_data()
    print(f"  rows         : {len(df)}")
    print(f"  range        : {df.index[0]} -> {df.index[-1]}")

    print_section(sep, 2, "Running config-driven indicators")
    indicator_run = pipeline.run_indicators()
    df = indicator_run.frame

    fvg_meta = indicator_run.metadata["fvg_main"]
    print(f"  indicator names: {list(indicator_run.metadata)}")
    print(f"  FVG outputs: {fvg_meta['output_columns']}")
    print(f"  FVG fill states: {fvg_meta['fill_state_encoding']}")

    active_bull = int(df["fvg_main_bull_active_count"].gt(0).sum())
    active_bear = int(df["fvg_main_bear_active_count"].gt(0).sum())
    print(f"  bars with active bullish gaps: {active_bull}")
    print(f"  bars with active bearish gaps: {active_bear}")

    print_section(sep, 3, "Building features and screening stationarity")
    features = pipeline.build_features()
    print(f"  feature count: {features.shape[1]}")

    stationarity = pipeline.check_stationarity()
    print_stationarity_summary(stationarity)

    print_section(sep, 4, "Previewing regime features")
    regime_result = pipeline.detect_regimes()
    regimes = regime_result["regimes"]
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

    print_section(sep, 9, "Generating signals")
    signal_result = pipeline.generate_signals()
    print_signal_summary(signal_result, allow_short=False)

    print_section(sep, 10, "Backtesting")
    backtest = pipeline.run_backtest()
    print_backtest_summary(backtest)

    print(f"\n{sep}\nPipeline complete.\n{sep}")


if __name__ == "__main__":
    main()