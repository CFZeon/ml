"""Run a constrained Optuna-backed AutoML search on the research pipeline.

Usage
-----
    python example_automl.py
"""

from pathlib import Path

from core import ATR, BollingerBands, MACD, RSI, ResearchPipeline
from example_utils import (
    build_example_universe_config,
    print_alignment_summary,
    print_automl_summary,
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
    end = "2024-05-01"
    context_symbols = ["ETHUSDT"]

    automl_storage = Path(".cache") / "automl" / "example_automl_v3.db"
    automl_storage.parent.mkdir(parents=True, exist_ok=True)
    if automl_storage.exists():
        automl_storage.unlink()

    pipeline = ResearchPipeline(
        {
            "data": {
                "symbol": symbol,
                "interval": interval,
                "start": start,
                "end": end,
                "futures_context": {"enabled": False},
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
                "schema_version": "indicator_aware_v7_example_workflow",
            },
            "feature_selection": {"enabled": True, "max_features": 96, "min_mi_threshold": 0.0005},
            "regime": {"method": "hmm"},  # HMM with stable norm-sorted state ordering
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
                "policy_mode": "validation_calibrated",
                "avg_win": 0.02,
                "avg_loss": 0.02,
                "shrinkage_alpha": 0.5,
                "fraction": 0.5,
                "min_trades_for_kelly": 30,
                "max_kelly_fraction": 0.5,
                "threshold": 0.01,
                "edge_threshold": 0.05,
                "meta_threshold": 0.55,
                "tuning_min_trades": 5,
            },
            "backtest": {
                "equity": 10_000,
                "fee_rate": 0.001,
                "slippage_rate": 0.0002,
                "slippage_model": "sqrt_impact",
                "use_open_execution": True,
                "signal_delay_bars": 2,
            },
            "automl": {
                "enabled": True,
                "n_trials": 2,
                "seed": 42,
                "validation_fraction": 0.2,
                "minimum_dsr_threshold": None,
                "locked_holdout_enabled": False,
                "enable_pruning": False,
                "objective": "sharpe_ratio",
                "study_name": "BTCUSDT_1h_example_automl_v3",
                "storage": str(automl_storage),
                "selection_policy": {
                    "enabled": False,
                },
                "overfitting_control": {
                    "enabled": False,
                    "deflated_sharpe": {"enabled": False},
                    "pbo": {"enabled": False},
                    "post_selection": {"enabled": False},
                },
                "search_space": {
                    "features": {
                        "lags": {"type": "categorical", "choices": ["1,3,6", "1,4,12"]},
                        "frac_diff_d": {"type": "categorical", "choices": [0.4, 0.6]},
                        "rolling_window": {"type": "categorical", "choices": [20, 28]},
                        "squeeze_quantile": {"type": "categorical", "choices": [0.15, 0.2]},
                    },
                    "feature_selection": {
                        "enabled": {"type": "categorical", "choices": [True]},
                        "max_features": {"type": "categorical", "choices": [48, 64]},
                        "min_mi_threshold": {"type": "categorical", "choices": [0.0, 0.0005]},
                    },
                    "labels": {
                        "pt_mult": {"type": "categorical", "choices": [1.5, 2.0]},
                        "sl_mult": {"type": "categorical", "choices": [1.5, 2.0]},
                        "max_holding": {"type": "categorical", "choices": [12, 24]},
                        "min_return": {"type": "categorical", "choices": [0.0005, 0.001]},
                        "volatility_window": {"type": "categorical", "choices": [24]},
                        "barrier_tie_break": {"type": "categorical", "choices": ["sl"]},
                    },
                    "regime": {
                        "n_regimes": {"type": "categorical", "choices": [2]},
                    },
                    "model": {
                        "type": {"type": "categorical", "choices": ["gbm"]},
                        "gap": {"type": "categorical", "choices": [24]},
                        "validation_fraction": {"type": "categorical", "choices": [0.2]},
                        "meta_n_splits": {"type": "categorical", "choices": [2]},
                    },
                },
            },
        }
    )

    print_section(sep, 1, "Fetching BTCUSDT spot data")
    data = pipeline.fetch_data()
    print(f"  rows         : {len(data)}")
    print(f"  range        : {data.index[0]} -> {data.index[-1]}")

    print_section(sep, 2, "Running indicators")
    indicator_run = pipeline.run_indicators()
    print(f"  indicators   : {[result.kind for result in indicator_run.results]}")

    print_section(sep, 3, "Running AutoML search")
    automl = pipeline.run_automl()
    print_automl_summary(automl)

    print_section(sep, 4, "Rebuilding the canonical workflow with the selected config")
    features = pipeline.build_features()
    print(f"  feature count: {features.shape[1]}")
    stationarity = pipeline.check_stationarity()
    print_stationarity_summary(stationarity)

    print_section(sep, 5, "Previewing regime features")
    regimes = pipeline.detect_regimes()["regimes"]
    print_regime_summary(regimes)

    print_section(sep, 6, "Building labels and aligning research matrix")
    labels = pipeline.build_labels()
    print_label_summary(labels)
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
    signals = pipeline.generate_signals()
    print_signal_summary(signals, allow_short=False)

    print_section(sep, 10, "Backtesting")
    backtest = pipeline.run_backtest()
    print_backtest_summary(backtest)


if __name__ == "__main__":
    main()