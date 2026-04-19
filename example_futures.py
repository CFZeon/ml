"""USDT-M futures example using real Binance data and the VectorBT-first adapter.

Usage
-----
    python example_futures.py
"""

from core import ATR, MACD, RSI, ResearchPipeline
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
    sep = "=" * 60
    pipeline = ResearchPipeline(
        {
            "data": {
                "symbol": "BTCUSDT",
                "interval": "1h",
                "start": "2024-01-01",
                "end": "2024-04-01",
                "market": "um_futures",
                "futures_context": {"enabled": True, "include_recent_stats": True},
                "cross_asset_context": {"symbols": ["ETHUSDT", "SOLUSDT"], "market": "um_futures"},
            },
            "indicators": [RSI(14), MACD(), ATR(14)],
            "features": {
                "lags": [1, 3, 6],
                "frac_diff_d": 0.4,
                "rolling_window": 24,
                "context_timeframes": ["4h"],
            },
            "feature_selection": {"enabled": True, "max_features": 64, "min_mi_threshold": 0.0},
            "regime": {"method": "hmm"},  # HMM with stable norm-sorted state ordering
            "labels": {
                "kind": "triple_barrier",
                "pt_sl": (1.5, 1.5),
                "max_holding": 12,
                "min_return": 0.0005,
                "volatility_window": 24,
                "barrier_tie_break": "sl",
            },
            "model": {
                "type": "logistic",
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
                "fee_rate": 0.0004,
                "slippage_rate": 0.0002,
                "slippage_model": "sqrt_impact",
                "engine": "pandas",
                "valuation_price": "mark",
                "apply_funding": True,
                "allow_short": True,
                "leverage": 1.5,
                "use_open_execution": True,
                "signal_delay_bars": 1,
                "futures_account": {
                    "enabled": True,
                    "margin_mode": "isolated",
                    "warning_margin_ratio": 0.8,
                    "leverage_brackets_data": {
                        "symbol": "BTCUSDT",
                        "brackets": [
                            {
                                "bracket": 1,
                                "initial_leverage": 20.0,
                                "notional_floor": 0.0,
                                "notional_cap": 50_000.0,
                                "maint_margin_ratio": 0.02,
                                "cum": 0.0,
                            },
                            {
                                "bracket": 2,
                                "initial_leverage": 10.0,
                                "notional_floor": 50_000.0,
                                "notional_cap": 250_000.0,
                                "maint_margin_ratio": 0.04,
                                "cum": 0.0,
                            },
                        ],
                    },
                },
            },
        }
    )

    print_section(sep, 1, "Fetching BTCUSDT futures data")
    data = pipeline.fetch_data()
    filters = pipeline.state.get("symbol_filters", {})
    futures_context = pipeline.state.get("futures_context", {})
    contract_spec = pipeline.state.get("futures_contract_spec", {})
    print(f"  rows         : {len(data)}")
    print(f"  range        : {data.index[0]} -> {data.index[-1]}")
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

    print_section(sep, 7, "CPCV training")
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