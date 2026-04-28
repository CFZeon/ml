"""Futures example focused on trend-strength and breakout indicators.

Usage
-----
    python example_trend_breakout_futures.py
    python example_trend_breakout_futures.py --local-certification
"""

from core import ADX, ATR, DonchianChannels, MACD, RSI, ResearchPipeline
from core.execution import NAUTILUS_AVAILABLE
from example_utils import (
    build_example_universe_config,
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
    args = parse_local_certification_args("Run the trend-breakout futures example.")
    sep = "=" * 60
    symbol = "BTCUSDT"
    interval = "1h"
    start = "2024-01-01"
    end = "2024-04-01"
    context_symbols = ["ETHUSDT"]

    config = {
        "data": {
            "symbol": symbol,
            "interval": interval,
            "start": start,
            "end": end,
            "market": "um_futures",
            "futures_context": {"enabled": True, "include_recent_stats": True},
            "cross_asset_context": {"symbols": context_symbols, "market": "um_futures"},
        },
        "universe": build_example_universe_config(
            symbol,
            context_symbols=context_symbols,
            market="um_futures",
            snapshot_timestamp=start,
        ),
        "indicators": [RSI(14), MACD(), ATR(14), ADX(14), DonchianChannels(20)],
        "features": {
            "lags": [1, 2, 3],
            "frac_diff_d": 0.4,
            "rolling_window": 16,
            "context_timeframes": ["4h"],
        },
        "feature_selection": {"enabled": True, "max_features": 56, "min_mi_threshold": 0.0},
        "regime": {"method": "hmm", "enabled": False},
        "labels": {
            "kind": "fixed_horizon",
            "horizon": 3,
            "threshold": 0.0,
            "cost_rate": 0.0,
        },
        "model": {
            "type": "logistic",
            "cv_method": "walk_forward",
            "n_splits": 4,
            "train_size": 360,
            "test_size": 96,
            "gap": 3,
            "meta_n_splits": 2,
        },
        "signals": {
            "policy_mode": "theory_only",
            "avg_win": 0.012,
            "avg_loss": 0.008,
            "threshold": 0.0,
            "edge_threshold": 0.0,
            "fraction": 0.75,
            "min_trades_for_kelly": 10,
            "max_kelly_fraction": 0.5,
            "meta_threshold": 0.0,
            "profitability_threshold": 0.0,
            "expected_edge_threshold": 0.0,
            "sizing_mode": "expected_utility",
            "tuning_min_trades": 1,
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
            "leverage": 2.0,
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
                        }
                    ],
                },
            },
        },
    }
    try:
        config = prepare_example_runtime_config(
            config,
            market="um_futures",
            local_certification=args.local_certification,
            nautilus_available=NAUTILUS_AVAILABLE,
            example_name="example_trend_breakout_futures.py",
        )
    except RuntimeError as exc:
        print(str(exc))
        raise SystemExit(2) from exc

    pipeline = ResearchPipeline(config)

    print_section(sep, 1, "Fetching futures data")
    data = pipeline.fetch_data()
    example_runtime = dict(config.get("example_runtime") or {})
    filters = pipeline.state.get("symbol_filters", {})
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

    print_section(sep, 2, "Running trend-breakout indicators")
    indicator_run = pipeline.run_indicators()
    print(f"  indicators   : {[result.kind for result in indicator_run.results]}")

    print_section(sep, 3, "Building features and screening stationarity")
    features = pipeline.build_features()
    print(f"  feature count: {features.shape[1]}")
    stationarity = pipeline.check_stationarity()
    print_stationarity_summary(stationarity)

    print_section(sep, 4, "Previewing regime features")
    regimes = pipeline.detect_regimes()["regimes"]
    print("  mode         : disabled for this compact futures breakout demo")
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

    print_section(sep, 8, "Walk-forward training")
    training = pipeline.train_models()
    print_training_summary(training)

    print_section(sep, 9, "Generating signals")
    signals = pipeline.generate_signals()
    print_signal_summary(signals)

    print_section(sep, 10, "Backtesting")
    backtest = pipeline.run_backtest()
    print_backtest_summary(backtest)

    print(f"\n{sep}\nTrend-breakout futures example complete.\n{sep}")


if __name__ == "__main__":
    main()