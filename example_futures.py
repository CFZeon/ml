"""USDT-M futures example using real Binance data and the VectorBT-first adapter.

Usage
-----
    python example_futures.py
"""

from core import ATR, MACD, RSI, ResearchPipeline


def print_training_summary(training):
    print(f"  avg accuracy : {training['avg_accuracy']:.4f}")
    print(f"  avg f1       : {training['avg_f1_macro']:.4f}")
    print(f"  avg selected : {training['feature_selection']['avg_selected_features']}")
    print(f"  tuned signals: {training['last_signal_params']}")


def print_backtest_summary(backtest):
    print(f"  engine       : {backtest['engine']}")
    print(f"  start equity : ${backtest['starting_equity']:,.2f}")
    print(f"  end equity   : ${backtest['ending_equity']:,.2f}")
    print(f"  net profit   : ${backtest['net_profit']:,.2f} ({backtest['net_profit_pct']:.2%})")
    print(f"  funding pnl  : ${backtest['funding_pnl']:,.2f}")
    print(f"  fees paid    : ${backtest['fees_paid']:,.2f}")
    print(f"  slippage     : ${backtest['slippage_paid']:,.2f}")
    print(f"  sharpe ratio : {backtest['sharpe_ratio']}")
    print(f"  max drawdown : {backtest['max_drawdown']:.2%}")
    print(f"  trades       : {backtest['total_trades']}")
    print(f"  closed trades: {backtest['closed_trades']}")
    print(f"  trade win rt : {backtest['trade_win_rate']:.2%}")


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
            "regime": {"n_regimes": 2},
            "labels": {"kind": "fixed_horizon", "horizon": 8, "threshold": 0.0005},
            "model": {
                "type": "logistic",
                "n_splits": 3,
                "gap": 8,
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
                "equity": 10_000,
                "fee_rate": 0.0004,
                "slippage_rate": 0.0002,
                "engine": "vectorbt",
                "valuation_price": "mark",
                "apply_funding": True,
                "allow_short": True,
                "leverage": 1.5,
                "use_open_execution": True,
                "signal_delay_bars": 1,
            },
        }
    )

    print(f"\n{sep}\nStep 1 - Fetching BTCUSDT futures data\n{sep}")
    data = pipeline.fetch_data()
    filters = pipeline.state.get("symbol_filters", {})
    futures_context = pipeline.state.get("futures_context", {})
    print(f"  rows={len(data)}  range={data.index[0]} -> {data.index[-1]}")
    print(
        "  symbol filters: "
        f"tick={filters.get('tick_size')}  step={filters.get('step_size')}  min_notional={filters.get('min_notional')}"
    )
    print(f"  futures context tables: {sorted(futures_context)}")

    print(f"\n{sep}\nStep 2 - Running indicators and building features\n{sep}")
    indicator_run = pipeline.run_indicators()
    features = pipeline.build_features()
    stationarity = pipeline.check_stationarity()
    screening = stationarity["feature_screening"]["summary"]
    print(f"  indicators    : {[result.kind for result in indicator_run.results]}")
    print(f"  feature count : {features.shape[1]}")
    print(
        f"  screening     : {screening['screened_feature_count']}/{screening['total_features']}  "
        f"transformed={screening['transformed_features']}  dropped={screening['dropped_features']}"
    )

    print(f"\n{sep}\nStep 3 - Regimes, labels, and alignment\n{sep}")
    regimes = pipeline.detect_regimes()["regimes"]
    labels = pipeline.build_labels()
    aligned = pipeline.align_data()
    pipeline.select_features()
    weights = pipeline.compute_sample_weights()
    print(f"  regime counts : {regimes.value_counts().to_dict()}")
    print(f"  labels        : {labels['label'].value_counts().to_dict()}")
    print(f"  samples       : {len(aligned['X'])}")
    print(f"  weight range  : [{weights.min():.3f}, {weights.max():.3f}]")

    print(f"\n{sep}\nStep 4 - Walk-forward training\n{sep}")
    training = pipeline.train_models()
    print_training_summary(training)

    print(f"\n{sep}\nStep 5 - Signals and backtest\n{sep}")
    signals = pipeline.generate_signals()
    signal_classes = signals["signals"]
    backtest = pipeline.run_backtest()
    print(
        f"  long={int((signal_classes == 1).sum())}  "
        f"short={int((signal_classes == -1).sum())}  flat={int((signal_classes == 0).sum())}"
    )
    print_backtest_summary(backtest)

    print(f"\n{sep}\nFutures example complete.\n{sep}")


if __name__ == "__main__":
    main()