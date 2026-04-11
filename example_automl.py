"""Run a constrained Optuna-backed AutoML search on the research pipeline.

Usage
-----
    python example_automl.py
"""

from core import ATR, BollingerBands, MACD, RSI, ResearchPipeline


def main():
    sep = "=" * 60
    pipeline = ResearchPipeline(
        {
            "data": {
                "symbol": "BTCUSDT",
                "interval": "1h",
                "start": "2024-01-01",
                "end": "2026-01-01",
            },
            "indicators": [RSI(14), MACD(), BollingerBands(20), ATR(14)],
            "features": {
                "lags": [1, 3, 6],
                "frac_diff_d": 0.4,
                "rolling_window": 20,
                "squeeze_quantile": 0.2,
                "schema_version": "indicator_aware_v5_foldlocal",
            },
            "feature_selection": {"enabled": True, "max_features": 96, "min_mi_threshold": 0.0005},
            "regime": {"n_regimes": 2},
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
            "backtest": {"equity": 10_000, "fee_rate": 0.001, "use_open_execution": True},
            "automl": {
                "enabled": True,
                "n_trials": 8,
                "objective": "composite",
                "seed": 42,
                "min_trades": 10,
                "study_name": "BTCUSDT_1h_composite_oos_demo",
            },
        }
    )

    print(f"\n{sep}\nStep 1 · Fetching data and indicators\n{sep}")
    pipeline.fetch_data()
    pipeline.run_indicators()

    print(f"\n{sep}\nStep 2 · AutoML search\n{sep}")
    automl = pipeline.run_automl()
    print(f"  study name : {automl['study_name']}")
    print(f"  objective  : {automl['objective']}")
    print(f"  trials     : {automl['trial_count']}")
    print(f"  best value : {automl['best_value']:.4f}")
    print(f"  best params: {automl['best_params']}")
    print(f"  best bt    : {automl['best_backtest']}")

    print(f"\n{sep}\nStep 3 · Rebuild pipeline with best config\n{sep}")
    features = pipeline.build_features()
    stationarity = pipeline.check_stationarity()
    pipeline.detect_regimes()
    pipeline.build_labels()
    pipeline.align_data()
    pipeline.select_features()
    pipeline.compute_sample_weights()
    training = pipeline.train_models()
    signals = pipeline.generate_signals()
    backtest = pipeline.run_backtest()

    screening = stationarity["feature_screening"]["summary"]
    print(f"  feature count: {features.shape[1]}")
    print(
        f"  screened     : {screening['screened_feature_count']}/{screening['total_features']}  "
        f"transformed={screening['transformed_features']}  dropped={screening['dropped_features']}"
    )
    if screening["transform_usage"]:
        print(f"  transforms   : {screening['transform_usage']}")
    print(f"  avg selected : {training['feature_selection']['avg_selected_features']}")
    print(f"  avg accuracy : {training['avg_accuracy']:.4f}")
    print(f"  avg f1       : {training['avg_f1_macro']:.4f}")
    block_diag = training["feature_block_diagnostics"]
    if block_diag["summary"]:
        print("  top blocks   :")
        for block in block_diag["summary"][:5]:
            print(
                f"    {block['block']}: f1_drop={block['avg_f1_drop']:.4f}  "
                f"acc_drop={block['avg_accuracy_drop']:.4f}  "
                f"native={block['avg_native_importance']:.4f}"
            )
    print(f"  long signals : {int((signals['signals'] == 1).sum())}")
    print(f"  short signals: {int((signals['signals'] == -1).sum())}")
    print(f"  net profit   : ${backtest['net_profit']:,.2f}")
    print(f"  sharpe ratio : {backtest['sharpe_ratio']}")
    print(f"  max drawdown : {backtest['max_drawdown']:.2%}")
    print(f"  start equity : ${backtest['starting_equity']:,.2f}")
    print(f"  end equity   : ${backtest['ending_equity']:,.2f}")
    print(f"  net profit   : ${backtest['net_profit']:,.2f} ({backtest['net_profit_pct']:.2%})")
    print(f"  gross profit : ${backtest['gross_profit']:,.2f}")
    print(f"  gross loss   : ${backtest['gross_loss']:,.2f}")
    print(f"  fees paid    : ${backtest['fees_paid']:,.2f}")
    print(f"  sharpe ratio : {backtest['sharpe_ratio']}")
    print(f"  sortino      : {backtest['sortino_ratio']}")
    print(f"  calmar       : {backtest['calmar_ratio']}")
    print(f"  CAGR         : {backtest['cagr']:.2%}")
    print(f"  volatility   : {backtest['annualized_volatility']:.2%}")
    print(f"  max drawdown : {backtest['max_drawdown']:.2%} (${backtest['max_drawdown_amount']:,.2f})")
    print(f"  dd duration  : {backtest['max_drawdown_duration_bars']} bars ({backtest['max_drawdown_duration']})")
    print(f"  exposure     : {backtest['exposure_rate']:.2%}")
    print(f"  profit factor: {backtest['profit_factor']}")
    print(f"  expectancy   : ${backtest['expectancy']:,.2f} per active bar")
    print(f"  avg win      : ${backtest['avg_win']:,.2f}")
    print(f"  avg loss     : ${backtest['avg_loss']:,.2f}")
    print(f"  trades       : {backtest['total_trades']}")
    print(f"  win rate     : {backtest['win_rate']:.2%} (active bars)")


if __name__ == "__main__":
    main()