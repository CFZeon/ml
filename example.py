"""End-to-end pipeline: fetch → indicators → features → labels → train → backtest.

Usage
-----
    python example.py
"""

from core import RSI, ATR, BollingerBands, MACD, ResearchPipeline


def main():
    SEP = "=" * 60
    pipeline = ResearchPipeline(
        {
            "data": {
                "symbol": "BTCUSDT",
                "interval": "1h",
                "start": "2024-01-01",
                "end": "2024-12-01",
            },
            "indicators": [RSI(14), MACD(), BollingerBands(20), ATR(14)],
            "features": {"lags": [1, 3, 6], "frac_diff_d": 0.4},
            "feature_selection": {"enabled": True, "max_features": 96, "min_mi_threshold": 0.0005},
            "regime": {"n_regimes": 2},
            "labels": {
                "kind": "triple_barrier",
                "pt_sl": (2.0, 2.0),
                "max_holding": 24,
                "min_return": 0.001,
                "volatility_window": 24,
            },
            "model": {"type": "gbm", "n_splits": 3, "gap": 24},
            "signals": {
                "avg_win": 0.02,
                "avg_loss": 0.02,
                "fraction": 0.5,
                "threshold": 0.01,
                "edge_threshold": 0.05,
                "meta_threshold": 0.55,
            },
            "backtest": {"equity": 10_000, "fee_rate": 0.001},
        }
    )

    # ── 1. Fetch data ─────────────────────────────────────────────────────
    print(f"\n{SEP}\nStep 1 · Fetching BTCUSDT 1h from Binance Vision\n{SEP}")
    df = pipeline.fetch_data()
    print(f"  rows={len(df)}  range={df.index[0]} → {df.index[-1]}")

    # ── 2. Indicators ─────────────────────────────────────────────────────
    print(f"\n{SEP}\nStep 2 · Attaching indicators\n{SEP}")
    indicator_run = pipeline.run_indicators()
    df = indicator_run.frame
    print(f"  columns: {list(df.columns)}")

    # ── 3. Features ───────────────────────────────────────────────────────
    print(f"\n{SEP}\nStep 3 · Building features\n{SEP}")
    features = pipeline.build_features()
    print(f"  feature count: {features.shape[1]}")

    stationarity = pipeline.check_stationarity()
    raw_stat = stationarity["close"]
    fd_stat = stationarity["close_fracdiff"]
    screening = stationarity["feature_screening"]["summary"]
    print(f"  close stationary?  {raw_stat['stationary']}  (p={raw_stat['p_value']})")
    print(f"  frac-diff close?   {fd_stat['stationary']}  (p={fd_stat['p_value']})")
    print(
        f"  screened features: {screening['screened_feature_count']}/{screening['total_features']}  "
        f"transformed={screening['transformed_features']}  dropped={screening['dropped_features']}"
    )
    if screening["transform_usage"]:
        print(f"  transforms used : {screening['transform_usage']}")

    # ── 4. Regime detection ───────────────────────────────────────────────
    print(f"\n{SEP}\nStep 4 · Regime detection\n{SEP}")
    regime_result = pipeline.detect_regimes()
    regimes = regime_result["regimes"]
    features = pipeline.state["features"]
    print(f"  regime counts:\n{regimes.value_counts().to_string()}")

    # ── 5. Triple-barrier labels ──────────────────────────────────────────
    print(f"\n{SEP}\nStep 5 · Triple-barrier labeling\n{SEP}")
    labels = pipeline.build_labels()
    print(f"  labels: {len(labels)}")
    print(f"  distribution:\n{labels['label'].value_counts().to_string()}")
    print(f"  barriers:\n{labels['barrier'].value_counts().to_string()}")

    # ── 6. Align & clean ─────────────────────────────────────────────────
    print(f"\n{SEP}\nStep 6 · Aligning features ↔ labels\n{SEP}")
    aligned = pipeline.align_data()
    X = aligned["X"]
    y = aligned["y"]
    labels_aligned = aligned["labels_aligned"]
    print(f"  samples={len(X)}  features={X.shape[1]}")

    # ── 6b. Feature selection ─────────────────────────────────────────────
    print(f"\n{SEP}\nStep 6b · Feature selection (mutual information)\n{SEP}")
    selection = pipeline.select_features()
    sel_report = selection.report
    X = pipeline.state["X"]
    print("  global preselection disabled; supervised MI filtering runs inside each walk-forward fold")
    print(
        f"  configured cap: {sel_report['max_features'] or 'auto'}  "
        f"min_mi={sel_report['min_mi_threshold']}"
    )

    # ── 7. Sample weights ─────────────────────────────────────────────────
    print(f"\n{SEP}\nStep 7 · Sample weights by uniqueness\n{SEP}")
    weights = pipeline.compute_sample_weights()
    print(f"  range=[{weights.min():.3f}, {weights.max():.3f}]  mean={weights.mean():.3f}")

    # ── 8. Walk-forward training ──────────────────────────────────────────
    print(f"\n{SEP}\nStep 8 · Walk-forward training (3 folds)\n{SEP}")
    training = pipeline.train_models()
    for metric in training["fold_metrics"]:
        print(f"  fold {metric['fold']}: acc={metric['accuracy']}  f1={metric['f1_macro']}")
    print(f"  avg  acc={training['avg_accuracy']:.4f}  f1={training['avg_f1_macro']:.4f}")
    print(f"  avg selected : {training['feature_selection']['avg_selected_features']}")

    block_diag = training["feature_block_diagnostics"]
    if block_diag["summary"]:
        print("  top feature blocks:")
        for block in block_diag["summary"][:5]:
            print(
                f"    {block['block']}: f1_drop={block['avg_f1_drop']:.4f}  "
                f"acc_drop={block['avg_accuracy_drop']:.4f}  "
                f"native={block['avg_native_importance']:.4f}"
            )

    # ── 9. Signals + Kelly sizing ─────────────────────────────────────────
    print(f"\n{SEP}\nStep 9 · Generating signals with Kelly sizing\n{SEP}")
    signal_result = pipeline.generate_signals()
    sig_cat = signal_result["signals"]
    print(f"  long={int((sig_cat == 1).sum())}  short={int((sig_cat == -1).sum())}  "
          f"flat={int((sig_cat == 0).sum())}")

    # ── 10. Backtest ──────────────────────────────────────────────────────
    print(f"\n{SEP}\nStep 10 · Backtest\n{SEP}")
    bt = pipeline.run_backtest()
    print(f"  start equity : ${bt['starting_equity']:,.2f}")
    print(f"  end equity   : ${bt['ending_equity']:,.2f}")
    print(f"  net profit   : ${bt['net_profit']:,.2f} ({bt['net_profit_pct']:.2%})")
    print(f"  gross profit : ${bt['gross_profit']:,.2f}")
    print(f"  gross loss   : ${bt['gross_loss']:,.2f}")
    print(f"  fees paid    : ${bt['fees_paid']:,.2f}")
    print(f"  sharpe ratio : {bt['sharpe_ratio']}")
    print(f"  sortino      : {bt['sortino_ratio']}")
    print(f"  calmar       : {bt['calmar_ratio']}")
    print(f"  CAGR         : {bt['cagr']:.2%}")
    print(f"  volatility   : {bt['annualized_volatility']:.2%}")
    print(f"  max drawdown : {bt['max_drawdown']:.2%} (${bt['max_drawdown_amount']:,.2f})")
    print(f"  dd duration  : {bt['max_drawdown_duration_bars']} bars ({bt['max_drawdown_duration']})")
    print(f"  exposure     : {bt['exposure_rate']:.2%}")
    print(f"  profit factor: {bt['profit_factor']}")
    print(f"  expectancy   : ${bt['expectancy']:,.2f} per active bar")
    print(f"  avg win      : ${bt['avg_win']:,.2f}")
    print(f"  avg loss     : ${bt['avg_loss']:,.2f}")
    print(f"  trades       : {bt['total_trades']}")
    print(f"  win rate     : {bt['win_rate']:.2%} (active bars)")

    print(f"\n{SEP}\nPipeline complete.\n{SEP}")


if __name__ == "__main__":
    main()
