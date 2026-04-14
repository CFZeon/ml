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
                "futures_context": {"enabled": True, "include_recent_stats": True},
                "cross_asset_context": {"symbols": ["ETHUSDT", "SOLUSDT", "BNBUSDT"]},
            },
            "indicators": [RSI(14), MACD(), BollingerBands(20), ATR(14)],
            "features": {"lags": [1, 3, 6], "frac_diff_d": 0.4, "context_timeframes": ["4h", "1d"]},
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
            "backtest": {"equity": 10_000, "fee_rate": 0.001, "use_open_execution": True, "signal_delay_bars": 2},
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
    print(f"  stationarity preview: {stationarity.get('mode', 'disabled')}")
    print(f"  note: {stationarity.get('note', 'N/A')}")

    # ── 4. Regime detection ───────────────────────────────────────────────
    print(f"\n{SEP}\nStep 4 · Regime detection\n{SEP}")
    regime_result = pipeline.detect_regimes()
    print(f"  regime preview: {regime_result.get('mode', 'disabled')}")
    print(f"  note: {pipeline.state.get('regime_detection', {}).get('note', 'N/A')}")

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
    print(f"  selection mode: {sel_report.get('mode', 'N/A')}")
    print(f"  note: {sel_report.get('note', 'N/A')}")

    # ── 7. Sample weights ─────────────────────────────────────────────────
    print(f"\n{SEP}\nStep 7 · Sample weights by uniqueness\n{SEP}")
    weights = pipeline.compute_sample_weights()
    print(f"  range=[{weights.min():.3f}, {weights.max():.3f}]  mean={weights.mean():.3f}")

    # ── 8. Walk-forward training ──────────────────────────────────────────
    print(f"\n{SEP}\nStep 8 · Walk-forward training (purging + embargo + rolling Kelly)\n{SEP}")
    training = pipeline.train_models()
    for metric in training["fold_metrics"]:
        print(f"  fold {metric['fold']}: acc={metric['accuracy']}  f1={metric['f1_macro']}")
    print(f"  avg  acc={training['avg_accuracy']:.4f}  f1={training['avg_f1_macro']:.4f}")
    directional_accuracy = [metric.get("directional_accuracy") for metric in training["fold_metrics"] if metric.get("directional_accuracy") is not None]
    if directional_accuracy:
        print(f"  avg dir acc={sum(directional_accuracy) / len(directional_accuracy):.4f}")
    print(f"  OOS avg win : {training.get('oos_avg_win', 0):.4%}")
    print(f"  OOS avg loss: {training.get('oos_avg_loss', 0):.4%}")

    # ── 9. Signals ────────────────────────────────────────────────────────
    print(f"\n{SEP}\nStep 9 · Generating signals (with rolling Kelly sizing)\n{SEP}")
    signal_result = pipeline.generate_signals()
    sig_cat = signal_result["signals"]
    print(f"  long={int((sig_cat == 1).sum())}  short={int((sig_cat == -1).sum())}  "
          f"flat={int((sig_cat == 0).sum())}")
    print(f"  avg win used: {signal_result.get('avg_win_used', 0):.4%}")
    print(f"  avg loss used: {signal_result.get('avg_loss_used', 0):.4%}")

    # ── 10. Backtest ──────────────────────────────────────────────────────
    print(f"\n{SEP}\nStep 10 · Backtest\n{SEP}")
    bt = pipeline.run_backtest()
    print(f"  net profit   : {bt['net_profit_pct']:.2%}")
    print(f"  sharpe ratio : {bt['sharpe_ratio']}")
    print(f"  max drawdown : {bt['max_drawdown']:.2%}")
    print(f"  win rate     : {bt['win_rate']:.2%} (active bars)")
    print(f"  closed trades: {bt['closed_trades']}")
    print(f"  trade win rt : {bt['trade_win_rate']:.2%}")

    print(f"\n{SEP}\nPipeline complete.\n{SEP}")


if __name__ == "__main__":
    main()
