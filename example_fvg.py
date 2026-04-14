"""End-to-end pipeline using a config-driven Fair Value Gap indicator setup.

Usage
-----
    python example_fvg.py
"""

import pandas as pd

from core import (
    ResearchPipeline,
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
                "end": "2024-12-01",
            },
            "indicators": [
                {"kind": "rsi", "params": {"period": 14}},
                {"kind": "atr", "params": {"period": 14}},
                {"kind": "fvg", "params": {"name": "fvg_main", "min_gap_pct": 0.0005}},
            ],
            "features": {
                "lags": [1, 3, 6],
                "frac_diff_d": 0.4,
            },
            "feature_selection": {"enabled": True, "max_features": 96, "min_mi_threshold": 0.0005},
            "regime": {"n_regimes": 2, "builder": build_fvg_regime_features},
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

    print(f"\n{sep}\nStep 1 · Fetching BTCUSDT 1h from Binance Vision\n{sep}")
    df = pipeline.fetch_data()
    print(f"  rows={len(df)}  range={df.index[0]} → {df.index[-1]}")

    print(f"\n{sep}\nStep 2 · Running config-driven indicators\n{sep}")
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

    print(f"\n{sep}\nStep 3 · Building features from FVG outputs\n{sep}")
    features = pipeline.build_features()
    print(f"  feature count: {features.shape[1]}")

    stationarity = pipeline.check_stationarity()
    print(f"  stationarity : {stationarity['mode']}")
    print(f"  note         : {stationarity['note']}")

    print(f"\n{sep}\nStep 4 · Regime detection\n{sep}")
    regime_result = pipeline.detect_regimes()
    print(f"  regimes      : {regime_result['mode']}")
    print(f"  note         : {pipeline.state['regime_detection']['note']}")

    print(f"\n{sep}\nStep 5 · Triple-barrier labeling\n{sep}")
    labels = pipeline.build_labels()
    print(f"  labels: {len(labels)}")
    print(f"  distribution:\n{labels['label'].value_counts().to_string()}")

    print(f"\n{sep}\nStep 6 · Aligning features and labels\n{sep}")
    aligned = pipeline.align_data()
    X = aligned["X"]
    labels_aligned = aligned["labels_aligned"]
    print(f"  samples={len(X)}  features={X.shape[1]}")

    print(f"\n{sep}\nStep 6b · Feature selection (mutual information)\n{sep}")
    selection = pipeline.select_features()
    sel_report = selection.report
    X = pipeline.state["X"]
    print("  global preselection disabled; supervised MI filtering runs inside each walk-forward fold")
    print(
        f"  configured cap: {sel_report['max_features'] or 'auto'}  "
        f"min_mi={sel_report['min_mi_threshold']}"
    )

    print(f"\n{sep}\nStep 7 · Uniqueness weights\n{sep}")
    weights = pipeline.compute_sample_weights()
    print(f"  range=[{weights.min():.3f}, {weights.max():.3f}]  mean={weights.mean():.3f}")

    print(f"\n{sep}\nStep 8 · Walk-forward training\n{sep}")
    training = pipeline.train_models()
    for metric in training["fold_metrics"]:
        print(f"  fold {metric['fold']}: acc={metric['accuracy']}  f1={metric['f1_macro']}")
    print(f"  avg acc={training['avg_accuracy']:.4f}  f1={training['avg_f1_macro']:.4f}")
    print(f"  avg selected : {training['feature_selection']['avg_selected_features']}")
    print(f"  stationarity : {training['stationarity']['mode']}")
    print(f"  regime mode  : {training['regime']['mode']}")

    block_diag = training["feature_block_diagnostics"]
    if block_diag["summary"]:
        print("  top feature blocks:")
        for block in block_diag["summary"][:5]:
            print(
                f"    {block['block']}: f1_drop={block['avg_f1_drop']:.4f}  "
                f"acc_drop={block['avg_accuracy_drop']:.4f}  "
                f"native={block['avg_native_importance']:.4f}"
            )

    print(f"\n{sep}\nStep 9 · Signals with meta-labeling and Kelly sizing\n{sep}")
    signal_result = pipeline.generate_signals()
    signal_classes = signal_result["signals"]
    print(
        f"  long={int((signal_classes == 1).sum())}  "
        f"short={int((signal_classes == -1).sum())}  "
        f"flat={int((signal_classes == 0).sum())}"
    )

    print(f"\n{sep}\nStep 10 · Backtest\n{sep}")
    backtest = pipeline.run_backtest()
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

    print(f"\n{sep}\nPipeline complete.\n{sep}")


if __name__ == "__main__":
    main()