"""End-to-end pipeline using a config-driven Fair Value Gap indicator setup.

Usage
-----
    python example_fvg.py
"""

import numpy as np
import pandas as pd

from core import (
    ResearchPipeline,
)


def add_fvg_features(pipeline, features):
    data = pipeline.require("data")
    features = features.copy()
    features["fvg_gap_imbalance"] = (
        data["fvg_main_bull_active_count"].fillna(0)
        - data["fvg_main_bear_active_count"].fillna(0)
    )
    features["fvg_fill_imbalance"] = (
        data["fvg_main_bull_fill_state"].fillna(0)
        - data["fvg_main_bear_fill_state"].fillna(0)
    )
    features["fvg_distance_spread"] = (
        data["fvg_main_bull_distance_pct"].fillna(0)
        - data["fvg_main_bear_distance_pct"].fillna(0)
    )
    return features


def build_fvg_regime_features(pipeline):
    data = pipeline.require("data")
    features = pipeline.require("features")
    return pd.DataFrame(
        {
            "vol_20": data["close"].pct_change().rolling(20).std(),
            "fvg_gap_imbalance": features["fvg_gap_imbalance"],
            "fvg_distance_spread": features["fvg_distance_spread"],
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
                "end": "2024-04-01",
            },
            "indicators": [
                {"kind": "rsi", "params": {"period": 14}},
                {"kind": "atr", "params": {"period": 14}},
                {"kind": "fvg", "params": {"name": "fvg_main", "min_gap_pct": 0.0005}},
            ],
            "features": {
                "lags": [1, 3, 6],
                "frac_diff_d": 0.4,
                "builders": [add_fvg_features],
            },
            "regime": {"n_regimes": 2, "builder": build_fvg_regime_features},
            "labels": {
                "kind": "triple_barrier",
                "pt_sl": (2.0, 2.0),
                "max_holding": 24,
                "min_return": 0.001,
                "volatility_window": 24,
            },
            "model": {"type": "rf", "n_splits": 3, "gap": 24},
            "signals": {"avg_win": 0.02, "avg_loss": 0.02, "fraction": 0.5, "threshold": 0.05},
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
    raw_stat = stationarity["close"]
    fd_stat = stationarity["close_fracdiff"]
    print(f"  close stationary?  {raw_stat['stationary']}  (p={raw_stat['p_value']})")
    print(f"  frac-diff close?   {fd_stat['stationary']}  (p={fd_stat['p_value']})")

    print(f"\n{sep}\nStep 4 · Regime detection\n{sep}")
    regime_result = pipeline.detect_regimes()
    regimes = regime_result["regimes"]
    features = pipeline.state["features"]
    print(f"  regime counts:\n{regimes.value_counts().to_string()}")

    print(f"\n{sep}\nStep 5 · Triple-barrier labeling\n{sep}")
    labels = pipeline.build_labels()
    print(f"  labels: {len(labels)}")
    print(f"  distribution:\n{labels['label'].value_counts().to_string()}")

    print(f"\n{sep}\nStep 6 · Aligning features and labels\n{sep}")
    aligned = pipeline.align_data()
    X = aligned["X"]
    labels_aligned = aligned["labels_aligned"]
    print(f"  samples={len(X)}  features={X.shape[1]}")

    print(f"\n{sep}\nStep 7 · Uniqueness weights\n{sep}")
    weights = pipeline.compute_sample_weights()
    print(f"  range=[{weights.min():.3f}, {weights.max():.3f}]  mean={weights.mean():.3f}")

    print(f"\n{sep}\nStep 8 · Walk-forward training\n{sep}")
    training = pipeline.train_models()
    for metric in training["fold_metrics"]:
        print(f"  fold {metric['fold']}: acc={metric['accuracy']}  f1={metric['f1_macro']}")
    print(f"  avg acc={training['avg_accuracy']:.4f}  f1={training['avg_f1_macro']:.4f}")

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
    print(f"  total return : {backtest['total_return']:.2%}")
    print(f"  sharpe ratio : {backtest['sharpe_ratio']}")
    print(f"  max drawdown : {backtest['max_drawdown']:.2%}")
    print(f"  trades       : {backtest['total_trades']}")
    print(f"  win rate     : {backtest['win_rate']:.2%}")

    print(f"\n{sep}\nPipeline complete.\n{sep}")


if __name__ == "__main__":
    main()