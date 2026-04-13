"""Spot example with a point-in-time safe custom data join.

Usage
-----
    python example_custom_data.py
"""

import numpy as np
import pandas as pd

from core import ATR, BollingerBands, RSI, ResearchPipeline, fetch_binance_bars


def build_custom_feed(symbol, interval, start, end):
    bars = fetch_binance_bars(symbol=symbol, interval=interval, start=start, end=end, market="spot")
    custom = pd.DataFrame(
        {
            "timestamp": bars.index,
            "available_at": bars.index + pd.Timedelta(hours=1),
            "realized_vol_24h": bars["close"].pct_change().rolling(24).std(),
            "trend_12h": bars["close"].pct_change(12),
            "volume_zscore_24h": (
                (bars["volume"] - bars["volume"].rolling(24).mean())
                / bars["volume"].rolling(24).std()
            ),
            "session_bias": np.sin(np.arange(len(bars)) / 12.0),
        }
    ).dropna()
    return bars, custom.reset_index(drop=True)


def print_backtest_summary(backtest):
    print(f"  engine       : {backtest['engine']}")
    print(f"  end equity   : ${backtest['ending_equity']:,.2f}")
    print(f"  net profit   : ${backtest['net_profit']:,.2f} ({backtest['net_profit_pct']:.2%})")
    print(f"  sharpe ratio : {backtest['sharpe_ratio']}")
    print(f"  max drawdown : {backtest['max_drawdown']:.2%}")
    print(f"  trades       : {backtest['total_trades']}")


def main():
    sep = "=" * 60
    symbol = "BTCUSDT"
    interval = "1h"
    start = "2024-01-01"
    end = "2024-04-01"

    print(f"\n{sep}\nStep 1 - Building delayed custom data feed\n{sep}")
    base_bars, custom_feed = build_custom_feed(symbol=symbol, interval=interval, start=start, end=end)
    print(f"  source rows   : {len(base_bars)}")
    print(f"  custom rows   : {len(custom_feed)}")
    print(f"  feed columns  : {[column for column in custom_feed.columns if column not in {'timestamp', 'available_at'}]}")

    pipeline = ResearchPipeline(
        {
            "data": {
                "symbol": symbol,
                "interval": interval,
                "start": start,
                "end": end,
                "market": "spot",
                "futures_context": {"enabled": False},
                "cross_asset_context": {"symbols": ["ETHUSDT", "BNBUSDT"]},
                "custom_data": [
                    {
                        "name": "delayed_market_microstructure",
                        "frame": custom_feed,
                        "timestamp_column": "timestamp",
                        "availability_column": "available_at",
                        "prefix": "exo",
                    }
                ],
            },
            "indicators": [RSI(14), BollingerBands(20), ATR(14)],
            "features": {
                "lags": [1, 3, 6],
                "frac_diff_d": 0.4,
                "rolling_window": 24,
                "context_timeframes": ["4h"],
            },
            "feature_selection": {"enabled": True, "max_features": 72, "min_mi_threshold": 0.0},
            "regime": {"n_regimes": 2},
            "labels": {"kind": "fixed_horizon", "horizon": 12, "threshold": 0.0005},
            "model": {
                "type": "gbm",
                "n_splits": 3,
                "gap": 12,
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
                "fee_rate": 0.001,
                "slippage_rate": 0.0003,
                "engine": "vectorbt",
                "use_open_execution": True,
                "signal_delay_bars": 1,
            },
        }
    )

    print(f"\n{sep}\nStep 2 - Fetching market data and joining custom feed\n{sep}")
    data = pipeline.fetch_data()
    custom_report = pipeline.state["custom_data_report"][0]
    exogenous_columns = [column for column in data.columns if column.startswith("exo_")]
    print(f"  rows          : {len(data)}")
    print(f"  joined cols   : {custom_report['joined_columns']}")
    print(f"  join coverage : {custom_report['coverage']:.2%}")
    print(f"  raw exogenous : {exogenous_columns}")

    print(f"\n{sep}\nStep 3 - Indicators, features, and labels\n{sep}")
    pipeline.run_indicators()
    features = pipeline.build_features()
    labels = pipeline.build_labels()
    aligned = pipeline.align_data()
    pipeline.select_features()
    weights = pipeline.compute_sample_weights()
    feature_columns = [column for column in features.columns if column.startswith("exo_")]
    print(f"  feature count : {features.shape[1]}")
    print(f"  exogenous fts : {feature_columns[:10]}")
    print(f"  labels        : {labels['label'].value_counts().to_dict()}")
    print(f"  samples       : {len(aligned['X'])}")
    print(f"  weight mean   : {weights.mean():.3f}")

    print(f"\n{sep}\nStep 4 - Training, signals, and backtest\n{sep}")
    training = pipeline.train_models()
    signals = pipeline.generate_signals()
    backtest = pipeline.run_backtest()
    signal_classes = signals["signals"]
    print(f"  avg accuracy  : {training['avg_accuracy']:.4f}")
    print(f"  avg f1        : {training['avg_f1_macro']:.4f}")
    print(f"  avg selected  : {training['feature_selection']['avg_selected_features']}")
    print(
        f"  long={int((signal_classes == 1).sum())}  "
        f"short={int((signal_classes == -1).sum())}  flat={int((signal_classes == 0).sum())}"
    )
    print_backtest_summary(backtest)

    print(f"\n{sep}\nCustom data example complete.\n{sep}")


if __name__ == "__main__":
    main()