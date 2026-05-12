"""Spot example with a point-in-time safe custom data join.

Usage
-----
    python example_custom_data.py
    python example_custom_data.py --local-certification
"""

import numpy as np
import pandas as pd

from core import ATR, BollingerBands, RSI, fetch_binance_bars
from core.execution import NAUTILUS_AVAILABLE
from example_entrypoints import parse_example_args, run_example
from example_utils import (
    build_custom_data_entry,
    build_spot_research_config,
    clone_config_with_overrides,
    print_section,
)


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


def main():
    args = parse_example_args("Run the custom-data spot example.")
    sep = "=" * 60
    symbol = "BTCUSDT"
    interval = "1h"
    start = "2024-01-01"
    end = "2024-03-01" if args.quick else "2024-05-01"
    context_symbols = [] if args.quick else ["ETHUSDT", "BNBUSDT"]

    print_section(sep, 1, "Building delayed custom data feed")
    base_bars, custom_feed = build_custom_feed(symbol=symbol, interval=interval, start=start, end=end)
    print(f"  source rows   : {len(base_bars)}")
    print(f"  custom rows   : {len(custom_feed)}")
    print(f"  feed columns  : {[column for column in custom_feed.columns if column not in {'timestamp', 'available_at'}]}")

    custom_entry = build_custom_data_entry(
        "delayed_market_microstructure",
        custom_feed,
        timestamp_column="timestamp",
        availability_column="available_at",
        value_columns=[
            "realized_vol_24h",
            "trend_12h",
            "volume_zscore_24h",
            "session_bias",
        ],
        prefix="exo",
        max_feature_age="6h",
    )

    config = build_spot_research_config(
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
        indicators=[RSI(14), BollingerBands(20), ATR(14)],
        context_symbols=context_symbols,
        custom_data=[custom_entry],
    )
    config = clone_config_with_overrides(
        config,
        {
            "experiment": {
                "name": "custom_data_spot",
                "description": "Spot example with a point-in-time safe custom data join.",
            },
            "data": {
                "futures_context": {"enabled": False},
            },
            "features": {
                "rolling_window": 24,
                "context_timeframes": ["4h"],
            },
            "feature_selection": {"max_features": 72, "min_mi_threshold": 0.0},
            "labels": {
                "max_holding": 12,
                "min_return": 0.0005,
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
                "slippage_rate": 0.0003,
                "signal_delay_bars": 1,
            },
            "quick_overrides": {
                "data": {
                    "end": "2024-02-15",
                    "cross_asset_context": {"symbols": []},
                },
                "regime": {"enabled": False},
                "model": {
                    "type": "logistic",
                    "cv_method": "walk_forward",
                    "n_splits": 1,
                    "train_size": 240,
                    "test_size": 48,
                    "gap": 6,
                },
            },
        },
    )
    run_example(
        config,
        market="spot",
        local_certification=args.local_certification,
        quick=args.quick,
        quiet=args.quiet,
        nautilus_available=NAUTILUS_AVAILABLE,
        example_name="example_custom_data.py",
    )


if __name__ == "__main__":
    main()