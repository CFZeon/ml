"""Spot example with a point-in-time safe custom data join.

Usage
-----
    python example_custom_data.py
    python example_custom_data.py --local-certification
"""

import numpy as np
import pandas as pd

from core import ATR, BollingerBands, RSI, ResearchPipeline, fetch_binance_bars
from core.execution import NAUTILUS_AVAILABLE
from example_utils import (
    build_custom_data_entry,
    build_spot_research_config,
    clone_config_with_overrides,
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
    args = parse_local_certification_args("Run the custom-data spot example.")
    sep = "=" * 60
    symbol = "BTCUSDT"
    interval = "1h"
    start = "2024-01-01"
    end = "2024-05-01"
    context_symbols = ["ETHUSDT", "BNBUSDT"]

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
        },
    )
    try:
        config = prepare_example_runtime_config(
            config,
            market="spot",
            local_certification=args.local_certification,
            nautilus_available=NAUTILUS_AVAILABLE,
            example_name="example_custom_data.py",
        )
    except RuntimeError as exc:
        print(str(exc))
        raise SystemExit(2) from exc

    pipeline = ResearchPipeline(config)

    print_section(sep, 2, "Fetching market data and joining custom feed")
    data = pipeline.fetch_data()
    example_runtime = dict(config.get("example_runtime") or {})
    custom_report = pipeline.state["custom_data_report"][0]
    exogenous_columns = [column for column in data.columns if column.startswith("exo_")]
    print(f"  rows          : {len(data)}")
    if example_runtime:
        print(f"  runtime mode  : {example_runtime.get('mode')}")
        print(f"  runtime note  : {example_runtime.get('note')}")
    print(f"  joined cols   : {custom_report['joined_columns']}")
    print(f"  join coverage : {custom_report['coverage']:.2%}")
    print(f"  exact matches : {custom_report['exact_match_count']}")
    print(f"  exact allowed : {custom_report['allow_exact_matches']}")
    print(f"  stale hits    : {custom_report['stale_hit_count']}")
    print(f"  median age    : {custom_report['median_feature_age']}")
    print(f"  max age       : {custom_report['max_feature_age_observed']}")
    print(f"  raw exogenous : {exogenous_columns}")

    print_section(sep, 3, "Running indicators")
    indicator_run = pipeline.run_indicators()
    print(f"  indicators   : {[result.kind for result in indicator_run.results]}")

    print_section(sep, 4, "Building features and screening stationarity")
    features = pipeline.build_features()
    family_summary = pipeline.state.get("feature_family_summary", {})
    print(f"  feature count : {features.shape[1]}")
    print(f"  families      : {family_summary.get('selected_family_counts', {})}")
    feature_columns = [column for column in features.columns if column.startswith("exo_")]
    print(f"  exogenous fts : {feature_columns[:10]}")
    stationarity = pipeline.check_stationarity()
    print_stationarity_summary(stationarity)

    print_section(sep, 5, "Previewing regime features")
    regimes = pipeline.detect_regimes()["regimes"]
    print_regime_summary(regimes)

    print_section(sep, 6, "Building labels and aligning research matrix")
    labels = pipeline.build_labels()
    aligned = pipeline.align_data()
    print_label_summary(labels)
    print_alignment_summary(aligned)

    print_section(sep, 7, "Previewing feature-selection and weighting")
    selection = pipeline.select_features()
    print_feature_selection_summary(selection)
    weights = pipeline.compute_sample_weights()
    print_weight_summary(weights)

    print_section(sep, 8, "CPCV training")
    training = pipeline.train_models()
    print_training_summary(training)

    print_section(sep, 9, "Generating signals")
    signals = pipeline.generate_signals()
    print_signal_summary(signals, allow_short=False)

    print_section(sep, 10, "Backtesting")
    backtest = pipeline.run_backtest()
    print_backtest_summary(backtest)

    print(f"\n{sep}\nCustom data example complete.\n{sep}")


if __name__ == "__main__":
    main()