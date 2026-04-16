"""Offline synthetic example covering derivatives context and futures-style backtesting.

Usage
-----
    python example_synthetic_derivatives.py
"""

import numpy as np
import pandas as pd

from core import ATR, RSI, ResearchPipeline
from example_utils import (
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


def make_ohlcv(index, drift=18.0, amplitude=2.5, volume_base=1_500.0):
    steps = np.linspace(0.0, 1.0, len(index))
    cycle = np.sin(np.linspace(0.0, 12.0 * np.pi, len(index)))
    shock = 0.8 * np.sin(np.linspace(0.0, 2.0 * np.pi, len(index)) ** 2)
    close = 100.0 + drift * steps + amplitude * cycle + shock
    open_ = np.roll(close, 1)
    open_[0] = close[0] * 0.998
    high = np.maximum(open_, close) * 1.004
    low = np.minimum(open_, close) * 0.996
    volume = volume_base + 150.0 * (1.0 + np.cos(np.linspace(0.0, 8.0 * np.pi, len(index))))
    quote_volume = close * volume
    trades = 200 + (30 * (1.0 + np.sin(np.linspace(0.0, 7.0 * np.pi, len(index))))).astype(int)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": quote_volume,
            "trades": trades,
        },
        index=index,
    )


def make_futures_context(index, spot_close):
    mark_close = spot_close * (1.0 + 0.0007 * np.sin(np.linspace(0.0, 8.0 * np.pi, len(index))))
    mark_open = np.roll(mark_close, 1)
    mark_open[0] = mark_close[0]
    premium_close = mark_close - spot_close
    premium_open = np.roll(premium_close, 1)
    premium_open[0] = premium_close[0]

    return {
        "mark_price": pd.DataFrame(
            {
                "mark_open": mark_open,
                "mark_high": np.maximum(mark_open, mark_close) * 1.001,
                "mark_low": np.minimum(mark_open, mark_close) * 0.999,
                "mark_close": mark_close,
            },
            index=index,
        ),
        "premium_index": pd.DataFrame(
            {
                "premium_open": premium_open,
                "premium_high": premium_close + 0.0001,
                "premium_low": premium_close - 0.0001,
                "premium_close": premium_close,
            },
            index=index,
        ),
        "funding": pd.DataFrame(
            {
                "funding_rate": 0.00008 * np.sin(np.linspace(0.0, 4.0 * np.pi, len(index[::8]))),
                "funding_mark_price": spot_close[::8],
            },
            index=index[::8],
        ),
        "open_interest": pd.DataFrame(
            {
                "sumOpenInterest": 50_000 + np.linspace(0.0, 6_000.0, len(index)),
                "sumOpenInterestValue": 5_000_000 + np.linspace(0.0, 800_000.0, len(index)),
            },
            index=index,
        ),
        "taker_flow": pd.DataFrame(
            {
                "buySellRatio": 1.0 + 0.15 * np.sin(np.linspace(0.0, 7.0 * np.pi, len(index))),
                "buyVol": 3_000 + 250 * (1.0 + np.cos(np.linspace(0.0, 6.0 * np.pi, len(index)))),
                "sellVol": 2_800 + 200 * (1.0 + np.sin(np.linspace(0.0, 5.0 * np.pi, len(index)))),
            },
            index=index,
        ),
        "global_long_short": pd.DataFrame(
            {
                "longShortRatio": 1.05 + 0.06 * np.sin(np.linspace(0.0, 5.0 * np.pi, len(index))),
                "longAccount": 0.52 + 0.02 * np.sin(np.linspace(0.0, 3.0 * np.pi, len(index))),
                "shortAccount": 0.48 - 0.02 * np.sin(np.linspace(0.0, 3.0 * np.pi, len(index))),
            },
            index=index,
        ),
        "basis": pd.DataFrame(
            {
                "basisRate": 0.0003 * np.sin(np.linspace(0.0, 2.0 * np.pi, len(index))),
                "basis": 12.0 + 3.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, len(index))),
                "futuresPrice": mark_close * 1.0004,
                "indexPrice": spot_close,
            },
            index=index,
        ),
    }


def main():
    sep = "=" * 60
    index = pd.date_range("2025-01-01", periods=720, freq="1h", tz="UTC")
    raw_data = make_ohlcv(index)
    eth_data = make_ohlcv(index, drift=14.0, amplitude=2.0, volume_base=1_200.0)
    sol_data = make_ohlcv(index, drift=22.0, amplitude=4.0, volume_base=900.0)

    pipeline = ResearchPipeline(
        {
            "data": {"symbol": "BTCUSDT", "interval": "1h", "market": "um_futures"},
            "indicators": [RSI(14), ATR(14)],
            "features": {
                "lags": [1, 3, 6],
                "frac_diff_d": 0.4,
                "rolling_window": 24,
                "context_timeframes": ["4h"],
            },
            "feature_selection": {"enabled": True, "max_features": 48, "min_mi_threshold": 0.0},
            "regime": {"method": "hmm"},  # HMM with stable norm-sorted state ordering
            "labels": {
                "kind": "trend_scanning",
                "min_horizon": 6,
                "max_horizon": 24,
                "step": 3,
                "min_t_value": 0.5,
                "min_return": 0.0001,
            },
            "model": {
                "type": "logistic",
                "cv_method": "cpcv",
                "n_blocks": 4,
                "test_blocks": 2,
                "validation_fraction": 0.2,
                "meta_n_splits": 2,
            },
            "signals": {
                "threshold": 0.0,
                "edge_threshold": 0.0,
                "fraction": 1.0,
                "meta_threshold": 0.5,
                "expected_edge_threshold": 0.0,
                "sizing_mode": "expected_utility",
            },
            "backtest": {
                "equity": 10_000,
                "fee_rate": 0.0005,
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

    pipeline.state["raw_data"] = raw_data
    pipeline.state["data"] = raw_data.copy()
    pipeline.state["futures_context"] = make_futures_context(index, raw_data["close"].to_numpy())
    pipeline.state["cross_asset_context"] = {"ETHUSDT": eth_data, "SOLUSDT": sol_data}
    pipeline.state["symbol_filters"] = {"tick_size": 0.1, "step_size": 0.001, "min_notional": 10.0}

    print_section(sep, 1, "Loading offline synthetic market state")
    print(f"  bars         : {len(raw_data)}")
    print(f"  range        : {raw_data.index[0]} -> {raw_data.index[-1]}")
    print("  network fetch: skipped")

    print_section(sep, 2, "Running indicators")
    indicator_run = pipeline.run_indicators()
    print(f"  indicators   : {[result.kind for result in indicator_run.results]}")

    print_section(sep, 3, "Building features and screening stationarity")
    features = pipeline.build_features()
    print(f"  feature count: {features.shape[1]}")
    print(f"  has fut block: {'fut_funding_rate' in features.columns}")
    print(f"  has ctx block: {'ctx_ethusdt_ret_1' in features.columns}")
    print(f"  has mtf block: {'mtf_4h_trend' in features.columns}")
    stationarity = pipeline.check_stationarity()
    print_stationarity_summary(stationarity)

    print_section(sep, 4, "Previewing regime features")
    regimes = pipeline.detect_regimes()["regimes"]
    print_regime_summary(regimes)

    print_section(sep, 5, "Building labels and aligning research matrix")
    labels = pipeline.build_labels()
    aligned = pipeline.align_data()
    print_label_summary(labels)
    print_alignment_summary(aligned)

    print_section(sep, 6, "Previewing feature-selection and weighting")
    selection = pipeline.select_features()
    print_feature_selection_summary(selection)
    weights = pipeline.compute_sample_weights()
    print_weight_summary(weights)

    print_section(sep, 7, "CPCV training")
    training = pipeline.train_models()
    print_training_summary(training)

    print_section(sep, 8, "Generating signals")
    signals = pipeline.generate_signals()
    print_signal_summary(signals)

    print_section(sep, 9, "Backtesting")
    backtest = pipeline.run_backtest()
    print_backtest_summary(backtest)

    print(f"\n{sep}\nSynthetic derivatives example complete.\n{sep}")


if __name__ == "__main__":
    main()