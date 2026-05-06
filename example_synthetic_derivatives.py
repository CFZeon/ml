"""Offline synthetic example covering derivatives context and futures-style backtesting.

Usage
-----
    python example_synthetic_derivatives.py
"""

import numpy as np
import pandas as pd

from core import ATR, RSI, ResearchPipeline
from example_entrypoints import parse_example_args, run_example
from example_utils import (
    seed_offline_pipeline_state,
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
    args = parse_example_args("Run the offline synthetic derivatives example.", include_local_certification=False)
    periods = 240 if args.quick else 720
    index = pd.date_range("2025-01-01", periods=periods, freq="1h", tz="UTC")
    raw_data = make_ohlcv(index)
    eth_data = make_ohlcv(index, drift=14.0, amplitude=2.0, volume_base=1_200.0)
    sol_data = make_ohlcv(index, drift=22.0, amplitude=4.0, volume_base=900.0)

    config = {
        "experiment": {
            "name": "synthetic_derivatives_offline",
            "description": "Offline synthetic derivatives example covering context and futures-style backtesting.",
        },
        "data": {"symbol": "BTCUSDT", "interval": "1h", "market": "um_futures", "start": str(index[0]), "end": str(index[-1])},
        "indicators": [RSI(14), ATR(14)],
        "features": {
            "lags": [1, 3, 6],
            "frac_diff_d": 0.4,
            "rolling_window": 24,
            "context_timeframes": ["4h"],
        },
        "feature_selection": {"enabled": True, "max_features": 48, "min_mi_threshold": 0.0},
        "regime": {"method": "hmm"},
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
            "policy_mode": "frozen_manual",
            "avg_win": 0.04,
            "avg_loss": 0.01,
            "threshold": 0.0,
            "edge_threshold": 0.0,
            "shrinkage_alpha": 0.5,
            "fraction": 0.75,
            "min_trades_for_kelly": 30,
            "max_kelly_fraction": 0.5,
            "meta_threshold": 0.0,
            "profitability_threshold": 0.0,
            "expected_edge_threshold": 0.0,
            "sizing_mode": "expected_utility",
        },
        "backtest": {
            "equity": 10_000,
            "fee_rate": 0.0005,
            "slippage_rate": 0.0002,
            "slippage_model": "sqrt_impact",
            "engine": "vectorbt",
            "valuation_price": "mark",
            "apply_funding": True,
            "allow_short": True,
            "leverage": 1.5,
            "use_open_execution": True,
            "signal_delay_bars": 1,
        },
    }
    pipeline = ResearchPipeline(config)

    seed_offline_pipeline_state(
        pipeline,
        raw_data,
        futures_context=make_futures_context(index, raw_data["close"].to_numpy()),
        cross_asset_context={"ETHUSDT": eth_data, "SOLUSDT": sol_data},
        symbol_filters={"tick_size": 0.1, "step_size": 0.001, "min_notional": 10.0},
    )
    run_example(
        config,
        market="um_futures",
        quiet=args.quiet,
        example_name="example_synthetic_derivatives.py",
        pipeline=pipeline,
    )


if __name__ == "__main__":
    main()