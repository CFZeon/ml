"""Multi-timeframe FVG + SMA(200) daily futures example with derivative-based derivatives context.

Feature stack
-------------
  Base market
    - BTCUSDT USD-M futures, 15m base bars

  Base indicators (15m)
    - RSI(14), ATR(14)
        - WaveTrend oscillator (10, 21, 4)
    - FVG with 1 % minimum gap width filter

  Optional derivatives indicators (fetched inside indicators, no pipeline changes)
    - Funding-rate first derivative from Binance USD-M futures
        - Open-interest non-level change context from Binance USD-M futures
    - Combined funding/OI derivative interactions

  Multi-timeframe context
    - 1h, 4h, 1d OHLCV-derived statistics

  Custom features
    - Enhanced FVG signal levels
    - Daily SMA(200) computed from futures daily closes

Usage
-----
    python example_mtf_fvg_futures.py
    python example_mtf_fvg_futures.py --local-certification
"""

import pandas as pd

from core import fetch_binance_bars
from core.execution import NAUTILUS_AVAILABLE
from core.indicators._derivatives_binance import fetch_funding_history
from example_entrypoints import parse_example_args, run_example
from example_mtf_fvg import (
    BASE_INTERVAL,
    DERIVATIVES_FLAGS,
    DERIVATIVES_SETTINGS,
    MIN_GAP_PCT,
    SYMBOL,
    SMA_WARMUP_DAYS,
    _fill_sparse_feature_columns,
    build_daily_sma200,
    build_indicator_specs,
    build_mtf_regime_features,
    compute_enhanced_fvg_features,
    format_config_timestamp,
    resolve_example_window,
)
from example_utils import (
    build_custom_data_entry,
    build_futures_research_config,
    print_section,
)


def align_runtime_funding_to_index(funding_frame: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Map funding events onto the nearest prior available bar."""
    if funding_frame is None or funding_frame.empty:
        return pd.DataFrame(columns=["timestamp", "funding_rate", "funding_mark_price"])

    aligned_index = pd.DatetimeIndex(index)
    funding = funding_frame.copy().sort_values("timestamp")
    funding_timestamps = pd.DatetimeIndex(pd.to_datetime(funding["timestamp"], utc=True))
    positions = aligned_index.get_indexer(funding_timestamps, method="pad")
    valid_mask = positions >= 0
    if not valid_mask.any():
        return pd.DataFrame(columns=funding.columns)

    funding = funding.loc[valid_mask].copy()
    funding["timestamp"] = aligned_index.take(positions[valid_mask]).to_numpy()
    funding = funding.drop_duplicates(subset=["timestamp"], keep="last")
    return funding.reset_index(drop=True)


def _attach_runtime_funding_context(pipeline, data: pd.DataFrame) -> pd.DataFrame:
    funding_frame = fetch_funding_history(
        SYMBOL,
        data.index.min(),
        data.index.max() + pd.Timedelta(milliseconds=1),
    )
    funding_frame = align_runtime_funding_to_index(funding_frame, data.index)
    pipeline.state["futures_context"] = {
        **dict(pipeline.state.get("futures_context") or {}),
        "funding": funding_frame.set_index("timestamp").sort_index(),
    }
    return data


def main():
    args = parse_example_args(
        "Multi-timeframe FVG + SMA(200) daily futures example (BTCUSDT 15m USD-M futures)."
    )
    sep = "=" * 60
    window = resolve_example_window()
    model_start = window["model_start"]
    model_end = window["model_end"]
    if args.quick:
        model_start = max(model_start, model_end - pd.Timedelta(days=10))
    sma_history_start = model_start - pd.Timedelta(days=SMA_WARMUP_DAYS)
    config_model_start = format_config_timestamp(model_start)
    config_model_end = format_config_timestamp(model_end)
    indicators = build_indicator_specs(SYMBOL)

    if window["note"]:
        print(f"{sep}\nDerivatives window mode: {window['window_mode']}\n{sep}")
        print(f"  note  : {window['note']}")
        print(f"  start : {model_start}")
        print(f"  end   : {model_end}")

    print_section(sep, 1, f"Pre-fetching {SYMBOL} futures data for custom feature computation")
    base_bars = fetch_binance_bars(
        symbol=SYMBOL,
        interval=BASE_INTERVAL,
        start=model_start,
        end=model_end,
        market="um_futures",
    )
    print(f"  {BASE_INTERVAL} futures bars : {len(base_bars)}")
    print(f"  range               : {base_bars.index[0]} ... {base_bars.index[-1]}")

    daily_bars = fetch_binance_bars(
        symbol=SYMBOL,
        interval="1d",
        start=sma_history_start,
        end=model_end,
        market="um_futures",
    )
    print(f"  1d futures bars     : {len(daily_bars)}")
    print(f"  range               : {daily_bars.index[0]} ... {daily_bars.index[-1]}")

    print_section(sep, 2, "Computing enhanced FVG signal levels (min_gap_pct=1 %)")
    efvg_df = compute_enhanced_fvg_features(base_bars, min_gap_pct=MIN_GAP_PCT)
    print(f"  bars with active bull sell level : {int((efvg_df['efvg_bull_active_count'] > 0).sum())}")
    print(f"  bars with active bear buy  level : {int((efvg_df['efvg_bear_active_count'] > 0).sum())}")

    print_section(sep, 3, "Computing daily SMA(200) from futures history")
    sma200_df = build_daily_sma200(daily_bars)
    print(f"  valid SMA(200) readings       : {int(sma200_df['sma200_daily'].notna().sum())}")
    if sma200_df.empty:
        raise RuntimeError("No valid SMA(200) values produced for futures example.")

    print_section(sep, 4, "Assembling custom-data entries")
    efvg_entry = build_custom_data_entry(
        "enhanced_fvg",
        efvg_df,
        value_columns=[
            "efvg_bull_sell_level",
            "efvg_bull_sell_dist_pct",
            "efvg_bull_sell_touch",
            "efvg_bull_active_count",
            "efvg_bear_buy_level",
            "efvg_bear_buy_dist_pct",
            "efvg_bear_buy_touch",
            "efvg_bear_active_count",
        ],
        allow_exact_matches=True,
    )
    sma200_entry = build_custom_data_entry(
        "sma200_daily",
        sma200_df,
        value_columns=["sma200_daily", "price_vs_sma200_daily", "sma200_slope_5d"],
        allow_exact_matches=False,
    )
    print(f"  custom entries: enhanced_fvg ({len(efvg_df)} rows), sma200_daily ({len(sma200_df)} rows)")

    print_section(sep, 5, "Building futures pipeline config")
    config = build_futures_research_config(
        SYMBOL,
        BASE_INTERVAL,
        config_model_start,
        config_model_end,
        indicators=indicators,
        context_symbols=[],
        config_overrides={
            "data": {
                "futures_context": {"enabled": False},
                "cross_asset_context": {"symbols": [], "market": "um_futures"},
            },
            "features": {
                "context_timeframes": ["1h", "4h", "1d"],
                "lags": [1, 3, 6],
                "frac_diff_d": 0.4,
                "rolling_window": 20,
                "squeeze_quantile": 0.2,
                "futures_context_ttl": {
                    "mark_price": "2h",
                    "premium_index": "2h",
                    "funding": "12h",
                    "recent": "4h",
                    "max_stale_rate": 1.0,
                    "max_unknown_rate": 1.0,
                },
                "cross_asset_context_ttl": {"max_age": "2h", "max_unknown_rate": 1.0},
            },
            "regime": {"method": "hmm", "builder": build_mtf_regime_features},
            "labels": {
                "kind": "triple_barrier",
                "pt_sl": (1.5, 1.5),
                "max_holding": 16,
                "min_return": 0.0005,
                "volatility_window": 48,
                "barrier_tie_break": "conservative",
            },
            "model": {
                "type": "gbm",
                "cv_method": "cpcv",
                "n_blocks": 4,
                "test_blocks": 2,
                "validation_fraction": 0.2,
                "meta_n_splits": 2,
            },
            "evaluation_mode": "research_only",
            "signals": {
                "policy_mode": "frozen_manual",
                "profitability_threshold": -1.0,
                "min_position_size": 0.1,
                "avg_win": 0.015,
                "avg_loss": 0.01,
                "shrinkage_alpha": 0.5,
                "fraction": 0.5,
                "min_trades_for_kelly": 30,
                "max_kelly_fraction": 0.5,
                "threshold": 0.0,
                "edge_threshold": 0.0,
                "meta_threshold": 0.0,
                "expected_edge_threshold": -1.0,
                "sizing_mode": "expected_utility",
                "tuning_min_trades": 5,
            },
            "backtest": {
                "equity": 10_000,
                "fee_rate": 0.0004,
                "slippage_rate": 0.0002,
                "slippage_model": "sqrt_impact",
                "engine": "pandas",
                "funding_missing_policy": {
                    "mode": "zero_fill_debug",
                    "expected_interval": "8h",
                    "max_gap_multiplier": 1.5,
                },
                "use_open_execution": True,
                "signal_delay_bars": 2,
            },
        },
    )
    config["data"]["custom_data"] = [efvg_entry, sma200_entry]
    config["experiment"] = {
        "name": "mtf_fvg_futures",
        "description": "Multi-timeframe FVG + SMA(200) daily futures example.",
    }
    config["derivatives"] = {
        **DERIVATIVES_FLAGS,
        "window_mode": window["window_mode"],
        "recent_window_days": DERIVATIVES_SETTINGS["recent_window_days"],
    }

    run_example(
        config,
        market="um_futures",
        local_certification=args.local_certification,
        quiet=args.quiet,
        nautilus_available=NAUTILUS_AVAILABLE,
        example_name="example_mtf_fvg_futures.py",
        hooks={
            "after_fetch_data": _attach_runtime_funding_context,
            "after_build_features": _fill_sparse_feature_columns,
        },
    )


if __name__ == "__main__":
    main()