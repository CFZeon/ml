"""Multi-timeframe FVG + SMA(200) daily spot example."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import fetch_binance_bars
from core.execution import NAUTILUS_AVAILABLE
from core.indicators.derivatives_combined import DerivativesCombined
from core.indicators.funding_rate import FundingRateContext
from core.indicators.open_interest import OpenInterestContext
from example_entrypoints import parse_example_args, run_example
from example_utils import build_custom_data_entry, build_spot_research_config, print_section

SYMBOL = "BTCUSDT"
BASE_INTERVAL = "15m"
BASE_INTERVAL_FLOOR = "15min"
MODEL_START_DEFAULT = pd.Timestamp("2024-01-01", tz="UTC")
MODEL_END_DEFAULT = pd.Timestamp("2024-07-01", tz="UTC")
SMA_WARMUP_DAYS = 730
MIN_GAP_PCT = 0.01

DERIVATIVES_FLAGS = {
    "use_funding_rate": True,
    "use_open_interest": True,
    "use_derivatives_combined": True,
}

DERIVATIVES_SETTINGS = {
    "recent_window_days": 28,
    "funding_window": 9,
    "funding_warmup": "10D",
    "funding_max_age": "8h",
    "funding_zscore_threshold": 2.0,
    "funding_min_coverage": 0.5,
    "oi_period": "5m",
    "oi_change_horizon": "4h",
    "oi_trend_short_span": 12,
    "oi_trend_long_span": 48,
    "oi_warmup": "2D",
    "oi_max_age": "30m",
    "oi_min_coverage": 0.5,
    "combined_volatility_window": 20,
}


def resolve_example_window() -> dict[str, object]:
    if not DERIVATIVES_FLAGS.get("use_open_interest"):
        model_start = MODEL_START_DEFAULT
        model_end = MODEL_END_DEFAULT
        window_mode = "historical_static"
        note = None
    else:
        model_end = pd.Timestamp.now(tz="UTC").floor(BASE_INTERVAL_FLOOR)
        model_start = model_end - pd.Timedelta(days=DERIVATIVES_SETTINGS["recent_window_days"])
        window_mode = "recent_live_derivatives"
        note = (
            "openInterestHist is a recent-history endpoint, so the example uses a "
            f"rolling {DERIVATIVES_SETTINGS['recent_window_days']}-day window when OI is enabled."
        )

    return {
        "model_start": model_start,
        "model_end": model_end,
        "sma_history_start": model_start - pd.Timedelta(days=SMA_WARMUP_DAYS),
        "window_mode": window_mode,
        "note": note,
    }


def format_config_timestamp(timestamp: pd.Timestamp) -> str:
    return timestamp.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")


def build_indicator_specs(symbol: str) -> list[object]:
    indicators: list[object] = [
        {"kind": "rsi", "params": {"period": 14}},
        {"kind": "atr", "params": {"period": 14}},
        {
            "kind": "wavetrend",
            "params": {"channel_length": 10, "average_length": 21, "signal_length": 4},
        },
        {"kind": "fvg", "params": {"name": "fvg_base", "min_gap_pct": MIN_GAP_PCT}},
    ]
    if DERIVATIVES_FLAGS.get("use_funding_rate"):
        indicators.append(
            FundingRateContext(
                symbol=symbol,
                name="funding_ctx",
                rolling_window=DERIVATIVES_SETTINGS["funding_window"],
                warmup=DERIVATIVES_SETTINGS["funding_warmup"],
                max_age=DERIVATIVES_SETTINGS["funding_max_age"],
                zscore_threshold=DERIVATIVES_SETTINGS["funding_zscore_threshold"],
                min_coverage=DERIVATIVES_SETTINGS["funding_min_coverage"],
            )
        )
    if DERIVATIVES_FLAGS.get("use_open_interest"):
        indicators.append(
            OpenInterestContext(
                symbol=symbol,
                name="oi_ctx",
                period=DERIVATIVES_SETTINGS["oi_period"],
                change_horizon=DERIVATIVES_SETTINGS["oi_change_horizon"],
                trend_short_span=DERIVATIVES_SETTINGS["oi_trend_short_span"],
                trend_long_span=DERIVATIVES_SETTINGS["oi_trend_long_span"],
                warmup=DERIVATIVES_SETTINGS["oi_warmup"],
                max_age=DERIVATIVES_SETTINGS["oi_max_age"],
                min_coverage=DERIVATIVES_SETTINGS["oi_min_coverage"],
            )
        )
    if (
        DERIVATIVES_FLAGS.get("use_derivatives_combined")
        and DERIVATIVES_FLAGS.get("use_funding_rate")
        and DERIVATIVES_FLAGS.get("use_open_interest")
    ):
        indicators.append(
            DerivativesCombined(
                name="deriv_combo",
                funding_prefix="funding_ctx",
                oi_prefix="oi_ctx",
                funding_window=DERIVATIVES_SETTINGS["funding_window"],
                oi_change_horizon=DERIVATIVES_SETTINGS["oi_change_horizon"],
                volatility_window=DERIVATIVES_SETTINGS["combined_volatility_window"],
            )
        )
    return indicators


def compute_enhanced_fvg_features(df: pd.DataFrame, min_gap_pct: float = MIN_GAP_PCT) -> pd.DataFrame:
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()
    n = len(df)

    bull_sell_level = np.full(n, np.nan)
    bull_sell_dist_pct = np.full(n, np.nan)
    bull_sell_touch = np.zeros(n)
    bull_active_count = np.zeros(n)
    bear_buy_level = np.full(n, np.nan)
    bear_buy_dist_pct = np.full(n, np.nan)
    bear_buy_touch = np.zeros(n)
    bear_active_count = np.zeros(n)

    active_bulls: list[dict[str, float]] = []
    active_bears: list[dict[str, float]] = []

    for index in range(n):
        cur_close = close[index]
        cur_high = high[index]
        cur_low = low[index]

        active_bulls = [gap for gap in active_bulls if cur_low > gap["gap_bottom"]]
        active_bears = [gap for gap in active_bears if cur_high < gap["gap_top"]]

        if index >= 2 and cur_close > 0.0:
            bull_gap_bottom = high[index - 2]
            bull_gap_top = low[index]
            bull_gap_size = bull_gap_top - bull_gap_bottom
            if bull_gap_size > 0.0 and (bull_gap_size / cur_close) >= min_gap_pct:
                active_bulls.append({"sell_level": bull_gap_top, "gap_bottom": bull_gap_bottom})

            bear_gap_top = low[index - 2]
            bear_gap_bottom = high[index]
            bear_gap_size = bear_gap_top - bear_gap_bottom
            if bear_gap_size > 0.0 and (bear_gap_size / cur_close) >= min_gap_pct:
                active_bears.append({"buy_level": bear_gap_bottom, "gap_top": bear_gap_top})

        bull_active_count[index] = float(len(active_bulls))
        bear_active_count[index] = float(len(active_bears))

        if active_bulls and cur_close > 0.0:
            nearest_bull = min(active_bulls, key=lambda gap: abs(cur_close - gap["sell_level"]))
            bull_sell_level[index] = nearest_bull["sell_level"]
            bull_sell_dist_pct[index] = (cur_close - nearest_bull["sell_level"]) / cur_close
            bull_sell_touch[index] = 1.0 if cur_high >= nearest_bull["sell_level"] else 0.0

        if active_bears and cur_close > 0.0:
            nearest_bear = min(active_bears, key=lambda gap: abs(cur_close - gap["buy_level"]))
            bear_buy_level[index] = nearest_bear["buy_level"]
            bear_buy_dist_pct[index] = (cur_close - nearest_bear["buy_level"]) / cur_close
            bear_buy_touch[index] = 1.0 if cur_low <= nearest_bear["buy_level"] else 0.0

    return pd.DataFrame(
        {
            "timestamp": df.index,
            "available_at": df.index,
            "efvg_bull_sell_level": bull_sell_level,
            "efvg_bull_sell_dist_pct": bull_sell_dist_pct,
            "efvg_bull_sell_touch": bull_sell_touch,
            "efvg_bull_active_count": bull_active_count,
            "efvg_bear_buy_level": bear_buy_level,
            "efvg_bear_buy_dist_pct": bear_buy_dist_pct,
            "efvg_bear_buy_touch": bear_buy_touch,
            "efvg_bear_active_count": bear_active_count,
        }
    )


def build_daily_sma200(daily_bars: pd.DataFrame) -> pd.DataFrame:
    daily = daily_bars.copy()
    daily["sma200_daily"] = daily["close"].rolling(200, min_periods=200).mean()
    daily["price_vs_sma200_daily"] = (daily["close"] - daily["sma200_daily"]) / daily["close"].replace(0.0, np.nan)
    daily["sma200_slope_5d"] = daily["sma200_daily"].pct_change(5)
    daily = daily.loc[:, ["sma200_daily", "price_vs_sma200_daily", "sma200_slope_5d"]].dropna(how="all")
    daily = daily.reset_index().rename(columns={daily.index.name or "index": "timestamp"})
    daily["timestamp"] = pd.to_datetime(daily["timestamp"], utc=True)
    daily["available_at"] = daily["timestamp"] + pd.Timedelta(days=1)
    return daily[["timestamp", "available_at", "sma200_daily", "price_vs_sma200_daily", "sma200_slope_5d"]]


def build_mtf_regime_features(pipeline) -> pd.DataFrame:
    data = pipeline.require("data")
    features = pipeline.require("features")
    available = set(features.columns)
    columns: dict[str, pd.Series] = {
        "vol_20": data["close"].pct_change().rolling(20).std(),
    }
    for column in [
        "efvg_bull_sell_dist_pct",
        "efvg_bear_buy_dist_pct",
        "efvg_bull_active_count",
        "efvg_bear_active_count",
        "price_vs_sma200_daily",
        "funding_ctx_z_9",
        "funding_ctx_delta",
        "oi_ctx_log_change_4h",
        "oi_ctx_trend_spread",
        "oi_ctx_pressure_vs_notional_volume",
        "deriv_combo_crowding_proxy",
    ]:
        if column in available:
            columns[column] = features[column]
    return pd.DataFrame(columns).dropna()


def _fill_sparse_feature_columns(pipeline, features: pd.DataFrame) -> pd.DataFrame:
    sparse_columns = [
        column
        for column in features.columns
        if (
            column.startswith("enhanced_fvg_")
            or column.startswith("sma200_daily_")
            or column.startswith("funding_ctx_")
            or column.startswith("oi_ctx_")
            or column.startswith("deriv_combo_")
        )
    ]
    if not sparse_columns:
        return features
    filled = features.copy()
    filled.loc[:, sparse_columns] = filled[sparse_columns].fillna(0.0)
    pipeline.state["features"] = filled
    return filled


def main() -> None:
    args = parse_example_args("Multi-timeframe FVG + SMA(200) daily example (BTCUSDT 15m spot).")
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

    print_section(sep, 1, f"Pre-fetching {SYMBOL} {BASE_INTERVAL} data for custom feature computation")
    base_bars = fetch_binance_bars(symbol=SYMBOL, interval=BASE_INTERVAL, start=model_start, end=model_end, market="spot")
    print(f"  {BASE_INTERVAL} bars : {len(base_bars)}")
    print(f"  range         : {base_bars.index[0]} ... {base_bars.index[-1]}")

    print_section(sep, 2, "Computing enhanced FVG signal levels (min_gap_pct=1 %)")
    efvg_df = compute_enhanced_fvg_features(base_bars, min_gap_pct=MIN_GAP_PCT)
    print(f"  bars with active bull sell level : {int((efvg_df['efvg_bull_active_count'] > 0).sum())}")
    print(f"  bars with active bear buy  level : {int((efvg_df['efvg_bear_active_count'] > 0).sum())}")

    print_section(sep, 3, "Computing daily SMA(200) custom feed")
    daily_bars = fetch_binance_bars(symbol=SYMBOL, interval="1d", start=sma_history_start, end=model_end, market="spot")
    sma200_df = build_daily_sma200(daily_bars)
    print(f"  1d bars fetched                : {len(daily_bars)}")
    print(f"  valid SMA(200) rows            : {int(sma200_df['sma200_daily'].notna().sum())}")

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

    print_section(sep, 5, "Building pipeline config")
    config = build_spot_research_config(
        SYMBOL,
        BASE_INTERVAL,
        config_model_start,
        config_model_end,
        indicators=indicators,
        custom_data=[efvg_entry, sma200_entry],
        config_overrides={
            "data": {"futures_context": {"enabled": False}},
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
                "fee_rate": 0.001,
                "slippage_rate": 0.0002,
                "slippage_model": "sqrt_impact",
                "engine": "vectorbt",
                "use_open_execution": True,
                "signal_delay_bars": 2,
            },
            "quick_overrides": {
                "regime": {"enabled": False},
                "features": {"context_timeframes": []},
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
    config["experiment"] = {
        "name": "mtf_fvg_spot",
        "description": "Multi-timeframe FVG + SMA(200) daily spot example.",
    }
    config["derivatives"] = {
        **DERIVATIVES_FLAGS,
        "window_mode": window["window_mode"],
        "recent_window_days": DERIVATIVES_SETTINGS["recent_window_days"],
    }

    run_example(
        config,
        market="spot",
        local_certification=args.local_certification,
        quick=args.quick,
        quiet=args.quiet,
        nautilus_available=NAUTILUS_AVAILABLE,
        example_name="example_mtf_fvg.py",
        hooks={"after_build_features": _fill_sparse_feature_columns},
    )


if __name__ == "__main__":
    main()
