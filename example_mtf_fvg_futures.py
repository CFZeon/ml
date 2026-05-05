"""Multi-timeframe FVG + SMA(200) daily futures example with derivative-based derivatives context.

Feature stack
-------------
  Base market
    - BTCUSDT USD-M futures, 15m base bars

  Base indicators (15m)
    - RSI(14), ATR(14)
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

from core import ResearchPipeline, fetch_binance_bars
from core.execution import NAUTILUS_AVAILABLE
from core.indicators._derivatives_binance import fetch_funding_history
from example_mtf_fvg import (
    BASE_INTERVAL,
    DERIVATIVES_FLAGS,
    DERIVATIVES_SETTINGS,
    MIN_GAP_PCT,
    SYMBOL,
    build_daily_sma200,
    build_indicator_specs,
    build_mtf_regime_features,
    compute_enhanced_fvg_features,
    format_config_timestamp,
    print_derivatives_indicator_summary,
    resolve_example_window,
)
from example_utils import (
    build_custom_data_entry,
    build_futures_research_config,
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


def main():
    args = parse_local_certification_args(
        "Multi-timeframe FVG + SMA(200) daily futures example (BTCUSDT 15m USD-M futures)."
    )
    sep = "=" * 60
    window = resolve_example_window()
    model_start = window["model_start"]
    model_end = window["model_end"]
    sma_history_start = window["sma_history_start"]
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
    config["derivatives"] = {
        **DERIVATIVES_FLAGS,
        "window_mode": window["window_mode"],
        "recent_window_days": DERIVATIVES_SETTINGS["recent_window_days"],
    }

    try:
        config = prepare_example_runtime_config(
            config,
            market="um_futures",
            local_certification=args.local_certification,
            nautilus_available=NAUTILUS_AVAILABLE,
            example_name="example_mtf_fvg_futures.py",
        )
    except RuntimeError as exc:
        print(str(exc))
        raise SystemExit(2) from exc

    pipeline = ResearchPipeline(config)
    example_runtime = dict(config.get("example_runtime") or {})

    print_section(sep, 6, f"Fetching {SYMBOL} {BASE_INTERVAL} futures data via pipeline")
    data = pipeline.fetch_data()
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
    filters = pipeline.state.get("symbol_filters", {})
    contract_spec = pipeline.state.get("futures_contract_spec", {})
    print(f"  rows  : {len(data)}")
    print(f"  range : {data.index[0]} ... {data.index[-1]}")
    if example_runtime:
        print(f"  mode  : {example_runtime.get('mode')}")
    if filters:
        print(
            "  filters: "
            f"tick={filters.get('tick_size')}  step={filters.get('step_size')}  min_notional={filters.get('min_notional')}"
        )
    if contract_spec:
        print(
            "  contract: "
            f"margin={contract_spec.get('margin_asset')}  liq_fee={contract_spec.get('liquidation_fee_rate')}"
        )
    print(f"  runtime funding rows: {len(funding_frame)}")

    print_section(sep, 7, "Running indicators  (RSI, ATR, FVG, funding delta, OI change, combined)")
    indicator_run = pipeline.run_indicators()
    fvg_meta = indicator_run.metadata.get("fvg_base", {})
    print(f"  indicator names  : {list(indicator_run.metadata)}")
    print(f"  FVG output cols  : {fvg_meta.get('output_columns', [])}")
    print_derivatives_indicator_summary(indicator_run)

    print_section(sep, 8, "Building features  (MTF + eFVG + SMA200 + derivatives)")
    features = pipeline.build_features()
    mtf_cols = [column for column in features.columns if column.startswith("mtf_")]
    efvg_cols = [column for column in features.columns if "efvg_" in column]
    sma_cols = [column for column in features.columns if "sma200" in column]
    deriv_cols = [
        column
        for column in features.columns
        if column.startswith("funding_ctx_") or column.startswith("oi_ctx_") or column.startswith("deriv_combo_")
    ]
    print(f"  total features    : {features.shape[1]}")
    print(f"  MTF features      : {len(mtf_cols)}")
    print(f"  eFVG features     : {len(efvg_cols)}")
    print(f"  SMA200 features   : {len(sma_cols)}")
    print(f"  derivatives cols  : {len(deriv_cols)}")

    custom_sparse_cols = [
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
    if custom_sparse_cols:
        na_before = int(features[custom_sparse_cols].isna().sum().sum())
        features.loc[:, custom_sparse_cols] = features[custom_sparse_cols].fillna(0.0)
        na_after = int(features[custom_sparse_cols].isna().sum().sum())
        pipeline.state["features"] = features
        print(f"  custom NaN fill  : cols={len(custom_sparse_cols)}  before={na_before}  after={na_after}")

    stationarity = pipeline.check_stationarity()
    print_stationarity_summary(stationarity)

    print_section(sep, 9, "Detecting regimes")
    print_regime_summary(pipeline.detect_regimes()["regimes"])

    print_section(sep, 10, "Building labels and aligning research matrix")
    labels = pipeline.build_labels()
    aligned = pipeline.align_data()
    print_label_summary(labels)
    print_alignment_summary(aligned)

    print_section(sep, 11, "Feature selection + sample weighting")
    selection = pipeline.select_features()
    print_feature_selection_summary(selection)
    weights = pipeline.compute_sample_weights()
    print_weight_summary(weights)

    print_section(sep, 12, "CPCV training")
    training = pipeline.train_models()
    print_training_summary(training)

    print_section(sep, 13, "Generating signals")
    signals = pipeline.generate_signals()
    print_signal_summary(signals)

    print_section(sep, 14, "Backtesting")
    backtest = pipeline.run_backtest()
    print_backtest_summary(backtest)

    print(f"\n{sep}\nFutures MTF FVG example complete.\n{sep}")


if __name__ == "__main__":
    main()