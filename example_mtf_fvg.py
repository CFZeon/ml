"""Multi-timeframe FVG + SMA(200) daily spot example — BTCUSDT 15m base.

Feature stack
-------------
  Base indicators (15m)
    - RSI(14), ATR(14)
    - FVG with 1 % minimum gap width filter

  Multi-timeframe context (resampled from 15m base via context_timeframes)
    - 1h, 4h, 1d OHLCV-derived statistics (trend, volatility, breakout, etc.)

  Enhanced FVG signal levels (injected as custom data)
    Bullish FVG: low[t] > high[t-2]
      - sell_level = low of the 3rd candle  (price retrace to gap top = sell zone)
    Bearish FVG: high[t] < low[t-2]
      - buy_level  = high of the 3rd candle (price retrace to gap bottom = buy zone)
    Gaps narrower than 1 % of close price are discarded.
    A level deactivates when price fully trades through the gap.
    Features: nearest active level, pct distance, touch flag, active count.

  Daily SMA(200) (injected as point-in-time-safe custom data)
    Computed from a 2-year history so the SMA is fully warm before the model
    window starts.  available_at is offset +1 day (daily-bar resolution).
    Features: sma200_daily, price_vs_sma200_daily (pct deviation), sma200_slope_5d.

Usage
-----
    python example_mtf_fvg.py
    python example_mtf_fvg.py --local-certification
"""

import numpy as np
import pandas as pd

from core import ResearchPipeline, fetch_binance_bars
from core.execution import NAUTILUS_AVAILABLE
from example_utils import (
    build_custom_data_entry,
    build_spot_research_config,
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

# ── configuration constants ───────────────────────────────────────────────────

SYMBOL = "BTCUSDT"
BASE_INTERVAL = "15m"

# Model window: 6 months of 15 m bars  ≈ 17 520 rows
MODEL_START = "2024-01-01"
MODEL_END = "2024-07-01"

# SMA(200) needs 200 daily bars of warmup before MODEL_START.
# Fetching 1d bars from 2 years prior ensures the SMA is fully warm on day 1.
SMA_HISTORY_START = "2022-01-01"

# Minimum FVG width as a fraction of close price (1 %).
MIN_GAP_PCT = 0.01


# ── enhanced FVG feature builder ──────────────────────────────────────────────

def compute_enhanced_fvg_features(df: pd.DataFrame, min_gap_pct: float = MIN_GAP_PCT) -> pd.DataFrame:
    """Compute enhanced FVG signal-level features from a bar DataFrame.

    Bullish FVG detected when  low[t] > high[t-2].
      sell_level = low[t]  (the gap's upper boundary / 3rd-candle low).
      Rationale: price retracing back to this level enters the unfilled gap
      from above; treat as a short-term sell / resistance zone.

    Bearish FVG detected when  high[t] < low[t-2].
      buy_level = high[t]  (the gap's lower boundary / 3rd-candle high).
      Rationale: price retracing back to this level enters the unfilled gap
      from below; treat as a short-term buy / support zone.

    Gaps narrower than min_gap_pct of the current close are discarded.
    A gap is deactivated when price fully trades through it:
      - bullish gap: current low <= gap_bottom (= high[t-2])
      - bearish gap: current high >= gap_top   (= low[t-2])

    Output columns (all float, NaN when no active gap exists):
      efvg_bull_sell_level     closest active bull sell-level price
      efvg_bull_sell_dist_pct  (close - sell_level) / close
                               positive  - close is above the sell level
                               negative  - close has fallen below it
      efvg_bull_sell_touch     1.0 if this bar's high >= sell_level, else 0.0
      efvg_bull_active_count   number of currently active bullish levels
      efvg_bear_buy_level      closest active bear buy-level price
      efvg_bear_buy_dist_pct   (close - buy_level) / close
                               positive  - close is above the buy level
                               negative  - close has fallen below it
      efvg_bear_buy_touch      1.0 if this bar's low <= buy_level, else 0.0
      efvg_bear_active_count   number of currently active bearish levels
    """
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

    # Each entry: {"sell_level": float, "gap_bottom": float}
    active_bulls: list = []
    # Each entry: {"buy_level": float, "gap_top": float}
    active_bears: list = []

    for i in range(n):
        cur_close = close[i]
        cur_high = high[i]
        cur_low = low[i]

        # --- expire fully-filled gaps -------------------------------------------
        # Bullish gap is filled when price trades down through gap_bottom (high[t-2]).
        active_bulls = [g for g in active_bulls if cur_low > g["gap_bottom"]]
        # Bearish gap is filled when price trades up through gap_top (low[t-2]).
        active_bears = [g for g in active_bears if cur_high < g["gap_top"]]

        # --- detect new FVGs formed at bar i (needs i-2 and i) ------------------
        if i >= 2:
            # Bullish FVG
            gap_bottom = high[i - 2]
            gap_top = low[i]
            gap_size = gap_top - gap_bottom
            if (
                gap_size > 0
                and cur_close > 0
                and (gap_size / cur_close) >= min_gap_pct
            ):
                active_bulls.append(
                    {"sell_level": gap_top, "gap_bottom": gap_bottom}
                )

            # Bearish FVG
            bear_gap_top = low[i - 2]
            bear_gap_bottom = high[i]
            bear_gap_size = bear_gap_top - bear_gap_bottom
            if (
                bear_gap_size > 0
                and cur_close > 0
                and (bear_gap_size / cur_close) >= min_gap_pct
            ):
                active_bears.append(
                    {"buy_level": bear_gap_bottom, "gap_top": bear_gap_top}
                )

        # --- record nearest active level ----------------------------------------
        bull_active_count[i] = float(len(active_bulls))
        bear_active_count[i] = float(len(active_bears))

        if active_bulls and cur_close > 0:
            nearest = min(active_bulls, key=lambda g: abs(cur_close - g["sell_level"]))
            bull_sell_level[i] = nearest["sell_level"]
            bull_sell_dist_pct[i] = (cur_close - nearest["sell_level"]) / cur_close
            bull_sell_touch[i] = 1.0 if cur_high >= nearest["sell_level"] else 0.0

        if active_bears and cur_close > 0:
            nearest = min(active_bears, key=lambda g: abs(cur_close - g["buy_level"]))
            bear_buy_level[i] = nearest["buy_level"]
            bear_buy_dist_pct[i] = (cur_close - nearest["buy_level"]) / cur_close
            bear_buy_touch[i] = 1.0 if cur_low <= nearest["buy_level"] else 0.0

    return pd.DataFrame(
        {
            "timestamp": df.index,
            "available_at": df.index,   # known at bar close
            "efvg_bull_sell_level": bull_sell_level,
            "efvg_bull_sell_dist_pct": bull_sell_dist_pct,
            "efvg_bull_sell_touch": bull_sell_touch,
            "efvg_bull_active_count": bull_active_count,
            "efvg_bear_buy_level": bear_buy_level,
            "efvg_bear_buy_dist_pct": bear_buy_dist_pct,
            "efvg_bear_buy_touch": bear_buy_touch,
            "efvg_bear_active_count": bear_active_count,
        }
    ).reset_index(drop=True)


# ── daily SMA(200) builder ────────────────────────────────────────────────────

def build_daily_sma200(daily_bars: pd.DataFrame) -> pd.DataFrame:
    """Compute SMA(200) on daily close bars and return a custom-data DataFrame.

    daily_bars must cover at least 200 days before the model window start so
    that the SMA has a full warmup period.

    available_at is set to timestamp + 1 day so the value is never used on
    the same bar that produced it (point-in-time safety at daily resolution).

    Output columns:
      sma200_daily            the 200-day simple moving average of daily close
      price_vs_sma200_daily   (close - sma200) / sma200  (pct deviation)
      sma200_slope_5d         5-day percentage change of sma200 (trend direction)
    """
    daily_close = (
        daily_bars["close"]
        .resample("1D", label="right", closed="left")
        .last()
        .dropna()
    )
    sma200 = daily_close.rolling(200, min_periods=200).mean()
    price_vs_sma200 = (daily_close - sma200) / sma200
    sma200_slope_5d = sma200.pct_change(5)

    out = pd.DataFrame(
        {
            "timestamp": sma200.index,
            "available_at": sma200.index + pd.Timedelta(days=1),
            "sma200_daily": sma200.values,
            "price_vs_sma200_daily": price_vs_sma200.values,
            "sma200_slope_5d": sma200_slope_5d.values,
        }
    ).dropna()
    return out.reset_index(drop=True)


# ── regime feature builder ─────────────────────────────────────────────────────

def build_mtf_regime_features(pipeline) -> pd.DataFrame:
    """Regime features: close volatility + enhanced FVG level imbalance."""
    data = pipeline.require("data")
    features = pipeline.require("features")
    available = set(features.columns)

    cols = {}
    cols["vol_20"] = data["close"].pct_change().rolling(20).std()

    for col in [
        "efvg_bull_sell_dist_pct",
        "efvg_bear_buy_dist_pct",
        "efvg_bull_active_count",
        "efvg_bear_active_count",
        "price_vs_sma200_daily",
    ]:
        if col in available:
            cols[col] = features[col]

    return pd.DataFrame(cols).dropna()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_local_certification_args(
        "Multi-timeframe FVG + SMA(200) daily example (BTCUSDT 15m spot)."
    )
    sep = "=" * 60

    # ── 1. Pre-fetch data for custom feature computation ───────────────────────
    print_section(sep, 1, f"Pre-fetching {SYMBOL} data for custom feature computation")

    base_bars = fetch_binance_bars(
        symbol=SYMBOL,
        interval=BASE_INTERVAL,
        start=MODEL_START,
        end=MODEL_END,
        market="spot",
    )
    print(f"  {BASE_INTERVAL} bars (model window) : {len(base_bars)}")
    print(f"  range                      : {base_bars.index[0]} ... {base_bars.index[-1]}")

    daily_bars = fetch_binance_bars(
        symbol=SYMBOL,
        interval="1d",
        start=SMA_HISTORY_START,
        end=MODEL_END,
        market="spot",
    )
    print(f"  1d bars (SMA history)      : {len(daily_bars)}")
    print(f"  range                      : {daily_bars.index[0]} ... {daily_bars.index[-1]}")

    # ── 2. Compute enhanced FVG signal levels (1 % width filter) ──────────────
    print_section(sep, 2, "Computing enhanced FVG signal levels (min_gap_pct=1 %)")

    efvg_df = compute_enhanced_fvg_features(base_bars, min_gap_pct=MIN_GAP_PCT)
    bull_bars = int((efvg_df["efvg_bull_active_count"] > 0).sum())
    bear_bars = int((efvg_df["efvg_bear_active_count"] > 0).sum())
    bull_touches = int(efvg_df["efvg_bull_sell_touch"].sum())
    bear_touches = int(efvg_df["efvg_bear_buy_touch"].sum())
    print(f"  bars with active bull sell level : {bull_bars}")
    print(f"  bars with active bear buy  level : {bear_bars}")
    print(f"  bull sell-level touch events     : {bull_touches}  (potential sell signal)")
    print(f"  bear buy-level  touch events     : {bear_touches}  (potential buy  signal)")

    # ── 3. Compute daily SMA(200) ──────────────────────────────────────────────
    print_section(sep, 3, "Computing daily SMA(200) from 2-year history")

    sma200_df = build_daily_sma200(daily_bars)
    valid_sma200 = int(sma200_df["sma200_daily"].notna().sum())
    print(f"  daily bars in history         : {len(daily_bars)}")
    print(f"  valid SMA(200) readings       : {valid_sma200}")
    if valid_sma200 == 0:
        raise RuntimeError(
            "No valid SMA(200) values produced — extend SMA_HISTORY_START further back."
        )
    if not sma200_df.empty:
        price_above = float(
            (sma200_df["price_vs_sma200_daily"] > 0).sum() / len(sma200_df) * 100
        )
        print(f"  % daily closes above SMA(200) : {price_above:.1f} %")

    # ── 4. Assemble custom-data entries ───────────────────────────────────────
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
        allow_exact_matches=True,   # 15m bar aligns exactly to its own timestamp
    )

    sma200_entry = build_custom_data_entry(
        "sma200_daily",
        sma200_df,
        value_columns=["sma200_daily", "price_vs_sma200_daily", "sma200_slope_5d"],
        allow_exact_matches=False,  # available_at is offset +1 day; use asof join
    )
    print(f"  custom entries: enhanced_fvg ({len(efvg_df)} rows), "
          f"sma200_daily ({len(sma200_df)} rows)")

    # ── 5. Build pipeline config ───────────────────────────────────────────────
    print_section(sep, 5, "Building pipeline config")

    config = build_spot_research_config(
        SYMBOL,
        BASE_INTERVAL,
        MODEL_START,
        MODEL_END,
        indicators=[
            {"kind": "rsi",  "params": {"period": 14}},
            {"kind": "atr",  "params": {"period": 14}},
            {
                "kind": "fvg",
                "params": {
                    "name": "fvg_base",
                    "min_gap_pct": MIN_GAP_PCT,   # 1 % width filter on the indicator too
                },
            },
        ],
        custom_data=[efvg_entry, sma200_entry],
        config_overrides={
            "data": {
                # Futures context data is not aligned to 15m bars — disable it
                # to avoid the TTL unknown-rate guard dropping all rows.
                "futures_context": {"enabled": False},
            },
            "features": {
                # Add 1h to the default [4h, 1d] set
                "context_timeframes": ["1h", "4h", "1d"],
                "lags": [1, 3, 6],
                "frac_diff_d": 0.4,
                "rolling_window": 20,
                "squeeze_quantile": 0.2,
                # Relax TTL unknown-rate — futures context is disabled but the
                # config key is still present from the base template.
                "futures_context_ttl": {
                    "mark_price": "2h",
                    "premium_index": "2h",
                    "funding": "12h",
                    "recent": "4h",
                    "max_stale_rate": 1.0,
                    "max_unknown_rate": 1.0,
                },
                # No cross-asset symbols — relax the unknown-rate guard so
                # the alignment step does not drop all rows.
                "cross_asset_context_ttl": {"max_age": "2h", "max_unknown_rate": 1.0},
            },
            "regime": {
                "method": "hmm",
                "builder": build_mtf_regime_features,
            },
            "labels": {
                "kind": "triple_barrier",
                # Tighter barriers for 15m bars (smaller expected moves)
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
                "profitability_threshold": -1.0,  # Disable profitability filter for demo
                "min_position_size": 0.1,         # Force non-zero size even with negative edge (demo only)
                "avg_win": 0.015,
                "avg_loss": 0.01,
                "shrinkage_alpha": 0.5,
                "fraction": 0.5,
                "min_trades_for_kelly": 30,
                "max_kelly_fraction": 0.5,
                "threshold": 0.0,
                "edge_threshold": 0.0,
                "meta_threshold": 0.0,
                "expected_edge_threshold": -1.0,  # Disable edge filter for demo
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
        },
    )

    try:
        config = prepare_example_runtime_config(
            config,
            market="spot",
            local_certification=args.local_certification,
            nautilus_available=NAUTILUS_AVAILABLE,
            example_name="example_mtf_fvg.py",
        )
    except RuntimeError as exc:
        print(str(exc))
        raise SystemExit(2) from exc

    pipeline = ResearchPipeline(config)
    example_runtime = dict(config.get("example_runtime") or {})

    # ── 6. Fetch data ──────────────────────────────────────────────────────────
    print_section(sep, 6, f"Fetching {SYMBOL} {BASE_INTERVAL} spot data via pipeline")

    df = pipeline.fetch_data()
    print(f"  rows  : {len(df)}")
    print(f"  range : {df.index[0]} ... {df.index[-1]}")
    if example_runtime:
        print(f"  mode  : {example_runtime.get('mode')}")

    # ── 7. Indicators ──────────────────────────────────────────────────────────
    print_section(sep, 7, "Running indicators  (RSI, ATR, FVG 1 % min-gap)")

    indicator_run = pipeline.run_indicators()
    fvg_meta = indicator_run.metadata.get("fvg_base", {})
    bull_col = "fvg_base_bull_active_count"
    bear_col = "fvg_base_bear_active_count"
    active_bull_ind = int(
        indicator_run.frame.get(bull_col, pd.Series(dtype=float)).gt(0).sum()
    )
    active_bear_ind = int(
        indicator_run.frame.get(bear_col, pd.Series(dtype=float)).gt(0).sum()
    )
    print(f"  indicator names  : {list(indicator_run.metadata)}")
    print(f"  FVG output cols  : {fvg_meta.get('output_columns', [])}")
    print(f"  bars w/ active bull gap (indicator): {active_bull_ind}")
    print(f"  bars w/ active bear gap (indicator): {active_bear_ind}")

    # ── 8. Features ────────────────────────────────────────────────────────────
    print_section(sep, 8, "Building features  (MTF 1h/4h/1d + eFVG + SMA200)")

    features = pipeline.build_features()
    mtf_cols  = [c for c in features.columns if c.startswith("mtf_")]
    efvg_cols = [c for c in features.columns if "efvg_" in c]
    sma_cols  = [c for c in features.columns if "sma200" in c]
    fvg_ind_cols = [c for c in features.columns if c.startswith("fvg_base")]

    print(f"  total features    : {features.shape[1]}")
    print(f"  MTF features      : {len(mtf_cols)}")
    if mtf_cols:
        sample = ", ".join(mtf_cols[:6])
        suffix = "..." if len(mtf_cols) > 6 else ""
        print(f"    sample: {sample}{suffix}")
    print(f"  eFVG features     : {len(efvg_cols)}  {efvg_cols}")
    print(f"  SMA200 features   : {len(sma_cols)}  {sma_cols}")
    print(f"  FVG-indicator cols: {len(fvg_ind_cols)}")

    # Sparse event-driven custom features can remain NaN for long stretches
    # (for example when one side of FVG is inactive). Use neutral zero values
    # so alignment does not drop the entire sample matrix.
    custom_sparse_cols = [
        col
        for col in features.columns
        if col.startswith("enhanced_fvg_") or col.startswith("sma200_daily_")
    ]
    if custom_sparse_cols:
        na_before = int(features[custom_sparse_cols].isna().sum().sum())
        features.loc[:, custom_sparse_cols] = features[custom_sparse_cols].fillna(0.0)
        na_after = int(features[custom_sparse_cols].isna().sum().sum())
        pipeline.state["features"] = features
        print(
            "  custom NaN fill  : "
            f"cols={len(custom_sparse_cols)}  before={na_before}  after={na_after}"
        )

    stationarity = pipeline.check_stationarity()
    print_stationarity_summary(stationarity)

    # ── 9. Regimes ─────────────────────────────────────────────────────────────
    print_section(sep, 9, "Detecting regimes")
    regime_result = pipeline.detect_regimes()
    print_regime_summary(regime_result["regimes"])

    # ── 10. Labels ─────────────────────────────────────────────────────────────
    print_section(sep, 10, "Triple-barrier labels  (pt/sl=1.5×ATR, max_holding=16 bars)")
    labels = pipeline.build_labels()
    print_label_summary(labels)

    # ── 11. Alignment ──────────────────────────────────────────────────────────
    print_section(sep, 11, "Aligning research matrix")
    aligned = pipeline.align_data()
    print_alignment_summary(aligned)

    # ── diagnostic: show NaN counts if alignment produced 0 samples ───────────
    if len(aligned["X"]) == 0:
        features_df = pipeline.require("features")
        labels_df = pipeline.require("labels")
        common_idx = features_df.index.intersection(labels_df.index)
        X_diag = features_df.loc[common_idx]
        nan_counts = X_diag.isna().sum()
        all_nan = nan_counts[nan_counts == len(X_diag)]
        any_nan = nan_counts[(nan_counts > 0) & (nan_counts < len(X_diag))].nlargest(10)
        full_nan = nan_counts[nan_counts == len(X_diag)]
        print(f"  [DIAG] common rows: {len(common_idx)}")
        print(f"  [DIAG] cols with ALL NaN  : {len(full_nan)} / {list(full_nan.index[:10])}")
        print(f"  [DIAG] cols with SOME NaN (top-10): {any_nan.to_dict()}")
        # find rows where all features are non-NaN
        row_nan_counts = X_diag.isna().sum(axis=1)
        print(f"  [DIAG] row NaN count stats: min={row_nan_counts.min()}  max={row_nan_counts.max()}  mean={row_nan_counts.mean():.1f}")
        raise SystemExit("Alignment diagnostic complete — 0 samples detected")

    # ── 12. Feature selection + weighting ─────────────────────────────────────
    print_section(sep, 12, "Feature selection + sample weighting")
    selection = pipeline.select_features()
    print_feature_selection_summary(selection)
    weights = pipeline.compute_sample_weights()
    print_weight_summary(weights)

    # ── 13. Training ───────────────────────────────────────────────────────────
    print_section(sep, 13, "CPCV training")
    training = pipeline.train_models()
    print_training_summary(training)

    # ── 14. Signals ────────────────────────────────────────────────────────────
    print_section(sep, 14, "Generating signals")
    signal_result = pipeline.generate_signals()
    print_signal_summary(signal_result, allow_short=False)

    # ── 15. Backtest ───────────────────────────────────────────────────────────
    print_section(sep, 15, "Backtesting")
    backtest = pipeline.run_backtest()
    print_backtest_summary(backtest)
    if float(backtest.get("total_trades") or 0.0) == 0.0:
        print(
            "  note: profitability filter abstained on all CPCV paths — "
            "use as a feature-and-validation smoke test."
        )

    print(f"\n{sep}\nPipeline complete.\n{sep}")


if __name__ == "__main__":
    main()
