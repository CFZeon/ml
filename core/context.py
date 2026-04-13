"""Market context loaders and feature builders for constrained crypto alpha research."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .data import fetch_binance_bars, _interval_timedelta, _parse_bound, _parse_interval
from .features import FeatureBlock


_FAPI_BASE = "https://fapi.binance.com"
_SUPPORTED_PERIODS = {"5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"}
_DEFAULT_CONTEXT_SYMBOLS = ["ETHUSDT", "SOLUSDT", "BNBUSDT"]


def _cache_path(cache_dir, namespace, payload):
    if cache_dir is None:
        return None

    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha1(encoded).hexdigest()
    return Path(cache_dir) / "context" / namespace / f"{digest}.pkl"


def _read_cache(path):
    if path is None or not path.exists():
        return None
    return pd.read_pickle(path)


def _write_cache(path, frame):
    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_pickle(temp_path)
    temp_path.replace(path)


def _request_json(session, path, params):
    response = session.get(f"{_FAPI_BASE}{path}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _interval_to_pandas_freq(interval):
    value, unit = _parse_interval(interval)
    if unit == "m":
        return f"{value}min"
    if unit == "h":
        return f"{value}h"
    if unit == "d":
        return f"{value}d"
    if unit == "w":
        return f"{value}w"
    if unit == "mo":
        return f"{value}MS"
    if unit == "s":
        return f"{value}s"
    raise ValueError(f"Unsupported interval={interval!r}")


def _context_period(interval):
    if interval in _SUPPORTED_PERIODS:
        return interval
    return "1h"


def _empty_feature_block(index, block_name):
    return FeatureBlock(frame=pd.DataFrame(index=index), laggable_columns=[], block_name=block_name)


def _rolling_zscore(series, window):
    mean = series.rolling(window, min_periods=max(2, window // 2)).mean()
    std = series.rolling(window, min_periods=max(2, window // 2)).std().replace(0, np.nan)
    return (series - mean) / std


def _safe_divide(numerator, denominator):
    if isinstance(denominator, pd.Series):
        denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def _asof_reindex(base_index, frame):
    if frame is None or frame.empty:
        return pd.DataFrame(index=base_index)

    context = frame.sort_index().reset_index().rename(columns={frame.index.name or "index": "timestamp"})
    anchor = pd.DataFrame({"timestamp": pd.DatetimeIndex(base_index)})
    joined = pd.merge_asof(anchor, context, on="timestamp", direction="backward")
    return joined.set_index("timestamp").reindex(base_index)


def _coverage_ratio(frame):
    if frame is None or frame.empty:
        return 0.0
    return float(frame.notna().all(axis=1).mean())


def _slug_symbol(symbol):
    return symbol.lower()


def _parse_kline_rows(rows, prefix):
    if not rows:
        columns = [f"{prefix}_open", f"{prefix}_high", f"{prefix}_low", f"{prefix}_close"]
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([row[0] for row in rows], unit="ms", utc=True),
            f"{prefix}_open": pd.Series([row[1] for row in rows], dtype=float),
            f"{prefix}_high": pd.Series([row[2] for row in rows], dtype=float),
            f"{prefix}_low": pd.Series([row[3] for row in rows], dtype=float),
            f"{prefix}_close": pd.Series([row[4] for row in rows], dtype=float),
        }
    )
    return frame.set_index("timestamp").sort_index()


def _fetch_kline_endpoint(path, symbol, interval, start_dt, end_dt, session, cache_dir, prefix):
    cache_file = _cache_path(
        cache_dir,
        namespace=prefix,
        payload={"path": path, "symbol": symbol, "interval": interval, "start": start_dt, "end": end_dt},
    )
    cached = _read_cache(cache_file)
    if cached is not None:
        return cached

    interval_delta = _interval_timedelta(interval)
    if interval_delta is None:
        raise ValueError(f"Expected fixed-width interval for futures kline fetch, got {interval!r}")

    rows = []
    current_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    interval_ms = int(interval_delta.total_seconds() * 1000)

    while current_ms < end_ms:
        batch = _request_json(
            session,
            path,
            {"symbol": symbol, "interval": interval, "startTime": current_ms, "endTime": end_ms - 1, "limit": 1500},
        )
        if not batch:
            break

        rows.extend(batch)
        last_open_ms = int(batch[-1][0])
        next_ms = last_open_ms + interval_ms
        if next_ms <= current_ms:
            break
        current_ms = next_ms

    frame = _parse_kline_rows(rows, prefix=prefix)
    if not frame.empty:
        frame = frame[(frame.index >= start_dt) & (frame.index < end_dt)]
    _write_cache(cache_file, frame)
    return frame


def _fetch_funding_history(symbol, start_dt, end_dt, session, cache_dir):
    cache_file = _cache_path(
        cache_dir,
        namespace="funding_rate",
        payload={"symbol": symbol, "start": start_dt, "end": end_dt},
    )
    cached = _read_cache(cache_file)
    if cached is not None:
        return cached

    rows = []
    current_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    while current_ms < end_ms:
        batch = _request_json(
            session,
            "/fapi/v1/fundingRate",
            {"symbol": symbol, "startTime": current_ms, "endTime": end_ms - 1, "limit": 1000},
        )
        if not batch:
            break

        rows.extend(batch)
        last_time_ms = int(batch[-1]["fundingTime"])
        next_ms = last_time_ms + 1
        if next_ms <= current_ms:
            break
        current_ms = next_ms

    if not rows:
        frame = pd.DataFrame(columns=["funding_rate", "funding_mark_price"])
    else:
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime([row["fundingTime"] for row in rows], unit="ms", utc=True),
                "funding_rate": pd.Series([row["fundingRate"] for row in rows], dtype=float),
                "funding_mark_price": pd.Series([row.get("markPrice", np.nan) for row in rows], dtype=float),
            }
        ).set_index("timestamp").sort_index()

    if not frame.empty:
        frame = frame[(frame.index >= start_dt) & (frame.index < end_dt)]
    _write_cache(cache_file, frame)
    return frame


def _recent_window(start_dt, end_dt, max_days=30):
    floor = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=max_days)
    if end_dt <= floor:
        return None
    return max(start_dt, floor), end_dt


def _fetch_period_endpoint(path, params, start_dt, end_dt, session, cache_dir, namespace):
    recent_window = _recent_window(start_dt, end_dt)
    if recent_window is None:
        return pd.DataFrame()

    recent_start, recent_end = recent_window
    cache_file = _cache_path(
        cache_dir,
        namespace=namespace,
        payload={**params, "path": path, "start": recent_start, "end": recent_end},
    )
    cached = _read_cache(cache_file)
    if cached is not None:
        return cached

    period = params["period"]
    period_delta = _interval_timedelta(period)
    if period_delta is None:
        raise ValueError(f"Expected fixed-width period for recent endpoint, got {period!r}")

    rows = []
    current_ms = int(recent_start.timestamp() * 1000)
    end_ms = int(recent_end.timestamp() * 1000)
    step_ms = int(period_delta.total_seconds() * 1000)

    while current_ms < end_ms:
        batch = _request_json(
            session,
            path,
            {**params, "startTime": current_ms, "endTime": end_ms - 1, "limit": 500},
        )
        if not batch:
            break

        rows.extend(batch)
        last_ms = int(batch[-1]["timestamp"])
        next_ms = last_ms + step_ms
        if next_ms <= current_ms:
            break
        current_ms = next_ms

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"].astype("int64"), unit="ms", utc=True)
        frame = frame.set_index("timestamp").sort_index()
        for column in frame.columns:
            if column in {"symbol", "pair", "contractType"}:
                continue
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame[(frame.index >= recent_start) & (frame.index < recent_end)]

    _write_cache(cache_file, frame)
    return frame


def fetch_binance_futures_context(symbol="BTCUSDT", interval="1h", start="2024-01-01", end="2024-03-01", cache_dir=".cache", include_recent_stats=True):
    """Fetch lightweight Binance futures context data for a spot-model feature set."""
    start_dt = _parse_bound(start)
    end_dt = _parse_bound(end)
    period = _context_period(interval)

    with requests.Session() as session:
        context = {
            "mark_price": _fetch_kline_endpoint(
                "/fapi/v1/markPriceKlines",
                symbol=symbol,
                interval=interval,
                start_dt=start_dt,
                end_dt=end_dt,
                session=session,
                cache_dir=cache_dir,
                prefix="mark",
            ),
            "premium_index": _fetch_kline_endpoint(
                "/fapi/v1/premiumIndexKlines",
                symbol=symbol,
                interval=interval,
                start_dt=start_dt,
                end_dt=end_dt,
                session=session,
                cache_dir=cache_dir,
                prefix="premium",
            ),
            "funding": _fetch_funding_history(
                symbol=symbol,
                start_dt=start_dt,
                end_dt=end_dt,
                session=session,
                cache_dir=cache_dir,
            ),
        }

        if include_recent_stats:
            context["open_interest"] = _fetch_period_endpoint(
                "/futures/data/openInterestHist",
                params={"symbol": symbol, "period": period},
                start_dt=start_dt,
                end_dt=end_dt,
                session=session,
                cache_dir=cache_dir,
                namespace="open_interest",
            )
            context["taker_flow"] = _fetch_period_endpoint(
                "/futures/data/takerlongshortRatio",
                params={"symbol": symbol, "period": period},
                start_dt=start_dt,
                end_dt=end_dt,
                session=session,
                cache_dir=cache_dir,
                namespace="taker_flow",
            )
            context["global_long_short"] = _fetch_period_endpoint(
                "/futures/data/globalLongShortAccountRatio",
                params={"symbol": symbol, "period": period},
                start_dt=start_dt,
                end_dt=end_dt,
                session=session,
                cache_dir=cache_dir,
                namespace="global_long_short",
            )
            context["basis"] = _fetch_period_endpoint(
                "/futures/data/basis",
                params={"pair": symbol, "contractType": "PERPETUAL", "period": period},
                start_dt=start_dt,
                end_dt=end_dt,
                session=session,
                cache_dir=cache_dir,
                namespace="basis",
            )

    return context


def fetch_context_symbol_bars(symbols=None, interval="1h", start="2024-01-01", end="2024-03-01", cache_dir=".cache", market="spot"):
    """Fetch a small set of spot symbols used as cross-asset market context."""
    symbol_list = list(symbols or _DEFAULT_CONTEXT_SYMBOLS)
    frames = {}
    for symbol in symbol_list:
        frames[symbol] = fetch_binance_bars(
            symbol=symbol,
            interval=interval,
            start=start,
            end=end,
            cache_dir=cache_dir,
            market=market,
        )
    return frames


def build_futures_context_feature_block(base_data, futures_context, rolling_window=20, min_recent_coverage=0.6):
    """Build normalized derivatives-context features aligned to the base spot index."""
    if not futures_context:
        return _empty_feature_block(base_data.index, block_name="futures_context")

    close = base_data["close"].astype(float)
    base_return = close.pct_change()
    feature_frame = pd.DataFrame(index=base_data.index)

    mark_frame = futures_context.get("mark_price")
    if mark_frame is not None and not mark_frame.empty:
        aligned_mark = _asof_reindex(base_data.index, mark_frame).ffill()
        mark_close = aligned_mark["mark_close"].astype(float)
        mark_range = _safe_divide(aligned_mark["mark_high"] - aligned_mark["mark_low"], mark_close)
        spread = _safe_divide(mark_close - close, close)
        feature_frame["fut_mark_spread_pct"] = spread
        feature_frame["fut_mark_spread_z"] = _rolling_zscore(spread, rolling_window)
        feature_frame["fut_mark_range_pct"] = mark_range
        feature_frame["fut_mark_return_diff"] = mark_close.pct_change() - base_return

    premium_frame = futures_context.get("premium_index")
    if premium_frame is not None and not premium_frame.empty:
        aligned_premium = _asof_reindex(base_data.index, premium_frame).ffill()
        premium_close = aligned_premium["premium_close"].astype(float)
        feature_frame["fut_premium_close"] = premium_close
        feature_frame["fut_premium_change"] = premium_close.diff()
        feature_frame["fut_premium_abs"] = premium_close.abs()
        feature_frame["fut_premium_z"] = _rolling_zscore(premium_close, rolling_window)

    funding_frame = futures_context.get("funding")
    if funding_frame is not None and not funding_frame.empty:
        aligned_funding = _asof_reindex(base_data.index, funding_frame).ffill()
        funding_rate = aligned_funding["funding_rate"].astype(float).fillna(0.0)
        funding_window = max(4, min(rolling_window, 24))
        feature_frame["fut_funding_rate"] = funding_rate
        feature_frame["fut_funding_abs"] = funding_rate.abs()
        feature_frame["fut_funding_mean"] = funding_rate.rolling(funding_window, min_periods=1).mean()
        feature_frame["fut_funding_z"] = _rolling_zscore(funding_rate, funding_window)
        feature_frame["fut_funding_sign"] = np.sign(funding_rate)

    recent_specs = [
        ("open_interest", {"sumOpenInterestValue": "fut_oi_value", "sumOpenInterest": "fut_oi_contracts"}),
        ("taker_flow", {"buySellRatio": "fut_taker_buy_sell_ratio", "buyVol": "fut_taker_buy_vol", "sellVol": "fut_taker_sell_vol"}),
        ("global_long_short", {"longShortRatio": "fut_global_ls_ratio", "longAccount": "fut_global_long_account", "shortAccount": "fut_global_short_account"}),
        ("basis", {"basisRate": "fut_basis_rate", "basis": "fut_basis_value", "futuresPrice": "fut_basis_futures_price", "indexPrice": "fut_basis_index_price"}),
    ]
    for key, rename_map in recent_specs:
        recent_frame = futures_context.get(key)
        if recent_frame is None or recent_frame.empty:
            continue
        aligned_recent = _asof_reindex(base_data.index, recent_frame)
        if _coverage_ratio(aligned_recent[list(set(rename_map) & set(aligned_recent.columns))]) < min_recent_coverage:
            continue
        for source_column, target_column in rename_map.items():
            if source_column not in aligned_recent.columns:
                continue
            series = aligned_recent[source_column].astype(float).ffill()
            feature_frame[target_column] = series
            feature_frame[f"{target_column}_change"] = series.pct_change().replace([np.inf, -np.inf], np.nan)
            feature_frame[f"{target_column}_z"] = _rolling_zscore(series, rolling_window)

    if "fut_taker_buy_vol" in feature_frame.columns and "fut_taker_sell_vol" in feature_frame.columns:
        total_taker = feature_frame["fut_taker_buy_vol"] + feature_frame["fut_taker_sell_vol"]
        feature_frame["fut_taker_imbalance"] = _safe_divide(
            feature_frame["fut_taker_buy_vol"] - feature_frame["fut_taker_sell_vol"],
            total_taker,
        )

    if "fut_oi_value_change" in feature_frame.columns:
        feature_frame["fut_oi_price_divergence"] = feature_frame["fut_oi_value_change"] - base_return

    if "fut_global_long_account" in feature_frame.columns and "fut_global_short_account" in feature_frame.columns:
        feature_frame["fut_global_account_spread"] = (
            feature_frame["fut_global_long_account"] - feature_frame["fut_global_short_account"]
        )

    if "fut_basis_futures_price" in feature_frame.columns and "fut_basis_index_price" in feature_frame.columns:
        feature_frame["fut_basis_spread_pct"] = _safe_divide(
            feature_frame["fut_basis_futures_price"] - feature_frame["fut_basis_index_price"],
            feature_frame["fut_basis_index_price"],
        )

    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    laggable_columns = list(feature_frame.columns)
    return FeatureBlock(frame=feature_frame, laggable_columns=laggable_columns, block_name="futures_context")


def build_cross_asset_context_feature_block(base_data, context_symbol_data, rolling_window=20):
    """Build a compact cross-asset context block from a small spot leader basket."""
    if not context_symbol_data:
        return _empty_feature_block(base_data.index, block_name="cross_asset_context")

    base_close = base_data["close"].astype(float)
    base_return = base_close.pct_change()
    feature_frame = pd.DataFrame(index=base_data.index)
    basket_returns = []
    basket_medium_returns = []
    basket_vols = []

    medium_horizon = max(6, min(24, rolling_window))
    for symbol, data in context_symbol_data.items():
        if data is None or data.empty:
            continue

        prefix = _slug_symbol(symbol)
        aligned = data.reindex(base_data.index).ffill()
        close = aligned["close"].astype(float)
        volume = aligned["volume"].astype(float)
        ret_1 = close.pct_change()
        ret_medium = close.pct_change(medium_horizon)
        vol_rolling = ret_1.rolling(rolling_window, min_periods=max(2, rolling_window // 2)).std()
        feature_frame[f"ctx_{prefix}_ret_1"] = ret_1
        feature_frame[f"ctx_{prefix}_ret_{medium_horizon}"] = ret_medium
        feature_frame[f"ctx_{prefix}_vol"] = vol_rolling
        feature_frame[f"ctx_{prefix}_rel_ret_1"] = ret_1 - base_return
        feature_frame[f"ctx_{prefix}_volume_z"] = _rolling_zscore(volume, rolling_window)
        basket_returns.append(ret_1.rename(symbol))
        basket_medium_returns.append(ret_medium.rename(symbol))
        basket_vols.append(vol_rolling.rename(symbol))

    if basket_returns:
        returns_frame = pd.concat(basket_returns, axis=1)
        feature_frame["ctx_basket_ret_1_mean"] = returns_frame.mean(axis=1)
        feature_frame["ctx_basket_ret_1_dispersion"] = returns_frame.std(axis=1)
        feature_frame["ctx_basket_breadth_positive"] = returns_frame.gt(0).mean(axis=1)

    if basket_medium_returns:
        medium_frame = pd.concat(basket_medium_returns, axis=1)
        feature_frame[f"ctx_basket_ret_{medium_horizon}_mean"] = medium_frame.mean(axis=1)

    if basket_vols:
        vol_frame = pd.concat(basket_vols, axis=1)
        feature_frame["ctx_basket_vol_mean"] = vol_frame.mean(axis=1)

    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    laggable_columns = list(feature_frame.columns)
    return FeatureBlock(frame=feature_frame, laggable_columns=laggable_columns, block_name="cross_asset_context")


def build_multi_timeframe_context_feature_block(base_data, base_interval, timeframes=None, rolling_window=20):
    """Build higher-timeframe trend and volatility context features."""
    selected_timeframes = list(timeframes or [])
    if not selected_timeframes:
        return _empty_feature_block(base_data.index, block_name="multi_timeframe")

    base_delta = _interval_timedelta(base_interval)
    ohlcv = base_data[["open", "high", "low", "close", "volume"]].astype(float)
    feature_frame = pd.DataFrame(index=base_data.index)

    for timeframe in selected_timeframes:
        timeframe_delta = _interval_timedelta(timeframe)
        if base_delta is not None and timeframe_delta is not None and timeframe_delta <= base_delta:
            continue

        resampled = ohlcv.resample(
            _interval_to_pandas_freq(timeframe),
            label="right",
            closed="left",
        ).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        ).dropna()
        if resampled.empty:
            continue

        tf_prefix = timeframe.lower()
        tf_close = resampled["close"]
        tf_return = tf_close.pct_change()
        tf_window = max(4, min(rolling_window, 8))
        rolling_high = resampled["high"].rolling(tf_window, min_periods=2).max().shift(1)
        rolling_low = resampled["low"].rolling(tf_window, min_periods=2).min().shift(1)

        tf_features = pd.DataFrame(index=resampled.index)
        tf_features[f"mtf_{tf_prefix}_ret_1"] = tf_return
        tf_features[f"mtf_{tf_prefix}_ret_3"] = tf_close.pct_change(3)
        tf_features[f"mtf_{tf_prefix}_trend"] = _safe_divide(
            tf_close - tf_close.rolling(tf_window, min_periods=2).mean(),
            tf_close,
        )
        tf_features[f"mtf_{tf_prefix}_range_pct"] = _safe_divide(resampled["high"] - resampled["low"], tf_close)
        tf_features[f"mtf_{tf_prefix}_vol"] = tf_return.rolling(tf_window, min_periods=2).std()
        tf_features[f"mtf_{tf_prefix}_close_location"] = _safe_divide(
            tf_close - resampled["low"],
            resampled["high"] - resampled["low"],
        ) - 0.5
        tf_features[f"mtf_{tf_prefix}_volume_z"] = _rolling_zscore(resampled["volume"], tf_window)
        tf_features[f"mtf_{tf_prefix}_breakout_up"] = _safe_divide(tf_close - rolling_high, rolling_high)
        tf_features[f"mtf_{tf_prefix}_breakout_down"] = _safe_divide(tf_close - rolling_low, rolling_low)

        aligned = _asof_reindex(base_data.index, tf_features).ffill()
        feature_frame = feature_frame.join(aligned)

    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    laggable_columns = list(feature_frame.columns)
    return FeatureBlock(frame=feature_frame, laggable_columns=laggable_columns, block_name="multi_timeframe")