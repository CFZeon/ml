"""Reference-overlay feature adapters and cross-venue validation helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .data import _interval_timedelta, _parse_bound, _parse_interval
from .features import FeatureBlock
from .storage import read_parquet_frame, write_parquet_frame


_COINBASE_BASE = "https://api.exchange.coinbase.com"
_KRAKEN_BASE = "https://api.kraken.com"
_BYBIT_BASE = "https://api.bybit.com"
_COINBASE_GRANULARITIES = (60, 300, 900, 3600, 21600, 86400)
_KRAKEN_INTERVALS = (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
_BYBIT_INTERVALS = ("1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M")
_QUOTE_SUFFIXES = ("USDT", "USDC", "BUSD", "FDUSD", "TUSD", "USD", "BTC", "ETH", "EUR")
_KRAKEN_ASSET_MAP = {
    "BTC": "XBT",
    "DOGE": "XDG",
}


def _empty_feature_block(index, block_name):
    return FeatureBlock(frame=pd.DataFrame(index=index), laggable_columns=[], block_name=block_name)


def _safe_divide(numerator, denominator):
    if isinstance(denominator, pd.Series):
        denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def _rolling_zscore(series, window):
    mean = series.rolling(window, min_periods=max(2, window // 2)).mean()
    std = series.rolling(window, min_periods=max(2, window // 2)).std().replace(0, np.nan)
    return (series - mean) / std


def _asof_reindex(base_index, frame):
    if frame is None or frame.empty:
        return pd.DataFrame(index=base_index)

    context = frame.sort_index().reset_index().rename(columns={frame.index.name or "index": "timestamp"})
    anchor = pd.DataFrame({"timestamp": pd.DatetimeIndex(base_index)})
    joined = pd.merge_asof(anchor, context, on="timestamp", direction="backward")
    return joined.set_index("timestamp").reindex(base_index)


def _cache_path(cache_dir, namespace, payload):
    if cache_dir is None:
        return None
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha1(encoded).hexdigest()
    return Path(cache_dir) / "reference_data" / namespace / f"{digest}.parquet"


def _read_cache(path):
    if path is None or not path.exists():
        return None
    return read_parquet_frame(path)


def _write_cache(path, frame):
    if path is None:
        return
    write_parquet_frame(path, pd.DataFrame(frame))


def _coverage_ratio(series):
    values = pd.Series(series)
    if values.empty:
        return 0.0
    return float(values.notna().mean())


def _interval_to_seconds(interval):
    value, unit = _parse_interval(interval)
    multipliers = {
        "s": 1,
        "m": 60,
        "h": 3600,
        "d": 86400,
        "w": 7 * 86400,
    }
    if unit not in multipliers:
        raise ValueError(f"Unsupported reference interval={interval!r}")
    return int(value) * multipliers[unit]


def _interval_to_pandas_freq(interval):
    value, unit = _parse_interval(interval)
    if unit == "s":
        return f"{value}s"
    if unit == "m":
        return f"{value}min"
    if unit == "h":
        return f"{value}h"
    if unit == "d":
        return f"{value}d"
    if unit == "w":
        return f"{value}w"
    raise ValueError(f"Unsupported reference interval={interval!r}")


def _drop_uncommitted_tail(frame, interval, *, always_drop_last=False):
    data = pd.DataFrame(frame).copy()
    if data.empty:
        return data
    if always_drop_last:
        return data.iloc[:-1].copy() if len(data) > 1 else data.iloc[:0].copy()

    interval_delta = _interval_timedelta(interval)
    if interval_delta is None:
        return data
    now = pd.Timestamp.now(tz="UTC")
    last_timestamp = pd.DatetimeIndex(data.index)[-1]
    if last_timestamp + interval_delta > now:
        return data.iloc[:-1].copy() if len(data) > 1 else data.iloc[:0].copy()
    return data


def _resample_ohlcv(frame, interval):
    data = pd.DataFrame(frame).copy()
    if data.empty:
        return data
    resampled = data.resample(_interval_to_pandas_freq(interval), label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return resampled.dropna(subset=["open", "high", "low", "close"], how="any")


def _split_symbol(symbol):
    normalized = str(symbol or "").upper().replace("-", "").replace("/", "")
    for suffix in _QUOTE_SUFFIXES:
        if normalized.endswith(suffix) and len(normalized) > len(suffix):
            return normalized[: -len(suffix)], suffix
    if len(normalized) > 3:
        return normalized[:-3], normalized[-3:]
    raise ValueError(f"Could not infer base/quote assets from symbol={symbol!r}")


def _normalize_reference_quote(quote_asset):
    quote_asset = str(quote_asset or "").upper()
    if quote_asset in {"USDT", "USDC", "BUSD", "FDUSD", "TUSD"}:
        return "USD"
    return quote_asset


def _coinbase_product_id(symbol):
    base_asset, quote_asset = _split_symbol(symbol)
    return f"{base_asset}-{_normalize_reference_quote(quote_asset)}"


def _kraken_pair(symbol):
    base_asset, quote_asset = _split_symbol(symbol)
    return f"{_KRAKEN_ASSET_MAP.get(base_asset, base_asset)}{_normalize_reference_quote(quote_asset)}"


def _select_supported_seconds(target_seconds, supported_seconds):
    eligible = [value for value in supported_seconds if value <= target_seconds and target_seconds % value == 0]
    if eligible:
        return max(eligible)
    smaller = [value for value in supported_seconds if value <= target_seconds]
    if smaller:
        return max(smaller)
    return min(supported_seconds)


def _select_coinbase_granularity(interval):
    target_seconds = _interval_to_seconds(interval)
    return _select_supported_seconds(target_seconds, _COINBASE_GRANULARITIES)


def _select_kraken_interval(interval):
    target_minutes = max(1, _interval_to_seconds(interval) // 60)
    return _select_supported_seconds(target_minutes, _KRAKEN_INTERVALS)


def _target_requires_resample(interval, base_seconds):
    return int(base_seconds) != _interval_to_seconds(interval)


def _normalize_ohlcv_frame(frame):
    data = pd.DataFrame(frame).copy()
    if data.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if not isinstance(data.index, pd.DatetimeIndex):
        if "timestamp" in data.columns:
            data.index = pd.to_datetime(data.pop("timestamp"), utc=True)
        else:
            raise ValueError("Reference frame requires a DatetimeIndex or timestamp column")
    elif data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    else:
        data.index = data.index.tz_convert("UTC")
    data = data.sort_index()
    for column in ["open", "high", "low", "close", "volume"]:
        if column not in data.columns:
            if column == "volume":
                data[column] = np.nan
            else:
                data[column] = data.get("close")
    return data[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")


def _normalize_prefixed_ohlc_frame(frame, *, prefix):
    data = pd.DataFrame(frame).copy()
    if data.empty:
        columns = [f"{prefix}_open", f"{prefix}_high", f"{prefix}_low", f"{prefix}_close"]
        return pd.DataFrame(columns=columns)
    if not isinstance(data.index, pd.DatetimeIndex):
        if "timestamp" in data.columns:
            data.index = pd.to_datetime(data.pop("timestamp"), utc=True)
        else:
            raise ValueError("Reference frame requires a DatetimeIndex or timestamp column")
    elif data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    else:
        data.index = data.index.tz_convert("UTC")
    data = data.sort_index()
    ordered = [f"{prefix}_open", f"{prefix}_high", f"{prefix}_low", f"{prefix}_close"]
    for column in ordered:
        if column not in data.columns:
            data[column] = np.nan
    return data[ordered].apply(pd.to_numeric, errors="coerce")


def _coerce_mapping(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    return {}


def _fetch_json(session, url, *, params=None, headers=None):
    response = session.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_coinbase_reference_bars(symbol="BTCUSDT", interval="1h", start=None, end=None, cache_dir=".cache", session=None):
    start_dt = _parse_bound(start) if start is not None else pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)
    end_dt = _parse_bound(end) if end is not None else pd.Timestamp.now(tz="UTC")
    granularity = _select_coinbase_granularity(interval)
    cache_file = _cache_path(
        cache_dir,
        namespace="coinbase_spot",
        payload={"symbol": symbol, "interval": interval, "start": start_dt, "end": end_dt, "granularity": granularity},
    )
    cached = _read_cache(cache_file)
    if cached is not None:
        return _normalize_ohlcv_frame(cached)

    own_session = session is None
    session = session or requests.Session()
    try:
        product_id = _coinbase_product_id(symbol)
        rows = []
        cursor = start_dt
        step = pd.Timedelta(seconds=granularity)
        max_window = step * 300
        while cursor < end_dt:
            window_end = min(end_dt, cursor + max_window)
            payload = _fetch_json(
                session,
                f"{_COINBASE_BASE}/products/{product_id}/candles",
                params={
                    "start": cursor.isoformat().replace("+00:00", "Z"),
                    "end": window_end.isoformat().replace("+00:00", "Z"),
                    "granularity": str(granularity),
                },
                headers={"Accept": "application/json"},
            )
            if payload:
                batch = pd.DataFrame(payload, columns=["time", "low", "high", "open", "close", "volume"])
                batch["timestamp"] = pd.to_datetime(batch["time"], unit="s", utc=True)
                rows.append(batch[["timestamp", "open", "high", "low", "close", "volume"]])
            cursor = window_end

        if rows:
            frame = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["timestamp"]).set_index("timestamp").sort_index()
        else:
            frame = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        frame = frame[(frame.index >= start_dt) & (frame.index < end_dt)]
        frame = _drop_uncommitted_tail(frame, interval)
        if _target_requires_resample(interval, granularity):
            frame = _resample_ohlcv(frame, interval)
        frame = frame[(frame.index >= start_dt) & (frame.index < end_dt)]
        _write_cache(cache_file, frame)
        return _normalize_ohlcv_frame(frame)
    finally:
        if own_session:
            session.close()


def fetch_kraken_reference_bars(symbol="BTCUSDT", interval="1h", start=None, end=None, cache_dir=".cache", session=None):
    start_dt = _parse_bound(start) if start is not None else pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)
    end_dt = _parse_bound(end) if end is not None else pd.Timestamp.now(tz="UTC")
    interval_minutes = _select_kraken_interval(interval)
    cache_file = _cache_path(
        cache_dir,
        namespace="kraken_spot",
        payload={"symbol": symbol, "interval": interval, "start": start_dt, "end": end_dt, "interval_minutes": interval_minutes},
    )
    cached = _read_cache(cache_file)
    if cached is not None:
        return _normalize_ohlcv_frame(cached)

    own_session = session is None
    session = session or requests.Session()
    try:
        payload = _fetch_json(
            session,
            f"{_KRAKEN_BASE}/0/public/OHLC",
            params={
                "pair": _kraken_pair(symbol),
                "interval": interval_minutes,
                "since": int(start_dt.timestamp()),
            },
        )
        result = dict(payload.get("result") or {})
        pair_key = next((key for key in result.keys() if key != "last"), None)
        rows = list(result.get(pair_key) or []) if pair_key is not None else []
        if rows:
            frame = pd.DataFrame(
                rows,
                columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"],
            )
            frame["timestamp"] = pd.to_datetime(frame["time"], unit="s", utc=True)
            frame = frame[["timestamp", "open", "high", "low", "close", "volume"]].set_index("timestamp").sort_index()
        else:
            frame = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        frame = frame[(frame.index >= start_dt) & (frame.index < end_dt)]
        frame = _drop_uncommitted_tail(frame, interval, always_drop_last=True)
        if _target_requires_resample(interval, int(interval_minutes) * 60):
            frame = _resample_ohlcv(frame, interval)
        frame = frame[(frame.index >= start_dt) & (frame.index < end_dt)]
        _write_cache(cache_file, frame)
        return _normalize_ohlcv_frame(frame)
    finally:
        if own_session:
            session.close()


def _normalize_bybit_interval(interval):
    seconds = _interval_to_seconds(interval)
    mapping = {
        60: "1",
        180: "3",
        300: "5",
        900: "15",
        1800: "30",
        3600: "60",
        7200: "120",
        14400: "240",
        21600: "360",
        43200: "720",
        86400: "D",
        7 * 86400: "W",
    }
    if seconds in mapping:
        return mapping[seconds]
    eligible = [value for value in _BYBIT_INTERVALS if value.isdigit() and int(value) * 60 <= seconds and seconds % (int(value) * 60) == 0]
    if eligible:
        return max(eligible, key=lambda value: int(value))
    return "60"


def _fetch_bybit_kline(path, symbol, interval, *, category="linear", start=None, end=None, cache_dir=".cache", session=None, namespace="bybit"):
    start_dt = _parse_bound(start) if start is not None else pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)
    end_dt = _parse_bound(end) if end is not None else pd.Timestamp.now(tz="UTC")
    bybit_interval = _normalize_bybit_interval(interval)
    cache_file = _cache_path(
        cache_dir,
        namespace=namespace,
        payload={"path": path, "symbol": symbol, "interval": interval, "start": start_dt, "end": end_dt, "category": category},
    )
    cached = _read_cache(cache_file)
    if cached is not None:
        return _normalize_ohlcv_frame(cached)

    own_session = session is None
    session = session or requests.Session()
    try:
        cursor_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
        rows = []
        while cursor_ms < end_ms:
            payload = _fetch_json(
                session,
                f"{_BYBIT_BASE}{path}",
                params={
                    "category": category,
                    "symbol": str(symbol).upper(),
                    "interval": bybit_interval,
                    "start": cursor_ms,
                    "end": end_ms,
                    "limit": 1000,
                },
            )
            batch = list(((payload.get("result") or {}).get("list") or []))
            if not batch:
                break
            rows.extend(batch)
            newest_timestamp = max(int(row[0]) for row in batch)
            if newest_timestamp < cursor_ms:
                break
            cursor_ms = newest_timestamp + 1
            if len(batch) == 1:
                break

        if rows:
            frame = pd.DataFrame(rows, columns=["startTime", "openPrice", "highPrice", "lowPrice", "closePrice"])
            frame["timestamp"] = pd.to_datetime(frame["startTime"].astype("int64"), unit="ms", utc=True)
            frame = frame.rename(
                columns={
                    "openPrice": "open",
                    "highPrice": "high",
                    "lowPrice": "low",
                    "closePrice": "close",
                }
            )
            frame["volume"] = np.nan
            frame = frame[["timestamp", "open", "high", "low", "close", "volume"]].drop_duplicates(subset=["timestamp"]).set_index("timestamp").sort_index()
        else:
            frame = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        frame = frame[(frame.index >= start_dt) & (frame.index < end_dt)]
        frame = _drop_uncommitted_tail(frame, interval)
        _write_cache(cache_file, frame)
        return _normalize_ohlcv_frame(frame)
    finally:
        if own_session:
            session.close()


def fetch_bybit_futures_reference(symbol="BTCUSDT", interval="1h", start=None, end=None, cache_dir=".cache", session=None, category="linear"):
    own_session = session is None
    session = session or requests.Session()
    try:
        return {
            "mark_price": _normalize_prefixed_ohlc_frame(
                _fetch_bybit_kline(
                    "/v5/market/mark-price-kline",
                    symbol,
                    interval,
                    category=category,
                    start=start,
                    end=end,
                    cache_dir=cache_dir,
                    session=session,
                    namespace="bybit_mark",
                ).rename(columns={
                    "open": "mark_open",
                    "high": "mark_high",
                    "low": "mark_low",
                    "close": "mark_close",
                }),
                prefix="mark",
            ),
            "index_price": _normalize_prefixed_ohlc_frame(
                _fetch_bybit_kline(
                    "/v5/market/index-price-kline",
                    symbol,
                    interval,
                    category=category,
                    start=start,
                    end=end,
                    cache_dir=cache_dir,
                    session=session,
                    namespace="bybit_index",
                ).rename(columns={
                    "open": "index_open",
                    "high": "index_high",
                    "low": "index_low",
                    "close": "index_close",
                }),
                prefix="index",
            ),
        }
    finally:
        if own_session:
            session.close()


def normalize_okx_reference_bundle(bundle=None):
    payload = _coerce_mapping(bundle)
    return {
        "mark_price": _normalize_prefixed_ohlc_frame(payload.get("mark_price"), prefix="mark"),
        "index_price": _normalize_prefixed_ohlc_frame(payload.get("index_price"), prefix="index"),
    }


def _resolve_reference_section(config, section_name):
    return dict((config or {}).get(section_name) or {})


def _series_metric_summary(series, *, threshold):
    values = pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {
            "available": False,
            "median": None,
            "max": None,
            "breach_share": None,
            "threshold": float(threshold),
            "passed": True,
        }
    breach_share = float((values > float(threshold)).mean())
    return {
        "available": True,
        "median": float(values.median()),
        "max": float(values.max()),
        "breach_share": breach_share,
        "threshold": float(threshold),
        "passed": breach_share <= 0.05,
    }


def _build_spot_overlay(base_data, venue_frames, *, composite_policy="liquidity_weighted", minimum_quorum=1):
    base_frame = pd.DataFrame(base_data).copy()
    close_matrix = pd.DataFrame(index=base_frame.index)
    volume_matrix = pd.DataFrame(index=base_frame.index)
    for venue, frame in dict(venue_frames or {}).items():
        normalized = _normalize_ohlcv_frame(frame)
        close_matrix[venue] = normalized.reindex(base_frame.index)["close"]
        volume_matrix[venue] = normalized.reindex(base_frame.index)["volume"]

    overlay = pd.DataFrame(index=base_frame.index)
    if not close_matrix.empty:
        composite_policy = str(composite_policy or "liquidity_weighted").lower()
        if composite_policy == "liquidity_weighted":
            weight_matrix = volume_matrix.where(volume_matrix > 0.0)
            weighted_close = close_matrix.mul(weight_matrix)
            weight_sums = weight_matrix.sum(axis=1, skipna=True)
            reference_close = weighted_close.sum(axis=1, skipna=True).div(weight_sums.replace(0.0, np.nan))
            reference_close = reference_close.where(weight_sums > 0.0, close_matrix.median(axis=1, skipna=True))
        elif composite_policy == "strict_quorum":
            venue_count = close_matrix.notna().sum(axis=1)
            reference_close = close_matrix.median(axis=1, skipna=True).where(venue_count >= int(max(1, minimum_quorum)))
        else:
            reference_close = close_matrix.median(axis=1, skipna=True)
        overlay["reference_price"] = reference_close
        overlay["reference_close"] = reference_close
        overlay["composite_price"] = reference_close
        breadth = close_matrix.pct_change().apply(np.sign).replace(0.0, np.nan).mean(axis=1)
        overlay["breadth"] = breadth.fillna(0.0)
    if not volume_matrix.empty:
        overlay["reference_volume"] = volume_matrix.median(axis=1, skipna=True)

    if overlay.empty or not overlay.notna().any(axis=0).any():
        return pd.DataFrame(index=base_frame.index)
    return overlay


def build_spot_reference_validation(
    base_data,
    *,
    symbol="BTCUSDT",
    interval="1h",
    start=None,
    end=None,
    cache_dir=".cache",
    config=None,
    session=None,
):
    base_frame = pd.DataFrame(base_data).copy()
    config = dict(config or {})
    section = _resolve_reference_section(config, "spot")
    venues = list(dict.fromkeys(section.get("venues") or config.get("spot_venues") or ["coinbase", "kraken"]))
    partial_mode = str(section.get("partial_coverage_mode", config.get("partial_coverage_mode", "allow"))).lower()
    divergence_mode = str(section.get("divergence_mode", config.get("divergence_mode", "advisory"))).lower()
    composite_policy = str(section.get("composite_policy", config.get("composite_policy", "liquidity_weighted"))).lower()
    minimum_quorum = int(section.get("minimum_quorum", config.get("minimum_quorum", 1)))
    min_coverage_ratio = float(section.get("min_coverage_ratio", config.get("min_coverage_ratio", 0.95)))
    max_price_divergence_bps = float(section.get("max_price_divergence_bps", config.get("max_price_divergence_bps", 75.0)))
    provided_frames = dict(section.get("frames") or config.get("spot_frames") or {})
    fetch_live = bool(section.get("fetch_live", config.get("fetch_live", True)))
    venue_frames = {}
    venue_details = {}
    warnings = []

    for venue in venues:
        frame = None
        error = None
        try:
            if venue in provided_frames:
                frame = _normalize_ohlcv_frame(provided_frames[venue])
            elif fetch_live:
                if venue == "coinbase":
                    frame = fetch_coinbase_reference_bars(symbol=symbol, interval=interval, start=start, end=end, cache_dir=cache_dir, session=session)
                elif venue == "kraken":
                    frame = fetch_kraken_reference_bars(symbol=symbol, interval=interval, start=start, end=end, cache_dir=cache_dir, session=session)
        except Exception as exc:
            error = str(exc)
            warnings.append(f"{venue}_fetch_failed")

        if frame is not None and not frame.empty:
            venue_frames[venue] = frame
        aligned_close = frame.reindex(base_frame.index)["close"] if frame is not None and not frame.empty else pd.Series(index=base_frame.index, dtype=float)
        venue_details[venue] = {
            "available": bool(frame is not None and not frame.empty),
            "rows": int(len(frame)) if frame is not None else 0,
            "coverage_ratio": _coverage_ratio(aligned_close),
            "last_timestamp": frame.index[-1].isoformat() if frame is not None and not frame.empty else None,
            "error": error,
        }

    overlay = _build_spot_overlay(
        base_frame,
        venue_frames,
        composite_policy=composite_policy,
        minimum_quorum=minimum_quorum,
    )
    reference_close = overlay.get("reference_price", pd.Series(index=base_frame.index, dtype=float))
    divergence_bps = (reference_close.sub(base_frame["close"].astype(float)).abs() / base_frame["close"].replace(0.0, np.nan).abs()) * 1e4
    divergence_summary = _series_metric_summary(divergence_bps, threshold=max_price_divergence_bps)
    full_cohort_available = bool(venues) and all(
        venue_details.get(venue, {}).get("coverage_ratio", 0.0) >= min_coverage_ratio
        for venue in venues
    )
    partial_coverage = not full_cohort_available
    severe_divergence = bool(full_cohort_available and not divergence_summary.get("passed", True))

    reasons = []
    if partial_coverage:
        reasons.append("partial_reference_coverage")
    if severe_divergence:
        reasons.append("spot_reference_divergence")

    promotion_pass = True
    if severe_divergence and divergence_mode == "blocking":
        promotion_pass = False

    warnings.extend(
        reason
        for reason in reasons
        if (reason == "partial_reference_coverage" and partial_mode in {"allow", "advisory"})
        or (reason == "spot_reference_divergence" and divergence_mode == "advisory")
    )

    if severe_divergence and divergence_mode == "blocking":
        gate_mode = "blocking"
    elif partial_coverage and partial_mode == "blocking":
        gate_mode = "blocking"
    elif reasons:
        gate_mode = "advisory"
    else:
        gate_mode = "advisory"

    report = {
        "kind": "spot_reference_validation",
        "requested_venues": venues,
        "partial_coverage_mode": partial_mode,
        "divergence_mode": divergence_mode,
        "composite_policy": composite_policy,
        "minimum_quorum": int(max(1, minimum_quorum)),
        "min_coverage_ratio": min_coverage_ratio,
        "full_cohort_available": full_cohort_available,
        "available_venue_count": int(sum(1 for details in venue_details.values() if details.get("available"))),
        "promotion_pass": promotion_pass,
        "gate_mode": gate_mode,
        "reasons": reasons,
        "warnings": warnings,
        "venues": venue_details,
        "divergence": divergence_summary,
        "overlay_columns": list(overlay.columns),
    }
    return {
        "overlay": overlay,
        "report": report,
        "venue_frames": venue_frames,
    }


def _normalize_futures_reference_inputs(*, symbol, interval, start, end, cache_dir, config=None, session=None):
    config = dict(config or {})
    section = _resolve_reference_section(config, "futures")
    venues = list(dict.fromkeys(section.get("venues") or config.get("futures_venues") or ["bybit"]))
    provided = dict(section.get("frames") or config.get("futures_frames") or {})
    fetch_live = bool(section.get("fetch_live", config.get("fetch_live", True)))
    category = str(section.get("bybit_category", config.get("bybit_category", "linear")))
    venue_bundles = {}
    venue_details = {}
    warnings = []

    for venue in venues:
        bundle = None
        error = None
        try:
            if venue in provided:
                if venue == "okx":
                    bundle = normalize_okx_reference_bundle(provided[venue])
                else:
                    raw_bundle = _coerce_mapping(provided[venue])
                    bundle = {
                        "mark_price": _normalize_prefixed_ohlc_frame(raw_bundle.get("mark_price"), prefix="mark"),
                        "index_price": _normalize_prefixed_ohlc_frame(raw_bundle.get("index_price"), prefix="index"),
                    }
            elif fetch_live and venue == "bybit":
                bundle = fetch_bybit_futures_reference(
                    symbol=symbol,
                    interval=interval,
                    start=start,
                    end=end,
                    cache_dir=cache_dir,
                    session=session,
                    category=category,
                )
        except Exception as exc:
            error = str(exc)
            warnings.append(f"{venue}_fetch_failed")

        bundle_payload = _coerce_mapping(bundle)
        mark_frame = bundle_payload.get("mark_price")
        index_frame = bundle_payload.get("index_price")
        available = bool(
            (mark_frame is not None and not pd.DataFrame(mark_frame).empty)
            or (index_frame is not None and not pd.DataFrame(index_frame).empty)
        )
        if available:
            venue_bundles[venue] = bundle_payload
        venue_details[venue] = {
            "available": available,
            "mark_rows": int(len(mark_frame)) if mark_frame is not None else 0,
            "index_rows": int(len(index_frame)) if index_frame is not None else 0,
            "error": error,
        }
    return venue_bundles, venue_details, warnings


def _build_futures_overlay(base_index, futures_context, external_bundles):
    overlay = pd.DataFrame(index=base_index)
    index_candidates = []
    basis_candidates = []

    futures_context = _coerce_mapping(futures_context)
    basis_source = futures_context.get("basis")
    basis_frame = pd.DataFrame(basis_source).copy() if basis_source is not None else pd.DataFrame()
    if not basis_frame.empty:
        if not isinstance(basis_frame.index, pd.DatetimeIndex):
            basis_frame.index = pd.to_datetime(basis_frame.index, utc=True)
        elif basis_frame.index.tz is None:
            basis_frame.index = basis_frame.index.tz_localize("UTC")
        else:
            basis_frame.index = basis_frame.index.tz_convert("UTC")
        basis_frame = basis_frame.sort_index()
        aligned_basis = _asof_reindex(base_index, basis_frame)
        if "indexPrice" in aligned_basis.columns:
            index_candidates.append(aligned_basis["indexPrice"].astype(float))
        if "basisRate" in aligned_basis.columns:
            basis_candidates.append(aligned_basis["basisRate"].astype(float))

    funding_source = futures_context.get("funding")
    funding_frame = pd.DataFrame(funding_source).copy() if funding_source is not None else pd.DataFrame()
    if not funding_frame.empty:
        if not isinstance(funding_frame.index, pd.DatetimeIndex):
            funding_frame.index = pd.to_datetime(funding_frame.index, utc=True)
        elif funding_frame.index.tz is None:
            funding_frame.index = funding_frame.index.tz_localize("UTC")
        else:
            funding_frame.index = funding_frame.index.tz_convert("UTC")
        funding_frame = funding_frame.sort_index()
        aligned_funding = _asof_reindex(base_index, funding_frame)
        if "funding_rate" in aligned_funding.columns:
            overlay["composite_funding_rate"] = aligned_funding["funding_rate"].astype(float)

    for bundle in dict(external_bundles or {}).values():
        bundle_payload = _coerce_mapping(bundle)
        index_frame = _normalize_prefixed_ohlc_frame(bundle_payload.get("index_price"), prefix="index")
        mark_frame = _normalize_prefixed_ohlc_frame(bundle_payload.get("mark_price"), prefix="mark")
        if not index_frame.empty:
            aligned_index = index_frame.reindex(base_index)
            index_candidates.append(aligned_index["index_close"].astype(float))
        if not mark_frame.empty and not index_frame.empty:
            aligned_mark = mark_frame.reindex(base_index)
            aligned_index = index_frame.reindex(base_index)
            basis_candidates.append(
                _safe_divide(
                    aligned_mark["mark_close"].astype(float) - aligned_index["index_close"].astype(float),
                    aligned_index["index_close"].astype(float),
                )
            )

    if index_candidates:
        index_matrix = pd.concat(index_candidates, axis=1)
        reference_price = index_matrix.median(axis=1, skipna=True)
        overlay["reference_price"] = reference_price
        overlay["reference_close"] = reference_price
        overlay["composite_price"] = reference_price
    if basis_candidates:
        basis_matrix = pd.concat(basis_candidates, axis=1)
        overlay["composite_basis"] = basis_matrix.median(axis=1, skipna=True)

    if overlay.empty or not overlay.notna().any(axis=0).any():
        return pd.DataFrame(index=base_index)
    return overlay


def build_futures_reference_validation(
    base_data,
    *,
    futures_context=None,
    symbol="BTCUSDT",
    interval="1h",
    start=None,
    end=None,
    cache_dir=".cache",
    config=None,
    session=None,
):
    base_frame = pd.DataFrame(base_data).copy()
    config = dict(config or {})
    section = _resolve_reference_section(config, "futures")
    max_trade_mark_gap_bps = float(section.get("max_trade_mark_gap_bps", config.get("max_trade_mark_gap_bps", 125.0)))
    max_mark_index_gap_bps = float(section.get("max_mark_index_gap_bps", config.get("max_mark_index_gap_bps", 125.0)))
    max_basis_error_bps = float(section.get("max_basis_error_bps", config.get("max_basis_error_bps", 25.0)))

    external_bundles, venue_details, warnings = _normalize_futures_reference_inputs(
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
        cache_dir=cache_dir,
        config=config,
        session=session,
    )

    futures_context = _coerce_mapping(futures_context)
    mark_frame = _normalize_prefixed_ohlc_frame(futures_context.get("mark_price"), prefix="mark")
    basis_source = futures_context.get("basis")
    basis_frame = pd.DataFrame(basis_source).copy() if basis_source is not None else pd.DataFrame()
    if not basis_frame.empty:
        if not isinstance(basis_frame.index, pd.DatetimeIndex):
            basis_frame.index = pd.to_datetime(basis_frame.index, utc=True)
        elif basis_frame.index.tz is None:
            basis_frame.index = basis_frame.index.tz_localize("UTC")
        else:
            basis_frame.index = basis_frame.index.tz_convert("UTC")
        basis_frame = basis_frame.sort_index()

    aligned_mark = mark_frame.reindex(base_frame.index) if not mark_frame.empty else pd.DataFrame(index=base_frame.index)
    aligned_basis = _asof_reindex(base_frame.index, basis_frame) if not basis_frame.empty else pd.DataFrame(index=base_frame.index)
    base_close = base_frame["close"].astype(float)

    trade_mark_gap_bps = pd.Series(index=base_frame.index, dtype=float)
    if "mark_close" in aligned_mark.columns:
        trade_mark_gap_bps = _safe_divide((aligned_mark["mark_close"].astype(float) - base_close).abs(), base_close.abs()) * 1e4

    mark_index_gap_bps = pd.Series(index=base_frame.index, dtype=float)
    if "mark_close" in aligned_mark.columns and "indexPrice" in aligned_basis.columns:
        mark_index_gap_bps = _safe_divide(
            (aligned_mark["mark_close"].astype(float) - aligned_basis["indexPrice"].astype(float)).abs(),
            aligned_basis["indexPrice"].astype(float).abs(),
        ) * 1e4

    basis_error_bps = pd.Series(index=base_frame.index, dtype=float)
    if {"basis", "futuresPrice", "indexPrice"}.issubset(aligned_basis.columns):
        reconstructed_basis = aligned_basis["futuresPrice"].astype(float) - aligned_basis["indexPrice"].astype(float)
        basis_error_bps = _safe_divide(
            (aligned_basis["basis"].astype(float) - reconstructed_basis).abs(),
            aligned_basis["indexPrice"].astype(float).abs(),
        ) * 1e4

    trade_mark_summary = _series_metric_summary(trade_mark_gap_bps, threshold=max_trade_mark_gap_bps)
    mark_index_summary = _series_metric_summary(mark_index_gap_bps, threshold=max_mark_index_gap_bps)
    basis_error_summary = _series_metric_summary(basis_error_bps, threshold=max_basis_error_bps)

    required_inputs_available = bool(trade_mark_summary.get("available") and mark_index_summary.get("available") and basis_error_summary.get("available"))
    self_consistency_pass = bool(
        required_inputs_available
        and trade_mark_summary.get("passed", True)
        and mark_index_summary.get("passed", True)
        and basis_error_summary.get("passed", True)
    )

    reasons = []
    if not required_inputs_available:
        reasons.append("missing_futures_self_consistency_inputs")
    elif not self_consistency_pass:
        reasons.append("futures_self_consistency_failed")

    external_comparison = {}
    for venue, bundle in external_bundles.items():
        venue_report = {
            "mark_rows": venue_details.get(venue, {}).get("mark_rows", 0),
            "index_rows": venue_details.get(venue, {}).get("index_rows", 0),
        }
        bundle_payload = _coerce_mapping(bundle)
        external_mark = _normalize_prefixed_ohlc_frame(bundle_payload.get("mark_price"), prefix="mark").reindex(base_frame.index)
        external_index = _normalize_prefixed_ohlc_frame(bundle_payload.get("index_price"), prefix="index").reindex(base_frame.index)
        if "mark_close" in aligned_mark.columns and "mark_close" in external_mark.columns:
            venue_report["mark_gap_bps"] = _series_metric_summary(
                _safe_divide(
                    (external_mark["mark_close"].astype(float) - aligned_mark["mark_close"].astype(float)).abs(),
                    aligned_mark["mark_close"].astype(float).abs(),
                ) * 1e4,
                threshold=max_mark_index_gap_bps,
            )
        if "indexPrice" in aligned_basis.columns and "index_close" in external_index.columns:
            venue_report["index_gap_bps"] = _series_metric_summary(
                _safe_divide(
                    (external_index["index_close"].astype(float) - aligned_basis["indexPrice"].astype(float)).abs(),
                    aligned_basis["indexPrice"].astype(float).abs(),
                ) * 1e4,
                threshold=max_mark_index_gap_bps,
            )
        external_comparison[venue] = venue_report

    overlay = _build_futures_overlay(base_frame.index, futures_context, external_bundles)
    report = {
        "kind": "futures_reference_validation",
        "requested_venues": list(venue_details.keys()),
        "promotion_pass": self_consistency_pass,
        "gate_mode": "blocking",
        "reasons": reasons,
        "warnings": warnings,
        "self_consistency": {
            "required_inputs_available": required_inputs_available,
            "trade_mark_gap_bps": trade_mark_summary,
            "mark_index_gap_bps": mark_index_summary,
            "basis_reconstruction_error_bps": basis_error_summary,
        },
        "external_comparison": external_comparison,
        "venues": venue_details,
        "overlay_columns": list(overlay.columns),
    }
    return {
        "overlay": overlay,
        "report": report,
        "venue_frames": external_bundles,
    }


def build_reference_validation_bundle(
    base_data,
    *,
    market="spot",
    symbol="BTCUSDT",
    interval="1h",
    start=None,
    end=None,
    cache_dir=".cache",
    futures_context=None,
    config=None,
    session=None,
):
    normalized_market = str(market or "spot").lower()
    if normalized_market == "spot":
        return build_spot_reference_validation(
            base_data,
            symbol=symbol,
            interval=interval,
            start=start,
            end=end,
            cache_dir=cache_dir,
            config=config,
            session=session,
        )
    return build_futures_reference_validation(
        base_data,
        futures_context=futures_context,
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
        cache_dir=cache_dir,
        config=config,
        session=session,
    )


def build_reference_overlay_feature_block(base_data, reference_data=None, rolling_window=20):
    base_frame = pd.DataFrame(base_data)
    if reference_data is None:
        return _empty_feature_block(base_frame.index, "reference_overlay")

    reference_frame = pd.DataFrame(reference_data)
    if reference_frame.empty:
        return _empty_feature_block(base_frame.index, "reference_overlay")

    aligned = _asof_reindex(base_frame.index, reference_frame)
    frame = pd.DataFrame(index=base_frame.index)

    reference_price = None
    for column in ["reference_price", "reference_close", "composite_price"]:
        if column in aligned.columns:
            reference_price = aligned[column].astype(float)
            break
    if reference_price is not None and "close" in base_frame.columns:
        base_close = base_frame["close"].astype(float)
        gap = _safe_divide(reference_price - base_close, base_close)
        frame["ref_price_gap"] = gap
        frame["ref_price_gap_z"] = _rolling_zscore(gap, rolling_window)

    if "reference_volume" in aligned.columns and "volume" in base_frame.columns:
        ref_volume = aligned["reference_volume"].astype(float)
        frame["ref_volume_ratio"] = _safe_divide(ref_volume, base_frame["volume"].astype(float))
        frame["ref_volume_ratio_z"] = _rolling_zscore(frame["ref_volume_ratio"], rolling_window)

    if "breadth" in aligned.columns:
        frame["ref_breadth"] = aligned["breadth"].astype(float)
        frame["ref_breadth_z"] = _rolling_zscore(frame["ref_breadth"], rolling_window)

    if "composite_funding_rate" in aligned.columns:
        frame["composite_funding_rate"] = aligned["composite_funding_rate"].astype(float)
        frame["composite_funding_z"] = _rolling_zscore(frame["composite_funding_rate"], rolling_window)

    if "composite_basis" in aligned.columns:
        frame["composite_basis"] = aligned["composite_basis"].astype(float)
        frame["composite_basis_z"] = _rolling_zscore(frame["composite_basis"], rolling_window)

    if frame.empty:
        return _empty_feature_block(base_frame.index, "reference_overlay")

    laggable_columns = list(frame.columns)
    return FeatureBlock(frame=frame, laggable_columns=laggable_columns, block_name="reference_overlay")


__all__ = [
    "build_futures_reference_validation",
    "build_reference_overlay_feature_block",
    "build_reference_validation_bundle",
    "build_spot_reference_validation",
    "fetch_bybit_futures_reference",
    "fetch_coinbase_reference_bars",
    "fetch_kraken_reference_bars",
    "normalize_okx_reference_bundle",
]