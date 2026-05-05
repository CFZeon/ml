"""Helpers for indicator-local Binance USD-M derivatives context."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests


BINANCE_FAPI_BASE_URL = "https://fapi.binance.com"
FUNDING_RATE_LIMIT = 1000
OPEN_INTEREST_LIMIT = 500
OPEN_INTEREST_MAX_LOOKBACK = pd.Timedelta(days=30)
_PERIOD_PATTERN = re.compile(r"^(\d+)([mhd])$")


def ensure_utc_timestamp(value):
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def ensure_utc_index(index):
    normalized = pd.DatetimeIndex(index)
    if normalized.tz is None:
        normalized = normalized.tz_localize("UTC")
    else:
        normalized = normalized.tz_convert("UTC")
    if not normalized.is_monotonic_increasing:
        raise ValueError("Indicator context alignment requires a monotonic increasing DatetimeIndex")
    return normalized


def resolve_symbol(df, explicit_symbol=None):
    if explicit_symbol is not None:
        return str(explicit_symbol).upper()

    if "symbol" not in df.columns:
        raise ValueError("Derivatives indicators require an explicit symbol when the frame has no 'symbol' column")

    symbols = pd.Series(df["symbol"], copy=False).dropna().astype(str).str.upper().unique()
    if len(symbols) != 1:
        raise ValueError(f"Expected one symbol in frame, got {list(symbols)}")
    return str(symbols[0])


def live_open_interest_supported(start_dt, end_dt, *, now=None):
    now_ts = ensure_utc_timestamp(now or pd.Timestamp.now(tz="UTC"))
    start_ts = ensure_utc_timestamp(start_dt)
    end_ts = ensure_utc_timestamp(end_dt)
    earliest_supported = now_ts - OPEN_INTEREST_MAX_LOOKBACK
    return end_ts >= earliest_supported and start_ts < now_ts


def rolling_zscore(series, window, min_periods=3):
    min_periods = max(1, min(int(window), int(min_periods)))
    rolling_mean = series.rolling(window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window, min_periods=min_periods).std(ddof=0)
    zscore = (series - rolling_mean) / rolling_std.replace(0.0, np.nan)
    return zscore.replace([np.inf, -np.inf], np.nan)


def rolling_slope(series, window):
    lookback = max(int(window) - 1, 1)
    return (series - series.shift(lookback)) / float(lookback)


def period_to_timedelta(period):
    match = _PERIOD_PATTERN.match(str(period).strip())
    if match is None:
        raise ValueError(f"Unsupported period={period!r}")
    value = int(match.group(1))
    unit = match.group(2)
    if unit == "m":
        return pd.Timedelta(minutes=value)
    if unit == "h":
        return pd.Timedelta(hours=value)
    if unit == "d":
        return pd.Timedelta(days=value)
    raise ValueError(f"Unsupported period={period!r}")


def _json_default(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    return str(value)


def _cache_path(namespace, cache_dir, params):
    cache_root = Path(cache_dir or ".cache") / "binance_derivatives" / namespace
    cache_root.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(params, sort_keys=True, default=_json_default)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    symbol = str(params.get("symbol") or "unknown").lower()
    return cache_root / f"{symbol}_{digest}.csv"


def _read_cached_frame(path):
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="mixed")
    return frame


def _write_cached_frame(path, frame):
    serializable = frame.copy()
    if "timestamp" in serializable.columns:
        serializable["timestamp"] = pd.to_datetime(serializable["timestamp"], utc=True)
    serializable.to_csv(path, index=False)


def _fetch_paginated_rows(endpoint, params, *, start_dt, end_dt, limit, time_key, session=None):
    http = session or requests.Session()
    cursor_ms = int(ensure_utc_timestamp(start_dt).value // 1_000_000)
    end_ms = int(ensure_utc_timestamp(end_dt).value // 1_000_000)
    rows = []
    last_seen_ms = None

    while cursor_ms <= end_ms:
        query = dict(params)
        query.update({"startTime": cursor_ms, "endTime": end_ms, "limit": limit})
        response = http.get(f"{BINANCE_FAPI_BASE_URL}{endpoint}", params=query, timeout=30)
        response.raise_for_status()
        batch = response.json() or []
        if not batch:
            break

        rows.extend(batch)
        timestamps = [
            int(row[time_key])
            for row in batch
            if row.get(time_key) is not None
        ]
        if not timestamps:
            break

        batch_last_ms = max(timestamps)
        if batch_last_ms == last_seen_ms:
            break

        last_seen_ms = batch_last_ms
        cursor_ms = batch_last_ms + 1
        if len(batch) < limit:
            break

    return rows


def fetch_funding_history(symbol, start_dt, end_dt, *, cache_dir=".cache", session=None):
    start_ts = ensure_utc_timestamp(start_dt)
    end_ts = ensure_utc_timestamp(end_dt)
    cache_key = {
        "symbol": str(symbol).upper(),
        "start": start_ts,
        "end": end_ts,
        "endpoint": "funding_rate",
    }
    cache_path = _cache_path("funding_rate", cache_dir, cache_key)
    cached = _read_cached_frame(cache_path)
    if cached is not None:
        return cached

    rows = _fetch_paginated_rows(
        "/fapi/v1/fundingRate",
        {"symbol": str(symbol).upper()},
        start_dt=start_ts,
        end_dt=end_ts,
        limit=FUNDING_RATE_LIMIT,
        time_key="fundingTime",
        session=session,
    )
    if not rows:
        return pd.DataFrame(columns=["timestamp", "funding_rate", "funding_mark_price"])

    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["fundingTime"], unit="ms", utc=True)
    frame["funding_rate"] = pd.to_numeric(frame["fundingRate"], errors="coerce")
    frame["funding_mark_price"] = pd.to_numeric(frame.get("markPrice"), errors="coerce")
    output = (
        frame[["timestamp", "funding_rate", "funding_mark_price"]]
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    _write_cached_frame(cache_path, output)
    return output


def fetch_open_interest_history(symbol, period, start_dt, end_dt, *, cache_dir=".cache", session=None):
    start_ts = ensure_utc_timestamp(start_dt)
    end_ts = ensure_utc_timestamp(end_dt)
    recent_floor = ensure_utc_timestamp(pd.Timestamp.now(tz="UTC")) - OPEN_INTEREST_MAX_LOOKBACK
    if end_ts <= recent_floor:
        return pd.DataFrame(columns=["timestamp", "sumOpenInterest", "sumOpenInterestValue"])
    start_ts = max(start_ts, recent_floor)
    period_delta = period_to_timedelta(period)
    cache_key = {
        "symbol": str(symbol).upper(),
        "period": str(period),
        "start": start_ts,
        "end": end_ts,
        "endpoint": "open_interest",
    }
    cache_path = _cache_path("open_interest", cache_dir, cache_key)
    cached = _read_cached_frame(cache_path)
    if cached is not None:
        return cached

    rows = []
    chunk_span = min(
        OPEN_INTEREST_MAX_LOOKBACK - period_delta,
        period_delta * max(OPEN_INTEREST_LIMIT - 1, 1),
    )
    chunk_start = start_ts
    while chunk_start <= end_ts:
        chunk_end = min(chunk_start + chunk_span, end_ts)
        rows.extend(
            _fetch_paginated_rows(
                "/futures/data/openInterestHist",
                {"symbol": str(symbol).upper(), "period": str(period)},
                start_dt=chunk_start,
                end_dt=chunk_end,
                limit=OPEN_INTEREST_LIMIT,
                time_key="timestamp",
                session=session,
            )
        )
        chunk_start = chunk_end + period_delta
    if not rows:
        return pd.DataFrame(columns=["timestamp", "sumOpenInterest", "sumOpenInterestValue"])

    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame["sumOpenInterest"] = pd.to_numeric(frame["sumOpenInterest"], errors="coerce")
    frame["sumOpenInterestValue"] = pd.to_numeric(frame["sumOpenInterestValue"], errors="coerce")
    output = (
        frame[["timestamp", "sumOpenInterest", "sumOpenInterestValue"]]
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    _write_cached_frame(cache_path, output)
    return output


def align_context_frame(
    base_index,
    context_frame,
    *,
    value_columns,
    max_age=None,
    allow_exact_matches=True,
):
    raw_index = pd.DatetimeIndex(base_index)
    normalized_index = ensure_utc_index(raw_index)
    empty_aligned = pd.DataFrame(index=raw_index)
    for column in value_columns:
        empty_aligned[column] = np.nan
    empty_aligned["source_timestamp"] = pd.NaT
    empty_aligned["source_age"] = pd.NaT

    if context_frame is None or context_frame.empty:
        report = {
            "source_rows": 0,
            "matched_rows": 0,
            "coverage": 0.0,
            "stale_rows": 0,
            "max_age": str(max_age) if max_age is not None else None,
            "first_source_timestamp": None,
            "last_source_timestamp": None,
            "max_observed_age": None,
        }
        return empty_aligned, report

    context = pd.DataFrame(context_frame).copy()
    context["source_timestamp"] = pd.to_datetime(context["timestamp"], utc=True)
    context = context.sort_values("source_timestamp")
    anchor = pd.DataFrame({"decision_time": normalized_index})
    joined = pd.merge_asof(
        anchor,
        context[["source_timestamp", *value_columns]],
        left_on="decision_time",
        right_on="source_timestamp",
        direction="backward",
        allow_exact_matches=allow_exact_matches,
    )

    age = joined["decision_time"] - joined["source_timestamp"]
    stale_mask = pd.Series(False, index=joined.index, dtype=bool)
    max_age_td = pd.Timedelta(max_age) if max_age is not None else None
    if max_age_td is not None:
        stale_mask = joined["source_timestamp"].notna() & age.ge(max_age_td)
        if stale_mask.any():
            joined.loc[stale_mask, value_columns] = np.nan
            joined.loc[stale_mask, "source_timestamp"] = pd.NaT
            age.loc[stale_mask] = pd.NaT

    joined["source_age"] = age
    joined.index = raw_index
    coverage_mask = joined["source_timestamp"].notna()
    observed_age = joined.loc[coverage_mask, "source_age"] if coverage_mask.any() else pd.Series(dtype="timedelta64[ns]")
    report = {
        "source_rows": int(len(context)),
        "matched_rows": int(coverage_mask.sum()),
        "coverage": float(coverage_mask.mean()) if len(joined) else 0.0,
        "stale_rows": int(stale_mask.sum()),
        "max_age": str(max_age_td) if max_age_td is not None else None,
        "first_source_timestamp": context["source_timestamp"].min(),
        "last_source_timestamp": context["source_timestamp"].max(),
        "max_observed_age": observed_age.max() if not observed_age.empty else None,
    }
    return joined, report