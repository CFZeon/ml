"""Fetch OHLCV data from Binance Vision public archives.

Binance Vision spot klines are ZIP-compressed CSV files with 12 columns matching
the `/api/v3/klines` payload: open time, OHLCV, close time, quote volume,
trade count, taker buy base volume, taker buy quote volume, and an ignored
trailing field. Remote archives live under both `monthly/klines/...` and
`daily/klines/...`; local cache files here are periodized by month for `1h+`
intervals and ISO week for sub-hour intervals.
"""

import io
import hashlib
import hmac
import json
import os
import random
import re
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests

from .data_contracts import validate_custom_source_contract, validate_market_frame_contract
from .storage import read_json, read_parquet_frame, write_json, write_parquet_frame

_VISION_BASES = {
    "spot": "https://data.binance.vision/data/spot",
    "um_futures": "https://data.binance.vision/data/futures/um",
    "cm_futures": "https://data.binance.vision/data/futures/cm",
}
_REST_BASES = {
    "spot": "https://api.binance.com",
    "um_futures": "https://fapi.binance.com",
    "cm_futures": "https://dapi.binance.com",
}
_INTERVAL_PATTERN = re.compile(r"^(\d+)(mo|[smhdw])$")

_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base_vol", "taker_buy_quote_vol", "ignore",
]

_FLOAT_COLUMNS = [
    "open", "high", "low", "close", "volume", "quote_volume",
    "taker_buy_base_vol", "taker_buy_quote_vol",
]
_OPTIONAL_FLOAT_COLUMNS = ["taker_buy_base_vol", "taker_buy_quote_vol"]
_OUTPUT_COLUMNS = [
    "open", "high", "low", "close", "volume", "quote_volume", "trades",
    "taker_buy_base_vol", "taker_buy_quote_vol",
]
_DEFAULT_RETRY_POLICY = {
    "max_retries": 3,
    "backoff_factor": 0.5,
    "backoff_max": 8.0,
    "backoff_jitter": 0.1,
    "retry_statuses": {429, 500, 502, 503, 504},
    "retry_after_max": 30.0,
    "timeout": 30,
}
_VALID_GAP_POLICIES = {"fail", "warn", "flag", "drop_windows"}
_VALID_DUPLICATE_POLICIES = {"fail", "warn", "flag"}


@dataclass(frozen=True)
class CustomDataset:
    name: str
    frame: pd.DataFrame
    availability_column: str
    source_path: str | None = None
    availability_is_assumed: bool = False
    default_allow_exact_matches: bool = False
    max_feature_age: pd.Timedelta | None = None
    value_columns: tuple[str, ...] = ()
    dataset_manifest: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CachePeriod:
    kind: str
    key: str
    start: pd.Timestamp
    end: pd.Timestamp


def _parse_bound(value):
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _normalize_market(market="spot", futures_type=None):
    normalized = (market or "spot").lower()
    if normalized in {"spot", "cash"}:
        return "spot"
    if normalized in {"um", "usdm", "usdtm", "um_futures", "futures", "futures_um"}:
        return "um_futures"
    if normalized in {"cm", "coinm", "cm_futures", "futures_cm"}:
        return "cm_futures"
    if futures_type is not None:
        return _normalize_market(f"{futures_type}_futures")
    raise ValueError(f"Unsupported Binance market={market!r}")


def _vision_base_url(market):
    normalized = _normalize_market(market)
    return _VISION_BASES[normalized]


def _rest_base_url(market):
    normalized = _normalize_market(market)
    return _REST_BASES[normalized]


def _parse_interval(interval):
    match = _INTERVAL_PATTERN.fullmatch(interval)
    if match is None:
        raise ValueError(f"Unsupported Binance interval={interval!r}")
    return int(match.group(1)), match.group(2)


def _uses_weekly_cache(interval):
    value, unit = _parse_interval(interval)
    if unit == "s":
        return True
    if unit == "m":
        return value < 60
    return False


def _interval_timedelta(interval):
    value, unit = _parse_interval(interval)
    if unit == "s":
        return pd.Timedelta(seconds=value)
    if unit == "m":
        return pd.Timedelta(minutes=value)
    if unit == "h":
        return pd.Timedelta(hours=value)
    if unit == "d":
        return pd.Timedelta(days=value)
    if unit == "w":
        return pd.Timedelta(weeks=value)
    return None


def _expected_index(start, end, interval):
    if end <= start:
        return pd.DatetimeIndex([], tz="UTC")

    delta = _interval_timedelta(interval)
    if delta is not None:
        return pd.date_range(start=start, end=end, freq=delta, inclusive="left")

    value, _ = _parse_interval(interval)
    return pd.date_range(start=start, end=end, freq=f"{value}MS", inclusive="left")


def _iter_monthly_periods(start, end):
    current = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")

    while current < end:
        next_period = current + pd.offsets.MonthBegin(1)
        yield CachePeriod(
            kind="monthly",
            key=current.strftime("%Y-%m"),
            start=current,
            end=next_period,
        )
        current = next_period


def _iter_weekly_periods(start, end):
    current = start.normalize() - pd.Timedelta(days=start.weekday())

    while current < end:
        iso_year, iso_week, _ = current.isocalendar()
        next_period = current + pd.Timedelta(days=7)
        yield CachePeriod(
            kind="weekly",
            key=f"{iso_year}-W{iso_week:02d}",
            start=current,
            end=next_period,
        )
        current = next_period


def _iter_cache_periods(start, end, interval):
    if _uses_weekly_cache(interval):
        yield from _iter_weekly_periods(start, end)
        return
    yield from _iter_monthly_periods(start, end)


def _cache_path(cache_dir, symbol, interval, period, market="spot"):
    market_key = _normalize_market(market)
    return (
        Path(cache_dir)
        / market_key
        / "klines"
        / symbol
        / interval
        / period.kind
        / f"{period.key}.parquet"
    )


def _read_cache(path):
    if not path.exists():
        return None
    return _normalize_output_schema(read_parquet_frame(path))


def _write_cache(path, frame):
    write_parquet_frame(path, pd.DataFrame(frame))


def _read_object_cache(path):
    if path is None:
        return None
    return read_json(path)


def _write_object_cache(path, payload):
    if path is None:
        return
    write_json(path, payload)


def _normalize_output_schema(frame):
    if frame is None:
        return None

    normalized = frame.copy()
    # Older cache files were written before taker-side volumes became part of
    # the canonical output schema, so backfill those optional columns as zeros.
    for column in _OPTIONAL_FLOAT_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = 0.0
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").fillna(0.0).astype(float)

    return normalized


def _empty_duplicate_report():
    return {
        "status": "clean",
        "exact_duplicate_timestamps": 0,
        "exact_duplicate_rows": 0,
        "conflicting_duplicate_timestamps": 0,
        "conflicting_duplicate_rows": 0,
        "exact_timestamps": [],
        "conflicting_timestamps": [],
    }


def _finalize_duplicate_report(report):
    payload = dict(_empty_duplicate_report())
    payload.update(dict(report or {}))
    exact_count = int(payload.get("exact_duplicate_timestamps", 0))
    conflict_count = int(payload.get("conflicting_duplicate_timestamps", 0))
    if conflict_count > 0:
        payload["status"] = "conflicts_detected"
    elif exact_count > 0:
        payload["status"] = "exact_duplicates_deduplicated"
    else:
        payload["status"] = "clean"
    return payload


def _combine_duplicate_reports(*reports):
    combined = _empty_duplicate_report()
    for report in reports:
        if not report:
            continue
        payload = dict(report)
        combined["exact_duplicate_timestamps"] += int(payload.get("exact_duplicate_timestamps", 0))
        combined["exact_duplicate_rows"] += int(payload.get("exact_duplicate_rows", 0))
        combined["conflicting_duplicate_timestamps"] += int(payload.get("conflicting_duplicate_timestamps", 0))
        combined["conflicting_duplicate_rows"] += int(payload.get("conflicting_duplicate_rows", 0))
        combined["exact_timestamps"].extend(list(payload.get("exact_timestamps", [])))
        combined["conflicting_timestamps"].extend(list(payload.get("conflicting_timestamps", [])))
    return _finalize_duplicate_report(combined)


def _deduplicate_timestamp_index(frame):
    if frame is None or frame.empty:
        empty = pd.DataFrame(columns=_OUTPUT_COLUMNS)
        empty.attrs["duplicate_report"] = _empty_duplicate_report()
        return empty

    working = pd.DataFrame(frame).copy().sort_index()
    duplicate_mask = working.index.duplicated(keep=False)
    if not duplicate_mask.any():
        working.attrs["duplicate_report"] = _empty_duplicate_report()
        return working

    deduplicated_rows = []
    report = _empty_duplicate_report()
    for _, rows in working.groupby(level=0, sort=True):
        timestamp = rows.index[0]
        deduplicated_rows.append(rows.iloc[[0]])
        if len(rows) <= 1:
            continue

        comparison = rows[_OUTPUT_COLUMNS] if set(_OUTPUT_COLUMNS).issubset(rows.columns) else rows
        is_conflict = bool((comparison.nunique(dropna=False) > 1).any())
        if is_conflict:
            report["conflicting_duplicate_timestamps"] += 1
            report["conflicting_duplicate_rows"] += int(len(rows) - 1)
            report["conflicting_timestamps"].append(timestamp)
        else:
            report["exact_duplicate_timestamps"] += 1
            report["exact_duplicate_rows"] += int(len(rows) - 1)
            report["exact_timestamps"].append(timestamp)

    deduplicated = pd.concat(deduplicated_rows, axis=0).sort_index()
    deduplicated.attrs["duplicate_report"] = _finalize_duplicate_report(report)
    return deduplicated


def _infer_timestamp_unit(raw_times):
    max_value = pd.to_numeric(raw_times, errors="raise").abs().max()
    return "us" if max_value >= 10**14 else "ms"


def _prepare_frame(frame):
    if frame is None or frame.empty:
        empty = pd.DataFrame(columns=_OUTPUT_COLUMNS)
        empty.attrs["duplicate_report"] = _empty_duplicate_report()
        return empty

    prepared = frame.copy()
    if not prepared.empty and str(prepared.iloc[0]["open_time"]).lower() == "open_time":
        prepared = prepared.iloc[1:].copy()
    prepared["open_time"] = pd.to_numeric(prepared["open_time"], errors="raise")
    prepared["timestamp"] = pd.to_datetime(
        prepared["open_time"],
        unit=_infer_timestamp_unit(prepared["open_time"]),
        utc=True,
    )
    prepared = prepared.set_index("timestamp").sort_index()
    prepared = _normalize_output_schema(prepared)

    for column in _FLOAT_COLUMNS:
        prepared[column] = prepared[column].astype(float)
    prepared["trades"] = prepared["trades"].astype(int)
    prepared = prepared[_OUTPUT_COLUMNS]
    return _deduplicate_timestamp_index(prepared)


def _merge_frames(frames):
    valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid_frames:
        empty = pd.DataFrame(columns=_OUTPUT_COLUMNS)
        empty.attrs["duplicate_report"] = _empty_duplicate_report()
        return empty

    merged = pd.concat([_normalize_output_schema(frame) for frame in valid_frames]).sort_index()
    merged = _deduplicate_timestamp_index(merged[_OUTPUT_COLUMNS])
    merged.attrs["duplicate_report"] = _combine_duplicate_reports(
        *(dict(getattr(frame, "attrs", {}).get("duplicate_report") or {}) for frame in valid_frames),
        merged.attrs.get("duplicate_report"),
    )
    return merged[_OUTPUT_COLUMNS]


def _period_window(period, request_start, request_end):
    return max(period.start, request_start), min(period.end, request_end)


def _has_all_expected_rows(frame, interval, start, end):
    if end <= start:
        return True
    if frame is None or frame.empty:
        return False

    expected = _expected_index(start, end, interval)
    if expected.empty:
        return True

    window = frame[(frame.index >= start) & (frame.index < end)]
    return expected.difference(window.index).empty


def _resolve_retry_policy(retry_policy=None):
    resolved = {**_DEFAULT_RETRY_POLICY, **dict(retry_policy or {})}
    resolved["max_retries"] = max(0, int(resolved.get("max_retries", 0)))
    resolved["backoff_factor"] = max(0.0, float(resolved.get("backoff_factor", 0.0)))
    resolved["backoff_max"] = max(0.0, float(resolved.get("backoff_max", 0.0)))
    resolved["backoff_jitter"] = max(0.0, float(resolved.get("backoff_jitter", 0.0)))
    resolved["retry_after_max"] = max(0.0, float(resolved.get("retry_after_max", 0.0)))
    resolved["retry_statuses"] = {int(status) for status in resolved.get("retry_statuses", set())}
    resolved["timeout"] = float(resolved.get("timeout", 30))
    return resolved


def _parse_retry_after_seconds(response, retry_after_max):
    header_value = response.headers.get("Retry-After") if response is not None else None
    if header_value in (None, ""):
        return None
    try:
        return min(float(header_value), float(retry_after_max))
    except (TypeError, ValueError):
        return None


def _retry_delay_seconds(attempt_number, retry_policy, response=None):
    retry_after = _parse_retry_after_seconds(response, retry_policy["retry_after_max"])
    if retry_after is not None:
        return retry_after
    if attempt_number <= 1:
        base_delay = 0.0
    else:
        base_delay = retry_policy["backoff_factor"] * (2 ** (attempt_number - 2))
    if retry_policy["backoff_jitter"] > 0.0:
        base_delay += random.uniform(0.0, retry_policy["backoff_jitter"])
    return min(base_delay, retry_policy["backoff_max"])


def _missing_segments(missing_index, interval):
    missing_index = pd.DatetimeIndex(missing_index)
    if missing_index.empty:
        return []

    delta = _interval_timedelta(interval)
    segments = []
    start = missing_index[0]
    previous = missing_index[0]
    count = 1

    for timestamp in missing_index[1:]:
        contiguous = delta is not None and timestamp - previous == delta
        if contiguous:
            previous = timestamp
            count += 1
            continue
        segments.append({"start": start, "end": previous, "count": count})
        start = timestamp
        previous = timestamp
        count = 1

    segments.append({"start": start, "end": previous, "count": count})
    return segments


def _download_archive(url, session, retry_policy=None):
    retry_policy = _resolve_retry_policy(retry_policy)
    attempt = 0

    while True:
        attempt += 1
        print(f"  Fetching {url} (attempt {attempt}) ...")
        try:
            response = session.get(url, timeout=retry_policy["timeout"])
        except requests.RequestException as exc:
            if attempt > retry_policy["max_retries"] + 1:
                raise
            delay = _retry_delay_seconds(attempt, retry_policy)
            if delay > 0.0:
                time.sleep(delay)
            continue

        if response.status_code == 404:
            return None, {
                "url": url,
                "status": "not_found",
                "http_status": 404,
                "attempts": attempt,
                "retry_count": max(0, attempt - 1),
            }
        if response.status_code in retry_policy["retry_statuses"] and attempt <= retry_policy["max_retries"]:
            delay = _retry_delay_seconds(attempt, retry_policy, response=response)
            if delay > 0.0:
                time.sleep(delay)
            continue
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            with zf.open(zf.namelist()[0]) as handle:
                raw = pd.read_csv(handle, header=None, names=_COLUMNS)
        return _prepare_frame(raw), {
            "url": url,
            "status": "downloaded",
            "http_status": int(response.status_code),
            "attempts": attempt,
            "retry_count": max(0, attempt - 1),
        }


def _monthly_archive_url(symbol, interval, period, market="spot"):
    base_url = _vision_base_url(market)
    return (
        f"{base_url}/monthly/klines/{symbol}/{interval}/"
        f"{symbol}-{interval}-{period.start:%Y-%m}.zip"
    )


def _daily_archive_url(symbol, interval, day, market="spot"):
    base_url = _vision_base_url(market)
    return (
        f"{base_url}/daily/klines/{symbol}/{interval}/"
        f"{symbol}-{interval}-{day:%Y-%m-%d}.zip"
    )


def _fetch_daily_range(symbol, interval, period, session, market="spot", retry_policy=None):
    frames = []
    download_reports = []
    available_end = min(period.end, pd.Timestamp.now(tz="UTC").normalize())
    for day in pd.date_range(period.start, available_end, freq="1D", inclusive="left"):
        frame, report = _download_archive(
            _daily_archive_url(symbol, interval, day, market=market),
            session,
            retry_policy=retry_policy,
        )
        download_reports.append(report)
        if frame is not None and not frame.empty:
            frames.append(frame)
    return _merge_frames(frames), {
        "source": "daily",
        "downloads": download_reports,
        "retry_count": sum(report.get("retry_count", 0) for report in download_reports),
    }


def _fetch_period(symbol, interval, period, session, market="spot", retry_policy=None):
    if not _uses_weekly_cache(interval):
        monthly_frame, monthly_report = _download_archive(
            _monthly_archive_url(symbol, interval, period, market=market),
            session,
            retry_policy=retry_policy,
        )
        if monthly_frame is not None:
            return monthly_frame, {
                "source": "monthly",
                "downloads": [monthly_report],
                "retry_count": monthly_report.get("retry_count", 0),
            }

        daily_frame, daily_report = _fetch_daily_range(
            symbol,
            interval,
            period,
            session,
            market=market,
            retry_policy=retry_policy,
        )
        return daily_frame, {
            "source": "daily_fallback",
            "downloads": [monthly_report, *daily_report["downloads"]],
            "retry_count": monthly_report.get("retry_count", 0) + daily_report.get("retry_count", 0),
        }

    return _fetch_daily_range(symbol, interval, period, session, market=market, retry_policy=retry_policy)


def _load_period(symbol, interval, period, request_start, request_end, cache_dir, session,
                 market="spot", retry_policy=None):
    cache_file = _cache_path(cache_dir, symbol, interval, period, market=market) if cache_dir else None
    cached = _read_cache(cache_file) if cache_file is not None else None
    window_start, window_end = _period_window(period, request_start, request_end)
    expected = _expected_index(window_start, window_end, interval)

    if _has_all_expected_rows(cached, interval, window_start, window_end):
        if cache_file is not None:
            print(f"  Loading cached {period.kind[:-2]} {period.key} from {cache_file}")
        window = cached[(cached.index >= window_start) & (cached.index < window_end)] if cached is not None else pd.DataFrame(columns=_OUTPUT_COLUMNS)
        return cached, {
            "period_kind": period.kind,
            "period_key": period.key,
            "window_start": window_start,
            "window_end": window_end,
            "expected_rows": int(len(expected)),
            "observed_rows": int(len(window)),
            "missing_rows": 0,
            "missing_segments": [],
            "status": "complete",
            "used_cache": True,
            "refreshed": False,
            "retry_count": 0,
            "downloads": [],
            "dropped_window": False,
            "duplicate_report": dict((cached.attrs or {}).get("duplicate_report") or {}),
        }

    if cache_file is not None:
        action = "Refreshing" if cached is not None and not cached.empty else "Building"
        print(f"  {action} {period.kind} cache {period.key}")

    refreshed, fetch_report = _fetch_period(symbol, interval, period, session, market=market, retry_policy=retry_policy)
    merged = _merge_frames([cached, refreshed])

    if cache_file is not None:
        _write_cache(cache_file, merged)

    window = merged[(merged.index >= window_start) & (merged.index < window_end)]
    missing_index = expected.difference(window.index)
    return merged, {
        "period_kind": period.kind,
        "period_key": period.key,
        "window_start": window_start,
        "window_end": window_end,
        "expected_rows": int(len(expected)),
        "observed_rows": int(len(window)),
        "missing_rows": int(len(missing_index)),
        "missing_segments": _missing_segments(missing_index, interval),
        "status": "complete" if missing_index.empty else "incomplete",
        "used_cache": cached is not None and not cached.empty,
        "refreshed": True,
        "retry_count": int(fetch_report.get("retry_count", 0)),
        "downloads": fetch_report.get("downloads", []),
        "dropped_window": False,
        "duplicate_report": dict((merged.attrs or {}).get("duplicate_report") or {}),
    }


def _build_integrity_report(df, interval, start_dt, end_dt, market, symbol, gap_policy, period_reports):
    expected = _expected_index(start_dt, end_dt, interval)
    observed = pd.DataFrame(df).copy()
    observed = observed[(observed.index >= start_dt) & (observed.index < end_dt)] if not observed.empty else observed
    missing_index = expected.difference(observed.index) if not expected.empty else pd.DatetimeIndex([], tz="UTC")
    status = "complete" if missing_index.empty else "incomplete"
    if any(report.get("dropped_window") for report in period_reports):
        status = "dropped_windows" if missing_index.size > 0 else "complete"
    duplicate_report = _combine_duplicate_reports(
        *(dict(report.get("duplicate_report") or {}) for report in period_reports)
    )
    return {
        "symbol": symbol,
        "market": market,
        "interval": interval,
        "gap_policy": gap_policy,
        "duplicate_policy": None,
        "expected_rows": int(len(expected)),
        "observed_rows": int(len(observed)),
        "missing_rows": int(len(missing_index)),
        "missing_segments": _missing_segments(missing_index, interval),
        "status": status,
        "retry_count": int(sum(report.get("retry_count", 0) for report in period_reports)),
        "duplicate_report": duplicate_report,
        "periods": period_reports,
    }


def fetch_binance_vision(symbol="BTCUSDT", interval="1h",
                         start="2024-01-01", end="2024-03-01",
                         cache_dir=".cache", market="spot", futures_type=None,
                         gap_policy="warn", duplicate_policy="fail", retry_policy=None,
                         return_report=False):
    """Load spot or futures klines from Binance Vision with periodized local cache files.

    Parameters
    ----------
    symbol : str   – e.g. "BTCUSDT"
    interval : str – e.g. "30m", "1h", "4h", "1d"
    start, end : str – ISO timestamps interpreted as UTC, with end exclusive
    cache_dir : str or None – directory for caching; None disables caching

    market : str – "spot", "um_futures", or "cm_futures"
    futures_type : str or None – backward-compatible alias for futures family
    gap_policy : str – one of {"fail", "warn", "flag", "drop_windows"}
    duplicate_policy : str – one of {"fail", "warn", "flag"}; only conflicting duplicates trigger policy
    retry_policy : dict or None – bounded download retry/backoff settings
    return_report : bool – when True, return (dataframe, integrity_report)

    Returns
    -------
    pd.DataFrame with datetime UTC index and float columns:
        open, high, low, close, volume, quote_volume, trades
    """
    market = _normalize_market(market, futures_type=futures_type)
    if gap_policy not in _VALID_GAP_POLICIES:
        raise ValueError(f"Unsupported gap_policy={gap_policy!r}")
    duplicate_policy = str(duplicate_policy or "fail").lower()
    if duplicate_policy not in _VALID_DUPLICATE_POLICIES:
        raise ValueError(f"Unsupported duplicate_policy={duplicate_policy!r}")
    start_dt = _parse_bound(start)
    end_dt = _parse_bound(end)
    if end_dt <= start_dt:
        raise ValueError(f"Expected start < end, got start={start!r} end={end!r}")

    period_frames = []
    period_reports = []
    with requests.Session() as session:
        for period in _iter_cache_periods(start_dt, end_dt, interval):
            period_frame, period_report = _load_period(
                symbol,
                interval,
                period,
                start_dt,
                end_dt,
                cache_dir,
                session,
                market=market,
                retry_policy=retry_policy,
            )
            duplicate_report = dict(period_report.get("duplicate_report") or {})
            conflicting_timestamps = list(duplicate_report.get("conflicting_timestamps") or [])
            if duplicate_report.get("conflicting_duplicate_timestamps", 0) > 0:
                timestamp_preview = ", ".join(str(pd.Timestamp(ts)) for ts in conflicting_timestamps[:3])
                if duplicate_policy == "fail":
                    raise RuntimeError(
                        f"Conflicting duplicate bars detected for {symbol} {interval} {period_report['period_key']}: "
                        f"{duplicate_report['conflicting_duplicate_timestamps']} timestamps"
                        + (f" ({timestamp_preview})" if timestamp_preview else "")
                    )
                if duplicate_policy == "warn":
                    print(
                        f"  WARNING: conflicting duplicate bars remain in {period_report['period_key']}: "
                        f"{duplicate_report['conflicting_duplicate_timestamps']} timestamps"
                    )
            if period_report["missing_rows"] > 0:
                if gap_policy == "fail":
                    raise RuntimeError(
                        f"Incomplete data window for {symbol} {interval} {period_report['period_key']}: "
                        f"{period_report['missing_rows']} candles missing"
                    )
                if gap_policy == "warn":
                    print(f"  WARNING: {period_report['period_key']} still has missing candles after refresh")
                if gap_policy == "drop_windows":
                    period_report["dropped_window"] = True
                    period_frame = period_frame.iloc[0:0]

            period_frames.append(period_frame)
            period_reports.append(period_report)

    df = _merge_frames(period_frames)
    df = df[(df.index >= start_dt) & (df.index < end_dt)]
    integrity_report = _build_integrity_report(df, interval, start_dt, end_dt, market, symbol, gap_policy, period_reports)
    integrity_report["duplicate_policy"] = duplicate_policy
    df.attrs["integrity_report"] = integrity_report

    if df.empty:
        raise RuntimeError(f"No data fetched for {symbol} {interval} [{start}, {end})")

    df, dataset_manifest = validate_market_frame_contract(
        df,
        market=market,
        dataset_name=f"binance_{market}_{symbol.lower()}_{interval}_bars",
        source={
            "source_name": "binance_vision",
            "symbol": symbol,
            "market": market,
            "interval": interval,
            "start": start_dt,
            "end": end_dt,
            "gap_policy": gap_policy,
            "duplicate_policy": duplicate_policy,
            "integrity_status": integrity_report.get("status"),
            "missing_rows": integrity_report.get("missing_rows"),
            "duplicate_conflicts": (integrity_report.get("duplicate_report") or {}).get("conflicting_duplicate_timestamps"),
            "retry_count": integrity_report.get("retry_count"),
        },
    )
    df.attrs["integrity_report"] = integrity_report
    df.attrs["dataset_manifest"] = dataset_manifest

    if return_report:
        return df, integrity_report
    return df


def fetch_binance_bars(symbol="BTCUSDT", interval="1h",
                       start="2024-01-01", end="2024-03-01",
                       cache_dir=".cache", market="spot", futures_type=None,
                       gap_policy="warn", duplicate_policy="fail", retry_policy=None,
                       return_report=False):
    """Unified market-data entrypoint for Binance Vision spot and futures bars."""
    return fetch_binance_vision(
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
        cache_dir=cache_dir,
        market=market,
        futures_type=futures_type,
        gap_policy=gap_policy,
        duplicate_policy=duplicate_policy,
        retry_policy=retry_policy,
        return_report=return_report,
    )


def _read_table(path, file_format=None):
    resolved = Path(path)
    suffix = (file_format or resolved.suffix.lstrip(".")).lower()
    if suffix == "csv":
        return pd.read_csv(resolved)
    if suffix in {"parquet", "pq"}:
        return pd.read_parquet(resolved)
    if suffix == "json":
        return pd.read_json(resolved)
    raise ValueError(f"Unsupported custom data format={suffix!r} for path={path!r}")


def _prefix_columns(columns, prefix):
    if not prefix:
        return columns
    return {column: f"{prefix}_{column}" for column in columns}


def load_custom_dataset(path=None, frame=None, name=None, file_format=None,
                        timestamp_column="timestamp", availability_column=None,
                        assume_event_time_is_available_time=False,
                        value_columns=None, value_dtypes=None, prefix=None, start=None, end=None,
                        max_feature_age=None):
    """Load a point-in-time safe custom dataset with explicit availability timestamps."""
    if frame is None:
        if path is None:
            raise ValueError("Either path or frame must be provided for custom data")
        raw = _read_table(path, file_format=file_format)
    else:
        raw = pd.DataFrame(frame).copy()

    availability_is_assumed = False
    if timestamp_column not in raw.columns:
        raise ValueError(f"Custom data missing timestamp column {timestamp_column!r}")
    if availability_column is None:
        if not assume_event_time_is_available_time:
            raise ValueError(
                "Custom data requires an explicit availability_column unless assume_event_time_is_available_time=True"
            )
        availability_column = timestamp_column
        availability_is_assumed = True
    if availability_column not in raw.columns:
        raise ValueError(f"Custom data missing availability column {availability_column!r}")

    dataset_name = name or (Path(path).stem if path is not None else "custom_data")
    dataset_prefix = prefix or re.sub(r"[^a-z0-9]+", "_", dataset_name.lower()).strip("_")
    selected_value_columns = list(value_columns or [])
    if not selected_value_columns:
        raise ValueError("Custom data contract requires explicit value_columns")

    prepared = raw.copy()

    if start is not None:
        start_dt = _parse_bound(start)
        prepared = prepared[prepared[availability_column] >= start_dt]
    if end is not None:
        end_dt = _parse_bound(end)
        prepared = prepared[prepared[availability_column] < end_dt]

    prepared, dataset_manifest = validate_custom_source_contract(
        prepared,
        dataset_name=dataset_name,
        timestamp_column=timestamp_column,
        availability_column=availability_column,
        value_columns=selected_value_columns,
        value_dtypes=value_dtypes,
        source={
            "source_name": "custom_dataset",
            "path": str(path) if path is not None else None,
            "file_format": file_format,
            "start": _parse_bound(start) if start is not None else None,
            "end": _parse_bound(end) if end is not None else None,
        },
        availability_is_assumed=availability_is_assumed,
    )
    prepared = prepared.sort_values(availability_column)
    rename_map = _prefix_columns(selected_value_columns, dataset_prefix)

    selected = prepared[selected_value_columns].copy()
    selected.insert(0, "event_timestamp", prepared[timestamp_column].to_numpy())
    selected.insert(1, "available_at", prepared[availability_column].to_numpy())
    selected = selected.rename(columns=rename_map)
    return CustomDataset(
        name=dataset_name,
        frame=selected.reset_index(drop=True),
        availability_column="available_at",
        source_path=str(path) if path is not None else None,
        availability_is_assumed=availability_is_assumed,
        default_allow_exact_matches=False,
        max_feature_age=_parse_optional_tolerance(max_feature_age),
        value_columns=tuple(selected_value_columns),
        dataset_manifest=dataset_manifest,
    )


def _parse_optional_tolerance(value):
    if value is None:
        return None
    return pd.Timedelta(value)


_CUSTOM_DATASET_LOAD_KEYS = {
    "path",
    "frame",
    "name",
    "file_format",
    "timestamp_column",
    "availability_column",
    "assume_event_time_is_available_time",
    "value_columns",
    "value_dtypes",
    "prefix",
    "start",
    "end",
    "max_feature_age",
}


def _custom_join_report_defaults(dataset, joined_columns, allow_exact_matches):
    return {
        "name": dataset.name,
        "source_path": dataset.source_path,
        "joined_columns": joined_columns,
        "coverage": 0.0,
        "matched_row_count": 0,
        "stale_hit_count": 0,
        "stale_hit_rate": 0.0,
        "median_feature_age": None,
        "max_feature_age_observed": None,
        "max_feature_age": dataset.max_feature_age,
        "fallback_assumption_used": bool(dataset.availability_is_assumed),
        "fallback_assumption_rows": 0,
        "fallback_assumption_rate": 0.0,
        "allow_exact_matches": bool(allow_exact_matches),
        "exact_match_count": 0,
        "exact_match_rate": 0.0,
        "dataset_manifest": dict(dataset.dataset_manifest or {}),
        "contract_hash": dict(dataset.dataset_manifest or {}).get("contract", {}).get("contract_hash"),
    }


def join_custom_dataset(base_frame, dataset, tolerance=None, allow_exact_matches=None, return_report=False):
    """Point-in-time join a custom dataset onto market data using availability timestamps."""
    base = pd.DataFrame(base_frame).copy()
    if not isinstance(base.index, pd.DatetimeIndex):
        raise ValueError("Base frame index must be a DatetimeIndex for point-in-time joins")
    feature_columns = [
        column for column in dataset.frame.columns
        if column not in {"event_timestamp", dataset.availability_column}
    ]

    effective_max_feature_age = dataset.max_feature_age
    if tolerance is not None:
        effective_max_feature_age = _parse_optional_tolerance(tolerance)
    if allow_exact_matches is None:
        allow_exact_matches = dataset.default_allow_exact_matches

    if dataset.frame.empty:
        report = _custom_join_report_defaults(dataset, feature_columns, allow_exact_matches)
        return (base, report) if return_report else base

    anchor = pd.DataFrame({"decision_time": pd.DatetimeIndex(base.index)}).sort_values("decision_time")
    custom = dataset.frame.sort_values(dataset.availability_column)
    joined = pd.merge_asof(
        anchor,
        custom,
        left_on="decision_time",
        right_on=dataset.availability_column,
        direction="backward",
        allow_exact_matches=allow_exact_matches,
    )

    matched_mask = joined[dataset.availability_column].notna()
    feature_age = pd.Series(pd.NaT, index=joined.index, dtype="timedelta64[ns]")
    if matched_mask.any():
        feature_age.loc[matched_mask] = joined.loc[matched_mask, "decision_time"] - joined.loc[matched_mask, dataset.availability_column]

    stale_mask = pd.Series(False, index=joined.index, dtype=bool)
    if effective_max_feature_age is not None:
        stale_mask = matched_mask & feature_age.gt(effective_max_feature_age)
        if stale_mask.any():
            for column in ["event_timestamp", dataset.availability_column, *feature_columns]:
                if column not in joined.columns:
                    continue
                null_value = pd.NaT if pd.api.types.is_datetime64_any_dtype(joined[column]) else np.nan
                joined.loc[stale_mask, column] = null_value

    matched_after_ttl_mask = matched_mask & ~stale_mask
    matched_feature_age = feature_age.loc[matched_after_ttl_mask] if matched_after_ttl_mask.any() else pd.Series(dtype="timedelta64[ns]")
    exact_match_mask = matched_after_ttl_mask & feature_age.eq(pd.Timedelta(0))

    joined = joined.set_index("decision_time")
    joined.index.name = base.index.name
    if dataset.availability_column in joined.columns:
        joined = joined.drop(columns=[dataset.availability_column])

    coverage = float(joined[feature_columns].notna().all(axis=1).mean()) if feature_columns else 0.0
    fallback_rows = int(matched_after_ttl_mask.sum()) if dataset.availability_is_assumed else 0
    report = {
        "name": dataset.name,
        "source_path": dataset.source_path,
        "joined_columns": feature_columns,
        "coverage": coverage,
        "matched_row_count": int(matched_after_ttl_mask.sum()),
        "stale_hit_count": int(stale_mask.sum()),
        "stale_hit_rate": float(stale_mask.mean()) if len(stale_mask) > 0 else 0.0,
        "median_feature_age": matched_feature_age.median() if not matched_feature_age.empty else None,
        "max_feature_age_observed": matched_feature_age.max() if not matched_feature_age.empty else None,
        "max_feature_age": effective_max_feature_age,
        "fallback_assumption_used": bool(dataset.availability_is_assumed),
        "fallback_assumption_rows": fallback_rows,
        "fallback_assumption_rate": (fallback_rows / len(base)) if len(base) > 0 else 0.0,
        "allow_exact_matches": bool(allow_exact_matches),
        "exact_match_count": int(exact_match_mask.sum()),
        "exact_match_rate": float(exact_match_mask.mean()) if len(exact_match_mask) > 0 else 0.0,
        "dataset_manifest": dict(dataset.dataset_manifest or {}),
        "contract_hash": dict(dataset.dataset_manifest or {}).get("contract", {}).get("contract_hash"),
    }
    result = base.join(joined)
    return (result, report) if return_report else result


def join_custom_data(base_frame, datasets):
    """Apply one or more point-in-time custom data joins to market data."""
    joined = pd.DataFrame(base_frame).copy()
    reports = []
    for config in datasets or []:
        dataset_kwargs = {key: value for key, value in config.items() if key in _CUSTOM_DATASET_LOAD_KEYS}
        dataset = load_custom_dataset(**dataset_kwargs)
        joined, report = join_custom_dataset(
            joined,
            dataset,
            tolerance=config.get("tolerance"),
            allow_exact_matches=config.get("allow_exact_matches"),
            return_report=True,
        )
        reports.append(report)
    return joined, reports


def _symbol_filters_cache_path(cache_dir, market, symbol):
    if cache_dir is None:
        return None
    return Path(cache_dir) / _normalize_market(market) / "symbol_filters" / f"{symbol}.json"


def _exchange_info_cache_path(cache_dir, market):
    if cache_dir is None:
        return None
    return Path(cache_dir) / _normalize_market(market) / "exchange_info" / "all.json"


def _futures_metadata_cache_path(cache_dir, market, symbol, namespace):
    if cache_dir is None:
        return None
    return Path(cache_dir) / _normalize_market(market) / "futures_metadata" / namespace / f"{symbol}.json"


def _coerce_filter_float(value):
    if value in (None, ""):
        return None
    coerced = float(value)
    return coerced if pd.notna(coerced) else None


def _coerce_filter_int(value):
    if value in (None, ""):
        return None
    return int(value)


def _coerce_filter_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _parse_symbol_filters(payload):
    filters = payload.get("filters", [])
    parsed = {
        "symbol": payload.get("symbol"),
        "raw_filters": {},
        "unsupported_filters": {},
    }
    for item in filters:
        filter_type = item.get("filterType")
        parsed["raw_filters"][filter_type] = dict(item)
        if filter_type == "PRICE_FILTER":
            parsed["min_price"] = _coerce_filter_float(item.get("minPrice"))
            parsed["max_price"] = _coerce_filter_float(item.get("maxPrice"))
            parsed["tick_size"] = _coerce_filter_float(item.get("tickSize"))
        elif filter_type == "LOT_SIZE":
            parsed["step_size"] = _coerce_filter_float(item.get("stepSize"))
            parsed["min_qty"] = _coerce_filter_float(item.get("minQty"))
            parsed["max_qty"] = _coerce_filter_float(item.get("maxQty"))
        elif filter_type == "MARKET_LOT_SIZE":
            parsed["market_step_size"] = _coerce_filter_float(item.get("stepSize"))
            parsed["market_min_qty"] = _coerce_filter_float(item.get("minQty"))
            parsed["market_max_qty"] = _coerce_filter_float(item.get("maxQty"))
        elif filter_type == "MIN_NOTIONAL":
            parsed["min_notional"] = _coerce_filter_float(item.get("minNotional"))
            parsed["min_notional_apply_to_market"] = _coerce_filter_bool(item.get("applyToMarket", True))
            parsed["min_notional_avg_price_mins"] = _coerce_filter_int(item.get("avgPriceMins", 0))
        elif filter_type == "NOTIONAL":
            parsed["notional_min_notional"] = _coerce_filter_float(item.get("minNotional"))
            parsed.setdefault("min_notional", parsed["notional_min_notional"])
            parsed["max_notional"] = _coerce_filter_float(item.get("maxNotional"))
            parsed["notional_apply_min_to_market"] = _coerce_filter_bool(item.get("applyMinToMarket", True))
            parsed["notional_apply_max_to_market"] = _coerce_filter_bool(item.get("applyMaxToMarket", True))
            parsed["notional_avg_price_mins"] = _coerce_filter_int(item.get("avgPriceMins", 0))
        elif filter_type == "PERCENT_PRICE":
            parsed["percent_price"] = {
                "multiplier_up": _coerce_filter_float(item.get("multiplierUp")),
                "multiplier_down": _coerce_filter_float(item.get("multiplierDown")),
                "avg_price_mins": _coerce_filter_int(item.get("avgPriceMins", 0)),
            }
        elif filter_type == "PERCENT_PRICE_BY_SIDE":
            parsed["percent_price_by_side"] = {
                "bid_multiplier_up": _coerce_filter_float(item.get("bidMultiplierUp")),
                "bid_multiplier_down": _coerce_filter_float(item.get("bidMultiplierDown")),
                "ask_multiplier_up": _coerce_filter_float(item.get("askMultiplierUp")),
                "ask_multiplier_down": _coerce_filter_float(item.get("askMultiplierDown")),
                "avg_price_mins": _coerce_filter_int(item.get("avgPriceMins", 0)),
            }
        elif filter_type == "MAX_POSITION":
            parsed["max_position"] = _coerce_filter_float(item.get("maxPosition"))
        else:
            parsed["unsupported_filters"][filter_type] = dict(item)
    return parsed


def _fetch_exchange_info_symbol_payload(symbol, market="spot", cache_dir=".cache"):
    normalized_market = _normalize_market(market)
    cache_path = _futures_metadata_cache_path(cache_dir, normalized_market, symbol, "exchange_info")
    cached = _read_object_cache(cache_path)
    if cached is not None:
        return cached

    payload = fetch_binance_exchange_info(market=normalized_market, cache_dir=cache_dir)
    symbols = payload.get("symbols", [])
    symbol_payload = next((dict(item) for item in symbols if item.get("symbol") == symbol), None)
    if symbol_payload is None:
        raise RuntimeError(f"No exchangeInfo filters returned for {symbol} on {normalized_market}")
    _write_object_cache(cache_path, symbol_payload)
    return symbol_payload


def fetch_binance_exchange_info(market="spot", cache_dir=".cache", force_refresh=False):
    """Fetch the full Binance exchangeInfo payload for a market and cache it locally."""
    normalized_market = _normalize_market(market)
    cache_path = _exchange_info_cache_path(cache_dir, normalized_market)
    if not force_refresh:
        cached = _read_object_cache(cache_path)
        if cached is not None:
            return cached

    base_url = _rest_base_url(normalized_market)
    endpoint = "/api/v3/exchangeInfo" if normalized_market == "spot" else (
        "/fapi/v1/exchangeInfo" if normalized_market == "um_futures" else "/dapi/v1/exchangeInfo"
    )

    with requests.Session() as session:
        response = session.get(f"{base_url}{endpoint}", timeout=30)
        response.raise_for_status()
        payload = response.json()

    _write_object_cache(cache_path, payload)
    return payload


def _parse_futures_contract_spec(payload, market="um_futures"):
    normalized_market = _normalize_market(market)
    if normalized_market == "spot":
        raise ValueError("Futures contract specs are only available for futures markets")

    return {
        "symbol": payload.get("symbol"),
        "pair": payload.get("pair") or payload.get("symbol"),
        "market": normalized_market,
        "contract_type": payload.get("contractType"),
        "base_asset": payload.get("baseAsset"),
        "quote_asset": payload.get("quoteAsset"),
        "margin_asset": payload.get("marginAsset") or payload.get("quoteAsset"),
        "status": payload.get("status"),
        "contract_size": _coerce_filter_float(payload.get("contractSize")) or 1.0,
        "liquidation_fee_rate": _coerce_filter_float(payload.get("liquidationFee")),
        "market_take_bound": _coerce_filter_float(payload.get("marketTakeBound")),
        "trigger_protect": _coerce_filter_float(payload.get("triggerProtect")),
        "price_precision": _coerce_filter_int(payload.get("pricePrecision")),
        "quantity_precision": _coerce_filter_int(payload.get("quantityPrecision")),
        "base_asset_precision": _coerce_filter_int(payload.get("baseAssetPrecision")),
        "quote_precision": _coerce_filter_int(payload.get("quotePrecision")),
        "onboard_date": (
            pd.to_datetime(int(payload.get("onboardDate")), unit="ms", utc=True)
            if payload.get("onboardDate") not in (None, "") else None
        ),
        "delivery_date": (
            pd.to_datetime(int(payload.get("deliveryDate")), unit="ms", utc=True)
            if payload.get("deliveryDate") not in (None, "") else None
        ),
    }


def _normalize_futures_leverage_brackets(payload, symbol=None, market="um_futures"):
    normalized_market = _normalize_market(market)
    if normalized_market == "spot":
        raise ValueError("Futures leverage brackets are only available for futures markets")

    selected_symbol = symbol
    notional_coef = 1.0
    bracket_rows = payload

    if isinstance(payload, dict):
        if "brackets" in payload:
            selected_symbol = payload.get("symbol", selected_symbol)
            notional_coef = float(payload.get("notionalCoef", 1.0) or 1.0)
            bracket_rows = payload.get("brackets", [])
        else:
            bracket_rows = [payload]
    elif isinstance(payload, list) and payload and isinstance(payload[0], dict) and "brackets" in payload[0]:
        candidates = payload
        if selected_symbol is not None:
            match = next((row for row in candidates if row.get("symbol") == selected_symbol), None)
            if match is None:
                raise ValueError(f"No leverage bracket payload found for symbol {selected_symbol!r}")
            payload = match
        else:
            payload = candidates[0]
        selected_symbol = payload.get("symbol", selected_symbol)
        notional_coef = float(payload.get("notionalCoef", 1.0) or 1.0)
        bracket_rows = payload.get("brackets", [])

    normalized_rows = []
    for row in bracket_rows or []:
        normalized_rows.append(
            {
                "bracket": _coerce_filter_int(row.get("bracket")) or len(normalized_rows) + 1,
                "initial_leverage": _coerce_filter_float(
                    row.get("initialLeverage", row.get("initial_leverage", row.get("maxLeverage")))
                ) or 1.0,
                "notional_floor": _coerce_filter_float(row.get("notionalFloor", row.get("notional_floor"))),
                "notional_cap": _coerce_filter_float(row.get("notionalCap", row.get("notional_cap"))),
                "qty_floor": _coerce_filter_float(row.get("qtyFloor", row.get("qty_floor"))),
                "qty_cap": _coerce_filter_float(row.get("qtyCap", row.get("qty_cap"))),
                "maint_margin_ratio": _coerce_filter_float(
                    row.get("maintMarginRatio", row.get("maint_margin_ratio"))
                ) or 0.0,
                "cum": _coerce_filter_float(row.get("cum")) or 0.0,
            }
        )

    normalized_rows.sort(
        key=lambda row: (
            row.get("notional_floor") if row.get("notional_floor") is not None else -np.inf,
            row.get("notional_cap") if row.get("notional_cap") is not None else np.inf,
        )
    )

    return {
        "symbol": selected_symbol,
        "market": normalized_market,
        "notional_coef": float(notional_coef),
        "brackets": normalized_rows,
    }


def _signed_request_json(base_url, endpoint, params, api_key, api_secret, timeout=30):
    query_params = {**dict(params or {}), "timestamp": int(time.time() * 1000)}
    query = urlencode(sorted(query_params.items()))
    signature = hmac.new(api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
    headers = {"X-MBX-APIKEY": api_key}
    with requests.Session() as session:
        response = session.get(
            f"{base_url}{endpoint}",
            params={**query_params, "signature": signature},
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()


def fetch_binance_futures_contract_spec(symbol, market="um_futures", cache_dir=".cache"):
    """Fetch normalized futures-only contract metadata from exchangeInfo."""
    normalized_market = _normalize_market(market)
    if normalized_market == "spot":
        raise ValueError("Futures contract specs are only available for futures markets")

    cache_path = _futures_metadata_cache_path(cache_dir, normalized_market, symbol, "contract_spec")
    cached = _read_object_cache(cache_path)
    if cached is not None:
        return cached

    payload = _fetch_exchange_info_symbol_payload(symbol, market=normalized_market, cache_dir=cache_dir)
    contract_spec = _parse_futures_contract_spec(payload, market=normalized_market)
    _write_object_cache(cache_path, contract_spec)
    return contract_spec


def load_futures_leverage_brackets(symbol, market="um_futures", brackets=None, path=None,
                                   cache_dir=".cache", use_signed_endpoint=False,
                                   api_key=None, api_secret=None):
    """Load normalized leverage brackets from config, cache, or the signed Binance endpoint."""
    normalized_market = _normalize_market(market)
    if normalized_market == "spot":
        raise ValueError("Futures leverage brackets are only available for futures markets")

    cache_path = _futures_metadata_cache_path(cache_dir, normalized_market, symbol, "leverage_brackets")
    if brackets is None and path is None and not use_signed_endpoint:
        return _read_object_cache(cache_path)

    payload = None
    if brackets is not None:
        payload = brackets
    elif path is not None:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        resolved_key = api_key or os.getenv("BINANCE_API_KEY") or os.getenv("BINANCE_FAPI_KEY")
        resolved_secret = api_secret or os.getenv("BINANCE_API_SECRET") or os.getenv("BINANCE_FAPI_SECRET")
        if not resolved_key or not resolved_secret:
            raise ValueError("Signed leverage-bracket fetch requires api_key/api_secret or Binance API env vars")
        endpoint = "/fapi/v1/leverageBracket" if normalized_market == "um_futures" else "/dapi/v1/leverageBracket"
        payload = _signed_request_json(
            base_url=_rest_base_url(normalized_market),
            endpoint=endpoint,
            params={"symbol": symbol},
            api_key=resolved_key,
            api_secret=resolved_secret,
            timeout=30,
        )

    normalized = _normalize_futures_leverage_brackets(payload, symbol=symbol, market=normalized_market)
    _write_object_cache(cache_path, normalized)
    return normalized


def fetch_binance_symbol_filters(symbol, market="spot", cache_dir=".cache"):
    """Fetch Binance symbol execution filters for spot or futures markets."""
    normalized_market = _normalize_market(market)
    cache_path = _symbol_filters_cache_path(cache_dir, normalized_market, symbol)
    cached = None
    if cache_path is not None:
        try:
            cached = _read_object_cache(cache_path)
        except Exception:
            cached = None
    if cached is not None:
        return dict(cached)

    payload = _fetch_exchange_info_symbol_payload(symbol, market=normalized_market, cache_dir=cache_dir)
    filters = _parse_symbol_filters(payload)
    if cache_path is not None:
        _write_object_cache(cache_path, filters)
    return filters
