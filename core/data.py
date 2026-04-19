"""Fetch OHLCV data from Binance Vision public archives.

Binance Vision spot klines are ZIP-compressed CSV files with 12 columns matching
the `/api/v3/klines` payload: open time, OHLCV, close time, quote volume,
trade count, taker buy base volume, taker buy quote volume, and an ignored
trailing field. Remote archives live under both `monthly/klines/...` and
`daily/klines/...`; local cache files here are periodized by month for `1h+`
intervals and ISO week for sub-hour intervals.
"""

import io
import random
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

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


@dataclass(frozen=True)
class CustomDataset:
    name: str
    frame: pd.DataFrame
    availability_column: str
    source_path: str | None = None


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
        / f"{period.key}.pkl"
    )


def _read_cache(path):
    if not path.exists():
        return None
    return _normalize_output_schema(pd.read_pickle(path))


def _write_cache(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_pickle(temp_path)
    temp_path.replace(path)


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


def _infer_timestamp_unit(raw_times):
    max_value = pd.to_numeric(raw_times, errors="raise").abs().max()
    return "us" if max_value >= 10**14 else "ms"


def _prepare_frame(frame):
    if frame is None or frame.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

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
    prepared = prepared[~prepared.index.duplicated(keep="first")]
    return prepared


def _merge_frames(frames):
    valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid_frames:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    merged = pd.concat([_normalize_output_schema(frame) for frame in valid_frames]).sort_index()
    merged = merged[~merged.index.duplicated(keep="first")]
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
    }


def _build_integrity_report(df, interval, start_dt, end_dt, market, symbol, gap_policy, period_reports):
    expected = _expected_index(start_dt, end_dt, interval)
    observed = pd.DataFrame(df).copy()
    observed = observed[(observed.index >= start_dt) & (observed.index < end_dt)] if not observed.empty else observed
    missing_index = expected.difference(observed.index) if not expected.empty else pd.DatetimeIndex([], tz="UTC")
    status = "complete" if missing_index.empty else "incomplete"
    if any(report.get("dropped_window") for report in period_reports):
        status = "dropped_windows" if missing_index.size > 0 else "complete"
    return {
        "symbol": symbol,
        "market": market,
        "interval": interval,
        "gap_policy": gap_policy,
        "expected_rows": int(len(expected)),
        "observed_rows": int(len(observed)),
        "missing_rows": int(len(missing_index)),
        "missing_segments": _missing_segments(missing_index, interval),
        "status": status,
        "retry_count": int(sum(report.get("retry_count", 0) for report in period_reports)),
        "periods": period_reports,
    }


def fetch_binance_vision(symbol="BTCUSDT", interval="1h",
                         start="2024-01-01", end="2024-03-01",
                         cache_dir=".cache", market="spot", futures_type=None,
                         gap_policy="warn", retry_policy=None,
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
    df.attrs["integrity_report"] = integrity_report

    if df.empty:
        raise RuntimeError(f"No data fetched for {symbol} {interval} [{start}, {end})")

    if return_report:
        return df, integrity_report
    return df


def fetch_binance_bars(symbol="BTCUSDT", interval="1h",
                       start="2024-01-01", end="2024-03-01",
                       cache_dir=".cache", market="spot", futures_type=None,
                       gap_policy="warn", retry_policy=None,
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
                        value_columns=None, prefix=None, start=None, end=None):
    """Load a point-in-time safe custom dataset with explicit availability timestamps."""
    if frame is None:
        if path is None:
            raise ValueError("Either path or frame must be provided for custom data")
        raw = _read_table(path, file_format=file_format)
    else:
        raw = pd.DataFrame(frame).copy()

    availability_column = availability_column or timestamp_column
    if timestamp_column not in raw.columns:
        raise ValueError(f"Custom data missing timestamp column {timestamp_column!r}")
    if availability_column not in raw.columns:
        raise ValueError(f"Custom data missing availability column {availability_column!r}")

    dataset_name = name or (Path(path).stem if path is not None else "custom_data")
    dataset_prefix = prefix or re.sub(r"[^a-z0-9]+", "_", dataset_name.lower()).strip("_")

    prepared = raw.copy()
    prepared[timestamp_column] = pd.to_datetime(prepared[timestamp_column], utc=True)
    prepared[availability_column] = pd.to_datetime(prepared[availability_column], utc=True)
    prepared = prepared.sort_values(availability_column)

    if start is not None:
        start_dt = _parse_bound(start)
        prepared = prepared[prepared[availability_column] >= start_dt]
    if end is not None:
        end_dt = _parse_bound(end)
        prepared = prepared[prepared[availability_column] < end_dt]

    selected_value_columns = list(value_columns or [
        column for column in prepared.columns
        if column not in {timestamp_column, availability_column}
    ])
    rename_map = _prefix_columns(selected_value_columns, dataset_prefix)

    selected = prepared[[timestamp_column, availability_column] + selected_value_columns].copy()
    selected = selected.rename(columns={**rename_map, timestamp_column: "event_timestamp", availability_column: "available_at"})
    return CustomDataset(
        name=dataset_name,
        frame=selected.reset_index(drop=True),
        availability_column="available_at",
        source_path=str(path) if path is not None else None,
    )


def _parse_optional_tolerance(value):
    if value is None:
        return None
    return pd.Timedelta(value)


def join_custom_dataset(base_frame, dataset, tolerance=None, allow_exact_matches=True):
    """Point-in-time join a custom dataset onto market data using availability timestamps."""
    base = pd.DataFrame(base_frame).copy()
    if not isinstance(base.index, pd.DatetimeIndex):
        raise ValueError("Base frame index must be a DatetimeIndex for point-in-time joins")
    if dataset.frame.empty:
        return base

    anchor = pd.DataFrame({"decision_time": pd.DatetimeIndex(base.index)}).sort_values("decision_time")
    custom = dataset.frame.sort_values(dataset.availability_column)
    joined = pd.merge_asof(
        anchor,
        custom,
        left_on="decision_time",
        right_on=dataset.availability_column,
        direction="backward",
        allow_exact_matches=allow_exact_matches,
        tolerance=_parse_optional_tolerance(tolerance),
    )
    joined = joined.set_index("decision_time")
    joined.index.name = base.index.name
    if dataset.availability_column in joined.columns:
        joined = joined.drop(columns=[dataset.availability_column])
    return base.join(joined)


def join_custom_data(base_frame, datasets):
    """Apply one or more point-in-time custom data joins to market data."""
    joined = pd.DataFrame(base_frame).copy()
    reports = []
    for config in datasets or []:
        dataset = load_custom_dataset(**config)
        joined = join_custom_dataset(
            joined,
            dataset,
            tolerance=config.get("tolerance"),
            allow_exact_matches=config.get("allow_exact_matches", True),
        )
        joined_columns = [column for column in dataset.frame.columns if column not in {"event_timestamp"}]
        coverage = float(joined[[column for column in joined_columns if column in joined.columns]].notna().all(axis=1).mean()) if joined_columns else 0.0
        reports.append(
            {
                "name": dataset.name,
                "source_path": dataset.source_path,
                "joined_columns": [column for column in joined_columns if column in joined.columns],
                "coverage": coverage,
            }
        )
    return joined, reports


def _symbol_filters_cache_path(cache_dir, market, symbol):
    if cache_dir is None:
        return None
    return Path(cache_dir) / _normalize_market(market) / "symbol_filters" / f"{symbol}.pkl"


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


def fetch_binance_symbol_filters(symbol, market="spot", cache_dir=".cache"):
    """Fetch Binance symbol execution filters for spot or futures markets."""
    normalized_market = _normalize_market(market)
    cache_path = _symbol_filters_cache_path(cache_dir, normalized_market, symbol)
    cached = _read_cache(cache_path) if cache_path is not None else None
    if cached is not None:
        return cached

    base_url = _rest_base_url(normalized_market)
    endpoint = "/api/v3/exchangeInfo" if normalized_market == "spot" else (
        "/fapi/v1/exchangeInfo" if normalized_market == "um_futures" else "/dapi/v1/exchangeInfo"
    )

    with requests.Session() as session:
        response = session.get(f"{base_url}{endpoint}", params={"symbol": symbol}, timeout=30)
        response.raise_for_status()
        payload = response.json()

    symbols = payload.get("symbols", [])
    if not symbols:
        raise RuntimeError(f"No exchangeInfo filters returned for {symbol} on {normalized_market}")
    filters = _parse_symbol_filters(symbols[0])
    if cache_path is not None:
        _write_cache(cache_path, filters)
    return filters
