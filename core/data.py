"""Fetch OHLCV data from Binance Vision public archives.

Binance Vision spot klines are ZIP-compressed CSV files with 12 columns matching
the `/api/v3/klines` payload: open time, OHLCV, close time, quote volume,
trade count, taker buy base volume, taker buy quote volume, and an ignored
trailing field. Remote archives live under both `monthly/klines/...` and
`daily/klines/...`; local cache files here are periodized by month for `1h+`
intervals and ISO week for sub-hour intervals.
"""

import io
import re
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
_OUTPUT_COLUMNS = [
    "open", "high", "low", "close", "volume", "quote_volume", "trades",
    "taker_buy_base_vol", "taker_buy_quote_vol",
]


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
    return pd.read_pickle(path)


def _write_cache(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_pickle(temp_path)
    temp_path.replace(path)


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

    merged = pd.concat(valid_frames).sort_index()
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


def _download_archive(url, session):
    print(f"  Fetching {url} ...")
    response = session.get(url, timeout=30)
    if response.status_code == 404:
        return None
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        with zf.open(zf.namelist()[0]) as handle:
            raw = pd.read_csv(handle, header=None, names=_COLUMNS)
    return _prepare_frame(raw)


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


def _fetch_daily_range(symbol, interval, period, session, market="spot"):
    frames = []
    available_end = min(period.end, pd.Timestamp.now(tz="UTC").normalize())
    for day in pd.date_range(period.start, available_end, freq="1D", inclusive="left"):
        frame = _download_archive(_daily_archive_url(symbol, interval, day, market=market), session)
        if frame is not None and not frame.empty:
            frames.append(frame)
    return _merge_frames(frames)


def _fetch_period(symbol, interval, period, session, market="spot"):
    if not _uses_weekly_cache(interval):
        monthly_frame = _download_archive(_monthly_archive_url(symbol, interval, period, market=market), session)
        if monthly_frame is not None:
            return monthly_frame

    return _fetch_daily_range(symbol, interval, period, session, market=market)


def _load_period(symbol, interval, period, request_start, request_end, cache_dir, session, market="spot"):
    cache_file = _cache_path(cache_dir, symbol, interval, period, market=market) if cache_dir else None
    cached = _read_cache(cache_file) if cache_file is not None else None
    window_start, window_end = _period_window(period, request_start, request_end)

    if _has_all_expected_rows(cached, interval, window_start, window_end):
        if cache_file is not None:
            print(f"  Loading cached {period.kind[:-2]} {period.key} from {cache_file}")
        return cached

    if cache_file is not None:
        action = "Refreshing" if cached is not None and not cached.empty else "Building"
        print(f"  {action} {period.kind} cache {period.key}")

    refreshed = _fetch_period(symbol, interval, period, session, market=market)
    merged = _merge_frames([cached, refreshed])

    if cache_file is not None:
        _write_cache(cache_file, merged)

    if not _has_all_expected_rows(merged, interval, window_start, window_end):
        print(f"  WARNING: {period.key} still has missing candles after refresh")

    return merged


def fetch_binance_vision(symbol="BTCUSDT", interval="1h",
                         start="2024-01-01", end="2024-03-01",
                         cache_dir=".cache", market="spot", futures_type=None):
    """Load spot or futures klines from Binance Vision with periodized local cache files.

    Parameters
    ----------
    symbol : str   – e.g. "BTCUSDT"
    interval : str – e.g. "30m", "1h", "4h", "1d"
    start, end : str – ISO timestamps interpreted as UTC, with end exclusive
    cache_dir : str or None – directory for caching; None disables caching

    market : str – "spot", "um_futures", or "cm_futures"
    futures_type : str or None – backward-compatible alias for futures family

    Returns
    -------
    pd.DataFrame with datetime UTC index and float columns:
        open, high, low, close, volume, quote_volume, trades
    """
    market = _normalize_market(market, futures_type=futures_type)
    start_dt = _parse_bound(start)
    end_dt = _parse_bound(end)
    if end_dt <= start_dt:
        raise ValueError(f"Expected start < end, got start={start!r} end={end!r}")

    period_frames = []
    with requests.Session() as session:
        for period in _iter_cache_periods(start_dt, end_dt, interval):
            period_frames.append(
                _load_period(symbol, interval, period, start_dt, end_dt, cache_dir, session, market=market)
            )

    df = _merge_frames(period_frames)
    df = df[(df.index >= start_dt) & (df.index < end_dt)]

    if df.empty:
        raise RuntimeError(f"No data fetched for {symbol} {interval} [{start}, {end})")

    return df


def fetch_binance_bars(symbol="BTCUSDT", interval="1h",
                       start="2024-01-01", end="2024-03-01",
                       cache_dir=".cache", market="spot", futures_type=None):
    """Unified market-data entrypoint for Binance Vision spot and futures bars."""
    return fetch_binance_vision(
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
        cache_dir=cache_dir,
        market=market,
        futures_type=futures_type,
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


def _parse_symbol_filters(payload):
    filters = payload.get("filters", [])
    parsed = {"symbol": payload.get("symbol")}
    for item in filters:
        filter_type = item.get("filterType")
        if filter_type == "PRICE_FILTER":
            parsed["tick_size"] = float(item.get("tickSize", 0.0))
        elif filter_type in {"LOT_SIZE", "MARKET_LOT_SIZE"}:
            parsed.setdefault("step_size", float(item.get("stepSize", 0.0)))
            parsed.setdefault("min_qty", float(item.get("minQty", 0.0)))
            parsed.setdefault("max_qty", float(item.get("maxQty", 0.0)))
        elif filter_type == "MIN_NOTIONAL":
            parsed["min_notional"] = float(item.get("minNotional", 0.0))
        elif filter_type == "NOTIONAL":
            parsed.setdefault("min_notional", float(item.get("minNotional", 0.0)))
            parsed["max_notional"] = float(item.get("maxNotional", 0.0))
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
