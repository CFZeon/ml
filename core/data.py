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

_BASE = "https://data.binance.vision/data/spot"
_INTERVAL_PATTERN = re.compile(r"^(\d+)(mo|[smhdw])$")

_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base_vol", "taker_buy_quote_vol", "ignore",
]

_FLOAT_COLUMNS = ["open", "high", "low", "close", "volume", "quote_volume"]
_OUTPUT_COLUMNS = ["open", "high", "low", "close", "volume", "quote_volume", "trades"]


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


def _cache_path(cache_dir, symbol, interval, period):
    return (
        Path(cache_dir)
        / "spot"
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


def _monthly_archive_url(symbol, interval, period):
    return (
        f"{_BASE}/monthly/klines/{symbol}/{interval}/"
        f"{symbol}-{interval}-{period.start:%Y-%m}.zip"
    )


def _daily_archive_url(symbol, interval, day):
    return (
        f"{_BASE}/daily/klines/{symbol}/{interval}/"
        f"{symbol}-{interval}-{day:%Y-%m-%d}.zip"
    )


def _fetch_daily_range(symbol, interval, period, session):
    frames = []
    for day in pd.date_range(period.start, period.end, freq="1D", inclusive="left"):
        frame = _download_archive(_daily_archive_url(symbol, interval, day), session)
        if frame is not None and not frame.empty:
            frames.append(frame)
    return _merge_frames(frames)


def _fetch_period(symbol, interval, period, session):
    if not _uses_weekly_cache(interval):
        monthly_frame = _download_archive(_monthly_archive_url(symbol, interval, period), session)
        if monthly_frame is not None:
            return monthly_frame

    return _fetch_daily_range(symbol, interval, period, session)


def _load_period(symbol, interval, period, request_start, request_end, cache_dir, session):
    cache_file = _cache_path(cache_dir, symbol, interval, period) if cache_dir else None
    cached = _read_cache(cache_file) if cache_file is not None else None
    window_start, window_end = _period_window(period, request_start, request_end)

    if _has_all_expected_rows(cached, interval, window_start, window_end):
        if cache_file is not None:
            print(f"  Loading cached {period.kind[:-2]} {period.key} from {cache_file}")
        return cached

    if cache_file is not None:
        action = "Refreshing" if cached is not None and not cached.empty else "Building"
        print(f"  {action} {period.kind} cache {period.key}")

    refreshed = _fetch_period(symbol, interval, period, session)
    merged = _merge_frames([cached, refreshed])

    if cache_file is not None:
        _write_cache(cache_file, merged)

    if not _has_all_expected_rows(merged, interval, window_start, window_end):
        print(f"  WARNING: {period.key} still has missing candles after refresh")

    return merged


def fetch_binance_vision(symbol="BTCUSDT", interval="1h",
                         start="2024-01-01", end="2024-03-01",
                         cache_dir=".cache"):
    """Load spot klines from Binance Vision with periodized local cache files.

    Parameters
    ----------
    symbol : str   – e.g. "BTCUSDT"
    interval : str – e.g. "30m", "1h", "4h", "1d"
    start, end : str – ISO timestamps interpreted as UTC, with end exclusive
    cache_dir : str or None – directory for caching; None disables caching

    Returns
    -------
    pd.DataFrame with datetime UTC index and float columns:
        open, high, low, close, volume, quote_volume, trades
    """
    start_dt = _parse_bound(start)
    end_dt = _parse_bound(end)
    if end_dt <= start_dt:
        raise ValueError(f"Expected start < end, got start={start!r} end={end!r}")

    period_frames = []
    with requests.Session() as session:
        for period in _iter_cache_periods(start_dt, end_dt, interval):
            period_frames.append(
                _load_period(symbol, interval, period, start_dt, end_dt, cache_dir, session)
            )

    df = _merge_frames(period_frames)
    df = df[(df.index >= start_dt) & (df.index < end_dt)]

    if df.empty:
        raise RuntimeError(f"No data fetched for {symbol} {interval} [{start}, {end})")

    return df
