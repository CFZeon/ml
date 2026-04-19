"""Historical universe snapshots and symbol lifecycle governance."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import _normalize_market, fetch_binance_exchange_info


_HALT_STATUSES = {
    "AUCTION_MATCH",
    "BREAK",
    "HALT",
    "HALTED",
    "PAUSED",
    "SUSPENDED",
}
_DELIST_STATUSES = {
    "CLOSE",
    "CLOSED",
    "DELISTED",
    "DELIVERED",
    "DELIVERING",
    "END_OF_LIFE",
    "EXPIRED",
    "SETTLING",
}


@dataclass(frozen=True)
class HistoricalUniverseSnapshot:
    snapshot_timestamp: pd.Timestamp | None
    market: str
    symbols: pd.DataFrame
    source: str | None = None


def _coerce_timestamp(value):
    if value in (None, "") or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
        unit = "ms" if abs(float(value)) >= 1e11 else "s"
        return pd.to_datetime(int(value), unit=unit, utc=True)
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _coerce_float(value):
    if value in (None, ""):
        return np.nan
    coerced = float(value)
    return coerced if np.isfinite(coerced) else np.nan


def _first_present(record, *keys):
    for key in keys:
        if key in record and record.get(key) not in (None, ""):
            return record.get(key)
    return None


def _looks_like_snapshot_manifest(value):
    return isinstance(value, dict) and (
        "symbols" in value or any(key in value for key in ["snapshot_timestamp", "as_of", "timestamp", "serverTime"])
    )


def _normalize_symbol_record(record, market, snapshot_timestamp):
    symbol = record.get("symbol")
    if symbol in (None, ""):
        return None

    listing_start = _coerce_timestamp(
        _first_present(record, "listing_start", "listed_at", "list_date", "onboard_date", "onboardDate")
    )
    delisting_end = _coerce_timestamp(
        _first_present(record, "delisting_end", "delisted_at", "end_date", "delivery_date", "deliveryDate")
    )
    status = str(record.get("status") or "TRADING").upper()

    return {
        "symbol": str(symbol),
        "market": _normalize_market(record.get("market", market)),
        "status": status,
        "listing_start": listing_start,
        "delisting_end": delisting_end,
        "avg_daily_quote_volume": _coerce_float(
            _first_present(
                record,
                "avg_daily_quote_volume",
                "avg_quote_volume",
                "quote_volume",
                "liquidity",
                "avgQuoteVolume",
            )
        ),
        "snapshot_timestamp": snapshot_timestamp,
    }


def normalize_universe_snapshot(snapshot, market="spot", snapshot_timestamp=None, source=None):
    if isinstance(snapshot, HistoricalUniverseSnapshot):
        resolved_timestamp = snapshot.snapshot_timestamp if snapshot_timestamp is None else _coerce_timestamp(snapshot_timestamp)
        resolved_market = _normalize_market(market or snapshot.market)
        frame = pd.DataFrame(snapshot.symbols).copy()
        if not frame.empty:
            frame["market"] = resolved_market
            frame["snapshot_timestamp"] = resolved_timestamp
        return HistoricalUniverseSnapshot(
            snapshot_timestamp=resolved_timestamp,
            market=resolved_market,
            symbols=frame,
            source=source or snapshot.source,
        )

    payload = snapshot
    resolved_market = _normalize_market(market)
    resolved_timestamp = _coerce_timestamp(snapshot_timestamp)
    resolved_source = source
    if isinstance(payload, pd.DataFrame):
        records = payload.to_dict("records")
    elif isinstance(payload, dict):
        resolved_market = _normalize_market(payload.get("market", resolved_market))
        resolved_timestamp = _coerce_timestamp(
            _first_present(payload, "snapshot_timestamp", "as_of", "timestamp", "serverTime")
        ) or resolved_timestamp
        resolved_source = resolved_source or payload.get("source")
        records = payload.get("symbols", [payload])
    else:
        records = list(payload or [])

    rows = []
    for record in records:
        normalized = _normalize_symbol_record(dict(record), resolved_market, resolved_timestamp)
        if normalized is not None:
            rows.append(normalized)

    frame = pd.DataFrame(
        rows,
        columns=[
            "symbol",
            "market",
            "status",
            "listing_start",
            "delisting_end",
            "avg_daily_quote_volume",
            "snapshot_timestamp",
        ],
    )
    if not frame.empty:
        frame = frame.sort_values("symbol").reset_index(drop=True)

    return HistoricalUniverseSnapshot(
        snapshot_timestamp=resolved_timestamp,
        market=resolved_market,
        symbols=frame,
        source=resolved_source,
    )


def _snapshot_cache_dir(cache_dir, market):
    if cache_dir is None:
        return None
    return Path(cache_dir) / _normalize_market(market) / "universe_snapshots"


def _serialize_snapshot(snapshot):
    records = []
    for record in snapshot.symbols.to_dict("records"):
        serialized = dict(record)
        for key in ["listing_start", "delisting_end", "snapshot_timestamp"]:
            value = serialized.get(key)
            if pd.isna(value) or value is None:
                serialized[key] = None
            else:
                serialized[key] = pd.Timestamp(value).isoformat()
        records.append(serialized)
    return {
        "snapshot_timestamp": snapshot.snapshot_timestamp.isoformat() if snapshot.snapshot_timestamp is not None else None,
        "market": snapshot.market,
        "source": snapshot.source,
        "symbols": records,
    }


def persist_historical_universe_snapshot(snapshot, cache_dir=".cache", snapshot_timestamp=None, market="spot"):
    normalized = normalize_universe_snapshot(snapshot, market=market, snapshot_timestamp=snapshot_timestamp)
    if normalized.snapshot_timestamp is None:
        raise ValueError("Historical universe snapshots require a snapshot_timestamp")

    cache_path = _snapshot_cache_dir(cache_dir, normalized.market)
    if cache_path is None:
        raise ValueError("cache_dir must be provided to persist a universe snapshot")

    file_name = normalized.snapshot_timestamp.strftime("%Y%m%dT%H%M%SZ") + ".json"
    target = cache_path / file_name
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_suffix(target.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(_serialize_snapshot(normalized), handle, indent=2, sort_keys=True)
    temp_path.replace(target)
    return target


def _load_snapshot_file(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return normalize_universe_snapshot(payload, source=str(path))


def _load_snapshot_candidates(market="spot", cache_dir=".cache", snapshots=None, path=None, fetch_if_missing=False):
    resolved_market = _normalize_market(market)
    candidates = []

    if snapshots is not None:
        if isinstance(snapshots, HistoricalUniverseSnapshot):
            candidates.append(normalize_universe_snapshot(snapshots, market=resolved_market))
        elif isinstance(snapshots, (dict, pd.DataFrame)):
            candidates.append(normalize_universe_snapshot(snapshots, market=resolved_market))
        else:
            snapshot_items = list(snapshots)
            if snapshot_items:
                if all(_looks_like_snapshot_manifest(item) or isinstance(item, HistoricalUniverseSnapshot) for item in snapshot_items):
                    for item in snapshot_items:
                        candidates.append(normalize_universe_snapshot(item, market=resolved_market))
                else:
                    candidates.append(
                        normalize_universe_snapshot(
                            {"market": resolved_market, "symbols": snapshot_items},
                            market=resolved_market,
                        )
                    )

    if path is not None:
        resolved = Path(path)
        paths = [resolved] if resolved.is_file() else sorted(resolved.glob("*.json"))
        for snapshot_path in paths:
            candidates.append(_load_snapshot_file(snapshot_path))
    elif cache_dir is not None:
        cache_path = _snapshot_cache_dir(cache_dir, resolved_market)
        if cache_path is not None and cache_path.exists():
            for snapshot_path in sorted(cache_path.glob("*.json")):
                candidates.append(_load_snapshot_file(snapshot_path))

    if not candidates and fetch_if_missing:
        payload = fetch_binance_exchange_info(market=resolved_market, cache_dir=cache_dir, force_refresh=True)
        snapshot = normalize_universe_snapshot(
            payload,
            market=resolved_market,
            snapshot_timestamp=_coerce_timestamp(payload.get("serverTime")) or pd.Timestamp.now(tz="UTC"),
            source="exchange_info",
        )
        if cache_dir is not None:
            persist_historical_universe_snapshot(snapshot, cache_dir=cache_dir)
        candidates.append(snapshot)

    return candidates


def load_historical_universe_snapshot(snapshot_timestamp=None, market="spot", cache_dir=".cache",
                                      snapshots=None, path=None, fetch_if_missing=False):
    candidates = _load_snapshot_candidates(
        market=market,
        cache_dir=cache_dir,
        snapshots=snapshots,
        path=path,
        fetch_if_missing=fetch_if_missing,
    )
    if not candidates:
        raise ValueError("No historical universe snapshots were available")

    if snapshot_timestamp is None:
        return max(candidates, key=lambda candidate: candidate.snapshot_timestamp or pd.Timestamp.min.tz_localize("UTC"))

    as_of = _coerce_timestamp(snapshot_timestamp)
    eligible = [candidate for candidate in candidates if candidate.snapshot_timestamp is not None and candidate.snapshot_timestamp <= as_of]
    if not eligible:
        raise ValueError(f"No historical universe snapshot is available at or before {as_of}")
    return max(eligible, key=lambda candidate: candidate.snapshot_timestamp)


def evaluate_universe_eligibility(snapshot, as_of=None, requested_symbols=None,
                                  min_history_days=0, min_liquidity=None):
    normalized = normalize_universe_snapshot(snapshot)
    universe_as_of = _coerce_timestamp(as_of) or normalized.snapshot_timestamp or pd.Timestamp.now(tz="UTC")
    frame = normalized.symbols.copy()
    if frame.empty:
        requested_map = {str(symbol): ["missing_from_snapshot"] for symbol in (requested_symbols or [])}
        return {
            "snapshot_timestamp": normalized.snapshot_timestamp,
            "market": normalized.market,
            "eligible_symbols": [],
            "ineligible_symbols": requested_map,
            "requested_symbols": list(requested_symbols or []),
            "requested_symbol_records": {},
            "eligibility_frame": frame,
        }

    requested_list = [str(symbol) for symbol in (requested_symbols or [])]
    requested_set = set(requested_list)
    min_history_delta = pd.Timedelta(days=max(0, int(min_history_days or 0)))
    liquidity_threshold = None if min_liquidity in (None, "") else float(min_liquidity)

    eligible_symbols = []
    ineligible_symbols = {}
    requested_symbol_records = {}
    evaluation_rows = []

    for record in frame.to_dict("records"):
        symbol = str(record["symbol"])
        reasons = []
        listing_start = _coerce_timestamp(record.get("listing_start"))
        delisting_end = _coerce_timestamp(record.get("delisting_end"))
        status = str(record.get("status") or "TRADING").upper()
        liquidity = _coerce_float(record.get("avg_daily_quote_volume"))

        if listing_start is not None and listing_start > universe_as_of:
            reasons.append("not_listed_yet")
        if delisting_end is not None and delisting_end <= universe_as_of:
            reasons.append("delisted")
        if status != "TRADING":
            reasons.append(f"status_{status.lower()}")
        if min_history_delta > pd.Timedelta(0):
            if listing_start is None or listing_start > universe_as_of - min_history_delta:
                reasons.append("insufficient_history")
        if liquidity_threshold is not None:
            if not np.isfinite(liquidity) or liquidity < liquidity_threshold:
                reasons.append("insufficient_liquidity")

        if not reasons:
            eligible_symbols.append(symbol)
        elif not requested_set or symbol in requested_set:
            ineligible_symbols[symbol] = reasons

        if symbol in requested_set:
            requested_symbol_records[symbol] = dict(record)

        evaluation_rows.append(
            {
                **record,
                "eligible": not reasons,
                "reasons": reasons,
            }
        )

    for symbol in requested_list:
        if symbol not in requested_symbol_records:
            ineligible_symbols.setdefault(symbol, ["missing_from_snapshot"])

    return {
        "snapshot_timestamp": normalized.snapshot_timestamp,
        "market": normalized.market,
        "eligible_symbols": sorted(eligible_symbols),
        "ineligible_symbols": ineligible_symbols,
        "requested_symbols": requested_list,
        "requested_symbol_records": requested_symbol_records,
        "eligibility_frame": pd.DataFrame(evaluation_rows),
    }


def _classify_lifecycle_status(status):
    normalized = str(status or "TRADING").upper()
    if normalized in _HALT_STATUSES or "HALT" in normalized or "SUSPEND" in normalized or "PAUSE" in normalized:
        return "halt"
    if normalized in _DELIST_STATUSES or "DELIST" in normalized or "EXPIRE" in normalized or "DELIVER" in normalized:
        return "delist"
    return None


def build_symbol_lifecycle_frame(index, symbol=None, snapshot=None, events=None):
    if index is None:
        return None
    lifecycle_index = pd.DatetimeIndex(index)
    if lifecycle_index.empty:
        return None

    statuses = pd.Series("TRADING", index=lifecycle_index, dtype=object)
    event_rows = []
    snapshot_record = None
    if snapshot is not None:
        normalized = normalize_universe_snapshot(snapshot)
        if not normalized.symbols.empty and symbol is not None:
            matches = normalized.symbols.loc[normalized.symbols["symbol"].eq(str(symbol))]
            if not matches.empty:
                snapshot_record = matches.iloc[-1].to_dict()
                if snapshot_record.get("status") not in (None, ""):
                    statuses.iloc[:] = str(snapshot_record.get("status")).upper()
                delisting_end = _coerce_timestamp(snapshot_record.get("delisting_end"))
                if delisting_end is not None:
                    event_rows.append(
                        {
                            "timestamp": delisting_end,
                            "symbol": str(symbol),
                            "status": "DELISTED",
                            "source": "snapshot",
                        }
                    )

    if isinstance(events, pd.DataFrame):
        if "status" in events.columns:
            lifecycle = events.copy()
            lifecycle.index = pd.to_datetime(lifecycle.index, utc=True)
            lifecycle = lifecycle.sort_index().reindex(lifecycle_index).ffill()
            lifecycle["status"] = lifecycle["status"].fillna("TRADING").astype(str).str.upper()
            return lifecycle[["status"]]
    elif events is not None:
        if isinstance(events, dict):
            raw_events = [events]
        else:
            raw_events = list(events)
        for event in raw_events:
            event_symbol = event.get("symbol")
            if symbol is not None and event_symbol not in (None, symbol):
                continue
            timestamp = _coerce_timestamp(_first_present(event, "timestamp", "event_time", "time"))
            status = str(event.get("status") or "TRADING").upper()
            if timestamp is None:
                continue
            event_rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": str(symbol or event_symbol or ""),
                    "status": status,
                    "source": event.get("source", "event"),
                }
            )

    if not event_rows and snapshot_record is None:
        return None

    event_frame = pd.DataFrame(event_rows)
    if not event_frame.empty:
        event_frame = event_frame.sort_values("timestamp")
        for row in event_frame.to_dict("records"):
            statuses.loc[statuses.index >= row["timestamp"]] = row["status"]

    lifecycle = pd.DataFrame({"status": statuses.astype(str).str.upper()})
    lifecycle["status_category"] = lifecycle["status"].map(_classify_lifecycle_status)
    return lifecycle


def apply_symbol_lifecycle_policy(position, lifecycle_frame, policy=None):
    series = pd.Series(position, copy=False).astype(float)
    keep_mask = pd.Series(True, index=series.index, dtype=bool)
    if lifecycle_frame is None or len(series) == 0:
        return series.copy(), keep_mask, {
            "enabled": False,
            "forced_liquidations": 0,
            "frozen_bars": 0,
            "dropped_rows": 0,
            "event_count": 0,
            "halt_events": 0,
            "delist_events": 0,
            "policy": {},
            "events": pd.DataFrame(),
        }

    lifecycle = pd.DataFrame(lifecycle_frame).copy()
    if "status" not in lifecycle.columns:
        raise ValueError("lifecycle_frame must include a 'status' column")
    lifecycle.index = pd.to_datetime(lifecycle.index, utc=True)
    lifecycle = lifecycle.reindex(series.index).ffill()
    lifecycle["status"] = lifecycle["status"].fillna("TRADING").astype(str).str.upper()
    lifecycle["status_category"] = lifecycle["status"].map(_classify_lifecycle_status)

    resolved_policy = {
        "halt_action": str((policy or {}).get("halt_action", "freeze")).lower(),
        "delist_action": str((policy or {}).get("delist_action", "liquidate")).lower(),
    }
    adjusted = series.copy()
    event_rows = []
    forced_liquidations = 0
    frozen_bars = 0
    previous_status = "TRADING"

    for position_number, timestamp in enumerate(adjusted.index):
        status = lifecycle.at[timestamp, "status"]
        category = lifecycle.at[timestamp, "status_category"]
        if category is None:
            previous_status = status
            continue

        action = resolved_policy.get(f"{category}_action", "ignore")
        previous_position = float(adjusted.iloc[position_number - 1]) if position_number > 0 else 0.0
        if status != previous_status:
            event_rows.append(
                {
                    "timestamp": timestamp,
                    "status": status,
                    "category": category,
                    "action": action,
                }
            )

        if action == "freeze":
            adjusted.iloc[position_number] = previous_position
            frozen_bars += 1
        elif action == "liquidate":
            if status != previous_status and abs(previous_position) > 1e-12:
                forced_liquidations += 1
            adjusted.iloc[position_number] = 0.0
        elif action == "drop":
            if status != previous_status and abs(previous_position) > 1e-12:
                forced_liquidations += 1
            adjusted.iloc[position_number] = 0.0
            if position_number + 1 < len(adjusted):
                keep_mask.iloc[position_number + 1:] = False
            previous_status = status
            break

        previous_status = status

    event_frame = pd.DataFrame(event_rows)
    return adjusted.loc[keep_mask], keep_mask, {
        "enabled": True,
        "forced_liquidations": int(forced_liquidations),
        "frozen_bars": int(frozen_bars),
        "dropped_rows": int((~keep_mask).sum()),
        "event_count": int(len(event_rows)),
        "halt_events": int(sum(1 for row in event_rows if row.get("category") == "halt")),
        "delist_events": int(sum(1 for row in event_rows if row.get("category") == "delist")),
        "policy": resolved_policy,
        "events": event_frame,
    }


__all__ = [
    "HistoricalUniverseSnapshot",
    "apply_symbol_lifecycle_policy",
    "build_symbol_lifecycle_frame",
    "evaluate_universe_eligibility",
    "load_historical_universe_snapshot",
    "normalize_universe_snapshot",
    "persist_historical_universe_snapshot",
]