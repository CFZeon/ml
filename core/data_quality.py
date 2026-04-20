"""Pre-feature data quality checks and quarantine controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DataQualityResult:
    clean_frame: pd.DataFrame
    quarantine_mask: pd.Series
    report: dict


def _rolling_mad(series, window):
    median = series.rolling(window, min_periods=max(3, window // 2)).median()
    abs_dev = (series - median).abs()
    mad = abs_dev.rolling(window, min_periods=max(3, window // 2)).median()
    return mad.replace(0.0, np.nan)


def _flag_ohlc_inconsistency(frame):
    if not {"open", "high", "low", "close"}.issubset(frame.columns):
        return pd.Series(False, index=frame.index, dtype=bool)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    open_ = frame["open"].astype(float)
    close = frame["close"].astype(float)
    return (high < pd.concat([open_, close, low], axis=1).max(axis=1)) | (low > pd.concat([open_, close, high], axis=1).min(axis=1))


def _flag_timestamp_anomalies(frame):
    index = pd.Index(frame.index)
    duplicate = pd.Series(index.duplicated(keep="first"), index=index, dtype=bool)
    retrograde = pd.Series(False, index=index, dtype=bool)
    if len(index) > 1:
        retrograde.iloc[1:] = pd.Series(index[1:] <= index[:-1], index=index[1:]).astype(bool)
    return duplicate, retrograde


def _flag_return_spikes(frame, window, threshold):
    if "close" not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    close = frame["close"].astype(float)
    returns = close.pct_change().abs()
    scale = _rolling_mad(returns.fillna(0.0), window).fillna(returns.median())
    floor = float(returns.median() or 0.0)
    limit = np.maximum(scale.to_numpy(dtype=float) * float(threshold), floor)
    return returns > pd.Series(limit, index=frame.index)


def _flag_range_spikes(frame, window, threshold):
    if not {"high", "low", "close"}.issubset(frame.columns):
        return pd.Series(False, index=frame.index, dtype=bool)
    range_pct = ((frame["high"].astype(float) - frame["low"].astype(float)).abs() / frame["close"].replace(0.0, np.nan)).fillna(0.0)
    scale = _rolling_mad(range_pct, window).fillna(range_pct.median())
    floor = float(range_pct.median() or 0.0)
    limit = np.maximum(scale.to_numpy(dtype=float) * float(threshold), floor)
    return range_pct > pd.Series(limit, index=frame.index)


def _flag_volume_anomalies(frame):
    if "volume" not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0) <= 0.0


def _flag_quote_volume_inconsistency(frame, tolerance):
    if not {"close", "volume", "quote_volume"}.issubset(frame.columns):
        return pd.Series(False, index=frame.index, dtype=bool)
    implied = frame["close"].astype(float) * frame["volume"].astype(float)
    quoted = frame["quote_volume"].astype(float)
    relative_error = (quoted - implied).abs().divide(implied.replace(0.0, np.nan)).fillna(0.0)
    return relative_error > float(tolerance)


def _flag_trade_count_anomalies(frame, window, spike_threshold):
    if "trades" not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    trades = pd.to_numeric(frame["trades"], errors="coerce").fillna(0.0)
    baseline = trades.rolling(window, min_periods=max(3, window // 2)).median().fillna(trades.median())
    return (trades <= 0.0) | (trades > baseline * float(spike_threshold))


def _winsorize_rows(frame, mask, columns, window, z_threshold):
    updated = frame.copy()
    if not mask.any():
        return updated
    for column in columns:
        if column not in updated.columns:
            continue
        series = pd.to_numeric(updated[column], errors="coerce")
        median = series.rolling(window, min_periods=max(3, window // 2)).median().fillna(series.median())
        mad = _rolling_mad(series, window).fillna(series.mad() if hasattr(series, "mad") else series.abs().median())
        scale = (mad * 1.4826).replace(0.0, np.nan).fillna(series.std() or 0.0)
        lower = median - float(z_threshold) * scale
        upper = median + float(z_threshold) * scale
        updated.loc[mask, column] = series.loc[mask].clip(lower=lower.loc[mask], upper=upper.loc[mask])
    return updated


def check_data_quality(frame, config=None):
    config = dict(config or {})
    actions = dict(config.get("actions", {}) or {})
    block_on_quarantine = bool(config.get("block_on_quarantine", False))
    frame = pd.DataFrame(frame).copy()
    window = max(5, int(config.get("rolling_window", 24)))
    return_spike_threshold = float(config.get("return_spike_threshold", 8.0))
    range_spike_threshold = float(config.get("range_spike_threshold", 8.0))
    quote_volume_tolerance = float(config.get("quote_volume_tolerance", 0.25))
    trade_count_spike_threshold = float(config.get("trade_count_spike_threshold", 10.0))
    winsorize_z_threshold = float(config.get("winsorize_z_threshold", 5.0))

    duplicate_timestamp, retrograde_timestamp = _flag_timestamp_anomalies(frame)
    anomalies = {
        "ohlc_inconsistency": _flag_ohlc_inconsistency(frame),
        "duplicate_timestamp": duplicate_timestamp,
        "retrograde_timestamp": retrograde_timestamp,
        "return_spike": _flag_return_spikes(frame, window=window, threshold=return_spike_threshold),
        "range_spike": _flag_range_spikes(frame, window=window, threshold=range_spike_threshold),
        "nonpositive_volume": _flag_volume_anomalies(frame),
        "quote_volume_inconsistency": _flag_quote_volume_inconsistency(frame, tolerance=quote_volume_tolerance),
        "trade_count_anomaly": _flag_trade_count_anomalies(frame, window=window, spike_threshold=trade_count_spike_threshold),
    }

    default_actions = {
        "ohlc_inconsistency": "drop",
        "duplicate_timestamp": "drop",
        "retrograde_timestamp": "drop",
        "return_spike": "flag",
        "range_spike": "flag",
        "nonpositive_volume": "drop",
        "quote_volume_inconsistency": "flag",
        "trade_count_anomaly": "flag",
    }
    clean = frame.copy()
    quarantine_mask = pd.Series(False, index=frame.index, dtype=bool)
    drop_mask = pd.Series(False, index=frame.index, dtype=bool)
    action_counts = {}
    anomaly_report = {}

    for anomaly_name, mask in anomalies.items():
        mask = pd.Series(mask, index=frame.index, dtype=bool).fillna(False)
        action = str(actions.get(anomaly_name, default_actions.get(anomaly_name, "flag"))).lower()
        if action not in {"flag", "null", "drop", "winsorize"}:
            raise ValueError(f"Unsupported data quality action {action!r} for anomaly {anomaly_name!r}")

        quarantine_mask |= mask
        action_counts[action] = action_counts.get(action, 0) + int(mask.sum())

        if action == "null" and mask.any():
            quality_columns = [column for column in ["open", "high", "low", "close", "volume", "quote_volume", "trades"] if column in clean.columns]
            clean.loc[mask, quality_columns] = np.nan
        elif action == "drop":
            drop_mask |= mask
        elif action == "winsorize" and mask.any():
            winsorize_columns = [column for column in ["open", "high", "low", "close", "volume", "quote_volume", "trades"] if column in clean.columns]
            clean = _winsorize_rows(clean, mask, winsorize_columns, window=window, z_threshold=winsorize_z_threshold)

        anomaly_report[anomaly_name] = {
            "count": int(mask.sum()),
            "action": action,
        }

    if drop_mask.any():
        clean = clean.loc[~drop_mask].copy()

    quarantined_rows = int(quarantine_mask.sum())
    status = "pass" if quarantined_rows == 0 else "quarantine"
    blocking = bool(block_on_quarantine and quarantined_rows > 0)

    report = {
        "status": status,
        "blocking": blocking,
        "summary": {
            "rows_input": int(len(frame)),
            "rows_output": int(len(clean)),
            "quarantined_rows": quarantined_rows,
            "dropped_rows": int(drop_mask.sum()),
            "anomaly_counts": {name: details["count"] for name, details in anomaly_report.items()},
            "action_counts": action_counts,
        },
        "anomalies": anomaly_report,
    }
    clean.attrs["data_quality_report"] = report
    clean.attrs["data_quality_quarantine_mask"] = quarantine_mask.reindex(clean.index).fillna(False)
    return DataQualityResult(
        clean_frame=clean,
        quarantine_mask=quarantine_mask,
        report=report,
    )


__all__ = ["DataQualityResult", "check_data_quality"]