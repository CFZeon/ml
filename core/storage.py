"""Safe storage helpers for portable caches and model manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def payload_sha256(payload):
    encoded = json.dumps(_json_encode(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_encode(value):
    if isinstance(value, dict):
        return {key: _json_encode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_encode(item) for item in value]
    if isinstance(value, tuple):
        return [_json_encode(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return {"__type__": "timestamp", "value": value.isoformat()}
    if isinstance(value, pd.Timedelta):
        return {"__type__": "timedelta", "value": str(value)}
    return value


def _json_decode(value):
    if isinstance(value, list):
        return [_json_decode(item) for item in value]
    if isinstance(value, dict):
        marker = value.get("__type__")
        if marker == "timestamp":
            return pd.Timestamp(value["value"])
        if marker == "timedelta":
            return pd.Timedelta(value["value"])
        return {key: _json_decode(item) for key, item in value.items()}
    return value


def read_json(path):
    resolved = Path(path)
    if not resolved.exists():
        return None
    with open(resolved, "r", encoding="utf-8") as handle:
        return _json_decode(json.load(handle))


def write_json(path, payload):
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temp_path = resolved.with_suffix(resolved.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(_json_encode(payload), handle, indent=2, sort_keys=True)
    temp_path.replace(resolved)


def read_parquet_frame(path):
    resolved = Path(path)
    if not resolved.exists():
        return None
    return pd.read_parquet(resolved)


def write_parquet_frame(path, frame):
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temp_path = resolved.with_suffix(resolved.suffix + ".tmp")
    pd.DataFrame(frame).to_parquet(temp_path)
    temp_path.replace(resolved)


def file_sha256(path):
    resolved = Path(path)
    digest = hashlib.sha256()
    with open(resolved, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_fingerprint(frame, *, include_index=True):
    normalized = pd.DataFrame(frame).copy()
    digest = hashlib.sha256()
    digest.update("|".join(map(str, normalized.columns)).encode("utf-8"))
    digest.update(normalized.dtypes.astype(str).to_json().encode("utf-8"))
    hashed = pd.util.hash_pandas_object(normalized, index=include_index, categorize=True)
    digest.update(hashed.to_numpy(dtype="uint64", copy=False).tobytes())
    return digest.hexdigest()


__all__ = [
    "file_sha256",
    "frame_fingerprint",
    "payload_sha256",
    "read_json",
    "read_parquet_frame",
    "write_json",
    "write_parquet_frame",
]