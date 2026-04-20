"""Dataset contracts and lineage manifests for research inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

import pandas as pd

from .storage import file_sha256, frame_fingerprint

try:
    import pandera.pandas as pa
    from pandera.errors import SchemaError, SchemaErrors
except ImportError:  # pragma: no cover - exercised in environments without optional deps.
    pa = None
    SchemaError = Exception
    SchemaErrors = Exception


_CONTRACT_SCHEMA_VERSION = 1
_CONTRACT_SEMANTIC_VERSION = "1.0.0"
_MARKET_BAR_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
]
_OPTIONAL_MARKET_BAR_COLUMNS = [
    "taker_buy_base_vol",
    "taker_buy_quote_vol",
]
_REFERENCE_COLUMNS = [
    "reference_price",
    "reference_close",
    "composite_price",
    "reference_volume",
    "breadth",
    "composite_funding_rate",
    "composite_basis",
]


@dataclass(frozen=True)
class DatasetManifest:
    name: str
    dataset_kind: str
    created_at: str
    row_count: int
    column_count: int
    columns: list[str]
    source_fingerprint: str
    contract: dict = field(default_factory=dict)
    source: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    upstream_contract_hashes: list[str] = field(default_factory=list)
    upstream_dataset_names: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _require_pandera():
    if pa is None:
        raise ImportError(
            "Dataset contract validation requires pandera. Install it with `python -m pip install pandera`."
        )


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items() if item is not None}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            pass
    return value


def _payload_hash(payload):
    encoded = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _schema_error_details(exc):
    if hasattr(exc, "failure_cases") and exc.failure_cases is not None:
        failure_cases = exc.failure_cases
        if hasattr(failure_cases, "head"):
            return failure_cases.head(10).to_dict(orient="records")
    if hasattr(exc, "schema_errors"):
        return [str(error) for error in getattr(exc, "schema_errors", [])[:10]]
    return [str(exc)]


def _contract_failure(dataset_name, exc):
    message = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
    raise ValueError(f"{dataset_name} failed dataset contract validation: {message}") from exc


def _ensure_utc_datetime_index(frame, dataset_name):
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{dataset_name} requires a DatetimeIndex")
    if frame.index.tz is None:
        raise ValueError(f"{dataset_name} requires a timezone-aware UTC index")
    if str(frame.index.tz) != "UTC":
        raise ValueError(f"{dataset_name} requires a UTC index")
    if frame.index.has_duplicates:
        raise ValueError(f"{dataset_name} requires a unique DatetimeIndex")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"{dataset_name} requires a monotonic increasing DatetimeIndex")


def _parse_epoch_unit(series):
    numeric = pd.to_numeric(pd.Series(series).dropna(), errors="raise")
    if numeric.empty:
        return "ms"
    max_abs = numeric.abs().max()
    if max_abs >= 10**14:
        return "us"
    if max_abs >= 10**11:
        return "ms"
    return "s"


def parse_timestamp_series_utc(series, column_name):
    values = pd.Series(series).copy()
    non_null = values.dropna()
    if non_null.empty:
        return pd.to_datetime(values)

    if isinstance(values.dtype, pd.DatetimeTZDtype):
        if str(values.dt.tz) != "UTC":
            raise ValueError(f"{column_name} must already be UTC")
        return values.dt.tz_convert("UTC")

    if pd.api.types.is_datetime64_any_dtype(values):
        raise ValueError(f"{column_name} must be timezone-aware UTC; naive datetimes are not allowed")

    if pd.api.types.is_numeric_dtype(non_null):
        return pd.to_datetime(values, unit=_parse_epoch_unit(non_null), utc=True)

    parsed = pd.to_datetime(values, errors="raise")
    if getattr(parsed.dt, "tz", None) is None:
        raise ValueError(f"{column_name} must be timezone-aware UTC; naive timestamps are not allowed")
    if str(parsed.dt.tz) != "UTC":
        raise ValueError(f"{column_name} must already be UTC")
    return parsed.dt.tz_convert("UTC")


def _float_column(*, nullable=False, required=True):
    _require_pandera()
    return pa.Column(float, nullable=nullable, required=required, coerce=True)


def _int_column(*, nullable=False, required=True):
    _require_pandera()
    return pa.Column(int, nullable=nullable, required=required, coerce=True)


def _object_column(*, nullable=True, required=True):
    _require_pandera()
    return pa.Column(object, nullable=nullable, required=required, coerce=False)


def _datetime_column(*, required=True):
    _require_pandera()
    return pa.Column(pd.DatetimeTZDtype(unit="ns", tz="UTC"), nullable=False, required=required, coerce=False)


def _custom_value_column(dtype_spec, *, nullable):
    if dtype_spec is None:
        return _object_column(nullable=nullable)

    normalized = str(dtype_spec).lower()
    if normalized in {"float", "float64", "double", "number"}:
        return _float_column(nullable=nullable)
    if normalized in {"int", "int64", "integer"}:
        return _int_column(nullable=nullable)
    if normalized in {"bool", "boolean"}:
        _require_pandera()
        return pa.Column(bool, nullable=nullable, required=True, coerce=True)
    if normalized in {"str", "string"}:
        _require_pandera()
        return pa.Column(str, nullable=nullable, required=True, coerce=False)
    if normalized.startswith("datetime"):
        return _datetime_column()

    _require_pandera()
    return pa.Column(dtype_spec, nullable=nullable, required=True, coerce=False)


def _build_contract_definition(
    *,
    name,
    dataset_kind,
    required_columns,
    optional_columns=None,
    allow_extra_columns=False,
    availability_timestamp_policy=None,
    timezone_policy="utc_required",
    duplicate_policy="index_unique",
    declared_value_columns=None,
    declared_value_dtypes=None,
):
    contract = {
        "name": str(name),
        "dataset_kind": str(dataset_kind),
        "schema_version": _CONTRACT_SCHEMA_VERSION,
        "semantic_version": _CONTRACT_SEMANTIC_VERSION,
        "required_columns": list(required_columns or []),
        "optional_columns": list(optional_columns or []),
        "allow_extra_columns": bool(allow_extra_columns),
        "availability_timestamp_policy": _json_ready(availability_timestamp_policy or {}),
        "timezone_policy": str(timezone_policy),
        "duplicate_policy": str(duplicate_policy),
        "declared_value_columns": list(declared_value_columns or []),
        "declared_value_dtypes": dict(declared_value_dtypes or {}),
    }
    contract["contract_hash"] = _payload_hash(contract)
    return contract


def build_dataset_manifest(
    *,
    name,
    dataset_kind,
    frame,
    contract,
    source=None,
    validation=None,
    upstream_manifests=None,
    source_fingerprint=None,
):
    upstream_payloads = [
        manifest.to_dict() if hasattr(manifest, "to_dict") else dict(manifest or {})
        for manifest in list(upstream_manifests or [])
    ]
    manifest = DatasetManifest(
        name=str(name),
        dataset_kind=str(dataset_kind),
        created_at=pd.Timestamp.now(tz="UTC").isoformat(),
        row_count=int(len(frame)),
        column_count=int(len(getattr(frame, "columns", []))),
        columns=list(getattr(frame, "columns", [])),
        source_fingerprint=str(source_fingerprint or frame_fingerprint(frame)),
        contract=dict(contract or {}),
        source=dict(_json_ready(source or {})),
        validation=dict(_json_ready(validation or {"status": "pass", "errors": []})),
        upstream_contract_hashes=[
            payload.get("contract", {}).get("contract_hash")
            for payload in upstream_payloads
            if payload.get("contract", {}).get("contract_hash") is not None
        ],
        upstream_dataset_names=[payload.get("name") for payload in upstream_payloads if payload.get("name")],
    )
    return manifest.to_dict()


def build_dataset_bundle_manifest(frame, *, name, upstream_manifests=None, validation=None, source=None):
    upstream_payloads = [
        manifest.to_dict() if hasattr(manifest, "to_dict") else dict(manifest or {})
        for manifest in list(upstream_manifests or [])
    ]
    contract = _build_contract_definition(
        name=f"{name}_bundle",
        dataset_kind="derived_dataset",
        required_columns=list(getattr(frame, "columns", [])),
        optional_columns=[],
        allow_extra_columns=False,
        duplicate_policy="index_unique",
    )
    return build_dataset_manifest(
        name=name,
        dataset_kind="derived_dataset",
        frame=pd.DataFrame(frame),
        contract=contract,
        source={
            "source_name": "pipeline_state",
            **dict(source or {}),
        },
        validation=validation or {"status": "pass", "errors": []},
        upstream_manifests=upstream_payloads,
    )


def _attach_manifest(frame, manifest):
    frame.attrs["dataset_manifest"] = manifest
    return frame


def _validate_with_schema(frame, schema, dataset_name):
    _require_pandera()
    try:
        return schema.validate(pd.DataFrame(frame).copy(), lazy=True)
    except (SchemaError, SchemaErrors) as exc:
        _contract_failure(dataset_name, exc)


def validate_market_frame_contract(frame, *, market="spot", dataset_name="market_data", source=None):
    validated = pd.DataFrame(frame).copy()
    for column in _OPTIONAL_MARKET_BAR_COLUMNS:
        if column not in validated.columns:
            validated[column] = 0.0
    _ensure_utc_datetime_index(validated, dataset_name)
    schema = pa.DataFrameSchema(
        {
            "open": _float_column(nullable=False),
            "high": _float_column(nullable=False),
            "low": _float_column(nullable=False),
            "close": _float_column(nullable=False),
            "volume": _float_column(nullable=False),
            "quote_volume": _float_column(nullable=False),
            "trades": _int_column(nullable=False),
            "taker_buy_base_vol": _float_column(nullable=False, required=False),
            "taker_buy_quote_vol": _float_column(nullable=False, required=False),
        },
        strict=False,
        coerce=True,
        ordered=False,
    )
    validated = _validate_with_schema(validated, schema, dataset_name)
    contract = _build_contract_definition(
        name=f"{market}_market_bars",
        dataset_kind="market_bars",
        required_columns=_MARKET_BAR_COLUMNS,
        optional_columns=_OPTIONAL_MARKET_BAR_COLUMNS,
        allow_extra_columns=True,
        availability_timestamp_policy={"index": "bar_open_time"},
        duplicate_policy="index_unique",
    )
    manifest = build_dataset_manifest(
        name=dataset_name,
        dataset_kind="market_bars",
        frame=validated,
        contract=contract,
        source=source,
        validation={"status": "pass", "errors": []},
    )
    return _attach_manifest(validated, manifest), manifest


def validate_prefixed_bar_frame_contract(frame, *, prefix, dataset_name, source=None):
    validated = pd.DataFrame(frame).copy()
    _ensure_utc_datetime_index(validated, dataset_name)
    required_columns = [f"{prefix}_open", f"{prefix}_high", f"{prefix}_low", f"{prefix}_close"]
    schema = pa.DataFrameSchema(
        {
            f"{prefix}_open": _float_column(nullable=False),
            f"{prefix}_high": _float_column(nullable=False),
            f"{prefix}_low": _float_column(nullable=False),
            f"{prefix}_close": _float_column(nullable=False),
        },
        strict=False,
        coerce=True,
        ordered=False,
    )
    validated = _validate_with_schema(validated, schema, dataset_name)
    contract = _build_contract_definition(
        name=f"{prefix}_bar_context",
        dataset_kind="context_bars",
        required_columns=required_columns,
        optional_columns=[],
        allow_extra_columns=True,
        availability_timestamp_policy={"index": "bar_open_time"},
        duplicate_policy="index_unique",
    )
    manifest = build_dataset_manifest(
        name=dataset_name,
        dataset_kind="context_bars",
        frame=validated,
        contract=contract,
        source=source,
        validation={"status": "pass", "errors": []},
    )
    return _attach_manifest(validated, manifest), manifest


def validate_funding_frame_contract(frame, *, dataset_name="funding_context", source=None):
    validated = pd.DataFrame(frame).copy()
    _ensure_utc_datetime_index(validated, dataset_name)
    schema = pa.DataFrameSchema(
        {
            "funding_rate": _float_column(nullable=False),
            "funding_mark_price": _float_column(nullable=True),
        },
        strict=False,
        coerce=True,
        ordered=False,
    )
    validated = _validate_with_schema(validated, schema, dataset_name)
    contract = _build_contract_definition(
        name="futures_funding_context",
        dataset_kind="funding_context",
        required_columns=["funding_rate"],
        optional_columns=["funding_mark_price"],
        allow_extra_columns=True,
        availability_timestamp_policy={"index": "publication_time"},
        duplicate_policy="index_unique",
    )
    manifest = build_dataset_manifest(
        name=dataset_name,
        dataset_kind="funding_context",
        frame=validated,
        contract=contract,
        source=source,
        validation={"status": "pass", "errors": []},
    )
    return _attach_manifest(validated, manifest), manifest


def validate_numeric_context_frame_contract(frame, *, dataset_name, contract_name, required_columns, source=None):
    validated = pd.DataFrame(frame).copy()
    _ensure_utc_datetime_index(validated, dataset_name)
    schema = pa.DataFrameSchema(
        {column: _float_column(nullable=True) for column in required_columns},
        strict=False,
        coerce=True,
        ordered=False,
    )
    validated = _validate_with_schema(validated, schema, dataset_name)
    contract = _build_contract_definition(
        name=contract_name,
        dataset_kind="context_metrics",
        required_columns=required_columns,
        optional_columns=[],
        allow_extra_columns=True,
        availability_timestamp_policy={"index": "publication_time"},
        duplicate_policy="index_unique",
    )
    manifest = build_dataset_manifest(
        name=dataset_name,
        dataset_kind="context_metrics",
        frame=validated,
        contract=contract,
        source=source,
        validation={"status": "pass", "errors": []},
    )
    return _attach_manifest(validated, manifest), manifest


def validate_futures_context_bundle(context, *, symbol=None, interval=None, source_name="binance_futures_context"):
    validated = {}
    manifests = {}
    spec_map = {
        "mark_price": lambda frame: validate_prefixed_bar_frame_contract(
            frame,
            prefix="mark",
            dataset_name=f"futures_context_mark_price_{str(symbol or 'unknown').lower()}",
            source={"source_name": source_name, "symbol": symbol, "interval": interval, "key": "mark_price"},
        ),
        "premium_index": lambda frame: validate_prefixed_bar_frame_contract(
            frame,
            prefix="premium",
            dataset_name=f"futures_context_premium_index_{str(symbol or 'unknown').lower()}",
            source={"source_name": source_name, "symbol": symbol, "interval": interval, "key": "premium_index"},
        ),
        "funding": lambda frame: validate_funding_frame_contract(
            frame,
            dataset_name=f"futures_context_funding_{str(symbol or 'unknown').lower()}",
            source={"source_name": source_name, "symbol": symbol, "interval": interval, "key": "funding"},
        ),
        "open_interest": lambda frame: validate_numeric_context_frame_contract(
            frame,
            dataset_name=f"futures_context_open_interest_{str(symbol or 'unknown').lower()}",
            contract_name="futures_open_interest_context",
            required_columns=["sumOpenInterest", "sumOpenInterestValue"],
            source={"source_name": source_name, "symbol": symbol, "interval": interval, "key": "open_interest"},
        ),
        "taker_flow": lambda frame: validate_numeric_context_frame_contract(
            frame,
            dataset_name=f"futures_context_taker_flow_{str(symbol or 'unknown').lower()}",
            contract_name="futures_taker_flow_context",
            required_columns=["buySellRatio", "buyVol", "sellVol"],
            source={"source_name": source_name, "symbol": symbol, "interval": interval, "key": "taker_flow"},
        ),
        "global_long_short": lambda frame: validate_numeric_context_frame_contract(
            frame,
            dataset_name=f"futures_context_global_long_short_{str(symbol or 'unknown').lower()}",
            contract_name="futures_global_long_short_context",
            required_columns=["longShortRatio", "longAccount", "shortAccount"],
            source={"source_name": source_name, "symbol": symbol, "interval": interval, "key": "global_long_short"},
        ),
        "basis": lambda frame: validate_numeric_context_frame_contract(
            frame,
            dataset_name=f"futures_context_basis_{str(symbol or 'unknown').lower()}",
            contract_name="futures_basis_context",
            required_columns=["basisRate", "basis", "futuresPrice", "indexPrice"],
            source={"source_name": source_name, "symbol": symbol, "interval": interval, "key": "basis"},
        ),
    }
    for key, value in dict(context or {}).items():
        if key in spec_map and value is not None and not pd.DataFrame(value).empty:
            validated_value, manifest = spec_map[key](value)
            validated[key] = validated_value
            manifests[key] = manifest
        else:
            validated[key] = value
    return validated, manifests


def validate_market_context_frames(frames, *, market="spot", interval=None, group_name="context"):
    validated = {}
    manifests = {}
    for symbol, frame in dict(frames or {}).items():
        dataset_name = f"{group_name}_{str(symbol).lower()}"
        validated_frame, manifest = validate_market_frame_contract(
            frame,
            market=market,
            dataset_name=dataset_name,
            source={"source_name": group_name, "symbol": symbol, "market": market, "interval": interval},
        )
        validated[symbol] = validated_frame
        manifests[symbol] = manifest
    return validated, manifests


def validate_reference_overlay_frame_contract(frame, *, dataset_name="reference_overlay_data", source=None):
    validated = pd.DataFrame(frame).copy()
    _ensure_utc_datetime_index(validated, dataset_name)
    if not validated.empty and not any(column in validated.columns for column in _REFERENCE_COLUMNS):
        raise ValueError(f"{dataset_name} must include at least one contracted reference column")
    schema = pa.DataFrameSchema(
        {column: _float_column(nullable=True, required=False) for column in _REFERENCE_COLUMNS},
        strict=False,
        coerce=True,
        ordered=False,
    )
    validated = _validate_with_schema(validated, schema, dataset_name)
    contract = _build_contract_definition(
        name="reference_overlay_data",
        dataset_kind="reference_overlay",
        required_columns=[],
        optional_columns=_REFERENCE_COLUMNS,
        allow_extra_columns=True,
        availability_timestamp_policy={"index": "publication_time"},
        duplicate_policy="index_unique",
    )
    manifest = build_dataset_manifest(
        name=dataset_name,
        dataset_kind="reference_overlay",
        frame=validated,
        contract=contract,
        source=source,
        validation={"status": "pass", "errors": []},
    )
    return _attach_manifest(validated, manifest), manifest


def validate_custom_source_contract(
    frame,
    *,
    dataset_name,
    timestamp_column,
    availability_column,
    value_columns,
    value_dtypes=None,
    source=None,
    availability_is_assumed=False,
):
    selected_value_columns = list(value_columns or [])
    if not selected_value_columns:
        raise ValueError("Custom data contract requires explicit value_columns")

    prepared = pd.DataFrame(frame).copy()
    prepared[timestamp_column] = parse_timestamp_series_utc(prepared[timestamp_column], timestamp_column)
    prepared[availability_column] = parse_timestamp_series_utc(prepared[availability_column], availability_column)

    if (prepared[availability_column] < prepared[timestamp_column]).fillna(False).any():
        raise ValueError(f"{dataset_name} requires {availability_column!r} >= {timestamp_column!r}")

    declared_dtypes = dict(value_dtypes or {})
    schema_columns = {
        timestamp_column: _datetime_column(),
        availability_column: _datetime_column(),
    }
    resolved_dtypes = {}
    for column in selected_value_columns:
        if column not in prepared.columns:
            raise ValueError(f"{dataset_name} missing declared custom value column {column!r}")
        resolved_dtypes[column] = declared_dtypes.get(column, str(prepared[column].dtype))
        schema_columns[column] = _custom_value_column(
            resolved_dtypes[column],
            nullable=bool(prepared[column].isna().any()),
        )

    schema = pa.DataFrameSchema(
        schema_columns,
        strict=True,
        coerce=False,
        ordered=False,
    )
    validated = _validate_with_schema(prepared, schema, dataset_name)
    contract = _build_contract_definition(
        name=f"custom_dataset_{dataset_name}",
        dataset_kind="custom_data",
        required_columns=[timestamp_column, availability_column, *selected_value_columns],
        optional_columns=[],
        allow_extra_columns=False,
        availability_timestamp_policy={
            "timestamp_column": timestamp_column,
            "availability_column": availability_column,
            "availability_is_assumed": bool(availability_is_assumed),
        },
        duplicate_policy="row_level_allowed",
        declared_value_columns=selected_value_columns,
        declared_value_dtypes=resolved_dtypes,
    )
    source_payload = dict(source or {})
    source_path = source_payload.get("path")
    if source_path is not None and Path(source_path).exists():
        source_payload["source_file_sha256"] = file_sha256(source_path)
    manifest = build_dataset_manifest(
        name=dataset_name,
        dataset_kind="custom_data",
        frame=validated,
        contract=contract,
        source=source_payload,
        validation={"status": "pass", "errors": []},
    )
    return _attach_manifest(validated, manifest), manifest


__all__ = [
    "DatasetManifest",
    "build_dataset_bundle_manifest",
    "build_dataset_manifest",
    "parse_timestamp_series_utc",
    "validate_custom_source_contract",
    "validate_funding_frame_contract",
    "validate_futures_context_bundle",
    "validate_market_context_frames",
    "validate_market_frame_contract",
    "validate_numeric_context_frame_contract",
    "validate_prefixed_bar_frame_contract",
    "validate_reference_overlay_frame_contract",
]