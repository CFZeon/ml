"""Config loading and validation for user-facing experiment entrypoints."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from core import build_indicator, list_available_indicators
from example_utils import build_futures_research_config, build_spot_research_config, clone_config_with_overrides


DEFAULT_INDICATORS = [
    {"kind": "returns", "params": {"periods": [1, 3, 6]}},
    {"kind": "volatility", "params": {"window": 24}},
]
SUPPORTED_MARKETS = {"spot", "um_futures"}
PIPELINE_MANAGED_SECTIONS = {
    "automl",
    "backtest",
    "data",
    "data_quality",
    "experiment",
    "feature_selection",
    "features",
    "indicators",
    "labels",
    "model",
    "quick_overrides",
    "regime",
    "signals",
    "universe",
}


def _normalize_time_value(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


@dataclass(frozen=True)
class ResolvedExperimentConfig:
    """Resolved experiment config ready for ``ResearchPipeline``."""

    name: str
    raw_config: dict[str, Any]
    config: dict[str, Any]
    config_path: Path | None = None
    quick_mode: bool = False


def _clone_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return clone_config_with_overrides(dict(value or {}), None)


def _clone_any(value: Any) -> Any:
    return copy.deepcopy(value)


def _read_raw_config(config_source: str | Path | Mapping[str, Any]) -> tuple[dict[str, Any], Path | None]:
    if isinstance(config_source, Mapping):
        return _clone_mapping(config_source), None

    config_path = Path(config_source).expanduser().resolve()
    raw_text = config_path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(raw_text) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Config file {config_path} must contain a mapping at the top level")
    return _clone_mapping(loaded), config_path


def _normalize_indicator_specs(indicators: Any) -> list[Any]:
    raw_specs = DEFAULT_INDICATORS if indicators in (None, []) else indicators
    if not isinstance(raw_specs, list):
        raise ValueError("Config field indicators must be a list")

    normalized_specs: list[Any] = []
    for raw_spec in raw_specs:
        if isinstance(raw_spec, str):
            build_indicator(raw_spec)
            normalized_specs.append(raw_spec)
            continue
        if isinstance(raw_spec, Mapping):
            spec = _clone_mapping(raw_spec)
            kind = spec.pop("kind", spec.pop("indicator", None))
            if kind is None:
                raise ValueError("Each indicator mapping must include a 'kind' field")
            name = spec.pop("name", None)
            params = _clone_mapping(spec.pop("params", {}))
            params.update(spec)
            normalized = {"kind": str(kind)}
            if name is not None:
                normalized["name"] = str(name)
            if params:
                normalized["params"] = params
            build_indicator(normalized)
            normalized_specs.append(normalized)
            continue
        build_indicator(raw_spec)
        normalized_specs.append(raw_spec)
    return normalized_specs


def _collect_validation_errors(config: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []

    data = dict(config.get("data") or {})
    for field in ["symbol", "interval", "start", "end"]:
        if data.get(field) in (None, ""):
            errors.append(f"data.{field} is required")

    market = str(data.get("market", "spot") or "spot").strip().lower()
    if market not in SUPPORTED_MARKETS:
        errors.append(f"data.market must be one of {sorted(SUPPORTED_MARKETS)}, got {market!r}")

    try:
        _normalize_indicator_specs(config.get("indicators"))
    except Exception as exc:
        errors.append(str(exc))

    quick_overrides = config.get("quick_overrides")
    if quick_overrides is not None and not isinstance(quick_overrides, Mapping):
        errors.append("quick_overrides must be a mapping when provided")

    model = dict(config.get("model") or {})
    cv_method = str(model.get("cv_method", "cpcv") or "cpcv").strip().lower()
    if cv_method not in {"cpcv", "walk_forward"}:
        errors.append(f"model.cv_method must be 'cpcv' or 'walk_forward', got {cv_method!r}")

    labels = dict(config.get("labels") or {})
    if labels:
        label_kind = str(labels.get("kind", "triple_barrier") or "triple_barrier").strip().lower()
        if label_kind not in {"triple_barrier", "fixed_horizon", "trend_scanning"}:
            errors.append(
                "labels.kind must be one of ['fixed_horizon', 'trend_scanning', 'triple_barrier']"
            )

    return errors


def _normalize_special_values(config: dict[str, Any]) -> dict[str, Any]:
    labels = dict(config.get("labels") or {})
    pt_sl = labels.get("pt_sl")
    if isinstance(pt_sl, list):
        labels["pt_sl"] = tuple(pt_sl)
        config["labels"] = labels
    return config


def load_experiment_config(config_source: str | Path | Mapping[str, Any], *, quick: bool = False) -> ResolvedExperimentConfig:
    """Load, validate, and expand a user-facing experiment config."""

    raw_config, config_path = _read_raw_config(config_source)
    working_config = _clone_mapping(raw_config)
    if quick:
        working_config = clone_config_with_overrides(working_config, _clone_mapping(working_config.get("quick_overrides") or {}))

    validation_errors = _collect_validation_errors(working_config)
    if validation_errors:
        available = list_available_indicators()
        joined_errors = "\n- ".join(validation_errors)
        raise ValueError(f"Invalid experiment config:\n- {joined_errors}\nAvailable indicators: {available}")

    data_section = _clone_mapping(working_config.get("data") or {})
    market = str(data_section.get("market", "spot") or "spot").strip().lower()
    symbol = str(data_section["symbol"])
    interval = str(data_section["interval"])
    start = _normalize_time_value(data_section["start"])
    end = _normalize_time_value(data_section["end"])
    data_section["start"] = start
    data_section["end"] = end
    data_source = str(data_section.pop("source", "binance_bars") or "binance_bars")
    context_symbols = list(
        data_section.pop("context_symbols", None)
        or (data_section.get("cross_asset_context") or {}).get("symbols")
        or []
    )
    indicator_specs = _normalize_indicator_specs(working_config.get("indicators"))

    if market == "spot":
        pipeline_config = build_spot_research_config(
            symbol=symbol,
            interval=interval,
            start=start,
            end=end,
            indicators=indicator_specs,
            context_symbols=context_symbols,
        )
    else:
        pipeline_config = build_futures_research_config(
            symbol=symbol,
            interval=interval,
            start=start,
            end=end,
            indicators=indicator_specs,
            context_symbols=context_symbols,
        )

    overrides = {}
    for section in [
        "automl",
        "backtest",
        "data",
        "data_quality",
        "feature_selection",
        "features",
        "labels",
        "model",
        "regime",
        "signals",
        "universe",
    ]:
        if section not in working_config:
            continue
        if section == "data":
            overrides[section] = data_section
        else:
            overrides[section] = _clone_mapping(working_config.get(section) or {})
    overrides["indicators"] = indicator_specs

    resolved = clone_config_with_overrides(pipeline_config, overrides)
    passthrough_sections = {
        key: _clone_any(value)
        for key, value in working_config.items()
        if key not in PIPELINE_MANAGED_SECTIONS and key != "indicators"
    }
    if passthrough_sections:
        resolved = clone_config_with_overrides(resolved, passthrough_sections)
    experiment_section = _clone_mapping(working_config.get("experiment") or {})
    resolved["experiment"] = {
        **experiment_section,
        "name": str(experiment_section.get("name") or (config_path.stem if config_path is not None else "adhoc_experiment")),
        "config_path": None if config_path is None else str(config_path),
        "quick_mode": bool(quick),
        "data_source": data_source,
    }
    resolved = _normalize_special_values(resolved)
    return ResolvedExperimentConfig(
        name=str(resolved["experiment"]["name"]),
        raw_config=working_config,
        config=resolved,
        config_path=config_path,
        quick_mode=bool(quick),
    )
