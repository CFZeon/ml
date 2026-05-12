"""Config loading and validation for user-facing experiment entrypoints."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from core import build_indicator, list_available_indicators
from core.regimes.detectors import is_native_regime_detector_spec
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


def _clone_mapping_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    cloned: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            cloned.append(_clone_mapping(item))
    return cloned


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


def _infer_legacy_regime_method(primary_detector: Mapping[str, Any] | None) -> str:
    detector = dict(primary_detector or {})
    params = dict(detector.get("params") or {})
    configured = str(params.get("method", detector.get("method", "")) or "").strip().lower()
    if configured in {"explicit", "hmm"}:
        return configured

    detector_type = str(detector.get("type", "explicit") or "explicit").strip().lower()
    if "hmm" in detector_type or detector_type in {"hidden_markov", "filtered_hmm"}:
        return "hmm"
    return "explicit"


def _select_primary_regime_detector(regime_config: Mapping[str, Any]) -> dict[str, Any] | None:
    detectors = _clone_mapping_list(regime_config.get("detectors"))
    if not detectors:
        return None

    ensemble = _clone_mapping(regime_config.get("ensemble") or {})
    primary_name = str(ensemble.get("primary_detector", "") or "").strip()
    if primary_name:
        for detector in detectors:
            if str(detector.get("name", "") or "").strip() == primary_name:
                return detector

    for detector in detectors:
        if bool(detector.get("primary", False)):
            return detector
    return detectors[0]


def _apply_regime_orchestration_compatibility(config: dict[str, Any]) -> dict[str, Any]:
    working = _clone_mapping(config)
    regime = _clone_mapping(working.get("regime") or {})
    feature_adaptation = _clone_mapping(working.get("feature_adaptation") or {})
    model_library = _clone_mapping(working.get("model_library") or {})
    router = _clone_mapping(working.get("router") or {})
    maintenance = _clone_mapping(working.get("maintenance") or {})
    model = _clone_mapping(working.get("model") or {})

    detectors = _clone_mapping_list(regime.get("detectors"))
    if detectors:
        primary_detector = _select_primary_regime_detector(regime)
        primary_params = dict((primary_detector or {}).get("params") or {})
        regime.setdefault("enabled", True)
        regime.setdefault("column_name", "regime")
        if is_native_regime_detector_spec(primary_detector):
            if regime.get("method") in (None, ""):
                regime["method"] = None
        else:
            regime.setdefault("method", _infer_legacy_regime_method(primary_detector))

        derived_n_regimes = (
            regime.get("n_regimes")
            or primary_params.get("n_regimes")
            or primary_params.get("state_count")
            or (primary_detector or {}).get("n_regimes")
        )
        if derived_n_regimes is not None:
            regime["n_regimes"] = int(derived_n_regimes)

        warmup_bars = [
            int(detector.get("warmup_bars") or 0)
            for detector in detectors
            if detector.get("warmup_bars") is not None
        ]
        if regime.get("feature_lookback") is not None:
            warmup_bars.append(int(regime.get("feature_lookback")))
        if warmup_bars:
            regime["feature_lookback"] = max(80, max(warmup_bars))

        if not is_native_regime_detector_spec(primary_detector):
            compatibility_adapter = _clone_mapping(regime.get("compatibility_adapter") or {})
            compatibility_adapter.update(
                {
                    "enabled": True,
                    "source": "experiments.config",
                    "primary_detector": str(
                        (primary_detector or {}).get("name")
                        or (primary_detector or {}).get("type")
                        or "detector_0"
                    ),
                    "derived_method": regime.get("method", "explicit"),
                    "derived_n_regimes": regime.get("n_regimes"),
                }
            )
            regime["compatibility_adapter"] = compatibility_adapter

    if feature_adaptation or model_library or router or maintenance:
        regime_aware = _clone_mapping(model.get("regime_aware") or {})
        regime_aware.setdefault("enabled", True)
        regime_aware.setdefault(
            "strategy",
            "specialist" if list(model_library.get("specialists") or []) else "feature",
        )
        regime_aware.setdefault(
            "min_samples_per_regime",
            int(
                model_library.get("min_samples_per_regime")
                or (_clone_mapping(model_library.get("specialist_defaults") or {}).get("min_training_samples") or 40)
            ),
        )
        regime_aware.setdefault("regime_column", str(regime.get("column_name", "regime")))

        scaling = _clone_mapping(feature_adaptation.get("scaling") or {})
        selection = _clone_mapping(feature_adaptation.get("selection") or {})
        interaction_budget = _clone_mapping(feature_adaptation.get("interaction_budget") or {})
        if interaction_budget or scaling:
            regime_aware.setdefault(
                "regime_interactions",
                bool(interaction_budget.get("enabled", scaling.get("mode") == "regime_conditioned")),
            )
        if interaction_budget.get("max_features") is not None:
            regime_aware.setdefault("max_interaction_features", int(interaction_budget.get("max_features")))
        if interaction_budget.get("max_regimes") is not None:
            regime_aware.setdefault("max_interaction_regimes", int(interaction_budget.get("max_regimes")))
        if selection.get("max_regime_states") is not None:
            regime_aware.setdefault("max_dummy_cardinality", int(selection.get("max_regime_states")))

        coverage_config = _clone_mapping(regime_aware.get("coverage_config") or {})
        coverage_config.update(_clone_mapping(model_library.get("coverage_config") or {}))
        if coverage_config:
            regime_aware["coverage_config"] = coverage_config

        if feature_adaptation:
            regime_aware.setdefault("feature_adaptation", feature_adaptation)
        if model_library:
            regime_aware.setdefault("model_library", model_library)
        if router:
            regime_aware.setdefault("router", router)
        if maintenance:
            regime_aware.setdefault("maintenance", maintenance)
        if feature_adaptation.get("disable_incompatible_features") is not None:
            regime_aware.setdefault(
                "disable_incompatible_features",
                bool(feature_adaptation.get("disable_incompatible_features")),
            )
        model["regime_aware"] = regime_aware

    if regime:
        working["regime"] = regime
    if model:
        working["model"] = model
    return working


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

    regime = config.get("regime")
    if regime is not None and not isinstance(regime, Mapping):
        errors.append("regime must be a mapping when provided")
    elif isinstance(regime, Mapping) and regime.get("detectors") is not None and not isinstance(regime.get("detectors"), list):
        errors.append("regime.detectors must be a list when provided")
    elif isinstance(regime, Mapping):
        detectors = [dict(item) for item in list(regime.get("detectors") or []) if isinstance(item, Mapping)]
        enabled_native = [detector for detector in detectors if detector.get("enabled", True) is not False and is_native_regime_detector_spec(detector)]
        if len(enabled_native) > 1:
            errors.append("regime.detectors supports at most one enabled native detector before Slice 5")

        primary_detector = _select_primary_regime_detector(regime)
        if is_native_regime_detector_spec(primary_detector) and regime.get("method") not in (None, ""):
            errors.append("regime.method cannot be combined with a native primary regime detector")

    feature_adaptation = config.get("feature_adaptation")
    if feature_adaptation is not None and not isinstance(feature_adaptation, Mapping):
        errors.append("feature_adaptation must be a mapping when provided")

    model_library = config.get("model_library")
    if model_library is not None and not isinstance(model_library, Mapping):
        errors.append("model_library must be a mapping when provided")
    elif isinstance(model_library, Mapping) and model_library.get("specialists") is not None and not isinstance(model_library.get("specialists"), list):
        errors.append("model_library.specialists must be a list when provided")

    router = config.get("router")
    if router is not None and not isinstance(router, Mapping):
        errors.append("router must be a mapping when provided")

    maintenance = config.get("maintenance")
    if maintenance is not None and not isinstance(maintenance, Mapping):
        errors.append("maintenance must be a mapping when provided")

    return errors


def _normalize_special_values(config: dict[str, Any]) -> dict[str, Any]:
    labels = dict(config.get("labels") or {})
    pt_sl = labels.get("pt_sl")
    if isinstance(pt_sl, list):
        labels["pt_sl"] = tuple(pt_sl)
        config["labels"] = labels
    return config


def _build_shared_quick_smoke_overrides(config: Mapping[str, Any]) -> dict[str, Any]:
    working = _clone_mapping(config)
    data = _clone_mapping(working.get("data") or {})
    features = _clone_mapping(working.get("features") or {})
    configured_lags = features.get("lags") or [1, 3, 6]
    try:
        feature_lag_bars = max(int(value) for value in configured_lags)
    except (TypeError, ValueError):
        feature_lag_bars = 6
    overrides: dict[str, Any] = {
        "features": {
            "lookahead_guard": {"enabled": False},
        },
        "feature_selection": {
            "enabled": False,
        },
        "feature_governance": {
            "enabled": False,
        },
    }

    context_symbols = list(
        data.get("context_symbols")
        or (_clone_mapping(data.get("cross_asset_context") or {}).get("symbols") or [])
    )
    if len(context_symbols) > 1:
        overrides["data"] = {"context_symbols": context_symbols[:1]}

    context_timeframes = list(features.get("context_timeframes") or [])
    if len(context_timeframes) > 1:
        overrides["features"]["context_timeframes"] = context_timeframes[:1]

    model = _clone_mapping(working.get("model") or {})
    cv_method = str(model.get("cv_method", "") or "").strip().lower()
    model_overrides: dict[str, Any] = {}
    if cv_method == "walk_forward":
        model_overrides.update(
            {
                "n_splits": 1,
                "train_size": min(int(model.get("train_size", 240) or 240), 240),
                "test_size": min(int(model.get("test_size", 48) or 48), 48),
                "gap": max(feature_lag_bars, min(int(model.get("gap", feature_lag_bars) or feature_lag_bars), 6)),
            }
        )
    elif cv_method == "cpcv":
        model_overrides.update(
            {
                "n_blocks": min(int(model.get("n_blocks", 2) or 2), 2),
                "test_blocks": 1,
                "validation_fraction": min(float(model.get("validation_fraction", 0.1) or 0.1), 0.1),
            }
        )
    if model.get("meta_n_splits") is not None:
        model_overrides["meta_n_splits"] = min(int(model.get("meta_n_splits", 2) or 2), 2)
    if model_overrides:
        overrides["model"] = model_overrides

    return overrides


def load_experiment_config(config_source: str | Path | Mapping[str, Any], *, quick: bool = False) -> ResolvedExperimentConfig:
    """Load, validate, and expand a user-facing experiment config."""

    raw_config, config_path = _read_raw_config(config_source)
    working_config = _clone_mapping(raw_config)
    if quick:
        working_config = clone_config_with_overrides(working_config, _clone_mapping(working_config.get("quick_overrides") or {}))
        if not bool((_clone_mapping(working_config.get("automl") or {})).get("enabled", False)):
            working_config = clone_config_with_overrides(
                working_config,
                _build_shared_quick_smoke_overrides(working_config),
            )
    user_config = _clone_any(working_config)
    working_config = _apply_regime_orchestration_compatibility(working_config)

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
    cross_asset_context = _clone_mapping(data_section.get("cross_asset_context") or {})
    if cross_asset_context or context_symbols:
        cross_asset_context["symbols"] = [str(context_symbol) for context_symbol in context_symbols]
        data_section["cross_asset_context"] = cross_asset_context
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

    resolved_regime = _clone_mapping(resolved.get("regime") or {})
    if is_native_regime_detector_spec(_select_primary_regime_detector(resolved_regime)) and resolved_regime.get("method") is None:
        resolved_regime.pop("method", None)
        resolved["regime"] = resolved_regime

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
        raw_config=user_config,
        config=resolved,
        config_path=config_path,
        quick_mode=bool(quick),
    )
