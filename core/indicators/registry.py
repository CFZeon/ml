"""Indicator registry and execution helpers."""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType
from typing import Any

import pandas as pd

from .base import Indicator, IndicatorRunResult


INDICATOR_REGISTRY: dict[str, type[Indicator]] = {}
INDICATOR_DISCOVERY_ERRORS: dict[str, str] = {}

_DISCOVERED_MODULES: set[str] = set()
_DISCOVERY_PACKAGES = ("core.indicators", "indicators")
_DISCOVERY_SKIP = {"base", "registry"}


def _normalize_kind(kind: Any) -> str:
    normalized = str(kind or "").strip()
    if not normalized:
        raise ValueError("Indicator kind must be a non-empty string")
    return normalized


def register_indicator(indicator_cls):
    if not isinstance(indicator_cls, type) or not issubclass(indicator_cls, Indicator):
        raise TypeError("Registered indicators must be Indicator subclasses")
    INDICATOR_REGISTRY[_normalize_kind(getattr(indicator_cls, "kind", None))] = indicator_cls
    return indicator_cls


def _normalize_function_output(*, input_frame: pd.DataFrame, output_frame: Any, kind: str, name: str) -> dict[str, pd.Series]:
    if isinstance(output_frame, pd.Series):
        output_frame = output_frame.to_frame(name=kind)
    if not isinstance(output_frame, pd.DataFrame):
        raise TypeError(f"Indicator {kind!r} must return a pandas DataFrame")
    if output_frame.empty:
        raise ValueError(f"Indicator {kind!r} did not return any feature columns")
    if not output_frame.index.equals(input_frame.index):
        raise ValueError(f"Indicator {kind!r} must preserve the input index exactly")
    if not output_frame.columns.is_unique:
        raise ValueError(f"Indicator {kind!r} returned duplicate output columns")

    namespace = f"{kind}_"
    normalized_outputs: dict[str, pd.Series] = {}
    for raw_column in output_frame.columns:
        column = _normalize_kind(raw_column)
        if column == kind:
            final_name = name
        elif column.startswith(namespace):
            final_name = f"{name}{column[len(kind):]}" if name != kind else column
        else:
            raise ValueError(
                f"Indicator {kind!r} output column {column!r} must equal {kind!r} or start with {namespace!r}"
            )
        if final_name in input_frame.columns:
            raise ValueError(f"Indicator {kind!r} output column {final_name!r} would overwrite an existing column")
        if final_name in normalized_outputs:
            raise ValueError(f"Indicator {kind!r} output column {final_name!r} is duplicated after namespacing")
        normalized_outputs[final_name] = output_frame[column].rename(final_name)
    return normalized_outputs


def _build_function_indicator_cls(module: ModuleType) -> type[Indicator]:
    kind = _normalize_kind(getattr(module, "KIND", None))
    compute_fn = getattr(module, "compute", None)
    if not callable(compute_fn):
        raise TypeError(f"Indicator module {module.__name__!r} must define compute(df, **params)")

    lookahead_safe = getattr(module, "LOOKAHEAD_SAFE", None)
    if lookahead_safe is not True:
        raise ValueError(f"Indicator module {module.__name__!r} must set LOOKAHEAD_SAFE = True")

    stateful = getattr(module, "STATEFUL", None)
    if not isinstance(stateful, bool):
        raise ValueError(f"Indicator module {module.__name__!r} must declare STATEFUL as a bool")

    required_columns = tuple(getattr(module, "REQUIRED_COLUMNS", ()) or ())
    for column in required_columns:
        _normalize_kind(column)

    default_name = str(getattr(module, "DEFAULT_NAME", kind) or kind)
    description = str(getattr(module, "DESCRIPTION", "") or "")
    indicator_kind = kind
    indicator_required_columns = required_columns
    indicator_stateful = stateful
    indicator_description = description
    indicator_default_name = default_name

    class FunctionalIndicator(Indicator):
        kind = indicator_kind
        required_columns = indicator_required_columns

        def __init__(self, name=None, **params):
            self._params = dict(params)
            super().__init__(name=name)

        def default_name(self):
            return indicator_default_name

        def params(self):
            return dict(self._params)

        def describe(self, outputs):
            metadata = super().describe(outputs)
            metadata.update(
                {
                    "module": module.__name__,
                    "stateful": indicator_stateful,
                    "lookahead_safe": True,
                    "description": indicator_description,
                }
            )
            return metadata

        def compute(self, df):
            computed = compute_fn(df.copy(), **self._params)
            return _normalize_function_output(input_frame=df, output_frame=computed, kind=self.kind, name=self.name)

    FunctionalIndicator.__name__ = "".join(part.title() for part in kind.split("_")) + "Indicator"
    FunctionalIndicator.__qualname__ = FunctionalIndicator.__name__
    FunctionalIndicator.__module__ = module.__name__
    return FunctionalIndicator


def _discover_package(package_name: str) -> None:
    try:
        package = importlib.import_module(package_name)
    except ModuleNotFoundError:
        return

    package_paths = getattr(package, "__path__", None)
    if package_paths is None:
        return

    for module_info in pkgutil.iter_modules(package_paths, f"{package_name}."):
        short_name = module_info.name.rsplit(".", 1)[-1]
        if short_name.startswith("_") or short_name in _DISCOVERY_SKIP or module_info.name in _DISCOVERED_MODULES:
            continue
        try:
            module = importlib.import_module(module_info.name)
            _DISCOVERED_MODULES.add(module_info.name)
            if hasattr(module, "KIND") and callable(getattr(module, "compute", None)):
                register_indicator(_build_function_indicator_cls(module))
        except Exception as exc:
            INDICATOR_DISCOVERY_ERRORS[module_info.name] = f"{type(exc).__name__}: {exc}"


def discover_indicators() -> dict[str, type[Indicator]]:
    for package_name in _DISCOVERY_PACKAGES:
        _discover_package(package_name)
    return INDICATOR_REGISTRY


def list_available_indicators() -> list[str]:
    discover_indicators()
    return sorted(INDICATOR_REGISTRY)


def _unknown_indicator_error(kind: Any) -> ValueError:
    available = list_available_indicators()
    message = f"Unknown indicator kind={kind!r}. Available: {available}"
    if INDICATOR_DISCOVERY_ERRORS:
        message += f". Discovery errors: {INDICATOR_DISCOVERY_ERRORS}"
    return ValueError(message)


def build_indicator(spec):
    discover_indicators()

    if isinstance(spec, Indicator):
        return spec

    if isinstance(spec, str):
        if spec not in INDICATOR_REGISTRY:
            raise _unknown_indicator_error(spec)
        return INDICATOR_REGISTRY[spec]()

    if isinstance(spec, dict):
        raw_spec = dict(spec)
        kind = raw_spec.pop("kind", raw_spec.pop("indicator", None))
        params = raw_spec.pop("params", {}) or {}
        params = {**raw_spec, **params}
        if kind not in INDICATOR_REGISTRY:
            raise _unknown_indicator_error(kind)
        return INDICATOR_REGISTRY[kind](**params)

    raise TypeError("Indicator spec must be an Indicator, string kind, or config dict")


def build_indicators(specs):
    discover_indicators()
    return [build_indicator(spec) for spec in specs]


def run_indicators(df, indicators):
    """Compute indicators and return an enriched DataFrame plus metadata."""
    out = df.copy()
    results = []
    for indicator in build_indicators(indicators):
        result = indicator.run(out)
        out = out.join(result.to_frame())
        results.append(result)
    return IndicatorRunResult(frame=out, results=results)


def attach_indicators(df, indicators):
    """Backward-compatible helper returning only the enriched DataFrame."""
    return run_indicators(df, indicators).frame