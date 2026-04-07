"""Indicator registry and execution helpers."""

from .base import Indicator, IndicatorRunResult


INDICATOR_REGISTRY = {}


def register_indicator(indicator_cls):
    INDICATOR_REGISTRY[indicator_cls.kind] = indicator_cls
    return indicator_cls


def build_indicator(spec):
    if isinstance(spec, Indicator):
        return spec

    if isinstance(spec, str):
        if spec not in INDICATOR_REGISTRY:
            raise ValueError(f"Unknown indicator kind={spec!r}. Available: {sorted(INDICATOR_REGISTRY)}")
        return INDICATOR_REGISTRY[spec]()

    if isinstance(spec, dict):
        raw_spec = dict(spec)
        kind = raw_spec.pop("kind", raw_spec.pop("indicator", None))
        params = raw_spec.pop("params", {}) or {}
        params = {**raw_spec, **params}
        if kind not in INDICATOR_REGISTRY:
            raise ValueError(f"Unknown indicator kind={kind!r}. Available: {sorted(INDICATOR_REGISTRY)}")
        return INDICATOR_REGISTRY[kind](**params)

    raise TypeError("Indicator spec must be an Indicator, string kind, or config dict")


def build_indicators(specs):
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