from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExecutionPolicy:
    adapter: str = "bar_surrogate"
    order_type: str = "market"
    time_in_force: str = "IOC"
    participation_cap: float = 0.10
    min_fill_ratio: float = 0.25
    action_latency_bars: int = 0
    max_order_age_bars: int = 1
    cancel_replace_bars: int = 1
    force_simulation: bool = False

    def to_dict(self):
        return asdict(self)


def resolve_execution_policy(config=None):
    if isinstance(config, ExecutionPolicy):
        return config

    config = dict(config or {})
    adapter = str(config.get("adapter", "bar_surrogate")).lower()
    adapter_aliases = {
        "bar": "bar_surrogate",
        "surrogate": "bar_surrogate",
        "nautilus_surrogate": "bar_surrogate",
    }
    adapter = adapter_aliases.get(adapter, adapter)
    return ExecutionPolicy(
        adapter=adapter,
        order_type=str(config.get("order_type", "market")).lower(),
        time_in_force=str(config.get("time_in_force", "IOC")).upper(),
        participation_cap=max(0.0, float(config.get("participation_cap", 0.10))),
        min_fill_ratio=max(0.0, min(1.0, float(config.get("min_fill_ratio", 0.25)))),
        action_latency_bars=max(0, int(config.get("action_latency_bars", 0))),
        max_order_age_bars=max(1, int(config.get("max_order_age_bars", 1))),
        cancel_replace_bars=max(1, int(config.get("cancel_replace_bars", 1))),
        force_simulation=bool(config.get("force_simulation", False)),
    )


__all__ = ["ExecutionPolicy", "resolve_execution_policy"]