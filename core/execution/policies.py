from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExecutionPolicy:
    adapter: str = "nautilus"
    order_type: str = "market"
    time_in_force: str = "IOC"
    participation_cap: float = 1.0
    min_fill_ratio: float = 0.0
    max_order_age_bars: int = 1
    cancel_replace_bars: int = 1

    def to_dict(self):
        return asdict(self)


def resolve_execution_policy(config=None):
    if isinstance(config, ExecutionPolicy):
        return config

    config = dict(config or {})
    return ExecutionPolicy(
        adapter=str(config.get("adapter", "nautilus")).lower(),
        order_type=str(config.get("order_type", "market")).lower(),
        time_in_force=str(config.get("time_in_force", "IOC")).upper(),
        participation_cap=max(0.0, float(config.get("participation_cap", 1.0))),
        min_fill_ratio=max(0.0, min(1.0, float(config.get("min_fill_ratio", 0.0)))),
        max_order_age_bars=max(1, int(config.get("max_order_age_bars", 1))),
        cancel_replace_bars=max(1, int(config.get("cancel_replace_bars", 1))),
    )


__all__ = ["ExecutionPolicy", "resolve_execution_policy"]