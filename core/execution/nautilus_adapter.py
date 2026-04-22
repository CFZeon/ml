from __future__ import annotations


try:  # pragma: no cover - optional dependency boundary only
    import nautilus_trader  # type: ignore

    NAUTILUS_AVAILABLE = True
except Exception:  # pragma: no cover
    nautilus_trader = None
    NAUTILUS_AVAILABLE = False


class ExecutionAdapterUnavailableError(RuntimeError):
    """Raised when a requested execution adapter is unavailable and no simulation override was provided."""


class NautilusExecutionAdapter:
    """Execution adapter boundary for future NautilusTrader-backed simulation.

    In environments without NautilusTrader installed, the backtest stack falls back
    to the repo's deterministic bar-volume execution surrogate while preserving the
    same adapter selection surface.
    """

    name = "nautilus"

    def __init__(self, scenario_schedule=None, scenario_policy=None):
        self.available = bool(NAUTILUS_AVAILABLE)
        self.backend = "nautilus" if self.available else "nautilus_surrogate"
        self.scenario_schedule = scenario_schedule
        self.scenario_policy = dict(scenario_policy or {})

    def describe_scenarios(self):
        return {
            "configured": self.scenario_schedule is not None,
            "policy": dict(self.scenario_policy),
        }


__all__ = ["ExecutionAdapterUnavailableError", "NAUTILUS_AVAILABLE", "NautilusExecutionAdapter"]