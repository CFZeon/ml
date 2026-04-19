from __future__ import annotations


try:  # pragma: no cover - optional dependency boundary only
    import nautilus_trader  # type: ignore

    NAUTILUS_AVAILABLE = True
except Exception:  # pragma: no cover
    nautilus_trader = None
    NAUTILUS_AVAILABLE = False


class NautilusExecutionAdapter:
    """Execution adapter boundary for future NautilusTrader-backed simulation.

    In environments without NautilusTrader installed, the backtest stack falls back
    to the repo's deterministic bar-volume execution surrogate while preserving the
    same adapter selection surface.
    """

    name = "nautilus"

    def __init__(self):
        self.available = bool(NAUTILUS_AVAILABLE)
        self.backend = "nautilus" if self.available else "nautilus_surrogate"


__all__ = ["NAUTILUS_AVAILABLE", "NautilusExecutionAdapter"]