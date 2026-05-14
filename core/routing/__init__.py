"""Router contracts package."""

from .contracts import (
    BaseRouter,
    RouterManifest,
    RouterStateSnapshot,
    RoutingDecisionContract,
    RoutingScoreComponent,
)
from .diagnostics import build_admissible_router_regime_trace, replay_router_trace
from .router import HardSwitchRouter, WeightedRouter, build_router

__all__ = [
    "BaseRouter",
    "build_router",
    "build_admissible_router_regime_trace",
    "HardSwitchRouter",
    "replay_router_trace",
    "RouterManifest",
    "RouterStateSnapshot",
    "RoutingDecisionContract",
    "RoutingScoreComponent",
    "WeightedRouter",
]