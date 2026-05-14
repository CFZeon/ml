import unittest

from core.regimes import RegimeStateContract
from core.routing import HardSwitchRouter, replay_router_trace
from core.specialists import SpecialistHealthContract, SpecialistLibrarySnapshot, SpecialistSpec


def _make_router_library():
    return SpecialistLibrarySnapshot(
        symbol="BTCUSDT",
        timeframe="1h",
        fallback_model_id="fallback_generalist",
        specialists=[
            SpecialistSpec(
                model_id="fallback_generalist",
                symbol="BTCUSDT",
                timeframe="1h",
                compatible_regimes=[],
                estimator_family="logisticregression",
                metadata={"fallback_only": True, "lifecycle_state": "active"},
            ),
            SpecialistSpec(
                model_id="specialist::bull",
                symbol="BTCUSDT",
                timeframe="1h",
                compatible_regimes=["bull"],
                estimator_family="logisticregression",
                metadata={"lifecycle_state": "active"},
            ),
        ],
        health=[
            SpecialistHealthContract(model_id="fallback_generalist", fallback_only=True, stability_score=0.55, decay_score=0.05),
            SpecialistHealthContract(model_id="specialist::bull", compatible_regimes=["bull"], stability_score=0.8, decay_score=0.08),
        ],
    )


class RouterTraceReplayTest(unittest.TestCase):
    def test_replay_router_trace_is_deterministic_and_summarizes_switches(self):
        router = HardSwitchRouter(hysteresis_margin=0.05, min_persistence_bars=2, cooldown_bars=1)
        regime_states = [
            RegimeStateContract(as_of="2026-05-09T00:00:00Z", available_at="2026-05-09T00:00:00Z", label="bull", confidence=0.9),
            RegimeStateContract(as_of="2026-05-09T01:00:00Z", available_at="2026-05-09T01:00:00Z", label="bull", confidence=0.9),
            RegimeStateContract(as_of="2026-05-09T02:00:00Z", available_at="2026-05-09T02:00:00Z", label="flat", confidence=0.2),
        ]

        first = replay_router_trace(router, _make_router_library(), regime_states)
        second = replay_router_trace(router, _make_router_library(), regime_states)

        self.assertEqual(first["decision_trace"], second["decision_trace"])
        self.assertEqual(first["summary"]["decision_count"], 3)
        self.assertEqual(first["summary"]["switch_count"], 2)
        self.assertEqual(first["summary"]["executed_weight_turnover_total"], 2.0)
        self.assertEqual(first["summary"]["blocked_allocation_count"], 0)
        self.assertEqual(first["summary"]["route_reason_counts"]["persistence_hold"], 1)
        self.assertEqual(first["summary"]["selected_model_ids"][1], "specialist::bull")

    def test_replay_router_trace_counts_degraded_regime_availability(self):
        router = HardSwitchRouter(hysteresis_margin=0.0, min_persistence_bars=1, cooldown_bars=0, safe_mode_policy="fallback_only")
        regime_states = [
            RegimeStateContract(as_of="2026-05-09T00:00:00Z", available_at="2026-05-09T00:00:00Z", label="bull", confidence=0.9),
            RegimeStateContract(as_of="2026-05-09T01:00:00Z", available_at="2026-05-09T01:00:00Z", label="bull", confidence=0.9, warm=False),
            RegimeStateContract(
                as_of="2026-05-09T02:00:00Z",
                available_at="2026-05-09T02:00:00Z",
                label=0,
                confidence=0.0,
                warm=False,
                detector_outputs={"unavailable": 1},
            ),
        ]

        trace = replay_router_trace(router, _make_router_library(), regime_states)

        self.assertEqual(trace["summary"]["regime_availability_counts"], {"known": 1, "warm": 1, "unavailable": 1})
        self.assertEqual(trace["summary"]["safe_mode_action_counts"], {"fallback_only": 2})

    def test_replay_router_trace_uses_decision_timestamps_for_delayed_regime_recognition(self):
        router = HardSwitchRouter(hysteresis_margin=0.0, min_persistence_bars=1, cooldown_bars=0)
        decision_index = [
            "2026-05-09T00:00:00Z",
            "2026-05-09T01:00:00Z",
            "2026-05-09T02:00:00Z",
        ]
        regime_states = [
            RegimeStateContract(
                as_of=decision_index[0],
                available_at=decision_index[1],
                label="bull",
                confidence=0.9,
                warm=True,
                detector_outputs={"timing_blocked": 1},
                metadata={"availability_state": "timing_blocked"},
            ),
            RegimeStateContract(
                as_of=decision_index[1],
                available_at=decision_index[1],
                label="bull",
                confidence=0.9,
                warm=True,
            ),
            RegimeStateContract(
                as_of=decision_index[2],
                available_at=decision_index[2],
                label="bull",
                confidence=0.9,
                warm=True,
            ),
        ]

        trace = replay_router_trace(
            router,
            _make_router_library(),
            regime_states,
            decision_timestamps=decision_index,
        )

        self.assertEqual(trace["decision_trace"][0]["decision_timestamp"], "2026-05-09T00:00:00+00:00")
        self.assertEqual(trace["summary"]["regime_availability_counts"]["timing_blocked"], 1)
        self.assertEqual(trace["decision_trace"][0]["selected_model_id"], "fallback_generalist")


if __name__ == "__main__":
    unittest.main()