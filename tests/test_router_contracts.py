import unittest

from core.routing import (
    RouterManifest,
    RouterStateSnapshot,
    RoutingDecisionContract,
    RoutingScoreComponent,
)


class RouterContractsTest(unittest.TestCase):
    def test_router_contracts_round_trip_via_dict_payloads(self):
        component = RoutingScoreComponent(
            name="regime_confidence",
            value=0.82,
            weight=0.6,
            penalized=False,
            metadata={"source": "detector"},
        )
        decision = RoutingDecisionContract(
            as_of="2026-05-01T00:00:00Z",
            available_at="2026-05-01T00:00:00Z",
            selected_model_id="specialist::bull",
            weights={"specialist::bull": 1.0},
            regime_label="bull",
            regime_confidence=0.82,
            route_reason="highest_score",
            hysteresis_applied=True,
            cooldown_active=False,
            candidate_scores={"specialist::bull": 0.82, "fallback_generalist": 0.55},
            components=[component],
        )
        manifest = RouterManifest(
            router_type="score_router",
            score_component_names=["regime_confidence", "specialist_health"],
            policy_name="default",
            hysteresis_margin=0.05,
            min_persistence_bars=3,
            cooldown_bars=2,
        )
        state = RouterStateSnapshot(
            active_model_id="specialist::bull",
            last_switch_at="2026-05-01T00:00:00Z",
            cooldown_active=False,
            pending_challenger_id=None,
            pending_challenger_streak=0,
        )

        decision_roundtrip = RoutingDecisionContract.from_dict(decision.to_dict())
        manifest_roundtrip = RouterManifest.from_dict(manifest.to_dict())
        state_roundtrip = RouterStateSnapshot.from_dict(state.to_dict())

        self.assertEqual(decision_roundtrip.selected_model_id, "specialist::bull")
        self.assertEqual(decision_roundtrip.components[0].name, "regime_confidence")
        self.assertAlmostEqual(decision_roundtrip.candidate_scores["fallback_generalist"], 0.55)
        self.assertEqual(manifest_roundtrip.router_type, "score_router")
        self.assertEqual(manifest_roundtrip.min_persistence_bars, 3)
        self.assertEqual(state_roundtrip.active_model_id, "specialist::bull")
