import unittest

from core.regimes import RegimeStateContract
from core.routing import HardSwitchRouter, RoutingDecisionContract, WeightedRouter, build_router
from core.specialists import SpecialistHealthContract, SpecialistLibrarySnapshot, SpecialistSpec


def _make_active_specialist_library():
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
            SpecialistSpec(
                model_id="specialist::risk_off",
                symbol="BTCUSDT",
                timeframe="1h",
                compatible_regimes=["risk_off"],
                estimator_family="logisticregression",
                metadata={"lifecycle_state": "active"},
            ),
        ],
        health=[
            SpecialistHealthContract(model_id="fallback_generalist", fallback_only=True, stability_score=0.55, decay_score=0.1),
            SpecialistHealthContract(model_id="specialist::bull", compatible_regimes=["bull"], stability_score=0.82, decay_score=0.08),
            SpecialistHealthContract(model_id="specialist::risk_off", compatible_regimes=["risk_off"], stability_score=0.76, decay_score=0.09),
        ],
    )


class RouterRuntimeTest(unittest.TestCase):
    def test_hard_switch_router_respects_persistence_and_cooldown(self):
        router = HardSwitchRouter(hysteresis_margin=0.05, min_persistence_bars=2, cooldown_bars=1)
        state = router.initialize(_make_active_specialist_library())

        state, decision1 = router.select(
            state,
            RegimeStateContract(as_of="2026-05-09T00:00:00Z", available_at="2026-05-09T00:00:00Z", label="bull", confidence=0.9),
        )
        self.assertEqual(decision1.selected_model_id, "fallback_generalist")
        self.assertEqual(decision1.route_reason, "persistence_hold")
        self.assertEqual(state.pending_challenger_id, "specialist::bull")

        state, decision2 = router.select(
            state,
            RegimeStateContract(as_of="2026-05-09T01:00:00Z", available_at="2026-05-09T01:00:00Z", label="bull", confidence=0.9),
        )
        self.assertEqual(decision2.selected_model_id, "specialist::bull")
        self.assertEqual(state.active_model_id, "specialist::bull")

        state, decision3 = router.select(
            state,
            RegimeStateContract(as_of="2026-05-09T02:00:00Z", available_at="2026-05-09T02:00:00Z", label="risk_off", confidence=0.95),
        )
        self.assertFalse(decision3.cooldown_active)
        self.assertEqual(decision3.selected_model_id, "specialist::risk_off")
        self.assertEqual(decision3.route_reason, "initial_selection")
        self.assertFalse(decision3.metadata["candidate_eligibility"]["specialist::bull"]["eligible"])
        self.assertIn("regime_incompatible", decision3.metadata["candidate_eligibility"]["specialist::bull"]["reasons"])

    def test_weighted_router_returns_normalized_weights_and_diagnostics(self):
        router = WeightedRouter(allocation_temperature=0.5, hysteresis_margin=0.0, min_persistence_bars=1, cooldown_bars=0)
        state = router.initialize(_make_active_specialist_library())

        state, decision = router.select(
            state,
            RegimeStateContract(as_of="2026-05-09T00:00:00Z", available_at="2026-05-09T00:00:00Z", label="bull", confidence=0.8),
        )
        roundtrip = RoutingDecisionContract.from_dict(decision.to_dict())

        self.assertEqual(state.active_model_id, "specialist::bull")
        self.assertEqual(decision.selected_model_id, "specialist::bull")
        self.assertAlmostEqual(sum(decision.weights.values()), 1.0)
        self.assertGreater(decision.effective_model_count, 1.0)
        self.assertEqual(decision.allocation_control_reason, "mixture_allocation")
        self.assertGreater(decision.executed_weight_turnover, 0.0)
        self.assertIn("fallback_generalist", decision.candidate_scores)
        self.assertTrue(decision.components)
        self.assertEqual(roundtrip.selected_model_id, "specialist::bull")
        self.assertEqual(roundtrip.executed_candidate_ids, decision.executed_candidate_ids)

    def test_weighted_router_zeroes_blocked_challenger_weights(self):
        router = WeightedRouter(allocation_temperature=1.0, hysteresis_margin=1.0, min_persistence_bars=1, cooldown_bars=0)
        state = router.initialize(_make_active_specialist_library())

        state, decision = router.select(
            state,
            RegimeStateContract(as_of="2026-05-09T00:00:00Z", available_at="2026-05-09T00:00:00Z", label="bull", confidence=1.0),
        )

        self.assertEqual(decision.selected_model_id, "fallback_generalist")
        self.assertEqual(decision.route_reason, "hysteresis_hold")
        self.assertEqual(decision.metadata["blocked_switch_reason"], "hysteresis_margin_not_met")
        self.assertEqual(decision.weights, {"fallback_generalist": 1.0})
        self.assertEqual(decision.executed_candidate_ids, ["fallback_generalist"])
        self.assertEqual(decision.effective_model_count, 1.0)
        self.assertEqual(decision.allocation_control_reason, "blocked_selection_only")

    def test_router_filters_failed_specialist_from_runtime_candidates(self):
        router = HardSwitchRouter(hysteresis_margin=0.0, min_persistence_bars=1, cooldown_bars=0)
        state = router.initialize(_make_active_specialist_library())

        state, decision = router.select(
            state,
            RegimeStateContract(as_of="2026-05-09T00:00:00Z", available_at="2026-05-09T00:00:00Z", label="bull", confidence=0.95),
            specialist_health={
                "health": [
                    {
                        "model_id": "specialist::bull",
                        "failure_flags": ["runtime_failure"],
                    }
                ]
            },
        )

        self.assertEqual(decision.selected_model_id, "fallback_generalist")
        self.assertNotIn("specialist::bull", decision.candidate_scores)
        self.assertFalse(decision.metadata["candidate_eligibility"]["specialist::bull"]["eligible"])
        self.assertIn("health_failure_flags", decision.metadata["candidate_eligibility"]["specialist::bull"]["reasons"])

    def test_warm_regime_state_routes_to_explicit_safe_fallback(self):
        router = HardSwitchRouter(hysteresis_margin=0.0, min_persistence_bars=1, cooldown_bars=0, safe_mode_policy="fallback_only")
        state = router.initialize(_make_active_specialist_library())

        state, decision = router.select(
            state,
            RegimeStateContract(
                as_of="2026-05-09T00:00:00Z",
                available_at="2026-05-09T00:00:00Z",
                label="bull",
                confidence=0.95,
                warm=False,
            ),
        )

        self.assertEqual(decision.selected_model_id, "fallback_generalist")
        self.assertEqual(decision.route_reason, "warm_safe_mode")
        self.assertEqual(decision.metadata["regime_availability_state"], "warm")
        self.assertEqual(decision.metadata["safe_mode_action"], "fallback_only")
        self.assertEqual(state.active_model_id, "fallback_generalist")

    def test_unavailable_regime_state_can_force_no_trade(self):
        router = WeightedRouter(
            allocation_temperature=0.5,
            hysteresis_margin=0.0,
            min_persistence_bars=1,
            cooldown_bars=0,
            safe_mode_policy="no_trade",
        )
        state = router.initialize(_make_active_specialist_library())

        state, decision = router.select(
            state,
            RegimeStateContract(
                as_of="2026-05-09T00:00:00Z",
                available_at="2026-05-09T00:00:00Z",
                label=0,
                confidence=0.0,
                warm=False,
                detector_outputs={"unavailable": 1},
            ),
        )

        self.assertIsNone(decision.selected_model_id)
        self.assertEqual(decision.weights, {})
        self.assertEqual(decision.route_reason, "unavailable_safe_mode")
        self.assertEqual(decision.metadata["safe_mode_action"], "no_trade")
        self.assertIsNone(state.active_model_id)

    def test_build_router_supports_hard_switch_and_weighted_types(self):
        hard_switch = build_router({"type": "hard_switch", "hysteresis_margin": 0.1, "safe_mode_policy": "fallback_only"})
        weighted = build_router({"type": "confidence_weighted", "allocation_temperature": 0.7})

        self.assertIsInstance(hard_switch, HardSwitchRouter)
        self.assertIsInstance(weighted, WeightedRouter)
        self.assertEqual(weighted.manifest().router_type, "confidence_weighted")
        self.assertEqual(hard_switch.manifest().metadata["safe_mode_policy"], "fallback_only")


if __name__ == "__main__":
    unittest.main()