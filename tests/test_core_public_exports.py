import unittest

import core


class CorePublicExportsTest(unittest.TestCase):
    def test_phase_zero_contracts_are_exposed_from_core(self):
        expected_names = [
            "BuildRegimeObservationsStep",
            "BaseRegimeDetector",
            "RegimeObservationContract",
            "RegimeStateContract",
            "RegimeTransitionContract",
            "RegimeDetectorManifest",
            "RegimeTraceSummary",
            "build_regime_observation_contracts",
            "build_regime_state_contracts",
            "build_regime_transition_contracts",
            "build_regime_trace_summary",
            "summarize_regime_detection_result",
            "SpecialistSpec",
            "SpecialistHealthContract",
            "SpecialistLibrarySnapshot",
            "SpecialistLifecycleState",
            "apply_specialist_health_update",
            "apply_specialist_governance",
            "normalize_specialist_library_snapshot",
            "normalize_specialist_health_update",
            "project_specialist_library_snapshot",
            "build_specialist_selection_contract",
            "evaluate_specialist_certification_policy",
            "evaluate_specialist_degradation_policy",
            "evaluate_specialist_library_governance",
            "merge_specialist_health_contracts",
            "resolve_specialist_lifecycle_transition",
            "upsert_specialist_performance_slices",
            "build_specialist_specs_from_bundle",
            "build_specialist_health_contracts",
            "build_specialist_library_snapshot",
            "RoutingScoreComponent",
            "RoutingDecisionContract",
            "RouterStateSnapshot",
            "RouterManifest",
            "BaseRouter",
            "build_router",
            "HardSwitchRouter",
            "replay_router_trace",
            "WeightedRouter",
        ]

        for name in expected_names:
            self.assertTrue(hasattr(core, name), msg=f"missing core export: {name}")
