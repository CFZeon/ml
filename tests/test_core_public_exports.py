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
            "build_specialist_specs_from_bundle",
            "build_specialist_health_contracts",
            "build_specialist_library_snapshot",
            "RoutingScoreComponent",
            "RoutingDecisionContract",
            "RouterStateSnapshot",
            "RouterManifest",
            "BaseRouter",
        ]

        for name in expected_names:
            self.assertTrue(hasattr(core, name), msg=f"missing core export: {name}")
