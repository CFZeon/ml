import unittest

from core.backtest import _build_execution_evidence


class ExecutionEvidenceContractTest(unittest.TestCase):
    def test_event_driven_execution_evidence_is_certification_class(self):
        evidence = _build_execution_evidence(
            "event_driven",
            True,
            [],
            execution_adapter="nautilus",
            execution_backend="nautilus",
            symbol_filters={"tick_size": 0.1, "step_size": 0.001, "min_notional": 10.0},
        )

        self.assertEqual(evidence["class"], "event_driven_certification")
        self.assertEqual(evidence["execution_mode"], "event_driven")
        self.assertTrue(evidence["promotion_execution_ready"])
        self.assertEqual(evidence["blocking_reasons"], [])

    def test_surrogate_execution_evidence_is_research_class(self):
        evidence = _build_execution_evidence(
            "conservative_bar_surrogate",
            False,
            ["bar_surrogate_only", "no_queue_position_model"],
            execution_adapter="bar_surrogate",
            execution_backend="bar_surrogate",
        )

        self.assertEqual(evidence["class"], "research_surrogate")
        self.assertEqual(evidence["execution_mode"], "conservative_bar_surrogate")
        self.assertFalse(evidence["promotion_execution_ready"])
        self.assertIn("execution_backend_not_event_driven", evidence["blocking_reasons"])
        self.assertIn("promotion_execution_not_ready", evidence["blocking_reasons"])
        self.assertIn("bar_surrogate_only", evidence["blocking_reasons"])

    def test_event_driven_execution_without_symbol_filters_is_unknown(self):
        evidence = _build_execution_evidence(
            "event_driven",
            True,
            [],
            execution_adapter="nautilus",
            execution_backend="nautilus",
        )

        self.assertEqual(evidence["class"], "unknown_execution_constraints")
        self.assertFalse(evidence["promotion_execution_ready"])
        self.assertFalse(evidence["venue_constraints_available"])
        self.assertIn("venue_constraints_unavailable", evidence["blocking_reasons"])


if __name__ == "__main__":
    unittest.main()