import unittest

from core.specialists import (
    SpecialistHealthContract,
    SpecialistLibrarySnapshot,
    SpecialistPerformanceSlice,
    SpecialistSpec,
    apply_specialist_governance,
    evaluate_specialist_certification_policy,
    evaluate_specialist_library_governance,
)


def _make_candidate_snapshot():
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
                metadata={"fallback_only": True, "lifecycle_state": "candidate"},
            ),
            SpecialistSpec(
                model_id="specialist::bull",
                symbol="BTCUSDT",
                timeframe="1h",
                compatible_regimes=["bull"],
                estimator_family="logisticregression",
                metadata={"regime_label": "bull", "lifecycle_state": "candidate"},
            ),
        ],
        health=[
            SpecialistHealthContract(
                model_id="fallback_generalist",
                compatible_regimes=[],
                fallback_only=True,
            ),
            SpecialistHealthContract(
                model_id="specialist::bull",
                compatible_regimes=["bull"],
                stability_score=0.74,
                decay_score=0.18,
                failure_flags=[],
            ),
        ],
        performance_slices=[
            SpecialistPerformanceSlice(
                model_id="specialist::bull",
                regime_label="bull",
                split_role="training_slice",
                row_count=120,
                metric_summary={"trained_rows": 120},
            )
        ],
    )


def _make_active_snapshot():
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
                metadata={"regime_label": "bull", "lifecycle_state": "active"},
            ),
        ],
        health=[
            SpecialistHealthContract(
                model_id="fallback_generalist",
                compatible_regimes=[],
                fallback_only=True,
            ),
            SpecialistHealthContract(
                model_id="specialist::bull",
                compatible_regimes=["bull"],
                stability_score=0.32,
                decay_score=0.63,
                failure_flags=["drawdown_watch"],
            ),
        ],
        performance_slices=[
            SpecialistPerformanceSlice(
                model_id="specialist::bull",
                regime_label="bull",
                split_role="monitoring_window",
                row_count=24,
                metric_summary={"f1_macro": 0.41},
            )
        ],
    )


class SpecialistGovernanceTest(unittest.TestCase):
    def test_library_governance_certifies_candidate_specialist(self):
        snapshot = _make_candidate_snapshot()

        report = evaluate_specialist_library_governance(
            snapshot,
            policy={
                "certification": {
                    "min_training_rows": 50,
                    "min_stability_score": 0.6,
                    "max_decay_score": 0.3,
                }
            },
        )
        updated = apply_specialist_governance(snapshot, report)

        self.assertEqual(report["recommended_transitions"][0]["model_id"], "specialist::bull")
        self.assertEqual(report["recommended_transitions"][0]["target_state"], "certified")
        self.assertEqual(updated.metadata["selection_contract"]["certified_model_ids"], ["specialist::bull"])
        self.assertEqual(updated.metadata["governance"]["summary"]["transition_model_ids"], ["specialist::bull"])

    def test_library_governance_degrades_unhealthy_active_specialist(self):
        snapshot = _make_active_snapshot()

        report = evaluate_specialist_library_governance(
            snapshot,
            policy={
                "degradation": {
                    "min_stability_score": 0.5,
                    "max_decay_score": 0.4,
                    "min_monitoring_rows": 12,
                    "metric_minimums": {"f1_macro": 0.5},
                    "degrading_failure_flags": ["drawdown_watch"],
                }
            },
        )
        updated = apply_specialist_governance(snapshot, report)

        self.assertEqual(report["recommended_transitions"][0]["target_state"], "degraded")
        self.assertEqual(updated.metadata["selection_contract"]["degraded_model_ids"], ["specialist::bull"])
        self.assertIn("specialist::bull", report["summary"]["blocked_model_ids"])

    def test_advisory_certification_gate_does_not_block(self):
        report = evaluate_specialist_certification_policy(
            {
                "model_id": "specialist::bull",
                "compatible_regimes": ["bull"],
                "stability_score": 0.49,
                "decay_score": 0.2,
                "failure_flags": [],
            },
            trained_rows=120,
            policy={
                "min_training_rows": 50,
                "min_stability_score": 0.5,
                "max_decay_score": 0.3,
                "gate_modes": {"stability_score": "advisory"},
            },
        )

        self.assertTrue(report["approved"])
        self.assertIn("stability_score_below_threshold", report["advisory_failures"])

    def test_library_governance_retires_terminal_specialist_and_assigns_shadow_replacement(self):
        snapshot = SpecialistLibrarySnapshot(
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
                    model_id="specialist::bull_backup",
                    symbol="BTCUSDT",
                    timeframe="1h",
                    compatible_regimes=["bull"],
                    estimator_family="logisticregression",
                    metadata={"lifecycle_state": "certified"},
                ),
            ],
            health=[
                SpecialistHealthContract(model_id="fallback_generalist", compatible_regimes=[], fallback_only=True),
                SpecialistHealthContract(
                    model_id="specialist::bull",
                    compatible_regimes=["bull"],
                    stability_score=0.21,
                    decay_score=0.81,
                    failure_flags=["terminal_drawdown"],
                ),
                SpecialistHealthContract(
                    model_id="specialist::bull_backup",
                    compatible_regimes=["bull"],
                    stability_score=0.74,
                    decay_score=0.18,
                    failure_flags=[],
                ),
            ],
        )

        report = evaluate_specialist_library_governance(
            snapshot,
            policy={
                "retirement": {
                    "retiring_failure_flags": ["terminal_drawdown"],
                    "min_stability_score": 0.3,
                    "max_decay_score": 0.7,
                },
                "replacement": {
                    "preferred_source_states": ["certified"],
                },
            },
        )
        updated = apply_specialist_governance(snapshot, report)

        transitions = {(item["model_id"], item["target_state"]) for item in report["recommended_transitions"]}
        selection_contract = updated.metadata["selection_contract"]

        self.assertIn(("specialist::bull", "retired"), transitions)
        self.assertIn(("specialist::bull_backup", "shadow_challenger"), transitions)
        self.assertIn("specialist::bull", selection_contract["retired_model_ids"])
        self.assertIn("specialist::bull_backup", selection_contract["shadow_model_ids"])


if __name__ == "__main__":
    unittest.main()