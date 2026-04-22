import tempfile
import unittest

import pandas as pd

from core import (
    LocalRegistryStore,
    build_model,
    create_promotion_eligibility_report,
    evaluate_execution_realism_gate,
    evaluate_challenger_promotion,
    finalize_promotion_eligibility_report,
    resolve_canonical_promotion_score,
    upsert_promotion_gate,
)


def _fit_logistic_model():
    X = pd.DataFrame({"f1": [0.0, 1.0, 0.0, 1.0], "f2": [1.0, 1.0, 0.0, 0.0]})
    y = pd.Series([0, 1, 0, 1])
    model = build_model("logistic", {"c": 1.0})
    model.fit(X, y)
    return model, list(X.columns)


def _make_report(score_value):
    score = resolve_canonical_promotion_score(
        locked_holdout_report={"raw_objective_value": score_value},
        selection_value=score_value,
    )
    report = create_promotion_eligibility_report(score_basis=score["basis"], score_value=score["value"])
    for gate_name in ("feature_admission", "feature_portability", "regime_stability", "operational_health"):
        report = upsert_promotion_gate(report, group="selection", name=gate_name, passed=True)
    report = upsert_promotion_gate(report, group="post_selection", name="locked_holdout", passed=True)
    report = upsert_promotion_gate(report, group="post_selection", name="locked_holdout_gap", passed=True)
    return finalize_promotion_eligibility_report(report)


class PromotionGateBindingTest(unittest.TestCase):
    def test_execution_realism_gate_blocks_surrogate_promotion(self):
        report = _make_report(0.15)
        execution_realism = evaluate_execution_realism_gate(
            {
                "execution_mode": "conservative_bar_surrogate",
                "promotion_execution_ready": False,
                "execution_adapter": "bar_surrogate",
                "execution_backend": "bar_surrogate",
                "execution_limitations": ["bar_surrogate_only"],
            }
        )
        report = upsert_promotion_gate(
            report,
            group="post_selection",
            name="execution_realism",
            passed=bool(execution_realism["passed"]),
            mode="blocking",
            measured=execution_realism["execution_mode"],
            threshold=execution_realism["required_execution_mode"],
            reason=execution_realism["reason"],
            details=execution_realism,
        )
        report = finalize_promotion_eligibility_report(report)

        decision = evaluate_challenger_promotion(
            {
                "promotion_eligibility_report": report,
                "selection_value": 0.15,
                "sample_count": 500,
            }
        )

        self.assertFalse(decision["approved"])
        self.assertIn("execution_realism_failed", decision["promotion_eligibility_report"]["blocking_failures"])

    def test_calibration_mode_downgrades_blocking_gate_to_advisory(self):
        report = create_promotion_eligibility_report(calibration_mode=True)
        report = upsert_promotion_gate(
            report,
            group="selection",
            name="feature_portability",
            passed=False,
            mode="blocking",
            reason="feature_portability_failed",
        )
        report = finalize_promotion_eligibility_report(report)

        self.assertTrue(report["eligible_before_post_checks"])
        self.assertIn("feature_portability_failed", report["advisory_failures"])
        self.assertNotIn("feature_portability_failed", report["blocking_failures"])

    def test_registry_blocks_legacy_champion_without_canonical_report(self):
        model, feature_columns = _fit_logistic_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            version_id = store.register_version(
                model,
                symbol="BTCUSDT",
                feature_columns=feature_columns,
                training_summary={"avg_f1_macro": 0.7},
                validation_summary={"raw_objective_value": 0.1},
            )
            store.promote(version_id, "champion", symbol="BTCUSDT", decision={"approved": True, "reasons": ["approved"]})

            decision = evaluate_challenger_promotion(
                {
                    "promotion_eligibility_report": _make_report(0.12),
                    "selection_value": 0.12,
                    "sample_count": 500,
                },
                champion_record=store.get_champion("BTCUSDT"),
            )

            self.assertFalse(decision["approved"])
            self.assertIn("legacy_champion_missing_canonical_eligibility_report", decision["reasons"])

    def test_registry_compares_canonical_locked_holdout_scores(self):
        model, feature_columns = _fit_logistic_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            champion_report = _make_report(0.12)
            version_id = store.register_version(
                model,
                symbol="BTCUSDT",
                feature_columns=feature_columns,
                training_summary={"avg_f1_macro": 0.75},
                validation_summary={"raw_objective_value": 0.11},
                locked_holdout={"raw_objective_value": 0.12},
                promotion_eligibility_report=champion_report,
            )
            store.promote(version_id, "champion", symbol="BTCUSDT", decision={"approved": True, "reasons": ["approved"]})

            decision = evaluate_challenger_promotion(
                {
                    "promotion_eligibility_report": _make_report(0.11),
                    "selection_value": 0.11,
                    "sample_count": 500,
                },
                champion_record=store.get_champion("BTCUSDT"),
            )

            self.assertFalse(decision["approved"])
            self.assertIn("challenger_not_better_than_champion", decision["reasons"])
            self.assertEqual(decision["score_basis"], "locked_holdout_raw_objective")


if __name__ == "__main__":
    unittest.main()