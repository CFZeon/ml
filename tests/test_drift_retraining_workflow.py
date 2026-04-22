import tempfile
import unittest

import numpy as np
import pandas as pd

from core import (
    LocalRegistryStore,
    build_model,
    create_promotion_eligibility_report,
    finalize_promotion_eligibility_report,
    resolve_canonical_promotion_score,
    run_drift_retraining_cycle,
    upsert_promotion_gate,
)
from core.storage import read_json


def _fit_logistic_model():
    X = pd.DataFrame({"f1": [0.0, 1.0, 0.0, 1.0], "f2": [1.0, 1.0, 0.0, 0.0]})
    y = pd.Series([0, 1, 0, 1])
    model = build_model("logistic", {"c": 1.0})
    model.fit(X, y)
    return model, list(X.columns)


def _make_eligibility_report(score_value=0.12):
    score = resolve_canonical_promotion_score(
        locked_holdout_report={"raw_objective_value": score_value},
        selection_value=score_value,
    )
    report = create_promotion_eligibility_report(score_basis=score["basis"], score_value=score["value"])
    for group, name in (
        ("selection", "feature_admission"),
        ("selection", "feature_portability"),
        ("selection", "regime_stability"),
        ("selection", "operational_health"),
        ("post_selection", "locked_holdout"),
        ("post_selection", "locked_holdout_gap"),
    ):
        report = upsert_promotion_gate(report, group=group, name=name, passed=True)
    return finalize_promotion_eligibility_report(report)


def _make_drift_inputs(current_periods=240):
    reference_index = pd.date_range("2026-10-01", periods=300, freq="1h", tz="UTC")
    current_index = pd.date_range("2026-11-01", periods=current_periods, freq="1h", tz="UTC")
    reference_features = pd.DataFrame(
        {
            "alpha": np.random.default_rng(1).normal(0.0, 1.0, len(reference_index)),
            "beta": np.random.default_rng(2).normal(0.0, 1.0, len(reference_index)),
        },
        index=reference_index,
    )
    current_features = pd.DataFrame(
        {
            "alpha": np.random.default_rng(3).normal(3.0, 1.0, len(current_index)),
            "beta": np.random.default_rng(4).normal(3.0, 1.0, len(current_index)),
        },
        index=current_index,
    )
    reference_predictions = pd.DataFrame({"p0": 0.8, "p1": 0.2}, index=reference_index)
    current_predictions = pd.DataFrame({"p0": 0.2, "p1": 0.8}, index=current_index)
    performance = pd.Series(
        np.r_[np.full(len(current_index) // 2, 0.6), np.full(len(current_index) - (len(current_index) // 2), -0.4)],
        index=current_index,
    )
    return reference_features, current_features, reference_predictions, current_predictions, performance


def _register_champion(store, symbol, score_value):
    model, feature_columns = _fit_logistic_model()
    report = _make_eligibility_report(score_value)
    version_id = store.register_version(
        model,
        symbol=symbol,
        feature_columns=feature_columns,
        training_summary={"avg_f1_macro": 0.75},
        validation_summary={"raw_objective_value": score_value, "promotion_ready": True},
        promotion_eligibility_report=report,
    )
    store.promote(
        version_id,
        "champion",
        symbol=symbol,
        decision={"approved": True, "reasons": ["approved"], "promotion_eligibility_report": report},
    )
    return version_id


def _make_challenger_payload(score_value):
    model, feature_columns = _fit_logistic_model()
    report = _make_eligibility_report(score_value)
    return {
        "model": model,
        "feature_columns": feature_columns,
        "training_summary": {"avg_f1_macro": 0.80},
        "validation_summary": {"raw_objective_value": score_value, "promotion_ready": True},
        "promotion_eligibility_report": report,
        "sample_count": 320,
        "monitoring_report": {"healthy": True, "reasons": []},
    }


class DriftRetrainingWorkflowTest(unittest.TestCase):
    def test_drift_recommendation_waits_for_scheduled_window_and_persists_metadata(self):
        reference_features, current_features, reference_predictions, current_predictions, performance = _make_drift_inputs()

        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            champion_id = _register_champion(store, "BTCUSDT", 0.12)
            build_calls = []

            result = run_drift_retraining_cycle(
                store=store,
                symbol="BTCUSDT",
                reference_features=reference_features,
                current_features=current_features,
                reference_predictions=reference_predictions,
                current_predictions=current_predictions,
                current_performance=performance,
                bars_since_last_retrain=800,
                scheduled_window_open=False,
                train_challenger=lambda: build_calls.append("called"),
            )

            self.assertEqual(result["retrain_status"], "scheduled_window_pending")
            self.assertTrue(result["drift_guardrails"]["approved"])
            self.assertEqual(build_calls, [])
            champion = store.get_champion("BTCUSDT")
            self.assertEqual(champion["version_id"], champion_id)

            persisted_report = read_json(champion["latest_drift_report"])
            self.assertEqual(persisted_report["reference_window"]["sample_count"], len(reference_features))
            self.assertEqual(persisted_report["current_window"]["sample_count"], len(current_features))
            self.assertEqual(persisted_report["recommendation"]["bars_since_last_retrain"], 800)
            self.assertGreaterEqual(int(persisted_report["evidence_count"]), 2)

    def test_drift_retrain_guardrails_block_on_minimum_samples_and_cooldown(self):
        reference_features, current_features, reference_predictions, current_predictions, performance = _make_drift_inputs(current_periods=120)

        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            _register_champion(store, "BTCUSDT", 0.12)
            build_calls = []

            result = run_drift_retraining_cycle(
                store=store,
                symbol="BTCUSDT",
                reference_features=reference_features,
                current_features=current_features,
                reference_predictions=reference_predictions,
                current_predictions=current_predictions,
                current_performance=performance,
                bars_since_last_retrain=100,
                scheduled_window_open=True,
                train_challenger=lambda: build_calls.append("called"),
                drift_config={"min_samples": 200, "cooldown_bars": 500, "min_drift_signals": 2},
            )

            self.assertEqual(result["retrain_status"], "not_recommended")
            self.assertIn("minimum_samples_not_met", result["drift_guardrails"]["reasons"])
            self.assertIn("cooldown_active", result["drift_guardrails"]["reasons"])
            self.assertEqual(build_calls, [])

    def test_scheduled_drift_retrain_promotes_approved_challenger(self):
        reference_features, current_features, reference_predictions, current_predictions, performance = _make_drift_inputs()

        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            champion_id = _register_champion(store, "BTCUSDT", 0.12)

            result = run_drift_retraining_cycle(
                store=store,
                symbol="BTCUSDT",
                reference_features=reference_features,
                current_features=current_features,
                reference_predictions=reference_predictions,
                current_predictions=current_predictions,
                current_performance=performance,
                bars_since_last_retrain=800,
                scheduled_window_open=True,
                train_challenger=lambda: _make_challenger_payload(0.18),
            )

            self.assertEqual(result["retrain_status"], "promoted")
            self.assertTrue(result["promotion_decision"]["approved"])
            self.assertNotEqual(result["candidate_version_id"], champion_id)
            self.assertEqual(store.get_champion("BTCUSDT")["version_id"], result["candidate_version_id"])

    def test_failed_challenger_can_trigger_hybrid_rollback_on_critical_degradation(self):
        reference_features, current_features, reference_predictions, current_predictions, performance = _make_drift_inputs()

        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            previous_champion = _register_champion(store, "BTCUSDT", 0.10)
            current_champion = _register_champion(store, "BTCUSDT", 0.14)

            result = run_drift_retraining_cycle(
                store=store,
                symbol="BTCUSDT",
                reference_features=reference_features,
                current_features=current_features,
                reference_predictions=reference_predictions,
                current_predictions=current_predictions,
                current_performance=performance,
                bars_since_last_retrain=800,
                scheduled_window_open=True,
                train_challenger=lambda: _make_challenger_payload(0.08),
                current_monitoring_report={"healthy": False, "reasons": ["feature_schema"]},
                rollback_policy={"mode": "hybrid", "critical_reasons": ["feature_schema"]},
            )

            self.assertEqual(result["retrain_status"], "challenger_rejected")
            self.assertFalse(result["promotion_decision"]["approved"])
            self.assertTrue(result["rollback"]["recommended"])
            self.assertTrue(result["rollback"]["executed"])
            self.assertEqual(result["rollback"]["restored_version_id"], previous_champion)
            self.assertEqual(store.get_champion("BTCUSDT")["version_id"], previous_champion)
            archived_current = next(row for row in store.list_versions("BTCUSDT") if row["version_id"] == current_champion)
            self.assertEqual(archived_current["current_status"], "archived")


if __name__ == "__main__":
    unittest.main()