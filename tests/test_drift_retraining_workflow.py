import tempfile
import unittest

import numpy as np
import pandas as pd

from core import (
    LocalRegistryStore,
    ResearchPipeline,
    build_operational_limits_report,
    build_model,
    create_promotion_eligibility_report,
    finalize_promotion_eligibility_report,
    resolve_canonical_promotion_score,
    run_drift_retraining_cycle,
    upsert_promotion_gate,
)
from core.drift import DriftMonitor
from core.specialists import SpecialistHealthContract, SpecialistLibrarySnapshot, SpecialistPerformanceSlice, SpecialistSpec
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


def _make_stable_drift_inputs(current_periods=240):
    reference_index = pd.date_range("2026-10-01", periods=300, freq="1h", tz="UTC")
    current_index = pd.date_range("2026-11-01", periods=current_periods, freq="1h", tz="UTC")
    reference_features = pd.DataFrame({"alpha": 0.0, "beta": 0.0}, index=reference_index)
    current_features = pd.DataFrame({"alpha": 0.0, "beta": 0.0}, index=current_index)
    reference_predictions = pd.DataFrame({"p0": 0.5, "p1": 0.5}, index=reference_index)
    current_predictions = pd.DataFrame({"p0": 0.5, "p1": 0.5}, index=current_index)
    performance = pd.Series(np.zeros(len(current_index)), index=current_index)
    return reference_features, current_features, reference_predictions, current_predictions, performance


def _make_discovery_drift_inputs(current_periods=240):
    reference_index = pd.date_range("2026-10-01", periods=300, freq="1h", tz="UTC")
    current_index = pd.date_range("2026-11-01", periods=current_periods, freq="1h", tz="UTC")
    reference_features = pd.DataFrame({"alpha": 0.0, "beta": 0.0, "regime": "range"}, index=reference_index)
    current_features = pd.DataFrame({"alpha": 2.0, "beta": 2.0, "regime": "trend"}, index=current_index)
    reference_predictions = pd.DataFrame({"p0": 0.5, "p1": 0.5}, index=reference_index)
    current_predictions = pd.DataFrame({"p0": 0.5, "p1": 0.5}, index=current_index)
    performance = pd.Series(np.zeros(len(current_index)), index=current_index)
    return reference_features, current_features, reference_predictions, current_predictions, performance


def _make_score_recalibration_inputs(current_periods=240):
    reference_index = pd.date_range("2026-10-01", periods=300, freq="1h", tz="UTC")
    current_index = pd.date_range("2026-11-01", periods=current_periods, freq="1h", tz="UTC")
    reference_features = pd.DataFrame({"alpha": 0.0, "beta": 0.0}, index=reference_index)
    current_features = pd.DataFrame({"alpha": 0.0, "beta": 0.0}, index=current_index)
    reference_predictions = pd.DataFrame({"p0": 0.62, "p1": 0.38}, index=reference_index)
    current_predictions = pd.DataFrame({"p0": 0.94, "p1": 0.06}, index=current_index)
    performance = pd.Series(np.zeros(len(current_index)), index=current_index)
    return reference_features, current_features, reference_predictions, current_predictions, performance


def _make_active_specialist_library(*, degraded=False):
    if degraded:
        specialist_health = SpecialistHealthContract(
            model_id="specialist::bull",
            compatible_regimes=["bull"],
            stability_score=0.32,
            decay_score=0.63,
            failure_flags=["drawdown_watch"],
        )
        performance_slice = SpecialistPerformanceSlice(
            model_id="specialist::bull",
            regime_label="bull",
            split_role="monitoring_window",
            row_count=24,
            metric_summary={"f1_macro": 0.41},
        )
    else:
        specialist_health = SpecialistHealthContract(
            model_id="specialist::bull",
            compatible_regimes=["bull"],
            stability_score=0.82,
            decay_score=0.08,
            failure_flags=[],
        )
        performance_slice = SpecialistPerformanceSlice(
            model_id="specialist::bull",
            regime_label="bull",
            split_role="monitoring_window",
            row_count=24,
            metric_summary={"f1_macro": 0.71},
        )

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
            specialist_health,
        ],
        performance_slices=[performance_slice],
    )


def _register_champion(store, symbol, score_value, *, specialist_library=None):
    model, feature_columns = _fit_logistic_model()
    report = _make_eligibility_report(score_value)
    version_id = store.register_version(
        model,
        symbol=symbol,
        feature_columns=feature_columns,
        training_summary={"avg_f1_macro": 0.75},
        validation_summary={"raw_objective_value": score_value, "promotion_ready": True},
        promotion_eligibility_report=report,
        specialist_library=specialist_library,
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

    def test_drift_cycle_reuses_persisted_monitor_state_between_runs(self):
        reference_features, _, reference_predictions, _, _ = _make_stable_drift_inputs(current_periods=80)
        first_index = pd.date_range("2026-11-01", periods=80, freq="1h", tz="UTC")
        second_index = pd.date_range("2026-11-05", periods=80, freq="1h", tz="UTC")
        first_features = pd.DataFrame({"alpha": 0.0, "beta": 0.0}, index=first_index)
        second_features = pd.DataFrame({"alpha": 0.0, "beta": 0.0}, index=second_index)
        first_predictions = pd.DataFrame({"p0": 0.5, "p1": 0.5}, index=first_index)
        second_predictions = pd.DataFrame({"p0": 0.5, "p1": 0.5}, index=second_index)
        first_performance = pd.Series(np.full(len(first_index), 0.1), index=first_index)
        second_performance = pd.Series(np.full(len(second_index), 2.5), index=second_index)

        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            _register_champion(store, "BTCUSDT", 0.12)

            first_result = run_drift_retraining_cycle(
                store=store,
                symbol="BTCUSDT",
                reference_features=reference_features,
                current_features=first_features,
                reference_predictions=reference_predictions,
                current_predictions=first_predictions,
                current_performance=first_performance,
                bars_since_last_retrain=800,
                scheduled_window_open=False,
                train_challenger=lambda: _make_challenger_payload(0.13),
                drift_config={"min_samples": 1, "min_drift_signals": 1, "max_bars_between_retrain": 5000},
            )
            second_result = run_drift_retraining_cycle(
                store=store,
                symbol="BTCUSDT",
                reference_features=reference_features,
                current_features=second_features,
                reference_predictions=reference_predictions,
                current_predictions=second_predictions,
                current_performance=second_performance,
                bars_since_last_retrain=801,
                scheduled_window_open=False,
                train_challenger=lambda: _make_challenger_payload(0.13),
                drift_config={"min_samples": 1, "min_drift_signals": 1, "max_bars_between_retrain": 5000},
            )

            self.assertFalse(first_result["drift_report"]["performance_drift"])
            self.assertTrue(second_result["drift_report"]["performance_drift"])
            self.assertEqual(
                second_result["drift_report"]["drift_monitor_state"]["performance_detector"]["history_length"],
                160,
            )
            champion = store.get_champion("BTCUSDT")
            persisted_report = read_json(champion["latest_drift_report"])
            self.assertEqual(persisted_report["drift_monitor_state"]["performance_detector"]["history_length"], 160)

    def test_library_review_policy_recommends_review_for_degraded_active_specialist_library(self):
        reference_features, current_features, reference_predictions, current_predictions, performance = _make_stable_drift_inputs()

        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            _register_champion(
                store,
                "BTCUSDT",
                0.12,
                specialist_library=_make_active_specialist_library(degraded=True),
            )
            build_calls = []

            result = run_drift_retraining_cycle(
                store=store,
                symbol="BTCUSDT",
                reference_features=reference_features,
                current_features=current_features,
                reference_predictions=reference_predictions,
                current_predictions=current_predictions,
                current_performance=performance,
                bars_since_last_retrain=520,
                scheduled_window_open=True,
                train_challenger=lambda: build_calls.append("called"),
                drift_config={"min_samples": 200, "min_drift_signals": 2, "max_bars_between_retrain": 672},
                library_review_policy={
                    "min_active_specialists": 1,
                    "governance": {
                        "degradation": {
                            "min_stability_score": 0.5,
                            "max_decay_score": 0.4,
                            "min_monitoring_rows": 12,
                            "metric_minimums": {"f1_macro": 0.5},
                            "degrading_failure_flags": ["drawdown_watch"],
                        }
                    },
                },
            )

            self.assertEqual(result["retrain_status"], "library_review_recommended")
            self.assertEqual(build_calls, [])
            self.assertTrue(result["library_review"]["recommended"])
            self.assertEqual(result["library_review"]["action"], "review_library")
            self.assertIn("blocked_specialists_present", result["library_review"]["reasons"])
            self.assertIn("specialist::bull", result["library_review"]["blocked_model_ids"])

    def test_router_recalibration_policy_recommends_router_action_before_retraining(self):
        reference_features, current_features, reference_predictions, current_predictions, performance = _make_stable_drift_inputs()

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
                current_router_stability_report={
                    "enabled": True,
                    "applicable": True,
                    "decision_count": 80,
                    "switch_count": 26,
                    "switch_rate": 0.3291,
                    "blocked_switch_count": 0,
                    "blocked_switch_rate": 0.0,
                    "configured_control_count": 1,
                },
                bars_since_last_retrain=520,
                scheduled_window_open=True,
                train_challenger=lambda: build_calls.append("called"),
                drift_config={"min_samples": 200, "min_drift_signals": 2, "max_bars_between_retrain": 672},
                router_recalibration_policy={
                    "min_router_decision_count": 25,
                    "max_router_switch_rate": 0.20,
                    "require_router_stability_controls": True,
                },
            )

            self.assertEqual(result["retrain_status"], "router_recalibration_recommended")
            self.assertEqual(build_calls, [])
            self.assertTrue(result["router_recalibration"]["recommended"])
            self.assertEqual(result["router_recalibration"]["action"], "recalibrate_router")
            self.assertIn("router_switch_rate_above_limit", result["router_recalibration"]["reasons"])

    def test_non_structural_regime_shift_recommends_discovery_before_retraining(self):
        reference_features, current_features, reference_predictions, current_predictions, performance = _make_discovery_drift_inputs()

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
                bars_since_last_retrain=520,
                scheduled_window_open=True,
                train_challenger=lambda: build_calls.append("called"),
                drift_config={"min_samples": 200, "min_drift_signals": 2, "max_bars_between_retrain": 672},
            )

            self.assertTrue(result["drift_guardrails"]["approved"])
            self.assertFalse(result["structural_invalidation"]["retrain_recommended"])
            self.assertTrue(result["structural_invalidation"]["discover_recommended"])
            self.assertEqual(result["retrain_status"], "discovery_recommended")
            self.assertEqual(result["action_report"]["recommended_action"], "discover")
            self.assertEqual(build_calls, [])

    def test_score_only_drift_recommends_recalibration_before_retraining(self):
        reference_features, current_features, reference_predictions, current_predictions, performance = _make_score_recalibration_inputs()

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
                bars_since_last_retrain=520,
                scheduled_window_open=True,
                train_challenger=lambda: build_calls.append("called"),
                drift_config={
                    "min_samples": 200,
                    "min_drift_signals": 1,
                    "max_bars_between_retrain": 672,
                    "confidence_ks_threshold": 0.05,
                },
            )

            self.assertTrue(result["drift_guardrails"]["approved"])
            self.assertTrue(result["drift_report"]["score_drift"])
            self.assertFalse(result["drift_report"]["action_drift"])
            self.assertFalse(result["structural_invalidation"]["retrain_recommended"])
            self.assertFalse(result["structural_invalidation"]["discover_recommended"])
            self.assertTrue(result["structural_invalidation"]["recalibrate_recommended"])
            self.assertEqual(result["action_report"]["recommended_action"], "recalibrate")
            self.assertEqual(result["retrain_status"], "recalibration_recommended")
            self.assertEqual(build_calls, [])

    def test_model_ttl_expiry_stays_in_maintenance_refresh_without_drift_signals(self):
        reference_features, current_features, reference_predictions, current_predictions, performance = _make_stable_drift_inputs()

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
                drift_config={"min_samples": 200, "min_drift_signals": 2, "max_bars_between_retrain": 672},
            )

            self.assertTrue(result["drift_report"]["model_ttl_expired"])
            self.assertFalse(result["drift_report"]["ttl_evidence_sufficient"])
            self.assertTrue(result["drift_guardrails"]["maintenance_refresh_recommended"])
            self.assertTrue(result["drift_guardrails"]["maintenance_only_approved"])
            self.assertFalse(result["drift_guardrails"]["adaptive_approved"])
            self.assertEqual(result["action_report"]["recommended_action"], "maintenance_refresh")
            self.assertEqual(result["retrain_status"], "maintenance_refresh_recommended")
            self.assertIsNone(result["candidate_version_id"])
            self.assertEqual(store.get_champion("BTCUSDT")["version_id"], champion_id)

    def test_canonical_regime_ids_prevent_label_rename_drift_noise(self):
        reference_index = pd.date_range("2026-10-01", periods=300, freq="1h", tz="UTC")
        current_index = pd.date_range("2026-11-01", periods=240, freq="1h", tz="UTC")
        reference_features = pd.DataFrame(
            {
                "alpha": 0.0,
                "beta": 0.0,
                "regime": "range_label",
                "canonical_regime_id": "family_0",
            },
            index=reference_index,
        )
        current_features = pd.DataFrame(
            {
                "alpha": 0.0,
                "beta": 0.0,
                "regime": "trend_label",
                "canonical_regime_id": "family_0",
            },
            index=current_index,
        )
        reference_predictions = pd.DataFrame({"p0": 0.5, "p1": 0.5}, index=reference_index)
        current_predictions = pd.DataFrame({"p0": 0.5, "p1": 0.5}, index=current_index)

        monitor = DriftMonitor(reference_features, reference_predictions=reference_predictions, config={"min_samples": 1})
        report = monitor.check(current_features, current_predictions=current_predictions)

        self.assertFalse(report["regime_drift"])
        self.assertEqual(report["regime_report"]["identity_column"], "canonical_regime_id")
        self.assertEqual(float((report["regime_report"]["distribution"] or {}).get("total_variation") or 0.0), 0.0)

    def test_taxonomy_shadow_remap_noise_does_not_count_as_regime_drift(self):
        reference_index = pd.date_range("2026-10-01", periods=300, freq="1h", tz="UTC")
        current_index = pd.date_range("2026-11-01", periods=240, freq="1h", tz="UTC")
        reference_features = pd.DataFrame(
            {
                "alpha": 0.0,
                "beta": 0.0,
                "regime": "range_label",
                "canonical_regime_id": "family_0",
                "regime_family_id": "family_0",
                "mapping_status": "stable_match",
            },
            index=reference_index,
        )
        current_features = pd.DataFrame(
            {
                "alpha": 0.0,
                "beta": 0.0,
                "regime": "range_label",
                "canonical_regime_id": "family_0__needs_shadow_period__state_0",
                "regime_family_id": "family_0",
                "mapping_status": "needs_shadow_period",
            },
            index=current_index,
        )
        reference_predictions = pd.DataFrame({"p0": 0.5, "p1": 0.5}, index=reference_index)
        current_predictions = pd.DataFrame({"p0": 0.5, "p1": 0.5}, index=current_index)

        monitor = DriftMonitor(reference_features, reference_predictions=reference_predictions, config={"min_samples": 1})
        report = monitor.check(current_features, current_predictions=current_predictions)

        self.assertFalse(report["regime_drift"])
        self.assertTrue(report["regime_report"]["taxonomy"]["taxonomy_only_instability"])
        self.assertEqual(report["regime_report"]["taxonomy"]["identity_column"], "regime_family_id")
        self.assertEqual(report["regime_report"]["taxonomy"]["status_counts"], {"needs_shadow_period": 240})

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

    def test_pipeline_wrapper_runs_drift_cycle_from_runtime_state(self):
        reference_features, current_features, reference_predictions, current_predictions, performance = _make_drift_inputs()

        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            _register_champion(store, "BTCUSDT", 0.12)

            pipeline = ResearchPipeline({"data": {"symbol": "BTCUSDT", "interval": "1h"}})
            pipeline.state["X"] = current_features
            pipeline.state["training"] = {"oos_probabilities": current_predictions}
            pipeline.state["backtest"] = {"equity_curve": (1.0 + performance).cumprod()}
            pipeline.state["operational_monitoring"] = {"healthy": True, "reasons": []}

            result = pipeline.run_drift_retraining_cycle(
                store=store,
                reference_features=reference_features,
                reference_predictions=reference_predictions,
                bars_since_last_retrain=800,
                scheduled_window_open=True,
                train_challenger=lambda: _make_challenger_payload(0.18),
            )

            self.assertEqual(result["retrain_status"], "promoted")
            self.assertTrue(result["promotion_decision"]["approved"])
            self.assertEqual(pipeline.state["drift_cycle"]["candidate_version_id"], result["candidate_version_id"])

    def test_drawdown_breach_can_trigger_default_hybrid_rollback(self):
        reference_features, current_features, reference_predictions, current_predictions, performance = _make_drift_inputs()

        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            previous_champion = _register_champion(store, "BTCUSDT", 0.10)
            _register_champion(store, "BTCUSDT", 0.14)

            operational_limits = build_operational_limits_report(
                operational_limits={"healthy": True, "kill_switch_ready": True},
                equity_curve=pd.Series([1.0, 1.12, 1.05, 0.94], dtype=float),
            )

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
                current_monitoring_report={"healthy": True, "reasons": []},
                operational_limits=operational_limits,
                rollback_policy={"mode": "hybrid"},
            )

            self.assertEqual(result["retrain_status"], "challenger_rejected")
            self.assertTrue(result["rollback"]["recommended"])
            self.assertTrue(result["rollback"]["executed"])
            self.assertIn("drawdown_limit_breached", result["rollback"]["reasons"])
            self.assertIn("kill_switch_triggered", result["rollback"]["reasons"])
            self.assertEqual(result["rollback"]["restored_version_id"], previous_champion)

    def test_request_weight_guard_defers_drift_retraining_before_training(self):
        reference_features, current_features, reference_predictions, current_predictions, performance = _make_drift_inputs()

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
                bars_since_last_retrain=800,
                scheduled_window_open=True,
                train_challenger=lambda: build_calls.append("called"),
                drift_config={
                    "request_weight_guard": {
                        "limit": 6000,
                        "used": 5900,
                        "retrain_cost": 200,
                        "reserve_ratio": 0.05,
                    }
                },
            )

            self.assertEqual(result["retrain_status"], "request_weight_deferred")
            self.assertEqual(build_calls, [])
            self.assertTrue(result["drift_guardrails"]["approved"])
            self.assertTrue(result["request_weight_guard"]["configured"])
            self.assertFalse(result["request_weight_guard"]["allowed"])
            self.assertIn("request_weight_headroom_insufficient", result["request_weight_guard"]["reasons"])


if __name__ == "__main__":
    unittest.main()