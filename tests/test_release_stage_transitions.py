import tempfile
import unittest

import pandas as pd

from core import (
    LocalRegistryStore,
    build_deployment_readiness_report,
    build_live_calibration_report,
    build_monitoring_report,
    build_operational_limits_report,
    build_model,
    create_promotion_eligibility_report,
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


def _make_paper_report():
    return build_live_calibration_report(
        certified_expectations={"modeled_slippage_bps": 1.8, "modeled_fill_ratio": 0.94},
        paper_metrics={
            "mode": "paper",
            "duration_days": 35,
            "modeled_slippage_bps": 1.8,
            "realized_slippage_bps": 1.9,
            "modeled_fill_ratio": 0.94,
            "realized_fill_ratio": 0.91,
            "data_breaches": 0,
            "funding_breaches": 0,
            "kill_switch_triggers": 0,
        },
        symbol_filters={"tick_size": 0.1, "step_size": 0.001, "min_notional": 10.0},
    )


def _make_validated_monitoring_report(*, deployment_profile="research_workstation"):
    return build_monitoring_report(
        policy={"policy_profile": "research"},
        deployment_profile=deployment_profile,
        expected_feature_columns=["f1"],
        actual_feature_columns=["f1"],
        inference_latencies_ms=[60.0, 75.0, 90.0],
        queue_backlog=[0, 0, 0],
        resource_telemetry={
            "peak_rss_mb": 4096.0,
            "model_load_latency_ms": 1200.0,
            "drift_cycle_latency_ms": 180_000.0,
            "storage_footprint_mb": 2048.0,
            "symbol_count": 2,
            "timeframe_count": 1,
        },
        replay_benchmark={
            "run_count": 2,
            "throughput_bars_per_sec": 400.0,
            "latency_p95_ms": 90.0,
            "latency_p99_ms": 120.0,
            "memory_spike_mb": 4600.0,
            "restart_recovery_time_ms": 45_000.0,
            "deterministic_pass": True,
        },
    )


class ReleaseStageTransitionsTest(unittest.TestCase):
    def test_release_stage_advances_from_certified_to_micro_and_scaled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            _register_champion(store, "BTCUSDT", 0.12)
            _register_champion(store, "BTCUSDT", 0.16)

            base_kwargs = {
                "store": store,
                "symbol": "BTCUSDT",
                "monitoring_report": _make_validated_monitoring_report(),
                "drift_cycle": {
                    "drift_guardrails": {"approved": False, "reasons": []},
                    "retrain_status": "not_recommended",
                },
                "backend_status": {"adapter": "nautilus", "available": True, "reasons": []},
            }

            certified = build_deployment_readiness_report(**base_kwargs)
            self.assertEqual(certified["capital_release_stage"], "research_certified")

            paper = build_deployment_readiness_report(**base_kwargs, paper_report=_make_paper_report())
            self.assertEqual(paper["capital_release_stage"], "paper_verified")

            micro = build_deployment_readiness_report(
                **base_kwargs,
                paper_report=_make_paper_report(),
                operational_limits={"healthy": True, "kill_switch_ready": True, "reasons": []},
                release_request={"requested_stage": "micro_capital", "manual_acknowledged": True},
            )
            self.assertEqual(micro["capital_release_stage"], "micro_capital")
            self.assertTrue(micro["capital_release_eligible"])

            scaled = build_deployment_readiness_report(
                **base_kwargs,
                paper_report=_make_paper_report(),
                operational_limits={"healthy": True, "kill_switch_ready": True, "reasons": []},
                release_request={
                    "requested_stage": "scaled_capital",
                    "manual_acknowledged": True,
                    "scaled_capital_approved": True,
                },
            )
            self.assertEqual(scaled["capital_release_stage"], "scaled_capital")
            self.assertTrue(scaled["capital_release_eligible"])

    def test_release_stage_stays_paper_verified_when_drawdown_limit_is_breached(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            _register_champion(store, "BTCUSDT", 0.12)
            _register_champion(store, "BTCUSDT", 0.16)

            operational_limits = build_operational_limits_report(
                operational_limits={"healthy": True, "kill_switch_ready": True},
                equity_curve=pd.Series([1.0, 1.08, 1.02, 0.95], dtype=float),
            )

            report = build_deployment_readiness_report(
                store=store,
                symbol="BTCUSDT",
                monitoring_report=_make_validated_monitoring_report(deployment_profile="consumer_laptop"),
                drift_cycle={
                    "drift_guardrails": {"approved": False, "reasons": []},
                    "retrain_status": "not_recommended",
                },
                backend_status={"adapter": "nautilus", "available": True, "reasons": []},
                paper_report=_make_paper_report(),
                operational_limits=operational_limits,
                release_request={"requested_stage": "micro_capital", "manual_acknowledged": True},
            )

            self.assertEqual(report["capital_release_stage"], "paper_verified")
            self.assertFalse(report["capital_release_eligible"])
            self.assertIn("operational_limits", report["summary"]["failed_components"])
            self.assertIn("drawdown_limit_breached", report["release_blockers"])
            self.assertIn("kill_switch_triggered", report["release_blockers"])


if __name__ == "__main__":
    unittest.main()