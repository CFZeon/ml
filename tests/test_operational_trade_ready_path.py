import tempfile
import unittest

import pandas as pd

from core import (
    LocalRegistryStore,
    ResearchPipeline,
    build_deployment_readiness_report,
    build_model,
    create_promotion_eligibility_report,
    finalize_promotion_eligibility_report,
    resolve_canonical_promotion_score,
    upsert_promotion_gate,
)
from core.pipeline import _build_pipeline_operational_monitoring


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


class OperationalTradeReadyPathTest(unittest.TestCase):
    def test_pipeline_trade_ready_mode_auto_applies_trade_ready_monitoring_profile(self):
        index = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
        pipeline = ResearchPipeline(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "backtest": {"evaluation_mode": "trade_ready"},
            }
        )
        pipeline.state["raw_data"] = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0]}, index=index)

        report = _build_pipeline_operational_monitoring(
            pipeline,
            expected_feature_columns=["feature_a"],
            actual_feature_columns=["feature_a"],
            scope="training",
        )

        self.assertEqual(report["policy"]["policy_profile"], "trade_ready")
        self.assertEqual(float(report["policy"]["min_fill_ratio"]), 0.25)
        self.assertEqual(report["policy"]["max_data_lag"], "2h")
        self.assertEqual(int(report["policy"]["max_queue_backlog"]), 0)

    def test_deployment_readiness_passes_for_healthy_champion_with_rollback_available(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            _register_champion(store, "BTCUSDT", 0.10)
            current_champion = _register_champion(store, "BTCUSDT", 0.14)

            report = build_deployment_readiness_report(
                store=store,
                symbol="BTCUSDT",
                monitoring_report={"healthy": True, "reasons": []},
                drift_cycle={
                    "drift_guardrails": {"approved": False, "reasons": []},
                    "retrain_status": "not_recommended",
                },
                backend_status={"adapter": "nautilus", "available": True, "reasons": []},
            )

            self.assertTrue(report["ready"])
            self.assertEqual(report["operator_action"], "deploy")
            self.assertEqual(report["version_id"], current_champion)
            self.assertEqual(report["summary"]["failed_components"], [])
            self.assertTrue(report["components"]["rollback"]["available"])

    def test_deployment_readiness_blocks_on_pending_drift_backend_and_missing_rollback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            _register_champion(store, "BTCUSDT", 0.12)

            report = build_deployment_readiness_report(
                store=store,
                symbol="BTCUSDT",
                monitoring_report={"healthy": True, "reasons": []},
                drift_cycle={
                    "drift_guardrails": {"approved": True, "reasons": ["feature_drift"]},
                    "retrain_status": "scheduled_window_pending",
                },
                backend_status={"adapter": "nautilus", "available": False, "reasons": ["maintenance"]},
            )

            self.assertFalse(report["ready"])
            self.assertIn("drift_status", report["summary"]["failed_components"])
            self.assertIn("backend", report["summary"]["failed_components"])
            self.assertIn("rollback", report["summary"]["failed_components"])
            self.assertIn("drift_retraining_recommended", report["reasons"])
            self.assertIn("scheduled_window_pending", report["reasons"])
            self.assertIn("backend_unavailable", report["reasons"])
            self.assertIn("rollback_unavailable", report["reasons"])

    def test_pipeline_wrapper_reuses_state_for_deployment_readiness(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            _register_champion(store, "BTCUSDT", 0.10)
            current_champion = _register_champion(store, "BTCUSDT", 0.14)

            pipeline = ResearchPipeline({"data": {"symbol": "BTCUSDT", "interval": "1h"}})
            pipeline.state["operational_monitoring"] = {"healthy": True, "reasons": []}
            pipeline.state["drift_cycle"] = {
                "drift_guardrails": {"approved": False, "reasons": []},
                "retrain_status": "not_recommended",
            }

            report = pipeline.inspect_deployment_readiness(
                store=store,
                backend_status={"adapter": "nautilus", "available": True, "reasons": []},
            )

            self.assertTrue(report["ready"])
            self.assertEqual(report["version_id"], current_champion)
            self.assertEqual(pipeline.state["deployment_readiness"]["operator_action"], "deploy")


if __name__ == "__main__":
    unittest.main()