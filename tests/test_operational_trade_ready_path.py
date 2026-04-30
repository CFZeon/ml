import tempfile
import unittest

import pandas as pd

from core import (
    LocalRegistryStore,
    ResearchPipeline,
    build_deployment_readiness_report,
    build_live_calibration_report,
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


def _make_paper_report(*, duration_days=35.0):
    return build_live_calibration_report(
        certified_expectations={"modeled_slippage_bps": 1.8, "modeled_fill_ratio": 0.94},
        paper_metrics={
            "mode": "paper",
            "duration_days": duration_days,
            "modeled_slippage_bps": 1.8,
            "realized_slippage_bps": 1.92,
            "modeled_fill_ratio": 0.94,
            "realized_fill_ratio": 0.90,
            "data_breaches": 0,
            "funding_breaches": 0,
            "kill_switch_triggers": 0,
        },
    )


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
                paper_report=_make_paper_report(),
            )

            self.assertTrue(report["ready"])
            self.assertFalse(report["capital_release_eligible"])
            self.assertEqual(report["capital_release_stage"], "paper_verified")
            self.assertEqual(report["operator_action"], "paper")
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
                paper_report=_make_paper_report(),
            )

            self.assertFalse(report["ready"])
            self.assertIn("drift_status", report["summary"]["failed_components"])
            self.assertIn("backend", report["summary"]["failed_components"])
            self.assertIn("rollback", report["summary"]["failed_components"])
            self.assertIn("drift_retraining_recommended", report["reasons"])
            self.assertIn("scheduled_window_pending", report["reasons"])
            self.assertIn("backend_unavailable", report["reasons"])
            self.assertIn("rollback_unavailable", report["reasons"])

    def test_deployment_readiness_blocks_on_expired_model_freshness(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            _register_champion(store, "BTCUSDT", 0.10)
            current_champion = _register_champion(store, "BTCUSDT", 0.14)

            champion_record = store.get_champion("BTCUSDT")
            freshness_anchor = pd.Timestamp(champion_record["promoted_at"] or champion_record["created_at"])
            report = build_deployment_readiness_report(
                store=store,
                symbol="BTCUSDT",
                monitoring_report={"healthy": True, "reasons": []},
                drift_cycle={
                    "drift_guardrails": {"approved": False, "reasons": []},
                    "retrain_status": "not_recommended",
                },
                backend_status={"adapter": "nautilus", "available": True, "reasons": []},
                paper_report=_make_paper_report(),
                policy={
                    "max_model_age_days": 28,
                    "warn_model_age_days": 21,
                    "as_of_timestamp": freshness_anchor + pd.Timedelta(days=35),
                },
            )

            self.assertFalse(report["ready"])
            self.assertEqual(report["version_id"], current_champion)
            self.assertEqual(report["capital_release_stage"], "paper_verified")
            self.assertEqual(report["operator_action"], "hold")
            self.assertIn("model_freshness", report["summary"]["failed_components"])
            self.assertIn("model_expired", report["reasons"])
            self.assertFalse(report["components"]["model_freshness"]["passed"])
            self.assertTrue(report["components"]["model_freshness"]["expired"])

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
                paper_report=_make_paper_report(),
            )

            self.assertTrue(report["ready"])
            self.assertFalse(report["capital_release_eligible"])
            self.assertEqual(report["capital_release_stage"], "paper_verified")
            self.assertEqual(report["version_id"], current_champion)
            self.assertEqual(pipeline.state["deployment_readiness"]["operator_action"], "paper")

    def test_pipeline_can_build_and_attach_paper_calibration_from_observations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            _register_champion(store, "BTCUSDT", 0.10)
            current_champion = _register_champion(store, "BTCUSDT", 0.14)

            observations = pd.DataFrame(
                {
                    "timestamp": pd.date_range("2026-12-01", periods=35, freq="D", tz="UTC"),
                    "mode": "paper",
                    "trade_count": 12,
                    "modeled_slippage_bps": 1.8,
                    "realized_slippage_bps": 1.9,
                    "modeled_fill_ratio": 0.94,
                    "realized_fill_ratio": 0.91,
                    "data_breach": 0,
                    "funding_breach": 0,
                    "kill_switch_trigger": 0,
                }
            )

            pipeline = ResearchPipeline({"data": {"symbol": "BTCUSDT", "interval": "1h"}})
            paper_report = pipeline.inspect_paper_trading_calibration(
                certified_expectations={"modeled_slippage_bps": 1.8, "modeled_fill_ratio": 0.94},
                paper_observations=observations,
            )
            self.assertTrue(paper_report["passed"])
            self.assertEqual(pipeline.state["paper_calibration"]["mode"], "paper")

            store.attach_paper_report(current_champion, paper_report, symbol="BTCUSDT")
            pipeline.state.pop("paper_calibration", None)
            pipeline.state["operational_monitoring"] = {"healthy": True, "reasons": []}
            pipeline.state["drift_cycle"] = {
                "drift_guardrails": {"approved": False, "reasons": []},
                "retrain_status": "not_recommended",
            }

            readiness = pipeline.inspect_deployment_readiness(
                store=store,
                backend_status={"adapter": "nautilus", "available": True, "reasons": []},
            )

            self.assertTrue(readiness["ready"])
            self.assertEqual(readiness["capital_release_stage"], "paper_verified")
            self.assertTrue(readiness["components"]["paper_calibration"]["passed"])

    def test_pipeline_operational_limits_refresh_when_equity_curve_changes(self):
        pipeline = ResearchPipeline({"data": {"symbol": "BTCUSDT", "interval": "1h"}})

        green = pipeline.inspect_operational_limits(
            operational_limits={"healthy": True, "kill_switch_ready": True},
            equity_curve=pd.Series([1.0, 1.10, 1.056], dtype=float),
        )
        breached = pipeline.inspect_operational_limits(
            operational_limits={"healthy": True, "kill_switch_ready": True},
            equity_curve=pd.Series([1.0, 1.10, 0.94], dtype=float),
        )

        self.assertFalse(green["drawdown_breached"])
        self.assertTrue(breached["drawdown_breached"])
        self.assertTrue(breached["kill_switch_triggered"])
        self.assertAlmostEqual(breached["current_drawdown"], 0.94 / 1.10 - 1.0, places=6)

    def test_promoted_drift_cycle_does_not_block_micro_capital_release(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            _register_champion(store, "BTCUSDT", 0.10)
            current_champion = _register_champion(store, "BTCUSDT", 0.14)

            pipeline = ResearchPipeline({"data": {"symbol": "BTCUSDT", "interval": "1h"}})
            pipeline.state["operational_monitoring"] = {"healthy": True, "reasons": []}
            pipeline.state["drift_cycle"] = {
                "drift_guardrails": {"approved": True, "reasons": ["approved", "model_ttl_expired"]},
                "retrain_status": "promoted",
            }
            pipeline.inspect_operational_limits(
                operational_limits={"healthy": True, "kill_switch_ready": True},
                equity_curve=pd.Series([1.0, 1.10, 1.06], dtype=float),
            )

            store.attach_paper_report(current_champion, _make_paper_report(), symbol="BTCUSDT")
            report = pipeline.inspect_deployment_readiness(
                store=store,
                backend_status={"adapter": "nautilus", "available": True, "reasons": []},
                release_request={"requested_stage": "micro_capital", "manual_acknowledged": True},
            )

            self.assertEqual(report["capital_release_stage"], "micro_capital")
            self.assertTrue(report["capital_release_eligible"])
            self.assertEqual(report["operator_action"], "deploy")
            self.assertNotIn("approved", report["release_blockers"])
            self.assertNotIn("model_ttl_expired", report["reasons"])


if __name__ == "__main__":
    unittest.main()