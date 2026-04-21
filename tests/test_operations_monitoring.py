import tempfile
import unittest
from pathlib import Path

import pandas as pd

from core import (
    LocalRegistryStore,
    build_model,
    build_monitoring_report,
    create_promotion_eligibility_report,
    evaluate_challenger_promotion,
    finalize_promotion_eligibility_report,
    resolve_canonical_promotion_score,
    run_backtest,
    upsert_promotion_gate,
    write_monitoring_artifacts,
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
    report = upsert_promotion_gate(
        report,
        group="selection",
        name="feature_admission",
        passed=True,
    )
    report = upsert_promotion_gate(
        report,
        group="selection",
        name="feature_portability",
        passed=True,
    )
    report = upsert_promotion_gate(
        report,
        group="selection",
        name="regime_stability",
        passed=True,
    )
    report = upsert_promotion_gate(
        report,
        group="selection",
        name="operational_health",
        passed=True,
    )
    report = upsert_promotion_gate(
        report,
        group="post_selection",
        name="locked_holdout",
        passed=True,
    )
    report = upsert_promotion_gate(
        report,
        group="post_selection",
        name="locked_holdout_gap",
        passed=True,
    )
    return finalize_promotion_eligibility_report(report)


class OperationsMonitoringTest(unittest.TestCase):
    def test_freshness_breach_is_reported_and_artifacts_are_written(self):
        index = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
        report = build_monitoring_report(
            data_index=index,
            expected_data_end=index[-1] + pd.Timedelta("2h"),
            max_data_lag="30min",
        )

        self.assertFalse(report["healthy"])
        self.assertIn("raw_data_freshness", report["reasons"])

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_monitoring_artifacts(report, temp_dir, run_id="freshness")
            self.assertTrue(Path(artifacts["json"]).exists())
            self.assertTrue(Path(artifacts["parquet"]).exists())
            self.assertTrue(Path(artifacts["markdown"]).exists())

    def test_schema_drift_fails_closed(self):
        report = build_monitoring_report(
            expected_feature_columns=["feature_a", "feature_b"],
            actual_feature_columns=["feature_a", "feature_c"],
        )

        self.assertFalse(report["healthy"])
        self.assertIn("feature_schema", report["reasons"])
        self.assertTrue(report["components"]["feature_schema"]["fail_closed"])
        self.assertEqual(report["components"]["feature_schema"]["missing_columns"], ["feature_b"])

    def test_fill_quality_and_slippage_drift_are_measurable_on_replay(self):
        index = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
        close = pd.Series([100.0, 100.0, 100.0, 100.0], index=index)
        signals = pd.Series([0.0, 1.0, 1.0, 0.0], index=index)

        baseline = run_backtest(
            close=close,
            signals=signals,
            execution_prices=close,
            equity=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            signal_delay_bars=0,
            engine="pandas",
            volume=pd.Series([10_000.0] * len(index), index=index),
            execution_policy={"adapter": "nautilus", "time_in_force": "IOC", "participation_cap": 1.0},
        )
        stressed = run_backtest(
            close=close,
            signals=signals,
            execution_prices=close,
            equity=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.01,
            signal_delay_bars=0,
            engine="pandas",
            volume=pd.Series([1.0] * len(index), index=index),
            execution_policy={"adapter": "nautilus", "time_in_force": "IOC", "participation_cap": 0.5},
        )

        report = build_monitoring_report(
            backtest_reports=[stressed],
            baseline_backtest_report=baseline,
            policy={
                "min_fill_ratio": 0.5,
                "max_fill_ratio_deterioration": 0.1,
                "max_slippage_drift": 0.5,
            },
        )
        execution = report["components"]["execution_quality"]

        self.assertFalse(report["healthy"])
        self.assertLess(execution["worst_fill_ratio"], baseline["fill_ratio"])
        self.assertGreater(execution["fill_ratio_deterioration"], 0.0)
        self.assertGreater(execution["slippage_drift"], 0.0)

    def test_registry_promotion_can_require_operational_health(self):
        model, feature_columns = _fit_logistic_model()
        unhealthy_report = build_monitoring_report(
            expected_feature_columns=feature_columns,
            actual_feature_columns=["f1", "unexpected_feature"],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            version_id = store.register_version(
                model,
                symbol="BTCUSDT",
                feature_columns=feature_columns,
                training_summary={"avg_f1_macro": 0.75},
                validation_summary={"raw_objective_value": 0.12},
            )
            monitoring_path = store.attach_monitoring_report(version_id, unhealthy_report, symbol="BTCUSDT")
            decision = evaluate_challenger_promotion(
                {
                    "promotion_ready": True,
                    "selection_value": 0.12,
                    "sample_count": 500,
                    "promotion_eligibility_report": _make_eligibility_report(0.12),
                },
                monitoring_report=unhealthy_report,
                policy={"require_operational_health": True},
            )

            self.assertTrue(Path(monitoring_path).exists())
            self.assertFalse(decision["approved"])
            self.assertIn("operational_monitoring_not_healthy", decision["reasons"])

    def test_signal_decay_deterioration_is_reported(self):
        report = build_monitoring_report(
            signal_decay_report={
                "promotion_pass": True,
                "gate_mode": "blocking",
                "trade_count": 40,
                "half_life_bars": 3,
                "net_edge_at_effective_delay": 0.004,
                "edge_retention_at_effective_delay": 0.25,
            },
            baseline_signal_decay_report={
                "half_life_bars": 6,
                "net_edge_at_effective_delay": 0.020,
            },
            policy={
                "max_signal_delay_edge_deterioration": 0.01,
            },
        )

        self.assertFalse(report["healthy"])
        self.assertIn("signal_decay", report["reasons"])
        self.assertEqual(report["components"]["signal_decay"]["reason"], "signal_delay_edge_deterioration")


if __name__ == "__main__":
    unittest.main()