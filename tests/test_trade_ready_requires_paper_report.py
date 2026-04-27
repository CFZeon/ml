import tempfile
import unittest

import pandas as pd

from core import (
    LocalRegistryStore,
    build_deployment_readiness_report,
    build_live_calibration_report,
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
    )


class TradeReadyPaperGateTest(unittest.TestCase):
    def test_deployment_readiness_blocks_without_paper_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            _register_champion(store, "BTCUSDT", 0.14)
            _register_champion(store, "BTCUSDT", 0.18)

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

            self.assertFalse(report["ready"])
            self.assertFalse(report["capital_release_eligible"])
            self.assertIn("paper_calibration", report["summary"]["failed_components"])
            self.assertIn("paper_calibration_unavailable", report["reasons"])

    def test_deployment_readiness_uses_attached_paper_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            _register_champion(store, "BTCUSDT", 0.14)
            current_champion = _register_champion(store, "BTCUSDT", 0.18)
            store.attach_paper_report(current_champion, _make_paper_report(), symbol="BTCUSDT")

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
            self.assertFalse(report["capital_release_eligible"])
            self.assertEqual(report["capital_release_stage"], "paper_verified")
            self.assertEqual(report["version_id"], current_champion)
            self.assertTrue(report["components"]["paper_calibration"]["passed"])


if __name__ == "__main__":
    unittest.main()