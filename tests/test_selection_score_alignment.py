import tempfile
import unittest

import pandas as pd

from core import (
    LocalRegistryStore,
    build_model,
    create_promotion_eligibility_report,
    evaluate_challenger_promotion,
    finalize_promotion_eligibility_report,
    upsert_promotion_gate,
)
from core.automl import compute_objective_value


def _fit_logistic_model():
    X = pd.DataFrame({"f1": [0.0, 1.0, 0.0, 1.0], "f2": [1.0, 1.0, 0.0, 0.0]})
    y = pd.Series([0, 1, 0, 1])
    model = build_model("logistic", {"c": 1.0})
    model.fit(X, y)
    return model, list(X.columns)


def _make_report(score_basis, score_value, *, feature_portability_pass=True, regime_stability_pass=True):
    report = create_promotion_eligibility_report(score_basis=score_basis, score_value=score_value)
    report = upsert_promotion_gate(
        report,
        group="selection",
        name="feature_portability",
        passed=feature_portability_pass,
        mode="blocking",
        reason=None if feature_portability_pass else "feature_portability_failed",
    )
    report = upsert_promotion_gate(
        report,
        group="selection",
        name="regime_stability",
        passed=regime_stability_pass,
        mode="blocking",
        reason=None if regime_stability_pass else "regime_stability_failed",
    )
    report = upsert_promotion_gate(report, group="post_selection", name="locked_holdout", passed=True)
    report = upsert_promotion_gate(report, group="post_selection", name="locked_holdout_gap", passed=True)
    return finalize_promotion_eligibility_report(report)


class SelectionScoreAlignmentTest(unittest.TestCase):
    def test_trading_objective_defaults_to_confidence_lower_bound(self):
        training = {"avg_directional_accuracy": 0.55, "avg_accuracy": 0.55}
        backtest = {
            "net_profit_pct": 0.10,
            "sharpe_ratio": 2.0,
            "max_drawdown": -0.05,
            "total_trades": 12,
            "bar_count": 120,
            "statistical_significance": {
                "metrics": {
                    "sharpe_ratio": {"confidence_interval": {"lower": 0.8, "upper": 2.9}},
                }
            },
        }

        default_score = compute_objective_value(
            "risk_adjusted_after_costs",
            training,
            backtest,
            {"objective_gates": {"enabled": False}},
        )
        lower_bound_score = compute_objective_value(
            "risk_adjusted_after_costs",
            training,
            backtest,
            {"objective_gates": {"enabled": False}, "objective_use_confidence_lower_bound": True},
        )
        point_estimate_score = compute_objective_value(
            "risk_adjusted_after_costs",
            training,
            backtest,
            {"objective_gates": {"enabled": False}, "objective_use_confidence_lower_bound": False},
        )

        self.assertAlmostEqual(default_score, lower_bound_score, places=6)
        self.assertGreater(point_estimate_score, default_score)

    def test_registry_blocks_score_basis_mismatch_between_champion_and_challenger(self):
        model, feature_columns = _fit_logistic_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalRegistryStore(root_dir=temp_dir)
            champion_report = _make_report("selection_value", 0.12)
            version_id = store.register_version(
                model,
                symbol="BTCUSDT",
                feature_columns=feature_columns,
                training_summary={"avg_f1_macro": 0.75},
                validation_summary={"raw_objective_value": 0.12},
                promotion_eligibility_report=champion_report,
            )
            store.promote(version_id, "champion", symbol="BTCUSDT", decision={"approved": True, "reasons": ["approved"]})

            decision = evaluate_challenger_promotion(
                {
                    "promotion_eligibility_report": _make_report("locked_holdout_raw_objective", 0.14),
                    "selection_value": 0.14,
                    "sample_count": 500,
                },
                champion_record=store.get_champion("BTCUSDT"),
            )

            self.assertFalse(decision["approved"])
            self.assertIn("promotion_score_basis_mismatch", decision["reasons"])

    def test_blocking_governance_gate_from_selection_report_blocks_promotion(self):
        decision = evaluate_challenger_promotion(
            {
                "promotion_eligibility_report": _make_report(
                    "locked_holdout_raw_objective",
                    0.15,
                    feature_portability_pass=False,
                    regime_stability_pass=True,
                ),
                "selection_value": 0.15,
                "sample_count": 500,
            }
        )

        self.assertFalse(decision["approved"])
        self.assertIn("feature_portability_failed", decision["promotion_eligibility_report"]["blocking_failures"])


if __name__ == "__main__":
    unittest.main()