import unittest

import pandas as pd

from core.automl import _resolve_evidence_gate
from core.feature_governance import evaluate_feature_portability, summarize_feature_admission_reports
from core.regime import summarize_regime_ablation_reports
from core.signal_decay import build_signal_decay_report


class MissingEvidenceGateResolutionTest(unittest.TestCase):
    def test_missing_feature_portability_evidence_is_unknown(self):
        diagnostics = evaluate_feature_portability({}, top_features=[])

        self.assertEqual(diagnostics["status"], "unknown")
        self.assertFalse(diagnostics["promotion_pass"])
        self.assertIn("feature_portability_evidence_missing", diagnostics["reasons"])

    def test_missing_feature_admission_evidence_is_unknown(self):
        summary = summarize_feature_admission_reports([])

        self.assertEqual(summary["status"], "unknown")
        self.assertFalse(summary["promotion_pass"])
        self.assertIn("feature_admission_evidence_missing", summary["reasons"])

    def test_missing_regime_ablation_evidence_is_unknown(self):
        summary = summarize_regime_ablation_reports([])

        self.assertEqual(summary["status"], "unknown")
        self.assertFalse(summary["promotion_pass"])
        self.assertIn("regime_ablation_evidence_missing", summary["reasons"])

    def test_low_sample_signal_decay_is_unknown_not_passed(self):
        index = pd.date_range("2026-03-12", periods=6, freq="1h", tz="UTC")
        valuation = pd.Series([100.0, 101.0, 100.5, 100.0, 100.0, 100.0], index=index)
        signals = pd.Series([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=index)

        report = build_signal_decay_report(
            [
                {
                    "predictions": signals,
                    "direction_edge": pd.Series([0.6, 0.0, 0.0, 0.0, 0.0, 0.0], index=index),
                    "event_signals": signals,
                    "valuation_prices": valuation,
                    "execution_prices": valuation,
                }
            ],
            holding_bars=1,
            signal_delay_bars=1,
            execution_policy={"max_order_age_bars": 1, "cancel_replace_bars": 1},
            config={
                "min_realized_trade_count": 5,
                "min_half_life_holding_ratio": 0.0,
            },
        )

        self.assertEqual(report["status"], "unknown")
        self.assertFalse(report["promotion_pass"])
        self.assertEqual(report["gate_mode"], "advisory")
        self.assertIn("signal_decay_evidence_insufficient", report["reasons"])

    def test_automl_gate_resolution_blocks_missing_portability_evidence(self):
        gate = _resolve_evidence_gate(
            "feature_portability",
            {},
            selection_policy={"gate_modes": {"feature_portability": "blocking"}},
            failure_reason="feature_portability_failed",
        )

        self.assertEqual(gate["status"], "unknown")
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["mode"], "blocking")
        self.assertEqual(gate["reason"], "feature_portability_evidence_missing")


if __name__ == "__main__":
    unittest.main()