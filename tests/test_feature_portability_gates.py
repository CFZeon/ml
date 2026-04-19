import unittest

import pandas as pd

from core import (
    build_reference_overlay_feature_block,
    derive_feature_metadata,
    evaluate_feature_portability,
)


class FeaturePortabilityGateTest(unittest.TestCase):
    def test_venue_specific_features_are_tagged_automatically(self):
        metadata = derive_feature_metadata(
            feature_blocks={
                "return_1": "price_volume",
                "fut_funding_rate": "futures_context",
                "ref_price_gap": "reference_overlay",
                "composite_basis": "reference_overlay",
            },
            columns=["return_1", "fut_funding_rate", "ref_price_gap", "composite_basis"],
        )

        self.assertEqual(metadata["return_1"]["portability_class"], "endogenous")
        self.assertEqual(metadata["fut_funding_rate"]["portability_class"], "venue_specific")
        self.assertTrue(metadata["fut_funding_rate"]["exchange_specific_semantics"])
        self.assertEqual(metadata["ref_price_gap"]["portability_class"], "reference_overlay")
        self.assertEqual(metadata["composite_basis"]["portability_class"], "cross_venue_composite")

    def test_promotion_gate_fails_when_edge_depends_on_venue_specific_features(self):
        metadata = derive_feature_metadata(
            feature_blocks={
                "fut_funding_rate": "futures_context",
                "return_1": "price_volume",
            },
            columns=["fut_funding_rate", "return_1"],
        )
        diagnostics = evaluate_feature_portability(
            metadata,
            top_features=[
                {"feature": "fut_funding_rate", "avg_native_importance": 0.85},
                {"feature": "return_1", "avg_native_importance": 0.15},
            ],
            family_diagnostics={
                "bundles": [
                    {
                        "bundle": "endogenous_only",
                        "avg_accuracy_drop_vs_full": 0.08,
                        "avg_f1_drop_vs_full": 0.09,
                    },
                    {
                        "bundle": "full_context",
                        "avg_accuracy_drop_vs_full": 0.0,
                        "avg_f1_drop_vs_full": 0.0,
                    },
                ]
            },
            config={
                "max_venue_specific_importance_share": 0.5,
                "max_venue_specific_top_feature_share": 0.5,
                "max_endogenous_accuracy_drop": 0.02,
                "max_endogenous_f1_drop": 0.02,
            },
        )

        self.assertFalse(diagnostics["promotion_pass"])
        self.assertIn("venue_specific_importance_dominates", diagnostics["reasons"])
        self.assertIn("endogenous_ablation_failed", diagnostics["reasons"])

    def test_generic_reference_overlay_avoids_venue_named_interface(self):
        index = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
        base_data = pd.DataFrame(
            {
                "close": [100.0, 101.0, 102.0, 103.0],
                "volume": [10.0, 12.0, 15.0, 11.0],
            },
            index=index,
        )
        reference_data = pd.DataFrame(
            {
                "reference_price": [100.2, 101.4, 101.8, 103.5],
                "reference_volume": [20.0, 21.0, 19.0, 22.0],
                "breadth": [0.1, 0.2, -0.1, 0.05],
                "composite_funding_rate": [0.0001, 0.0002, 0.0001, -0.0001],
                "composite_basis": [0.001, 0.0015, 0.0008, 0.0012],
            },
            index=index,
        )

        block = build_reference_overlay_feature_block(base_data, reference_data, rolling_window=2)

        self.assertFalse(block.frame.empty)
        self.assertIn("ref_price_gap", block.frame.columns)
        self.assertIn("ref_volume_ratio", block.frame.columns)
        self.assertIn("composite_basis", block.frame.columns)
        self.assertFalse(any("binance" in column.lower() for column in block.frame.columns))


if __name__ == "__main__":
    unittest.main()