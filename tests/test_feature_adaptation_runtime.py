import json
import unittest

import pandas as pd

from core import (
    FeaturePolicyContract,
    apply_feature_adaptation_to_splits,
    build_feature_adapter,
    resolve_feature_adaptation_config,
    validate_feature_adaptation_config_contract,
    validate_feature_adaptation_runtime_support,
)


class FeatureAdaptationRuntimeTest(unittest.TestCase):
    def test_feature_policy_contract_round_trips_cleanly(self):
        contract = FeaturePolicyContract(
            policy_id="identity",
            feature_columns=["ret_1", "vol_24"],
            disabled_columns=[],
            generated_columns=[],
            regime_column="regime",
            scaling_mode="identity",
            fallback_mode="global",
            sparse_regimes=["2"],
            metadata={"requested_enabled": True, "no_op": True},
        )

        payload = contract.to_dict()
        restored = FeaturePolicyContract.from_dict(payload)

        self.assertEqual(payload["policy_id"], "identity")
        self.assertEqual(restored.policy_id, contract.policy_id)
        self.assertEqual(restored.feature_columns, contract.feature_columns)
        self.assertEqual(restored.metadata["requested_enabled"], True)
        json.dumps(payload)

    def test_identity_adapter_is_no_op_when_only_deferred_sections_are_requested(self):
        index = pd.date_range("2026-05-01", periods=6, freq="1h", tz="UTC")
        X_fit = pd.DataFrame({"ret_1": [0, 1, 2], "vol_24": [3.0, 4.0, 5.0]}, index=index[:3])
        X_val = pd.DataFrame({"ret_1": [3, 4], "vol_24": [6.0, 7.0]}, index=index[3:5])
        X_test = pd.DataFrame({"ret_1": [5], "vol_24": [8.0]}, index=index[5:])
        fit_regime = pd.DataFrame({"regime": [0, 1, 0], "regime_confidence": [0.6, 0.7, 0.8]}, index=X_fit.index)
        val_regime = pd.DataFrame({"regime": [1, 1], "regime_confidence": [0.4, 0.5]}, index=X_val.index)
        test_regime = pd.DataFrame({"regime": [0], "regime_confidence": [0.55]}, index=X_test.index)
        config = {
            "scaling": {"mode": "identity", "fallback": "identity"},
            "interaction_budget": {"enabled": True, "max_features": 4},
        }

        resolved = resolve_feature_adaptation_config(config)
        adapter = build_feature_adapter(config, regime_column="regime")
        adapter.fit(X_fit, fit_regime)
        transformed_fit, policy = adapter.transform(X_fit, fit_regime)
        batch = apply_feature_adaptation_to_splits(
            X_fit,
            X_val,
            X_test,
            fit_regime_frame=fit_regime,
            validation_regime_frame=val_regime,
            test_regime_frame=test_regime,
            config=config,
            regime_column="regime",
        )

        self.assertTrue(transformed_fit.equals(X_fit))
        self.assertTrue(batch.fit_frame.equals(X_fit))
        self.assertTrue(batch.validation_frame.equals(X_val))
        self.assertTrue(batch.test_frame.equals(X_test))
        self.assertEqual(policy.scaling_mode, "identity")
        self.assertEqual(batch.summary["adapter_type"], "identity")
        self.assertEqual(batch.summary["input_features"], batch.summary["output_features"])
        self.assertTrue(batch.summary["deferred_runtime"])
        self.assertEqual(resolved["requested_scaling_mode"], "identity")
        self.assertEqual(resolved["requested_selection_mode"], "identity")
        self.assertEqual(batch.fit_policy.metadata["requested_scaling_mode"], "identity")
        json.dumps(batch.summary)
        json.dumps(batch.manifest)
        json.dumps(batch.fit_policy.to_dict())

    def test_feature_strategy_scaling_is_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "strategy='feature'"):
            validate_feature_adaptation_runtime_support(
                {"scaling": {"mode": "regime_conditioned", "fallback": "global"}},
                regime_aware_strategy="feature",
            )

    def test_feature_strategy_masking_is_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "strategy='feature'"):
            validate_feature_adaptation_runtime_support(
                {
                    "selection": {"mode": "per_regime_mask", "fallback": "global"},
                    "disable_incompatible_features": True,
                },
                regime_aware_strategy="feature",
            )

    def test_invalid_interaction_budget_contract_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "max_features"):
            validate_feature_adaptation_config_contract(
                {"interaction_budget": {"enabled": True, "max_features": -1}}
            )

        with self.assertRaisesRegex(ValueError, "must be a mapping"):
            resolve_feature_adaptation_config({"interaction_budget": ["bad"]})


if __name__ == "__main__":
    unittest.main()