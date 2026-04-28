import unittest

from example_local_certification_automl import (
    build_local_certification_example_config,
    prepare_local_certification_runtime_config,
)


class LocalCertificationProfileTest(unittest.TestCase):
    def test_build_local_certification_example_config_uses_strict_local_profile(self):
        config = build_local_certification_example_config(automl_storage="local-certification-test.db")

        self.assertEqual(config["backtest"]["evaluation_mode"], "local_certification")
        self.assertEqual(config["backtest"]["execution_profile"], "local_l1_certification")
        self.assertEqual(config["backtest"]["execution_policy"]["adapter"], "nautilus")
        self.assertFalse(config["backtest"]["execution_policy"]["force_simulation"])
        self.assertEqual(config["backtest"]["required_stress_scenarios"], ["downtime", "stale_mark", "halt"])
        self.assertEqual(config["automl"]["trade_ready_profile"]["name"], "local_certification")
        self.assertEqual(config["automl"]["trade_ready_profile"]["min_significance_observations"], 48)
        self.assertEqual(config["automl"]["selection_policy"]["min_validation_trade_count"], 20)
        self.assertEqual(config["monitoring"]["policy_profile"], "local_certification")
        self.assertTrue(config["data_certification"]["enabled"])
        self.assertEqual(config["example_runtime"]["mode"], "local_certification")

    def test_prepare_local_certification_runtime_config_fails_closed_without_nautilus(self):
        with self.assertRaisesRegex(RuntimeError, "Local certification requires a local Nautilus installation"):
            prepare_local_certification_runtime_config({"backtest": {}}, nautilus_available=False)

    def test_prepare_local_certification_runtime_config_returns_original_config_when_available(self):
        config = {"backtest": {"evaluation_mode": "local_certification"}}

        resolved = prepare_local_certification_runtime_config(config, nautilus_available=True)

        self.assertIs(resolved, config)