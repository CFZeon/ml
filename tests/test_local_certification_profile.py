import unittest

import pandas as pd

from core import ResearchPipeline
from core.automl import _build_state_bundle
from core.lookahead import _seed_replay_state
from core.pipeline import _resolve_backtest_execution_policy
from example_local_certification_automl import (
    build_local_certification_example_config,
    prepare_local_certification_runtime_config,
)
from example_utils import prepare_example_runtime_config


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

    def test_prepare_local_certification_runtime_config_falls_back_to_explicit_surrogate_without_nautilus(self):
        config = build_local_certification_example_config(automl_storage="local-certification-test.db")

        resolved = prepare_local_certification_runtime_config(config, nautilus_available=False)

        self.assertEqual(resolved["backtest"]["evaluation_mode"], "local_certification")
        self.assertEqual(resolved["backtest"]["execution_profile"], "local_surrogate_certification")
        self.assertEqual(resolved["backtest"]["execution_policy"]["adapter"], "bar_surrogate")
        self.assertFalse(bool(resolved["backtest"]["execution_policy"].get("force_simulation", False)))
        self.assertEqual(resolved["example_runtime"]["mode"], "local_certification_surrogate")

    def test_prepare_example_runtime_config_uses_surrogate_local_certification_without_nautilus(self):
        resolved = prepare_example_runtime_config(
            {"data": {"market": "spot"}, "backtest": {}},
            market="spot",
            local_certification=True,
            nautilus_available=False,
            example_name="example.py",
        )

        self.assertEqual(resolved["backtest"]["evaluation_mode"], "local_certification")
        self.assertEqual(resolved["backtest"]["execution_profile"], "local_surrogate_certification")
        self.assertEqual(resolved["backtest"]["execution_policy"]["adapter"], "bar_surrogate")
        self.assertEqual(resolved["example_runtime"]["mode"], "local_certification_surrogate")

    def test_local_certification_execution_policy_allows_explicit_bar_surrogate(self):
        pipeline = ResearchPipeline(
            {
                "backtest": {
                    "evaluation_mode": "local_certification",
                    "execution_policy": {"adapter": "bar_surrogate"},
                }
            }
        )

        resolved = _resolve_backtest_execution_policy(pipeline)

        self.assertEqual(resolved["adapter"], "bar_surrogate")

    def test_automl_state_bundle_preserves_data_certification_reports(self):
        index = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
        frame = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0]}, index=index)
        pipeline = ResearchPipeline({"data": {"symbol": "BTCUSDT", "interval": "1h"}})
        pipeline.state["raw_data"] = frame.copy()
        pipeline.state["data"] = frame.copy()
        pipeline.state["data_integrity_report"] = {"status": "complete", "missing_rows": 0}
        pipeline.state["data_quality_report"] = {"status": "pass", "summary": {"quarantined_rows": 0}}
        pipeline.state["reference_integrity_report"] = {"promotion_pass": True}

        bundle = _build_state_bundle(pipeline)

        self.assertEqual(bundle["data_integrity_report"]["status"], "complete")
        self.assertEqual(bundle["data_quality_report"]["status"], "pass")
        self.assertTrue(bundle["reference_integrity_report"]["promotion_pass"])

    def test_lookahead_replay_state_preserves_data_certification_reports(self):
        base_state = {
            "raw_data": pd.DataFrame(
                {"close": [100.0, 101.0, 102.0, 103.0]},
                index=pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
            ),
            "data": pd.DataFrame(
                {"close": [100.0, 101.0, 102.0, 103.0]},
                index=pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
            ),
            "data_integrity_report": {"status": "complete", "missing_rows": 0},
            "data_quality_report": {"status": "pass", "summary": {"quarantined_rows": 0}},
            "reference_integrity_report": {"promotion_pass": True},
        }

        seeded = _seed_replay_state(base_state)

        self.assertEqual(seeded["data_integrity_report"]["status"], "complete")
        self.assertEqual(seeded["data_quality_report"]["status"], "pass")
        self.assertTrue(seeded["reference_integrity_report"]["promotion_pass"])

    def test_prepare_local_certification_runtime_config_returns_original_config_when_available(self):
        config = {"backtest": {"evaluation_mode": "local_certification"}}

        resolved = prepare_local_certification_runtime_config(config, nautilus_available=True)

        self.assertIs(resolved, config)