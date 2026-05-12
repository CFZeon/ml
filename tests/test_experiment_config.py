import tempfile
import unittest
from pathlib import Path

from example_utils import build_spot_research_config
from experiments import load_experiment_config


class ExperimentConfigTest(unittest.TestCase):
    def test_load_experiment_config_expands_defaults_and_quick_overrides(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "baseline.yaml"
            config_path.write_text(
                """
experiment:
  name: btc_baseline
data:
  symbol: BTCUSDT
  interval: 1h
  start: 2024-01-01
  end: 2024-04-01
  context_symbols: [ETHUSDT]
indicators:
  - kind: returns
    params:
      periods: [1, 2]
model:
  cv_method: walk_forward
  n_splits: 4
  gap: 8
quick_overrides:
  data:
    end: 2024-02-01
""".strip(),
                encoding="utf-8",
            )

            resolved = load_experiment_config(config_path, quick=True)

        self.assertEqual(resolved.name, "btc_baseline")
        self.assertTrue(resolved.quick_mode)
        self.assertEqual(resolved.config["data"]["symbol"], "BTCUSDT")
        self.assertEqual(resolved.config["data"]["end"], "2024-02-01")
        self.assertEqual(resolved.config["model"]["cv_method"], "walk_forward")
        self.assertEqual(resolved.config["indicators"][0]["kind"], "returns")
        self.assertEqual(resolved.config["indicators"][0]["params"]["periods"], [1, 2])

    def test_load_experiment_config_quick_smoke_caps_context_and_respects_gap_floor(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "quick_smoke.yaml"
            config_path.write_text(
                """
experiment:
  name: btc_quick_smoke
data:
  symbol: BTCUSDT
  interval: 1h
  start: 2024-01-01
  end: 2024-04-01
  context_symbols: [ETHUSDT, SOLUSDT, BNBUSDT]
indicators:
  - kind: returns
features:
  lags: [1, 3, 6]
model:
  cv_method: walk_forward
  n_splits: 4
  train_size: 720
  test_size: 168
  gap: 3
""".strip(),
                encoding="utf-8",
            )

            resolved = load_experiment_config(config_path, quick=True)

        self.assertEqual(resolved.config["data"]["cross_asset_context"]["symbols"], ["ETHUSDT"])
        self.assertEqual(resolved.config["model"]["n_splits"], 1)
        self.assertEqual(resolved.config["model"]["train_size"], 240)
        self.assertEqual(resolved.config["model"]["test_size"], 48)
        self.assertEqual(resolved.config["model"]["gap"], 6)

    def test_load_experiment_config_quick_smoke_caps_context_for_mapping_configs(self):
        config = build_spot_research_config(
            symbol="BTCUSDT",
            interval="1h",
            start="2024-01-01",
            end="2024-04-01",
            indicators=[{"kind": "returns"}],
            context_symbols=["ETHUSDT", "BNBUSDT"],
        )

        resolved = load_experiment_config(config, quick=True)

        self.assertEqual(resolved.config["data"]["cross_asset_context"]["symbols"], ["ETHUSDT"])

    def test_load_experiment_config_derives_legacy_regime_aware_fields_from_orchestration_shape(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "regime_orchestration.yaml"
            config_path.write_text(
                """
experiment:
  name: btc_regime_router
data:
  symbol: BTCUSDT
  interval: 1h
  start: 2024-01-01
  end: 2024-04-01
indicators:
  - kind: returns
regime:
  detectors:
    - name: vol_trend
      type: volatility_trend_hybrid
      primary: true
      warmup_bars: 240
      params:
        n_regimes: 3
    - name: liquidity
      type: liquidity_state
      warmup_bars: 120
  ensemble:
    type: weighted_vote
    primary_detector: vol_trend
feature_adaptation:
  scaling:
    mode: regime_conditioned
    fallback: global
  interaction_budget:
    enabled: true
    max_features: 5
    max_regimes: 4
model:
  type: gbm
  cv_method: cpcv
model_library:
  min_samples_per_regime: 72
  specialists:
    - model_id: trend_model
      estimator: gbm
      compatible_regimes: [trend_up_low_vol]
router:
  type: confidence_weighted
maintenance:
  retraining_policy: structural_only
""".strip(),
                encoding="utf-8",
            )

            resolved = load_experiment_config(config_path)

        self.assertEqual(resolved.config["regime"]["method"], "explicit")
        self.assertEqual(resolved.config["regime"]["feature_lookback"], 240)
        self.assertEqual(resolved.config["regime"]["compatibility_adapter"]["primary_detector"], "vol_trend")
        self.assertTrue(resolved.config["model"]["regime_aware"]["enabled"])
        self.assertEqual(resolved.config["model"]["regime_aware"]["strategy"], "specialist")
        self.assertEqual(resolved.config["model"]["regime_aware"]["min_samples_per_regime"], 72)
        self.assertTrue(resolved.config["model"]["regime_aware"]["regime_interactions"])
        self.assertEqual(resolved.config["model_library"]["specialists"][0]["model_id"], "trend_model")
        self.assertEqual(resolved.config["router"]["type"], "confidence_weighted")

    def test_load_experiment_config_keeps_native_primary_detector_without_legacy_method_alias(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "native_detector.yaml"
            config_path.write_text(
                """
experiment:
  name: btc_native_regime
data:
  symbol: BTCUSDT
  interval: 1h
  start: 2024-01-01
  end: 2024-04-01
indicators:
  - kind: returns
regime:
  detectors:
    - name: trend_native
      type: trend_state
      primary: true
      warmup_bars: 96
      params:
        lower_quantile: 0.25
        upper_quantile: 0.75
""".strip(),
                encoding="utf-8",
            )

            resolved = load_experiment_config(config_path)

        self.assertNotIn("method", resolved.config["regime"])
        self.assertEqual(resolved.config["regime"]["feature_lookback"], 96)
        self.assertEqual(resolved.config["regime"]["detectors"][0]["type"], "trend_state")
        self.assertNotIn("compatibility_adapter", resolved.config["regime"])

    def test_load_experiment_config_rejects_native_primary_detector_with_legacy_method(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "invalid_native_detector.yaml"
            config_path.write_text(
                """
experiment:
  name: invalid_native_regime
data:
  symbol: BTCUSDT
  interval: 1h
  start: 2024-01-01
  end: 2024-04-01
indicators:
  - kind: returns
regime:
  method: explicit
  detectors:
    - name: trend_native
      type: trend_state
      primary: true
""".strip(),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "regime.method cannot be combined with a native primary regime detector"):
                load_experiment_config(config_path)

    def test_load_experiment_config_preserves_orchestration_bundle_automl_config(self):
        config_path = Path(__file__).resolve().parents[1] / "configs" / "btc_regime_bundle_automl.yaml"

        resolved = load_experiment_config(config_path, quick=True)

        self.assertEqual(resolved.name, "btc_regime_bundle_automl")
        self.assertTrue(resolved.quick_mode)
        self.assertEqual(resolved.config["data"]["end"], "2024-02-15")
        self.assertEqual(resolved.config["automl"]["n_trials"], 1)
        bundle_choices = resolved.config["automl"]["search_space"]["orchestration"]["bundle"]["choices"]
        self.assertEqual(len(bundle_choices), 2)
        self.assertEqual(bundle_choices[0]["name"], "trend_native_weighted")
        self.assertEqual(bundle_choices[0]["router"]["type"], "confidence_weighted")
        self.assertEqual(bundle_choices[1]["regime"]["detectors"][0]["type"], "filtered_hmm")


if __name__ == "__main__":
    unittest.main()
