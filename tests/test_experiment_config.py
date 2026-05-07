import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
