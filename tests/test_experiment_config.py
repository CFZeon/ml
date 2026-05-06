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


if __name__ == "__main__":
    unittest.main()
