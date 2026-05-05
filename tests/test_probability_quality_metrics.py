import unittest

import numpy as np
import pandas as pd

from core.models import evaluate_model


class _StaticProbabilityModel:
    def __init__(self, predictions, probabilities, classes):
        self._predictions = np.asarray(predictions)
        self._probabilities = np.asarray(probabilities, dtype=float)
        self.classes_ = np.asarray(classes)

    def predict(self, X):
        return self._predictions

    def predict_proba(self, X):
        return self._probabilities


class ProbabilityQualityMetricsTest(unittest.TestCase):
    def test_evaluate_model_reports_brier_decomposition(self):
        index = pd.date_range("2026-05-01", periods=4, freq="1h", tz="UTC")
        X = pd.DataFrame({"feature": [0.1, 0.2, 0.3, 0.4]}, index=index)
        y = pd.Series([1, -1, 1, -1], index=index)
        model = _StaticProbabilityModel(
            predictions=[1, -1, 1, -1],
            probabilities=[
                [0.15, 0.05, 0.80],
                [0.70, 0.10, 0.20],
                [0.25, 0.10, 0.65],
                [0.60, 0.10, 0.30],
            ],
            classes=[-1, 0, 1],
        )

        metrics = evaluate_model(model, X, y)

        self.assertEqual(int(metrics["probability_observation_count"]), 4)
        decomposition = metrics["brier_decomposition"]
        self.assertIn("reliability", decomposition)
        self.assertIn("resolution", decomposition)
        self.assertIn("uncertainty", decomposition)
        self.assertIn("residual", decomposition)
        reconstructed = (
            float(decomposition["reliability"])
            - float(decomposition["resolution"])
            + float(decomposition["uncertainty"])
            + float(decomposition["residual"])
        )
        self.assertAlmostEqual(reconstructed, float(metrics["brier_score"]), places=3)


if __name__ == "__main__":
    unittest.main()