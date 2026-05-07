import unittest

import pandas as pd

from core.regime import (
    RegimeObservationContract,
    RegimeStateContract,
    build_regime_observation_contracts,
    build_regime_state_contracts,
    build_regime_trace_summary,
    build_regime_transition_contracts,
    summarize_regime_detection_result,
)


class RegimeFacadeCompatibilityTest(unittest.TestCase):
    def test_regime_facade_helpers_project_current_outputs_into_contracts(self):
        index = pd.date_range("2026-05-01", periods=4, freq="1h", tz="UTC")
        observations = pd.DataFrame(
            {
                "trend_20": [0.1, 0.2, 0.15, -0.05],
                "vol_20": [1.0, 1.1, 1.2, 1.3],
            },
            index=index,
        )
        regimes = pd.DataFrame(
            {
                "regime": ["bull", "bull", "risk_off", "risk_off"],
                "prob_bull": [0.7, 0.8, 0.2, 0.1],
                "prob_risk_off": [0.3, 0.2, 0.8, 0.9],
                "regime_confidence": [0.7, 0.8, 0.8, 0.9],
            },
            index=index,
        )

        observation_contracts = build_regime_observation_contracts(
            observations,
            source_map={"trend_20": "instrument_state", "vol_20": "instrument_state"},
        )
        state_contracts = build_regime_state_contracts(regimes)
        transition_contracts = build_regime_transition_contracts(regimes)
        summary = build_regime_trace_summary(
            regimes,
            mode="walk_forward",
            observation_columns=list(observations.columns),
            provenance={"trend_20": "instrument_state"},
        )
        detection_summary = summarize_regime_detection_result(
            {
                "mode": "walk_forward",
                "regime_observations": observations,
                "regimes": regimes,
                "provenance": {"trend_20": "instrument_state"},
            }
        )

        self.assertEqual(len(observation_contracts), len(observations))
        self.assertIsInstance(observation_contracts[0], RegimeObservationContract)
        self.assertEqual(observation_contracts[0].source_map["trend_20"], "instrument_state")
        self.assertEqual(observation_contracts[0].values["trend_20"], observations.iloc[0]["trend_20"])

        self.assertEqual(len(state_contracts), len(regimes))
        self.assertIsInstance(state_contracts[0], RegimeStateContract)
        self.assertEqual(state_contracts[0].label, "bull")
        self.assertAlmostEqual(state_contracts[2].probabilities["prob_risk_off"], 0.8)

        self.assertEqual(len(transition_contracts), 1)
        self.assertEqual(transition_contracts[0].from_label, "bull")
        self.assertEqual(transition_contracts[0].to_label, "risk_off")

        self.assertEqual(summary.transition_count, 1)
        self.assertEqual(summary.dominant_label, "bull")
        self.assertIn("trend_20", summary.observation_columns)
        self.assertEqual(detection_summary.mode, "walk_forward")
        self.assertEqual(detection_summary.label_distribution["bull"], 2)
