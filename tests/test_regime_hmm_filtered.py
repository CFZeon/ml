import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from core import ResearchPipeline, build_regime_observation_contracts
from core.regimes.detectors import FilteredHMMDetector, build_regime_detector
from core.regimes.online_state import replay_regime_detector_trace


def _make_hmm_observations(index):
    half = len(index) // 2
    state_a = pd.DataFrame(
        {
            "feature_1": np.linspace(-2.0, -0.5, half),
            "feature_2": np.linspace(0.0, 0.5, half),
            "feature_3": np.linspace(-1.5, -0.8, half),
        },
        index=index[:half],
    )
    state_b = pd.DataFrame(
        {
            "feature_1": np.linspace(0.8, 2.0, len(index) - half),
            "feature_2": np.linspace(1.0, 2.0, len(index) - half),
            "feature_3": np.linspace(0.6, 1.4, len(index) - half),
        },
        index=index[half:],
    )
    return pd.concat([state_a, state_b]).sort_index()


def _make_ohlcv(index):
    observations = _make_hmm_observations(index)
    close = 100.0 + observations["feature_1"].cumsum()
    open_ = close.shift(1).fillna(close.iloc[0])
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    volume = 1_000.0 + 100.0 * (observations["feature_2"] - observations["feature_2"].min() + 1.0)
    quote_volume = close * volume
    trades = (150 + 10.0 * observations["feature_3"]).round().astype(int)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": quote_volume,
            "trades": trades,
        },
        index=index,
    )


class FilteredHMMRuntimeTest(unittest.TestCase):
    def test_build_regime_detector_creates_filtered_hmm_detector(self):
        detector = build_regime_detector({"type": "filtered_hmm"}, config={"column_name": "regime"})
        self.assertIsInstance(detector, FilteredHMMDetector)

    def test_filtered_hmm_replay_is_prefix_invariant_for_frozen_fit_prefix(self):
        index = pd.date_range("2026-08-01", periods=96, freq="1h", tz="UTC")
        observations = _make_hmm_observations(index)
        fit_prefix = observations.iloc[:48]
        config = {"column_name": "regime", "n_regimes": 2, "random_state": 7, "n_iter": 40}
        spec = {"name": "hmm_native", "type": "filtered_hmm"}

        prefix_only = replay_regime_detector_trace(
            observations.iloc[:72],
            detector=build_regime_detector(spec, config=config),
            source_map={column: "instrument_state" for column in observations.columns},
            provenance={"source_counts": {"instrument_state": len(observations.columns)}},
            fit_observations=fit_prefix,
        )
        extended = replay_regime_detector_trace(
            observations,
            detector=build_regime_detector(spec, config=config),
            source_map={column: "instrument_state" for column in observations.columns},
            provenance={"source_counts": {"instrument_state": len(observations.columns)}},
            fit_observations=fit_prefix,
        )

        pd.testing.assert_frame_equal(prefix_only["state_frame"], extended["state_frame"].iloc[:72])
        self.assertEqual(prefix_only["detector_manifests"][0].to_dict(), extended["detector_manifests"][0].to_dict())
        manifest_metadata = prefix_only["detector_manifests"][0].metadata
        self.assertEqual(manifest_metadata["semantic_schema_version"], "filtered_hmm.semantic.v1")
        self.assertTrue(manifest_metadata["semantic_state_map"])
        self.assertEqual(manifest_metadata["canonical_schema_version"], "filtered_hmm.canonical.v1")
        self.assertTrue(manifest_metadata["canonical_state_map"])
        self.assertIn("taxonomy_stability_report", manifest_metadata)
        self.assertTrue(prefix_only["state_frame"]["regime"].dropna().astype(str).str.startswith("hmm__").all())
        self.assertTrue(
            prefix_only["state_frame"]["canonical_regime_id"].dropna().astype(str).str.startswith("filtered_hmm__").all()
        )

    def test_filtered_hmm_can_preserve_reference_canonical_ids(self):
        index = pd.date_range("2026-08-01", periods=96, freq="1h", tz="UTC")
        observations = _make_hmm_observations(index)
        spec = {"name": "hmm_native", "type": "filtered_hmm"}
        config = {"column_name": "regime", "n_regimes": 2, "random_state": 7, "n_iter": 40}

        deployed = build_regime_detector(spec, config=config)
        deployed.fit(observations.iloc[:64])
        deployed_manifest = deployed.manifest().to_dict()
        reference_map = {
            state_index: {
                **dict(payload or {}),
                "canonical_regime_id": f"deployed_family_{state_index}",
            }
            for state_index, payload in dict(deployed_manifest["metadata"]["canonical_state_map"]).items()
        }

        remapped = build_regime_detector(
            spec,
            config={
                **config,
                "reference_canonical_state_map": reference_map,
            },
        )
        remapped.fit(observations.iloc[:64])
        remapped_manifest = remapped.manifest().to_dict()
        remapped_ids = {
            state_index: payload["canonical_regime_id"]
            for state_index, payload in dict(remapped_manifest["metadata"]["canonical_state_map"]).items()
        }

        self.assertEqual(remapped_ids, {state_index: payload["canonical_regime_id"] for state_index, payload in reference_map.items()})
        self.assertTrue(remapped_manifest["metadata"]["taxonomy_stability_report"]["reference_available"])
        self.assertEqual(remapped_manifest["metadata"]["taxonomy_stability_report"]["compatibility_break_count"], 0)

    def test_filtered_hmm_update_does_not_call_smoothed_or_batch_decoding_methods(self):
        from hmmlearn.hmm import GaussianHMM

        index = pd.date_range("2026-08-01", periods=64, freq="1h", tz="UTC")
        observations = _make_hmm_observations(index)
        detector = build_regime_detector(
            {"name": "hmm_native", "type": "filtered_hmm"},
            config={"column_name": "regime", "n_regimes": 2, "random_state": 11, "n_iter": 40},
        )
        detector.fit(observations.iloc[:32])
        contracts = build_regime_observation_contracts(
            observations.iloc[32:36],
            source_map={column: "instrument_state" for column in observations.columns},
        )

        state = detector.initialize(observations.iloc[32:36])
        with (
            patch.object(GaussianHMM, "predict", side_effect=AssertionError("predict should not be called during filtered replay")),
            patch.object(GaussianHMM, "predict_proba", side_effect=AssertionError("predict_proba should not be called during filtered replay")),
            patch.object(GaussianHMM, "decode", side_effect=AssertionError("decode should not be called during filtered replay")),
            patch.object(GaussianHMM, "score_samples", side_effect=AssertionError("score_samples should not be called during filtered replay")),
        ):
            for observation in contracts:
                state, contract = detector.update(state, observation)

        self.assertIn("prob_state_0", contract.detector_outputs)
        self.assertIn("prob_state_1", contract.detector_outputs)
        self.assertIn("regime_confidence", contract.detector_outputs)
        self.assertIn("latent_regime_id", contract.detector_outputs)
        self.assertIn("semantic_regime", contract.detector_outputs)
        self.assertIn("canonical_regime_id", contract.detector_outputs)
        self.assertNotIn("smoothed_probability", contract.detector_outputs)
        self.assertEqual(contract.metadata["posterior_mode"], "filtered")
        self.assertEqual(contract.metadata["semantic_schema_version"], "filtered_hmm.semantic.v1")
        self.assertEqual(contract.metadata["canonical_schema_version"], "filtered_hmm.canonical.v1")
        self.assertTrue(str(contract.metadata["semantic_label"]).startswith("hmm__"))
        self.assertTrue(str(contract.metadata["canonical_regime_id"]).startswith("filtered_hmm__"))

    def test_pipeline_detect_regimes_routes_legacy_hmm_through_filtered_replay(self):
        index = pd.date_range("2026-08-01", periods=120, freq="1h", tz="UTC")
        raw_data = _make_ohlcv(index)
        pipeline = ResearchPipeline(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "indicators": [],
                "regime": {"method": "hmm", "n_regimes": 2, "random_state": 5, "n_iter": 40},
            }
        )
        pipeline.state["raw_data"] = raw_data
        pipeline.state["data"] = raw_data.copy()

        result = pipeline.detect_regimes()

        self.assertEqual(pipeline.state["regime_detection"]["method"], "hmm")
        self.assertEqual(pipeline.state["regime_detection"]["detector_type"], "filtered_hmm")
        self.assertEqual(
            pipeline.state["regime_detection"]["detector_manifests"][0].metadata["posterior_mode"],
            "filtered",
        )
        self.assertIn("regime_confidence", result["regime_state_frame"].columns)
        self.assertIn("latent_regime_id", result["regime_state_frame"].columns)
        self.assertIn("canonical_regime_id", result["regime_state_frame"].columns)
        self.assertTrue(any(column.startswith("prob_state_") for column in result["regime_state_frame"].columns))
        self.assertFalse(any("smoothed" in column for column in result["regime_state_frame"].columns))
        self.assertTrue(result["regime_state_frame"]["regime"].dropna().astype(str).str.startswith("hmm__").all())
        self.assertTrue(
            pipeline.state["regime_detection"]["detector_manifests"][0].metadata["semantic_state_map"]
        )
        self.assertTrue(
            pipeline.state["regime_detection"]["detector_manifests"][0].metadata["canonical_state_map"]
        )


if __name__ == "__main__":
    unittest.main()