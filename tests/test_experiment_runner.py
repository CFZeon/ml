import unittest
from io import StringIO
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments import run_experiment


def _make_index(length: int = 72) -> pd.DatetimeIndex:
    return pd.date_range("2026-01-01", periods=length, freq="1h", tz="UTC")


def _make_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    close = 100.0 + np.linspace(0.0, 8.0, len(index))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": 1_000.0,
            "quote_volume": close * 1_000.0,
            "trades": 100,
        },
        index=index,
    )


class _FakeIndicatorRun:
    def __init__(self, frame):
        self.frame = frame
        self.results = [type("Result", (), {"kind": "returns"})(), type("Result", (), {"kind": "volatility"})()]


class _FakePipeline:
    def __init__(self, config):
        self.config = config
        self.index = _make_index()
        self.frame = _make_frame(self.index)

    def fetch_data(self):
        return self.frame

    def run_indicators(self):
        enriched = self.frame.assign(returns_1=self.frame["close"].pct_change(), volatility_24=0.01)
        return _FakeIndicatorRun(enriched)

    def build_features(self):
        return pd.DataFrame({"returns_1": np.linspace(-0.01, 0.01, len(self.index)), "volatility_24": 0.01}, index=self.index)

    def check_stationarity(self):
        return {
            "close": {"stationary": False, "p_value": 0.2},
            "close_fracdiff": {"stationary": True, "p_value": 0.01},
            "feature_screening": {"summary": {"screened_feature_count": 2, "total_features": 2, "transformed_features": 1, "dropped_features": 0}},
        }

    def detect_regimes(self):
        return {"regimes": pd.DataFrame({"regime": [0, 1, 0]}, index=self.index[:3])}

    def build_labels(self):
        return pd.DataFrame({"label": [1, 0, -1], "barrier": ["pt", "vb", "sl"]}, index=self.index[:3])

    def align_data(self):
        X = pd.DataFrame({"returns_1": np.linspace(-0.01, 0.01, 60), "volatility_24": 0.01}, index=self.index[:60])
        y = pd.Series([1, 0, -1] * 20, index=self.index[:60])
        return {"X": X, "y": y}

    def select_features(self):
        return type("Selection", (), {"report": {"max_features": 8, "min_mi_threshold": 0.0}})()

    def compute_sample_weights(self):
        return pd.Series(1.0, index=self.index[:60])

    def train_models(self):
        return {
            "validation": {"method": "cpcv", "split_count": 3, "n_blocks": 4, "test_blocks": 2, "embargo_bars": 2},
            "avg_accuracy": 0.74,
            "avg_f1_macro": 0.69,
            "last_selected_columns": ["returns_1", "volatility_24"],
            "feature_adaptation": {
                "enabled": True,
                "applied_in_any_fold": True,
                "deferred_runtime": False,
                "requested_scaling_mode": "regime_conditioned",
                "requested_selection_mode": "per_regime_mask",
                "last_manifest": {"adapter_type": "composite_feature_adaptation"},
            },
        }

    def generate_signals(self):
        return {"signals": pd.Series([1, 0, -1] * 20, index=self.index[:60])}

    def run_backtest(self):
        return {"engine": "vectorbt", "sharpe_ratio": 3.4, "net_profit_pct": 0.18, "max_drawdown": -0.04, "total_trades": 8}


class _FakeAutoMLPipeline(_FakePipeline):
    def __init__(self, config):
        super().__init__(config)
        self.state = {}

    def run_automl(self):
        return {
            "objective": "sharpe_ratio",
            "selection_metric": "sharpe_ratio",
            "selection_mode": "maximize",
            "trial_count": 2,
            "best_value": 0.84,
            "best_overrides": {
                "experiment": {"bundle_name": "trend_native_weighted"},
                "model": {"type": "gbm"},
            },
            "best_bundle_lineage": {
                "bundle_name": "trend_native_weighted",
                "bundle_description": "Native trend-state detector with weighted routing.",
                "primary_detector": {"name": "trend_native_primary", "type": "trend_state", "primary": True},
                "specialist_model_ids": ["trend_model", "breakout_model"],
                "router": {"type": "confidence_weighted"},
            },
        }

    def refit_selected_candidate(self, automl):
        features = self.build_features()
        stationarity = self.check_stationarity()
        regimes = self.detect_regimes()["regimes"]
        labels = self.build_labels()
        aligned = self.align_data()
        self.state = {
            "features": features,
            "stationarity": stationarity,
            "regimes": regimes,
            "labels": labels,
            "X": aligned["X"],
            "y": aligned["y"],
            "labels_aligned": labels,
        }
        return {
            "pipeline": self,
            "training": self.train_models(),
            "signals": self.generate_signals(),
            "backtest": {
                "engine": "vectorbt",
                "sharpe_ratio": 1.6,
                "net_profit_pct": 0.11,
                "max_drawdown": -0.05,
                "total_trades": 12,
                "router_stability_report": {
                    "enabled": True,
                    "applicable": True,
                    "switch_count": 3,
                    "switch_rate": 0.25,
                    "blocked_switch_count": 1,
                    "configured_control_count": 3,
                },
            },
        }


class ExperimentRunnerTest(unittest.TestCase):
    @patch("experiments.runner.ResearchPipeline", _FakePipeline)
    def test_run_experiment_returns_artifacts_and_warnings(self):
        result = run_experiment(
            {
                "experiment": {"name": "runner_smoke"},
                "data": {"symbol": "BTCUSDT", "interval": "1h", "start": "2024-01-01", "end": "2024-02-01"},
                "indicators": [{"kind": "returns"}],
            },
            quiet=True,
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.artifacts["training"]["last_selected_columns"], ["returns_1", "volatility_24"])
        self.assertTrue(any("Too-good-to-be-true" in warning for warning in result.warnings))

    @patch("experiments.runner.ResearchPipeline", _FakePipeline)
    def test_run_experiment_prints_feature_adaptation_summary(self):
        with patch("sys.stdout", new_callable=StringIO) as stdout:
            run_experiment(
                {
                    "experiment": {"name": "runner_feature_adaptation"},
                    "data": {"symbol": "BTCUSDT", "interval": "1h", "start": "2024-01-01", "end": "2024-02-01"},
                    "indicators": [{"kind": "returns"}],
                },
                quiet=False,
            )

        output = stdout.getvalue()
        self.assertIn("Feature adapt", output)
        self.assertIn("Adapt policy", output)

    @patch("experiments.runner.ResearchPipeline", _FakeAutoMLPipeline)
    def test_run_experiment_prints_bundle_and_router_summary_for_automl(self):
        with patch("sys.stdout", new_callable=StringIO) as stdout:
            run_experiment(
                {
                    "experiment": {"name": "runner_bundle_automl"},
                    "data": {"symbol": "BTCUSDT", "interval": "1h", "start": "2024-01-01", "end": "2024-02-01"},
                    "indicators": [{"kind": "returns"}],
                    "automl": {"enabled": True},
                },
                quiet=False,
            )

        output = stdout.getvalue()
        self.assertIn("Bundle        : trend_native_weighted", output)
        self.assertIn("Bundle path   : detector=trend_native_primary:trend_state", output)
        self.assertIn("Router        : switches=3", output)
        self.assertIn("switch_rate=25.00%", output)


if __name__ == "__main__":
    unittest.main()