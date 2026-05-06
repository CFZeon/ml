import unittest
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
        }

    def generate_signals(self):
        return {"signals": pd.Series([1, 0, -1] * 20, index=self.index[:60])}

    def run_backtest(self):
        return {"engine": "vectorbt", "sharpe_ratio": 3.4, "net_profit_pct": 0.18, "max_drawdown": -0.04, "total_trades": 8}


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


if __name__ == "__main__":
    unittest.main()