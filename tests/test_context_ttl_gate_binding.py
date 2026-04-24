import unittest

import numpy as np
import pandas as pd

from core import ResearchPipeline


def _make_raw(index):
    close = pd.Series(np.linspace(100.0, 120.0, len(index)), index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close),
            "low": np.minimum(open_, close),
            "close": close,
            "volume": 1_000.0,
        },
        index=index,
    )


class ContextTtlGateBindingTest(unittest.TestCase):
    def test_trade_ready_training_blocks_on_context_ttl_breach_even_with_zero_fill_mode(self):
        index = pd.date_range("2026-02-01", periods=48, freq="1h", tz="UTC")
        raw_data = _make_raw(index)
        X = pd.DataFrame({"feature": np.linspace(-1.0, 1.0, len(index))}, index=index)
        y = pd.Series(np.where(np.arange(len(index)) % 2 == 0, 1, -1), index=index)

        pipeline = ResearchPipeline(
            {
                "features": {"context_missing_policy": {"mode": "zero_fill"}},
                "feature_selection": {"enabled": False},
                "model": {"type": "logistic", "cv_method": "walk_forward", "n_splits": 1, "gap": 0},
                "signals": {"avg_win": 0.02, "avg_loss": 0.02},
                "backtest": {"evaluation_mode": "trade_ready"},
            }
        )
        pipeline.state["raw_data"] = raw_data
        pipeline.state["X"] = X
        pipeline.state["y"] = y
        pipeline.state["labels_aligned"] = pd.DataFrame({"label": y}, index=index)
        pipeline.state["context_ttl_report"] = {
            "futures_context": {
                "promotion_pass": False,
                "unknown_hit_rate": 0.25,
            }
        }

        with self.assertRaisesRegex(RuntimeError, "Context integrity gate failed: futures_context"):
            pipeline.train_models()


if __name__ == "__main__":
    unittest.main()