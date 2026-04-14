import unittest

import numpy as np
import pandas as pd

from core import ResearchPipeline, trend_scanning_labels


def _make_ohlcv(index, drift=12.0, amplitude=3.0, volume_base=1_000.0):
    steps = np.linspace(0.0, 1.0, len(index))
    cycle = np.sin(np.linspace(0.0, 8.0 * np.pi, len(index)))
    close = 100.0 + drift * steps + amplitude * cycle
    open_ = np.roll(close, 1)
    open_[0] = close[0] * 0.998
    high = np.maximum(open_, close) * 1.003
    low = np.minimum(open_, close) * 0.997
    volume = volume_base + 120.0 * (1.0 + np.cos(np.linspace(0.0, 4.0 * np.pi, len(index))))
    quote_volume = close * volume
    trades = 150 + (20 * (1.0 + np.sin(np.linspace(0.0, 6.0 * np.pi, len(index))))).astype(int)

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


def _make_futures_context(index, spot_close):
    mark_close = spot_close * (1.0 + 0.0008 * np.sin(np.linspace(0.0, 5.0 * np.pi, len(index))))
    mark_open = np.roll(mark_close, 1)
    mark_open[0] = mark_close[0]
    mark_price = pd.DataFrame(
        {
            "mark_open": mark_open,
            "mark_high": np.maximum(mark_open, mark_close) * 1.001,
            "mark_low": np.minimum(mark_open, mark_close) * 0.999,
            "mark_close": mark_close,
        },
        index=index,
    )

    premium_close = 0.0002 * np.sin(np.linspace(0.0, 4.0 * np.pi, len(index)))
    premium_open = np.roll(premium_close, 1)
    premium_open[0] = premium_close[0]
    premium_index = pd.DataFrame(
        {
            "premium_open": premium_open,
            "premium_high": np.maximum(premium_open, premium_close) + 0.00005,
            "premium_low": np.minimum(premium_open, premium_close) - 0.00005,
            "premium_close": premium_close,
        },
        index=index,
    )

    funding_index = index[::8]
    funding = pd.DataFrame(
        {
            "funding_rate": 0.0001 * np.sin(np.linspace(0.0, 3.0 * np.pi, len(funding_index))),
            "funding_mark_price": spot_close[::8],
        },
        index=funding_index,
    )

    open_interest = pd.DataFrame(
        {
            "sumOpenInterest": 50_000 + np.linspace(0.0, 4_000.0, len(index)),
            "sumOpenInterestValue": 5_000_000 + np.linspace(0.0, 500_000.0, len(index)),
        },
        index=index,
    )

    taker_flow = pd.DataFrame(
        {
            "buySellRatio": 1.0 + 0.1 * np.sin(np.linspace(0.0, 6.0 * np.pi, len(index))),
            "buyVol": 3_000 + 200 * (1.0 + np.cos(np.linspace(0.0, 5.0 * np.pi, len(index)))),
            "sellVol": 2_600 + 180 * (1.0 + np.sin(np.linspace(0.0, 5.0 * np.pi, len(index)))),
        },
        index=index,
    )

    global_long_short = pd.DataFrame(
        {
            "longShortRatio": 1.1 + 0.05 * np.sin(np.linspace(0.0, 4.0 * np.pi, len(index))),
            "longAccount": 0.53 + 0.03 * np.sin(np.linspace(0.0, 3.0 * np.pi, len(index))),
            "shortAccount": 0.47 - 0.03 * np.sin(np.linspace(0.0, 3.0 * np.pi, len(index))),
        },
        index=index,
    )

    basis = pd.DataFrame(
        {
            "basisRate": 0.0004 * np.sin(np.linspace(0.0, 2.0 * np.pi, len(index))),
            "basis": 10.0 + 2.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, len(index))),
            "futuresPrice": mark_close * 1.0005,
            "indexPrice": spot_close,
        },
        index=index,
    )

    return {
        "mark_price": mark_price,
        "premium_index": premium_index,
        "funding": funding,
        "open_interest": open_interest,
        "taker_flow": taker_flow,
        "global_long_short": global_long_short,
        "basis": basis,
    }


class DerivativesContextPipelineTest(unittest.TestCase):
    def test_trend_scanning_labels_capture_non_zero_events(self):
        index = pd.date_range("2026-01-01", periods=180, freq="1h", tz="UTC")
        prices = pd.Series(100.0 + np.linspace(0.0, 10.0, len(index)) + 2.0 * np.sin(np.linspace(0.0, 6.0 * np.pi, len(index))), index=index)

        labels = trend_scanning_labels(
            prices,
            min_horizon=6,
            max_horizon=24,
            step=3,
            min_t_value=0.75,
            min_return=0.0001,
        )

        self.assertIn("trend_t_value", labels.columns)
        self.assertIn("trend_horizon", labels.columns)
        self.assertGreater(int(labels["label"].abs().sum()), 0)

    def test_derivatives_context_pipeline_case(self):
        index = pd.date_range("2026-02-01", periods=240, freq="1h", tz="UTC")
        raw_data = _make_ohlcv(index)
        cross_asset_context = {
            "ETHUSDT": _make_ohlcv(index, drift=9.0, amplitude=2.0, volume_base=1_300.0),
            "SOLUSDT": _make_ohlcv(index, drift=15.0, amplitude=4.0, volume_base=1_100.0),
            "BNBUSDT": _make_ohlcv(index, drift=6.0, amplitude=1.5, volume_base=900.0),
        }

        pipeline = ResearchPipeline(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "indicators": [],
                "features": {"lags": [1, 3], "frac_diff_d": 0.4, "rolling_window": 20, "context_timeframes": ["4h", "1d"]},
                "regime": {"method": "explicit"},
                "labels": {
                    "kind": "trend_scanning",
                    "min_horizon": 6,
                    "max_horizon": 24,
                    "step": 3,
                    "min_t_value": 0.75,
                    "min_return": 0.0001,
                },
            }
        )
        pipeline.state["raw_data"] = raw_data
        pipeline.state["data"] = raw_data.copy()
        pipeline.state["futures_context"] = _make_futures_context(index, raw_data["close"].to_numpy())
        pipeline.state["cross_asset_context"] = cross_asset_context

        features = pipeline.build_features()
        self.assertIn("fut_funding_rate", features.columns)
        self.assertIn("ctx_ethusdt_ret_1", features.columns)
        self.assertIn("mtf_4h_trend", features.columns)

        regime_result = pipeline.detect_regimes()
        regimes = regime_result["regimes"]
        self.assertIsNone(regimes)

        labels = pipeline.build_labels()
        self.assertIn("trend_t_value", labels.columns)
        self.assertGreater(int(labels["label"].abs().sum()), 0)

        aligned = pipeline.align_data()
        self.assertGreater(len(aligned["X"]), 50)
        self.assertNotIn("regime", aligned["X"].columns)
        self.assertNotIn("close_fracdiff", aligned["X"].columns)
        self.assertNotIn("close_fracdiff_lag1", aligned["X"].columns)
        self.assertNotIn("close_fracdiff_lag3", aligned["X"].columns)


if __name__ == "__main__":
    unittest.main()