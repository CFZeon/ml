import types
import unittest

import numpy as np
import pandas as pd

from core import ResearchPipeline
from core.pipeline import _build_fold_local_regime_frame
from core.regime import RegimeFeatureSet, build_regime_ablation_report


def _make_ohlcv(index, *, drift=10.0, amplitude=2.0, volume_base=1_000.0):
    steps = np.linspace(0.0, 1.0, len(index))
    cycle = np.sin(np.linspace(0.0, 6.0 * np.pi, len(index)))
    close = 100.0 + drift * steps + amplitude * cycle
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    volume = volume_base + 100.0 * (1.0 + np.cos(np.linspace(0.0, 4.0 * np.pi, len(index))))
    quote_volume = close * volume
    trades = 120 + (15.0 * (1.0 + np.sin(np.linspace(0.0, 4.0 * np.pi, len(index))))).astype(int)
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
    mark_close = spot_close * (1.0 + 0.001 * np.sin(np.linspace(0.0, 4.0 * np.pi, len(index))))
    premium_close = 0.0003 * np.sin(np.linspace(0.0, 5.0 * np.pi, len(index)))
    return {
        "mark_price": pd.DataFrame(
            {
                "mark_open": np.roll(mark_close, 1),
                "mark_high": mark_close * 1.001,
                "mark_low": mark_close * 0.999,
                "mark_close": mark_close,
            },
            index=index,
        ),
        "premium_index": pd.DataFrame(
            {
                "premium_open": np.roll(premium_close, 1),
                "premium_high": premium_close + 0.00005,
                "premium_low": premium_close - 0.00005,
                "premium_close": premium_close,
            },
            index=index,
        ),
        "funding": pd.DataFrame(
            {
                "funding_rate": 0.0001 * np.cos(np.linspace(0.0, 3.0 * np.pi, len(index))),
                "funding_mark_price": spot_close,
            },
            index=index,
        ),
    }


class RegimeLayerAblationTest(unittest.TestCase):
    def test_context_aware_regime_preview_reports_distinct_provenance(self):
        index = pd.date_range("2026-08-01", periods=220, freq="1h", tz="UTC")
        raw_data = _make_ohlcv(index)
        pipeline = ResearchPipeline(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "indicators": [],
                "features": {"rolling_window": 20, "context_timeframes": ["4h"]},
                "regime": {"method": "explicit"},
            }
        )
        pipeline.state["raw_data"] = raw_data
        pipeline.state["data"] = raw_data.copy()
        pipeline.state["futures_context"] = _make_futures_context(index, raw_data["close"].to_numpy())
        pipeline.state["cross_asset_context"] = {
            "ETHUSDT": _make_ohlcv(index, drift=7.0, amplitude=1.5, volume_base=1_200.0),
            "SOLUSDT": _make_ohlcv(index, drift=14.0, amplitude=3.0, volume_base=900.0),
        }
        pipeline.state["reference_overlay_data"] = pd.DataFrame(
            {
                "reference_close": raw_data["close"] * (1.0 + 0.0015 * np.cos(np.linspace(0.0, 5.0 * np.pi, len(index)))),
                "reference_volume": raw_data["volume"] * 1.1,
                "breadth": np.sin(np.linspace(0.0, 3.0 * np.pi, len(index))),
            },
            index=index,
        )

        result = pipeline.detect_regimes()

        provenance = result["provenance"]
        self.assertGreater(provenance["source_counts"]["instrument_state"], 0)
        self.assertGreater(provenance["source_counts"]["market_state"], 0)
        self.assertGreater(provenance["source_counts"]["cross_asset_state"], 0)

        ablation = result["ablation"]
        self.assertTrue(ablation["contextual_sources_present"])
        self.assertGreater(ablation["full_provenance"]["contextual_share"], 0.0)
        self.assertEqual(ablation["endogenous_provenance"]["contextual_share"], 0.0)
        self.assertGreater(
            ablation["full_provenance"]["total_columns"],
            ablation["endogenous_provenance"]["total_columns"],
        )

    def test_regime_stability_gate_rejects_contextual_layer_that_increases_switching(self):
        index = pd.date_range("2026-08-01", periods=120, freq="1h", tz="UTC")
        baseline_step = np.r_[np.full(60, -1.0), np.full(60, 1.0)]
        noisy_toggle = np.where(np.arange(len(index)) % 2 == 0, -1.0, 1.0)
        frame = pd.DataFrame(
            {
                "trend_20": baseline_step,
                "vol_20": np.r_[np.full(60, 0.01), np.full(60, 0.05)],
                "liquidity_20": np.r_[np.full(60, -0.5), np.full(60, 0.5)],
                "ref_trend_overlay": noisy_toggle,
                "ref_vol_overlay": noisy_toggle,
                "ref_liquidity_overlay": noisy_toggle,
            },
            index=index,
        )
        feature_set = RegimeFeatureSet(
            frame=frame,
            source_map={
                "trend_20": "instrument_state",
                "vol_20": "instrument_state",
                "liquidity_20": "instrument_state",
                "ref_trend_overlay": "market_state",
                "ref_vol_overlay": "market_state",
                "ref_liquidity_overlay": "market_state",
            },
        )

        report = build_regime_ablation_report(
            feature_set,
            method="explicit",
            config={"require_stability_improvement": True},
        )

        self.assertTrue(report["stability_gate"]["required"])
        self.assertFalse(report["stability_gate"]["passed"])
        self.assertIsNotNone(report["stability_improvement"])
        self.assertLess(report["stability_improvement"], 0.0)
        self.assertLess(report["full_stability"]["persistence"], report["endogenous_stability"]["persistence"])

    def test_fold_local_builder_accepts_regime_feature_set_output(self):
        raw_data = _make_ohlcv(pd.date_range("2026-08-01", periods=200, freq="1h", tz="UTC"))
        builder_windows = []

        def builder(scoped_pipeline):
            scoped = scoped_pipeline.require("data")
            builder_windows.append(scoped.index)
            frame = pd.DataFrame(
                {
                    "trend_20": scoped["close"].pct_change(4).fillna(0.0),
                    "vol_20": scoped["close"].pct_change().rolling(5, min_periods=1).std().fillna(0.0),
                    "liquidity_20": np.log1p(scoped["quote_volume"]).rolling(5, min_periods=1).mean().fillna(0.0),
                    "ref_trend_overlay": np.sin(np.linspace(0.0, 4.0 * np.pi, len(scoped))),
                },
                index=scoped.index,
            )
            return RegimeFeatureSet(
                frame=frame,
                source_map={
                    "trend_20": "instrument_state",
                    "vol_20": "instrument_state",
                    "liquidity_20": "instrument_state",
                    "ref_trend_overlay": "market_state",
                },
            )

        fake_pipeline = types.SimpleNamespace(
            state={"raw_data": raw_data},
            section=lambda key: (
                {"enabled": True, "method": "explicit", "builder": builder}
                if key == "regime"
                else {}
            ),
            require=lambda key: raw_data,
        )

        fold_index = raw_data.index[90:160]
        fit_index = raw_data.index[90:130]
        frame, feature_blocks = _build_fold_local_regime_frame(fake_pipeline, fold_index, fit_index=fit_index)

        self.assertEqual(len(builder_windows), 1)
        self.assertLess(builder_windows[0][-1], raw_data.index[-1])
        self.assertEqual(list(frame.index), list(fold_index))
        self.assertTrue({"trend_regime", "volatility_regime", "liquidity_regime", "regime"}.issubset(frame.columns))
        self.assertEqual(set(feature_blocks.values()), {"regime"})

        details = fake_pipeline.state["_last_regime_details"]
        self.assertEqual(details["provenance"]["source_counts"]["market_state"], 1)
        self.assertTrue(details["ablation"]["contextual_sources_present"])


if __name__ == "__main__":
    unittest.main()