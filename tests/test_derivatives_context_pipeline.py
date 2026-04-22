import unittest

import numpy as np
import pandas as pd

from core import ResearchPipeline, build_reference_validation_bundle, select_features, train_model, trend_scanning_labels
from core.context import build_futures_context_feature_block
from core.models import compute_feature_family_diagnostics, summarize_feature_family_diagnostics


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
    def _make_context_pipeline(self, *, feature_selection=None, model=None, labels=None):
        return ResearchPipeline(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "indicators": [],
                "features": {"lags": [1, 3], "frac_diff_d": 0.4, "rolling_window": 20, "context_timeframes": ["4h", "1d"]},
                "feature_selection": feature_selection or {"enabled": True, "max_features": 48, "min_mi_threshold": 0.0},
                "regime": {"method": "explicit"},
                "labels": labels or {"kind": "fixed_horizon", "horizon": 6, "threshold": 0.0},
                "model": model or {"type": "logistic", "cv_method": "walk_forward", "n_splits": 1, "gap": 0, "validation_fraction": 0.2, "meta_n_splits": 2},
                "signals": {"avg_win": 0.02, "avg_loss": 0.02, "threshold": 0.0, "edge_threshold": 0.0},
                "backtest": {"use_open_execution": False, "signal_delay_bars": 1},
            }
        )

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
        self.assertIsInstance(regimes, pd.DataFrame)
        self.assertTrue({"trend_regime", "volatility_regime", "liquidity_regime", "regime"}.issubset(regimes.columns))

        labels = pipeline.build_labels()
        self.assertIn("trend_t_value", labels.columns)
        self.assertGreater(int(labels["label"].abs().sum()), 0)

        aligned = pipeline.align_data()
        self.assertGreater(len(aligned["X"]), 50)
        self.assertNotIn("regime", aligned["X"].columns)
        self.assertNotIn("close_fracdiff", aligned["X"].columns)
        self.assertNotIn("close_fracdiff_lag1", aligned["X"].columns)
        self.assertNotIn("close_fracdiff_lag3", aligned["X"].columns)

    def test_futures_context_ttl_blanks_stale_state_but_keeps_rows(self):
        index = pd.date_range("2026-02-01", periods=12, freq="1h", tz="UTC")
        raw_data = _make_ohlcv(index)
        futures_context = _make_futures_context(index, raw_data["close"].to_numpy())
        futures_context["mark_price"] = futures_context["mark_price"].iloc[[0]].copy()

        block = build_futures_context_feature_block(
            raw_data,
            futures_context,
            rolling_window=4,
            ttl_config={"mark_price": "2h", "premium_index": "2h", "funding": "12h", "recent": "4h", "max_stale_rate": 0.1},
        )

        self.assertIn("fut_context_stale_any", block.frame.columns)
        self.assertEqual(float(block.frame.loc[index[3], "fut_context_stale_any"]), 1.0)
        self.assertAlmostEqual(float(block.frame.loc[index[3], "fut_mark_spread_pct"]), 0.0, places=6)
        self.assertGreater(block.metadata["ttl_report"]["stale_hit_count"], 0)
        self.assertFalse(block.metadata["ttl_report"]["promotion_pass"])

    def test_pipeline_surfaces_context_ttl_breach_in_operational_monitoring(self):
        index = pd.date_range("2026-02-01", periods=240, freq="1h", tz="UTC")
        raw_data = _make_ohlcv(index)
        futures_context = _make_futures_context(index, raw_data["close"].to_numpy())
        futures_context["mark_price"] = futures_context["mark_price"].iloc[[0]].copy()
        cross_asset_context = {
            "ETHUSDT": _make_ohlcv(index, drift=9.0, amplitude=2.0, volume_base=1_300.0),
            "SOLUSDT": _make_ohlcv(index, drift=15.0, amplitude=4.0, volume_base=1_100.0),
            "BNBUSDT": _make_ohlcv(index, drift=6.0, amplitude=1.5, volume_base=900.0),
        }

        pipeline = self._make_context_pipeline(
            labels={
                "kind": "trend_scanning",
                "min_horizon": 6,
                "max_horizon": 24,
                "step": 3,
                "min_t_value": 0.75,
                "min_return": 0.0001,
            }
        )
        pipeline.config["features"]["futures_context_ttl"] = {
            "mark_price": "2h",
            "premium_index": "2h",
            "funding": "12h",
            "recent": "4h",
            "max_stale_rate": 0.05,
        }
        pipeline.state["raw_data"] = raw_data
        pipeline.state["data"] = raw_data.copy()
        pipeline.state["futures_context"] = futures_context
        pipeline.state["cross_asset_context"] = cross_asset_context
        pipeline.build_features()
        pipeline.detect_regimes()
        pipeline.build_labels()
        pipeline.align_data()
        training = pipeline.train_models()

        self.assertIn("context_ttl", training["operational_monitoring"])
        self.assertFalse(training["operational_monitoring"]["healthy"])
        self.assertIn("context_ttl_breached", training["operational_monitoring"]["reasons"])

    def test_mark_valuation_strict_policy_rejects_missing_leading_prices(self):
        index = pd.date_range("2026-02-01", periods=12, freq="1h", tz="UTC")
        raw_data = _make_ohlcv(index)
        futures_context = _make_futures_context(index, raw_data["close"].to_numpy())
        futures_context["mark_price"] = futures_context["mark_price"].copy()
        futures_context["mark_price"].iloc[0, futures_context["mark_price"].columns.get_loc("mark_close")] = np.nan

        pipeline = ResearchPipeline(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h", "market": "um_futures"},
                "backtest": {
                    "engine": "pandas",
                    "use_open_execution": False,
                    "valuation_price": "mark",
                    "valuation_price_policy": "strict",
                    "execution_price_policy": "strict",
                },
                "indicators": [],
                "features": {},
                "labels": {"kind": "fixed_horizon", "horizon": 3, "threshold": 0.0},
            }
        )
        pipeline.state["raw_data"] = raw_data
        pipeline.state["futures_context"] = futures_context
        pipeline.state["signals"] = {
            "continuous_signals": pd.Series([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=index),
            "signals": pd.Series([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=index),
        }

        with self.assertRaisesRegex(ValueError, "valuation price policy"):
            pipeline.run_backtest()

    def test_pipeline_futures_backtest_surfaces_margin_metrics(self):
        index = pd.date_range("2026-02-01", periods=6, freq="1h", tz="UTC")
        raw_data = _make_ohlcv(index)
        futures_context = _make_futures_context(index, raw_data["close"].to_numpy())
        futures_context["mark_price"] = futures_context["mark_price"].copy()
        futures_context["mark_price"]["mark_close"] = [100.0, 100.0, 60.0, 60.0, 60.0, 60.0]

        pipeline = ResearchPipeline(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h", "market": "um_futures"},
                "backtest": {
                    "engine": "pandas",
                    "use_open_execution": False,
                    "valuation_price": "mark",
                    "valuation_price_policy": "strict",
                    "execution_price_policy": "strict",
                    "leverage": 3.0,
                    "allow_short": True,
                    "futures_account": {
                        "enabled": True,
                        "margin_mode": "isolated",
                        "warning_margin_ratio": 0.8,
                        "liquidation_fee_rate": 0.01,
                        "leverage_brackets_data": {
                            "symbol": "BTCUSDT",
                            "brackets": [
                                {
                                    "bracket": 1,
                                    "initial_leverage": 3.0,
                                    "notional_floor": 0.0,
                                    "notional_cap": 1_000_000.0,
                                    "maint_margin_ratio": 0.05,
                                    "cum": 0.0,
                                }
                            ],
                        },
                    },
                },
                "indicators": [],
                "features": {},
                "labels": {"kind": "fixed_horizon", "horizon": 3, "threshold": 0.0},
            }
        )
        pipeline.state["raw_data"] = raw_data
        pipeline.state["data"] = raw_data.copy()
        pipeline.state["futures_context"] = futures_context
        pipeline.state["futures_contract_spec"] = {
            "symbol": "BTCUSDT",
            "market": "um_futures",
            "margin_asset": "USDT",
            "liquidation_fee_rate": 0.01,
        }
        pipeline.state["signals"] = {
            "continuous_signals": pd.Series([0.5, 0.5, 0.5, 0.0, 0.0, 0.0], index=index),
            "signals": pd.Series([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], index=index),
        }

        backtest = pipeline.run_backtest()

        self.assertEqual(backtest["account_model"], "futures_margin")
        self.assertGreaterEqual(int(backtest["futures_bracket_count"]), 1)
        self.assertIn("margin_balance_series", backtest)
        self.assertIn("realized_leverage_series", backtest)
        self.assertGreater(int(backtest["liquidation_event_count"]), 0)
        self.assertFalse(backtest["liquidation_events"].empty)

    def test_reference_validation_bundle_feeds_reference_overlay_features(self):
        index = pd.date_range("2026-02-01", periods=48, freq="1h", tz="UTC")
        raw_data = _make_ohlcv(index)
        futures_context = _make_futures_context(index, raw_data["close"].to_numpy())
        bybit_bundle = {
            "mark_price": pd.DataFrame(
                {
                    "mark_open": raw_data["close"].shift(1).fillna(raw_data["close"].iloc[0]) * 1.0004,
                    "mark_high": raw_data["close"] * 1.0012,
                    "mark_low": raw_data["close"] * 0.9992,
                    "mark_close": raw_data["close"] * 1.0006,
                },
                index=index,
            ),
            "index_price": pd.DataFrame(
                {
                    "index_open": raw_data["close"].shift(1).fillna(raw_data["close"].iloc[0]) * 1.0001,
                    "index_high": raw_data["close"] * 1.0008,
                    "index_low": raw_data["close"] * 0.9995,
                    "index_close": raw_data["close"] * 1.0002,
                },
                index=index,
            ),
        }

        reference_bundle = build_reference_validation_bundle(
            raw_data,
            market="um_futures",
            symbol="BTCUSDT",
            interval="1h",
            futures_context=futures_context,
            config={
                "fetch_live": False,
                "futures": {
                    "frames": {"bybit": bybit_bundle},
                },
            },
        )

        pipeline = ResearchPipeline(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h", "market": "um_futures"},
                "indicators": [],
                "features": {"rolling_window": 10, "context_timeframes": ["4h"]},
                "stationarity": {"enabled": False},
            }
        )
        pipeline.state["raw_data"] = raw_data
        pipeline.state["data"] = raw_data.copy()
        pipeline.state["futures_context"] = futures_context
        pipeline.state["reference_overlay_data"] = reference_bundle["overlay"]

        features = pipeline.build_features()

        self.assertIn("reference_price", reference_bundle["overlay"].columns)
        self.assertIn("ref_price_gap", features.columns)
        self.assertIn("composite_basis", features.columns)

    def test_feature_family_metadata_survives_alignment_and_selection(self):
        index = pd.date_range("2026-02-01", periods=240, freq="1h", tz="UTC")
        raw_data = _make_ohlcv(index)
        cross_asset_context = {
            "ETHUSDT": _make_ohlcv(index, drift=9.0, amplitude=2.0, volume_base=1_300.0),
            "SOLUSDT": _make_ohlcv(index, drift=15.0, amplitude=4.0, volume_base=1_100.0),
            "BNBUSDT": _make_ohlcv(index, drift=6.0, amplitude=1.5, volume_base=900.0),
        }
        pipeline = self._make_context_pipeline()
        pipeline.state["raw_data"] = raw_data
        pipeline.state["data"] = raw_data.copy()
        pipeline.state["futures_context"] = _make_futures_context(index, raw_data["close"].to_numpy())
        pipeline.state["cross_asset_context"] = cross_asset_context

        pipeline.build_features()
        pipeline.build_labels()
        aligned = pipeline.align_data()

        feature_families = pipeline.state["feature_families"]
        self.assertEqual(feature_families["return_1"], "endogenous_price")
        self.assertEqual(feature_families["fut_funding_rate"], "futures_context")
        self.assertEqual(feature_families["ctx_ethusdt_ret_1"], "cross_asset")
        self.assertEqual(feature_families["mtf_4h_trend"], "endogenous_price")
        self.assertTrue(set(aligned["X"].columns).issubset(feature_families))

        synthetic_target = (aligned["X"]["fut_funding_rate"] > aligned["X"]["fut_funding_rate"].median()).astype(int)
        selection = select_features(
            aligned["X"],
            synthetic_target,
            feature_blocks=pipeline.state["feature_blocks"],
            config={"enabled": True, "max_features": 8, "min_mi_threshold": 0.0},
        )
        self.assertIn("fut_funding_rate", selection.frame.columns)
        self.assertEqual(selection.feature_families["fut_funding_rate"], "futures_context")
        self.assertIn("futures_context", selection.report["selected_family_summary"]["selected_families"])

    def test_family_ablation_reporting_populated_when_multiple_families_present(self):
        index = pd.date_range("2026-02-01", periods=240, freq="1h", tz="UTC")
        raw_data = _make_ohlcv(index)
        cross_asset_context = {
            "ETHUSDT": _make_ohlcv(index, drift=9.0, amplitude=2.0, volume_base=1_300.0),
            "SOLUSDT": _make_ohlcv(index, drift=15.0, amplitude=4.0, volume_base=1_100.0),
        }
        pipeline = self._make_context_pipeline(feature_selection={"enabled": False})
        pipeline.state["raw_data"] = raw_data
        pipeline.state["data"] = raw_data.copy()
        pipeline.state["futures_context"] = _make_futures_context(index, raw_data["close"].to_numpy())
        pipeline.state["cross_asset_context"] = cross_asset_context

        pipeline.build_features()
        pipeline.build_labels()
        aligned = pipeline.align_data()

        X = aligned["X"]
        target = (
            X["return_1"].fillna(0.0)
            + X["fut_funding_rate"].fillna(0.0)
            + X["ctx_ethusdt_ret_1"].fillna(0.0)
        ) > 0.0
        y = target.astype(int)
        split = len(X) // 2
        X_train = X.iloc[:split]
        X_test = X.iloc[split:]
        y_train = y.iloc[:split]
        y_test = y.iloc[split:]

        model = train_model(X_train, y_train, model_type="logistic")
        diagnostics = compute_feature_family_diagnostics(
            model,
            X_train,
            X_test,
            y_test,
            feature_blocks=pipeline.state["feature_blocks"],
        )
        family_diagnostics = summarize_feature_family_diagnostics([diagnostics])

        self.assertTrue(family_diagnostics["summary"])
        bundle_names = {row["bundle"] for row in family_diagnostics["bundles"]}
        self.assertIn("endogenous_only", bundle_names)
        self.assertIn("full_context", bundle_names)
        self.assertTrue(any("futures_context" in row.get("families", []) for row in family_diagnostics["bundles"]))
        self.assertTrue(any("cross_asset" in row.get("families", []) for row in family_diagnostics["bundles"]))

    def test_pipeline_identifies_endogenous_only_selected_feature_set(self):
        index = pd.date_range("2026-02-01", periods=220, freq="1h", tz="UTC")
        raw_data = _make_ohlcv(index)
        pipeline = self._make_context_pipeline(feature_selection={"enabled": False})
        pipeline.state["raw_data"] = raw_data
        pipeline.state["data"] = raw_data.copy()

        pipeline.build_features()
        pipeline.build_labels()
        aligned = pipeline.align_data()

        synthetic_target = (aligned["X"]["return_1"] > aligned["X"]["return_1"].median()).astype(int)
        selection = select_features(
            aligned["X"],
            synthetic_target,
            feature_blocks=pipeline.state["feature_blocks"],
            config={"enabled": True, "max_features": 6, "min_mi_threshold": 0.0},
        )

        self.assertEqual(selection.report["selected_family_summary"]["selected_families"], ["endogenous_price"])
        self.assertTrue(selection.report["selected_family_summary"]["endogenous_only"])


if __name__ == "__main__":
    unittest.main()