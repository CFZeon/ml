import unittest

import numpy as np
import pandas as pd

from core import ResearchPipeline, build_execution_outcome_frame, build_trade_outcome_frame
from core.pipeline import (
    SignalPolicyBuilder,
    _build_signal_state,
    _compute_theory_thresholds,
    _estimate_trade_outcome_stats,
    _select_causal_prior_trade_outcomes,
)


class SignalProfitabilitySizingTest(unittest.TestCase):
    def test_kelly_fallback_to_flat_below_trade_threshold(self):
        index = pd.date_range("2026-03-01", periods=2, freq="1h", tz="UTC")
        predictions = pd.Series([1, -1], index=index)
        probability_frame = pd.DataFrame(
            {
                -1: [0.1, 0.9],
                0: [0.0, 0.0],
                1: [0.9, 0.1],
            },
            index=index,
        )
        profitability_prob = pd.Series([0.8, 0.7], index=index)

        with self.assertWarnsRegex(RuntimeWarning, "Kelly sizing fallback activated"):
            state = _build_signal_state(
                predictions,
                probability_frame,
                profitability_prob,
                {
                    "threshold": 0.0,
                    "edge_threshold": 0.0,
                    "meta_threshold": 0.55,
                    "fraction": 0.4,
                    "sizing_mode": "kelly",
                    "min_trades_for_kelly": 30,
                    "max_kelly_fraction": 0.5,
                },
                avg_win=0.03,
                avg_loss=0.01,
                holding_bars=1,
                kelly_trade_count=8,
            )

        self.assertTrue(state["used_flat_kelly_fallback"])
        self.assertEqual(int(state["kelly_trade_count"]), 8)
        self.assertAlmostEqual(float(state["position_size"].iloc[0]), 0.4, places=6)
        self.assertAlmostEqual(float(state["position_size"].iloc[1]), 0.4, places=6)
        self.assertAlmostEqual(float(state["event_signals"].iloc[0]), 0.4, places=6)
        self.assertAlmostEqual(float(state["event_signals"].iloc[1]), -0.4, places=6)

    def test_kelly_shrinkage_blends_fold_and_pooled(self):
        index = pd.date_range("2026-03-02", periods=3, freq="1h", tz="UTC")
        fold_trade_outcomes = pd.DataFrame(
            {
                "net_trade_return": [0.10, 0.06, -0.04],
                "trade_taken": [1, 1, 1],
            },
            index=index,
        )
        pooled_trade_outcomes = pd.DataFrame(
            {
                "net_trade_return": [0.02, -0.01, -0.03],
                "trade_taken": [1, 1, 1],
            },
            index=pd.date_range("2026-02-20", periods=3, freq="1h", tz="UTC"),
        )

        avg_win, avg_loss = _estimate_trade_outcome_stats(
            fold_trade_outcomes,
            default_win=0.02,
            default_loss=0.02,
            pooled_trade_outcomes=pooled_trade_outcomes,
            shrinkage_alpha=0.5,
        )

        self.assertAlmostEqual(avg_win, 0.05, places=6)
        self.assertAlmostEqual(avg_loss, 0.03, places=6)

    def test_kelly_capped_at_max_fraction(self):
        index = pd.date_range("2026-03-03", periods=1, freq="1h", tz="UTC")
        predictions = pd.Series([1], index=index)
        probability_frame = pd.DataFrame({-1: [0.05], 0: [0.0], 1: [0.95]}, index=index)
        profitability_prob = pd.Series([0.9], index=index)

        state = _build_signal_state(
            predictions,
            probability_frame,
            profitability_prob,
            {
                "threshold": 0.0,
                "edge_threshold": 0.0,
                "meta_threshold": 0.55,
                "fraction": 1.0,
                "sizing_mode": "kelly",
                "min_trades_for_kelly": 30,
                "max_kelly_fraction": 0.2,
            },
            avg_win=0.04,
            avg_loss=0.01,
            holding_bars=1,
            kelly_trade_count=50,
        )

        self.assertFalse(state["used_flat_kelly_fallback"])
        self.assertAlmostEqual(float(state["position_size"].iloc[0]), 0.2, places=6)

    def test_walk_forward_prior_trade_outcomes_require_strictly_earlier_timestamps(self):
        earlier = pd.DataFrame(
            {"net_trade_return": [0.03], "trade_taken": [1]},
            index=pd.date_range("2026-03-01", periods=1, freq="1h", tz="UTC"),
        )
        later = pd.DataFrame(
            {"net_trade_return": [0.07], "trade_taken": [1]},
            index=pd.date_range("2026-03-10", periods=1, freq="1h", tz="UTC"),
        )

        filtered, policy = _select_causal_prior_trade_outcomes(
            [later, earlier],
            validation_method="walk_forward",
            calibration_cutoff_timestamp=pd.Timestamp("2026-03-05", tz="UTC"),
        )

        self.assertListEqual(filtered.index.tolist(), earlier.index.tolist())
        self.assertEqual(policy["policy_name"], "strictly_earlier_oos_only")
        self.assertTrue(policy["allow_cross_fold_borrowing"])
        self.assertEqual(int(policy["causal_trade_rows"]), 1)

    def test_cpcv_prior_trade_outcomes_do_not_borrow_other_paths(self):
        earlier = pd.DataFrame(
            {"net_trade_return": [0.03], "trade_taken": [1]},
            index=pd.date_range("2026-03-01", periods=1, freq="1h", tz="UTC"),
        )
        later = pd.DataFrame(
            {"net_trade_return": [0.07], "trade_taken": [1]},
            index=pd.date_range("2026-03-10", periods=1, freq="1h", tz="UTC"),
        )

        filtered, policy = _select_causal_prior_trade_outcomes(
            [earlier, later],
            validation_method="cpcv",
            calibration_cutoff_timestamp=pd.Timestamp("2026-03-12", tz="UTC"),
        )

        self.assertTrue(filtered.empty)
        self.assertEqual(policy["policy_name"], "validation_only_or_defaults")
        self.assertFalse(policy["allow_cross_fold_borrowing"])
        self.assertEqual(int(policy["causal_trade_rows"]), 0)

    def test_trade_outcome_frame_uses_profitability_not_raw_correctness(self):
        index = pd.date_range("2026-03-01", periods=4, freq="1h", tz="UTC")
        predictions = pd.Series([1, -1, 1, 0], index=index)
        labels = pd.DataFrame(
            {
                "label": [0, 0, 1, -1],
                "gross_return": [0.0200, -0.0150, 0.0005, -0.0100],
                "cost_rate": [0.0010, 0.0010, 0.0010, 0.0010],
            },
            index=index,
        )

        outcomes = build_trade_outcome_frame(predictions, labels)

        self.assertListEqual(outcomes["profitable"].tolist(), [1, 1, 0, 0])
        self.assertAlmostEqual(float(outcomes.loc[index[0], "net_trade_return"]), 0.0190, places=6)
        self.assertAlmostEqual(float(outcomes.loc[index[1], "net_trade_return"]), 0.0140, places=6)
        self.assertAlmostEqual(float(outcomes.loc[index[2], "net_trade_return"]), -0.0005, places=6)

    def test_expected_utility_sizing_uses_break_even_probability_gate(self):
        index = pd.date_range("2026-03-05", periods=2, freq="1h", tz="UTC")
        predictions = pd.Series([1, -1], index=index)
        probability_frame = pd.DataFrame(
            {
                -1: [0.35, 0.70],
                0: [0.0, 0.0],
                1: [0.65, 0.30],
            },
            index=index,
        )
        profitability_prob = pd.Series([0.48, 0.62], index=index)

        state = _build_signal_state(
            predictions,
            probability_frame,
            profitability_prob,
            {
                "threshold": 0.0,
                "edge_threshold": 0.0,
                "meta_threshold": 0.55,
                "fraction": 1.0,
                "sizing_mode": "expected_utility",
            },
            avg_win=0.015,
            avg_loss=0.010,
            holding_bars=1,
        )

        self.assertLess(state["break_even_profit_prob"], 0.48)
        self.assertGreater(float(state["position_size"].iloc[0]), 0.0)
        self.assertGreater(float(state["position_size"].iloc[1]), 0.0)
        self.assertNotEqual(int(state["signals"].iloc[0]), 0)
        self.assertNotEqual(int(state["signals"].iloc[1]), 0)

    def test_execution_outcome_frame_matches_backtest_style_returns(self):
        index = pd.date_range("2026-03-07", periods=6, freq="1h", tz="UTC")
        valuation = pd.Series([100.0, 101.0, 103.0, 102.0, 100.0, 101.0], index=index)
        execution = valuation.copy()
        predictions = pd.Series([1, -1, 0], index=index[:3])

        outcomes = build_execution_outcome_frame(
            predictions,
            valuation_prices=valuation,
            execution_prices=execution,
            holding_bars=2,
            signal_delay_bars=1,
            fee_rate=0.001,
            slippage_rate=0.0,
        )

        long_expected = (1.0 + 0.01) * (1.0 + (103.0 / 101.0 - 1.0)) - 1.0 - 0.002
        short_expected = (1.0 - (103.0 / 101.0 - 1.0)) * (1.0 - (102.0 / 103.0 - 1.0)) - 1.0 - 0.002

        self.assertAlmostEqual(float(outcomes.loc[index[0], "net_trade_return"]), long_expected, places=6)
        self.assertAlmostEqual(float(outcomes.loc[index[1], "net_trade_return"]), short_expected, places=6)
        self.assertEqual(int(outcomes.loc[index[2], "trade_taken"]), 0)

    def test_execution_outcome_frame_uses_execution_price_path_when_prices_diverge(self):
        index = pd.date_range("2026-03-08", periods=6, freq="1h", tz="UTC")
        valuation = pd.Series([100.0, 101.0, 103.0, 102.0, 100.0, 99.0], index=index)
        execution = pd.Series([100.0, 105.0, 104.0, 109.0, 107.0, 108.0], index=index)
        predictions = pd.Series([1, -1], index=index[:2])

        outcomes = build_execution_outcome_frame(
            predictions,
            valuation_prices=valuation,
            execution_prices=execution,
            holding_bars=2,
            signal_delay_bars=1,
            fee_rate=0.0,
            slippage_rate=0.0,
        )

        execution_returns = execution.pct_change().fillna(0.0)
        valuation_returns = valuation.pct_change().fillna(0.0)
        long_expected = (1.0 + execution_returns.iloc[1]) * (1.0 + execution_returns.iloc[2]) - 1.0
        short_expected = (1.0 - execution_returns.iloc[2]) * (1.0 - execution_returns.iloc[3]) - 1.0
        long_valuation_path = (1.0 + valuation_returns.iloc[1]) * (1.0 + valuation_returns.iloc[2]) - 1.0

        self.assertAlmostEqual(float(outcomes.loc[index[0], "net_trade_return"]), long_expected, places=6)
        self.assertAlmostEqual(float(outcomes.loc[index[1], "net_trade_return"]), short_expected, places=6)
        self.assertNotAlmostEqual(float(outcomes.loc[index[0], "net_trade_return"]), long_valuation_path, places=4)

    def test_execution_outcome_frame_applies_dynamic_slippage_costs(self):
        index = pd.date_range("2026-03-09", periods=6, freq="1h", tz="UTC")
        valuation = pd.Series([100.0, 101.0, 103.0, 102.0, 100.0, 101.0], index=index)
        predictions = pd.Series([1, -1], index=index[:2])
        volume = pd.Series(5.0, index=index)

        baseline = build_execution_outcome_frame(
            predictions,
            valuation_prices=valuation,
            execution_prices=valuation,
            holding_bars=2,
            signal_delay_bars=1,
            fee_rate=0.0,
            slippage_rate=0.0,
            equity=10_000.0,
            volume=volume,
        )
        impacted = build_execution_outcome_frame(
            predictions,
            valuation_prices=valuation,
            execution_prices=valuation,
            holding_bars=2,
            signal_delay_bars=1,
            fee_rate=0.0,
            slippage_rate=0.0,
            equity=10_000.0,
            volume=volume,
            slippage_model="sqrt_impact",
        )

        self.assertGreater(float(impacted.loc[index[0], "slippage_cost_rate"]), 0.0)
        self.assertLess(float(impacted.loc[index[0], "net_trade_return"]), float(baseline.loc[index[0], "net_trade_return"]))
        self.assertLess(float(impacted.loc[index[1], "net_trade_return"]), float(baseline.loc[index[1], "net_trade_return"]))

    def test_execution_outcome_frame_requires_volume_for_dynamic_slippage(self):
        index = pd.date_range("2026-03-09", periods=4, freq="1h", tz="UTC")
        valuation = pd.Series([100.0, 101.0, 102.0, 103.0], index=index)
        predictions = pd.Series([1], index=index[:1])

        with self.assertRaisesRegex(ValueError, "volume is required"):
            build_execution_outcome_frame(
                predictions,
                valuation_prices=valuation,
                execution_prices=valuation,
                holding_bars=1,
                signal_delay_bars=1,
                slippage_model="sqrt_impact",
            )

    def test_theory_thresholds_use_realized_dynamic_costs(self):
        trade_outcomes = pd.DataFrame(
            {
                "trade_taken": [1, 1, 0],
                "round_trip_cost_rate": [0.0040, 0.0060, 0.0],
            },
            index=pd.date_range("2026-03-09", periods=3, freq="1h", tz="UTC"),
        )

        thresholds = _compute_theory_thresholds(
            avg_win=0.03,
            avg_loss=0.02,
            backtest_config={"fee_rate": 0.0, "slippage_rate": 0.0},
            signal_config={"threshold": 0.0, "edge_threshold": 0.0, "fraction": 1.0},
            trade_outcomes=trade_outcomes,
        )

        self.assertAlmostEqual(float(thresholds["params"]["threshold"]), 0.005, places=6)

    def test_signal_policy_builder_validation_calibrated_uses_validation_trade_outcomes(self):
        trade_outcomes = pd.DataFrame(
            {
                "trade_taken": [1, 1, 0],
                "round_trip_cost_rate": [0.0040, 0.0060, 0.0],
            },
            index=pd.date_range("2026-03-09", periods=3, freq="1h", tz="UTC"),
        )
        builder = SignalPolicyBuilder(
            {"policy_mode": "validation_calibrated", "threshold": 0.0, "edge_threshold": 0.0, "fraction": 1.0},
            {"fee_rate": 0.0, "slippage_rate": 0.0},
        )

        report = builder.build(
            avg_win=0.03,
            avg_loss=0.02,
            trade_outcomes=trade_outcomes,
            calibration_context={"source": "validation_trade_outcomes", "calibration_rows": 3},
        )

        self.assertEqual(report["mode"], "validation_calibrated")
        self.assertEqual(report["policy_quality"]["source"], "validation_trade_outcomes")
        self.assertTrue(report["policy_quality"]["used_trade_outcomes"])
        self.assertAlmostEqual(float(report["params"]["threshold"]), 0.005, places=6)

    def test_signal_policy_builder_frozen_manual_preserves_manual_policy(self):
        builder = SignalPolicyBuilder(
            {
                "policy_mode": "frozen_manual",
                "threshold": 0.07,
                "edge_threshold": 0.11,
                "meta_threshold": 0.61,
                "fraction": 0.4,
            },
            {"fee_rate": 0.001, "slippage_rate": 0.001},
        )

        report = builder.build(avg_win=0.03, avg_loss=0.02)

        self.assertEqual(report["mode"], "frozen_manual")
        self.assertEqual(report["policy_quality"]["source"], "frozen_manual")
        self.assertAlmostEqual(float(report["params"]["threshold"]), 0.07, places=6)
        self.assertAlmostEqual(float(report["params"]["edge_threshold"]), 0.11, places=6)
        self.assertAlmostEqual(float(report["params"]["meta_threshold"]), 0.61, places=6)
        self.assertAlmostEqual(float(report["params"]["fraction"]), 0.4, places=6)

    def test_pipeline_oos_avg_win_loss_ignore_label_gross_return(self):
        index = pd.date_range("2026-03-10", periods=220, freq="1h", tz="UTC")
        steps = np.linspace(0.0, 1.0, len(index))
        close = 100.0 + 8.0 * steps + 1.5 * np.sin(np.linspace(0.0, 8.0 * np.pi, len(index)))
        open_ = np.r_[close[0], close[:-1]]
        high = np.maximum(open_, close) * 1.002
        low = np.minimum(open_, close) * 0.998
        volume = 1_000.0 + 40.0 * np.sin(np.linspace(0.0, 4.0 * np.pi, len(index)))
        raw = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "quote_volume": close * volume,
                "trades": 100,
            },
            index=index,
        )

        pipeline = ResearchPipeline(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "indicators": [],
                "features": {"lags": [1, 3], "frac_diff_d": 0.4, "rolling_window": 20},
                "labels": {"kind": "fixed_horizon", "horizon": 6, "threshold": 0.0001},
                "model": {"type": "gbm", "cv_method": "walk_forward", "n_splits": 1, "gap": 0},
                "feature_selection": {"enabled": True, "max_features": 12},
                "signals": {"avg_win": 0.02, "avg_loss": 0.02},
                "backtest": {
                    "equity": 10_000.0,
                    "use_open_execution": False,
                    "signal_delay_bars": 1,
                    "fee_rate": 0.0,
                    "slippage_rate": 0.0,
                    "slippage_model": "sqrt_impact",
                },
            }
        )
        pipeline.state["raw_data"] = raw
        pipeline.state["data"] = raw.copy()
        pipeline.build_features()
        labels = pipeline.build_labels().copy()
        labels["gross_return"] = np.where(labels["label"] >= 0, 0.35, -0.35)
        pipeline.state["labels"] = labels
        pipeline.align_data()
        training = pipeline.train_models()

        expected_outcomes = build_execution_outcome_frame(
            training["oos_predictions"],
            valuation_prices=raw["close"],
            execution_prices=raw["close"],
            holding_bars=6,
            signal_delay_bars=1,
            fee_rate=0.0,
            slippage_rate=0.0,
            equity=10_000.0,
            volume=raw["volume"],
            slippage_model="sqrt_impact",
            cutoff_timestamp=training["oos_predictions"].index[-1],
        )
        realized = expected_outcomes.loc[expected_outcomes["trade_taken"].eq(1), "net_trade_return"].dropna()
        expected_avg_win = float(realized[realized > 0].mean())
        expected_avg_loss = float(realized[realized < 0].abs().mean())

        self.assertAlmostEqual(float(training["oos_avg_win"]), expected_avg_win, places=6)
        self.assertAlmostEqual(float(training["oos_avg_loss"]), expected_avg_loss, places=6)
        self.assertIn("signal_policy", training)
        self.assertEqual(training["signal_policy"]["mode"], "validation_calibrated")
        self.assertGreaterEqual(len(training["signal_policy"]["folds"]), 1)
        self.assertEqual(
            training["signal_policy"]["last_policy_quality"]["source"],
            "validation_unavailable_static_fallback",
        )
        self.assertEqual(training["signal_policy"]["calibration_policy"]["policy_name"], "strictly_earlier_oos_only")
        self.assertIn("causal_calibration_rows", training["signal_policy"]["last_policy_quality"])
        self.assertIn("causal_cutoff_timestamp", training["signal_policy"]["last_policy_quality"])
        self.assertIn("cross_fold_borrowing_allowed", training["signal_policy"]["last_policy_quality"])

    def test_pipeline_persists_signal_decay_in_training_and_backtest(self):
        index = pd.date_range("2026-03-13", periods=220, freq="1h", tz="UTC")
        steps = np.linspace(0.0, 1.0, len(index))
        close = 100.0 + 6.0 * steps + 2.0 * np.sin(np.linspace(0.0, 10.0 * np.pi, len(index)))
        open_ = np.r_[close[0], close[:-1]]
        high = np.maximum(open_, close) * 1.002
        low = np.minimum(open_, close) * 0.998
        volume = 1_200.0 + 30.0 * np.cos(np.linspace(0.0, 5.0 * np.pi, len(index)))
        raw = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "quote_volume": close * volume,
                "trades": 100,
            },
            index=index,
        )

        pipeline = ResearchPipeline(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "indicators": [],
                "features": {"lags": [1, 3], "frac_diff_d": 0.4, "rolling_window": 20},
                "labels": {"kind": "fixed_horizon", "horizon": 6, "threshold": 0.0001},
                "model": {"type": "gbm", "cv_method": "walk_forward", "n_splits": 1, "gap": 0},
                "feature_selection": {"enabled": True, "max_features": 12},
                "signals": {
                    "avg_win": 0.02,
                    "avg_loss": 0.02,
                    "decay": {
                        "min_realized_trade_count": 1,
                        "min_half_life_holding_ratio": 0.0,
                    },
                },
                "backtest": {
                    "equity": 10_000.0,
                    "use_open_execution": False,
                    "signal_delay_bars": 1,
                    "fee_rate": 0.0,
                    "slippage_rate": 0.0,
                },
            }
        )
        pipeline.state["raw_data"] = raw
        pipeline.state["data"] = raw.copy()

        pipeline.build_features()
        pipeline.build_labels()
        pipeline.align_data()
        training = pipeline.train_models()
        pipeline.generate_signals()
        backtest = pipeline.run_backtest()

        self.assertIn("signal_decay", training)
        self.assertIn("net_edge_by_horizon", training["signal_decay"])
        self.assertIn("signal_decay", backtest)
        self.assertIn("signal_decay", backtest["operational_monitoring"]["components"])


if __name__ == "__main__":
    unittest.main()