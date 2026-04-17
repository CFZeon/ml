import unittest

import numpy as np
import pandas as pd

from core import ResearchPipeline, build_execution_outcome_frame, build_trade_outcome_frame
from core.pipeline import _build_signal_state, _compute_theory_thresholds


class SignalProfitabilitySizingTest(unittest.TestCase):
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
        self.assertGreaterEqual(float(training["last_signal_params"]["threshold"]), 0.001)


if __name__ == "__main__":
    unittest.main()