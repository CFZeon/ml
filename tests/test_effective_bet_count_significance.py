import unittest

import numpy as np
import pandas as pd

from core import run_backtest


class EffectiveBetCountSignificanceTest(unittest.TestCase):
    def test_significance_is_underpowered_when_bar_count_is_high_but_bet_count_is_low(self):
        index = pd.date_range("2026-04-01", periods=24, freq="1h", tz="UTC")
        close = pd.Series(np.linspace(100.0, 110.0, len(index)), index=index)
        signals = pd.Series(0.0, index=index, dtype=float)
        signals.iloc[1:6] = 1.0
        signals.iloc[12:18] = -1.0

        report = run_backtest(
            close=close,
            signals=signals,
            equity=1_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            execution_prices=close,
            signal_delay_bars=0,
            allow_short=True,
            significance={"enabled": True, "min_observations": 8, "min_effective_bets": 8},
        )

        significance = report["statistical_significance"]
        self.assertLess(int(report["effective_bet_count"]), 8)
        self.assertFalse(significance["enabled"])
        self.assertTrue(significance["underpowered"])
        self.assertEqual(significance["underpowered_reason"], "insufficient_effective_bets")
        self.assertEqual(int(significance["observation_count"]), len(close))
        self.assertEqual(int(significance["effective_bet_count"]), int(report["effective_bet_count"]))
        self.assertEqual(int(significance["min_effective_bets"]), 8)

        qualification = report["metric_qualification"]
        self.assertTrue(qualification["trade_level"]["low_sample_advisory"])
        self.assertTrue(qualification["portfolio_level"]["low_sample_advisory"])
        self.assertIn(
            "insufficient_realized_trade_count_for_trade_metrics",
            qualification["warnings"],
        )
        self.assertIn(
            "insufficient_effective_bet_count_for_portfolio_metrics",
            qualification["warnings"],
        )
        self.assertIsNone(report["sample_qualified_metrics"]["trade_profit_factor"])
        self.assertIsNone(report["sample_qualified_metrics"]["calmar_ratio"])
        self.assertLess(report["trade_risk_summary"]["closed_trades"], report["trade_risk_summary"]["minimum_closed_trades"])
        self.assertLess(
            report["portfolio_risk_summary"]["effective_bet_count"],
            report["portfolio_risk_summary"]["minimum_effective_bets"],
        )


if __name__ == "__main__":
    unittest.main()