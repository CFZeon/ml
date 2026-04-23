import unittest

import pandas as pd

from core import run_backtest
from core.scenarios import run_scenario_matrix


class ExchangeFailureScenariosTest(unittest.TestCase):
    def test_downtime_windows_suppress_fills(self):
        index = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
        close = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=index)
        signals = pd.Series([0.0, 1.0, 0.0, 1.0, 0.0], index=index)
        volume = pd.Series([1_000.0] * len(index), index=index)

        baseline = run_backtest(
            close=close,
            signals=signals,
            execution_prices=close,
            equity=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            signal_delay_bars=0,
            engine="pandas",
            volume=volume,
            execution_policy={"adapter": "nautilus", "force_simulation": True, "time_in_force": "IOC", "participation_cap": 1.0},
        )
        stressed = run_backtest(
            close=close,
            signals=signals,
            execution_prices=close,
            equity=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            signal_delay_bars=0,
            engine="pandas",
            volume=volume,
            execution_policy={"adapter": "nautilus", "force_simulation": True, "time_in_force": "IOC", "participation_cap": 1.0},
            scenario_schedule=[
                {"event_type": "downtime", "timestamp": index[1]},
                {"event_type": "downtime", "timestamp": index[3]},
            ],
            scenario_policy={"downtime_action": "freeze"},
        )

        self.assertGreater(int(baseline["accepted_orders"]), 0)
        self.assertEqual(int(stressed["scenario_report"]["suppressed_orders"]), 2)
        self.assertEqual(int(stressed["accepted_orders"]), 0)
        self.assertEqual(int(stressed["total_trades"]), 0)

    def test_stale_marks_can_be_rejected(self):
        index = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
        close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=index)
        signals = pd.Series([0.0, 0.5, 0.5, 0.0, 0.0], index=index)

        result = run_backtest(
            close=close,
            signals=signals,
            execution_prices=close,
            equity=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            signal_delay_bars=0,
            engine="pandas",
            scenario_schedule=[{"event_type": "stale_mark", "timestamp": index[2]}],
            scenario_policy={"stale_mark_action": "reject"},
            valuation_price_policy="drop_rows",
        )

        self.assertEqual(int(result["scenario_report"]["stale_mark_rejections"]), 1)
        self.assertEqual(int(result["price_fill_actions"]["valuation"]["dropped_rows"]), 1)

    def test_stale_marks_can_raise_warnings_without_row_rejection(self):
        index = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
        close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=index)
        signals = pd.Series([0.0, 0.5, 0.5, 0.0, 0.0], index=index)

        result = run_backtest(
            close=close,
            signals=signals,
            execution_prices=close,
            equity=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            signal_delay_bars=0,
            engine="pandas",
            scenario_schedule=[{"event_type": "stale_mark", "timestamp": index[2]}],
            scenario_policy={"stale_mark_action": "warn"},
            valuation_price_policy="drop_rows",
        )

        self.assertEqual(int(result["scenario_report"]["stale_mark_rejections"]), 0)
        self.assertIn("stale_mark_warning", result["scenario_report"]["warnings"])
        self.assertEqual(int(result["price_fill_actions"]["valuation"]["dropped_rows"]), 0)

    def test_halt_scenarios_can_force_liquidation(self):
        index = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
        close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], index=index)
        signals = pd.Series([0.0, 1.0, 1.0, 1.0, 1.0, 1.0], index=index)

        result = run_backtest(
            close=close,
            signals=signals,
            execution_prices=close,
            equity=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            signal_delay_bars=0,
            engine="pandas",
            volume=pd.Series(1_000.0, index=index),
            scenario_schedule=[{"event_type": "halt", "start": index[3], "end": index[4]}],
            symbol_lifecycle_policy={"halt_action": "liquidate"},
        )

        self.assertEqual(int(result["symbol_lifecycle_report"]["halt_events"]), 1)
        self.assertEqual(int(result["symbol_lifecycle_report"]["forced_liquidations"]), 1)
        self.assertEqual(pd.Timestamp(result["trade_ledger"].iloc[0]["exit_time"]), index[3])

    def test_scenario_matrix_runs_base_and_stress_cases(self):
        index = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
        close = pd.Series([100.0, 100.0, 100.0, 100.0], index=index)
        signals = pd.Series([0.0, 1.0, 0.0, 0.0], index=index)

        results = run_scenario_matrix(
            run_backtest,
            {
                "close": close,
                "signals": signals,
                "execution_prices": close,
                "equity": 10_000.0,
                "fee_rate": 0.0,
                "slippage_rate": 0.0,
                "signal_delay_bars": 0,
                "engine": "pandas",
                "volume": pd.Series(1_000.0, index=index),
                "execution_policy": {"adapter": "nautilus", "force_simulation": True, "time_in_force": "IOC", "participation_cap": 1.0},
            },
            {
                "downtime": {
                    "events": [{"event_type": "downtime", "timestamp": index[1]}],
                    "policy": {"downtime_action": "freeze"},
                }
            },
        )

        self.assertIn("base", results)
        self.assertIn("downtime", results)
        self.assertGreaterEqual(int(results["base"]["accepted_orders"]), int(results["downtime"]["accepted_orders"]))

    def test_run_backtest_attaches_trade_ready_stress_matrix_summary(self):
        index = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
        close = pd.Series([100.0, 100.0, 101.0, 101.0, 100.5, 100.0], index=index)
        signals = pd.Series([0.0, 1.0, 1.0, 0.0, -1.0, 0.0], index=index)

        result = run_backtest(
            close=close,
            signals=signals,
            execution_prices=close,
            equity=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            signal_delay_bars=0,
            engine="pandas",
            volume=pd.Series(1_000.0, index=index),
            execution_policy={"adapter": "nautilus", "force_simulation": True, "time_in_force": "IOC", "participation_cap": 1.0},
            evaluation_mode="trade_ready",
            required_stress_scenarios=["downtime", "stale_mark", "halt"],
            scenario_matrix={
                "downtime": {
                    "events": [{"event_type": "downtime", "timestamp": index[1]}],
                    "policy": {"downtime_action": "freeze"},
                },
                "stale_mark": {
                    "events": [{"event_type": "stale_mark", "timestamp": index[2]}],
                    "policy": {"stale_mark_action": "reject"},
                },
                "halt": {
                    "events": [{"event_type": "halt", "start": index[3], "end": index[4]}],
                    "policy": {},
                },
            },
        )

        self.assertEqual(result["evaluation_mode"], "trade_ready")
        self.assertTrue(result["stress_matrix"]["configured"])
        self.assertEqual(result["stress_matrix"]["scenario_count"], 3)
        self.assertCountEqual(result["stress_matrix"]["scenario_names"], ["downtime", "stale_mark", "halt"])
        self.assertTrue(result["stress_realism_ready"])


if __name__ == "__main__":
    unittest.main()