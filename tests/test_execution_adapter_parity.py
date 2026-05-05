import unittest
from unittest import mock

import pandas as pd

import core.backtest as backtest_module
from core import run_backtest
from core.execution import ExecutionAdapterUnavailableError


class ExecutionAdapterParityTest(unittest.TestCase):
    def test_default_execution_uses_explicit_bar_surrogate_mode(self):
        index = pd.date_range("2024-01-01", periods=4, freq="h")
        close = pd.Series([100.0, 101.0, 102.0, 103.0], index=index)
        signals = pd.Series([0.0, 0.5, 0.5, 0.0], index=index)

        result = run_backtest(
            close,
            signals,
            execution_prices=close,
            engine="pandas",
            signal_delay_bars=0,
            volume=pd.Series(10_000.0, index=index),
        )

        self.assertEqual(result["execution_adapter"], "bar_surrogate")
        self.assertEqual(result["execution_mode"], "conservative_bar_surrogate")
        self.assertFalse(result["promotion_execution_ready"])

    def test_nautilus_adapter_requires_explicit_force_simulation_without_backend(self):
        index = pd.date_range("2024-01-01", periods=3, freq="h")
        close = pd.Series([100.0, 100.0, 100.0], index=index)
        signals = pd.Series([0.0, 1.0, 0.0], index=index)

        with self.assertRaises(ExecutionAdapterUnavailableError):
            run_backtest(
                close,
                signals,
                execution_prices=close,
                engine="pandas",
                signal_delay_bars=0,
                volume=pd.Series(1_000.0, index=index),
                execution_policy={"adapter": "nautilus", "time_in_force": "IOC", "participation_cap": 1.0},
            )

    def test_trade_ready_surrogate_backtest_records_execution_blocker(self):
        index = pd.date_range("2024-01-01", periods=4, freq="h")
        close = pd.Series([100.0, 101.0, 102.0, 103.0], index=index)
        signals = pd.Series([0.0, 0.5, 0.5, 0.0], index=index)

        result = run_backtest(
            close,
            signals,
            execution_prices=close,
            engine="pandas",
            signal_delay_bars=0,
            volume=pd.Series(10_000.0, index=index),
            evaluation_mode="trade_ready",
        )

        self.assertTrue(result["research_only"])
        self.assertIn("execution_backend_not_event_driven", result["trade_ready_blockers"])

    def test_vectorbt_import_error_requires_explicit_research_fallback_opt_in(self):
        index = pd.date_range("2024-01-01", periods=4, freq="h")
        close = pd.Series([100.0, 101.0, 102.0, 103.0], index=index)
        signals = pd.Series([0.0, 0.5, 0.5, 0.0], index=index)

        with mock.patch.object(backtest_module, "_run_vectorbt_backtest", side_effect=ImportError("vectorbt unavailable")):
            with self.assertRaisesRegex(ImportError, "allow_engine_fallback=True"):
                run_backtest(
                    close,
                    signals,
                    execution_prices=close,
                    engine="vectorbt",
                    signal_delay_bars=0,
                    volume=pd.Series(10_000.0, index=index),
                )

    def test_research_vectorbt_import_error_can_explicitly_fallback_to_pandas(self):
        index = pd.date_range("2024-01-01", periods=4, freq="h")
        close = pd.Series([100.0, 101.0, 102.0, 103.0], index=index)
        signals = pd.Series([0.0, 0.5, 0.5, 0.0], index=index)

        with mock.patch.object(backtest_module, "_run_vectorbt_backtest", side_effect=ImportError("vectorbt unavailable")):
            result = run_backtest(
                close,
                signals,
                execution_prices=close,
                engine="vectorbt",
                signal_delay_bars=0,
                volume=pd.Series(10_000.0, index=index),
                allow_engine_fallback=True,
            )

        self.assertEqual(result["requested_engine"], "vectorbt")
        self.assertEqual(result["engine"], "pandas")
        self.assertTrue(result["engine_fallback_used"])
        self.assertEqual(result["engine_fallback_reason"], "vectorbt_unavailable")
        self.assertIn("engine_fallback_to_pandas", result["backtest_warnings"])

    def test_capital_facing_vectorbt_import_error_fails_closed(self):
        index = pd.date_range("2024-01-01", periods=4, freq="h")
        close = pd.Series([100.0, 101.0, 102.0, 103.0], index=index)
        signals = pd.Series([0.0, 0.5, 0.5, 0.0], index=index)

        with mock.patch.object(backtest_module, "_run_vectorbt_backtest", side_effect=ImportError("vectorbt unavailable")):
            with self.assertRaisesRegex(ImportError, "require explicit engine parity"):
                run_backtest(
                    close,
                    signals,
                    execution_prices=close,
                    engine="vectorbt",
                    signal_delay_bars=0,
                    volume=pd.Series(10_000.0, index=index),
                    evaluation_mode="trade_ready",
                    allow_engine_fallback=True,
                )

    def test_legacy_and_default_execution_match_when_liquidity_is_abundant(self):
        index = pd.date_range("2024-01-01", periods=4, freq="h")
        close = pd.Series([100.0, 101.0, 102.0, 103.0], index=index)
        signals = pd.Series([0.0, 0.5, 0.5, 0.0], index=index)
        volume = pd.Series([10_000.0, 10_000.0, 10_000.0, 10_000.0], index=index)

        legacy = run_backtest(
            close,
            signals,
            execution_prices=close,
            engine="pandas",
            signal_delay_bars=0,
            volume=volume,
            execution_policy={"adapter": "legacy"},
        )
        current = run_backtest(
            close,
            signals,
            execution_prices=close,
            engine="pandas",
            signal_delay_bars=0,
            volume=volume,
            execution_policy={
                "adapter": "nautilus",
                "force_simulation": True,
                "time_in_force": "IOC",
                "participation_cap": 1.0,
            },
        )

        pd.testing.assert_series_equal(
            legacy["equity_curve"],
            current["equity_curve"],
            check_exact=False,
            atol=1e-10,
            rtol=1e-10,
        )
        self.assertAlmostEqual(float(current["fill_ratio"]), 1.0, places=6)
        self.assertEqual(int(legacy["blocked_orders"]), int(current["blocked_orders"]))
        self.assertEqual(int(legacy["adjusted_orders"]), int(current["adjusted_orders"]))
        self.assertEqual(legacy["order_rejection_reasons"], current["order_rejection_reasons"])


if __name__ == "__main__":
    unittest.main()