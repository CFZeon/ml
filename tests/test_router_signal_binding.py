import unittest

import pandas as pd

from core.backtest import run_backtest
from core.pipeline import _route_signal_state_with_router
from core.regimes import RegimeStateContract
from core.routing import HardSwitchRouter
from core.specialists import SpecialistHealthContract, SpecialistLibrarySnapshot, SpecialistSpec


class RouterSignalBindingTest(unittest.TestCase):
    def test_router_bound_signal_state_changes_executed_pnl(self):
        index = pd.date_range("2026-05-10", periods=4, freq="1h", tz="UTC")
        close = pd.Series([100.0, 105.0, 110.0, 115.0], index=index)
        zeros = pd.Series(0.0, index=index)
        ones = pd.Series(1.0, index=index)

        base_signal_state = {
            "event_signals": zeros,
            "continuous_signals": zeros,
            "signals": zeros.astype(int),
            "position_size": zeros,
            "meta_prob": zeros,
            "profitability_prob": zeros,
            "direction_edge": zeros,
            "confidence": zeros,
            "expected_trade_edge": zeros,
        }
        specialist_signal_surfaces = {
            "fallback_generalist": {
                "event_signals": zeros,
                "continuous_signals": zeros,
                "signals": zeros.astype(int),
                "position_size": zeros,
                "meta_prob": zeros,
                "profitability_prob": zeros,
                "direction_edge": zeros,
                "confidence": zeros,
                "expected_trade_edge": zeros,
            },
            "specialist::bull": {
                "event_signals": ones,
                "continuous_signals": ones,
                "signals": ones.astype(int),
                "position_size": ones,
                "meta_prob": ones,
                "profitability_prob": ones,
                "direction_edge": ones,
                "confidence": ones,
                "expected_trade_edge": ones,
            },
        }
        specialist_library = SpecialistLibrarySnapshot(
            symbol="BTCUSDT",
            timeframe="1h",
            fallback_model_id="fallback_generalist",
            specialists=[
                SpecialistSpec(
                    model_id="fallback_generalist",
                    symbol="BTCUSDT",
                    timeframe="1h",
                    compatible_regimes=[],
                    estimator_family="dummy",
                ),
                SpecialistSpec(
                    model_id="specialist::bull",
                    symbol="BTCUSDT",
                    timeframe="1h",
                    compatible_regimes=["bull"],
                    estimator_family="dummy",
                ),
            ],
        ).to_dict()
        regime_states = [
            RegimeStateContract(
                as_of=timestamp,
                available_at=timestamp,
                label="bull",
                confidence=0.95,
            )
            for timestamp in index
        ]

        routed_signal_state, router_trace_summary = _route_signal_state_with_router(
            base_signal_state,
            specialist_signal_surfaces,
            router=HardSwitchRouter(),
            specialist_library=specialist_library,
            regime_states=regime_states,
        )

        self.assertTrue(routed_signal_state["router_bound"])
        self.assertEqual(router_trace_summary["summary"]["binding_mode"], "executed_signals")
        self.assertTrue((routed_signal_state["continuous_signals"] == 1.0).all())

        baseline = run_backtest(
            close=close,
            signals=base_signal_state["continuous_signals"],
            execution_prices=close,
            signal_delay_bars=0,
            fee_rate=0.0,
            slippage_rate=0.0,
            engine="pandas",
        )
        routed = run_backtest(
            close=close,
            signals=routed_signal_state["continuous_signals"],
            execution_prices=close,
            signal_delay_bars=0,
            fee_rate=0.0,
            slippage_rate=0.0,
            engine="pandas",
        )

        self.assertEqual(float(baseline["net_profit_pct"]), 0.0)
        self.assertGreater(float(routed["net_profit_pct"]), 0.0)

    def test_router_binding_replays_with_row_level_specialist_health_trace(self):
        index = pd.date_range("2026-05-10", periods=3, freq="1h", tz="UTC")
        zeros = pd.Series(0.0, index=index)
        ones = pd.Series(1.0, index=index)
        base_signal_state = {
            "event_signals": zeros,
            "continuous_signals": zeros,
            "signals": zeros.astype(int),
            "position_size": zeros,
            "meta_prob": zeros,
            "profitability_prob": zeros,
            "direction_edge": zeros,
            "confidence": zeros,
            "expected_trade_edge": zeros,
        }
        specialist_signal_surfaces = {
            "fallback_generalist": {
                "event_signals": zeros,
                "continuous_signals": zeros,
                "signals": zeros.astype(int),
                "position_size": zeros,
                "meta_prob": zeros,
                "profitability_prob": zeros,
                "direction_edge": zeros,
                "confidence": zeros,
                "expected_trade_edge": zeros,
            },
            "specialist::bull": {
                "event_signals": ones,
                "continuous_signals": ones,
                "signals": ones.astype(int),
                "position_size": ones,
                "meta_prob": ones,
                "profitability_prob": ones,
                "direction_edge": ones,
                "confidence": ones,
                "expected_trade_edge": ones,
            },
        }
        specialist_library = SpecialistLibrarySnapshot(
            symbol="BTCUSDT",
            timeframe="1h",
            fallback_model_id="fallback_generalist",
            specialists=[
                SpecialistSpec(
                    model_id="fallback_generalist",
                    symbol="BTCUSDT",
                    timeframe="1h",
                    compatible_regimes=[],
                    estimator_family="dummy",
                    metadata={"fallback_only": True, "lifecycle_state": "active"},
                ),
                SpecialistSpec(
                    model_id="specialist::bull",
                    symbol="BTCUSDT",
                    timeframe="1h",
                    compatible_regimes=["bull"],
                    estimator_family="dummy",
                    metadata={"lifecycle_state": "active"},
                ),
            ],
            health=[
                SpecialistHealthContract(model_id="fallback_generalist", fallback_only=True, stability_score=0.55),
                SpecialistHealthContract(model_id="specialist::bull", compatible_regimes=["bull"], stability_score=0.8),
            ],
        ).to_dict()
        regime_states = [
            RegimeStateContract(as_of=timestamp, available_at=timestamp, label="bull", confidence=0.95)
            for timestamp in index
        ]
        specialist_health_trace = [
            {"health": []},
            {
                "health": [
                    SpecialistHealthContract(
                        model_id="specialist::bull",
                        compatible_regimes=["bull"],
                        stability_score=0.8,
                        failure_flags=["runtime_failure"],
                    ).to_dict()
                ]
            },
            {"health": []},
        ]

        routed_signal_state, router_trace_summary = _route_signal_state_with_router(
            base_signal_state,
            specialist_signal_surfaces,
            router=HardSwitchRouter(),
            specialist_library=specialist_library,
            regime_states=regime_states,
            specialist_health_trace=specialist_health_trace,
        )

        self.assertListEqual(routed_signal_state["continuous_signals"].tolist(), [1.0, 0.0, 1.0])
        self.assertEqual(router_trace_summary["summary"]["eligibility_blocked_rows"], 1)
        self.assertEqual(router_trace_summary["summary"]["fallback_rows_by_cause"], {"health_failure_flags": 1})
        self.assertTrue(pd.isna(routed_signal_state["routed_signal_surfaces"]["specialist::bull"].iloc[1]))

    def test_router_binding_fails_closed_until_health_binding_exists(self):
        index = pd.date_range("2026-05-10", periods=3, freq="1h", tz="UTC")
        zeros = pd.Series(0.0, index=index)
        ones = pd.Series(1.0, index=index)
        base_signal_state = {
            "event_signals": zeros,
            "continuous_signals": zeros,
            "signals": zeros.astype(int),
            "position_size": zeros,
            "meta_prob": zeros,
            "profitability_prob": zeros,
            "direction_edge": zeros,
            "confidence": zeros,
            "expected_trade_edge": zeros,
        }
        specialist_signal_surfaces = {
            "fallback_generalist": {
                "event_signals": zeros,
                "continuous_signals": zeros,
                "signals": zeros.astype(int),
                "position_size": zeros,
                "meta_prob": zeros,
                "profitability_prob": zeros,
                "direction_edge": zeros,
                "confidence": zeros,
                "expected_trade_edge": zeros,
            },
            "specialist::bull": {
                "event_signals": ones,
                "continuous_signals": ones,
                "signals": ones.astype(int),
                "position_size": ones,
                "meta_prob": ones,
                "profitability_prob": ones,
                "direction_edge": ones,
                "confidence": ones,
                "expected_trade_edge": ones,
            },
        }
        specialist_library = SpecialistLibrarySnapshot(
            symbol="BTCUSDT",
            timeframe="1h",
            fallback_model_id="fallback_generalist",
            specialists=[
                SpecialistSpec(
                    model_id="fallback_generalist",
                    symbol="BTCUSDT",
                    timeframe="1h",
                    compatible_regimes=[],
                    estimator_family="dummy",
                    metadata={"fallback_only": True, "lifecycle_state": "active"},
                ),
                SpecialistSpec(
                    model_id="specialist::bull",
                    symbol="BTCUSDT",
                    timeframe="1h",
                    compatible_regimes=["bull"],
                    estimator_family="dummy",
                    metadata={"lifecycle_state": "active"},
                ),
            ],
            health=[
                SpecialistHealthContract(
                    model_id="fallback_generalist",
                    compatible_regimes=[],
                    fallback_only=True,
                    metadata={"health_binding_resolved": True, "health_state": "fallback_only"},
                )
            ],
        ).to_dict()
        regime_states = [
            RegimeStateContract(as_of=timestamp, available_at=timestamp, label="bull", confidence=0.95)
            for timestamp in index
        ]

        routed_signal_state, router_trace_summary = _route_signal_state_with_router(
            base_signal_state,
            specialist_signal_surfaces,
            router=HardSwitchRouter(execution_policy="fallback_only_until_health_binding"),
            specialist_library=specialist_library,
            regime_states=regime_states,
        )

        self.assertListEqual(routed_signal_state["continuous_signals"].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(router_trace_summary["summary"]["eligibility_blocked_rows"], 3)
        self.assertEqual(router_trace_summary["summary"]["fallback_rows_by_cause"], {"health_unbound": 3})


if __name__ == "__main__":
    unittest.main()
