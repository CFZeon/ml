import unittest

import pandas as pd

from core import DepthCurveImpactModel, FillAwareCostModel, FlatSlippageModel, ProxyImpactModel
from core.slippage import _estimate_fill_event_costs


class MicrostructureCostModelTest(unittest.TestCase):
    def test_proxy_cost_increases_with_participation_spread_and_volatility(self):
        index = pd.date_range("2024-01-01", periods=3, freq="h")
        trade_notional = pd.Series([100.0, 20_000.0, 20_000.0], index=index)
        volume = pd.Series([1_000.0, 1_000.0, 1_000.0], index=index)
        price = pd.Series([100.0, 100.0, 100.0], index=index)
        volatility = pd.Series([0.01, 0.01, 0.05], index=index)

        low_spread = ProxyImpactModel(spread_bps=1.0, impact_bps=5.0, volatility_multiplier=1.0, adverse_selection_bps=1.0)
        high_spread = ProxyImpactModel(spread_bps=4.0, impact_bps=5.0, volatility_multiplier=1.0, adverse_selection_bps=1.0)

        low_costs = low_spread.estimate(trade_notional=trade_notional, volume=volume, volatility=volatility, price=price)
        high_costs = high_spread.estimate(trade_notional=trade_notional, volume=volume, volatility=volatility, price=price)

        self.assertLess(float(low_costs.iloc[0]), float(low_costs.iloc[1]))
        self.assertLess(float(low_costs.iloc[1]), float(low_costs.iloc[2]))
        self.assertLess(float(low_costs.iloc[1]), float(high_costs.iloc[1]))

    def test_l2_cost_reacts_to_thinner_depth_and_adverse_imbalance(self):
        index = pd.date_range("2024-01-01", periods=2, freq="h")
        trade_notional = pd.Series([2_000.0, 2_000.0], index=index)
        volume = pd.Series([5_000.0, 5_000.0], index=index)
        price = pd.Series([100.0, 100.0], index=index)
        volatility = pd.Series([0.01, 0.01], index=index)
        orderbook_depth = pd.DataFrame(
            {
                "bid_size_1": [500.0, 50.0],
                "ask_size_1": [500.0, 10.0],
                "bid_size_2": [500.0, 25.0],
                "ask_size_2": [500.0, 5.0],
                "best_bid": [99.9, 99.8],
                "best_ask": [100.1, 100.4],
                "snapshot_age_seconds": [0.1, 4.0],
                "queue_proxy": [0.0, 0.8],
            },
            index=index,
        )

        model = DepthCurveImpactModel()
        costs = model.estimate(
            trade_notional=trade_notional,
            volume=volume,
            volatility=volatility,
            price=price,
            orderbook_depth=orderbook_depth,
        )

        self.assertLess(float(costs.iloc[0]), float(costs.iloc[1]))

    def test_fill_aware_realized_cost_matches_fill_event_sum(self):
        index = pd.date_range("2024-01-01", periods=2, freq="h")
        order_ledger = pd.DataFrame(
            {
                "timestamp": index,
                "status": ["accepted", "partial_fill"],
                "executed_notional": [1_000.0, 500.0],
                "execution_price": [100.0, 100.0],
            }
        )
        execution_series = pd.Series([100.0, 100.0], index=index)
        volume = pd.Series([10_000.0, 10_000.0], index=index)

        report = _estimate_fill_event_costs(
            order_ledger=order_ledger,
            execution_series=execution_series,
            slippage_rate=0.0,
            slippage_model=FillAwareCostModel(base_model=FlatSlippageModel(rate=0.002)),
            volume=volume,
        )

        self.assertAlmostEqual(float(report["total_cost"]), 3.0, places=6)
        self.assertAlmostEqual(float(report["event_costs"]["cost"].sum()), 3.0, places=6)

    def test_cost_stress_sweeps_deteriorate_monotonically(self):
        index = pd.date_range("2024-01-01", periods=2, freq="h")
        order_ledger = pd.DataFrame(
            {
                "timestamp": index,
                "status": ["accepted", "accepted"],
                "executed_notional": [2_000.0, 2_000.0],
                "execution_price": [100.0, 100.0],
            }
        )
        execution_series = pd.Series([100.0, 100.0], index=index)
        volume = pd.Series([5_000.0, 5_000.0], index=index)

        report = _estimate_fill_event_costs(
            order_ledger=order_ledger,
            execution_series=execution_series,
            slippage_rate=0.0,
            slippage_model=ProxyImpactModel(),
            volume=volume,
        )

        stressed_costs = [report["stress_scenarios"][key] for key in ["1.0", "1.25", "1.5"]]
        self.assertLessEqual(stressed_costs[0], stressed_costs[1])
        self.assertLessEqual(stressed_costs[1], stressed_costs[2])


if __name__ == "__main__":
    unittest.main()