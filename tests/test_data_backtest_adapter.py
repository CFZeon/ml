import unittest

import pandas as pd

from core.data import (
    _COLUMNS,
    _normalize_futures_leverage_brackets,
    _parse_futures_contract_spec,
    _parse_symbol_filters,
    _prepare_frame,
)
from core import FlatSlippageModel, SquareRootImpactModel, join_custom_data, run_backtest
from core.slippage import OrderBookImpactModel


class DataBacktestAdapterTest(unittest.TestCase):
    def test_prepare_frame_accepts_header_row_archives(self):
        raw = pd.DataFrame(
            [
                _COLUMNS,
                [1704067200000, "100", "101", "99", "100.5", "10", 1704070799999, "1000", "12", "4", "400", "0"],
            ],
            columns=_COLUMNS,
        )

        prepared = _prepare_frame(raw)

        self.assertEqual(len(prepared), 1)
        self.assertAlmostEqual(float(prepared.iloc[0]["close"]), 100.5, places=6)

    def test_prepare_frame_backfills_missing_taker_columns(self):
        raw = pd.DataFrame(
            [
                [1704067200000, "100", "101", "99", "100.5", "10", 1704070799999, "1000", "12"],
            ],
            columns=_COLUMNS[:9],
        )

        prepared = _prepare_frame(raw)

        self.assertIn("taker_buy_base_vol", prepared.columns)
        self.assertIn("taker_buy_quote_vol", prepared.columns)
        self.assertAlmostEqual(float(prepared.iloc[0]["taker_buy_base_vol"]), 0.0, places=6)
        self.assertAlmostEqual(float(prepared.iloc[0]["taker_buy_quote_vol"]), 0.0, places=6)

    def test_parse_symbol_filters_preserves_extended_binance_rules(self):
        payload = {
            "symbol": "BTCUSDT",
            "filters": [
                {"filterType": "PRICE_FILTER", "minPrice": "10", "maxPrice": "1000000", "tickSize": "0.01"},
                {"filterType": "LOT_SIZE", "minQty": "0.001", "maxQty": "50", "stepSize": "0.001"},
                {"filterType": "MARKET_LOT_SIZE", "minQty": "0.01", "maxQty": "25", "stepSize": "0.01"},
                {"filterType": "MIN_NOTIONAL", "minNotional": "10", "applyToMarket": True, "avgPriceMins": 5},
                {
                    "filterType": "NOTIONAL",
                    "minNotional": "12",
                    "maxNotional": "1000",
                    "applyMinToMarket": False,
                    "applyMaxToMarket": True,
                    "avgPriceMins": 1,
                },
                {"filterType": "PERCENT_PRICE", "multiplierUp": "1.2", "multiplierDown": "0.8", "avgPriceMins": 5},
                {
                    "filterType": "PERCENT_PRICE_BY_SIDE",
                    "bidMultiplierUp": "1.1",
                    "bidMultiplierDown": "0.9",
                    "askMultiplierUp": "1.3",
                    "askMultiplierDown": "0.7",
                    "avgPriceMins": 1,
                },
                {"filterType": "MAX_POSITION", "maxPosition": "5"},
                {"filterType": "TRAILING_DELTA", "minTrailingAboveDelta": 10, "maxTrailingAboveDelta": 2000},
            ],
        }

        parsed = _parse_symbol_filters(payload)

        self.assertAlmostEqual(float(parsed["tick_size"]), 0.01, places=6)
        self.assertAlmostEqual(float(parsed["market_step_size"]), 0.01, places=6)
        self.assertTrue(parsed["min_notional_apply_to_market"])
        self.assertFalse(parsed["notional_apply_min_to_market"])
        self.assertAlmostEqual(float(parsed["max_notional"]), 1000.0, places=6)
        self.assertAlmostEqual(float(parsed["percent_price"]["multiplier_up"]), 1.2, places=6)
        self.assertAlmostEqual(float(parsed["percent_price_by_side"]["ask_multiplier_down"]), 0.7, places=6)
        self.assertAlmostEqual(float(parsed["max_position"]), 5.0, places=6)
        self.assertIn("TRAILING_DELTA", parsed["unsupported_filters"])
        self.assertIn("LOT_SIZE", parsed["raw_filters"])

    def test_parse_futures_contract_spec_extracts_liquidation_metadata(self):
        payload = {
            "symbol": "BTCUSDT",
            "pair": "BTCUSDT",
            "contractType": "PERPETUAL",
            "baseAsset": "BTC",
            "quoteAsset": "USDT",
            "marginAsset": "USDT",
            "status": "TRADING",
            "contractSize": "1",
            "liquidationFee": "0.0125",
            "marketTakeBound": "0.30",
            "triggerProtect": "0.15",
            "pricePrecision": 2,
            "quantityPrecision": 3,
            "onboardDate": 1704067200000,
            "deliveryDate": 4133404800000,
        }

        spec = _parse_futures_contract_spec(payload, market="um_futures")

        self.assertEqual(spec["symbol"], "BTCUSDT")
        self.assertEqual(spec["market"], "um_futures")
        self.assertEqual(spec["margin_asset"], "USDT")
        self.assertAlmostEqual(float(spec["contract_size"]), 1.0, places=6)
        self.assertAlmostEqual(float(spec["liquidation_fee_rate"]), 0.0125, places=6)
        self.assertAlmostEqual(float(spec["market_take_bound"]), 0.30, places=6)
        self.assertIsNotNone(spec["onboard_date"])
        self.assertIsNotNone(spec["delivery_date"])

    def test_normalize_futures_leverage_brackets_accepts_binance_shape(self):
        payload = {
            "symbol": "BTCUSDT",
            "notionalCoef": 1.0,
            "brackets": [
                {
                    "bracket": 1,
                    "initialLeverage": 75,
                    "notionalFloor": 0,
                    "notionalCap": 10000,
                    "maintMarginRatio": 0.0065,
                    "cum": 0,
                },
                {
                    "bracket": 2,
                    "initialLeverage": 50,
                    "notionalFloor": 10000,
                    "notionalCap": 50000,
                    "maintMarginRatio": 0.01,
                    "cum": 65,
                },
            ],
        }

        normalized = _normalize_futures_leverage_brackets(payload, symbol="BTCUSDT", market="um_futures")

        self.assertEqual(normalized["symbol"], "BTCUSDT")
        self.assertEqual(len(normalized["brackets"]), 2)
        self.assertAlmostEqual(float(normalized["brackets"][0]["initial_leverage"]), 75.0, places=6)
        self.assertAlmostEqual(float(normalized["brackets"][1]["maint_margin_ratio"]), 0.01, places=6)

    def test_slippage_sqrt_model_increases_with_volume_ratio(self):
        index = pd.date_range("2026-03-09", periods=4, freq="1h", tz="UTC")
        model = SquareRootImpactModel(adv_window=2)

        rates = model.estimate(
            trade_notional=pd.Series([0.0, 250.0, 5_000.0, 10_000.0], index=index),
            volume=pd.Series(1_000.0, index=index),
            volatility=pd.Series(20.0, index=index),
            price=pd.Series(100.0, index=index),
        )

        self.assertAlmostEqual(float(rates.iloc[0]), 0.0, places=6)
        self.assertGreater(float(rates.iloc[2]), float(rates.iloc[1]))
        self.assertGreater(float(rates.iloc[3]), float(rates.iloc[2]))

    def test_slippage_flat_model_matches_legacy_behavior(self):
        index = pd.date_range("2026-03-10", periods=6, freq="1h", tz="UTC")
        close = pd.Series([100.0, 101.0, 103.0, 102.0, 104.0, 103.0], index=index)
        signals = pd.Series([0.0, 1.0, 1.0, 0.0, -1.0, 0.0], index=index)
        volume = pd.Series(1_000.0, index=index)

        legacy = run_backtest(
            close=close,
            signals=signals,
            equity=10_000.0,
            fee_rate=0.001,
            slippage_rate=0.0005,
            signal_delay_bars=0,
            engine="pandas",
            volume=volume,
        )
        explicit = run_backtest(
            close=close,
            signals=signals,
            equity=10_000.0,
            fee_rate=0.001,
            slippage_rate=0.0,
            signal_delay_bars=0,
            engine="pandas",
            volume=volume,
            slippage_model=FlatSlippageModel(0.0005),
        )

        self.assertAlmostEqual(float(legacy["ending_equity"]), float(explicit["ending_equity"]), places=6)
        self.assertAlmostEqual(float(legacy["net_profit_pct"]), float(explicit["net_profit_pct"]), places=6)
        self.assertAlmostEqual(float(legacy["slippage_paid"]), float(explicit["slippage_paid"]), places=6)

    def test_slippage_orderbook_raises_not_implemented(self):
        index = pd.date_range("2026-03-11", periods=4, freq="1h", tz="UTC")
        close = pd.Series([100.0, 101.0, 100.5, 101.5], index=index)
        signals = pd.Series([0.0, 1.0, 0.0, 0.0], index=index)
        volume = pd.Series(500.0, index=index)

        with self.assertRaisesRegex(NotImplementedError, "L2 data adapter not yet available"):
            run_backtest(
                close=close,
                signals=signals,
                equity=10_000.0,
                fee_rate=0.0,
                slippage_rate=0.0,
                signal_delay_bars=0,
                engine="pandas",
                volume=volume,
                slippage_model=OrderBookImpactModel(),
            )

    def test_point_in_time_custom_join_uses_availability_timestamp(self):
        index = pd.date_range("2026-03-10", periods=4, freq="1h", tz="UTC")
        base = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0],
                "high": [101.0, 102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0, 102.0],
                "close": [100.5, 101.5, 102.5, 103.5],
                "volume": [10.0, 11.0, 12.0, 13.0],
                "quote_volume": [1005.0, 1116.5, 1230.0, 1345.5],
                "trades": [100, 101, 102, 103],
            },
            index=index,
        )
        custom_frame = pd.DataFrame(
            {
                "timestamp": [index[0], index[1]],
                "available_at": [index[0] + pd.Timedelta(minutes=30), index[1] + pd.Timedelta(minutes=30)],
                "sentiment": [0.25, 0.75],
            }
        )

        joined, reports = join_custom_data(
            base,
            [
                {
                    "name": "sentiment_feed",
                    "frame": custom_frame,
                    "timestamp_column": "timestamp",
                    "availability_column": "available_at",
                    "value_columns": ["sentiment"],
                    "prefix": "sent",
                    "max_feature_age": "45m",
                }
            ],
        )

        self.assertTrue(pd.isna(joined.loc[index[0], "sent_sentiment"]))
        self.assertAlmostEqual(float(joined.loc[index[1], "sent_sentiment"]), 0.25, places=6)
        self.assertAlmostEqual(float(joined.loc[index[2], "sent_sentiment"]), 0.75, places=6)
        self.assertTrue(pd.isna(joined.loc[index[3], "sent_sentiment"]))
        self.assertEqual(reports[0]["joined_columns"], ["sent_sentiment"])
        self.assertEqual(int(reports[0]["stale_hit_count"]), 1)
        self.assertEqual(reports[0]["max_feature_age"], pd.Timedelta("45m"))
        self.assertEqual(reports[0]["median_feature_age"], pd.Timedelta("30m"))
        self.assertFalse(reports[0]["fallback_assumption_used"])

    def test_custom_join_requires_explicit_availability_or_opt_in(self):
        index = pd.date_range("2026-03-10", periods=2, freq="1h", tz="UTC")
        base = pd.DataFrame(
            {"close": [100.0, 101.0]},
            index=index,
        )
        custom_frame = pd.DataFrame(
            {
                "timestamp": [index[0]],
                "sentiment": [0.25],
            }
        )

        with self.assertRaisesRegex(ValueError, "availability_column"):
            join_custom_data(
                base,
                [
                    {
                        "name": "sentiment_feed",
                        "frame": custom_frame,
                        "timestamp_column": "timestamp",
                        "value_columns": ["sentiment"],
                        "prefix": "sent",
                    }
                ],
            )

    def test_custom_join_requires_explicit_value_columns(self):
        index = pd.date_range("2026-03-10", periods=2, freq="1h", tz="UTC")
        base = pd.DataFrame({"close": [100.0, 101.0]}, index=index)
        custom_frame = pd.DataFrame(
            {
                "timestamp": [index[0]],
                "available_at": [index[0] + pd.Timedelta(minutes=30)],
                "sentiment": [0.25],
            }
        )

        with self.assertRaisesRegex(ValueError, "value_columns"):
            join_custom_data(
                base,
                [
                    {
                        "name": "sentiment_feed",
                        "frame": custom_frame,
                        "timestamp_column": "timestamp",
                        "availability_column": "available_at",
                        "prefix": "sent",
                    }
                ],
            )

    def test_exact_match_requires_explicit_opt_in(self):
        index = pd.date_range("2026-03-10", periods=1, freq="1h", tz="UTC")
        base = pd.DataFrame(
            {"close": [100.0]},
            index=index,
        )
        custom_frame = pd.DataFrame(
            {
                "timestamp": [index[0]],
                "sentiment": [0.25],
            }
        )

        assumed_joined, assumed_report = join_custom_data(
            base,
            [
                {
                    "name": "assumed_feed",
                    "frame": custom_frame,
                    "timestamp_column": "timestamp",
                    "assume_event_time_is_available_time": True,
                    "value_columns": ["sentiment"],
                    "prefix": "assumed",
                }
            ],
        )
        explicit_joined, explicit_report = join_custom_data(
            base,
            [
                {
                    "name": "explicit_feed",
                    "frame": custom_frame.assign(available_at=custom_frame["timestamp"]),
                    "timestamp_column": "timestamp",
                    "availability_column": "available_at",
                    "value_columns": ["sentiment"],
                    "prefix": "explicit",
                }
            ],
        )
        opted_in_joined, opted_in_report = join_custom_data(
            base,
            [
                {
                    "name": "opted_in_feed",
                    "frame": custom_frame.assign(available_at=custom_frame["timestamp"]),
                    "timestamp_column": "timestamp",
                    "availability_column": "available_at",
                    "value_columns": ["sentiment"],
                    "prefix": "opted_in",
                    "allow_exact_matches": True,
                }
            ],
        )

        self.assertTrue(pd.isna(assumed_joined.loc[index[0], "assumed_sentiment"]))
        self.assertFalse(assumed_report[0]["allow_exact_matches"])
        self.assertTrue(assumed_report[0]["fallback_assumption_used"])
        self.assertEqual(int(assumed_report[0]["exact_match_count"]), 0)
        self.assertTrue(pd.isna(explicit_joined.loc[index[0], "explicit_sentiment"]))
        self.assertFalse(explicit_report[0]["allow_exact_matches"])
        self.assertFalse(explicit_report[0]["fallback_assumption_used"])
        self.assertEqual(int(explicit_report[0]["exact_match_count"]), 0)
        self.assertAlmostEqual(float(opted_in_joined.loc[index[0], "opted_in_sentiment"]), 0.25, places=6)
        self.assertTrue(opted_in_report[0]["allow_exact_matches"])
        self.assertEqual(int(opted_in_report[0]["exact_match_count"]), 1)

    def test_vectorbt_backtest_adapter_supports_futures_execution_inputs(self):
        index = pd.date_range("2026-03-12", periods=6, freq="1h", tz="UTC")
        close = pd.Series([100.0, 101.0, 103.0, 102.0, 100.0, 99.0], index=index)
        signals = pd.Series([0.0, 0.5, 0.5, -0.5, -0.5, 0.0], index=index)
        funding_rates = pd.Series([0.0, 0.0, 0.0005, 0.0, -0.0003, 0.0], index=index)

        result = run_backtest(
            close=close,
            signals=signals,
            equity=10_000.0,
            fee_rate=0.001,
            slippage_rate=0.0005,
            execution_prices=close,
            signal_delay_bars=0,
            engine="vectorbt",
            market="um_futures",
            allow_short=True,
            leverage=1.0,
            funding_rates=funding_rates,
            symbol_filters={"tick_size": 0.1, "step_size": 0.001, "min_notional": 10.0},
        )

        self.assertEqual(result["engine"], "vectorbt")
        self.assertGreater(result["closed_trades"], 0)
        self.assertFalse(result["trade_ledger"].empty)
        self.assertIn("funding_pnl", result)
        self.assertIn("slippage_paid", result)
        self.assertIn("statistical_significance", result)
        self.assertFalse(result["statistical_significance"]["enabled"])

    def test_futures_account_liquidation_triggers_on_large_adverse_move(self):
        index = pd.date_range("2026-03-12", periods=5, freq="1h", tz="UTC")
        close = pd.Series([100.0, 100.0, 60.0, 60.0, 60.0], index=index)
        signals = pd.Series([0.5, 0.5, 0.5, 0.0, 0.0], index=index)

        result = run_backtest(
            close=close,
            signals=signals,
            equity=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            signal_delay_bars=0,
            engine="pandas",
            market="um_futures",
            allow_short=True,
            leverage=3.0,
            futures_account={
                "enabled": True,
                "margin_mode": "isolated",
                "warning_margin_ratio": 0.8,
                "liquidation_fee_rate": 0.01,
                "leverage_brackets": {
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
        )

        self.assertEqual(result["account_model"], "futures_margin")
        self.assertEqual(result["futures_margin_mode"], "isolated")
        self.assertGreater(int(result["liquidation_event_count"]), 0)
        self.assertFalse(result["liquidation_events"].empty)
        self.assertIn("margin_ratio_series", result)
        self.assertIn("position_notional_series", result)
        self.assertGreater(float(result["max_margin_ratio"]), 1.0)
        self.assertGreater(int(result["bars_above_margin_warning"]), 0)
        self.assertGreater(float(result["liquidation_fee_paid"]), 0.0)

    def test_futures_account_isolated_and_cross_produce_different_outcomes(self):
        index = pd.date_range("2026-03-12", periods=5, freq="1h", tz="UTC")
        close = pd.Series([100.0, 100.0, 60.0, 60.0, 60.0], index=index)
        signals = pd.Series([0.5, 0.5, 0.5, 0.0, 0.0], index=index)
        futures_account = {
            "enabled": True,
            "warning_margin_ratio": 0.8,
            "liquidation_fee_rate": 0.01,
            "leverage_brackets": {
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
        }

        isolated = run_backtest(
            close=close,
            signals=signals,
            equity=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            signal_delay_bars=0,
            engine="pandas",
            market="um_futures",
            allow_short=True,
            leverage=3.0,
            futures_account={**futures_account, "margin_mode": "isolated"},
        )
        cross = run_backtest(
            close=close,
            signals=signals,
            equity=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            signal_delay_bars=0,
            engine="pandas",
            market="um_futures",
            allow_short=True,
            leverage=3.0,
            futures_account={**futures_account, "margin_mode": "cross"},
        )

        self.assertGreater(int(isolated["liquidation_event_count"]), 0)
        self.assertEqual(int(cross["liquidation_event_count"]), 0)
        self.assertGreater(float(isolated["ending_equity"]), float(cross["ending_equity"]))
        self.assertGreater(float(cross["max_margin_ratio"]), 0.0)

    def test_execution_parity_rejects_same_min_notional_order_in_both_engines(self):
        index = pd.date_range("2026-03-12", periods=5, freq="1h", tz="UTC")
        close = pd.Series([100.0, 100.5, 101.0, 101.5, 102.0], index=index)
        signals = pd.Series([0.0, 0.05, 0.05, 0.0, 0.0], index=index)
        filters = {"tick_size": 0.1, "step_size": 0.01, "min_notional": 1_000.0}

        pandas_result = run_backtest(
            close=close,
            signals=signals,
            equity=1_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            signal_delay_bars=0,
            engine="pandas",
            symbol_filters=filters,
        )
        vectorbt_result = run_backtest(
            close=close,
            signals=signals,
            equity=1_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            signal_delay_bars=0,
            engine="vectorbt",
            symbol_filters=filters,
        )

        self.assertEqual(pandas_result["blocked_orders"], 2)
        self.assertEqual(vectorbt_result["blocked_orders"], 2)
        self.assertEqual(pandas_result["order_rejection_reasons"], {"min_notional": 2})
        self.assertEqual(vectorbt_result["order_rejection_reasons"], {"min_notional": 2})
        self.assertAlmostEqual(float(pandas_result["ending_equity"]), 1_000.0, places=6)
        self.assertAlmostEqual(float(vectorbt_result["ending_equity"]), 1_000.0, places=6)
        self.assertEqual(int(pandas_result["closed_trades"]), 0)
        self.assertEqual(int(vectorbt_result["closed_trades"]), 0)

    def test_execution_parity_matches_flip_trade_counts(self):
        index = pd.date_range("2026-03-12", periods=6, freq="1h", tz="UTC")
        close = pd.Series([100.0, 102.0, 101.0, 99.0, 98.0, 100.0], index=index)
        signals = pd.Series([0.0, 1.0, 1.0, -1.0, -1.0, 0.0], index=index)

        pandas_result = run_backtest(
            close=close,
            signals=signals,
            equity=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            signal_delay_bars=0,
            engine="pandas",
            allow_short=True,
        )
        vectorbt_result = run_backtest(
            close=close,
            signals=signals,
            equity=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            signal_delay_bars=0,
            engine="vectorbt",
            allow_short=True,
        )

        self.assertEqual(int(pandas_result["closed_trades"]), int(vectorbt_result["closed_trades"]))
        self.assertEqual(int(pandas_result["total_trades"]), int(vectorbt_result["total_trades"]))
        self.assertEqual(int(pandas_result["blocked_orders"]), 0)
        self.assertEqual(int(vectorbt_result["blocked_orders"]), 0)
        self.assertAlmostEqual(float(pandas_result["ending_equity"]), float(vectorbt_result["ending_equity"]), places=2)

    def test_execution_validator_rejects_percent_price_in_both_engines(self):
        index = pd.date_range("2026-03-12", periods=4, freq="1h", tz="UTC")
        close = pd.Series([100.0, 130.0, 130.0, 130.0], index=index)
        signals = pd.Series([0.0, 1.0, 0.0, 0.0], index=index)
        filters = {
            "percent_price": {"multiplier_up": 1.1, "multiplier_down": 0.9, "avg_price_mins": 0},
            "weighted_average_price": 100.0,
        }

        pandas_result = run_backtest(close=close, signals=signals, equity=1_000.0, fee_rate=0.0, slippage_rate=0.0, signal_delay_bars=0, engine="pandas", symbol_filters=filters)
        vectorbt_result = run_backtest(close=close, signals=signals, equity=1_000.0, fee_rate=0.0, slippage_rate=0.0, signal_delay_bars=0, engine="vectorbt", symbol_filters=filters)

        self.assertEqual(pandas_result["order_rejection_reasons"], {"percent_price": 1})
        self.assertEqual(vectorbt_result["order_rejection_reasons"], {"percent_price": 1})
        self.assertEqual(int(pandas_result["blocked_orders"]), 1)
        self.assertEqual(int(vectorbt_result["blocked_orders"]), 1)
        self.assertGreater(float(pandas_result["blocked_notional_share"]), 0.0)
        self.assertGreater(float(vectorbt_result["blocked_notional_share"]), 0.0)

    def test_execution_validator_applies_market_lot_size_before_both_engines(self):
        index = pd.date_range("2026-03-12", periods=4, freq="1h", tz="UTC")
        close = pd.Series([100.0, 100.0, 100.0, 100.0], index=index)
        signals = pd.Series([0.0, 0.67, 0.0, 0.0], index=index)
        filters = {"market_min_qty": 1.0, "market_max_qty": 5.0, "market_step_size": 1.0}

        pandas_result = run_backtest(close=close, signals=signals, equity=1_000.0, fee_rate=0.0, slippage_rate=0.0, signal_delay_bars=0, engine="pandas", symbol_filters=filters)
        vectorbt_result = run_backtest(close=close, signals=signals, equity=1_000.0, fee_rate=0.0, slippage_rate=0.0, signal_delay_bars=0, engine="vectorbt", symbol_filters=filters)

        self.assertEqual(int(pandas_result["adjusted_orders"]), 1)
        self.assertEqual(int(vectorbt_result["adjusted_orders"]), 1)
        self.assertEqual(str(pandas_result["order_ledger"].iloc[0]["reason"]), "market_lot_size")
        self.assertEqual(str(vectorbt_result["order_ledger"].iloc[0]["reason"]), "market_lot_size")
        self.assertAlmostEqual(float(pandas_result["order_ledger"].iloc[0]["executed_position"]), 0.5, places=6)
        self.assertAlmostEqual(float(vectorbt_result["order_ledger"].iloc[0]["executed_position"]), 0.5, places=6)

    def test_execution_validator_applies_notional_cap_before_both_engines(self):
        index = pd.date_range("2026-03-12", periods=4, freq="1h", tz="UTC")
        close = pd.Series([100.0, 100.0, 100.0, 100.0], index=index)
        signals = pd.Series([0.0, 1.0, 0.0, 0.0], index=index)
        filters = {"max_notional": 500.0, "notional_apply_max_to_market": True}

        pandas_result = run_backtest(close=close, signals=signals, equity=1_000.0, fee_rate=0.0, slippage_rate=0.0, signal_delay_bars=0, engine="pandas", symbol_filters=filters)
        vectorbt_result = run_backtest(close=close, signals=signals, equity=1_000.0, fee_rate=0.0, slippage_rate=0.0, signal_delay_bars=0, engine="vectorbt", symbol_filters=filters)

        self.assertEqual(int(pandas_result["adjusted_orders"]), 1)
        self.assertEqual(int(vectorbt_result["adjusted_orders"]), 1)
        self.assertEqual(str(pandas_result["order_ledger"].iloc[0]["reason"]), "max_notional")
        self.assertEqual(str(vectorbt_result["order_ledger"].iloc[0]["reason"]), "max_notional")
        self.assertAlmostEqual(float(pandas_result["order_ledger"].iloc[0]["executed_position"]), 0.5, places=6)
        self.assertAlmostEqual(float(vectorbt_result["order_ledger"].iloc[0]["executed_position"]), 0.5, places=6)

    def test_execution_validator_applies_spot_max_position_before_both_engines(self):
        index = pd.date_range("2026-03-12", periods=5, freq="1h", tz="UTC")
        close = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=index)
        signals = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0], index=index)
        filters = {"max_position": 4.0, "step_size": 1.0, "max_qty": 10.0}

        pandas_result = run_backtest(close=close, signals=signals, equity=1_000.0, fee_rate=0.0, slippage_rate=0.0, signal_delay_bars=0, engine="pandas", market="spot", symbol_filters=filters)
        vectorbt_result = run_backtest(close=close, signals=signals, equity=1_000.0, fee_rate=0.0, slippage_rate=0.0, signal_delay_bars=0, engine="vectorbt", market="spot", symbol_filters=filters)

        self.assertAlmostEqual(float(pandas_result["order_ledger"].iloc[0]["executed_position"]), 0.4, places=6)
        self.assertAlmostEqual(float(vectorbt_result["order_ledger"].iloc[0]["executed_position"]), 0.4, places=6)
        self.assertEqual(pandas_result["order_rejection_reasons"], {"max_position": 1})
        self.assertEqual(vectorbt_result["order_rejection_reasons"], {"max_position": 1})
        self.assertEqual(int(pandas_result["blocked_orders"]), 1)
        self.assertEqual(int(vectorbt_result["blocked_orders"]), 1)

    def test_missing_leading_execution_prices_do_not_backfill_from_future(self):
        index = pd.date_range("2026-03-12", periods=4, freq="1h", tz="UTC")
        close = pd.Series([100.0, 101.0, 102.0, 103.0], index=index)
        execution_prices = pd.Series([float("nan"), 101.0, 102.0, 103.0], index=index, dtype="float")
        signals = pd.Series([0.0, 1.0, 0.0, 0.0], index=index)

        with self.assertRaisesRegex(ValueError, "execution price policy"):
            run_backtest(
                close=close,
                signals=signals,
                equity=10_000.0,
                fee_rate=0.0,
                slippage_rate=0.0,
                signal_delay_bars=0,
                engine="pandas",
                execution_prices=execution_prices,
                execution_price_policy="strict",
            )

    def test_capital_facing_backtest_requires_explicit_execution_prices(self):
        index = pd.date_range("2026-03-12", periods=4, freq="1h", tz="UTC")
        close = pd.Series([100.0, 101.0, 102.0, 103.0], index=index)
        signals = pd.Series([0.0, 1.0, 0.0, 0.0], index=index)

        with self.assertRaisesRegex(ValueError, "require explicit execution_prices"):
            run_backtest(
                close=close,
                signals=signals,
                equity=10_000.0,
                fee_rate=0.0,
                slippage_rate=0.0,
                signal_delay_bars=0,
                engine="pandas",
                evaluation_mode="local_certification",
            )

    def test_research_backtest_surfaces_same_bar_close_fallback_warning(self):
        index = pd.date_range("2026-03-12", periods=4, freq="1h", tz="UTC")
        close = pd.Series([100.0, 101.0, 102.0, 103.0], index=index)
        signals = pd.Series([0.0, 1.0, 0.0, 0.0], index=index)

        result = run_backtest(
            close=close,
            signals=signals,
            equity=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            signal_delay_bars=0,
            engine="pandas",
            evaluation_mode="research_only",
        )

        self.assertEqual(result["execution_price_source"], "close_fallback")
        self.assertTrue(result["same_bar_execution_fallback"])
        self.assertEqual(result["execution_price_warning"], "same_bar_execution_fallback")
        self.assertIn("same_bar_execution_fallback", result["backtest_warnings"])

    def test_execution_prices_can_only_forward_fill_causally_when_enabled(self):
        index = pd.date_range("2026-03-12", periods=5, freq="1h", tz="UTC")
        close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=index)
        execution_prices = pd.Series([100.0, 101.0, float("nan"), 103.0, 104.0], index=index, dtype="float")
        signals = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0], index=index)

        result = run_backtest(
            close=close,
            signals=signals,
            equity=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            signal_delay_bars=0,
            engine="pandas",
            execution_prices=execution_prices,
            execution_price_policy="ffill_with_limit",
            execution_price_fill_limit=1,
        )

        self.assertEqual(int(result["price_fill_actions"]["execution"]["forward_filled_rows"]), 1)
        self.assertEqual(int(result["price_fill_actions"]["execution"]["invalid_rows"]), 0)
        self.assertEqual(int(result["price_fill_actions"]["execution"]["leading_missing_rows"]), 0)

    def test_backtest_with_sqrt_slippage_produces_lower_returns(self):
        index = pd.date_range("2026-03-13", periods=10, freq="1h", tz="UTC")
        close = pd.Series([100.0, 103.0, 99.0, 104.0, 98.0, 105.0, 97.0, 106.0, 96.0, 107.0], index=index)
        signals = pd.Series([0.0, 1.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0], index=index)
        volume = pd.Series(5.0, index=index)

        baseline = run_backtest(
            close=close,
            signals=signals,
            equity=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            signal_delay_bars=0,
            engine="vectorbt",
            volume=volume,
        )
        impacted = run_backtest(
            close=close,
            signals=signals,
            equity=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            signal_delay_bars=0,
            engine="vectorbt",
            volume=volume,
            slippage_model="sqrt_impact",
        )

        self.assertLess(float(impacted["ending_equity"]), float(baseline["ending_equity"]))
        self.assertGreater(float(impacted["slippage_paid"]), float(baseline["slippage_paid"]))

    def test_backtest_reports_stationary_bootstrap_confidence_intervals(self):
        index = pd.date_range("2026-03-12", periods=24, freq="1h", tz="UTC")
        close = pd.Series(
            [100.0, 100.8, 101.4, 101.1, 102.0, 102.8, 103.3, 103.0, 103.8, 104.5, 104.1, 105.0,
             105.7, 105.2, 106.0, 106.8, 106.5, 107.1, 107.9, 107.6, 108.4, 109.0, 108.7, 109.5],
            index=index,
        )
        signals = pd.Series(1.0, index=index)
        benchmark_close = pd.Series(
            [100.0, 100.4, 100.9, 100.7, 101.1, 101.4, 101.8, 101.7, 102.0, 102.4, 102.2, 102.6,
             102.9, 102.8, 103.1, 103.5, 103.4, 103.7, 104.0, 103.9, 104.2, 104.5, 104.4, 104.8],
            index=index,
        )

        result = run_backtest(
            close=close,
            signals=signals,
            equity=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            signal_delay_bars=0,
            engine="pandas",
            significance={"bootstrap_samples": 96, "mean_block_length": 4, "random_state": 7},
            benchmark_returns=benchmark_close.pct_change().fillna(0.0),
        )

        significance = result["statistical_significance"]
        self.assertTrue(significance["enabled"])
        self.assertEqual(significance["method"], "stationary_bootstrap")
        self.assertEqual(int(significance["bootstrap_samples"]), 96)
        self.assertEqual(int(significance["mean_block_length"]), 4)
        self.assertIn("benchmark_sharpe_ratio", significance)

        sharpe_stats = significance["metrics"]["sharpe_ratio"]
        self.assertIn("confidence_interval", sharpe_stats)
        self.assertIn("p_value_gt_zero", sharpe_stats)
        self.assertIn("p_value_gt_benchmark", sharpe_stats)
        self.assertLessEqual(sharpe_stats["confidence_interval"]["lower"], sharpe_stats["confidence_interval"]["upper"])
        self.assertGreaterEqual(float(sharpe_stats["p_value_gt_zero"]), 0.0)
        self.assertLessEqual(float(sharpe_stats["p_value_gt_zero"]), 1.0)
        self.assertGreaterEqual(float(sharpe_stats["p_value_gt_benchmark"]), 0.0)
        self.assertLessEqual(float(sharpe_stats["p_value_gt_benchmark"]), 1.0)

        for key in ["sortino_ratio", "calmar_ratio", "net_profit_pct", "max_drawdown"]:
            metric = significance["metrics"][key]
            self.assertIn("confidence_interval", metric)
            self.assertLessEqual(metric["confidence_interval"]["lower"], metric["confidence_interval"]["upper"])

    def test_backtest_significance_handles_short_samples(self):
        index = pd.date_range("2026-03-12", periods=5, freq="1h", tz="UTC")
        close = pd.Series([100.0, 100.5, 101.0, 100.8, 101.2], index=index)
        signals = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0], index=index)

        result = run_backtest(
            close=close,
            signals=signals,
            equity=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            signal_delay_bars=0,
            engine="pandas",
        )

        significance = result["statistical_significance"]
        self.assertFalse(significance["enabled"])
        self.assertEqual(significance["reason"], "insufficient_observations")
        self.assertEqual(int(significance["observation_count"]), 5)
        self.assertEqual(int(significance["min_observations"]), 8)
        self.assertTrue(significance["underpowered"])


if __name__ == "__main__":
    unittest.main()