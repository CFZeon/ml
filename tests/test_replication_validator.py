from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import core.automl as automl_module


def _build_market_frame(rows, *, start=100.0):
    index = pd.date_range("2026-02-01", periods=rows, freq="1h", tz="UTC")
    values = np.linspace(float(start), float(start) + 10.0, rows)
    return pd.DataFrame(
        {
            "open": values,
            "high": values + 0.5,
            "low": values - 0.5,
            "close": values,
            "volume": 1_000.0,
            "quote_volume": values * 100.0,
            "trades": 100,
        },
        index=index,
    )


def _make_training_backtest(state_bundle, returns):
    returns = np.asarray(returns, dtype=float)
    index = state_bundle["raw_data"].index[-len(returns) :]
    equity_curve = pd.Series(10_000.0 * np.cumprod(1.0 + returns), index=index)
    training = {
        "avg_accuracy": 0.55,
        "avg_f1_macro": 0.55,
        "avg_directional_accuracy": 0.55,
        "avg_directional_f1_macro": 0.55,
        "avg_log_loss": 0.2,
        "avg_brier_score": 0.1,
        "avg_calibration_error": 0.05,
        "headline_metrics": {"directional_accuracy": 0.55},
        "feature_selection": {
            "enabled": False,
            "avg_input_features": 1,
            "avg_selected_features": 1,
        },
        "bootstrap": {
            "model_type": "gbm",
            "used_in_any_fold": False,
            "warning_count": 0,
            "folds": [],
        },
        "feature_governance": {},
        "operational_monitoring": {},
        "cross_venue_integrity": {},
        "signal_decay": {},
        "promotion_gates": {},
        "fold_metrics": [{"accuracy": 0.55, "f1_macro": 0.55}],
    }
    backtest = {
        "equity_curve": equity_curve,
        "net_profit_pct": float(np.prod(1.0 + returns) - 1.0),
        "sharpe_ratio": float(returns.mean() / max(returns.std(ddof=1), 1e-12)),
        "max_drawdown": -0.02,
        "total_trades": int(len(returns)),
        "turnover_ratio": 0.0,
    }
    return training, backtest


class ReplicationValidatorTest(unittest.TestCase):
    def test_replication_requires_coverage_and_pass_rate(self):
        raw = _build_market_frame(120, start=100.0)
        eth = _build_market_frame(120, start=50.0)
        full_state_bundle = {
            "raw_data": raw,
            "data": raw.copy(),
            "indicator_run": None,
            "futures_context": None,
            "cross_asset_context": {"ETHUSDT": eth},
            "data_lineage": {},
            "symbol_filters": {},
            "symbol_lifecycle": None,
            "universe_policy": {},
            "universe_snapshot": None,
            "universe_snapshot_meta": {},
            "eligible_symbols": ["BTCUSDT", "ETHUSDT"],
            "universe_report": {},
        }
        base_pipeline = SimpleNamespace(state={"cross_asset_context": {"ETHUSDT": eth}, "eligible_symbols": ["BTCUSDT", "ETHUSDT"]})
        base_config = {
            "data": {"symbol": "BTCUSDT", "interval": "1h"},
            "automl": {
                "objective": "risk_adjusted_after_costs",
                "replication": {
                    "enabled": True,
                    "alternate_window_count": 1,
                    "alternate_window_fraction": 0.5,
                    "min_coverage": 2,
                    "min_pass_rate": 1.0,
                    "min_score": 0.05,
                    "min_rows": 24,
                },
            },
        }
        holdout_plan = {"enabled": True, "search_end_timestamp": raw.index[79]}

        def _side_effect(_base_config, overrides, _pipeline_class, _trial_step_classes, state_bundle):
            symbol = str(((overrides or {}).get("data") or {}).get("symbol", "BTCUSDT"))
            if symbol == "ETHUSDT":
                return _make_training_backtest(state_bundle, [0.010, 0.011, 0.012, 0.010, 0.011])
            return _make_training_backtest(state_bundle, [0.001, -0.001, 0.001, -0.001, 0.001])

        with mock.patch("core.automl._execute_trial_candidate", side_effect=_side_effect):
            report = automl_module._evaluate_replication_cohorts(
                base_config=base_config,
                best_overrides={"model": {"type": "gbm"}},
                pipeline_class=object,
                trial_step_classes=[],
                full_state_bundle=full_state_bundle,
                holdout_plan=holdout_plan,
                base_pipeline=base_pipeline,
            )

        self.assertTrue(report["enabled"])
        self.assertEqual(report["requested_cohort_count"], 2)
        self.assertEqual(report["completed_cohort_count"], 2)
        self.assertEqual(report["pass_count"], 1)
        self.assertAlmostEqual(float(report["pass_rate"]), 0.5)
        self.assertFalse(report["promotion_pass"])
        self.assertIn("replication_pass_rate_below_minimum", report["reasons"])
        self.assertEqual(report["summary_by_kind"]["symbol"]["passed"], 1)
        self.assertEqual(report["summary_by_kind"]["period"]["passed"], 0)


if __name__ == "__main__":
    unittest.main()