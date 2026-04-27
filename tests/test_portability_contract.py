from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import core.automl as automl_module


def _build_market_frame(rows, *, start=100.0):
    index = pd.date_range("2026-05-01", periods=rows, freq="1h", tz="UTC")
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
        "feature_selection": {"enabled": False, "avg_input_features": 1, "avg_selected_features": 1},
        "bootstrap": {"model_type": "gbm", "used_in_any_fold": False, "warning_count": 0, "folds": []},
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


class PortabilityContractTest(unittest.TestCase):
    def test_portability_contract_blocks_when_no_supporting_cohort_exists(self):
        raw = _build_market_frame(120)
        full_state_bundle = {
            "raw_data": raw,
            "data": raw.copy(),
            "indicator_run": None,
            "futures_context": None,
            "cross_asset_context": {},
            "data_lineage": {},
            "symbol_filters": {},
            "symbol_lifecycle": None,
            "universe_policy": {},
            "universe_snapshot": None,
            "universe_snapshot_meta": {},
            "eligible_symbols": ["BTCUSDT"],
            "universe_report": {},
        }
        base_config = {
            "data": {"symbol": "BTCUSDT", "interval": "1h"},
            "automl": {
                "objective": "risk_adjusted_after_costs",
                "replication": {
                    "enabled": True,
                    "include_symbol_cohorts": False,
                    "include_window_cohorts": False,
                    "min_coverage": 0,
                    "min_pass_rate": 0.0,
                    "min_rows": 24,
                },
                "portability_contract": {
                    "enabled": True,
                    "accepted_kinds": ["symbol", "period"],
                    "min_supporting_cohorts": 1,
                    "min_passed_supporting_cohorts": 1,
                },
            },
        }

        report = automl_module._evaluate_replication_cohorts(
            base_config=base_config,
            best_overrides={"model": {"type": "gbm"}},
            pipeline_class=object,
            trial_step_classes=[],
            full_state_bundle=full_state_bundle,
            holdout_plan={},
            base_pipeline=SimpleNamespace(state={}),
        )

        self.assertFalse(report["promotion_pass"])
        self.assertFalse(report["portability_contract"]["passed"])
        self.assertIn("portability_supporting_cohort_missing", report["reasons"])
        self.assertEqual(int(report["portability_contract"]["supporting_cohort_attempted_count"]), 0)

    def test_portability_contract_accepts_a_passing_period_cohort(self):
        raw = _build_market_frame(120)
        full_state_bundle = {
            "raw_data": raw,
            "data": raw.copy(),
            "indicator_run": None,
            "futures_context": None,
            "cross_asset_context": {},
            "data_lineage": {},
            "symbol_filters": {},
            "symbol_lifecycle": None,
            "universe_policy": {},
            "universe_snapshot": None,
            "universe_snapshot_meta": {},
            "eligible_symbols": ["BTCUSDT"],
            "universe_report": {},
        }
        base_config = {
            "data": {"symbol": "BTCUSDT", "interval": "1h"},
            "automl": {
                "objective": "risk_adjusted_after_costs",
                "replication": {
                    "enabled": True,
                    "include_symbol_cohorts": False,
                    "include_window_cohorts": True,
                    "alternate_window_count": 1,
                    "alternate_window_fraction": 0.5,
                    "min_coverage": 1,
                    "min_pass_rate": 1.0,
                    "min_score": 0.0,
                    "min_rows": 24,
                },
                "portability_contract": {
                    "enabled": True,
                    "accepted_kinds": ["period"],
                    "min_supporting_cohorts": 1,
                    "min_passed_supporting_cohorts": 1,
                },
            },
        }
        holdout_plan = {"enabled": True, "search_end_timestamp": raw.index[79]}

        with mock.patch(
            "core.automl._execute_trial_candidate",
            side_effect=lambda *_args, **_kwargs: _make_training_backtest(full_state_bundle, [0.01, 0.012, 0.011, 0.013]),
        ):
            report = automl_module._evaluate_replication_cohorts(
                base_config=base_config,
                best_overrides={"model": {"type": "gbm"}},
                pipeline_class=object,
                trial_step_classes=[],
                full_state_bundle=full_state_bundle,
                holdout_plan=holdout_plan,
                base_pipeline=SimpleNamespace(state={}),
                primary_score=0.02,
            )

        self.assertTrue(report["promotion_pass"])
        self.assertTrue(report["portability_contract"]["passed"])
        self.assertEqual(int(report["portability_contract"]["supporting_cohort_pass_count"]), 1)
        self.assertEqual(report["portability_contract"]["distinct_passed_kinds"], ["period"])


if __name__ == "__main__":
    unittest.main()