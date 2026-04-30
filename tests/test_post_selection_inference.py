import unittest
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd

import core.automl as automl_module
from core import (
    align_post_selection_return_matrix,
    compute_post_selection_inference,
    compute_hansen_spa,
    compute_white_reality_check,
    select_post_selection_candidates,
)


class PostSelectionInferenceTest(unittest.TestCase):
    @staticmethod
    def _make_return_matrix(seed=0, rows=320, cols=4, positive_mean=None):
        rng = np.random.default_rng(seed)
        values = rng.normal(0.0, 0.01, size=(rows, cols))
        values = values - values.mean(axis=0, keepdims=True)
        if positive_mean is not None:
            values[:, 0] += float(positive_mean)
        index = pd.date_range("2026-07-01", periods=rows, freq="1h", tz="UTC")
        columns = list(range(cols))
        return pd.DataFrame(values, index=index, columns=columns)

    @staticmethod
    def _make_trial_record(trial_number, returns, raw_value=None):
        returns = pd.Series(returns, copy=True)
        period_sharpe = float(returns.mean() / max(returns.std(ddof=1), 1e-12))
        raw_value = float(raw_value if raw_value is not None else period_sharpe)
        training = {
            "avg_accuracy": 0.51,
            "feature_selection": {
                "enabled": False,
                "avg_input_features": 24,
                "avg_selected_features": 24,
            },
            "fold_stability": {"enabled": False, "passed": True},
        }
        backtest = {
            "sharpe_ratio": raw_value,
            "net_profit_pct": float(np.prod(1.0 + returns.to_numpy(dtype=float)) - 1.0),
            "max_drawdown": -0.05,
            "total_trades": 30,
        }
        evaluation = {
            "training": training,
            "backtest": backtest,
            "returns": returns,
            "period_sharpe": period_sharpe,
            "raw_objective_value": raw_value,
            "objective_diagnostics": {"final_score": raw_value},
        }
        return {
            "overrides": {
                "model": {"type": "gbm"},
                "features": {"lags": [1, 3]},
                "labels": {"max_holding": 24},
            },
            "search": evaluation,
            "validation": evaluation,
            "training": training,
            "backtest": backtest,
            "returns": returns,
            "period_sharpe": period_sharpe,
            "objective_diagnostics": {"final_score": raw_value},
            "raw_objective_value": raw_value,
        }

    def test_white_reality_check_rejects_noise_candidates(self):
        matrix = self._make_return_matrix(seed=1, rows=320, cols=4)

        report = compute_white_reality_check(
            matrix,
            bootstrap_samples=200,
            mean_block_length=8,
            random_state=11,
        )

        self.assertTrue(report["enabled"])
        self.assertGreaterEqual(report["p_value"], 0.1)

    def test_hansen_spa_detects_superior_candidate(self):
        matrix = self._make_return_matrix(seed=2, rows=320, cols=4, positive_mean=0.004)

        report = compute_hansen_spa(
            matrix,
            bootstrap_samples=200,
            mean_block_length=8,
            random_state=17,
        )

        self.assertTrue(report["enabled"])
        self.assertLessEqual(report["p_value"], 0.05)
        self.assertEqual(report["best_trial_number"], 0)

    def test_aligned_return_matrix_honors_strict_intersection(self):
        index = pd.date_range("2026-07-01", periods=6, freq="1h", tz="UTC")
        frame = pd.DataFrame(
            {
                0: [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
                1: [np.nan, 0.01, 0.02, 0.03, 0.04, np.nan],
                2: [0.02, 0.03, np.nan, 0.05, 0.06, 0.07],
            },
            index=index,
        )

        aligned, report = align_post_selection_return_matrix(
            frame,
            selected_columns=[0, 1, 2],
            overlap_policy="strict_intersection",
            min_overlap_fraction=0.0,
            min_overlap_observations=1,
        )

        self.assertTrue(report["enabled"])
        self.assertEqual(report["rows"], 3)
        pd.testing.assert_index_equal(aligned.index, index[[1, 3, 4]])

    def test_candidate_selection_respects_cap_and_correlation_filter(self):
        base = self._make_return_matrix(seed=3, rows=200, cols=6)
        base[1] = base[0] * 0.99
        trial_reports = [{"number": int(column)} for column in base.columns]

        report = select_post_selection_candidates(
            trial_reports,
            base,
            max_candidates=3,
            correlation_threshold=0.95,
            min_overlap_observations=20,
        )

        self.assertLessEqual(report["selected_candidate_count"], 3)
        self.assertIn(0, report["selected_trial_numbers"])
        self.assertTrue(any(item["discarded_trial_number"] == 1 for item in report["discarded_due_to_correlation"]))

    def test_candidate_selection_handles_constant_series_without_runtime_warning(self):
        index = pd.date_range("2026-07-01", periods=64, freq="1h", tz="UTC")
        frame = pd.DataFrame(
            {
                0: np.full(len(index), 0.001, dtype=float),
                1: np.full(len(index), 0.001, dtype=float),
                2: np.full(len(index), -0.001, dtype=float),
            },
            index=index,
        )
        trial_reports = [{"number": 0}, {"number": 1}, {"number": 2}]

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            report = select_post_selection_candidates(
                trial_reports,
                frame,
                max_candidates=3,
                correlation_threshold=0.95,
                min_overlap_observations=20,
            )

        self.assertIn(0, report["selected_trial_numbers"])
        self.assertTrue(any(item["discarded_trial_number"] == 1 for item in report["discarded_due_to_correlation"]))


    def test_automl_selection_report_includes_post_selection_results_and_gate(self):
        matrix = self._make_return_matrix(seed=4, rows=260, cols=3)
        trial_records = {
            int(column): self._make_trial_record(int(column), matrix[column], raw_value=0.4 - 0.05 * int(column))
            for column in matrix.columns
        }
        completed_trials = [SimpleNamespace(number=int(column), params={}) for column in matrix.columns]

        report = automl_module._build_trial_selection_report(
            completed_trials,
            trial_records,
            "sharpe_ratio",
            {
                "overfitting_control": {
                    "post_selection": {
                        "enabled": True,
                        "require_pass": True,
                        "bootstrap_samples": 120,
                        "mean_block_length": 8,
                        "random_state": 23,
                    }
                }
            },
        )

        self.assertIn("post_selection", report["diagnostics"])
        self.assertTrue(report["diagnostics"]["post_selection"]["enabled"])
        self.assertIn("white_reality_check", report["diagnostics"]["post_selection"])
        self.assertIn("hansen_spa", report["diagnostics"]["post_selection"])
        self.assertEqual(
            report["diagnostics"]["eligible_trial_count_after_post_selection"],
            sum(1 for item in report["trial_reports"] if item["selection_policy"]["eligible_before_post_checks"]),
        )

    def test_post_selection_inference_blocks_noise_pool_when_required(self):
        matrix = self._make_return_matrix(seed=5, rows=260, cols=3)
        trial_reports = [{"number": int(column)} for column in matrix.columns]

        report = compute_post_selection_inference(
            trial_reports,
            matrix,
            config={
                "enabled": True,
                "require_pass": True,
                "bootstrap_samples": 120,
                "mean_block_length": 8,
                "random_state": 29,
            },
        )

        self.assertTrue(report["enabled"])
        self.assertFalse(report["passed"])
        self.assertGreaterEqual(report["white_reality_check"]["p_value"], 0.1)
        self.assertGreaterEqual(report["hansen_spa"]["p_value"], 0.1)


if __name__ == "__main__":
    unittest.main()