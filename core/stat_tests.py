"""Post-selection statistical tests for candidate strategy return paths."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .backtest import _default_mean_block_length, _stationary_bootstrap_indices


def _coerce_return_frame(return_frame, selected_columns=None):
    frame = pd.DataFrame(return_frame, copy=False).astype(float).replace([np.inf, -np.inf], np.nan)
    if selected_columns is not None:
        selected = [column for column in selected_columns if column in frame.columns]
        frame = frame.loc[:, selected]
    return frame.dropna(axis=1, how="all")


def _safe_abs_correlation(overlap_frame, variance_floor=1e-12):
    overlap = pd.DataFrame(overlap_frame, copy=False)
    if overlap.shape[1] < 2 or overlap.empty:
        return None

    values = overlap.to_numpy(dtype=float)
    left = values[:, 0]
    right = values[:, 1]
    if len(left) < 2:
        return None

    left_std = float(np.std(left, ddof=1))
    right_std = float(np.std(right, ddof=1))
    if not np.isfinite(left_std) or not np.isfinite(right_std):
        return None
    if left_std <= float(variance_floor) or right_std <= float(variance_floor):
        return 1.0 if np.allclose(left, right, atol=variance_floor, rtol=0.0) else 0.0

    left_centered = left - float(np.mean(left))
    right_centered = right - float(np.mean(right))
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= float(variance_floor):
        return 0.0

    correlation = float(np.dot(left_centered, right_centered) / denominator)
    return float(abs(np.clip(correlation, -1.0, 1.0)))


def _summarize_pairwise_overlap(coverage_frame, min_overlap_fraction=0.0, min_overlap_observations=0):
    coverage = pd.DataFrame(coverage_frame, copy=False).fillna(False).astype(bool)
    overlap_counts = []
    overlap_fractions = []
    insufficient_pairs = 0

    columns = list(coverage.columns)
    for left_idx, left in enumerate(columns):
        for right in columns[left_idx + 1 :]:
            left_mask = coverage[left]
            right_mask = coverage[right]
            union_count = int((left_mask | right_mask).sum())
            if union_count <= 0:
                continue

            overlap_count = int((left_mask & right_mask).sum())
            overlap_fraction = float(overlap_count / union_count)
            overlap_counts.append(overlap_count)
            overlap_fractions.append(overlap_fraction)
            if overlap_count < int(min_overlap_observations) or overlap_fraction < float(min_overlap_fraction):
                insufficient_pairs += 1

    if not overlap_counts:
        return {
            "pair_count": 0,
            "min_count": None,
            "median_count": None,
            "max_count": None,
            "min_fraction": None,
            "median_fraction": None,
            "max_fraction": None,
            "insufficient_pair_count": 0,
            "sufficient": False,
        }

    return {
        "pair_count": int(len(overlap_counts)),
        "min_count": int(np.min(overlap_counts)),
        "median_count": float(np.median(overlap_counts)),
        "max_count": int(np.max(overlap_counts)),
        "min_fraction": float(np.min(overlap_fractions)),
        "median_fraction": float(np.median(overlap_fractions)),
        "max_fraction": float(np.max(overlap_fractions)),
        "insufficient_pair_count": int(insufficient_pairs),
        "sufficient": bool(insufficient_pairs == 0),
    }


def select_post_selection_candidates(
    trial_reports,
    trial_return_frame,
    max_candidates=8,
    correlation_threshold=0.9,
    min_overlap_observations=10,
):
    frame = _coerce_return_frame(trial_return_frame)
    ranked_trial_numbers = [int(report["number"]) for report in (trial_reports or []) if int(report["number"]) in frame.columns]
    selected = []
    discarded_due_to_correlation = []
    discarded_due_to_cap = 0

    for trial_number in ranked_trial_numbers:
        if len(selected) >= int(max_candidates):
            discarded_due_to_cap += 1
            continue

        candidate_series = frame[trial_number]
        correlated_with = None
        for kept_trial in selected:
            overlap = pd.concat([candidate_series, frame[kept_trial]], axis=1).dropna()
            if len(overlap) < int(min_overlap_observations):
                continue
            correlation = _safe_abs_correlation(overlap)
            if np.isfinite(correlation) and correlation >= float(correlation_threshold):
                correlated_with = {
                    "correlated_with_trial_number": int(kept_trial),
                    "correlation": float(correlation),
                    "overlap_rows": int(len(overlap)),
                }
                break

        if correlated_with is not None:
            discarded_due_to_correlation.append({"discarded_trial_number": int(trial_number), **correlated_with})
            continue

        selected.append(int(trial_number))

    return {
        "candidate_cap": int(max_candidates),
        "correlation_threshold": float(correlation_threshold),
        "min_overlap_observations": int(min_overlap_observations),
        "ranked_candidate_count": int(len(ranked_trial_numbers)),
        "selected_candidate_count": int(len(selected)),
        "selected_trial_numbers": [int(value) for value in selected],
        "discarded_due_to_cap": int(discarded_due_to_cap),
        "discarded_due_to_correlation": discarded_due_to_correlation,
    }


def align_post_selection_return_matrix(
    trial_return_frame,
    selected_columns=None,
    overlap_policy="strict_intersection",
    min_overlap_fraction=0.5,
    min_overlap_observations=5,
):
    frame = _coerce_return_frame(trial_return_frame, selected_columns=selected_columns)
    report = {
        "enabled": False,
        "reason": None,
        "overlap_policy": str(overlap_policy or "strict_intersection").lower(),
        "selected_columns": list(frame.columns),
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "strict_overlap_rows": 0,
        "strict_overlap_fraction": None,
        "pairwise_overlap": {},
        "fill_method": None,
    }
    if frame.empty or frame.shape[1] == 0:
        report["reason"] = "insufficient_candidates"
        return pd.DataFrame(), report

    coverage = frame.notna()
    strict_overlap_rows = int(coverage.all(axis=1).sum())
    report["strict_overlap_rows"] = strict_overlap_rows
    report["strict_overlap_fraction"] = float(strict_overlap_rows / len(frame)) if len(frame) > 0 else None
    report["pairwise_overlap"] = _summarize_pairwise_overlap(
        coverage,
        min_overlap_fraction=min_overlap_fraction,
        min_overlap_observations=min_overlap_observations,
    )

    overlap_policy = report["overlap_policy"]
    if overlap_policy == "strict_intersection":
        aligned = frame.loc[coverage.all(axis=1)].copy()
        report["fill_method"] = "drop_non_intersection"
    elif overlap_policy == "pairwise_overlap":
        if not report["pairwise_overlap"].get("sufficient", False):
            report["reason"] = "insufficient_overlap"
            return pd.DataFrame(), report
        aligned = frame.fillna(0.0).copy()
        report["fill_method"] = "zero_fill_pairwise_overlap"
    elif overlap_policy == "zero_fill_debug":
        aligned = frame.fillna(0.0).copy()
        report["fill_method"] = "zero_fill"
    else:
        report["reason"] = "unknown_overlap_policy"
        return pd.DataFrame(), report

    if aligned.shape[0] < 2 or aligned.shape[1] < 1:
        report["reason"] = "insufficient_rows"
        return pd.DataFrame(), report

    report["enabled"] = True
    report["rows"] = int(aligned.shape[0])
    report["columns"] = int(aligned.shape[1])
    return aligned.astype(float), report


def _build_bootstrap_report_template(test_name, return_matrix, bootstrap_samples, mean_block_length):
    return {
        "enabled": False,
        "reason": None,
        "test": test_name,
        "bootstrap_samples": int(bootstrap_samples),
        "mean_block_length": None if mean_block_length is None else int(mean_block_length),
        "trial_count": int(getattr(return_matrix, "shape", [0, 0])[1]) if return_matrix is not None else 0,
        "observations": int(getattr(return_matrix, "shape", [0, 0])[0]) if return_matrix is not None else 0,
        "observed_stat": None,
        "p_value": None,
        "best_trial_number": None,
        "best_mean_return": None,
        "bootstrap_stat_mean": None,
        "bootstrap_stat_median": None,
        "bootstrap_stat_max": None,
    }


def compute_white_reality_check(return_matrix, bootstrap_samples=500, mean_block_length=None, random_state=42):
    frame = _coerce_return_frame(return_matrix).dropna()
    report = _build_bootstrap_report_template("white_reality_check", frame, bootstrap_samples, mean_block_length)
    if frame.empty or frame.shape[0] < 2 or frame.shape[1] < 1:
        report["reason"] = "insufficient_data"
        return report

    values = frame.to_numpy(dtype=float)
    sample_size = values.shape[0]
    candidate_means = values.mean(axis=0)
    observed_stat = float(max(np.sqrt(sample_size) * np.max(candidate_means), 0.0))
    mean_block_length = int(mean_block_length or _default_mean_block_length(sample_size))
    centered = values - candidate_means
    rng = np.random.default_rng(random_state)
    bootstrap_stats = np.empty(int(bootstrap_samples), dtype=float)

    for sample_idx in range(int(bootstrap_samples)):
        sampled_idx = _stationary_bootstrap_indices(sample_size, mean_block_length, rng)
        sampled_means = centered[sampled_idx, :].mean(axis=0)
        bootstrap_stats[sample_idx] = float(max(np.sqrt(sample_size) * np.max(sampled_means), 0.0))

    best_column = int(np.argmax(candidate_means))
    report.update(
        {
            "enabled": True,
            "mean_block_length": int(mean_block_length),
            "observed_stat": observed_stat,
            "p_value": float(np.mean(bootstrap_stats >= observed_stat - 1e-12)),
            "best_trial_number": int(frame.columns[best_column]),
            "best_mean_return": float(candidate_means[best_column]),
            "bootstrap_stat_mean": float(np.mean(bootstrap_stats)),
            "bootstrap_stat_median": float(np.median(bootstrap_stats)),
            "bootstrap_stat_max": float(np.max(bootstrap_stats)),
        }
    )
    return report


def compute_hansen_spa(
    return_matrix,
    bootstrap_samples=500,
    mean_block_length=None,
    random_state=42,
    variance_floor=1e-12,
):
    frame = _coerce_return_frame(return_matrix).dropna()
    report = _build_bootstrap_report_template("hansen_spa", frame, bootstrap_samples, mean_block_length)
    if frame.empty or frame.shape[0] < 2 or frame.shape[1] < 1:
        report["reason"] = "insufficient_data"
        return report

    values = frame.to_numpy(dtype=float)
    sample_size = values.shape[0]
    candidate_means = values.mean(axis=0)
    candidate_std = values.std(axis=0, ddof=1)
    candidate_std = np.where(np.isfinite(candidate_std) & (candidate_std > variance_floor), candidate_std, np.nan)
    if not np.isfinite(candidate_std).any():
        report["reason"] = "insufficient_variance"
        return report

    scaled_statistics = np.sqrt(sample_size) * candidate_means / candidate_std
    observed_stat = float(max(np.nanmax(scaled_statistics), 0.0))
    truncation_threshold = float(np.sqrt(max(0.0, 2.0 * np.log(max(np.log(max(sample_size, 16)), 1.0)))))
    null_mean = np.where(scaled_statistics <= -truncation_threshold, candidate_means, 0.0)
    centered = values - candidate_means + null_mean
    mean_block_length = int(mean_block_length or _default_mean_block_length(sample_size))
    rng = np.random.default_rng(random_state)
    bootstrap_stats = np.empty(int(bootstrap_samples), dtype=float)

    for sample_idx in range(int(bootstrap_samples)):
        sampled_idx = _stationary_bootstrap_indices(sample_size, mean_block_length, rng)
        sampled_means = centered[sampled_idx, :].mean(axis=0)
        studentized = np.sqrt(sample_size) * sampled_means / candidate_std
        bootstrap_stats[sample_idx] = float(max(np.nanmax(studentized), 0.0))

    best_column = int(np.nanargmax(scaled_statistics))
    report.update(
        {
            "enabled": True,
            "mean_block_length": int(mean_block_length),
            "observed_stat": observed_stat,
            "p_value": float(np.mean(bootstrap_stats >= observed_stat - 1e-12)),
            "best_trial_number": int(frame.columns[best_column]),
            "best_mean_return": float(candidate_means[best_column]),
            "bootstrap_stat_mean": float(np.mean(bootstrap_stats)),
            "bootstrap_stat_median": float(np.median(bootstrap_stats)),
            "bootstrap_stat_max": float(np.max(bootstrap_stats)),
            "truncation_threshold": truncation_threshold,
        }
    )
    return report


def _resolve_post_selection_pass(white_report, spa_report, alpha, pass_rule):
    white_passed = bool(
        white_report.get("enabled") and white_report.get("p_value") is not None and white_report["p_value"] <= alpha
    )
    spa_passed = bool(
        spa_report.get("enabled") and spa_report.get("p_value") is not None and spa_report["p_value"] <= alpha
    )
    normalized_rule = str(pass_rule or "spa").lower()
    if normalized_rule == "white_rc":
        return white_passed
    if normalized_rule == "both":
        return white_passed and spa_passed
    if normalized_rule == "any":
        return white_passed or spa_passed
    return spa_passed


def compute_post_selection_inference(trial_reports, trial_return_frame, config=None):
    config = copy.deepcopy(config or {})
    report = {
        "enabled": bool(config.get("enabled", True)),
        "reason": None,
        "alpha": float(config.get("alpha", 0.05)),
        "require_pass": bool(config.get("require_pass", False)),
        "pass_rule": str(config.get("pass_rule", "spa")).lower(),
        "candidate_selection": {},
        "aligned_return_matrix": {},
        "white_reality_check": {"enabled": False, "reason": "disabled"},
        "hansen_spa": {"enabled": False, "reason": "disabled"},
        "passed": True,
    }
    if not report["enabled"]:
        report["reason"] = "disabled"
        return report

    candidate_selection = select_post_selection_candidates(
        trial_reports,
        trial_return_frame,
        max_candidates=int(config.get("max_candidates", 8)),
        correlation_threshold=float(config.get("correlation_threshold", 0.9)),
        min_overlap_observations=int(config.get("min_overlap_observations", 10)),
    )
    report["candidate_selection"] = candidate_selection
    if candidate_selection["selected_candidate_count"] < 1:
        report["reason"] = "insufficient_candidates"
        report["passed"] = False
        return report

    aligned_matrix, matrix_report = align_post_selection_return_matrix(
        trial_return_frame,
        selected_columns=candidate_selection["selected_trial_numbers"],
        overlap_policy=config.get("overlap_policy", "strict_intersection"),
        min_overlap_fraction=float(config.get("min_overlap_fraction", 0.5)),
        min_overlap_observations=int(config.get("min_overlap_observations", 10)),
    )
    report["aligned_return_matrix"] = matrix_report
    if aligned_matrix.empty:
        report["reason"] = matrix_report.get("reason") or "insufficient_rows"
        report["passed"] = False
        return report

    white_report = compute_white_reality_check(
        aligned_matrix,
        bootstrap_samples=int(config.get("bootstrap_samples", 500)),
        mean_block_length=config.get("mean_block_length"),
        random_state=int(config.get("random_state", 42)),
    )
    spa_report = compute_hansen_spa(
        aligned_matrix,
        bootstrap_samples=int(config.get("bootstrap_samples", 500)),
        mean_block_length=config.get("mean_block_length"),
        random_state=int(config.get("random_state", 43)),
    )
    report["white_reality_check"] = white_report
    report["hansen_spa"] = spa_report
    report["passed"] = _resolve_post_selection_pass(
        white_report,
        spa_report,
        alpha=float(report["alpha"]),
        pass_rule=report["pass_rule"],
    )
    return report


__all__ = [
    "align_post_selection_return_matrix",
    "compute_hansen_spa",
    "compute_post_selection_inference",
    "compute_white_reality_check",
    "select_post_selection_candidates",
]