"""AutoML search helpers for the research pipeline."""

import copy
from itertools import combinations
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:  # pragma: no cover - handled at runtime when AutoML is enabled
    optuna = None
    TPESampler = None


DEFAULT_AUTOML_SEARCH_SPACE = {
    "features": {
        "lags": {
            "type": "categorical",
            "choices": ["1,3,6", "1,2,4,8", "1,4,12", "1,6,12"],
        },
        "frac_diff_d": {"type": "float", "low": 0.2, "high": 0.8, "step": 0.2},
        "rolling_window": {"type": "categorical", "choices": [14, 20, 28, 40]},
        "squeeze_quantile": {"type": "categorical", "choices": [0.1, 0.15, 0.2, 0.25]},
    },
    "feature_selection": {
        "enabled": {"type": "categorical", "choices": [True, False]},
        "max_features": {"type": "categorical", "choices": [32, 48, 64, 96, 128]},
        "min_mi_threshold": {"type": "categorical", "choices": [0.0, 0.0005, 0.001, 0.002]},
    },
    "labels": {
        "pt_mult": {"type": "float", "low": 1.0, "high": 3.0, "step": 0.5},
        "sl_mult": {"type": "float", "low": 1.0, "high": 3.0, "step": 0.5},
        "max_holding": {"type": "categorical", "choices": [12, 24, 48]},
        "min_return": {"type": "categorical", "choices": [0.0, 0.0005, 0.001, 0.002]},
        "volatility_window": {"type": "categorical", "choices": [12, 24, 48]},
        "barrier_tie_break": {"type": "categorical", "choices": ["sl", "pt"]},
    },
    "regime": {
        "n_regimes": {"type": "categorical", "choices": [2, 3, 4]},
    },
    "model": {
        "type": {"type": "categorical", "choices": ["rf", "gbm", "logistic"]},
        "gap": {"type": "categorical", "choices": [12, 24, 48]},
        "validation_fraction": {"type": "categorical", "choices": [0.15, 0.2, 0.25, 0.3]},
        "meta_n_splits": {"type": "categorical", "choices": [2, 3]},
        "calibration_params": {
            "c": {"type": "float", "low": 0.1, "high": 10.0, "log": True},
        },
        "meta_params": {
            "c": {"type": "float", "low": 0.05, "high": 5.0, "log": True},
        },
        "meta_calibration_params": {
            "c": {"type": "float", "low": 0.1, "high": 10.0, "log": True},
        },
        "params": {
            "rf": {
                "n_estimators": {"type": "categorical", "choices": [100, 200, 400]},
                "max_depth": {"type": "categorical", "choices": [3, 5, 8, None]},
                "min_samples_leaf": {"type": "categorical", "choices": [1, 3, 5, 10]},
            },
            "gbm": {
                "n_estimators": {"type": "categorical", "choices": [100, 200, 400]},
                "learning_rate": {"type": "float", "low": 0.03, "high": 0.2, "log": True},
                "max_depth": {"type": "categorical", "choices": [2, 3, 4]},
                "subsample": {"type": "categorical", "choices": [0.7, 0.85, 1.0]},
                "min_samples_leaf": {"type": "categorical", "choices": [1, 3, 5, 10]},
            },
            "logistic": {
                "c": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
            },
        },
    },
    "signals": {
        "threshold": {"type": "categorical", "choices": [0.01, 0.02, 0.03, 0.05]},
        "fraction": {"type": "categorical", "choices": [0.25, 0.5, 0.75]},
        "edge_threshold": {"type": "categorical", "choices": [0.03, 0.05, 0.08, 0.12]},
        "meta_threshold": {"type": "categorical", "choices": [0.5, 0.55, 0.6, 0.65]},
        "tuning_min_trades": {"type": "categorical", "choices": [3, 5, 8]},
    },
}


_NORMAL_DIST = NormalDist()
_EULER_MASCHERONI = 0.5772156649015329
_BACKTEST_OBJECTIVES = {"sharpe_ratio", "net_profit_pct", "profit_factor", "calmar_ratio"}


def _normalize_objective_name(objective_name):
    objective_name = (objective_name or "accuracy_first").lower()
    if objective_name == "composite":
        return "accuracy_first"
    return objective_name


def _resolve_study_name(base_config, automl_config):
    data_config = base_config.get("data", {})
    objective = _normalize_objective_name(automl_config.get("objective", "accuracy_first"))
    study_name = automl_config.get("study_name") or (
        f"{data_config.get('symbol', 'symbol')}_{data_config.get('interval', 'interval')}_{objective}"
    )
    schema_version = base_config.get("features", {}).get("schema_version")
    if schema_version and schema_version not in study_name:
        return f"{study_name}_{schema_version}"
    return study_name


def _clone_value(value):
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    return value


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            pass
    return value


def _deep_merge(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _slice_temporal_value(value, start_timestamp=None, end_timestamp=None):
    if isinstance(value, pd.DataFrame):
        sliced = value
        if isinstance(sliced.index, pd.DatetimeIndex):
            if start_timestamp is not None:
                sliced = sliced.loc[sliced.index >= start_timestamp]
            if end_timestamp is not None:
                sliced = sliced.loc[sliced.index <= end_timestamp]
        return sliced.copy()
    if isinstance(value, pd.Series):
        sliced = value
        if isinstance(sliced.index, pd.DatetimeIndex):
            if start_timestamp is not None:
                sliced = sliced.loc[sliced.index >= start_timestamp]
            if end_timestamp is not None:
                sliced = sliced.loc[sliced.index <= end_timestamp]
        return sliced.copy()
    if isinstance(value, dict):
        return {
            key: _slice_temporal_value(item, start_timestamp=start_timestamp, end_timestamp=end_timestamp)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_slice_temporal_value(item, start_timestamp=start_timestamp, end_timestamp=end_timestamp) for item in value]
    if isinstance(value, tuple):
        return tuple(
            _slice_temporal_value(item, start_timestamp=start_timestamp, end_timestamp=end_timestamp)
            for item in value
        )
    return copy.deepcopy(value)


def _build_state_bundle(base_pipeline):
    return {
        "raw_data": base_pipeline.require("raw_data").copy(),
        "data": base_pipeline.require("data").copy(),
        "indicator_run": base_pipeline.state.get("indicator_run"),
        "futures_context": _slice_temporal_value(base_pipeline.state.get("futures_context")),
        "cross_asset_context": _slice_temporal_value(base_pipeline.state.get("cross_asset_context")),
        "symbol_filters": copy.deepcopy(base_pipeline.state.get("symbol_filters")),
    }


def _build_temporal_state_bundle(full_state_bundle, end_timestamp=None):
    raw_data = pd.DataFrame(full_state_bundle["raw_data"], copy=True)
    if end_timestamp is not None:
        raw_data = raw_data.loc[raw_data.index <= end_timestamp].copy()

    slice_index = raw_data.index
    return {
        "raw_data": raw_data,
        "data": full_state_bundle["data"].reindex(slice_index).copy(),
        "indicator_run": full_state_bundle.get("indicator_run"),
        "futures_context": _slice_temporal_value(full_state_bundle.get("futures_context"), end_timestamp=end_timestamp),
        "cross_asset_context": _slice_temporal_value(
            full_state_bundle.get("cross_asset_context"),
            end_timestamp=end_timestamp,
        ),
        "symbol_filters": copy.deepcopy(full_state_bundle.get("symbol_filters")),
    }


def _seed_candidate_state(candidate, state_bundle):
    for key, value in state_bundle.items():
        if value is None:
            continue
        candidate.state[key] = _slice_temporal_value(value)


def _resolve_holdout_plan(raw_data, automl_config):
    plan = {
        "enabled": False,
        "reason": None,
        "search_rows": int(len(raw_data)),
        "validation_rows": 0,
        "holdout_rows": 0,
        "validation_start_timestamp": None,
        "validation_end_timestamp": None,
        "holdout_start_timestamp": None,
        "start_timestamp": None,
        "end_timestamp": None,
        "search_end_timestamp": None,
    }

    if raw_data is None or len(raw_data) < 3:
        plan["reason"] = "insufficient_rows"
        return plan
    if not automl_config.get("locked_holdout_enabled", True):
        plan["reason"] = "disabled"
        return plan

    holdout_rows = automl_config.get("locked_holdout_bars")
    explicit_holdout = holdout_rows is not None
    if holdout_rows is None:
        holdout_rows = int(round(len(raw_data) * float(automl_config.get("locked_holdout_fraction", 0.2))))
    holdout_rows = int(holdout_rows)
    if holdout_rows <= 0:
        plan["reason"] = "empty_holdout"
        return plan

    validation_rows = int(round(len(raw_data) * float(automl_config.get("validation_fraction", 0.2))))
    if validation_rows <= 0:
        plan["reason"] = "empty_validation"
        return plan

    min_search_rows = int(automl_config.get("locked_holdout_min_search_rows", 100))
    shortfall = max(min_search_rows - (len(raw_data) - validation_rows - holdout_rows), 0)
    if shortfall > 0:
        validation_reduction = min(shortfall, max(validation_rows - 1, 0))
        validation_rows -= validation_reduction
        shortfall -= validation_reduction

    if shortfall > 0 and not explicit_holdout:
        holdout_reduction = min(shortfall, max(holdout_rows - 1, 0))
        holdout_rows -= holdout_reduction
        shortfall -= holdout_reduction

    if holdout_rows <= 0:
        plan["reason"] = "empty_holdout"
        return plan
    if validation_rows <= 0:
        plan["reason"] = "empty_validation"
        return plan
    if shortfall > 0:
        plan["reason"] = "insufficient_search_rows"
        return plan

    search_rows = int(len(raw_data) - validation_rows - holdout_rows)
    if search_rows <= 0:
        plan["reason"] = "insufficient_search_rows"
        return plan

    validation_start_index = search_rows
    validation_end_index = search_rows + validation_rows - 1
    holdout_start_index = search_rows + validation_rows
    plan.update(
        {
            "enabled": True,
            "search_rows": search_rows,
            "validation_rows": validation_rows,
            "holdout_rows": holdout_rows,
            "validation_start_timestamp": raw_data.index[validation_start_index],
            "validation_end_timestamp": raw_data.index[validation_end_index],
            "holdout_start_timestamp": raw_data.index[holdout_start_index],
            "start_timestamp": raw_data.index[holdout_start_index],
            "end_timestamp": raw_data.index[-1],
            "search_end_timestamp": raw_data.index[search_rows - 1],
        }
    )
    return plan


def _sample_from_spec(trial, name, spec):
    if isinstance(spec, dict):
        param_type = spec.get("type")
        if param_type == "categorical":
            choices = [tuple(choice) if isinstance(choice, list) else choice for choice in spec["choices"]]
            return trial.suggest_categorical(name, choices)
        if param_type == "float":
            if "step" in spec:
                return trial.suggest_float(name, spec["low"], spec["high"], step=spec["step"])
            return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
        if param_type == "int":
            return trial.suggest_int(
                name,
                spec["low"],
                spec["high"],
                step=spec.get("step", 1),
                log=spec.get("log", False),
            )

    if isinstance(spec, list):
        choices = [tuple(choice) if isinstance(choice, list) else choice for choice in spec]
        return trial.suggest_categorical(name, choices)

    raise TypeError(f"Unsupported search spec for {name!r}: {spec!r}")


def _sample_param_group(trial, prefix, group_space):
    return {
        key: _sample_from_spec(trial, f"{prefix}.{key}", spec)
        for key, spec in (group_space or {}).items()
    }


def _build_study_storage_path(base_config, automl_config):
    explicit = automl_config.get("storage")
    if explicit:
        path = Path(explicit)
    else:
        study_name = _resolve_study_name(base_config, automl_config)
        path = Path(".cache") / "automl" / f"{study_name}.db"

    path.parent.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _summarize_training(training):
    return {
        "avg_accuracy": training.get("avg_accuracy"),
        "avg_f1_macro": training.get("avg_f1_macro"),
        "avg_directional_accuracy": training.get("avg_directional_accuracy"),
        "avg_directional_f1_macro": training.get("avg_directional_f1_macro"),
        "avg_log_loss": training.get("avg_log_loss"),
        "avg_brier_score": training.get("avg_brier_score"),
        "avg_calibration_error": training.get("avg_calibration_error"),
        "headline_metrics": training.get("headline_metrics", {}),
        "fold_count": len(training.get("fold_metrics", [])),
    }


def _summarize_backtest(backtest):
    keys = [
        "net_profit",
        "net_profit_pct",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "profit_factor",
        "max_drawdown",
        "total_trades",
        "win_rate",
        "ending_equity",
    ]
    summary = {key: backtest.get(key) for key in keys}
    if backtest.get("statistical_significance") is not None:
        summary["statistical_significance"] = backtest.get("statistical_significance")
    return summary


def _resolve_metric(training, key, fallback=None):
    value = training.get(key)
    if value is None and fallback is not None:
        value = training.get(fallback)
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _run_candidate_steps(candidate, step_names):
    for step_name in step_names:
        candidate.run_step(step_name)


def _execute_trial_candidate(base_config, overrides, pipeline_class, trial_step_classes, state_bundle):
    candidate_config = copy.deepcopy(base_config)
    _deep_merge(candidate_config, copy.deepcopy(overrides or {}))
    candidate_config["automl"] = {**candidate_config.get("automl", {}), "enabled": False}

    candidate = pipeline_class(candidate_config, steps=trial_step_classes)
    _seed_candidate_state(candidate, state_bundle)
    _run_candidate_steps(
        candidate,
        [
            "build_features",
            "detect_regimes",
            "build_labels",
            "align_data",
            "select_features",
            "compute_sample_weights",
            "train_models",
            "generate_signals",
            "run_backtest",
        ],
    )
    return candidate.state["training"], candidate.state["backtest"]


def _execute_temporal_split_candidate(
    base_config,
    overrides,
    pipeline_class,
    trial_step_classes,
    state_bundle,
    split_timestamp,
):
    candidate_config = copy.deepcopy(base_config)
    _deep_merge(candidate_config, copy.deepcopy(overrides or {}))
    candidate_config["automl"] = {**candidate_config.get("automl", {}), "enabled": False}

    candidate = pipeline_class(candidate_config, steps=trial_step_classes)
    _seed_candidate_state(candidate, state_bundle)
    _run_candidate_steps(
        candidate,
        [
            "build_features",
            "detect_regimes",
            "build_labels",
            "align_data",
        ],
    )

    aligned_index = candidate.state["X"].index
    aligned_train_rows = int((aligned_index < split_timestamp).sum())
    aligned_test_rows = int((aligned_index >= split_timestamp).sum())
    if aligned_train_rows <= 0 or aligned_test_rows <= 0:
        raise RuntimeError("Aligned split empty")

    candidate.config["model"] = {
        **candidate.config.get("model", {}),
        "cv_method": "walk_forward",
        "n_splits": 1,
        "train_size": aligned_train_rows,
        "test_size": aligned_test_rows,
    }
    _run_candidate_steps(
        candidate,
        [
            "compute_sample_weights",
            "train_models",
            "generate_signals",
            "run_backtest",
        ],
    )
    return candidate.state["training"], candidate.state["backtest"], {
        "aligned_train_rows": int(aligned_train_rows),
        "aligned_test_rows": int(aligned_test_rows),
    }


def _build_validation_holdout_report(best_trial_report, holdout_plan):
    report = {
        "enabled": bool(holdout_plan.get("enabled", False)),
        "reason": holdout_plan.get("reason"),
        "start_timestamp": _json_ready(holdout_plan.get("validation_start_timestamp")),
        "end_timestamp": _json_ready(holdout_plan.get("validation_end_timestamp")),
        "search_rows": int(holdout_plan.get("search_rows", 0)),
        "validation_rows": int(holdout_plan.get("validation_rows", 0)),
        "aligned_search_rows": 0,
        "aligned_validation_rows": 0,
        "training": None,
        "backtest": None,
        "raw_objective_value": None,
        "selection_value": None,
        "meets_minimum_dsr_threshold": None,
    }
    if not holdout_plan.get("enabled") or not best_trial_report:
        return report

    validation_metrics = best_trial_report.get("validation_metrics") or {}
    split = validation_metrics.get("split") or {}
    report["aligned_search_rows"] = int(split.get("aligned_train_rows", 0))
    report["aligned_validation_rows"] = int(split.get("aligned_test_rows", 0))
    report["training"] = validation_metrics.get("training")
    report["backtest"] = validation_metrics.get("backtest")
    report["raw_objective_value"] = validation_metrics.get("raw_objective_value")
    report["selection_value"] = best_trial_report.get("selection_value")
    report["meets_minimum_dsr_threshold"] = best_trial_report.get("meets_minimum_dsr_threshold")
    return report


def _extract_sharpe_ci_lower(backtest_summary):
    significance = (backtest_summary or {}).get("statistical_significance") or {}
    metrics = significance.get("metrics") or {}
    sharpe = metrics.get("sharpe_ratio") or {}
    confidence_interval = sharpe.get("confidence_interval") or {}
    lower = confidence_interval.get("lower")
    if lower is None or not np.isfinite(lower):
        return None
    return float(lower)


def _evaluate_locked_holdout(base_config, best_overrides, pipeline_class, trial_step_classes, full_state_bundle, holdout_plan):
    report = {
        "enabled": bool(holdout_plan.get("enabled", False)),
        "reason": holdout_plan.get("reason"),
        "start_timestamp": _json_ready(holdout_plan.get("holdout_start_timestamp")),
        "end_timestamp": _json_ready(holdout_plan.get("end_timestamp")),
        "search_rows": int(holdout_plan.get("search_rows", 0)),
        "validation_rows": int(holdout_plan.get("validation_rows", 0)),
        "pre_holdout_rows": int(holdout_plan.get("search_rows", 0) + holdout_plan.get("validation_rows", 0)),
        "holdout_rows": int(holdout_plan.get("holdout_rows", 0)),
        "aligned_search_rows": 0,
        "aligned_pre_holdout_rows": 0,
        "aligned_holdout_rows": 0,
        "training": None,
        "backtest": None,
        "holdout_warning": False,
    }
    if not holdout_plan.get("enabled"):
        return report

    try:
        training, backtest, split = _execute_temporal_split_candidate(
            base_config,
            best_overrides,
            pipeline_class,
            trial_step_classes,
            full_state_bundle,
            holdout_plan["holdout_start_timestamp"],
        )
    except RuntimeError as exc:
        if "Aligned split empty" in str(exc):
            report["reason"] = "aligned_split_empty"
            return report
        raise

    report["aligned_search_rows"] = int(split["aligned_train_rows"])
    report["aligned_pre_holdout_rows"] = int(split["aligned_train_rows"])
    report["aligned_holdout_rows"] = int(split["aligned_test_rows"])
    report["training"] = _json_ready(_summarize_training(training))
    report["backtest"] = _json_ready(_summarize_backtest(backtest))
    sharpe_ci_lower = _extract_sharpe_ci_lower(report["backtest"])
    report["holdout_warning"] = bool(sharpe_ci_lower is not None and sharpe_ci_lower < 0.0)
    return report


def _resolve_overfitting_control(automl_config=None):
    automl_config = automl_config or {}
    control = copy.deepcopy(automl_config.get("overfitting_control", {}))
    dsr_config = dict(control.get("deflated_sharpe", {}))
    pbo_config = dict(control.get("pbo", {}))

    return {
        "enabled": bool(control.get("enabled", True)),
        "selection_mode": str(control.get("selection_mode", "penalized_ranking")).lower(),
        "penalized_objectives": {
            str(value).lower()
            for value in control.get("penalized_objectives", sorted(_BACKTEST_OBJECTIVES))
        },
        "deflated_sharpe": {
            "enabled": bool(dsr_config.get("enabled", True)),
            "use_effective_trial_count": bool(dsr_config.get("use_effective_trial_count", True)),
            "min_track_record_length": int(dsr_config.get("min_track_record_length", 10)),
        },
        "pbo": {
            "enabled": bool(pbo_config.get("enabled", True)),
            "n_blocks": int(pbo_config.get("n_blocks", 8)),
            "test_blocks": pbo_config.get("test_blocks"),
            "min_block_size": int(pbo_config.get("min_block_size", 5)),
            "metric": str(pbo_config.get("metric", "sharpe_ratio")).lower(),
        },
    }


def _infer_periods_per_year(index):
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return 0.0

    deltas = index.to_series().diff().dropna()
    if deltas.empty:
        return 0.0

    seconds = deltas.median().total_seconds()
    if seconds <= 0:
        return 0.0

    return (365.25 * 24 * 60 * 60) / seconds


def _extract_backtest_returns(backtest):
    equity_curve = backtest.get("equity_curve")
    if not isinstance(equity_curve, pd.Series) or equity_curve.empty:
        return pd.Series(dtype=float)

    returns = (
        pd.Series(equity_curve, copy=False)
        .astype(float)
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    returns.name = "strategy_return"
    return returns


def _compute_period_sharpe(returns):
    series = pd.Series(returns, copy=False).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 2:
        return None

    volatility = float(series.std())
    if not np.isfinite(volatility) or volatility <= 0.0:
        return 0.0
    return float(series.mean() / volatility)


def _build_evaluation_record(training, backtest, objective_name, automl_config, split=None):
    training_summary = _json_ready(_summarize_training(training))
    backtest_summary = _json_ready(_summarize_backtest(backtest))
    returns = _extract_backtest_returns(backtest)
    period_sharpe = _compute_period_sharpe(returns)
    raw_objective_value = compute_objective_value(
        objective_name,
        training_summary,
        backtest_summary,
        automl_config,
    )
    record = {
        "training": training_summary,
        "backtest": backtest_summary,
        "returns": returns,
        "period_sharpe": period_sharpe,
        "raw_objective_value": float(raw_objective_value),
    }
    if split is not None:
        record["split"] = {
            "aligned_train_rows": int(split.get("aligned_train_rows", 0)),
            "aligned_test_rows": int(split.get("aligned_test_rows", 0)),
        }
    return record


def _build_trial_record(overrides, search_record, validation_record=None):
    validation_record = validation_record or search_record
    return {
        "overrides": copy.deepcopy(overrides or {}),
        "search": search_record,
        "validation": validation_record,
        "training": validation_record["training"],
        "backtest": validation_record["backtest"],
        "returns": validation_record["returns"],
        "period_sharpe": validation_record["period_sharpe"],
        "raw_objective_value": float(validation_record["raw_objective_value"]),
        "search_raw_objective_value": float(search_record["raw_objective_value"]),
    }


def compute_deflated_sharpe_ratio(
    returns,
    sharpe_mean=0.0,
    sharpe_std=0.0,
    trial_count=1.0,
    min_track_record_length=10,
):
    series = pd.Series(returns, copy=False).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    sample_size = int(len(series))
    periods_per_year = _infer_periods_per_year(series.index)
    annualization = np.sqrt(periods_per_year) if periods_per_year > 0 else 1.0
    observed_period_sharpe = _compute_period_sharpe(series)
    observed_sharpe_ratio = (
        float(observed_period_sharpe * annualization)
        if observed_period_sharpe is not None and np.isfinite(observed_period_sharpe)
        else None
    )
    report = {
        "enabled": True,
        "reason": None,
        "sample_size": sample_size,
        "trial_count": float(max(trial_count, 1.0)),
        "observed_sharpe_ratio": observed_sharpe_ratio,
        "observed_sharpe_per_period": (
            float(observed_period_sharpe) if observed_period_sharpe is not None and np.isfinite(observed_period_sharpe) else None
        ),
        "benchmark_sharpe_ratio": None,
        "benchmark_sharpe_per_period": None,
        "sharpe_mean": float(sharpe_mean) if np.isfinite(sharpe_mean) else 0.0,
        "sharpe_std": float(sharpe_std) if np.isfinite(sharpe_std) else 0.0,
        "skewness": None,
        "kurtosis": None,
        "z_score": None,
        "deflated_sharpe_ratio": 0.0,
    }

    if sample_size < max(2, int(min_track_record_length)):
        report["reason"] = "insufficient_track_record"
        return report
    if observed_period_sharpe is None or not np.isfinite(observed_period_sharpe):
        report["reason"] = "unavailable_sharpe"
        return report

    sharpe_mean = float(sharpe_mean) if np.isfinite(sharpe_mean) else 0.0
    sharpe_std = float(sharpe_std) if np.isfinite(sharpe_std) else 0.0
    effective_trials = float(max(trial_count, 1.0))
    benchmark_period_sharpe = sharpe_mean
    if effective_trials > 1.0 and sharpe_std > 0.0:
        quantile_primary = _NORMAL_DIST.inv_cdf(1.0 - 1.0 / effective_trials)
        quantile_secondary = _NORMAL_DIST.inv_cdf(1.0 - 1.0 / (effective_trials * np.e))
        benchmark_period_sharpe = sharpe_mean + sharpe_std * (
            (1.0 - _EULER_MASCHERONI) * quantile_primary
            + _EULER_MASCHERONI * quantile_secondary
        )

    skewness = float(series.skew()) if sample_size > 2 and np.isfinite(series.skew()) else 0.0
    excess_kurtosis = float(series.kurt()) if sample_size > 3 and np.isfinite(series.kurt()) else 0.0
    kurtosis = excess_kurtosis + 3.0
    denominator_term = 1.0 - skewness * observed_period_sharpe + ((kurtosis - 1.0) / 4.0) * (observed_period_sharpe ** 2)
    denominator_term = max(float(denominator_term), 1e-12)
    z_score = ((observed_period_sharpe - benchmark_period_sharpe) * np.sqrt(max(sample_size - 1, 1))) / np.sqrt(denominator_term)
    deflated_sharpe = float(_NORMAL_DIST.cdf(z_score))

    report.update(
        {
            "benchmark_sharpe_ratio": float(benchmark_period_sharpe * annualization),
            "benchmark_sharpe_per_period": float(benchmark_period_sharpe),
            "skewness": skewness,
            "kurtosis": kurtosis,
            "z_score": float(z_score),
            "deflated_sharpe_ratio": deflated_sharpe,
        }
    )
    return report


def _build_trial_return_frame(trial_records):
    series_by_trial = {}
    common_start = None
    common_end = None

    for trial_number, record in (trial_records or {}).items():
        returns = pd.Series(record.get("returns"), copy=False)
        if returns.empty:
            continue

        returns = returns.astype(float).replace([np.inf, -np.inf], np.nan).dropna().sort_index()
        if returns.empty:
            continue

        series_by_trial[trial_number] = returns
        start = returns.index[0]
        end = returns.index[-1]
        common_start = start if common_start is None or start > common_start else common_start
        common_end = end if common_end is None or end < common_end else common_end

    if len(series_by_trial) < 2 or common_start is None or common_end is None or common_start >= common_end:
        return pd.DataFrame()

    clipped = {}
    index_values = set()
    for trial_number, returns in series_by_trial.items():
        window = returns.loc[(returns.index >= common_start) & (returns.index <= common_end)]
        if window.empty:
            continue
        clipped[trial_number] = window
        index_values.update(window.index.tolist())

    if len(clipped) < 2 or len(index_values) < 2:
        return pd.DataFrame()

    if all(isinstance(value, pd.Timestamp) for value in index_values):
        common_index = pd.DatetimeIndex(sorted(index_values))
    else:
        common_index = pd.Index(sorted(index_values))

    aligned = {}
    for trial_number, returns in clipped.items():
        aligned[trial_number] = returns.reindex(common_index).fillna(0.0).astype(float)
    return pd.DataFrame(aligned, index=common_index)


def _estimate_effective_trial_count(trial_return_frame):
    if trial_return_frame.empty or trial_return_frame.shape[1] <= 1:
        return float(max(trial_return_frame.shape[1], 1)), None

    corr = trial_return_frame.corr().to_numpy(dtype=float)
    upper = corr[np.triu_indices_from(corr, k=1)]
    upper = upper[np.isfinite(upper)]
    if len(upper) == 0:
        return float(trial_return_frame.shape[1]), None

    average_corr = float(np.clip(np.mean(upper), 0.0, 1.0))
    effective_trial_count = average_corr + (1.0 - average_corr) * float(trial_return_frame.shape[1])
    effective_trial_count = max(1.0, min(effective_trial_count, float(trial_return_frame.shape[1])))
    return float(effective_trial_count), average_corr


def _score_return_window(returns, metric="sharpe_ratio"):
    series = pd.Series(returns, copy=False).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return None

    metric = (metric or "sharpe_ratio").lower()
    if metric == "net_profit_pct":
        return float(np.prod(1.0 + series.to_numpy(dtype=float)) - 1.0)
    if metric == "calmar_ratio":
        equity_curve = (1.0 + series).cumprod()
        peak = equity_curve.cummax()
        max_drawdown = float(((equity_curve - peak) / peak).min()) if not equity_curve.empty else 0.0
        total_return = float(equity_curve.iloc[-1] - 1.0) if not equity_curve.empty else 0.0
        return 0.0 if max_drawdown == 0.0 else total_return / abs(max_drawdown)

    volatility = float(series.std())
    if not np.isfinite(volatility) or volatility <= 0.0:
        return 0.0
    return float(series.mean() / volatility)


def compute_cpcv_pbo(trial_return_frame, n_blocks=8, test_blocks=None, min_block_size=5, metric="sharpe_ratio"):
    report = {
        "enabled": False,
        "reason": None,
        "metric": (metric or "sharpe_ratio").lower(),
        "trial_count": int(getattr(trial_return_frame, "shape", [0, 0])[1]) if trial_return_frame is not None else 0,
        "path_rows": int(getattr(trial_return_frame, "shape", [0, 0])[0]) if trial_return_frame is not None else 0,
        "block_count": 0,
        "test_block_count": 0,
        "split_count": 0,
        "probability_of_backtest_overfitting": None,
        "lambda_mean": None,
        "lambda_median": None,
        "lambda_min": None,
        "lambda_max": None,
        "oos_top_half_rate": None,
    }

    if trial_return_frame is None or trial_return_frame.empty or trial_return_frame.shape[1] < 2:
        report["reason"] = "insufficient_trials"
        return report

    frame = trial_return_frame.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(frame) < max(2, int(min_block_size) * 2):
        report["reason"] = "insufficient_rows"
        return report

    block_count = int(max(2, min(int(n_blocks), len(frame))))
    blocks = []
    while block_count >= 2:
        candidate_blocks = [block for block in np.array_split(np.arange(len(frame)), block_count) if len(block) > 0]
        if candidate_blocks and min(len(block) for block in candidate_blocks) >= int(max(1, min_block_size)):
            blocks = candidate_blocks
            break
        block_count -= 1

    if len(blocks) < 2:
        report["reason"] = "insufficient_blocks"
        return report

    if test_blocks is None:
        test_block_count = max(1, len(blocks) // 2)
    else:
        test_block_count = int(test_blocks)
    test_block_count = max(1, min(test_block_count, len(blocks) - 1))

    lambda_values = []
    logit_values = []
    for test_combo in combinations(range(len(blocks)), test_block_count):
        test_positions = np.concatenate([blocks[idx] for idx in test_combo])
        train_positions = np.concatenate([blocks[idx] for idx in range(len(blocks)) if idx not in test_combo])
        if len(train_positions) == 0 or len(test_positions) == 0:
            continue

        in_sample_scores = frame.iloc[train_positions].apply(_score_return_window, metric=metric)
        out_of_sample_scores = frame.iloc[test_positions].apply(_score_return_window, metric=metric)
        valid_mask = in_sample_scores.notna() & out_of_sample_scores.notna()
        if int(valid_mask.sum()) < 2:
            continue

        in_sample_scores = in_sample_scores.loc[valid_mask].astype(float)
        out_of_sample_scores = out_of_sample_scores.loc[valid_mask].astype(float)
        winner = in_sample_scores.idxmax()
        oos_rank = float(out_of_sample_scores.rank(method="average", ascending=True).loc[winner])
        relative_rank = min(max(oos_rank / (len(out_of_sample_scores) + 1.0), 1e-6), 1.0 - 1e-6)
        lambda_values.append(relative_rank)
        logit_values.append(float(np.log(relative_rank / (1.0 - relative_rank))))

    if not lambda_values:
        report["reason"] = "no_valid_splits"
        return report

    lambda_array = np.asarray(lambda_values, dtype=float)
    logit_array = np.asarray(logit_values, dtype=float)
    report.update(
        {
            "enabled": True,
            "block_count": int(len(blocks)),
            "test_block_count": int(test_block_count),
            "split_count": int(len(lambda_array)),
            "probability_of_backtest_overfitting": float(np.mean(logit_array <= 0.0)),
            "lambda_mean": float(np.mean(lambda_array)),
            "lambda_median": float(np.median(lambda_array)),
            "lambda_min": float(np.min(lambda_array)),
            "lambda_max": float(np.max(lambda_array)),
            "oos_top_half_rate": float(np.mean(lambda_array > 0.5)),
        }
    )
    return report


def _build_trial_selection_report(completed_trials, trial_records, objective_name, automl_config):
    control = _resolve_overfitting_control(automl_config)
    minimum_dsr_threshold = automl_config.get("minimum_dsr_threshold", 0.3)
    if minimum_dsr_threshold is not None:
        minimum_dsr_threshold = float(minimum_dsr_threshold)
    trial_return_frame = _build_trial_return_frame(trial_records)
    effective_trial_count = float(len(completed_trials))
    average_pairwise_correlation = None
    if control["enabled"] and control["deflated_sharpe"]["use_effective_trial_count"]:
        effective_trial_count, average_pairwise_correlation = _estimate_effective_trial_count(trial_return_frame)

    period_sharpes = []
    for trial in completed_trials:
        record = trial_records.get(trial.number)
        if record is None:
            continue
        period_sharpe = record.get("period_sharpe")
        if period_sharpe is not None and np.isfinite(period_sharpe):
            period_sharpes.append(float(period_sharpe))

    sharpe_mean = float(np.mean(period_sharpes)) if period_sharpes else 0.0
    sharpe_std = float(np.std(period_sharpes, ddof=1)) if len(period_sharpes) > 1 else 0.0
    selection_metric = objective_name
    selection_mode = "validation_objective_gated_dsr" if minimum_dsr_threshold is not None else "validation_objective"

    trial_reports = []
    for trial in completed_trials:
        record = trial_records.get(trial.number)
        if record is None:
            continue

        search_metrics = {
            "training": record.get("search", {}).get("training"),
            "backtest": record.get("search", {}).get("backtest"),
            "raw_objective_value": record.get("search", {}).get("raw_objective_value"),
        }
        validation_metrics = {
            "training": record.get("validation", {}).get("training"),
            "backtest": record.get("validation", {}).get("backtest"),
            "raw_objective_value": record.get("validation", {}).get("raw_objective_value"),
            "split": _json_ready(record.get("validation", {}).get("split")),
        }

        deflated_sharpe = compute_deflated_sharpe_ratio(
            record.get("returns"),
            sharpe_mean=sharpe_mean,
            sharpe_std=sharpe_std,
            trial_count=effective_trial_count,
            min_track_record_length=control["deflated_sharpe"]["min_track_record_length"],
        )
        selection_value = compute_objective_value(
            objective_name,
            record["training"],
            record["backtest"],
            automl_config,
        )
        meets_minimum_dsr_threshold = True
        dsr_value = deflated_sharpe.get("deflated_sharpe_ratio")
        if minimum_dsr_threshold is not None:
            if dsr_value is None or not np.isfinite(dsr_value) or dsr_value < minimum_dsr_threshold:
                selection_value = float("-inf")
                meets_minimum_dsr_threshold = False

        trial_reports.append(
            {
                "number": trial.number,
                "params": _json_ready(trial.params),
                "overrides": copy.deepcopy(record["overrides"]),
                "training": record["training"],
                "backtest": record["backtest"],
                "raw_objective_value": float(record["raw_objective_value"]),
                "selection_value": float(selection_value),
                "search_metrics": search_metrics,
                "validation_metrics": validation_metrics,
                "meets_minimum_dsr_threshold": meets_minimum_dsr_threshold,
                "overfitting": {"deflated_sharpe": deflated_sharpe},
            }
        )

    if not trial_reports:
        raise RuntimeError("AutoML could not build trial diagnostics for completed trials")

    trial_reports.sort(
        key=lambda item: (item["selection_value"], item["raw_objective_value"]),
        reverse=True,
    )

    pbo_config = control["pbo"]
    if control["enabled"] and pbo_config["enabled"]:
        pbo_report = compute_cpcv_pbo(
            trial_return_frame,
            n_blocks=pbo_config["n_blocks"],
            test_blocks=pbo_config["test_blocks"],
            min_block_size=pbo_config["min_block_size"],
            metric=pbo_config["metric"],
        )
    else:
        pbo_report = {
            "enabled": False,
            "reason": "disabled",
            "metric": pbo_config["metric"],
        }

    best_trial = trial_reports[0]
    diagnostics = {
        "enabled": control["enabled"],
        "selection_mode": selection_mode,
        "selection_metric": selection_metric,
        "minimum_dsr_threshold": minimum_dsr_threshold,
        "trial_count": int(len(completed_trials)),
        "effective_trial_count": float(effective_trial_count),
        "average_pairwise_correlation": average_pairwise_correlation,
        "trial_path_rows": int(len(trial_return_frame)) if not trial_return_frame.empty else 0,
        "sharpe_distribution": {
            "mean": sharpe_mean,
            "std": sharpe_std,
            "min": float(np.min(period_sharpes)) if period_sharpes else None,
            "median": float(np.median(period_sharpes)) if period_sharpes else None,
            "max": float(np.max(period_sharpes)) if period_sharpes else None,
        },
        "best_trial": {
            "number": int(best_trial["number"]),
            "raw_objective_value": float(best_trial["raw_objective_value"]),
            "selection_value": float(best_trial["selection_value"]),
            "meets_minimum_dsr_threshold": bool(best_trial["meets_minimum_dsr_threshold"]),
            "deflated_sharpe": best_trial["overfitting"]["deflated_sharpe"],
        },
        "pbo": pbo_report,
    }
    return {
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
        "trial_reports": trial_reports,
        "diagnostics": diagnostics,
    }


def compute_objective_value(objective_name, training, backtest, automl_config=None, overfitting_context=None):
    """Compute the scalar objective value used by the AutoML study."""
    automl_config = automl_config or {}
    objective_name = _normalize_objective_name(objective_name)

    directional_accuracy = _resolve_metric(training, "avg_directional_accuracy", fallback="avg_accuracy") or 0.0
    log_loss_value = _resolve_metric(training, "avg_log_loss")
    brier_score_value = _resolve_metric(training, "avg_brier_score")
    calibration_error = _resolve_metric(training, "avg_calibration_error")

    if objective_name == "directional_accuracy":
        raw_score = float(directional_accuracy)
    elif objective_name in {"neg_log_loss", "log_loss"}:
        raw_score = float(-(log_loss_value if log_loss_value is not None else 1e6))
    elif objective_name in {"neg_brier_score", "brier_score"}:
        raw_score = float(-(brier_score_value if brier_score_value is not None else 1e6))
    elif objective_name in {"neg_calibration_error", "calibration_error"}:
        raw_score = float(-(calibration_error if calibration_error is not None else 1e6))
    elif objective_name == "net_profit_pct":
        raw_score = float(backtest.get("net_profit_pct", 0.0))
    elif objective_name == "sharpe_ratio":
        raw_score = float(backtest.get("sharpe_ratio", 0.0))
    elif objective_name == "profit_factor":
        profit_factor = backtest.get("profit_factor", 0.0)
        if not np.isfinite(profit_factor):
            raw_score = float(automl_config.get("profit_factor_cap", 5.0))
        else:
            raw_score = float(profit_factor)
    elif objective_name == "calmar_ratio":
        raw_score = float(backtest.get("calmar_ratio", 0.0))
    else:
        score = automl_config.get("weight_directional_accuracy", 100.0) * directional_accuracy
        score += automl_config.get("weight_accuracy", 5.0) * (_resolve_metric(training, "avg_accuracy") or directional_accuracy)

        if log_loss_value is not None:
            score -= automl_config.get("weight_log_loss", 1.0) * log_loss_value
        if brier_score_value is not None:
            score -= automl_config.get("weight_brier_score", 0.5) * brier_score_value
        if calibration_error is not None:
            score -= automl_config.get("weight_calibration_error", 0.5) * calibration_error
        raw_score = float(score)

    if overfitting_context and overfitting_context.get("apply_penalty"):
        deflated_sharpe = overfitting_context.get("deflated_sharpe_ratio")
        if deflated_sharpe is None or not np.isfinite(deflated_sharpe):
            return 0.0
        return float(deflated_sharpe)

    return float(raw_score)


def _sample_trial_overrides(trial, search_space):
    overrides = {}

    feature_space = search_space.get("features", {})
    if feature_space:
        feature_overrides = {}
        if "lags" in feature_space:
            lag_choice = _sample_from_spec(trial, "features.lags", feature_space["lags"])
            if isinstance(lag_choice, str):
                feature_overrides["lags"] = [int(value) for value in lag_choice.split(",") if value]
            else:
                feature_overrides["lags"] = list(lag_choice)
        if "frac_diff_d" in feature_space:
            feature_overrides["frac_diff_d"] = _sample_from_spec(
                trial,
                "features.frac_diff_d",
                feature_space["frac_diff_d"],
            )
        for key in ["rolling_window", "squeeze_quantile"]:
            if key in feature_space:
                feature_overrides[key] = _sample_from_spec(trial, f"features.{key}", feature_space[key])
        if feature_overrides:
            overrides["features"] = feature_overrides

    selection_space = search_space.get("feature_selection", {})
    if selection_space:
        selection_overrides = {}
        for key in ["enabled", "max_features", "min_mi_threshold"]:
            if key in selection_space:
                selection_overrides[key] = _sample_from_spec(
                    trial,
                    f"feature_selection.{key}",
                    selection_space[key],
                )
        if selection_overrides:
            overrides["feature_selection"] = selection_overrides

    label_space = search_space.get("labels", {})
    if label_space:
        label_overrides = {}
        pt_mult = _sample_from_spec(trial, "labels.pt_mult", label_space["pt_mult"])
        sl_mult = _sample_from_spec(trial, "labels.sl_mult", label_space["sl_mult"])
        label_overrides["pt_sl"] = (pt_mult, sl_mult)
        for key in ["max_holding", "min_return", "volatility_window", "barrier_tie_break"]:
            if key in label_space:
                label_overrides[key] = _sample_from_spec(trial, f"labels.{key}", label_space[key])
        overrides["labels"] = label_overrides

    regime_space = search_space.get("regime", {})
    if regime_space:
        regime_overrides = {}
        if "n_regimes" in regime_space:
            regime_overrides["n_regimes"] = _sample_from_spec(
                trial,
                "regime.n_regimes",
                regime_space["n_regimes"],
            )
        if regime_overrides:
            overrides["regime"] = regime_overrides

    model_space = search_space.get("model", {})
    if model_space:
        model_type = _sample_from_spec(trial, "model.type", model_space["type"])
        model_overrides = {"type": model_type}
        if "gap" in model_space:
            model_overrides["gap"] = _sample_from_spec(trial, "model.gap", model_space["gap"])
        for key in ["validation_fraction", "meta_n_splits"]:
            if key in model_space:
                model_overrides[key] = _sample_from_spec(trial, f"model.{key}", model_space[key])
        for key, prefix in [
            ("calibration_params", "model.calibration_params"),
            ("meta_params", "model.meta_params"),
            ("meta_calibration_params", "model.meta_calibration_params"),
        ]:
            params = _sample_param_group(trial, prefix, model_space.get(key, {}))
            if params:
                model_overrides[key] = params
        model_params_space = model_space.get("params", {}).get(model_type, {})
        if model_params_space:
            model_overrides["params"] = {
                key: _sample_from_spec(trial, f"model.{model_type}.{key}", spec)
                for key, spec in model_params_space.items()
            }
        overrides["model"] = model_overrides

    signal_space = search_space.get("signals", {})
    if signal_space:
        signal_overrides = {}
        for key in ["threshold", "fraction", "edge_threshold", "meta_threshold", "tuning_min_trades"]:
            if key in signal_space:
                signal_overrides[key] = _sample_from_spec(trial, f"signals.{key}", signal_space[key])
        if signal_overrides:
            overrides["signals"] = signal_overrides

    return overrides


def run_automl_study(base_pipeline, pipeline_class, trial_step_classes):
    """Run an Optuna study against the pipeline's configurable search space."""
    if optuna is None:
        raise ImportError(
            "AutoML requires optuna. Install it with `python -m pip install optuna` or via requirements.txt."
        )

    base_config = copy.deepcopy(base_pipeline.config)
    automl_config = copy.deepcopy(base_config.get("automl", {}))
    search_space = copy.deepcopy(DEFAULT_AUTOML_SEARCH_SPACE)
    _deep_merge(search_space, automl_config.get("search_space", {}))

    storage_path = _build_study_storage_path(base_config, automl_config)
    storage_url = f"sqlite:///{storage_path.as_posix()}"
    study_name = _resolve_study_name(base_config, automl_config)
    sampler = TPESampler(seed=automl_config.get("seed", 42))
    objective_name = _normalize_objective_name(automl_config.get("objective", "accuracy_first"))

    full_state_bundle = _build_state_bundle(base_pipeline)
    holdout_plan = _resolve_holdout_plan(full_state_bundle["raw_data"], automl_config)
    search_state_bundle = full_state_bundle
    validation_state_bundle = full_state_bundle
    if holdout_plan["enabled"]:
        search_state_bundle = _build_temporal_state_bundle(full_state_bundle, holdout_plan["search_end_timestamp"])
        validation_state_bundle = _build_temporal_state_bundle(full_state_bundle, holdout_plan["validation_end_timestamp"])

    enable_pruning = bool(automl_config.get("enable_pruning", True))
    study_kwargs = {
        "direction": "maximize",
        "sampler": sampler,
        "study_name": study_name,
        "storage": storage_url,
        "load_if_exists": True,
    }
    if enable_pruning:
        study_kwargs["pruner"] = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    study = optuna.create_study(**study_kwargs)
    trial_records = {}

    def objective(trial):
        overrides = _sample_trial_overrides(trial, search_space)
        try:
            search_training, search_backtest = _execute_trial_candidate(
                base_config,
                overrides,
                pipeline_class,
                trial_step_classes,
                search_state_bundle,
            )
            search_record = _build_evaluation_record(
                search_training,
                search_backtest,
                objective_name,
                automl_config,
            )
        except RuntimeError as exc:
            if (
                "No validation splits were generated" in str(exc)
                or "No walk-forward folds were generated" in str(exc)
                or "Aligned split empty" in str(exc)
            ):
                raise optuna.TrialPruned(str(exc)) from exc
            raise

        trial.set_user_attr("overrides", _json_ready(_clone_value(overrides)))
        trial.set_user_attr(
            "search_metrics",
            {
                "training": search_record["training"],
                "backtest": search_record["backtest"],
                "raw_objective_value": float(search_record["raw_objective_value"]),
            },
        )
        trial.set_user_attr("search_raw_objective_value", float(search_record["raw_objective_value"]))

        trial.report(float(search_record["raw_objective_value"]), step=0)
        if enable_pruning and trial.should_prune():
            raise optuna.TrialPruned("Pruned after search-stage objective")

        validation_record = search_record
        if holdout_plan["enabled"]:
            try:
                validation_training, validation_backtest, validation_split = _execute_temporal_split_candidate(
                    base_config,
                    overrides,
                    pipeline_class,
                    trial_step_classes,
                    validation_state_bundle,
                    holdout_plan["validation_start_timestamp"],
                )
            except RuntimeError as exc:
                if (
                    "No validation splits were generated" in str(exc)
                    or "No walk-forward folds were generated" in str(exc)
                    or "Aligned split empty" in str(exc)
                ):
                    raise optuna.TrialPruned(str(exc)) from exc
                raise
            validation_record = _build_evaluation_record(
                validation_training,
                validation_backtest,
                objective_name,
                automl_config,
                split=validation_split,
            )
            trial.report(float(validation_record["raw_objective_value"]), step=1)

        record = _build_trial_record(overrides, search_record, validation_record)
        trial_records[trial.number] = record
        value = float(record["raw_objective_value"])

        trial.set_user_attr("training", record["training"])
        trial.set_user_attr("backtest", record["backtest"])
        trial.set_user_attr(
            "validation_metrics",
            {
                "training": record["training"],
                "backtest": record["backtest"],
                "raw_objective_value": value,
                "split": _json_ready(record.get("validation", {}).get("split")),
            },
        )
        trial.set_user_attr("raw_objective_value", value)
        return value

    study.optimize(
        objective,
        n_trials=automl_config.get("n_trials", 25),
        gc_after_trial=automl_config.get("gc_after_trial", True),
        show_progress_bar=False,
        catch=(ValueError, RuntimeError),
    )

    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        raise RuntimeError("AutoML finished without any completed trials")

    # DSR depends on the completed trial population, so final selection is re-ranked
    # after the study finishes rather than trusting the raw Optuna best trial.
    for trial in completed_trials:
        if trial.number in trial_records:
            continue
        overrides = trial.user_attrs.get("overrides")
        if overrides is None:
            continue
        try:
            search_training, search_backtest = _execute_trial_candidate(
                base_config,
                overrides,
                pipeline_class,
                trial_step_classes,
                search_state_bundle,
            )
            search_record = _build_evaluation_record(
                search_training,
                search_backtest,
                objective_name,
                automl_config,
            )
            validation_record = search_record
            if holdout_plan["enabled"]:
                validation_training, validation_backtest, validation_split = _execute_temporal_split_candidate(
                    base_config,
                    overrides,
                    pipeline_class,
                    trial_step_classes,
                    validation_state_bundle,
                    holdout_plan["validation_start_timestamp"],
                )
                validation_record = _build_evaluation_record(
                    validation_training,
                    validation_backtest,
                    objective_name,
                    automl_config,
                    split=validation_split,
                )
        except RuntimeError:
            continue
        trial_records[trial.number] = _build_trial_record(overrides, search_record, validation_record)

    selection_report = _build_trial_selection_report(completed_trials, trial_records, objective_name, automl_config)
    best_trial_report = selection_report["trial_reports"][0]
    best_trial_number = int(best_trial_report["number"])
    best_optuna_trial = study.best_trial
    best_overrides = copy.deepcopy(best_trial_report["overrides"])
    best_overrides_summary = _json_ready(_clone_value(best_overrides))

    validation_holdout = _build_validation_holdout_report(best_trial_report, holdout_plan)

    locked_holdout = _evaluate_locked_holdout(
        base_config=base_config,
        best_overrides=best_overrides,
        pipeline_class=pipeline_class,
        trial_step_classes=trial_step_classes,
        full_state_bundle=full_state_bundle,
        holdout_plan=holdout_plan,
    )

    top_trials = []
    for report in selection_report["trial_reports"][:5]:
        top_trials.append(
            {
                "number": int(report["number"]),
                "value": float(report["selection_value"]),
                "raw_value": float(report["raw_objective_value"]),
                "params": report["params"],
                "training": report["training"],
                "backtest": report["backtest"],
                "search_metrics": report["search_metrics"],
                "validation_metrics": report["validation_metrics"],
                "meets_minimum_dsr_threshold": report["meets_minimum_dsr_threshold"],
                "overfitting": report["overfitting"],
            }
        )

    summary = {
        "study_name": study.study_name,
        "storage": str(storage_path),
        "objective": objective_name,
        "selection_metric": selection_report["selection_metric"],
        "selection_mode": selection_report["selection_mode"],
        "feature_schema_version": base_config.get("features", {}).get("schema_version"),
        "best_value": float(best_trial_report["selection_value"]),
        "best_value_raw": float(best_trial_report["raw_objective_value"]),
        "best_value_penalized": float(best_trial_report["selection_value"]),
        "optuna_best_value": float(best_optuna_trial.value),
        "best_trial_number": best_trial_number,
        "optuna_best_trial_number": int(best_optuna_trial.number),
        "best_params": best_trial_report["params"],
        "best_overrides": best_overrides_summary,
        "best_backtest": best_trial_report["backtest"],
        "best_training": best_trial_report["training"],
        "best_overfitting": best_trial_report["overfitting"],
        "validation_holdout": validation_holdout,
        "locked_holdout": locked_holdout,
        "overfitting_diagnostics": selection_report["diagnostics"],
        "top_trials": top_trials,
        "trial_count": len(completed_trials),
    }

    engine = getattr(getattr(study, "_storage", None), "engine", None)
    if engine is not None:
        try:
            engine.dispose()
        except Exception:
            pass

    return summary