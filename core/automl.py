"""AutoML search helpers for the research pipeline."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:  # pragma: no cover - handled at runtime when AutoML is enabled
    optuna = None
    TPESampler = None

from .backtest import probabilistic_sharpe_ratio, deflated_sharpe_ratio


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


def _seed_candidate_state(candidate, state_bundle):
    for key, value in state_bundle.items():
        if value is None:
            continue
        candidate.state[key] = _slice_temporal_value(value)


def _resolve_locked_holdout_plan(raw_data, automl_config):
    plan = {
        "enabled": False,
        "reason": None,
        "search_rows": int(len(raw_data)),
        "holdout_rows": 0,
        "start_timestamp": None,
        "end_timestamp": None,
        "search_end_timestamp": None,
    }

    if raw_data is None or len(raw_data) < 2:
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

    min_search_rows = int(automl_config.get("locked_holdout_min_search_rows", 100))
    if not explicit_holdout and len(raw_data) - holdout_rows < min_search_rows:
        holdout_rows = len(raw_data) - min_search_rows
    if holdout_rows <= 0:
        plan["reason"] = "insufficient_search_rows"
        return plan

    search_rows = int(len(raw_data) - holdout_rows)
    plan.update(
        {
            "enabled": True,
            "search_rows": search_rows,
            "holdout_rows": holdout_rows,
            "start_timestamp": raw_data.index[search_rows],
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
        "probabilistic_sharpe_ratio",
        "deflated_sharpe_ratio",
        "skewness",
        "kurtosis",
        "sortino_ratio",
        "calmar_ratio",
        "profit_factor",
        "max_drawdown",
        "total_trades",
        "win_rate",
        "ending_equity",
    ]
    return {key: backtest.get(key) for key in keys}


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


def _evaluate_locked_holdout(base_config, best_overrides, pipeline_class, trial_step_classes, full_state_bundle, holdout_plan):
    report = {
        "enabled": bool(holdout_plan.get("enabled", False)),
        "reason": holdout_plan.get("reason"),
        "start_timestamp": _json_ready(holdout_plan.get("start_timestamp")),
        "end_timestamp": _json_ready(holdout_plan.get("end_timestamp")),
        "search_rows": int(holdout_plan.get("search_rows", 0)),
        "holdout_rows": int(holdout_plan.get("holdout_rows", 0)),
        "aligned_search_rows": 0,
        "aligned_holdout_rows": 0,
        "training": None,
        "backtest": None,
    }
    if not holdout_plan.get("enabled"):
        return report

    candidate_config = copy.deepcopy(base_config)
    _deep_merge(candidate_config, copy.deepcopy(best_overrides or {}))
    candidate_config["automl"] = {**candidate_config.get("automl", {}), "enabled": False}

    candidate = pipeline_class(candidate_config, steps=trial_step_classes)
    _seed_candidate_state(candidate, full_state_bundle)
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
    holdout_start = holdout_plan["start_timestamp"]
    aligned_search_rows = int((aligned_index < holdout_start).sum())
    aligned_holdout_rows = int((aligned_index >= holdout_start).sum())
    report["aligned_search_rows"] = aligned_search_rows
    report["aligned_holdout_rows"] = aligned_holdout_rows
    if aligned_search_rows <= 0 or aligned_holdout_rows <= 0:
        report["reason"] = "aligned_split_empty"
        return report

    candidate.config["model"] = {
        **candidate.config.get("model", {}),
        "n_splits": 1,
        "train_size": aligned_search_rows,
        "test_size": aligned_holdout_rows,
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

    report["training"] = _json_ready(_summarize_training(candidate.state["training"]))
    report["backtest"] = _json_ready(_summarize_backtest(candidate.state["backtest"]))
    return report


def compute_objective_value(objective_name, training, backtest, automl_config=None):
    """Compute institutional-grade multi-objective values for AutoML.
    
    Returns a tuple: (financial_score, model_integrity_score).
    """
    automl_config = automl_config or {}
    
    # 1. Financial Score (Objective 1)
    # We use per-bar stats for PSR to maintain statistical validity with high N.
    per_bar_sr = float(backtest.get("per_bar_sharpe", 0.0))
    sk = float(backtest.get("skewness", 0.0))
    kurt = float(backtest.get("kurtosis", 0.0))
    n_obs = len(backtest.get("equity_curve", []))
    
    # Zero-Benchmark PSR
    psr = probabilistic_sharpe_ratio(per_bar_sr, sk, kurt, n_obs, benchmark_sr=0.0)
    
    # Confidence Penalization (Soft filtering)
    # If confidence < 95%, we apply a sharp non-linear decay.
    financial_score = psr
    if psr < 0.95:
        financial_score = psr * np.exp(5.0 * (psr - 0.95))
        
    # Hard Constraints (Soft Pruning)
    total_trades = int(backtest.get("total_trades", 0))
    min_trades = int(automl_config.get("min_total_trades", 15))
    if total_trades < min_trades:
        # Severe penalty for under-trading
        financial_score -= 1e6 * (1.0 - total_trades / max(1, min_trades))
        
    max_dd = abs(float(backtest.get("max_drawdown", 0.0)))
    if max_dd > float(automl_config.get("max_drawdown_limit", 0.4)):
        financial_score -= 1e6
        
    # 2. Model Integrity Score (Objective 2)
    # Balanced composite of accuracy, log loss, and calibration.
    directional_accuracy = _resolve_metric(training, "avg_directional_accuracy", fallback="avg_accuracy") or 0.0
    log_loss_value = _resolve_metric(training, "avg_log_loss") or 1.0
    calibration_error = _resolve_metric(training, "avg_calibration_error") or 0.5
    
    # Normalize components to [0, 1] range where possible
    # Negative Log Loss is clipped and shifted
    integrity_log_loss = np.clip(1.0 - log_loss_value, 0.0, 1.0)
    integrity_calibration = np.clip(1.0 - (calibration_error * 2.0), 0.0, 1.0) # 0.5 error is "zero" integrity
    
    # Weighted composite for integrity
    model_integrity_score = (
        0.50 * directional_accuracy +
        0.30 * integrity_log_loss +
        0.20 * integrity_calibration
    )
    
    return float(financial_score), float(model_integrity_score)


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
        if "pt_mult" in label_space and "sl_mult" in label_space:
            pt_mult = _sample_from_spec(trial, "labels.pt_mult", label_space["pt_mult"])
            sl_mult = _sample_from_spec(trial, "labels.sl_mult", label_space["sl_mult"])
            label_overrides["pt_sl"] = (pt_mult, sl_mult)
        for key in ["max_holding", "min_return", "volatility_window", "collision_penalty"]:
            if key in label_space:
                label_overrides[key] = _sample_from_spec(trial, f"labels.{key}", label_space[key])
        if label_overrides:
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

    full_state_bundle = _build_state_bundle(base_pipeline)
    holdout_plan = _resolve_locked_holdout_plan(full_state_bundle["raw_data"], automl_config)
    search_state_bundle = _build_state_bundle(base_pipeline)
    if holdout_plan["enabled"]:
        search_index = full_state_bundle["raw_data"].index[: holdout_plan["search_rows"]]
        search_state_bundle["raw_data"] = full_state_bundle["raw_data"].loc[search_index].copy()
        search_state_bundle["data"] = full_state_bundle["data"].reindex(search_index).copy()
        search_end_timestamp = holdout_plan["search_end_timestamp"]
        search_state_bundle["futures_context"] = _slice_temporal_value(
            full_state_bundle.get("futures_context"),
            end_timestamp=search_end_timestamp,
        )
        search_state_bundle["cross_asset_context"] = _slice_temporal_value(
            full_state_bundle.get("cross_asset_context"),
            end_timestamp=search_end_timestamp,
        )

    study = optuna.create_study(
        directions=["maximize", "maximize"],
        sampler=sampler,
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
    )

    def objective(trial):
        overrides = _sample_trial_overrides(trial, search_space)
        candidate_config = copy.deepcopy(base_config)
        _deep_merge(candidate_config, overrides)
        candidate_config["automl"] = {**candidate_config.get("automl", {}), "enabled": False}

        candidate = pipeline_class(candidate_config, steps=trial_step_classes)
        _seed_candidate_state(candidate, search_state_bundle)

        try:
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
        except RuntimeError as exc:
            if "No walk-forward folds were generated" in str(exc):
                raise optuna.TrialPruned(str(exc)) from exc
            raise

        training = candidate.state["training"]
        backtest = candidate.state["backtest"]
        
        # Track trial metrics in user attributes for cross-trial calculations (like DSR)
        current_sr = float(backtest.get("sharpe_ratio", 0.0))
        trial.set_user_attr("sharpe_ratio", current_sr)

        # To compute DSR, we need the variance of SRs across all trials so far
        completed_trials = [t for t in trial.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        all_srs = [t.user_attrs.get("sharpe_ratio", 0.0) for t in completed_trials]
        all_srs.append(current_sr)
        
        sr_variance = float(np.var(all_srs)) if len(all_srs) > 1 else 0.0
        
        # Inject running stats into automl_config for objective computation
        running_config = copy.deepcopy(automl_config)
        running_config["sr_variance"] = sr_variance
        running_config["n_trials"] = len(all_srs)

        value = compute_objective_value(
            _normalize_objective_name(automl_config.get("objective", "accuracy_first")), 
            training, 
            backtest, 
            running_config
        )

        trial.set_user_attr("overrides", _json_ready(_clone_value(overrides)))
        trial.set_user_attr("training", _json_ready(_summarize_training(training)))
        trial.set_user_attr("backtest", _json_ready(_summarize_backtest(backtest)))
        return value # Returns tuple (financial, integrity)

    study.optimize(
        objective,
        n_trials=automl_config.get("n_trials", 25),
        gc_after_trial=automl_config.get("gc_after_trial", True),
        show_progress_bar=False,
        catch=(ValueError, RuntimeError),
    )

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        raise RuntimeError("AutoML finished without any completed trials")

    # Selection Heuristic: 60% Financial / 40% Model Integrity
    def selection_score(t):
        f_score, i_score = t.values
        return 0.6 * f_score + 0.4 * i_score

    best_trial = max(completed_trials, key=selection_score)
    best_overrides = best_trial.user_attrs.get("overrides", {})
    
    locked_holdout = _evaluate_locked_holdout(
        base_config=base_config,
        best_overrides=best_overrides,
        pipeline_class=pipeline_class,
        trial_step_classes=trial_step_classes,
        full_state_bundle=full_state_bundle,
        holdout_plan=holdout_plan,
    )

    top_trials = []
    # Sort by the selection heuristic for the top 5 report
    for trial in sorted(completed_trials, key=selection_score, reverse=True)[:5]:
        top_trials.append(
            {
                "number": trial.number,
                "values": trial.values,
                "params": _json_ready(trial.params),
                "training": trial.user_attrs.get("training", {}),
                "backtest": trial.user_attrs.get("backtest", {}),
            }
        )

    summary = {
        "study_name": study.study_name,
        "storage": str(storage_path),
        "objective": "multi_objective_psr_integrity",
        "feature_schema_version": base_config.get("features", {}).get("schema_version"),
        "best_values": best_trial.values,
        "best_params": _json_ready(best_trial.params),
        "best_overrides": best_overrides,
        "best_backtest": best_trial.user_attrs.get("backtest", {}),
        "best_training": best_trial.user_attrs.get("training", {}),
        "locked_holdout": locked_holdout,
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