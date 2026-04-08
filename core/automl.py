"""AutoML search helpers for the research pipeline."""

import copy
from pathlib import Path

import numpy as np

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
    },
    "labels": {
        "pt_mult": {"type": "float", "low": 1.0, "high": 3.0, "step": 0.5},
        "sl_mult": {"type": "float", "low": 1.0, "high": 3.0, "step": 0.5},
        "max_holding": {"type": "categorical", "choices": [12, 24, 48]},
        "min_return": {"type": "categorical", "choices": [0.0, 0.0005, 0.001, 0.002]},
        "volatility_window": {"type": "categorical", "choices": [12, 24, 48]},
    },
    "regime": {
        "n_regimes": {"type": "categorical", "choices": [2, 3, 4]},
    },
    "model": {
        "type": {"type": "categorical", "choices": ["rf", "gbm", "logistic"]},
        "gap": {"type": "categorical", "choices": [12, 24, 48]},
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
            },
            "logistic": {
                "c": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
            },
        },
    },
    "signals": {
        "threshold": {"type": "categorical", "choices": [0.02, 0.05, 0.08, 0.12]},
        "fraction": {"type": "categorical", "choices": [0.25, 0.5, 0.75]},
    },
}


def _resolve_study_name(base_config, automl_config):
    data_config = base_config.get("data", {})
    objective = automl_config.get("objective", "composite")
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
    return {key: backtest.get(key) for key in keys}


def compute_objective_value(objective_name, training, backtest, automl_config=None):
    """Compute the scalar objective value used by the AutoML study."""
    automl_config = automl_config or {}
    objective_name = objective_name or "composite"

    if objective_name == "net_profit_pct":
        return float(backtest.get("net_profit_pct", 0.0))
    if objective_name == "sharpe_ratio":
        return float(backtest.get("sharpe_ratio", 0.0))
    if objective_name == "profit_factor":
        profit_factor = backtest.get("profit_factor", 0.0)
        if not np.isfinite(profit_factor):
            return float(automl_config.get("profit_factor_cap", 5.0))
        return float(profit_factor)
    if objective_name == "calmar_ratio":
        return float(backtest.get("calmar_ratio", 0.0))

    profit_factor = backtest.get("profit_factor", 0.0)
    if not np.isfinite(profit_factor):
        profit_factor = automl_config.get("profit_factor_cap", 5.0)
    profit_factor = min(float(profit_factor), automl_config.get("profit_factor_cap", 5.0))

    score = (
        automl_config.get("weight_net_profit_pct", 2.0) * float(backtest.get("net_profit_pct", 0.0))
        + automl_config.get("weight_sharpe", 0.25) * float(backtest.get("sharpe_ratio", 0.0))
        + automl_config.get("weight_profit_factor", 0.25) * profit_factor
        - automl_config.get("weight_max_drawdown", 2.0) * abs(float(backtest.get("max_drawdown", 0.0)))
        + automl_config.get("weight_f1", 0.05) * float(training.get("avg_f1_macro", 0.0))
    )

    min_trades = automl_config.get("min_trades", 10)
    total_trades = int(backtest.get("total_trades", 0))
    if total_trades < min_trades:
        score -= automl_config.get("trade_count_penalty", 0.05) * (min_trades - total_trades)

    return float(score)


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
        if feature_overrides:
            overrides["features"] = feature_overrides

    label_space = search_space.get("labels", {})
    if label_space:
        label_overrides = {}
        pt_mult = _sample_from_spec(trial, "labels.pt_mult", label_space["pt_mult"])
        sl_mult = _sample_from_spec(trial, "labels.sl_mult", label_space["sl_mult"])
        label_overrides["pt_sl"] = (pt_mult, sl_mult)
        for key in ["max_holding", "min_return", "volatility_window"]:
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
        for key in ["threshold", "fraction"]:
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

    raw_data = base_pipeline.require("raw_data")
    indicator_data = base_pipeline.require("data").copy()

    study = optuna.create_study(
        direction="maximize",
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
        candidate.state["raw_data"] = raw_data
        candidate.state["data"] = indicator_data.copy()

        try:
            for step_name in [
                "build_features",
                "detect_regimes",
                "build_labels",
                "align_data",
                "compute_sample_weights",
                "train_models",
                "generate_signals",
                "run_backtest",
            ]:
                candidate.run_step(step_name)
        except RuntimeError as exc:
            if "No walk-forward folds were generated" in str(exc):
                raise optuna.TrialPruned(str(exc)) from exc
            raise

        training = candidate.state["training"]
        backtest = candidate.state["backtest"]
        value = compute_objective_value(automl_config.get("objective", "composite"), training, backtest, automl_config)

        trial.set_user_attr("overrides", _json_ready(_clone_value(overrides)))
        trial.set_user_attr("training", _json_ready(_summarize_training(training)))
        trial.set_user_attr("backtest", _json_ready(_summarize_backtest(backtest)))
        return value

    study.optimize(
        objective,
        n_trials=automl_config.get("n_trials", 25),
        gc_after_trial=automl_config.get("gc_after_trial", True),
        show_progress_bar=False,
        catch=(ValueError, RuntimeError),
    )

    completed_trials = [trial for trial in study.trials if trial.value is not None]
    if not completed_trials:
        raise RuntimeError("AutoML finished without any completed trials")

    top_trials = []
    for trial in sorted(completed_trials, key=lambda item: item.value, reverse=True)[:5]:
        top_trials.append(
            {
                "number": trial.number,
                "value": trial.value,
                "params": _json_ready(trial.params),
                "backtest": trial.user_attrs.get("backtest", {}),
            }
        )

    return {
        "study_name": study.study_name,
        "storage": str(storage_path),
        "objective": automl_config.get("objective", "composite"),
        "feature_schema_version": base_config.get("features", {}).get("schema_version"),
        "best_value": study.best_value,
        "best_params": _json_ready(study.best_trial.params),
        "best_overrides": study.best_trial.user_attrs.get("overrides", {}),
        "best_backtest": study.best_trial.user_attrs.get("backtest", {}),
        "best_training": study.best_trial.user_attrs.get("training", {}),
        "top_trials": top_trials,
        "trial_count": len(completed_trials),
    }