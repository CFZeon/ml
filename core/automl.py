"""AutoML search helpers for the research pipeline."""

import copy
import hashlib
import json
from itertools import combinations
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from .promotion import (
    build_promotion_gate_check_map,
    create_promotion_eligibility_report,
    evaluate_execution_realism_gate,
    finalize_promotion_eligibility_report,
    resolve_canonical_promotion_score,
    resolve_promotion_gate_mode,
    set_promotion_score,
    upsert_promotion_gate,
)
from .registry import LocalRegistryStore, evaluate_challenger_promotion
from .stat_tests import compute_post_selection_inference

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
}


_NORMAL_DIST = NormalDist()
_EULER_MASCHERONI = 0.5772156649015329
_BACKTEST_OBJECTIVES = {
    "sharpe_ratio",
    "net_profit_pct",
    "profit_factor",
    "calmar_ratio",
    "risk_adjusted_after_costs",
    "benchmark_excess_sharpe",
    "net_profit_pct_vs_benchmark",
}
_CLASSIFICATION_OBJECTIVES = {
    "directional_accuracy",
    "accuracy_first",
    "neg_log_loss",
    "log_loss",
    "neg_brier_score",
    "brier_score",
    "neg_calibration_error",
    "calibration_error",
}


def _normalize_objective_name(objective_name):
    objective_name = (objective_name or "risk_adjusted_after_costs").lower()
    aliases = {
        "composite": "accuracy_first",
        "trading_first": "risk_adjusted_after_costs",
        "after_cost_sharpe": "risk_adjusted_after_costs",
    }
    if objective_name in aliases:
        return aliases[objective_name]
    if objective_name == "composite":
        return "accuracy_first"
    return objective_name


def _resolve_study_name(base_config, automl_config):
    data_config = base_config.get("data", {})
    objective = _normalize_objective_name(automl_config.get("objective", "risk_adjusted_after_costs"))
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


def _stable_payload_hash(payload):
    serialized = json.dumps(
        _json_ready(payload),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _build_selection_snapshot(best_trial_report):
    selection_policy = best_trial_report.get("selection_policy") or {}
    snapshot = {
        "trial_number": int(best_trial_report["number"]),
        "trial_params": copy.deepcopy(best_trial_report.get("params") or {}),
        "frozen_overrides": copy.deepcopy(best_trial_report.get("overrides") or {}),
        "validation_metrics": copy.deepcopy(best_trial_report.get("validation_metrics") or {}),
        "eligibility": {
            "eligible": bool(selection_policy.get("eligible", False)),
            "eligible_before_post_checks": bool(selection_policy.get("eligible_before_post_checks", False)),
            "eligibility_checks": copy.deepcopy(selection_policy.get("eligibility_checks") or {}),
            "eligibility_reasons": list(selection_policy.get("eligibility_reasons") or []),
        },
        "selection_value": _coerce_float(best_trial_report.get("selection_value")),
        "raw_objective_value": _coerce_float(best_trial_report.get("raw_objective_value")),
        "selection_timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    snapshot["candidate_hash"] = _stable_payload_hash(
        {
            "trial_number": snapshot["trial_number"],
            "trial_params": snapshot["trial_params"],
            "frozen_overrides": snapshot["frozen_overrides"],
        }
    )
    return _json_ready(snapshot)


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
        "data_lineage": copy.deepcopy(base_pipeline.state.get("data_lineage")),
        "symbol_filters": copy.deepcopy(base_pipeline.state.get("symbol_filters")),
        "symbol_lifecycle": _slice_temporal_value(base_pipeline.state.get("symbol_lifecycle")),
        "universe_policy": copy.deepcopy(base_pipeline.state.get("universe_policy")),
        "universe_snapshot": _slice_temporal_value(base_pipeline.state.get("universe_snapshot")),
        "universe_snapshot_meta": copy.deepcopy(base_pipeline.state.get("universe_snapshot_meta")),
        "eligible_symbols": copy.deepcopy(base_pipeline.state.get("eligible_symbols")),
        "universe_report": _slice_temporal_value(base_pipeline.state.get("universe_report")),
    }


def _build_window_state_bundle(full_state_bundle, start_timestamp=None, end_timestamp=None):
    raw_data = pd.DataFrame(full_state_bundle["raw_data"], copy=True)
    if start_timestamp is not None:
        raw_data = raw_data.loc[raw_data.index >= start_timestamp].copy()
    if end_timestamp is not None:
        raw_data = raw_data.loc[raw_data.index <= end_timestamp].copy()

    slice_index = raw_data.index
    return {
        "raw_data": raw_data,
        "data": full_state_bundle["data"].reindex(slice_index).copy(),
        "indicator_run": _slice_temporal_value(
            full_state_bundle.get("indicator_run"),
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        ),
        "futures_context": _slice_temporal_value(
            full_state_bundle.get("futures_context"),
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        ),
        "cross_asset_context": _slice_temporal_value(
            full_state_bundle.get("cross_asset_context"),
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        ),
        "data_lineage": copy.deepcopy(full_state_bundle.get("data_lineage")),
        "symbol_filters": copy.deepcopy(full_state_bundle.get("symbol_filters")),
        "symbol_lifecycle": _slice_temporal_value(
            full_state_bundle.get("symbol_lifecycle"),
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        ),
        "universe_policy": copy.deepcopy(full_state_bundle.get("universe_policy")),
        "universe_snapshot": _slice_temporal_value(
            full_state_bundle.get("universe_snapshot"),
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        ),
        "universe_snapshot_meta": copy.deepcopy(full_state_bundle.get("universe_snapshot_meta")),
        "eligible_symbols": copy.deepcopy(full_state_bundle.get("eligible_symbols")),
        "universe_report": _slice_temporal_value(
            full_state_bundle.get("universe_report"),
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        ),
    }


def _build_temporal_state_bundle(full_state_bundle, end_timestamp=None):
    return _build_window_state_bundle(full_state_bundle, end_timestamp=end_timestamp)


def _resolve_stage_gap_defaults(base_config, automl_config):
    labels_config = base_config.get("labels", {}) or {}
    backtest_config = base_config.get("backtest", {}) or {}
    model_config = base_config.get("model", {}) or {}

    label_gap = int(
        labels_config.get(
            "max_holding",
            labels_config.get("horizon", 0),
        ) or 0
    )
    configured_embargo = max(
        0,
        int(
            automl_config.get(
                "stage_embargo_bars",
                automl_config.get(
                    "embargo_bars",
                    model_config.get("embargo_bars", model_config.get("gap", 0)),
                ),
            ) or 0
        ),
    )
    signal_delay = backtest_config.get("signal_delay_bars")
    if signal_delay is None:
        signal_delay = 2 if backtest_config.get("use_open_execution", True) else 1
    signal_delay = max(0, int(signal_delay))
    execution_policy = dict(backtest_config.get("execution_policy") or {})
    action_latency_bars = execution_policy.get("action_latency_bars", backtest_config.get("action_latency_bars", 0))
    action_latency_bars = max(0, int(action_latency_bars or 0))

    default_gap = max(label_gap, signal_delay + action_latency_bars, configured_embargo)
    search_validation_gap = automl_config.get("search_validation_gap_bars")
    validation_holdout_gap = automl_config.get("validation_holdout_gap_bars")
    return {
        "default_gap_bars": int(default_gap),
        "search_validation_gap_bars": max(0, int(default_gap if search_validation_gap is None else search_validation_gap)),
        "validation_holdout_gap_bars": max(0, int(default_gap if validation_holdout_gap is None else validation_holdout_gap)),
        "label_gap_bars": int(label_gap),
        "signal_delay_bars": int(signal_delay),
        "action_latency_bars": int(action_latency_bars),
        "configured_embargo_bars": int(configured_embargo),
    }


def _seed_candidate_state(candidate, state_bundle):
    for key, value in state_bundle.items():
        if value is None:
            continue
        candidate.state[key] = _slice_temporal_value(value)


def _resolve_holdout_plan(raw_data, automl_config, base_config=None):
    gap_defaults = _resolve_stage_gap_defaults(base_config or {}, automl_config)
    plan = {
        "enabled": False,
        "reason": None,
        "search_rows": int(len(raw_data)),
        "validation_rows": 0,
        "holdout_rows": 0,
        "search_validation_gap_bars": int(gap_defaults["search_validation_gap_bars"]),
        "validation_holdout_gap_bars": int(gap_defaults["validation_holdout_gap_bars"]),
        "default_stage_gap_bars": int(gap_defaults["default_gap_bars"]),
        "label_gap_bars": int(gap_defaults["label_gap_bars"]),
        "signal_delay_bars": int(gap_defaults["signal_delay_bars"]),
        "configured_embargo_bars": int(gap_defaults["configured_embargo_bars"]),
        "dropped_gap_rows": int(gap_defaults["search_validation_gap_bars"] + gap_defaults["validation_holdout_gap_bars"]),
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
    search_validation_gap_rows = int(plan["search_validation_gap_bars"])
    validation_holdout_gap_rows = int(plan["validation_holdout_gap_bars"])
    shortfall = max(
        min_search_rows - (len(raw_data) - validation_rows - holdout_rows - search_validation_gap_rows - validation_holdout_gap_rows),
        0,
    )
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

    search_rows = int(len(raw_data) - validation_rows - holdout_rows - search_validation_gap_rows - validation_holdout_gap_rows)
    if search_rows <= 0:
        plan["reason"] = "insufficient_search_rows"
        return plan

    validation_start_index = search_rows + search_validation_gap_rows
    validation_end_index = validation_start_index + validation_rows - 1
    holdout_start_index = validation_end_index + 1 + validation_holdout_gap_rows
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
            "search_validation_gap_start_timestamp": (
                raw_data.index[search_rows] if search_validation_gap_rows > 0 else None
            ),
            "search_validation_gap_end_timestamp": (
                raw_data.index[validation_start_index - 1] if search_validation_gap_rows > 0 else None
            ),
            "validation_holdout_gap_start_timestamp": (
                raw_data.index[validation_end_index + 1] if validation_holdout_gap_rows > 0 else None
            ),
            "validation_holdout_gap_end_timestamp": (
                raw_data.index[holdout_start_index - 1] if validation_holdout_gap_rows > 0 else None
            ),
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


def _validate_signal_policy_search_space(search_space):
    signal_space = copy.deepcopy((search_space or {}).get("signals") or {})
    if not signal_space:
        return

    disallowed_keys = ", ".join(sorted(signal_space))
    raise ValueError(
        "AutoML signal-policy search is disabled. Remove automl.search_space.signals entries "
        f"({disallowed_keys}) and keep signal policy outside the search space."
    )


def _validate_trial_overrides(overrides):
    signal_overrides = copy.deepcopy((overrides or {}).get("signals") or {})
    if not signal_overrides:
        return

    disallowed_keys = ", ".join(sorted(signal_overrides))
    raise ValueError(
        "AutoML signal-policy overrides are disabled. Remove signals overrides from trial sampling "
        f"({disallowed_keys}) and keep signal policy outside the search space."
    )


def _resolve_primary_training_payload(training):
    executable_validation = training.get("executable_validation") or {}
    replay_training = executable_validation.get("training") if executable_validation.get("enabled") else None
    if isinstance(replay_training, dict):
        return replay_training, "executable_validation"
    return training, "validation"


def _summarize_training(training):
    primary_training, primary_source = _resolve_primary_training_payload(training)
    feature_selection = training.get("feature_selection") or {}
    bootstrap = training.get("bootstrap") or {}
    feature_governance = training.get("feature_governance") or {}
    operational_monitoring = training.get("operational_monitoring") or {}
    cross_venue_integrity = training.get("cross_venue_integrity") or {}
    signal_decay = training.get("signal_decay") or {}
    return {
        "avg_accuracy": primary_training.get("avg_accuracy"),
        "avg_f1_macro": primary_training.get("avg_f1_macro"),
        "avg_directional_accuracy": primary_training.get("avg_directional_accuracy"),
        "avg_directional_f1_macro": primary_training.get("avg_directional_f1_macro"),
        "avg_log_loss": primary_training.get("avg_log_loss"),
        "avg_brier_score": primary_training.get("avg_brier_score"),
        "avg_calibration_error": primary_training.get("avg_calibration_error"),
        "headline_metrics": primary_training.get("headline_metrics", {}),
        "selection_metrics_source": primary_source,
        "feature_selection": {
            "enabled": bool(feature_selection.get("enabled", False)),
            "avg_input_features": feature_selection.get("avg_input_features"),
            "avg_selected_features": feature_selection.get("avg_selected_features"),
        },
        "bootstrap": {
            "model_type": bootstrap.get("model_type"),
            "used_in_any_fold": bootstrap.get("used_in_any_fold"),
            "warning_count": bootstrap.get("warning_count"),
            "folds": bootstrap.get("folds", []),
        },
        "feature_governance": {
            "retirement": feature_governance.get("retirement", {}),
            "admission_summary": feature_governance.get("admission_summary", {}),
        },
        "operational_monitoring": {
            "healthy": bool(operational_monitoring.get("healthy", True)),
            "reasons": list(operational_monitoring.get("reasons", [])),
            "summary": operational_monitoring.get("summary", {}),
            "artifacts": operational_monitoring.get("artifacts", {}),
        },
        "cross_venue_integrity": {
            "kind": cross_venue_integrity.get("kind"),
            "promotion_pass": bool(cross_venue_integrity.get("promotion_pass", True)),
            "gate_mode": cross_venue_integrity.get("gate_mode"),
            "reasons": list(cross_venue_integrity.get("reasons", [])),
            "warnings": list(cross_venue_integrity.get("warnings", [])),
            "venues": cross_venue_integrity.get("venues", {}),
            "self_consistency": cross_venue_integrity.get("self_consistency", {}),
            "divergence": cross_venue_integrity.get("divergence", {}),
            "overlay_columns": list(cross_venue_integrity.get("overlay_columns", [])),
        },
        "signal_decay": signal_decay,
        "promotion_gates": training.get("promotion_gates", {}),
        "fold_stability": primary_training.get("fold_stability", training.get("fold_stability")),
        "fold_count": len(primary_training.get("fold_metrics", [])),
        "diagnostic_validation": training.get("diagnostic_validation", {}),
        "executable_validation": {
            "enabled": bool((training.get("executable_validation") or {}).get("enabled", False)),
            "source": (training.get("executable_validation") or {}).get("source"),
            "method": (((training.get("executable_validation") or {}).get("training") or {}).get("validation") or {}).get("method"),
        },
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
        "fill_ratio",
        "cancelled_orders",
        "unfilled_notional",
        "average_action_delay_bars",
        "average_fill_delay_bars",
        "max_fill_delay_bars",
    ]
    summary = {key: backtest.get(key) for key in keys}
    equity_curve = backtest.get("equity_curve")
    summary["bar_count"] = int(len(equity_curve)) if isinstance(equity_curve, pd.Series) else None
    if backtest.get("statistical_significance") is not None:
        summary["statistical_significance"] = backtest.get("statistical_significance")
    if backtest.get("signal_decay") is not None:
        summary["signal_decay"] = backtest.get("signal_decay")
    return summary


def _resolve_metric(training, key, fallback=None):
    primary_training, _ = _resolve_primary_training_payload(training)
    value = primary_training.get(key)
    if value is None and primary_training is not training:
        value = training.get(key)
    if value is None and fallback is not None:
        value = primary_training.get(fallback)
        if value is None and primary_training is not training:
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
    train_end_timestamp,
    test_start_timestamp,
    excluded_intervals=None,
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
    aligned_mask = (aligned_index <= train_end_timestamp) | (aligned_index >= test_start_timestamp)
    for interval in excluded_intervals or []:
        interval_start, interval_end = interval
        if interval_start is None or interval_end is None:
            continue
        aligned_mask &= ~((aligned_index >= interval_start) & (aligned_index <= interval_end))
    candidate.state["X"] = candidate.state["X"].loc[aligned_mask].copy()
    candidate.state["y"] = candidate.state["y"].loc[aligned_mask].copy()
    candidate.state["labels_aligned"] = candidate.state["labels_aligned"].loc[aligned_mask].copy()

    aligned_index = candidate.state["X"].index
    aligned_train_rows = int((aligned_index <= train_end_timestamp).sum())
    aligned_test_rows = int((aligned_index >= test_start_timestamp).sum())
    aligned_gap_rows = int((~aligned_mask).sum())
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
        "aligned_gap_rows": int(aligned_gap_rows),
        "train_end_timestamp": _json_ready(train_end_timestamp),
        "test_start_timestamp": _json_ready(test_start_timestamp),
        "excluded_intervals": _json_ready(excluded_intervals or []),
    }


def _build_validation_holdout_report(best_trial_report, holdout_plan):
    report = {
        "enabled": bool(holdout_plan.get("enabled", False)),
        "reason": holdout_plan.get("reason"),
        "start_timestamp": _json_ready(holdout_plan.get("validation_start_timestamp")),
        "end_timestamp": _json_ready(holdout_plan.get("validation_end_timestamp")),
        "search_rows": int(holdout_plan.get("search_rows", 0)),
        "validation_rows": int(holdout_plan.get("validation_rows", 0)),
        "search_validation_gap_rows": int(holdout_plan.get("search_validation_gap_bars", 0)),
        "search_validation_gap_start_timestamp": _json_ready(holdout_plan.get("search_validation_gap_start_timestamp")),
        "search_validation_gap_end_timestamp": _json_ready(holdout_plan.get("search_validation_gap_end_timestamp")),
        "stage_gap_rows_dropped": int(holdout_plan.get("search_validation_gap_bars", 0)),
        "aligned_search_rows": 0,
        "aligned_validation_rows": 0,
        "aligned_gap_rows": 0,
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
    report["aligned_gap_rows"] = int(split.get("aligned_gap_rows", 0))
    report["training"] = validation_metrics.get("training")
    report["backtest"] = validation_metrics.get("backtest")
    report["raw_objective_value"] = validation_metrics.get("raw_objective_value")
    report["selection_value"] = best_trial_report.get("selection_value")
    report["meets_minimum_dsr_threshold"] = best_trial_report.get("meets_minimum_dsr_threshold")
    return report


def _decorate_locked_holdout_report(locked_holdout_report, selection_snapshot, access_count):
    report = copy.deepcopy(locked_holdout_report or {})
    report["access_count"] = int(access_count)
    report["evaluated_once"] = bool(report.get("enabled") and access_count == 1)
    report["evaluated_after_freeze"] = bool(report.get("enabled") and selection_snapshot is not None)
    report["frozen_candidate_hash"] = (selection_snapshot or {}).get("candidate_hash")
    return report


def _build_locked_holdout_promotion_report(selection_policy, best_trial_report, locked_holdout_report):
    holdout_gap = _build_generalization_gap_report(
        best_trial_report.get("raw_objective_value"),
        (locked_holdout_report or {}).get("raw_objective_value"),
    )
    require_locked_holdout_pass = bool(selection_policy.get("require_locked_holdout_pass", False))
    holdout_value = _coerce_float((locked_holdout_report or {}).get("raw_objective_value"))
    locked_holdout_pass = True
    if (locked_holdout_report or {}).get("enabled"):
        locked_holdout_pass = bool(
            holdout_value is not None
            and holdout_value >= selection_policy.get("min_locked_holdout_score", 0.0)
            and not locked_holdout_report.get("holdout_warning", False)
        )
    locked_holdout_gap_pass = bool(
        not require_locked_holdout_pass
        or (holdout_gap.get("normalized_degradation") or 0.0)
        <= selection_policy.get("max_generalization_gap", np.inf)
    )

    promotion_reasons = []
    if require_locked_holdout_pass and not locked_holdout_pass:
        promotion_reasons.append("locked_holdout_failed")
    if not locked_holdout_gap_pass:
        promotion_reasons.append("validation_holdout_gap_above_limit")

    return {
        "generalization_gap": holdout_gap,
        "locked_holdout_pass": locked_holdout_pass,
        "locked_holdout_gap_pass": locked_holdout_gap_pass,
        "promotion_ready": not promotion_reasons,
        "promotion_reasons": promotion_reasons,
    }


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
        "validation_holdout_gap_rows": int(holdout_plan.get("validation_holdout_gap_bars", 0)),
        "validation_holdout_gap_start_timestamp": _json_ready(holdout_plan.get("validation_holdout_gap_start_timestamp")),
        "validation_holdout_gap_end_timestamp": _json_ready(holdout_plan.get("validation_holdout_gap_end_timestamp")),
        "stage_gap_rows_dropped": int(holdout_plan.get("dropped_gap_rows", 0)),
        "aligned_search_rows": 0,
        "aligned_pre_holdout_rows": 0,
        "aligned_holdout_rows": 0,
        "aligned_gap_rows": 0,
        "training": None,
        "backtest": None,
        "raw_objective_value": None,
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
            train_end_timestamp=holdout_plan["validation_end_timestamp"],
            test_start_timestamp=holdout_plan["holdout_start_timestamp"],
            excluded_intervals=[
                (
                    holdout_plan.get("search_validation_gap_start_timestamp"),
                    holdout_plan.get("search_validation_gap_end_timestamp"),
                ),
                (
                    holdout_plan.get("validation_holdout_gap_start_timestamp"),
                    holdout_plan.get("validation_holdout_gap_end_timestamp"),
                ),
            ],
        )
    except RuntimeError as exc:
        if "Aligned split empty" in str(exc):
            report["reason"] = "aligned_split_empty"
            return report
        raise

    report["aligned_search_rows"] = int(split["aligned_train_rows"])
    report["aligned_pre_holdout_rows"] = int(split["aligned_train_rows"])
    report["aligned_holdout_rows"] = int(split["aligned_test_rows"])
    report["aligned_gap_rows"] = int(split.get("aligned_gap_rows", 0))
    report["training"] = _json_ready(_summarize_training(training))
    report["backtest"] = _json_ready(_summarize_backtest(backtest))
    report["objective_diagnostics"] = _build_objective_diagnostics(
        base_config.get("automl", {}).get("objective", "risk_adjusted_after_costs"),
        report["training"],
        report["backtest"],
        base_config.get("automl", {}),
    )
    report["raw_objective_value"] = float(report["objective_diagnostics"]["final_score"])
    sharpe_ci_lower = _extract_sharpe_ci_lower(report["backtest"])
    report["holdout_warning"] = bool(sharpe_ci_lower is not None and sharpe_ci_lower < 0.0)
    return report


def _resolve_replication_config(base_config, automl_config=None, base_pipeline=None):
    config = copy.deepcopy((automl_config or {}).get("replication") or {})
    include_symbol_cohorts = bool(config.get("include_symbol_cohorts", True))
    include_window_cohorts = bool(config.get("include_window_cohorts", True))
    include_regime_slices = bool(config.get("include_regime_slices", False))

    primary_symbol = str((base_config.get("data") or {}).get("symbol", "unknown"))
    timeframe = str((base_config.get("data") or {}).get("interval", "unknown"))

    symbols = [str(symbol) for symbol in (config.get("symbols") or []) if symbol is not None]
    if not symbols and base_pipeline is not None and include_symbol_cohorts:
        cross_asset_context = base_pipeline.state.get("cross_asset_context") or {}
        symbols = [str(symbol) for symbol in cross_asset_context.keys()]

    eligible_symbols = {str(symbol) for symbol in ((base_pipeline.state.get("eligible_symbols") or []) if base_pipeline is not None else [])}
    if eligible_symbols:
        symbols = [symbol for symbol in symbols if symbol in eligible_symbols]
    symbols = [symbol for symbol in symbols if symbol != primary_symbol]

    max_symbol_cohorts = int(config.get("max_symbol_cohorts", len(symbols) if symbols else 0))
    if max_symbol_cohorts >= 0:
        symbols = symbols[:max_symbol_cohorts]

    requested_defaults = int(include_symbol_cohorts) + int(include_window_cohorts)
    return {
        "enabled": bool(config.get("enabled", False)),
        "metric": _normalize_objective_name(config.get("metric", "risk_adjusted_after_costs")),
        "min_score": float(config.get("min_score", 0.0)),
        "min_coverage": int(config.get("min_coverage", max(requested_defaults, 1) if requested_defaults else 1)),
        "min_pass_rate": float(config.get("min_pass_rate", 0.6)),
        "min_rows": int(config.get("min_rows", 64)),
        "symbols": symbols,
        "include_symbol_cohorts": include_symbol_cohorts,
        "include_window_cohorts": include_window_cohorts,
        "include_regime_slices": include_regime_slices,
        "alternate_window_count": int(config.get("alternate_window_count", 1)),
        "alternate_window_fraction": float(config.get("alternate_window_fraction", 0.5)),
        "periods": copy.deepcopy(config.get("periods") or config.get("time_windows") or []),
        "primary_symbol": primary_symbol,
        "timeframe": timeframe,
    }


def _build_symbol_replication_state_bundle(full_state_bundle, symbol, start_timestamp=None, end_timestamp=None):
    cross_asset_context = dict(full_state_bundle.get("cross_asset_context") or {})
    symbol_frame = cross_asset_context.get(symbol)
    if not isinstance(symbol_frame, pd.DataFrame) or symbol_frame.empty:
        return None

    raw_data = pd.DataFrame(symbol_frame, copy=True)
    if start_timestamp is not None:
        raw_data = raw_data.loc[raw_data.index >= start_timestamp].copy()
    if end_timestamp is not None:
        raw_data = raw_data.loc[raw_data.index <= end_timestamp].copy()
    if raw_data.empty:
        return None

    return {
        "raw_data": raw_data,
        "data": raw_data.copy(),
        "indicator_run": None,
        "futures_context": None,
        "cross_asset_context": {
            key: _slice_temporal_value(value, start_timestamp=start_timestamp, end_timestamp=end_timestamp)
            for key, value in cross_asset_context.items()
            if str(key) != str(symbol)
        },
        "data_lineage": copy.deepcopy(full_state_bundle.get("data_lineage")),
        "symbol_filters": {},
        "symbol_lifecycle": _slice_temporal_value(
            full_state_bundle.get("symbol_lifecycle"),
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        ),
        "universe_policy": copy.deepcopy(full_state_bundle.get("universe_policy")),
        "universe_snapshot": _slice_temporal_value(
            full_state_bundle.get("universe_snapshot"),
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        ),
        "universe_snapshot_meta": copy.deepcopy(full_state_bundle.get("universe_snapshot_meta")),
        "eligible_symbols": copy.deepcopy(full_state_bundle.get("eligible_symbols")),
        "universe_report": _slice_temporal_value(
            full_state_bundle.get("universe_report"),
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        ),
    }


def _build_generated_replication_periods(raw_index, replication_config, holdout_plan):
    explicit_periods = list(replication_config.get("periods") or [])
    if explicit_periods:
        periods = []
        for position, period in enumerate(explicit_periods, start=1):
            start_timestamp = period.get("start_timestamp", period.get("start"))
            end_timestamp = period.get("end_timestamp", period.get("end"))
            if start_timestamp is not None:
                start_timestamp = pd.Timestamp(start_timestamp)
            if end_timestamp is not None:
                end_timestamp = pd.Timestamp(end_timestamp)
            if start_timestamp is not None and end_timestamp is not None and start_timestamp > end_timestamp:
                continue
            periods.append(
                {
                    "cohort_id": f"period:{position}",
                    "kind": "period",
                    "label": str(period.get("label") or f"period_{position}"),
                    "start_timestamp": start_timestamp,
                    "end_timestamp": end_timestamp,
                }
            )
        return periods

    candidate_index = pd.DatetimeIndex(raw_index)
    search_end_timestamp = holdout_plan.get("search_end_timestamp") if holdout_plan else None
    if search_end_timestamp is not None:
        candidate_index = candidate_index[candidate_index <= pd.Timestamp(search_end_timestamp)]

    min_rows = int(replication_config.get("min_rows", 0))
    window_count = max(0, int(replication_config.get("alternate_window_count", 0)))
    if window_count <= 0 or len(candidate_index) < max(2, min_rows):
        return []

    window_fraction = float(replication_config.get("alternate_window_fraction", 0.5))
    window_fraction = min(max(window_fraction, 0.05), 1.0)
    window_size = int(round(len(candidate_index) * window_fraction))
    window_size = max(window_size, min_rows)
    if window_size >= len(candidate_index):
        return []

    available_start = len(candidate_index) - window_size
    if available_start <= 0:
        return []

    if window_count == 1:
        start_positions = [0]
    else:
        step = max(1, available_start // max(window_count - 1, 1))
        start_positions = sorted({min(available_start, step * position) for position in range(window_count)})

    periods = []
    for position, start_location in enumerate(start_positions, start=1):
        end_location = start_location + window_size - 1
        periods.append(
            {
                "cohort_id": f"period:{position}",
                "kind": "period",
                "label": f"period_{position}",
                "start_timestamp": candidate_index[start_location],
                "end_timestamp": candidate_index[end_location],
            }
        )
    return periods


def _build_replication_cohort_specs(base_config, full_state_bundle, holdout_plan, replication_config):
    timeframe = str((base_config.get("data") or {}).get("interval", "unknown"))
    cohort_specs = []

    if replication_config.get("include_symbol_cohorts", True):
        for symbol in replication_config.get("symbols") or []:
            state_bundle = _build_symbol_replication_state_bundle(full_state_bundle, symbol)
            cohort_specs.append(
                {
                    "cohort_id": f"symbol:{symbol}",
                    "kind": "symbol",
                    "label": symbol,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "state_bundle": state_bundle,
                    "config_overrides": {"data": {"symbol": symbol}},
                }
            )

    if replication_config.get("include_window_cohorts", True):
        for period in _build_generated_replication_periods(
            full_state_bundle["raw_data"].index,
            replication_config,
            holdout_plan,
        ):
            state_bundle = _build_window_state_bundle(
                full_state_bundle,
                start_timestamp=period.get("start_timestamp"),
                end_timestamp=period.get("end_timestamp"),
            )
            cohort_specs.append(
                {
                    "cohort_id": period["cohort_id"],
                    "kind": period["kind"],
                    "label": period["label"],
                    "symbol": replication_config.get("primary_symbol"),
                    "timeframe": timeframe,
                    "start_timestamp": period.get("start_timestamp"),
                    "end_timestamp": period.get("end_timestamp"),
                    "state_bundle": state_bundle,
                    "config_overrides": {},
                }
            )

    return cohort_specs


def _evaluate_replication_cohorts(
    base_config,
    best_overrides,
    pipeline_class,
    trial_step_classes,
    full_state_bundle,
    holdout_plan,
    base_pipeline=None,
):
    automl_config = base_config.get("automl", {}) or {}
    replication_config = _resolve_replication_config(
        base_config,
        automl_config,
        base_pipeline=base_pipeline,
    )
    report = {
        "enabled": bool(replication_config.get("enabled", False)),
        "kind": "replication",
        "metric": replication_config.get("metric"),
        "primary_symbol": replication_config.get("primary_symbol"),
        "timeframe": replication_config.get("timeframe"),
        "min_score": float(replication_config.get("min_score", 0.0)),
        "min_coverage": int(replication_config.get("min_coverage", 0)),
        "min_pass_rate": float(replication_config.get("min_pass_rate", 0.0)),
        "min_rows": int(replication_config.get("min_rows", 0)),
        "include_symbol_cohorts": bool(replication_config.get("include_symbol_cohorts", True)),
        "include_window_cohorts": bool(replication_config.get("include_window_cohorts", True)),
        "include_regime_slices": bool(replication_config.get("include_regime_slices", False)),
        "requested_cohort_count": 0,
        "completed_cohort_count": 0,
        "coverage_ratio": None,
        "pass_count": 0,
        "pass_rate": None,
        "median_score": None,
        "tail_score": None,
        "median_net_profit_pct": None,
        "promotion_pass": True,
        "gate_mode": "disabled",
        "reasons": [],
        "warnings": [],
        "cohorts": [],
        "summary_by_kind": {},
    }
    if not report["enabled"]:
        return report

    cohort_specs = _build_replication_cohort_specs(
        base_config,
        full_state_bundle,
        holdout_plan,
        replication_config,
    )
    report["requested_cohort_count"] = int(len(cohort_specs))
    if not cohort_specs:
        report["promotion_pass"] = bool(report["min_coverage"] <= 0)
        report["warnings"].append("replication_cohorts_unavailable")
        if report["min_coverage"] > 0:
            report["reasons"].append("replication_coverage_below_minimum")
        return report

    objective_name = replication_config.get("metric")
    min_score = float(replication_config.get("min_score", 0.0))
    min_rows = int(replication_config.get("min_rows", 0))

    for spec in cohort_specs:
        state_bundle = spec.get("state_bundle")
        cohort_row = {
            "cohort_id": spec.get("cohort_id"),
            "kind": spec.get("kind"),
            "label": spec.get("label"),
            "symbol": spec.get("symbol"),
            "timeframe": spec.get("timeframe"),
            "start_timestamp": _json_ready(spec.get("start_timestamp")),
            "end_timestamp": _json_ready(spec.get("end_timestamp")),
            "row_count": 0,
            "completed": False,
            "passed": False,
            "score": None,
            "net_profit_pct": None,
            "total_trades": None,
            "reason": None,
        }

        if state_bundle is None:
            cohort_row["reason"] = "cohort_data_unavailable"
            report["cohorts"].append(cohort_row)
            continue

        raw_data = state_bundle.get("raw_data")
        row_count = int(len(raw_data)) if raw_data is not None else 0
        cohort_row["row_count"] = row_count
        if row_count < min_rows:
            cohort_row["reason"] = "replication_min_rows_not_met"
            report["cohorts"].append(cohort_row)
            continue

        overrides = copy.deepcopy(best_overrides or {})
        _deep_merge(overrides, copy.deepcopy(spec.get("config_overrides") or {}))
        try:
            training, backtest = _execute_trial_candidate(
                base_config,
                overrides,
                pipeline_class,
                trial_step_classes,
                state_bundle,
            )
            evaluation = _build_evaluation_record(
                training,
                backtest,
                objective_name,
                automl_config,
            )
        except RuntimeError as exc:
            cohort_row["reason"] = str(exc)
            report["cohorts"].append(cohort_row)
            continue

        score = _coerce_float(evaluation.get("raw_objective_value"))
        backtest_summary = evaluation.get("backtest") or {}
        cohort_row["completed"] = True
        cohort_row["score"] = score
        cohort_row["net_profit_pct"] = _coerce_float(backtest_summary.get("net_profit_pct"))
        cohort_row["total_trades"] = int(backtest_summary.get("total_trades") or 0)
        cohort_row["passed"] = bool(score is not None and score >= min_score)
        report["cohorts"].append(cohort_row)

    requested = int(len(report["cohorts"]))
    completed = [cohort for cohort in report["cohorts"] if cohort.get("completed")]
    completed_count = int(len(completed))
    pass_count = int(sum(1 for cohort in completed if cohort.get("passed")))
    scores = [float(cohort["score"]) for cohort in completed if cohort.get("score") is not None]
    net_profit_values = [
        float(cohort["net_profit_pct"])
        for cohort in completed
        if cohort.get("net_profit_pct") is not None
    ]

    report["completed_cohort_count"] = completed_count
    report["coverage_ratio"] = (float(completed_count) / float(requested)) if requested > 0 else None
    report["pass_count"] = pass_count
    report["pass_rate"] = (float(pass_count) / float(completed_count)) if completed_count > 0 else None
    report["median_score"] = float(np.median(scores)) if scores else None
    report["tail_score"] = float(min(scores)) if scores else None
    report["median_net_profit_pct"] = float(np.median(net_profit_values)) if net_profit_values else None

    summary_by_kind = {}
    for kind in sorted({str(cohort.get("kind")) for cohort in report["cohorts"]}):
        kind_rows = [cohort for cohort in report["cohorts"] if str(cohort.get("kind")) == kind]
        kind_completed = [cohort for cohort in kind_rows if cohort.get("completed")]
        kind_passed = [cohort for cohort in kind_completed if cohort.get("passed")]
        summary_by_kind[kind] = {
            "requested": int(len(kind_rows)),
            "completed": int(len(kind_completed)),
            "passed": int(len(kind_passed)),
            "pass_rate": (float(len(kind_passed)) / float(len(kind_completed))) if kind_completed else None,
        }
    report["summary_by_kind"] = summary_by_kind

    coverage_pass = bool(completed_count >= report["min_coverage"])
    pass_rate_pass = bool(report["pass_rate"] is not None and report["pass_rate"] >= report["min_pass_rate"])
    report["promotion_pass"] = bool(coverage_pass and pass_rate_pass)
    if not coverage_pass:
        report["reasons"].append("replication_coverage_below_minimum")
    if completed_count > 0 and not pass_rate_pass:
        report["reasons"].append("replication_pass_rate_below_minimum")
    if completed_count == 0:
        report["reasons"].append("replication_coverage_below_minimum")
    return report


def _resolve_overfitting_control(automl_config=None):
    automl_config = automl_config or {}
    control = copy.deepcopy(automl_config.get("overfitting_control", {}))
    dsr_config = dict(control.get("deflated_sharpe", {}))
    pbo_config = dict(control.get("pbo", {}))
    post_selection_config = dict(control.get("post_selection", {}))

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
            "overlap_policy": str(pbo_config.get("overlap_policy", "strict_intersection")).lower(),
            "min_overlap_fraction": float(pbo_config.get("min_overlap_fraction", 0.5)),
            "min_overlap_observations": pbo_config.get("min_overlap_observations"),
        },
        "post_selection": {
            "enabled": bool(post_selection_config.get("enabled", True)),
            "require_pass": bool(post_selection_config.get("require_pass", False)),
            "pass_rule": str(post_selection_config.get("pass_rule", "spa")).lower(),
            "alpha": float(post_selection_config.get("alpha", 0.05)),
            "max_candidates": int(post_selection_config.get("max_candidates", 8)),
            "correlation_threshold": float(post_selection_config.get("correlation_threshold", 0.9)),
            "min_overlap_fraction": float(post_selection_config.get("min_overlap_fraction", 0.5)),
            "min_overlap_observations": int(post_selection_config.get("min_overlap_observations", 10)),
            "overlap_policy": str(post_selection_config.get("overlap_policy", "strict_intersection")).lower(),
            "bootstrap_samples": int(post_selection_config.get("bootstrap_samples", 300)),
            "mean_block_length": post_selection_config.get("mean_block_length"),
            "random_state": int(post_selection_config.get("random_state", 42)),
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
    objective_diagnostics = _build_objective_diagnostics(
        objective_name,
        training_summary,
        backtest_summary,
        automl_config,
    )
    raw_objective_value = float(objective_diagnostics["final_score"])
    record = {
        "training": training_summary,
        "backtest": backtest_summary,
        "returns": returns,
        "period_sharpe": period_sharpe,
        "raw_objective_value": float(raw_objective_value),
        "objective_diagnostics": objective_diagnostics,
    }
    if split is not None:
        record["split"] = {
            "aligned_train_rows": int(split.get("aligned_train_rows", 0)),
            "aligned_test_rows": int(split.get("aligned_test_rows", 0)),
            "aligned_gap_rows": int(split.get("aligned_gap_rows", 0)),
            "train_end_timestamp": _json_ready(split.get("train_end_timestamp")),
            "test_start_timestamp": _json_ready(split.get("test_start_timestamp")),
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
        "objective_diagnostics": validation_record.get("objective_diagnostics"),
        "raw_objective_value": float(validation_record["raw_objective_value"]),
        "search_raw_objective_value": float(search_record["raw_objective_value"]),
    }


def _resolve_selection_policy(automl_config=None):
    policy = copy.deepcopy((automl_config or {}).get("selection_policy") or {})
    enabled = bool(policy.get("enabled", True))
    if not enabled:
        return {
            "enabled": False,
            "calibration_mode": False,
            "gate_modes": {},
            "max_generalization_gap": float("inf"),
            "max_param_fragility": float("inf"),
            "max_complexity_score": float("inf"),
            "min_validation_trade_count": 0,
            "require_locked_holdout_pass": False,
            "min_locked_holdout_score": float("-inf"),
            "max_feature_count_ratio": float("inf"),
            "max_trials_per_model_family": int(1e9),
            "local_perturbation_limit": 0,
            "require_fold_stability_pass": False,
        }
    return {
        "enabled": True,
        "calibration_mode": bool(policy.get("calibration_mode", False)),
        "gate_modes": copy.deepcopy(policy.get("gate_modes") or {}),
        "max_generalization_gap": float(policy.get("max_generalization_gap", 0.35)),
        "max_param_fragility": float(policy.get("max_param_fragility", 0.30)),
        "max_complexity_score": float(policy.get("max_complexity_score", 18.0)),
        "min_validation_trade_count": int(policy.get("min_validation_trade_count", 10)),
        "require_locked_holdout_pass": bool(policy.get("require_locked_holdout_pass", False)),
        "min_locked_holdout_score": float(policy.get("min_locked_holdout_score", 0.0)),
        "max_feature_count_ratio": float(policy.get("max_feature_count_ratio", 1.0)),
        "max_trials_per_model_family": int(policy.get("max_trials_per_model_family", 64)),
        "local_perturbation_limit": int(policy.get("local_perturbation_limit", 8)),
        "require_fold_stability_pass": bool(policy.get("require_fold_stability_pass", True)),
    }


def _resolve_fold_stability_gate(training_summary, selection_policy):
    stability = dict((training_summary or {}).get("fold_stability") or {})
    policy_enabled = bool(stability.get("policy_enabled", False))
    applies = bool(selection_policy.get("require_fold_stability_pass", True) and policy_enabled)
    return {
        "policy_enabled": policy_enabled,
        "applies": applies,
        "passed": bool(stability.get("passed", True)),
        "reasons": list(stability.get("reasons", [])),
        "summary": stability,
    }


def _first_failure_reason(payload, fallback):
    reasons = list((payload or {}).get("reasons") or [])
    if reasons:
        return str(reasons[0])
    return fallback


def _update_selection_policy_report(policy_report, promotion_eligibility_report, *, include_post_selection=False):
    report = finalize_promotion_eligibility_report(promotion_eligibility_report)
    checks = dict(policy_report.get("eligibility_checks") or {})
    checks.update(build_promotion_gate_check_map(report))
    selection_group = dict((report.get("groups") or {}).get("selection") or {})

    policy_report["promotion_eligibility_report"] = report
    policy_report["eligibility_checks"] = checks
    policy_report["eligible_before_post_checks"] = bool(report.get("eligible_before_post_checks", False))
    policy_report["eligibility_reasons"] = list(selection_group.get("blocking_failures") or [])
    if include_post_selection:
        policy_report["promotion_ready"] = bool(report.get("promotion_ready", False))
        policy_report["promotion_reasons"] = list(report.get("blocking_failures") or [])
    return policy_report


def _infer_model_family(overrides):
    return str((overrides or {}).get("model", {}).get("type", "unknown")).lower()


def _count_model_family_trials(trial_records):
    counts = {}
    for record in (trial_records or {}).values():
        family = _infer_model_family(record.get("overrides"))
        counts[family] = counts.get(family, 0) + 1
    return counts


def _coerce_float(value):
    if value is None:
        return None
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(coerced):
        return None
    return coerced


def _build_generalization_gap_report(reference_value, compared_value):
    reference_value = _coerce_float(reference_value)
    compared_value = _coerce_float(compared_value)
    report = {
        "reference_value": reference_value,
        "compared_value": compared_value,
        "absolute_gap": None,
        "degradation": None,
        "normalized_degradation": None,
    }
    if reference_value is None or compared_value is None:
        return report

    absolute_gap = float(reference_value - compared_value)
    degradation = max(absolute_gap, 0.0)
    scale = max(abs(reference_value), abs(compared_value), 1.0)
    report.update(
        {
            "absolute_gap": absolute_gap,
            "degradation": float(degradation),
            "normalized_degradation": float(degradation / scale),
        }
    )
    return report


def _count_configured_lags(raw_lags):
    if raw_lags is None:
        return 0
    if isinstance(raw_lags, str):
        return len([part for part in raw_lags.split(",") if str(part).strip()])
    if isinstance(raw_lags, (list, tuple)):
        return len(raw_lags)
    return 1


def _compute_trial_complexity(overrides, training_summary):
    feature_selection = (training_summary or {}).get("feature_selection") or {}
    feature_selection_enabled = bool(feature_selection.get("enabled", False))
    avg_input_features = _coerce_float(feature_selection.get("avg_input_features"))
    avg_selected_features = _coerce_float(feature_selection.get("avg_selected_features"))
    feature_count_ratio = None
    if (
        feature_selection_enabled
        and avg_input_features is not None
        and avg_input_features > 0.0
        and avg_selected_features is not None
    ):
        feature_count_ratio = float(avg_selected_features / avg_input_features)

    feature_overrides = (overrides or {}).get("features") or {}
    label_overrides = (overrides or {}).get("labels") or {}
    regime_overrides = (overrides or {}).get("regime") or {}
    model_overrides = (overrides or {}).get("model") or {}
    model_params = model_overrides.get("params") or {}
    model_family = _infer_model_family(overrides)

    lag_count = _count_configured_lags(feature_overrides.get("lags"))
    holding_period = int(label_overrides.get("max_holding") or label_overrides.get("horizon") or 0)
    n_regimes = int(regime_overrides.get("n_regimes") or 0)
    n_estimators = _coerce_float(model_params.get("n_estimators")) or 0.0
    max_depth = model_params.get("max_depth")
    if max_depth is None and model_family == "rf":
        max_depth = 12.0
    elif max_depth is None and model_family == "gbm":
        max_depth = 6.0
    max_depth = _coerce_float(max_depth) or 0.0
    meta_layers = int(bool(model_overrides.get("calibration_params")))
    meta_layers += int(bool(model_overrides.get("meta_params")))
    meta_layers += int(bool(model_overrides.get("meta_calibration_params")))

    score = 0.0
    score += min(avg_selected_features or 0.0, 256.0) / 32.0
    score += (feature_count_ratio or 0.0) * 4.0
    score += float(lag_count) * 0.5
    score += min(float(holding_period), 96.0) / 24.0
    score += max(n_regimes - 1, 0) * 0.75
    if model_family in {"rf", "gbm"}:
        score += min(n_estimators, 800.0) / 200.0
        score += min(max_depth, 12.0) / 3.0
    elif model_family == "logistic":
        score += 0.5
    score += meta_layers * 0.75

    return {
        "trial_complexity_score": float(score),
        "avg_input_features": avg_input_features,
        "avg_selected_features": avg_selected_features,
        "feature_count_ratio": feature_count_ratio,
        "lag_count": int(lag_count),
        "holding_period": int(holding_period),
        "n_regimes": int(n_regimes),
        "model_family": model_family,
        "meta_layers": int(meta_layers),
        "tree_count": int(n_estimators),
        "tree_depth": max_depth,
    }


def _ordered_choices_from_spec(spec):
    if not isinstance(spec, dict):
        return []
    spec_type = spec.get("type")
    if spec_type == "categorical":
        return [tuple(choice) if isinstance(choice, list) else choice for choice in spec.get("choices", [])]
    return []


def _neighbor_values_from_spec(current_value, spec):
    if not isinstance(spec, dict):
        return []

    spec_type = spec.get("type")
    if spec_type == "categorical":
        choices = _ordered_choices_from_spec(spec)
        if current_value not in choices:
            return []
        index = choices.index(current_value)
        neighbors = []
        if index - 1 >= 0:
            neighbors.append(choices[index - 1])
        if index + 1 < len(choices):
            neighbors.append(choices[index + 1])
        return neighbors

    current_numeric = _coerce_float(current_value)
    if current_numeric is None:
        return []

    if spec_type in {"float", "int"}:
        step = spec.get("step")
        if step is None:
            return []
        neighbors = []
        low = _coerce_float(spec.get("low"))
        high = _coerce_float(spec.get("high"))
        for candidate in (current_numeric - float(step), current_numeric + float(step)):
            if low is not None and candidate < low:
                continue
            if high is not None and candidate > high:
                continue
            if spec_type == "int":
                candidate = int(round(candidate))
            neighbors.append(candidate)
        return neighbors

    return []


def _set_override_value(overrides, path, value):
    updated = copy.deepcopy(overrides)
    target = updated
    for key in path[:-1]:
        target = target.setdefault(key, {})
    target[path[-1]] = value
    return updated


def _generate_local_perturbations(overrides, search_space, limit=8):
    overrides = overrides or {}
    search_space = search_space or {}
    perturbations = []
    seen = set()

    candidates = [
        (("feature_selection", "max_features"), ((search_space.get("feature_selection") or {}).get("max_features"))),
        (("labels", "max_holding"), ((search_space.get("labels") or {}).get("max_holding"))),
    ]
    model_family = _infer_model_family(overrides)
    model_params = ((overrides.get("model") or {}).get("params") or {})
    model_param_space = (((search_space.get("model") or {}).get("params") or {}).get(model_family) or {})
    for key in sorted(model_params):
        candidates.append((("model", "params", key), model_param_space.get(key)))

    for path, spec in candidates:
        current = overrides
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if current is None:
            continue

        for neighbor in _neighbor_values_from_spec(current, spec):
            updated = _set_override_value(overrides, path, neighbor)
            signature = str(_json_ready(updated))
            if signature in seen:
                continue
            seen.add(signature)
            perturbations.append(
                {
                    "field": ".".join(path),
                    "baseline_value": _json_ready(current),
                    "perturbed_value": _json_ready(neighbor),
                    "overrides": updated,
                }
            )
            if len(perturbations) >= max(1, int(limit)):
                return perturbations
    return perturbations


def _evaluate_candidate_fragility(
    base_config,
    overrides,
    pipeline_class,
    trial_step_classes,
    evaluation_state_bundle,
    evaluation_split,
    objective_name,
    automl_config,
    search_space,
    baseline_value,
    selection_policy,
):
    report = {
        "enabled": bool(selection_policy.get("enabled", True)),
        "baseline_value": _coerce_float(baseline_value),
        "param_fragility_score": 0.0,
        "dispersion": 0.0,
        "max_downside": 0.0,
        "evaluated_count": 0,
        "perturbations": [],
        "reason": None,
        "passed": True,
    }
    if not report["enabled"]:
        report["reason"] = "disabled"
        return report
    if report["baseline_value"] is None:
        report["reason"] = "unavailable_baseline"
        report["passed"] = False
        return report

    perturbations = _generate_local_perturbations(
        overrides,
        search_space,
        limit=selection_policy.get("local_perturbation_limit", 8),
    )
    if not perturbations:
        report["reason"] = "no_local_perturbations"
        return report

    scale = max(abs(report["baseline_value"]), 1.0)
    evaluated_scores = []
    for perturbation in perturbations:
        try:
            if evaluation_split is None:
                training, backtest = _execute_trial_candidate(
                    base_config,
                    perturbation["overrides"],
                    pipeline_class,
                    trial_step_classes,
                    evaluation_state_bundle,
                )
                evaluation = _build_evaluation_record(training, backtest, objective_name, automl_config)
            else:
                training, backtest, split = _execute_temporal_split_candidate(
                    base_config,
                    perturbation["overrides"],
                    pipeline_class,
                    trial_step_classes,
                    evaluation_state_bundle,
                    train_end_timestamp=evaluation_split["train_end_timestamp"],
                    test_start_timestamp=evaluation_split["test_start_timestamp"],
                    excluded_intervals=evaluation_split.get("excluded_intervals"),
                )
                evaluation = _build_evaluation_record(
                    training,
                    backtest,
                    objective_name,
                    automl_config,
                    split=split,
                )
            value = _coerce_float(evaluation.get("raw_objective_value"))
        except (RuntimeError, ValueError, KeyError) as exc:
            perturbation["error"] = str(exc)
            report["perturbations"].append(perturbation)
            continue

        perturbation["raw_objective_value"] = value
        perturbation["normalized_gap"] = (
            float(abs(report["baseline_value"] - value) / scale) if value is not None else None
        )
        report["perturbations"].append(perturbation)
        if value is not None:
            evaluated_scores.append(value)

    report["evaluated_count"] = int(len(evaluated_scores))
    if not evaluated_scores:
        report["reason"] = "no_successful_perturbations"
        return report

    evaluated_array = np.asarray(evaluated_scores, dtype=float)
    relative_moves = np.abs(evaluated_array - report["baseline_value"]) / scale
    downside_moves = np.maximum(report["baseline_value"] - evaluated_array, 0.0) / scale
    dispersion = float(np.std(np.append(evaluated_array, report["baseline_value"]))) / scale
    max_downside = float(np.max(downside_moves)) if len(downside_moves) else 0.0
    fragility_score = float(max(np.mean(relative_moves), max_downside))
    report.update(
        {
            "param_fragility_score": fragility_score,
            "dispersion": dispersion,
            "max_downside": max_downside,
            "passed": bool(fragility_score <= selection_policy.get("max_param_fragility", 0.30)),
        }
    )
    return report


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
        aligned[trial_number] = returns.reindex(common_index).astype(float)
    return pd.DataFrame(aligned, index=common_index)


def _summarize_pairwise_overlap(coverage_frame, min_overlap_fraction=0.0, min_overlap_observations=0):
    coverage = pd.DataFrame(coverage_frame, copy=False).fillna(False).astype(bool)
    overlap_counts = []
    overlap_fractions = []
    insufficient_pairs = 0

    for left, right in combinations(list(coverage.columns), 2):
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


def compute_cpcv_pbo(
    trial_return_frame,
    n_blocks=8,
    test_blocks=None,
    min_block_size=5,
    metric="sharpe_ratio",
    overlap_policy="strict_intersection",
    min_overlap_fraction=0.5,
    min_overlap_observations=None,
):
    report = {
        "enabled": False,
        "reason": None,
        "metric": (metric or "sharpe_ratio").lower(),
        "overlap_policy": str(overlap_policy or "strict_intersection").lower(),
        "min_overlap_fraction": float(min_overlap_fraction),
        "min_overlap_observations": None if min_overlap_observations is None else int(min_overlap_observations),
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
        "strict_overlap_rows": 0,
        "strict_overlap_fraction": None,
        "pairwise_overlap_min_fraction": None,
        "pairwise_overlap_median_fraction": None,
        "pairwise_overlap_max_fraction": None,
        "pairwise_overlap_min_count": None,
        "pairwise_overlap_median_count": None,
        "pairwise_overlap_max_count": None,
        "excluded_low_overlap_split_count": 0,
        "excluded_low_overlap_trial_pairs": 0,
    }

    if trial_return_frame is None or trial_return_frame.empty or trial_return_frame.shape[1] < 2:
        report["reason"] = "insufficient_trials"
        return report

    overlap_policy = report["overlap_policy"]
    if overlap_policy not in {"strict_intersection", "pairwise_overlap", "zero_fill_debug"}:
        report["reason"] = "unknown_overlap_policy"
        return report

    if min_overlap_observations is None:
        min_overlap_observations = max(2, int(min_block_size))
        report["min_overlap_observations"] = int(min_overlap_observations)

    raw_frame = trial_return_frame.astype(float).replace([np.inf, -np.inf], np.nan)
    raw_frame = raw_frame.dropna(axis=1, how="all")
    if raw_frame.shape[1] < 2:
        report["reason"] = "insufficient_trials"
        return report

    coverage = raw_frame.notna()
    strict_overlap_rows = int(coverage.all(axis=1).sum())
    report["strict_overlap_rows"] = strict_overlap_rows
    report["strict_overlap_fraction"] = (
        float(strict_overlap_rows / len(raw_frame))
        if len(raw_frame) > 0
        else None
    )
    overlap_summary = _summarize_pairwise_overlap(
        coverage,
        min_overlap_fraction=min_overlap_fraction,
        min_overlap_observations=min_overlap_observations,
    )
    report["pairwise_overlap_min_fraction"] = overlap_summary["min_fraction"]
    report["pairwise_overlap_median_fraction"] = overlap_summary["median_fraction"]
    report["pairwise_overlap_max_fraction"] = overlap_summary["max_fraction"]
    report["pairwise_overlap_min_count"] = overlap_summary["min_count"]
    report["pairwise_overlap_median_count"] = overlap_summary["median_count"]
    report["pairwise_overlap_max_count"] = overlap_summary["max_count"]

    if overlap_policy == "strict_intersection":
        if report["strict_overlap_fraction"] is not None and report["strict_overlap_fraction"] < float(min_overlap_fraction):
            report["reason"] = "insufficient_overlap"
            return report
        frame = raw_frame.loc[coverage.all(axis=1)].copy()
    elif overlap_policy == "pairwise_overlap":
        frame = raw_frame.copy()
    else:
        frame = raw_frame.fillna(0.0).astype(float)

    if len(frame) < max(2, int(min_block_size) * 2):
        report["reason"] = "insufficient_overlap" if overlap_policy != "zero_fill_debug" else "insufficient_rows"
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
    excluded_low_overlap_split_count = 0
    excluded_low_overlap_trial_pairs = 0
    for test_combo in combinations(range(len(blocks)), test_block_count):
        test_positions = np.concatenate([blocks[idx] for idx in test_combo])
        train_positions = np.concatenate([blocks[idx] for idx in range(len(blocks)) if idx not in test_combo])
        if len(train_positions) == 0 or len(test_positions) == 0:
            continue

        if overlap_policy == "pairwise_overlap":
            train_overlap = _summarize_pairwise_overlap(
                frame.iloc[train_positions].notna(),
                min_overlap_fraction=min_overlap_fraction,
                min_overlap_observations=min_overlap_observations,
            )
            test_overlap = _summarize_pairwise_overlap(
                frame.iloc[test_positions].notna(),
                min_overlap_fraction=min_overlap_fraction,
                min_overlap_observations=min_overlap_observations,
            )
            excluded_pairs = int(train_overlap["insufficient_pair_count"] + test_overlap["insufficient_pair_count"])
            if not train_overlap["sufficient"] or not test_overlap["sufficient"]:
                excluded_low_overlap_split_count += 1
                excluded_low_overlap_trial_pairs += excluded_pairs
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
        report.update(
            {
                "block_count": int(len(blocks)),
                "test_block_count": int(test_block_count),
                "excluded_low_overlap_split_count": int(excluded_low_overlap_split_count),
                "excluded_low_overlap_trial_pairs": int(excluded_low_overlap_trial_pairs),
            }
        )
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
            "excluded_low_overlap_split_count": int(excluded_low_overlap_split_count),
            "excluded_low_overlap_trial_pairs": int(excluded_low_overlap_trial_pairs),
        }
    )
    return report


def _build_trial_selection_report(completed_trials, trial_records, objective_name, automl_config):
    control = _resolve_overfitting_control(automl_config)
    selection_policy = _resolve_selection_policy(automl_config)
    explicit_minimum_dsr_threshold = "minimum_dsr_threshold" in (automl_config or {})
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
    model_family_counts = _count_model_family_trials(trial_records)

    trial_reports = []
    for trial in completed_trials:
        record = trial_records.get(trial.number)
        if record is None:
            continue

        search_metrics = {
            "training": record.get("search", {}).get("training"),
            "backtest": record.get("search", {}).get("backtest"),
            "raw_objective_value": record.get("search", {}).get("raw_objective_value"),
            "objective_diagnostics": record.get("search", {}).get("objective_diagnostics"),
        }
        validation_metrics = {
            "training": record.get("validation", {}).get("training"),
            "backtest": record.get("validation", {}).get("backtest"),
            "raw_objective_value": record.get("validation", {}).get("raw_objective_value"),
            "objective_diagnostics": record.get("validation", {}).get("objective_diagnostics"),
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

        complexity = _compute_trial_complexity(record.get("overrides"), validation_metrics["training"])
        search_gap = _build_generalization_gap_report(
            search_metrics.get("raw_objective_value"),
            validation_metrics.get("raw_objective_value"),
        )
        validation_trade_count = int((validation_metrics.get("backtest") or {}).get("total_trades") or 0)
        model_family = complexity["model_family"]
        fold_stability_gate = _resolve_fold_stability_gate(validation_metrics.get("training") or {}, selection_policy)
        dsr_reason = deflated_sharpe.get("reason")
        dsr_gate_applies = bool(
            minimum_dsr_threshold is not None
            and (
                explicit_minimum_dsr_threshold
                or dsr_reason not in {"insufficient_track_record", "unavailable_sharpe"}
            )
        )
        objective_diagnostics = validation_metrics.get("objective_diagnostics") or {}
        objective_gate_passed = bool(
            (objective_diagnostics.get("classification_gates") or {}).get("passed", True)
        )
        training_summary = validation_metrics.get("training") or {}
        promotion_gates = dict(training_summary.get("promotion_gates") or {})
        feature_admission_summary = dict((training_summary.get("feature_governance") or {}).get("admission_summary") or {})
        feature_portability_diagnostics = dict(training_summary.get("feature_portability_diagnostics") or {})
        cross_venue_integrity = dict(training_summary.get("cross_venue_integrity") or {})
        signal_decay = dict(training_summary.get("signal_decay") or {})
        regime_ablation_summary = dict((training_summary.get("regime") or {}).get("ablation_summary") or {})
        operational_monitoring = dict(training_summary.get("operational_monitoring") or {})
        feature_portability_pass = bool(promotion_gates.get("feature_portability", True))
        feature_admission_pass = bool(promotion_gates.get("feature_admission", True))
        regime_stability_pass = bool(promotion_gates.get("regime_stability", True))
        operational_health_pass = bool(promotion_gates.get("operational_health", True))
        cross_venue_integrity_pass = bool(promotion_gates.get("cross_venue_integrity", True))
        signal_decay_pass = bool(promotion_gates.get("signal_decay", signal_decay.get("promotion_pass", True)))

        eligibility_checks = {
            "minimum_dsr": bool(meets_minimum_dsr_threshold or not dsr_gate_applies),
            "objective_constraints": objective_gate_passed,
            "validation_trade_count": bool(
                validation_trade_count >= selection_policy.get("min_validation_trade_count", 0)
            ),
            "complexity": bool(
                complexity["trial_complexity_score"] <= selection_policy.get("max_complexity_score", np.inf)
            ),
            "feature_count_ratio": bool(
                complexity.get("feature_count_ratio") is None
                or complexity["feature_count_ratio"] <= selection_policy.get("max_feature_count_ratio", np.inf)
            ),
            "generalization_gap": bool(
                (search_gap.get("normalized_degradation") or 0.0)
                <= selection_policy.get("max_generalization_gap", np.inf)
            ),
            "model_family_trial_count": bool(
                model_family_counts.get(model_family, 0)
                <= selection_policy.get("max_trials_per_model_family", np.inf)
            ),
            "feature_portability": feature_portability_pass,
            "feature_admission": feature_admission_pass,
            "regime_stability": regime_stability_pass,
            "operational_health": operational_health_pass,
            "cross_venue_integrity": cross_venue_integrity_pass,
            "signal_decay": signal_decay_pass,
            "fold_stability": bool(fold_stability_gate["passed"] or not fold_stability_gate["applies"]),
            "param_fragility": None,
            "locked_holdout": None,
            "locked_holdout_gap": None,
        }

        promotion_eligibility_report = create_promotion_eligibility_report(
            calibration_mode=selection_policy.get("calibration_mode", False)
        )
        promotion_eligibility_report = set_promotion_score(
            promotion_eligibility_report,
            basis="selection_value",
            value=selection_value,
            metadata={
                "raw_objective_value": float(record["raw_objective_value"]),
                "selection_value": float(selection_value),
            },
        )

        selection_policy_enabled = bool(selection_policy.get("enabled", True))
        if selection_policy_enabled:
            gate_specs = [
                {
                    "name": "minimum_dsr",
                    "passed": eligibility_checks["minimum_dsr"],
                    "measured": dsr_value,
                    "threshold": minimum_dsr_threshold if dsr_gate_applies else None,
                    "reason": None if eligibility_checks["minimum_dsr"] else "deflated_sharpe_below_threshold",
                    "details": deflated_sharpe,
                },
                {
                    "name": "objective_constraints",
                    "passed": eligibility_checks["objective_constraints"],
                    "measured": objective_gate_passed,
                    "threshold": True,
                    "reason": None if eligibility_checks["objective_constraints"] else "objective_constraints_failed",
                    "details": objective_diagnostics,
                },
                {
                    "name": "validation_trade_count",
                    "passed": eligibility_checks["validation_trade_count"],
                    "measured": validation_trade_count,
                    "threshold": int(selection_policy.get("min_validation_trade_count", 0)),
                    "reason": None if eligibility_checks["validation_trade_count"] else "validation_trade_count_below_minimum",
                    "details": {"validation_trade_count": validation_trade_count},
                },
                {
                    "name": "complexity",
                    "passed": eligibility_checks["complexity"],
                    "measured": complexity["trial_complexity_score"],
                    "threshold": selection_policy.get("max_complexity_score", np.inf),
                    "reason": None if eligibility_checks["complexity"] else "complexity_score_above_limit",
                    "details": complexity,
                },
                {
                    "name": "feature_count_ratio",
                    "passed": eligibility_checks["feature_count_ratio"],
                    "measured": complexity.get("feature_count_ratio"),
                    "threshold": selection_policy.get("max_feature_count_ratio", np.inf),
                    "reason": None if eligibility_checks["feature_count_ratio"] else "feature_count_ratio_above_limit",
                    "details": complexity,
                },
                {
                    "name": "generalization_gap",
                    "passed": eligibility_checks["generalization_gap"],
                    "measured": (search_gap.get("normalized_degradation") or 0.0),
                    "threshold": selection_policy.get("max_generalization_gap", np.inf),
                    "reason": None if eligibility_checks["generalization_gap"] else "search_validation_gap_above_limit",
                    "details": search_gap,
                },
                {
                    "name": "model_family_trial_count",
                    "passed": eligibility_checks["model_family_trial_count"],
                    "measured": model_family_counts.get(model_family, 0),
                    "threshold": selection_policy.get("max_trials_per_model_family", np.inf),
                    "reason": None if eligibility_checks["model_family_trial_count"] else "model_family_trial_count_above_limit",
                    "details": {"model_family": model_family, "trial_count": model_family_counts.get(model_family, 0)},
                },
                {
                    "name": "feature_portability",
                    "passed": eligibility_checks["feature_portability"],
                    "measured": feature_portability_diagnostics.get("venue_specific_importance_share"),
                    "threshold": feature_portability_diagnostics.get("config"),
                    "reason": None if eligibility_checks["feature_portability"] else _first_failure_reason(feature_portability_diagnostics, "feature_portability_failed"),
                    "details": feature_portability_diagnostics,
                },
                {
                    "name": "feature_admission",
                    "passed": eligibility_checks["feature_admission"],
                    "measured": feature_admission_summary.get("promotion_pass"),
                    "threshold": True,
                    "reason": None if eligibility_checks["feature_admission"] else _first_failure_reason(feature_admission_summary, "feature_admission_failed"),
                    "details": feature_admission_summary,
                },
                {
                    "name": "regime_stability",
                    "passed": eligibility_checks["regime_stability"],
                    "measured": regime_ablation_summary.get("stability_improvement"),
                    "threshold": True,
                    "reason": None if eligibility_checks["regime_stability"] else _first_failure_reason(regime_ablation_summary, "regime_stability_failed"),
                    "details": regime_ablation_summary,
                },
                {
                    "name": "operational_health",
                    "passed": eligibility_checks["operational_health"],
                    "measured": operational_monitoring.get("healthy"),
                    "threshold": True,
                    "reason": None if eligibility_checks["operational_health"] else _first_failure_reason(operational_monitoring, "operational_monitoring_failed"),
                    "details": operational_monitoring,
                },
                {
                    "name": "cross_venue_integrity",
                    "passed": eligibility_checks["cross_venue_integrity"],
                    "measured": cross_venue_integrity.get("promotion_pass"),
                    "threshold": True,
                    "reason": None if eligibility_checks["cross_venue_integrity"] else _first_failure_reason(cross_venue_integrity, "cross_venue_integrity_failed"),
                    "details": cross_venue_integrity,
                },
                {
                    "name": "signal_decay",
                    "passed": eligibility_checks["signal_decay"],
                    "measured": signal_decay.get("net_edge_at_effective_delay"),
                    "threshold": signal_decay.get("policy"),
                    "reason": None if eligibility_checks["signal_decay"] else _first_failure_reason(signal_decay, "signal_decay_failed"),
                    "details": signal_decay,
                },
                {
                    "name": "fold_stability",
                    "passed": eligibility_checks["fold_stability"],
                    "measured": fold_stability_gate["summary"].get("persistence"),
                    "threshold": True,
                    "reason": None if eligibility_checks["fold_stability"] else "fold_stability_failed",
                    "details": fold_stability_gate["summary"],
                },
            ]
            for gate in gate_specs:
                promotion_eligibility_report = upsert_promotion_gate(
                    promotion_eligibility_report,
                    group="selection",
                    name=gate["name"],
                    passed=gate["passed"],
                    mode=resolve_promotion_gate_mode(selection_policy, gate["name"]),
                    measured=gate["measured"],
                    threshold=gate["threshold"],
                    reason=gate["reason"],
                    details=gate["details"],
                )
            promotion_eligibility_report = finalize_promotion_eligibility_report(promotion_eligibility_report)

            selection_policy_report = {
                "enabled": True,
                "eligible_before_post_checks": False,
                "eligible": None,
                "promotion_ready": None,
                "promotion_reasons": [],
                "frozen": False,
                "holdout_consulted_for_selection": False,
                "eligibility_checks": eligibility_checks,
                "eligibility_reasons": [],
                "promotion_eligibility_report": promotion_eligibility_report,
            }
            selection_policy_report = _update_selection_policy_report(
                selection_policy_report,
                promotion_eligibility_report,
                include_post_selection=False,
            )
        else:
            promotion_eligibility_report = finalize_promotion_eligibility_report(promotion_eligibility_report)
            selection_policy_report = {
                "enabled": False,
                "eligible_before_post_checks": True,
                "eligible": True,
                "promotion_ready": True,
                "promotion_reasons": [],
                "frozen": False,
                "holdout_consulted_for_selection": False,
                "eligibility_checks": eligibility_checks,
                "eligibility_reasons": [],
                "promotion_eligibility_report": promotion_eligibility_report,
            }

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
                "objective_diagnostics": objective_diagnostics,
                "meets_minimum_dsr_threshold": meets_minimum_dsr_threshold,
                "overfitting": {"deflated_sharpe": deflated_sharpe},
                "model_family": model_family,
                "trial_complexity_score": complexity["trial_complexity_score"],
                "feature_count_ratio": complexity.get("feature_count_ratio"),
                "complexity": complexity,
                "generalization_gap": {
                    "search_to_validation": search_gap,
                    "validation_to_locked_holdout": None,
                },
                "fold_stability": fold_stability_gate["summary"],
                "param_fragility_score": None,
                "param_fragility": None,
                "locked_holdout": None,
                "selection_policy": selection_policy_report,
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
            overlap_policy=pbo_config["overlap_policy"],
            min_overlap_fraction=pbo_config["min_overlap_fraction"],
            min_overlap_observations=pbo_config["min_overlap_observations"],
        )
    else:
        pbo_report = {
            "enabled": False,
            "reason": "disabled",
            "metric": pbo_config["metric"],
        }

    post_selection_config = control["post_selection"]
    if control["enabled"] and post_selection_config["enabled"]:
        post_selection_report = compute_post_selection_inference(
            trial_reports,
            trial_return_frame,
            config=post_selection_config,
        )
    else:
        post_selection_report = {
            "enabled": False,
            "reason": "disabled",
            "require_pass": bool(post_selection_config.get("require_pass", False)),
            "passed": True,
        }

    post_selection_required = bool(post_selection_report.get("enabled", False) and post_selection_config.get("require_pass", False))
    post_selection_gate = bool(post_selection_report.get("passed", True) or not post_selection_required)
    if post_selection_required and not post_selection_report.get("passed", False):
        for report in trial_reports:
            policy = report["selection_policy"]
            if not policy.get("eligible_before_post_checks", False):
                continue
            policy["eligibility_checks"]["post_selection"] = False
            policy["eligibility_reasons"].append("post_selection_inference_failed")
            policy["eligible_before_post_checks"] = False
    else:
        for report in trial_reports:
            policy = report["selection_policy"]
            policy["eligibility_checks"]["post_selection"] = bool(post_selection_gate)

    best_trial = trial_reports[0]
    diagnostics = {
        "enabled": control["enabled"],
        "selection_mode": selection_mode,
        "selection_metric": selection_metric,
        "minimum_dsr_threshold": minimum_dsr_threshold,
        "selection_policy": selection_policy,
        "trial_count": int(len(completed_trials)),
        "effective_trial_count": float(effective_trial_count),
        "model_family_counts": model_family_counts,
        "eligible_trial_count_before_post_checks": int(
            sum(1 for report in trial_reports if report["selection_policy"]["eligible_before_post_checks"])
        ),
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
            "trial_complexity_score": best_trial["trial_complexity_score"],
            "feature_count_ratio": best_trial.get("feature_count_ratio"),
            "generalization_gap": best_trial.get("generalization_gap"),
        },
        "pbo": pbo_report,
        "post_selection": post_selection_report,
        "eligible_trial_count_after_post_selection": int(
            sum(1 for report in trial_reports if report["selection_policy"].get("eligible_before_post_checks"))
        ),
    }
    return {
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
        "trial_reports": trial_reports,
        "diagnostics": diagnostics,
    }


def _resolve_objective_gates(automl_config, objective_name):
    objective_name = _normalize_objective_name(objective_name)
    gate_config = copy.deepcopy((automl_config or {}).get("objective_gates") or {})
    default_enabled = objective_name in {
        "risk_adjusted_after_costs",
        "benchmark_excess_sharpe",
        "net_profit_pct_vs_benchmark",
    }
    enabled = bool(gate_config.get("enabled", default_enabled))
    if not enabled:
        return {
            "enabled": False,
            "min_directional_accuracy": None,
            "max_log_loss": None,
            "max_calibration_error": None,
            "min_trade_count": None,
        }
    return {
        "enabled": True,
        "min_directional_accuracy": _coerce_float(gate_config.get("min_directional_accuracy", 0.45)),
        "max_log_loss": _coerce_float(gate_config.get("max_log_loss", 1.0)),
        "max_calibration_error": _coerce_float(gate_config.get("max_calibration_error", 0.35)),
        "min_trade_count": int(gate_config.get("min_trade_count", 5)),
    }


def _build_gate_result(name, value, minimum=None, maximum=None):
    passed = True
    if minimum is not None:
        passed = value is not None and value >= minimum
    if maximum is not None:
        passed = passed and value is not None and value <= maximum
    return {
        "name": name,
        "value": value,
        "minimum": minimum,
        "maximum": maximum,
        "passed": bool(passed),
    }


def _evaluate_objective_gates(training, backtest, automl_config, objective_name):
    gates = _resolve_objective_gates(automl_config, objective_name)
    report = {
        "enabled": bool(gates.get("enabled", False)),
        "passed": True,
        "failed": [],
        "checks": {},
    }
    if not report["enabled"]:
        return report

    directional_accuracy = _resolve_metric(training, "avg_directional_accuracy", fallback="avg_accuracy")
    log_loss_value = _resolve_metric(training, "avg_log_loss")
    calibration_error = _resolve_metric(training, "avg_calibration_error")
    trade_count = _coerce_float((backtest or {}).get("total_trades"))

    checks = {
        "directional_accuracy": _build_gate_result(
            "directional_accuracy",
            directional_accuracy,
            minimum=gates.get("min_directional_accuracy"),
        ),
        "log_loss": _build_gate_result(
            "log_loss",
            log_loss_value,
            maximum=gates.get("max_log_loss"),
        ),
        "calibration_error": _build_gate_result(
            "calibration_error",
            calibration_error,
            maximum=gates.get("max_calibration_error"),
        ),
        "trade_count": _build_gate_result(
            "trade_count",
            trade_count,
            minimum=float(gates.get("min_trade_count")) if gates.get("min_trade_count") is not None else None,
        ),
    }
    report["checks"] = checks
    report["failed"] = [name for name, payload in checks.items() if not payload["passed"]]
    report["passed"] = not report["failed"]
    return report


def _get_significance_payload(backtest, metric_name):
    significance = (backtest or {}).get("statistical_significance") or {}
    metrics = significance.get("metrics") or {}
    return metrics.get(metric_name) or {}


def _resolve_metric_value_with_significance(backtest, metric_name, use_lower_bound=False):
    point_estimate = _coerce_float((backtest or {}).get(metric_name))
    metric_payload = _get_significance_payload(backtest, metric_name)
    confidence_interval = metric_payload.get("confidence_interval") or {}
    lower_bound = _coerce_float(confidence_interval.get("lower"))

    if use_lower_bound and lower_bound is not None:
        return lower_bound, "confidence_lower_bound", lower_bound
    return point_estimate if point_estimate is not None else 0.0, "point_estimate", lower_bound


def _should_use_objective_lower_bound(objective_name, automl_config=None):
    configured = (automl_config or {}).get("objective_use_confidence_lower_bound")
    if configured is not None:
        return bool(configured)
    return _normalize_objective_name(objective_name) in _BACKTEST_OBJECTIVES


def _resolve_benchmark_reference(backtest, objective_name, automl_config):
    objective_name = _normalize_objective_name(objective_name)
    significance = (backtest or {}).get("statistical_significance") or {}
    if objective_name == "benchmark_excess_sharpe":
        benchmark_value = _coerce_float(significance.get("benchmark_sharpe_ratio"))
        if benchmark_value is None:
            benchmark_value = _coerce_float((automl_config or {}).get("benchmark_sharpe"))
        return benchmark_value
    if objective_name == "net_profit_pct_vs_benchmark":
        benchmark_value = _coerce_float((backtest or {}).get("benchmark_net_profit_pct"))
        if benchmark_value is None:
            benchmark_value = _coerce_float((automl_config or {}).get("benchmark_net_profit_pct"))
        return benchmark_value
    return None


def _compute_turnover_ratio(backtest):
    trade_count = _coerce_float((backtest or {}).get("total_trades"))
    bar_count = _coerce_float((backtest or {}).get("bar_count"))
    if trade_count is None or bar_count is None or bar_count <= 0.0:
        return None
    return float(min(trade_count / bar_count, 5.0))


def _build_objective_diagnostics(objective_name, training, backtest, automl_config=None):
    automl_config = automl_config or {}
    objective_name = _normalize_objective_name(objective_name)
    gate_report = _evaluate_objective_gates(training, backtest, automl_config, objective_name)

    directional_accuracy = _resolve_metric(training, "avg_directional_accuracy", fallback="avg_accuracy") or 0.0
    log_loss_value = _resolve_metric(training, "avg_log_loss")
    brier_score_value = _resolve_metric(training, "avg_brier_score")
    calibration_error = _resolve_metric(training, "avg_calibration_error")
    avg_accuracy = _resolve_metric(training, "avg_accuracy") or directional_accuracy
    net_profit_pct = _coerce_float((backtest or {}).get("net_profit_pct")) or 0.0
    max_drawdown = abs(_coerce_float((backtest or {}).get("max_drawdown")) or 0.0)
    turnover_ratio = _compute_turnover_ratio(backtest) or 0.0

    diagnostics = {
        "objective_name": objective_name,
        "classification_gates": gate_report,
        "components": {},
        "raw_score": None,
        "final_score": None,
        "primary_metric": None,
        "primary_metric_source": None,
        "primary_metric_lower_bound": None,
        "benchmark_reference": None,
    }
    use_lower_bound = _should_use_objective_lower_bound(objective_name, automl_config)

    if objective_name == "directional_accuracy":
        raw_score = float(directional_accuracy)
        diagnostics["components"] = {"directional_accuracy": float(directional_accuracy)}
    elif objective_name in {"neg_log_loss", "log_loss"}:
        raw_score = float(-(log_loss_value if log_loss_value is not None else 1e6))
        diagnostics["components"] = {"log_loss": log_loss_value}
    elif objective_name in {"neg_brier_score", "brier_score"}:
        raw_score = float(-(brier_score_value if brier_score_value is not None else 1e6))
        diagnostics["components"] = {"brier_score": brier_score_value}
    elif objective_name in {"neg_calibration_error", "calibration_error"}:
        raw_score = float(-(calibration_error if calibration_error is not None else 1e6))
        diagnostics["components"] = {"calibration_error": calibration_error}
    elif objective_name == "net_profit_pct":
        primary_metric, primary_source, primary_lower = _resolve_metric_value_with_significance(
            backtest,
            "net_profit_pct",
            use_lower_bound=use_lower_bound,
        )
        raw_score = float(primary_metric)
        diagnostics.update(
            {
                "primary_metric": "net_profit_pct",
                "primary_metric_source": primary_source,
                "primary_metric_lower_bound": primary_lower,
                "components": {"net_profit_pct": float(primary_metric)},
            }
        )
    elif objective_name == "sharpe_ratio":
        primary_metric, primary_source, primary_lower = _resolve_metric_value_with_significance(
            backtest,
            "sharpe_ratio",
            use_lower_bound=use_lower_bound,
        )
        raw_score = float(primary_metric)
        diagnostics.update(
            {
                "primary_metric": "sharpe_ratio",
                "primary_metric_source": primary_source,
                "primary_metric_lower_bound": primary_lower,
                "components": {"sharpe_ratio": float(raw_score)},
            }
        )
    elif objective_name == "profit_factor":
        primary_metric, primary_source, primary_lower = _resolve_metric_value_with_significance(
            backtest,
            "profit_factor",
            use_lower_bound=use_lower_bound,
        )
        if not np.isfinite(primary_metric):
            raw_score = float(automl_config.get("profit_factor_cap", 5.0))
        else:
            raw_score = float(primary_metric)
        diagnostics.update(
            {
                "primary_metric": "profit_factor",
                "primary_metric_source": primary_source,
                "primary_metric_lower_bound": primary_lower,
                "components": {"profit_factor": float(raw_score)},
            }
        )
    elif objective_name == "calmar_ratio":
        primary_metric, primary_source, primary_lower = _resolve_metric_value_with_significance(
            backtest,
            "calmar_ratio",
            use_lower_bound=use_lower_bound,
        )
        raw_score = float(primary_metric)
        diagnostics.update(
            {
                "primary_metric": "calmar_ratio",
                "primary_metric_source": primary_source,
                "primary_metric_lower_bound": primary_lower,
                "components": {"calmar_ratio": float(raw_score)},
            }
        )
    elif objective_name in {"risk_adjusted_after_costs", "benchmark_excess_sharpe", "net_profit_pct_vs_benchmark"}:
        metric_name = "sharpe_ratio"
        if objective_name == "net_profit_pct_vs_benchmark":
            metric_name = "net_profit_pct"
        primary_metric, primary_source, primary_lower = _resolve_metric_value_with_significance(
            backtest,
            metric_name,
            use_lower_bound=use_lower_bound,
        )
        benchmark_reference = _resolve_benchmark_reference(backtest, objective_name, automl_config)
        benchmark_reference = benchmark_reference if benchmark_reference is not None else 0.0
        drawdown_penalty = float(automl_config.get("objective_drawdown_penalty", 2.0)) * max_drawdown
        turnover_penalty = float(automl_config.get("objective_turnover_penalty", 0.25)) * turnover_ratio
        net_profit_bonus = float(automl_config.get("objective_net_profit_weight", 0.5)) * net_profit_pct
        raw_score = float(primary_metric) - float(benchmark_reference) + net_profit_bonus - drawdown_penalty - turnover_penalty
        diagnostics.update(
            {
                "primary_metric": metric_name,
                "primary_metric_source": primary_source,
                "primary_metric_lower_bound": primary_lower,
                "benchmark_reference": benchmark_reference,
                "components": {
                    metric_name: float(primary_metric),
                    "benchmark_reference": float(benchmark_reference),
                    "net_profit_pct": float(net_profit_pct),
                    "drawdown_penalty": float(drawdown_penalty),
                    "turnover_ratio": float(turnover_ratio),
                    "turnover_penalty": float(turnover_penalty),
                },
            }
        )
    else:
        score = automl_config.get("weight_directional_accuracy", 100.0) * directional_accuracy
        score += automl_config.get("weight_accuracy", 5.0) * avg_accuracy
        if log_loss_value is not None:
            score -= automl_config.get("weight_log_loss", 1.0) * log_loss_value
        if brier_score_value is not None:
            score -= automl_config.get("weight_brier_score", 0.5) * brier_score_value
        if calibration_error is not None:
            score -= automl_config.get("weight_calibration_error", 0.5) * calibration_error
        raw_score = float(score)
        diagnostics["components"] = {
            "directional_accuracy": float(directional_accuracy),
            "accuracy": float(avg_accuracy),
            "log_loss": log_loss_value,
            "brier_score": brier_score_value,
            "calibration_error": calibration_error,
        }

    diagnostics["raw_score"] = float(raw_score)
    diagnostics["final_score"] = float(raw_score if gate_report["passed"] else float("-inf"))
    return diagnostics


def compute_objective_value(objective_name, training, backtest, automl_config=None, overfitting_context=None):
    """Compute the scalar objective value used by the AutoML study."""
    diagnostics = _build_objective_diagnostics(objective_name, training, backtest, automl_config or {})
    raw_score = float(diagnostics["final_score"])

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
    _validate_signal_policy_search_space(search_space)

    storage_path = _build_study_storage_path(base_config, automl_config)
    storage_url = f"sqlite:///{storage_path.as_posix()}"
    study_name = _resolve_study_name(base_config, automl_config)
    sampler = TPESampler(seed=automl_config.get("seed", 42))
    objective_name = _normalize_objective_name(automl_config.get("objective", "risk_adjusted_after_costs"))

    full_state_bundle = _build_state_bundle(base_pipeline)
    holdout_plan = _resolve_holdout_plan(full_state_bundle["raw_data"], automl_config, base_config=base_config)
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
        _validate_trial_overrides(overrides)
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
                "objective_diagnostics": search_record.get("objective_diagnostics"),
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
                    train_end_timestamp=holdout_plan["search_end_timestamp"],
                    test_start_timestamp=holdout_plan["validation_start_timestamp"],
                    excluded_intervals=[
                        (
                            holdout_plan.get("search_validation_gap_start_timestamp"),
                            holdout_plan.get("search_validation_gap_end_timestamp"),
                        )
                    ],
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
                "objective_diagnostics": record.get("objective_diagnostics"),
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
                    train_end_timestamp=holdout_plan["search_end_timestamp"],
                    test_start_timestamp=holdout_plan["validation_start_timestamp"],
                    excluded_intervals=[
                        (
                            holdout_plan.get("search_validation_gap_start_timestamp"),
                            holdout_plan.get("search_validation_gap_end_timestamp"),
                        )
                    ],
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
    selection_policy = _resolve_selection_policy(automl_config)
    best_trial_report = None
    evaluation_split = None
    if holdout_plan["enabled"]:
        evaluation_split = {
            "train_end_timestamp": holdout_plan["search_end_timestamp"],
            "test_start_timestamp": holdout_plan["validation_start_timestamp"],
            "excluded_intervals": [
                (
                    holdout_plan.get("search_validation_gap_start_timestamp"),
                    holdout_plan.get("search_validation_gap_end_timestamp"),
                )
            ],
        }
    for report in selection_report["trial_reports"]:
        policy_report = report["selection_policy"]
        if not policy_report["eligible_before_post_checks"]:
            policy_report["eligible"] = False
            continue

        if not selection_policy.get("enabled", True):
            policy_report["eligible"] = True
            best_trial_report = report
            break

        fragility = _evaluate_candidate_fragility(
            base_config=base_config,
            overrides=report["overrides"],
            pipeline_class=pipeline_class,
            trial_step_classes=trial_step_classes,
            evaluation_state_bundle=validation_state_bundle,
            evaluation_split=evaluation_split,
            objective_name=objective_name,
            automl_config=automl_config,
            search_space=search_space,
            baseline_value=report["raw_objective_value"],
            selection_policy=selection_policy,
        )
        report["param_fragility"] = fragility
        report["param_fragility_score"] = fragility.get("param_fragility_score")
        promotion_eligibility_report = policy_report.get("promotion_eligibility_report") or create_promotion_eligibility_report()
        promotion_eligibility_report = upsert_promotion_gate(
            promotion_eligibility_report,
            group="selection",
            name="param_fragility",
            passed=bool(fragility.get("passed", True)),
            mode=resolve_promotion_gate_mode(selection_policy, "param_fragility"),
            measured=fragility.get("param_fragility_score"),
            threshold=selection_policy.get("max_param_fragility", np.inf),
            reason=None if fragility.get("passed", True) else "parameter_fragility_above_limit",
            details=fragility,
        )
        policy_report = _update_selection_policy_report(
            policy_report,
            promotion_eligibility_report,
            include_post_selection=False,
        )
        policy_report["eligible"] = bool(policy_report.get("eligible_before_post_checks", False))
        if policy_report["eligible"]:
            best_trial_report = report
            break

    if best_trial_report is None:
        raise RuntimeError("AutoML found no eligible trial under the configured selection policy")

    best_trial_number = int(best_trial_report["number"])
    best_optuna_trial = study.best_trial
    best_overrides = copy.deepcopy(best_trial_report["overrides"])
    best_overrides_summary = _json_ready(_clone_value(best_overrides))
    best_trial_report["selection_policy"]["frozen"] = True
    selection_snapshot = _build_selection_snapshot(best_trial_report)

    selection_report["diagnostics"]["promoted_trial"] = {
        "number": int(best_trial_report["number"]),
        "raw_objective_value": float(best_trial_report["raw_objective_value"]),
        "selection_value": float(best_trial_report["selection_value"]),
        "trial_complexity_score": float(best_trial_report["trial_complexity_score"]),
        "feature_count_ratio": best_trial_report.get("feature_count_ratio"),
        "fold_stability": best_trial_report.get("fold_stability"),
        "param_fragility_score": best_trial_report.get("param_fragility_score"),
        "generalization_gap": best_trial_report.get("generalization_gap"),
        "eligibility_reasons": best_trial_report["selection_policy"].get("eligibility_reasons", []),
        "candidate_hash": selection_snapshot.get("candidate_hash"),
        "selection_timestamp": selection_snapshot.get("selection_timestamp"),
    }
    selection_report["diagnostics"]["selection_freeze"] = selection_snapshot

    validation_holdout = _build_validation_holdout_report(best_trial_report, holdout_plan)

    locked_holdout_access_count = 0
    locked_holdout = best_trial_report.get("locked_holdout")
    if locked_holdout is None:
        locked_holdout_access_count += int(bool(holdout_plan.get("enabled", False)))
        locked_holdout = _evaluate_locked_holdout(
            base_config=base_config,
            best_overrides=best_overrides,
            pipeline_class=pipeline_class,
            trial_step_classes=trial_step_classes,
            full_state_bundle=full_state_bundle,
            holdout_plan=holdout_plan,
        )
    locked_holdout = _decorate_locked_holdout_report(
        locked_holdout,
        selection_snapshot=selection_snapshot,
        access_count=locked_holdout_access_count,
    )
    post_selection_holdout = _build_locked_holdout_promotion_report(
        selection_policy,
        best_trial_report,
        locked_holdout,
    )
    replication_report = _evaluate_replication_cohorts(
        base_config=base_config,
        best_overrides=best_overrides,
        pipeline_class=pipeline_class,
        trial_step_classes=trial_step_classes,
        full_state_bundle=full_state_bundle,
        holdout_plan=holdout_plan,
        base_pipeline=base_pipeline,
    )
    replication_report["gate_mode"] = resolve_promotion_gate_mode(selection_policy, "replication")
    replication_report = _json_ready(replication_report)
    best_trial_report["locked_holdout"] = locked_holdout
    best_trial_report["replication"] = replication_report
    best_trial_report["generalization_gap"]["validation_to_locked_holdout"] = post_selection_holdout["generalization_gap"]
    best_trial_report["selection_policy"]["eligibility_checks"]["locked_holdout"] = post_selection_holdout[
        "locked_holdout_pass"
    ]
    best_trial_report["selection_policy"]["eligibility_checks"]["locked_holdout_gap"] = post_selection_holdout[
        "locked_holdout_gap_pass"
    ]
    promotion_eligibility_report = best_trial_report["selection_policy"].get("promotion_eligibility_report") or create_promotion_eligibility_report()
    score = resolve_canonical_promotion_score(
        locked_holdout_report=locked_holdout,
        selection_value=best_trial_report.get("selection_value"),
        preference="locked_holdout_first",
    )
    promotion_eligibility_report = set_promotion_score(
        promotion_eligibility_report,
        basis=score.get("basis"),
        value=score.get("value"),
        metadata={
            "selection_value": best_trial_report.get("selection_value"),
            "locked_holdout_score": (locked_holdout or {}).get("raw_objective_value"),
        },
    )
    promotion_eligibility_report = upsert_promotion_gate(
        promotion_eligibility_report,
        group="post_selection",
        name="locked_holdout",
        passed=bool(post_selection_holdout["locked_holdout_pass"]),
        mode=resolve_promotion_gate_mode(selection_policy, "locked_holdout"),
        measured=(locked_holdout or {}).get("raw_objective_value"),
        threshold=selection_policy.get("min_locked_holdout_score", 0.0),
        reason=None if post_selection_holdout["locked_holdout_pass"] else "locked_holdout_failed",
        details=locked_holdout,
    )
    promotion_eligibility_report = upsert_promotion_gate(
        promotion_eligibility_report,
        group="post_selection",
        name="locked_holdout_gap",
        passed=bool(post_selection_holdout["locked_holdout_gap_pass"]),
        mode=resolve_promotion_gate_mode(selection_policy, "locked_holdout_gap"),
        measured=(post_selection_holdout["generalization_gap"] or {}).get("normalized_degradation"),
        threshold=selection_policy.get("max_generalization_gap", np.inf),
        reason=None if post_selection_holdout["locked_holdout_gap_pass"] else "validation_holdout_gap_above_limit",
        details=post_selection_holdout["generalization_gap"],
    )
    if replication_report.get("enabled"):
        promotion_eligibility_report = upsert_promotion_gate(
            promotion_eligibility_report,
            group="post_selection",
            name="replication",
            passed=bool(replication_report.get("promotion_pass", True)),
            mode=replication_report.get("gate_mode"),
            measured=replication_report.get("pass_rate"),
            threshold={
                "min_coverage": replication_report.get("min_coverage"),
                "min_pass_rate": replication_report.get("min_pass_rate"),
                "min_score": replication_report.get("min_score"),
            },
            reason=None if replication_report.get("promotion_pass", True) else _first_failure_reason(replication_report, "replication_failed"),
            details=replication_report,
        )
    execution_realism = evaluate_execution_realism_gate(
        (locked_holdout or {}).get("backtest")
        or validation_holdout.get("backtest")
        or best_trial_report.get("backtest")
        or {},
        policy=selection_policy,
    )
    best_trial_report["selection_policy"]["eligibility_checks"]["execution_realism"] = bool(execution_realism["passed"])
    promotion_eligibility_report = upsert_promotion_gate(
        promotion_eligibility_report,
        group="post_selection",
        name="execution_realism",
        passed=bool(execution_realism["passed"]),
        mode=resolve_promotion_gate_mode(selection_policy, "execution_realism"),
        measured=execution_realism.get("execution_mode"),
        threshold=execution_realism.get("required_execution_mode"),
        reason=execution_realism.get("reason"),
        details=execution_realism,
    )
    best_trial_report["selection_policy"] = _update_selection_policy_report(
        best_trial_report["selection_policy"],
        promotion_eligibility_report,
        include_post_selection=True,
    )
    best_trial_report["selection_policy"]["holdout_consulted_for_selection"] = False
    selection_report["diagnostics"]["holdout_access_count"] = int(locked_holdout_access_count)
    selection_report["diagnostics"]["holdout_evaluated_once"] = bool(locked_holdout.get("evaluated_once", False))
    selection_report["diagnostics"]["holdout_evaluated_after_freeze"] = bool(
        locked_holdout.get("evaluated_after_freeze", False)
    )
    selection_report["diagnostics"]["execution_realism"] = execution_realism
    selection_report["diagnostics"]["replication"] = replication_report
    selection_report["diagnostics"]["promotion_ready"] = bool(best_trial_report["selection_policy"]["promotion_ready"])
    selection_report["diagnostics"]["promotion_reasons"] = list(best_trial_report["selection_policy"]["promotion_reasons"])

    top_trials = []
    for report in selection_report["trial_reports"][:5]:
        top_trials.append(
            {
                "number": int(report["number"]),
                "value": float(report["selection_value"]),
                "raw_value": float(report["raw_objective_value"]),
                "model_family": report.get("model_family"),
                "params": report["params"],
                "training": report["training"],
                "backtest": report["backtest"],
                "search_metrics": report["search_metrics"],
                "validation_metrics": report["validation_metrics"],
                "objective_diagnostics": report.get("objective_diagnostics"),
                "meets_minimum_dsr_threshold": report["meets_minimum_dsr_threshold"],
                "trial_complexity_score": report["trial_complexity_score"],
                "feature_count_ratio": report.get("feature_count_ratio"),
                "fold_stability": report.get("fold_stability"),
                "generalization_gap": report.get("generalization_gap"),
                "param_fragility_score": report.get("param_fragility_score"),
                "param_fragility": report.get("param_fragility"),
                "selection_policy": report.get("selection_policy"),
                "locked_holdout": report.get("locked_holdout"),
                "replication": report.get("replication"),
                "overfitting": report["overfitting"],
            }
        )

    data_lineage = _json_ready(base_pipeline.state.get("data_lineage") or {})

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
        "best_objective_diagnostics": best_trial_report.get("objective_diagnostics"),
        "best_selection_policy": {
            "trial_complexity_score": best_trial_report["trial_complexity_score"],
            "feature_count_ratio": best_trial_report.get("feature_count_ratio"),
            "fold_stability": best_trial_report.get("fold_stability"),
            "generalization_gap": best_trial_report.get("generalization_gap"),
            "param_fragility_score": best_trial_report.get("param_fragility_score"),
            "param_fragility": best_trial_report.get("param_fragility"),
            "replication": best_trial_report.get("replication"),
            "selection_policy": best_trial_report.get("selection_policy"),
            "promotion_eligibility_report": best_trial_report.get("selection_policy", {}).get("promotion_eligibility_report"),
        },
        "selection_freeze": selection_snapshot,
        "validation_holdout": validation_holdout,
        "locked_holdout": locked_holdout,
        "replication": best_trial_report.get("replication"),
        "promotion_ready": bool(best_trial_report["selection_policy"]["promotion_ready"]),
        "promotion_reasons": list(best_trial_report["selection_policy"]["promotion_reasons"]),
        "promotion_eligibility_report": best_trial_report.get("selection_policy", {}).get("promotion_eligibility_report"),
        "overfitting_diagnostics": selection_report["diagnostics"],
        "data_lineage": data_lineage,
        "top_trials": top_trials,
        "trial_count": len(completed_trials),
    }

    registry_config = dict(base_config.get("registry") or {})
    if registry_config.get("enabled", False):
        symbol = base_config.get("data", {}).get("symbol", "unknown")
        registry_store = LocalRegistryStore(
            root_dir=registry_config.get("root_dir", ".cache/registry"),
            max_versions_per_symbol=registry_config.get("max_versions_per_symbol", 10),
        )
        champion_before = registry_store.get_champion(symbol)
        best_training = best_trial_report.get("training") or {}
        feature_columns = list(best_training.get("last_selected_columns") or [])
        version_id = registry_store.register_version(
            best_training.get("last_model"),
            symbol=symbol,
            feature_columns=feature_columns,
            metadata={
                "study_name": summary["study_name"],
                "best_trial_number": summary["best_trial_number"],
                "selection_freeze": selection_snapshot,
                "best_params": summary["best_params"],
                "best_overrides": summary["best_overrides"],
                "data_lineage": data_lineage,
            },
            training_summary=_json_ready(_summarize_training(best_training)),
            validation_summary=_json_ready(best_trial_report.get("validation_metrics") or {}),
            locked_holdout=_json_ready(locked_holdout),
            replication=_json_ready(best_trial_report.get("replication") or {}),
            promotion_eligibility_report=_json_ready(
                best_trial_report.get("selection_policy", {}).get("promotion_eligibility_report") or {}
            ),
            lineage={
                "candidate_hash": selection_snapshot.get("candidate_hash"),
                "selection_timestamp": selection_snapshot.get("selection_timestamp"),
                "data_lineage": data_lineage,
            },
            status="challenger",
            meta_model=best_training.get("last_meta"),
        )
        monitoring_report = _json_ready(best_training.get("operational_monitoring") or {})
        monitoring_path = None
        if monitoring_report:
            monitoring_path = registry_store.attach_monitoring_report(version_id, monitoring_report, symbol=symbol)
        promotion_decision = evaluate_challenger_promotion(
            {
                "promotion_ready": bool(summary["promotion_ready"]),
                "promotion_eligibility_report": _json_ready(
                    best_trial_report.get("selection_policy", {}).get("promotion_eligibility_report") or {}
                ),
                "selection_value": summary["best_value"],
                "sample_count": int(
                    (locked_holdout.get("backtest") or {}).get("total_trades")
                    or best_training.get("oos_trade_count")
                    or 0
                ),
            },
            champion_record=champion_before,
            monitoring_report=monitoring_report,
            policy=registry_config.get("promotion_policy"),
        )
        registry_store.record_promotion_decision(version_id, promotion_decision, symbol=symbol)
        if promotion_decision.get("approved", False):
            registry_store.promote(version_id, "champion", symbol=symbol, decision=promotion_decision)
        registry_entry = registry_store._find_row(version_id, symbol=symbol)
        summary["registry"] = {
            "version_id": version_id,
            "symbol": symbol,
            "current_status": registry_entry.get("current_status") if registry_entry else "challenger",
            "promotion_decision": promotion_decision,
            "champion_before": champion_before.get("version_id") if champion_before else None,
            "monitoring_report": str(monitoring_path) if monitoring_path is not None else None,
        }

    engine = getattr(getattr(study, "_storage", None), "engine", None)
    if engine is not None:
        try:
            engine.dispose()
        except Exception:
            pass

    return summary