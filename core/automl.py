"""AutoML search helpers for the research pipeline."""

import copy
import hashlib
import json
import os
import subprocess
import warnings
from itertools import combinations
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from .automl_contracts import validate_summary_contract
from .evaluation_modes import resolve_evaluation_mode
from .promotion import (
    build_promotion_gate_check_map,
    create_promotion_eligibility_report,
    evaluate_execution_realism_gate,
    evaluate_stress_realism_gate,
    finalize_promotion_eligibility_report,
    resolve_canonical_promotion_score,
    resolve_promotion_gate_mode,
    set_promotion_score,
    upsert_promotion_gate,
)
from .registry import LocalRegistryStore, evaluate_challenger_promotion
from .storage import frame_fingerprint, payload_sha256, read_json, write_json
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
        "min_return": {"type": "categorical", "choices": [0.0, 0.0005, 0.001, 0.002]},
    },
    "regime": {
        "n_regimes": {"type": "categorical", "choices": [2, 3, 4]},
    },
    "model": {
        "type": {"type": "categorical", "choices": ["rf", "gbm", "logistic"]},
        "gap": {"type": "categorical", "choices": [12, 24, 48]},
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
_THESIS_SPACE_PATHS = {
    ("features", "lags"),
    ("features", "frac_diff_d"),
    ("features", "rolling_window"),
    ("features", "squeeze_quantile"),
    ("feature_selection", "enabled"),
    ("feature_selection", "max_features"),
    ("feature_selection", "min_mi_threshold"),
    ("labels", "min_return"),
    ("labels", "barrier_tie_break"),
    ("regime", "n_regimes"),
    ("model", "gap"),
}
_MODEL_FAMILY_SPACE_PATHS = {
    ("model", "type"),
}
_FORBIDDEN_SEARCH_PATHS = {
    ("model", "validation_fraction"),
    ("model", "meta_n_splits"),
    ("labels", "pt_mult"),
    ("labels", "sl_mult"),
    ("labels", "max_holding"),
    ("labels", "volatility_window"),
}


def _spec_allows_variation(spec):
    if not isinstance(spec, dict):
        return False
    if "choices" in spec:
        return len(list(spec.get("choices") or [])) > 1
    spec_type = str(spec.get("type", "")).lower()
    if spec_type in {"float", "int"}:
        low = spec.get("low")
        high = spec.get("high")
        step = spec.get("step")
        if low is None or high is None:
            return False
        if low != high:
            return True
        return step not in (None, 0)
    return False


def _classify_search_space(search_space):
    thesis_space = {}
    model_family_space = {}
    hyperparameter_space = {}
    for section_name, section in dict(search_space or {}).items():
        section = dict(section or {})
        for key, spec in section.items():
            path = (section_name, key)
            if path in _THESIS_SPACE_PATHS:
                thesis_space.setdefault(section_name, {})[key] = copy.deepcopy(spec)
            elif path in _MODEL_FAMILY_SPACE_PATHS:
                model_family_space.setdefault(section_name, {})[key] = copy.deepcopy(spec)
            else:
                hyperparameter_space.setdefault(section_name, {})[key] = copy.deepcopy(spec)
    return {
        "thesis_space": thesis_space,
        "model_family_space": model_family_space,
        "hyperparameter_space": hyperparameter_space,
    }


def _find_varying_thesis_paths(search_space_tiers):
    violations = []
    for section_name, section in dict((search_space_tiers or {}).get("thesis_space") or {}).items():
        for key, spec in dict(section or {}).items():
            if _spec_allows_variation(spec):
                violations.append(f"{section_name}.{key}")
    return sorted(violations)


def _find_forbidden_search_paths(search_space):
    violations = []
    for section_name, section in dict(search_space or {}).items():
        for key in dict(section or {}).keys():
            if (section_name, key) in _FORBIDDEN_SEARCH_PATHS:
                violations.append(f"{section_name}.{key}")
    return sorted(violations)


def _validate_forbidden_search_space_paths(search_space):
    violations = _find_forbidden_search_paths(search_space)
    if violations:
        joined = ", ".join(violations)
        raise ValueError(
            "Search space includes forbidden thesis/data-split parameters. Remove these entries: "
            f"{joined}"
        )


def _validate_trade_ready_search_space(search_space, automl_config):
    trade_ready_profile = dict((automl_config or {}).get("trade_ready_profile") or {})
    if not trade_ready_profile or bool(trade_ready_profile.get("reduced_power", False)):
        return
    search_space_tiers = _classify_search_space(search_space)
    varying_paths = _find_varying_thesis_paths(search_space_tiers)
    if varying_paths:
        joined = ", ".join(varying_paths)
        raise ValueError(
            "Certification AutoML cannot search thesis_space parameters. Freeze these entries before running the study: "
            f"{joined}"
        )


def _resolve_experiment_family_id(search_space):
    tiers = _classify_search_space(search_space)
    return payload_sha256(
        {
            "thesis_space": tiers.get("thesis_space") or {},
            "model_family_space": tiers.get("model_family_space") or {},
        }
    )


def _normalize_objective_name(objective_name):
    objective_name = (objective_name or "risk_adjusted_after_costs").lower()
    aliases = {
        "composite": "accuracy_first",
        "trading_first": "risk_adjusted_after_costs",
        "after_cost_sharpe": "risk_adjusted_after_costs",
    }
    if objective_name in aliases:
        return aliases[objective_name]
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


def _resolve_resume_mode(automl_config):
    resume_mode = str((automl_config or {}).get("resume_mode", "never")).strip().lower()
    if resume_mode not in {"never", "if_manifest_matches", "force"}:
        raise ValueError(
            "Invalid automl.resume_mode. Expected one of never, if_manifest_matches, or force. "
            f"Received: {resume_mode!r}"
        )
    return resume_mode


def _resolve_code_revision(base_config, automl_config):
    explicit = (
        (automl_config or {}).get("code_revision")
        or (base_config or {}).get("code_revision")
        or os.getenv("BUILD_SOURCEVERSION")
        or os.getenv("GIT_COMMIT")
        or os.getenv("GITHUB_SHA")
    )
    if explicit:
        return str(explicit)

    repo_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return "unknown"

    revision = str(result.stdout or "").strip()
    return revision or "unknown"


def _resolve_experiment_storage_root(base_config, automl_config):
    explicit = (automl_config or {}).get("storage")
    if explicit:
        path = Path(explicit)
        if path.suffix:
            root = path.parent / path.stem
        else:
            root = path
    else:
        study_name = _resolve_study_name(base_config, automl_config)
        root = Path(".cache") / "automl" / study_name

    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _next_experiment_run_id(runs_dir):
    existing = []
    for path in Path(runs_dir).glob("run_*"):
        suffix = path.name.replace("run_", "", 1)
        if suffix.isdigit():
            existing.append(int(suffix))
    next_id = max(existing, default=0) + 1
    return f"run_{next_id:04d}"


def _build_experiment_manifest(base_config, automl_config, state_bundle, search_space):
    raw_data_value = state_bundle.get("raw_data")
    raw_data = pd.DataFrame(raw_data_value, copy=True) if raw_data_value is not None else pd.DataFrame()
    raw_bounds = {
        "start_timestamp": None,
        "end_timestamp": None,
        "row_count": 0,
        "raw_data_fingerprint": None,
    }
    if not raw_data.empty:
        raw_bounds = {
            "start_timestamp": raw_data.index[0].isoformat(),
            "end_timestamp": raw_data.index[-1].isoformat(),
            "row_count": int(len(raw_data)),
            "raw_data_fingerprint": frame_fingerprint(raw_data),
        }

    feature_contract = {
        "features": _json_ready(copy.deepcopy(base_config.get("features") or {})),
        "feature_selection": _json_ready(copy.deepcopy(base_config.get("feature_selection") or {})),
        "custom_data": _json_ready(copy.deepcopy(base_config.get("custom_data") or {})),
    }
    objective_contract = {
        "objective": _normalize_objective_name((automl_config or {}).get("objective", "risk_adjusted_after_costs")),
        "objective_gates": _json_ready(copy.deepcopy((automl_config or {}).get("objective_gates") or {})),
        "selection_policy": _json_ready(copy.deepcopy((automl_config or {}).get("selection_policy") or {})),
        "overfitting_control": _json_ready(copy.deepcopy((automl_config or {}).get("overfitting_control") or {})),
        "replication": _json_ready(copy.deepcopy((automl_config or {}).get("replication") or {})),
        "trade_ready_profile": _json_ready(copy.deepcopy((automl_config or {}).get("trade_ready_profile") or {})),
    }
    data_contract = {
        "symbol": str((base_config.get("data") or {}).get("symbol", "unknown")),
        "interval": str((base_config.get("data") or {}).get("interval", "unknown")),
        **raw_bounds,
        "data_lineage": _json_ready(copy.deepcopy(state_bundle.get("data_lineage") or {})),
        "symbol_filters": _json_ready(copy.deepcopy(state_bundle.get("symbol_filters") or {})),
        "universe_policy": _json_ready(copy.deepcopy(state_bundle.get("universe_policy") or {})),
        "universe_snapshot_meta": _json_ready(copy.deepcopy(state_bundle.get("universe_snapshot_meta") or {})),
        "eligible_symbols": _json_ready(copy.deepcopy(state_bundle.get("eligible_symbols") or [])),
    }

    manifest = {
        "manifest_schema": "automl_experiment_manifest.v1",
        "study_label": _resolve_study_name(base_config, automl_config),
        "resume_mode": _resolve_resume_mode(automl_config),
        "data_contract": data_contract,
        "feature_contract": feature_contract,
        "objective_contract": objective_contract,
        "search_space": _json_ready(copy.deepcopy(search_space)),
        "search_space_tiers": _json_ready(_classify_search_space(search_space)),
        "search_space_hash": payload_sha256(search_space),
        "feature_schema_hash": payload_sha256(feature_contract),
        "objective_hash": payload_sha256(objective_contract),
        "data_lineage_hash": payload_sha256(data_contract),
        "code_revision": _resolve_code_revision(base_config, automl_config),
    }
    manifest["experiment_family_id"] = _resolve_experiment_family_id(search_space)
    manifest["experiment_id"] = payload_sha256(
        {
            "data_lineage_hash": manifest["data_lineage_hash"],
            "feature_schema_hash": manifest["feature_schema_hash"],
            "objective_hash": manifest["objective_hash"],
            "search_space_hash": manifest["search_space_hash"],
            "code_revision": manifest["code_revision"],
        }
    )
    return manifest


def _build_experiment_storage_context(base_config, automl_config, experiment_manifest):
    storage_root = _resolve_experiment_storage_root(base_config, automl_config)
    experiment_dir = storage_root / "experiments" / experiment_manifest["experiment_id"]
    resume_mode = experiment_manifest["resume_mode"]
    if resume_mode == "never":
        runs_dir = experiment_dir / "runs"
        run_id = _next_experiment_run_id(runs_dir)
        run_dir = runs_dir / run_id
    else:
        run_id = None
        run_dir = experiment_dir

    return {
        "storage_root": storage_root,
        "experiment_dir": experiment_dir,
        "run_dir": run_dir,
        "run_id": run_id,
        "study_path": run_dir / "study.db",
        "manifest_path": experiment_dir / "manifest.json",
        "lineage_path": experiment_dir / "lineage.json",
        "summary_path": run_dir / "summary.json",
    }


def _validate_resume_manifest(storage_context, experiment_manifest):
    resume_mode = experiment_manifest["resume_mode"]
    experiment_dir = Path(storage_context["experiment_dir"])
    manifest_path = Path(storage_context["manifest_path"])
    study_path = Path(storage_context["study_path"])
    stored_manifest = read_json(manifest_path)
    study_exists = study_path.exists()

    if resume_mode == "never":
        return {
            "resume_mode": resume_mode,
            "load_if_exists": False,
            "resumed_existing_study": False,
            "manifest_matches": False,
            "mismatched_fields": [],
            "experiment_dir": str(experiment_dir),
            "study_path": str(study_path),
        }

    if not study_exists:
        return {
            "resume_mode": resume_mode,
            "load_if_exists": False,
            "resumed_existing_study": False,
            "manifest_matches": stored_manifest == experiment_manifest if stored_manifest is not None else False,
            "mismatched_fields": [],
            "experiment_dir": str(experiment_dir),
            "study_path": str(study_path),
        }

    if resume_mode == "force":
        mismatched_fields = []
        if isinstance(stored_manifest, dict):
            for field in [
                "experiment_id",
                "data_lineage_hash",
                "feature_schema_hash",
                "objective_hash",
                "search_space_hash",
                "code_revision",
            ]:
                if stored_manifest.get(field) != experiment_manifest.get(field):
                    mismatched_fields.append(field)
        return {
            "resume_mode": resume_mode,
            "load_if_exists": True,
            "resumed_existing_study": True,
            "manifest_matches": not mismatched_fields,
            "mismatched_fields": mismatched_fields,
            "experiment_dir": str(experiment_dir),
            "study_path": str(study_path),
        }

    if stored_manifest is None:
        raise RuntimeError(
            "AutoML resume requested with resume_mode=if_manifest_matches, but the stored experiment manifest is missing. "
            f"Archive or delete {experiment_dir} or rerun with resume_mode=force."
        )

    mismatched_fields = []
    for field in [
        "experiment_id",
        "data_lineage_hash",
        "feature_schema_hash",
        "objective_hash",
        "search_space_hash",
        "code_revision",
    ]:
        if stored_manifest.get(field) != experiment_manifest.get(field):
            mismatched_fields.append(field)

    if mismatched_fields:
        raise RuntimeError(
            "AutoML resume manifest mismatch. "
            f"stored_experiment_id={stored_manifest.get('experiment_id')} current_experiment_id={experiment_manifest.get('experiment_id')} "
            f"mismatched_fields={','.join(mismatched_fields)} archive_or_delete={experiment_dir}"
        )

    return {
        "resume_mode": resume_mode,
        "load_if_exists": True,
        "resumed_existing_study": True,
        "manifest_matches": True,
        "mismatched_fields": [],
        "experiment_dir": str(experiment_dir),
        "study_path": str(study_path),
    }


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
_TEMPORAL_STATE_KEYS = [
    "indicator_run", "futures_context", "cross_asset_context",
    "symbol_lifecycle", "universe_snapshot", "universe_report",
]
_DEEP_COPY_STATE_KEYS = [
    "data_lineage", "symbol_filters", "universe_policy",
    "universe_snapshot_meta", "eligible_symbols",
    "custom_data_report", "data_integrity_report", "data_quality_mask",
    "data_quality_report", "reference_integrity_report",
]


def _build_state_bundle(base_pipeline):
    bundle = {
        "raw_data": base_pipeline.require("raw_data").copy(),
        "data": base_pipeline.require("data").copy(),
    }
    for k in _TEMPORAL_STATE_KEYS:
        bundle[k] = _slice_temporal_value(base_pipeline.state.get(k))
    for k in _DEEP_COPY_STATE_KEYS:
        bundle[k] = copy.deepcopy(base_pipeline.state.get(k))
    return bundle


def _build_window_state_bundle(full_state_bundle, start_timestamp=None, end_timestamp=None):
    raw_data = pd.DataFrame(full_state_bundle["raw_data"], copy=True)
    if start_timestamp is not None:
        raw_data = raw_data.loc[raw_data.index >= start_timestamp].copy()
    if end_timestamp is not None:
        raw_data = raw_data.loc[raw_data.index <= end_timestamp].copy()

    slice_index = raw_data.index
    bundle = {
        "raw_data": raw_data,
        "data": full_state_bundle["data"].reindex(slice_index).copy(),
    }
    for k in _TEMPORAL_STATE_KEYS:
        bundle[k] = _slice_temporal_value(full_state_bundle.get(k), start_timestamp=start_timestamp, end_timestamp=end_timestamp)
    for k in _DEEP_COPY_STATE_KEYS:
        bundle[k] = copy.deepcopy(full_state_bundle.get(k))
    return bundle


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


def _build_capital_evidence_contract(base_config, *, holdout_plan=None, base_pipeline=None):
    automl_config = dict((base_config or {}).get("automl") or {})
    evaluation_mode = resolve_evaluation_mode((base_config or {}).get("backtest") or {})
    selection_policy = _resolve_selection_policy(automl_config)
    overfitting_control = _resolve_overfitting_control(automl_config)
    replication_config = _resolve_replication_config(base_config or {}, automl_config, base_pipeline=base_pipeline)
    trade_ready_profile = dict(automl_config.get("trade_ready_profile") or {})

    required_controls = {
        "locked_holdout": bool(evaluation_mode.is_capital_facing),
        "selection_policy": bool(evaluation_mode.is_capital_facing),
        "post_selection": bool(evaluation_mode.is_capital_facing),
        "replication": bool(evaluation_mode.is_capital_facing),
    }
    observed_controls = {
        "locked_holdout": bool((holdout_plan or {}).get("enabled", False)),
        "selection_policy": bool(selection_policy.get("enabled", False)),
        "post_selection": bool(
            overfitting_control.get("enabled", False) and overfitting_control.get("post_selection", {}).get("enabled", False)
        ),
        "replication": bool(replication_config.get("enabled", False)),
    }

    blocking_reasons = []
    if evaluation_mode.is_capital_facing:
        for control_name, required in required_controls.items():
            if required and not observed_controls.get(control_name, False):
                blocking_reasons.append(f"required_control_disabled:{control_name}")
        if bool(trade_ready_profile.get("reduced_power", False)):
            blocking_reasons.append("reduced_power_profile_not_capital_eligible")

    return {
        "requested_mode": evaluation_mode.requested_mode,
        "effective_mode": evaluation_mode.effective_mode,
        "capital_path_eligible": bool(evaluation_mode.is_capital_facing and not blocking_reasons),
        "required_controls": required_controls,
        "observed_controls": observed_controls,
        "blocking_reasons": blocking_reasons,
    }


def _validate_capital_evidence_contract(base_config, *, holdout_plan=None, base_pipeline=None):
    contract = _build_capital_evidence_contract(
        base_config,
        holdout_plan=holdout_plan,
        base_pipeline=base_pipeline,
    )
    hard_failures = [
        reason for reason in contract.get("blocking_reasons", []) if str(reason).startswith("required_control_disabled:")
    ]
    if hard_failures:
        raise RuntimeError(
            "Capital-facing AutoML configuration is invalid because required evidence controls are disabled: "
            f"{', '.join(hard_failures)}"
        )
    return contract


def _build_oos_evidence(base_config, *, holdout_plan=None, selection_diagnostics=None, training=None, base_pipeline=None):
    base_config = base_config or {}
    automl_config = dict(base_config.get("automl") or {})
    model_config = dict(base_config.get("model") or {})
    evaluation_mode = resolve_evaluation_mode(base_config.get("backtest") or {})
    validation_contract = _resolve_validation_contract(
        base_config,
        automl_config,
        training=training,
        holdout_enabled=bool((holdout_plan or {}).get("enabled", False)),
    )
    overfitting_control = _resolve_overfitting_control(automl_config)
    replication_config = _resolve_replication_config(base_config, automl_config, base_pipeline=base_pipeline)
    post_selection_report = dict((selection_diagnostics or {}).get("post_selection") or {})

    search_ranker = _normalize_validation_source_name(validation_contract.get("search_ranker"))
    model_validation_method = _normalize_validation_source_name(model_config.get("cv_method", "cpcv"))
    purging_rows = 0
    for row in list((training or {}).get("purging") or []):
        purging_rows += int(row.get("outer_purged_rows", 0))
        purging_rows += int(row.get("inner_purged_rows", 0))
        purging_rows += int(row.get("embargo_rows", 0))

    configured_embargo_bars = int((holdout_plan or {}).get("configured_embargo_bars", 0))
    search_stage_method = search_ranker or model_validation_method or "walk_forward"
    purged_temporal_search = bool(
        search_stage_method in {"walk_forward", "walk_forward_replay"}
        and max(configured_embargo_bars, purging_rows) > 0
    )

    controls = {
        "cpcv_or_purged_temporal_search": {
            "required": bool(evaluation_mode.is_capital_facing),
            "complete": bool(search_stage_method == "cpcv" or purged_temporal_search),
            "provenance": {
                "validation_contract": search_ranker,
                "model_cv_method": model_validation_method,
                "configured_embargo_bars": configured_embargo_bars,
                "observed_purging_rows": int(purging_rows),
            },
        },
        "search_stage_embargo": {
            "required": bool(evaluation_mode.is_capital_facing),
            "complete": bool(int((holdout_plan or {}).get("search_validation_gap_bars", 0)) > 0),
            "provenance": {
                "gap_rows": int((holdout_plan or {}).get("search_validation_gap_bars", 0)),
                "gap_start": _json_ready((holdout_plan or {}).get("search_validation_gap_start_timestamp")),
                "gap_end": _json_ready((holdout_plan or {}).get("search_validation_gap_end_timestamp")),
            },
        },
        "validation_holdout_gap": {
            "required": bool(evaluation_mode.is_capital_facing),
            "complete": bool(int((holdout_plan or {}).get("validation_holdout_gap_bars", 0)) > 0),
            "provenance": {
                "gap_rows": int((holdout_plan or {}).get("validation_holdout_gap_bars", 0)),
                "gap_start": _json_ready((holdout_plan or {}).get("validation_holdout_gap_start_timestamp")),
                "gap_end": _json_ready((holdout_plan or {}).get("validation_holdout_gap_end_timestamp")),
            },
        },
        "locked_holdout": {
            "required": bool(evaluation_mode.is_capital_facing),
            "complete": bool(
                validation_contract.get("locked_holdout") == "single_access_contiguous"
                and bool((holdout_plan or {}).get("enabled", False))
            ),
            "provenance": {
                "validation_contract": validation_contract.get("locked_holdout"),
                "enabled": bool((holdout_plan or {}).get("enabled", False)),
                "holdout_rows": int((holdout_plan or {}).get("holdout_rows", 0)),
            },
        },
        "post_selection_inference": {
            "required": bool(evaluation_mode.is_capital_facing),
            "complete": bool(
                overfitting_control.get("enabled", False)
                and overfitting_control.get("post_selection", {}).get("enabled", False)
            ),
            "provenance": {
                "overfitting_enabled": bool(overfitting_control.get("enabled", False)),
                "configured": bool(overfitting_control.get("post_selection", {}).get("enabled", False)),
                "report_enabled": bool(post_selection_report.get("enabled", False)),
                "require_pass": bool(post_selection_report.get("require_pass", False)),
                "passed": bool(post_selection_report.get("passed", True)),
                "reason": post_selection_report.get("reason"),
            },
        },
        "replication": {
            "required": bool(evaluation_mode.is_capital_facing),
            "complete": bool(replication_config.get("enabled", False)),
            "provenance": {
                "configured": bool(replication_config.get("enabled", False)),
                "min_coverage": int(replication_config.get("min_coverage", 0)),
                "include_symbol_cohorts": bool(replication_config.get("include_symbol_cohorts", False)),
                "include_window_cohorts": bool(replication_config.get("include_window_cohorts", False)),
            },
        },
    }

    evidence_stack_complete = bool(all(bool(control.get("complete", False)) for control in controls.values()))
    blocking_reasons = [
        f"oos_control_incomplete:{name}" for name, control in controls.items() if not bool(control.get("complete", False))
    ]
    any_complete = any(bool(control.get("complete", False)) for control in controls.values())
    oos_class = "adversarial_oos" if evidence_stack_complete else ("partial_oos" if any_complete else "search_only")

    return {
        "class": oos_class,
        "evidence_stack_complete": evidence_stack_complete,
        "controls": controls,
        "blocking_reasons": blocking_reasons,
    }


def _validate_oos_evidence_preconditions(base_config, *, holdout_plan=None, base_pipeline=None):
    oos_evidence = _build_oos_evidence(
        base_config,
        holdout_plan=holdout_plan,
        selection_diagnostics=None,
        training=None,
        base_pipeline=base_pipeline,
    )
    if resolve_evaluation_mode((base_config or {}).get("backtest") or {}).is_capital_facing and not oos_evidence[
        "evidence_stack_complete"
    ]:
        raise RuntimeError(
            "Capital-facing AutoML requires a complete adversarial OOS evidence stack before optimization starts: "
            f"{', '.join(oos_evidence['blocking_reasons'])}"
        )
    return oos_evidence


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


def _normalize_validation_source_name(source, *, executable=False):
    value = str(source or "").strip().lower()
    if not value:
        return None
    aliases = {
        "walk-forward": "walk_forward",
        "walkforward": "walk_forward",
        "wf": "walk_forward",
        "validation": "walk_forward",
        "executable_validation": "walk_forward_replay",
    }
    value = aliases.get(value, value)
    if executable and value == "walk_forward":
        return "walk_forward_replay"
    return value


def _resolve_validation_contract(base_config, automl_config, *, training=None, holdout_enabled=None):
    base_config = base_config or {}
    automl_config = automl_config or {}
    trade_ready_profile = dict(automl_config.get("trade_ready_profile") or {})
    model_config = dict(base_config.get("model") or {})
    training = training or {}
    training_validation = dict(training.get("validation") or {})
    executable_validation = dict(training.get("executable_validation") or {})
    executable_training = dict(executable_validation.get("training") or {}) if executable_validation.get("enabled") else {}
    executable_training_validation = dict(executable_training.get("validation") or {})

    primary_source = _normalize_validation_source_name(
        training_validation.get("method") or model_config.get("cv_method", "cpcv")
    )
    tradable_source = _normalize_validation_source_name(
        executable_training_validation.get("method") or model_config.get("cv_method", "cpcv"),
        executable=bool(executable_validation.get("enabled")),
    )
    if tradable_source == "cpcv" and executable_validation.get("enabled"):
        tradable_source = "walk_forward_replay"

    replication_enabled = bool((automl_config.get("replication") or {}).get("enabled", False))
    if holdout_enabled is None:
        holdout_enabled = bool(
            automl_config.get("locked_holdout_enabled", False)
            or automl_config.get("locked_holdout_fraction")
        )

    if trade_ready_profile:
        defaults = {
            "search_ranker": "cpcv",
            "contiguous_validation": "walk_forward_replay",
            "locked_holdout": "single_access_contiguous",
            "replication": "required" if replication_enabled else "disabled",
        }
    else:
        defaults = {
            "search_ranker": primary_source or "walk_forward",
            "contiguous_validation": tradable_source or primary_source or "walk_forward",
            "locked_holdout": "single_access_contiguous" if holdout_enabled else "disabled",
            "replication": "required" if replication_enabled else "disabled",
        }

    contract = copy.deepcopy(automl_config.get("validation_contract") or {})
    return {
        "search_ranker": str(contract.get("search_ranker", defaults["search_ranker"])),
        "contiguous_validation": str(contract.get("contiguous_validation", defaults["contiguous_validation"])),
        "locked_holdout": str(contract.get("locked_holdout", defaults["locked_holdout"])),
        "replication": str(contract.get("replication", defaults["replication"])),
    }


def _validation_contract_stage_pass(stage_source, *, diagnostic_source, selection_source, tradable_source, enabled):
    stage_source = _normalize_validation_source_name(stage_source)
    if stage_source in {None, "disabled"}:
        return True
    if stage_source == "cpcv":
        return diagnostic_source == "cpcv"
    if stage_source == "single_access_contiguous":
        return bool(enabled)
    if stage_source == "required":
        return bool(enabled)
    if stage_source == "walk_forward_replay":
        return tradable_source == "walk_forward_replay"
    if stage_source == "walk_forward":
        return tradable_source in {"walk_forward", "walk_forward_replay"} or selection_source in {
            "walk_forward",
            "walk_forward_replay",
        }
    return bool(enabled)


def _resolve_validation_sources(training, backtest, validation_contract, *, holdout_enabled=False, replication_enabled=False):
    training = training or {}
    backtest = backtest or {}
    validation = dict(training.get("validation") or {})
    executable_validation = dict(training.get("executable_validation") or {})
    executable_training = dict(executable_validation.get("training") or {}) if executable_validation.get("enabled") else {}
    executable_training_validation = dict(executable_training.get("validation") or {})

    diagnostic_source = _normalize_validation_source_name(
        backtest.get("diagnostic_validation_method") or validation.get("method")
    )
    selection_source = _normalize_validation_source_name(
        executable_training_validation.get("method") or backtest.get("validation_method") or validation.get("method"),
        executable=bool(executable_validation.get("enabled")),
    )
    tradable_source = _normalize_validation_source_name(
        backtest.get("validation_method") or executable_training_validation.get("method") or validation.get("method"),
        executable=bool(executable_validation.get("enabled")),
    )

    source_checks = {
        "search_ranker": _validation_contract_stage_pass(
            validation_contract.get("search_ranker"),
            diagnostic_source=diagnostic_source,
            selection_source=selection_source,
            tradable_source=tradable_source,
            enabled=bool(diagnostic_source),
        ),
        "contiguous_validation": _validation_contract_stage_pass(
            validation_contract.get("contiguous_validation"),
            diagnostic_source=diagnostic_source,
            selection_source=selection_source,
            tradable_source=tradable_source,
            enabled=bool(tradable_source),
        ),
        "locked_holdout": _validation_contract_stage_pass(
            validation_contract.get("locked_holdout"),
            diagnostic_source=diagnostic_source,
            selection_source=selection_source,
            tradable_source=tradable_source,
            enabled=bool(holdout_enabled),
        ),
        "replication": _validation_contract_stage_pass(
            validation_contract.get("replication"),
            diagnostic_source=diagnostic_source,
            selection_source=selection_source,
            tradable_source=tradable_source,
            enabled=bool(replication_enabled),
        ),
    }
    return {
        "validation_contract": dict(validation_contract or {}),
        "selection_metric_source": selection_source,
        "diagnostic_metric_source": diagnostic_source,
        "tradable_metric_source": tradable_source,
        "required_source_checks": source_checks,
        "all_required_sources_passed": bool(all(source_checks.values())),
    }


def _summarize_training(training):
    primary_training, primary_source = _resolve_primary_training_payload(training)
    feature_selection = training.get("feature_selection") or {}
    bootstrap = training.get("bootstrap") or {}
    feature_governance = training.get("feature_governance") or {}
    feature_portability_diagnostics = training.get("feature_portability_diagnostics") or {}
    regime = training.get("regime") or {}
    operational_monitoring = training.get("operational_monitoring") or {}
    cross_venue_integrity = training.get("cross_venue_integrity") or {}
    data_certification = training.get("data_certification") or {}
    signal_decay = training.get("signal_decay") or {}
    validation_sources = dict(training.get("validation_sources") or {})
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
        "selection_metric_source": validation_sources.get("selection_metric_source", primary_source),
        "diagnostic_metric_source": validation_sources.get("diagnostic_metric_source"),
        "tradable_metric_source": validation_sources.get("tradable_metric_source"),
        "all_required_sources_passed": bool(validation_sources.get("all_required_sources_passed", True)),
        "validation_sources": validation_sources,
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
        "cross_venue_integrity": (
            {
                "kind": cross_venue_integrity.get("kind"),
                "promotion_pass": bool(cross_venue_integrity.get("promotion_pass", False)),
                "gate_mode": cross_venue_integrity.get("gate_mode"),
                "reasons": list(cross_venue_integrity.get("reasons", [])),
                "warnings": list(cross_venue_integrity.get("warnings", [])),
                "venues": cross_venue_integrity.get("venues", {}),
                "self_consistency": cross_venue_integrity.get("self_consistency", {}),
                "divergence": cross_venue_integrity.get("divergence", {}),
                "overlay_columns": list(cross_venue_integrity.get("overlay_columns", [])),
            }
            if cross_venue_integrity
            else {}
        ),
        "data_certification": data_certification,
        "signal_decay": signal_decay,
        "feature_portability_diagnostics": feature_portability_diagnostics,
        "regime": regime,
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


def _summarize_backtest(backtest, *, evidence_class=None):
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
    summary["evidence_class"] = str(evidence_class or backtest.get("evidence_class") or "backtest_payload")
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


def _build_explicit_temporal_split(index, train_end_timestamp, test_start_timestamp, excluded_intervals=None):
    aligned_index = pd.Index(index)
    if aligned_index.empty:
        raise RuntimeError("Aligned split empty")

    train_mask = np.asarray(aligned_index <= train_end_timestamp, dtype=bool)
    test_mask = np.asarray(aligned_index >= test_start_timestamp, dtype=bool)
    excluded_mask = np.zeros(len(aligned_index), dtype=bool)
    for interval_start, interval_end in excluded_intervals or []:
        if interval_start is None or interval_end is None:
            continue
        excluded_mask |= np.asarray((aligned_index >= interval_start) & (aligned_index <= interval_end), dtype=bool)

    train_mask &= ~excluded_mask
    test_mask &= ~excluded_mask
    gap_mask = ~(train_mask | test_mask)

    train_index = np.flatnonzero(train_mask)
    test_index = np.flatnonzero(test_mask)
    gap_index = np.flatnonzero(gap_mask)
    if len(train_index) <= 0 or len(test_index) <= 0:
        raise RuntimeError("Aligned split empty")

    return {
        "split_id": "validation_stage_0",
        "train_index": train_index.tolist(),
        "test_index": test_index.tolist(),
        "gap_index": gap_index.tolist(),
        "gap_bars": int(len(gap_index)),
        "source": "staged_holdout_plan",
        "timestamp_bounds": {
            "train_end": _json_ready(train_end_timestamp),
            "test_start": _json_ready(test_start_timestamp),
        },
        "excluded_intervals": _json_ready(excluded_intervals or []),
    }


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

    explicit_split = _build_explicit_temporal_split(
        candidate.state["X"].index,
        train_end_timestamp,
        test_start_timestamp,
        excluded_intervals=excluded_intervals,
    )
    aligned_train_rows = int(len(explicit_split["train_index"]))
    aligned_test_rows = int(len(explicit_split["test_index"]))
    aligned_gap_rows = int(len(explicit_split["gap_index"]))

    candidate.config["model"] = {
        **candidate.config.get("model", {}),
        "cv_method": "walk_forward",
        "n_splits": 1,
        "train_size": aligned_train_rows,
        "test_size": aligned_test_rows,
        "gap": 0,
        "explicit_splits": [explicit_split],
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
        "gap_audit": {
            "split_id": explicit_split["split_id"],
            "source": explicit_split["source"],
            "gap_rows": int(aligned_gap_rows),
            "gap_bars": int(explicit_split["gap_bars"]),
            "timestamp_bounds": dict(explicit_split["timestamp_bounds"]),
        },
    }


def _build_validation_holdout_report(best_trial_report, holdout_plan):
    report = {
        "evidence_class": "outer_replay",
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
        "evidence_class": "locked_holdout",
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
    report["backtest"] = _json_ready(_summarize_backtest(backtest, evidence_class="locked_holdout"))
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
    portability_contract = copy.deepcopy((automl_config or {}).get("portability_contract") or {})
    accepted_kinds = [
        str(kind).strip().lower()
        for kind in (portability_contract.get("accepted_kinds") or ["symbol", "period", "venue"])
        if str(kind).strip()
    ]
    universe_snapshot_meta = dict((base_pipeline.state.get("universe_snapshot_meta") or {}) if base_pipeline is not None else {})
    universe_snapshot_source = str(universe_snapshot_meta.get("source") or "").strip().lower() or None
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
        "portability_contract": {
            "enabled": bool(portability_contract.get("enabled", False)),
            "accepted_kinds": accepted_kinds,
            "min_supporting_cohorts": int(portability_contract.get("min_supporting_cohorts", 1)),
            "min_passed_supporting_cohorts": int(portability_contract.get("min_passed_supporting_cohorts", 1)),
            "require_frozen_universe": bool(portability_contract.get("require_frozen_universe", False)),
            "frozen_universe_available": bool(
                universe_snapshot_meta.get("snapshot_timestamp") is not None
                and universe_snapshot_source not in {"exchange_info", "synthetic_example_snapshot"}
            ),
            "universe_snapshot_source": universe_snapshot_meta.get("source"),
        },
    }


def _finalize_portability_contract(report, primary_score=None):
    contract = dict(report.get("portability_contract") or {})
    contract.setdefault("enabled", False)
    contract.setdefault("accepted_kinds", [])
    contract.setdefault("min_supporting_cohorts", 1)
    contract.setdefault("min_passed_supporting_cohorts", 1)
    contract.setdefault("require_frozen_universe", False)
    contract.setdefault("frozen_universe_available", False)
    contract.setdefault("universe_snapshot_source", None)
    contract["supporting_cohort_attempted_count"] = 0
    contract["supporting_cohort_completed_count"] = 0
    contract["supporting_cohort_pass_count"] = 0
    contract["distinct_supporting_kinds"] = []
    contract["distinct_passed_kinds"] = []
    contract["median_degradation_vs_primary"] = None
    contract["tail_degradation_vs_primary"] = None
    contract["passed"] = True
    contract["reasons"] = []

    if not bool(contract.get("enabled", False)):
        report["portability_contract"] = contract
        return report

    accepted_kinds = {str(kind).strip().lower() for kind in (contract.get("accepted_kinds") or []) if str(kind).strip()}
    supporting_rows = [
        cohort
        for cohort in (report.get("cohorts") or [])
        if str(cohort.get("kind") or "").strip().lower() in accepted_kinds
    ]
    supporting_completed = [cohort for cohort in supporting_rows if cohort.get("completed")]
    supporting_passed = [cohort for cohort in supporting_completed if cohort.get("passed")]
    contract["supporting_cohort_attempted_count"] = int(len(supporting_rows))
    contract["supporting_cohort_completed_count"] = int(len(supporting_completed))
    contract["supporting_cohort_pass_count"] = int(len(supporting_passed))
    contract["distinct_supporting_kinds"] = sorted(
        {str(cohort.get("kind") or "").strip().lower() for cohort in supporting_completed}
    )
    contract["distinct_passed_kinds"] = sorted(
        {str(cohort.get("kind") or "").strip().lower() for cohort in supporting_passed}
    )
    degradations = []
    if primary_score is not None:
        for cohort in supporting_completed:
            score = _coerce_float(cohort.get("score"))
            if score is None:
                continue
            degradation = float(primary_score) - float(score)
            cohort["score_degradation_vs_primary"] = degradation
            degradations.append(degradation)
    if degradations:
        contract["median_degradation_vs_primary"] = float(np.median(degradations))
        contract["tail_degradation_vs_primary"] = float(max(degradations))

    reasons = []
    if (
        bool(contract.get("require_frozen_universe", False))
        and bool(report.get("include_symbol_cohorts", False))
        and "symbol" in accepted_kinds
        and not bool(contract.get("frozen_universe_available", False))
    ):
        reasons.append("portability_requires_frozen_universe_snapshot")
    if (
        bool(report.get("include_symbol_cohorts", False))
        and "symbol" in accepted_kinds
        and str(contract.get("universe_snapshot_source") or "").strip().lower() == "synthetic_example_snapshot"
    ):
        reasons.append("synthetic_universe_snapshot_not_allowed")
    if int(len(supporting_rows)) < int(contract.get("min_supporting_cohorts", 1)):
        reasons.append("portability_supporting_cohort_missing")
    if int(len(supporting_passed)) < int(contract.get("min_passed_supporting_cohorts", 1)):
        reasons.append("portability_supporting_pass_count_below_minimum")

    deduped = []
    for reason in list(report.get("reasons") or []) + reasons:
        text = str(reason or "").strip()
        if text and text not in deduped:
            deduped.append(text)
    report["reasons"] = deduped
    contract["reasons"] = reasons
    contract["passed"] = not reasons
    report["portability_contract"] = contract
    report["promotion_pass"] = bool(report.get("promotion_pass", False) and contract["passed"])
    return report


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
    primary_score=None,
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
        "portability_contract": copy.deepcopy(replication_config.get("portability_contract") or {}),
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
        return _finalize_portability_contract(report, primary_score=primary_score)

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
                evidence_class="replication",
            )
        except RuntimeError as exc:
            cohort_row["reason"] = str(exc)
            report["cohorts"].append(cohort_row)
            continue

        score = _coerce_float((evaluation.get("objective_diagnostics") or {}).get("raw_score"))
        if score is None:
            score = _coerce_float(evaluation.get("raw_objective_value"))
        backtest_summary = evaluation.get("backtest") or {}
        cohort_row["completed"] = True
        cohort_row["score"] = score
        cohort_row["net_profit_pct"] = _coerce_float(backtest_summary.get("net_profit_pct"))
        cohort_row["total_trades"] = int(backtest_summary.get("total_trades") or 0)
        cohort_row["score_degradation_vs_primary"] = (
            None if primary_score is None or score is None else float(primary_score) - float(score)
        )
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
    return _finalize_portability_contract(report, primary_score=primary_score)


def _resolve_overfitting_control(automl_config=None):
    policy_profile = _resolve_automl_policy_profile(automl_config)
    legacy_profile = policy_profile == "legacy_permissive"
    automl_config = automl_config or {}
    control = copy.deepcopy(automl_config.get("overfitting_control", {}))
    dsr_config = dict(control.get("deflated_sharpe", {}))
    pbo_config = dict(control.get("pbo", {}))
    post_selection_config = dict(control.get("post_selection", {}))

    return {
        "enabled": bool(control.get("enabled", True)),
        "policy_profile": policy_profile,
        "deprecation_warning": "legacy_permissive_policy_profile_deprecated" if legacy_profile else None,
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
            "require_pass": bool(post_selection_config.get("require_pass", not legacy_profile)),
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


def _resolve_automl_policy_profile(automl_config=None):
    profile = str((automl_config or {}).get("policy_profile", "hardened_default")).strip().lower()
    if profile == "legacy_permissive":
        return "legacy_permissive"
    return "hardened_default"


_HARDENED_DEFAULT_SELECTION_GATE_MODES = {
    "locked_holdout": "blocking",
    "locked_holdout_gap": "blocking",
    "replication": "blocking",
    "execution_realism": "blocking",
    "stress_realism": "blocking",
    "data_certification": "blocking",
    "param_fragility": "blocking",
    "lookahead_guard": "blocking",
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


def _build_evaluation_record(training, backtest, objective_name, automl_config, split=None, *, evidence_class=None):
    training_summary = _json_ready(_summarize_training(training))
    backtest_summary = _json_ready(_summarize_backtest(backtest, evidence_class=evidence_class))
    returns = _extract_backtest_returns(backtest)
    period_sharpe = _compute_period_sharpe(returns)
    objective_diagnostics = _build_objective_diagnostics(
        objective_name,
        training_summary,
        backtest_summary,
        automl_config,
    )
    raw_objective_value = _coerce_float(objective_diagnostics.get("raw_score"))
    if raw_objective_value is None:
        raw_objective_value = float(objective_diagnostics["final_score"])
    gated_objective_value = float(objective_diagnostics["final_score"])
    record = {
        "training": training_summary,
        "backtest": backtest_summary,
        "returns": returns,
        "period_sharpe": period_sharpe,
        "raw_objective_value": float(raw_objective_value),
        "gated_objective_value": float(gated_objective_value),
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


def _resolve_selection_policy(automl_config=None, evaluation_mode=None):
    policy_profile = _resolve_automl_policy_profile(automl_config)
    legacy_profile = policy_profile == "legacy_permissive"
    policy = copy.deepcopy((automl_config or {}).get("selection_policy") or {})
    is_capital_facing = bool(getattr(evaluation_mode, "is_capital_facing", False))
    enabled = bool(policy.get("enabled", True))
    if not enabled:
        return {
            "enabled": False,
            "policy_profile": policy_profile,
            "deprecation_warning": "legacy_permissive_policy_profile_deprecated" if legacy_profile else None,
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
    default_gate_modes = {}
    if is_capital_facing and not legacy_profile:
        default_gate_modes = copy.deepcopy(_HARDENED_DEFAULT_SELECTION_GATE_MODES)
    gate_modes = default_gate_modes
    gate_modes.update(copy.deepcopy(policy.get("gate_modes") or {}))
    return {
        "enabled": True,
        "policy_profile": policy_profile,
        "is_capital_facing": is_capital_facing,
        "deprecation_warning": "legacy_permissive_policy_profile_deprecated" if legacy_profile else None,
        "calibration_mode": bool(policy.get("calibration_mode", False)),
        "gate_modes": gate_modes,
        "max_generalization_gap": float(policy.get("max_generalization_gap", 0.35)),
        "max_param_fragility": float(policy.get("max_param_fragility", 0.30)),
        "max_complexity_score": float(policy.get("max_complexity_score", 18.0)),
        "min_validation_trade_count": int(policy.get("min_validation_trade_count", 10)),
        "min_locked_holdout_score": float(policy.get("min_locked_holdout_score", 0.0)),
        "max_feature_count_ratio": float(policy.get("max_feature_count_ratio", 1.0)),
        "max_trials_per_model_family": int(policy.get("max_trials_per_model_family", 64)),
        "local_perturbation_limit": int(policy.get("local_perturbation_limit", 8)),
        "require_locked_holdout_pass": bool(
            policy.get("require_locked_holdout_pass", bool(is_capital_facing and not legacy_profile))
        ),
        "require_fold_stability_pass": bool(policy.get("require_fold_stability_pass", bool(is_capital_facing))),
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


def _resolve_evidence_gate(gate_name, report, *, selection_policy=None, failure_reason=None, missing_reason=None):
    details = dict(report or {})
    failure_reason = failure_reason or f"{gate_name}_failed"
    missing_reason = missing_reason or f"{gate_name}_evidence_missing"
    policy_profile = str((selection_policy or {}).get("policy_profile") or "").strip().lower()
    is_capital_facing = bool((selection_policy or {}).get("is_capital_facing", False))
    default_mode = "advisory" if policy_profile == "legacy_permissive" else "blocking"
    gate_mode = resolve_promotion_gate_mode(selection_policy, gate_name, default=default_mode)
    if not details:
        passed = gate_mode != "blocking" or not is_capital_facing
        return {
            "details": {},
            "status": "unknown",
            "passed": passed,
            "mode": gate_mode,
            "reason": None if passed else missing_reason,
        }

    status = str(details.get("status") or "").strip().lower()
    if status not in {"passed", "failed", "unknown"}:
        if "promotion_pass" in details:
            status = "passed" if bool(details.get("promotion_pass", False)) else "failed"
        else:
            status = "unknown"

    reasons = list(details.get("reasons") or [])
    canonical_reason_gates = {"feature_portability", "feature_admission", "regime_stability"}
    if status == "unknown" and not reasons:
        reasons = [missing_reason]
    elif status == "failed":
        if not reasons:
            reasons = [failure_reason]
        elif gate_name in canonical_reason_gates and failure_reason not in reasons:
            reasons = [failure_reason] + reasons

    normalized_details = dict(details)
    normalized_details["status"] = status
    normalized_details["reasons"] = reasons
    passed = status == "passed" or (status == "unknown" and (gate_mode != "blocking" or not is_capital_facing))
    return {
        "details": normalized_details,
        "status": status,
        "passed": passed,
        "mode": str(details.get("gate_mode") or gate_mode).lower(),
        "reason": None if passed else (reasons[0] if reasons else failure_reason),
    }


def _resolve_lookahead_guard_gate(training_summary):
    lookahead_guard = dict((training_summary or {}).get("lookahead_guard") or {})
    if not lookahead_guard:
        return {
            "details": {},
            "passed": False,
            "reason": "lookahead_guard_missing",
        }

    status = str(lookahead_guard.get("status") or "").strip().lower()
    if status == "disabled":
        return {
            "details": lookahead_guard,
            "passed": False,
            "reason": "lookahead_guard_disabled",
        }
    if status == "skipped":
        return {
            "details": lookahead_guard,
            "passed": False,
            "reason": "lookahead_guard_skipped",
        }
    if not bool(lookahead_guard.get("enabled", False)):
        return {
            "details": lookahead_guard,
            "passed": False,
            "reason": "lookahead_guard_disabled",
        }

    passed = bool(lookahead_guard.get("promotion_pass", False))
    return {
        "details": lookahead_guard,
        "passed": passed,
        "reason": None if passed else _first_failure_reason(lookahead_guard, "lookahead_guard_failed"),
    }


def _gs(name, passed, measured, threshold, failure_reason, details, mode=None):
    return {
        "name": name,
        "passed": passed,
        "measured": measured,
        "threshold": threshold,
        "reason": None if passed else failure_reason,
        "details": details,
        "mode": mode,
    }


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
                evaluation = _build_evaluation_record(
                    training,
                    backtest,
                    objective_name,
                    automl_config,
                    evidence_class="outer_replay",
                )
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
                    evidence_class="outer_replay",
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
    evaluation_mode = resolve_evaluation_mode((automl_config or {}).get("backtest") or {})
    selection_policy = _resolve_selection_policy(automl_config, evaluation_mode=evaluation_mode)
    validation_contract = _resolve_validation_contract({}, automl_config)
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
        validation_sources = _resolve_validation_sources(
            record.get("training"),
            record.get("backtest"),
            validation_contract,
            holdout_enabled=bool(
                (automl_config or {}).get("locked_holdout_enabled", False)
                or (automl_config or {}).get("locked_holdout_fraction")
            ),
            replication_enabled=bool(((automl_config or {}).get("replication") or {}).get("enabled", False)),
        )

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
        data_certification = dict(training_summary.get("data_certification") or {})
        signal_decay = dict(training_summary.get("signal_decay") or {})
        regime_ablation_summary = dict((training_summary.get("regime") or {}).get("ablation_summary") or {})
        operational_monitoring = dict(training_summary.get("operational_monitoring") or {})
        feature_portability_gate = _resolve_evidence_gate(
            "feature_portability",
            feature_portability_diagnostics,
            selection_policy=selection_policy,
            failure_reason="feature_portability_failed",
        )
        feature_admission_gate = _resolve_evidence_gate(
            "feature_admission",
            feature_admission_summary,
            selection_policy=selection_policy,
            failure_reason="feature_admission_failed",
        )
        regime_stability_gate = _resolve_evidence_gate(
            "regime_stability",
            regime_ablation_summary,
            selection_policy=selection_policy,
            failure_reason="regime_stability_failed",
            missing_reason="regime_ablation_evidence_missing",
        )
        operational_health_pass = bool(promotion_gates.get("operational_health", True))
        cross_venue_integrity_gate = _resolve_evidence_gate(
            "cross_venue_integrity",
            cross_venue_integrity,
            selection_policy=selection_policy,
            failure_reason="cross_venue_integrity_failed",
        )
        data_certification_gate = _resolve_evidence_gate(
            "data_certification",
            data_certification,
            selection_policy=selection_policy,
            failure_reason="data_certification_failed",
        )
        signal_decay_gate = _resolve_evidence_gate(
            "signal_decay",
            signal_decay,
            selection_policy=selection_policy,
            failure_reason="signal_decay_failed",
        )
        feature_portability_pass = feature_portability_gate["passed"]
        feature_admission_pass = feature_admission_gate["passed"]
        regime_stability_pass = regime_stability_gate["passed"]
        cross_venue_integrity_pass = cross_venue_integrity_gate["passed"]
        data_certification_pass = data_certification_gate["passed"]
        signal_decay_pass = signal_decay_gate["passed"]

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
            "data_certification": data_certification_pass,
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
            ec = eligibility_checks
            gate_specs = [
                _gs("minimum_dsr", ec["minimum_dsr"], dsr_value,
                    minimum_dsr_threshold if dsr_gate_applies else None,
                    "deflated_sharpe_below_threshold", deflated_sharpe),
                _gs("objective_constraints", ec["objective_constraints"],
                    objective_gate_passed, True, "objective_constraints_failed", objective_diagnostics),
                _gs("validation_trade_count", ec["validation_trade_count"],
                    validation_trade_count, int(selection_policy.get("min_validation_trade_count", 0)),
                    "validation_trade_count_below_minimum", {"validation_trade_count": validation_trade_count}),
                _gs("complexity", ec["complexity"],
                    complexity["trial_complexity_score"], selection_policy.get("max_complexity_score", np.inf),
                    "complexity_score_above_limit", complexity),
                _gs("feature_count_ratio", ec["feature_count_ratio"],
                    complexity.get("feature_count_ratio"), selection_policy.get("max_feature_count_ratio", np.inf),
                    "feature_count_ratio_above_limit", complexity),
                _gs("generalization_gap", ec["generalization_gap"],
                    search_gap.get("normalized_degradation") or 0.0, selection_policy.get("max_generalization_gap", np.inf),
                    "search_validation_gap_above_limit", search_gap),
                _gs("model_family_trial_count", ec["model_family_trial_count"],
                    model_family_counts.get(model_family, 0), selection_policy.get("max_trials_per_model_family", np.inf),
                    "model_family_trial_count_above_limit", {"model_family": model_family, "trial_count": model_family_counts.get(model_family, 0)}),
                _gs("feature_portability", ec["feature_portability"],
                    feature_portability_diagnostics.get("venue_specific_importance_share"), feature_portability_diagnostics.get("config"),
                    feature_portability_gate["reason"], feature_portability_gate["details"], mode=feature_portability_gate["mode"]),
                _gs("feature_admission", ec["feature_admission"],
                    feature_admission_summary.get("promotion_pass"), True,
                    feature_admission_gate["reason"], feature_admission_gate["details"], mode=feature_admission_gate["mode"]),
                _gs("regime_stability", ec["regime_stability"],
                    regime_ablation_summary.get("stability_improvement"), True,
                    regime_stability_gate["reason"], regime_stability_gate["details"], mode=regime_stability_gate["mode"]),
                _gs("operational_health", ec["operational_health"],
                    operational_monitoring.get("healthy"), True,
                    _first_failure_reason(operational_monitoring, "operational_monitoring_failed"), operational_monitoring),
                _gs("cross_venue_integrity", ec["cross_venue_integrity"],
                    cross_venue_integrity.get("promotion_pass"), True,
                    cross_venue_integrity_gate["reason"], cross_venue_integrity_gate["details"], mode=cross_venue_integrity_gate["mode"]),
                _gs("data_certification", ec["data_certification"],
                    data_certification.get("promotion_pass"), True,
                    data_certification_gate["reason"], data_certification_gate["details"], mode=data_certification_gate["mode"]),
                _gs("signal_decay", ec["signal_decay"],
                    signal_decay.get("net_edge_at_effective_delay"), signal_decay.get("policy"),
                    signal_decay_gate["reason"], signal_decay_gate["details"], mode=signal_decay_gate["mode"]),
                _gs("fold_stability", ec["fold_stability"],
                    fold_stability_gate["summary"].get("persistence"), True,
                    "fold_stability_failed", fold_stability_gate["summary"]),
            ]
            for gate in gate_specs:
                promotion_eligibility_report = upsert_promotion_gate(
                    promotion_eligibility_report,
                    group="selection",
                    name=gate["name"],
                    passed=gate["passed"],
                    mode=gate.get("mode") or resolve_promotion_gate_mode(selection_policy, gate["name"]),
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
                "validation_sources": validation_sources,
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
        "validation_contract": validation_contract,
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


def _build_top_trial_reports(selection_report):
    top_trials = []
    for report in list((selection_report or {}).get("trial_reports") or [])[:5]:
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
    return top_trials


def _build_selection_outcome(trial_reports, *, best_trial_report=None, selection_snapshot=None, status=None, error=None):
    reports = list(trial_reports or [])
    eligible_trial_count = int(
        sum(1 for report in reports if (report.get("selection_policy") or {}).get("eligible"))
    )
    rejection_counts = {}
    for report in reports:
        for reason in list((report.get("selection_policy") or {}).get("eligibility_reasons") or []):
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    top_rejection_reasons = [
        reason
        for reason, _ in sorted(rejection_counts.items(), key=lambda item: (-item[1], item[0]))
    ]

    if status is None:
        status = "selected" if best_trial_report is not None else "abstain_no_eligible_trial"

    return {
        "status": status,
        "eligible_trial_count": eligible_trial_count,
        "completed_trial_count": int(len(reports)),
        "rejected_trial_count": int(len(reports) - eligible_trial_count),
        "top_rejection_reasons": top_rejection_reasons,
        "selection_freeze": selection_snapshot,
        "promotion_ready": bool(
            best_trial_report is not None
            and bool((best_trial_report.get("selection_policy") or {}).get("promotion_ready", False))
        ),
        "selected_trial_number": None if best_trial_report is None else int(best_trial_report["number"]),
        "candidate_hash": None if selection_snapshot is None else selection_snapshot.get("candidate_hash"),
        "holdout_consulted_for_selection": bool(
            best_trial_report is not None
            and bool((best_trial_report.get("selection_policy") or {}).get("holdout_consulted_for_selection", False))
        ),
        "evaluated_after_freeze": bool(
            best_trial_report is not None
            and bool(((best_trial_report.get("locked_holdout") or {}).get("evaluated_after_freeze", False)))
        ),
        "error": error,
    }


def _cleanup_optuna_study_resources(study):
    storage_backend = getattr(study, "_storage", None)
    if storage_backend is not None and hasattr(storage_backend, "remove_session"):
        try:
            storage_backend.remove_session()
        except Exception:
            pass

    engine = getattr(storage_backend, "engine", None)
    if engine is not None:
        try:
            engine.dispose()
        except Exception:
            pass


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
            "min_effective_bet_count": None,
            "require_statistical_significance": False,
            "min_significance_observations": None,
            "min_sharpe_ci_lower": None,
            "min_net_profit_pct_ci_lower": None,
        }
    return {
        "enabled": True,
        "min_directional_accuracy": _coerce_float(gate_config.get("min_directional_accuracy", 0.52)),
        "max_log_loss": _coerce_float(gate_config.get("max_log_loss", 0.78)),
        "max_calibration_error": _coerce_float(gate_config.get("max_calibration_error", 0.15)),
        "min_trade_count": int(gate_config.get("min_trade_count", 30)),
        "min_effective_bet_count": _coerce_float(gate_config.get("min_effective_bet_count")),
        "require_statistical_significance": bool(gate_config.get("require_statistical_significance", False)),
        "min_significance_observations": _coerce_float(gate_config.get("min_significance_observations")),
        "min_sharpe_ci_lower": _coerce_float(gate_config.get("min_sharpe_ci_lower")),
        "min_net_profit_pct_ci_lower": _coerce_float(gate_config.get("min_net_profit_pct_ci_lower")),
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
        "reasons": [],
        "checks": {},
    }
    if not report["enabled"]:
        return report

    directional_accuracy = _resolve_metric(training, "avg_directional_accuracy", fallback="avg_accuracy")
    log_loss_value = _resolve_metric(training, "avg_log_loss")
    calibration_error = _resolve_metric(training, "avg_calibration_error")
    trade_count = _coerce_float((backtest or {}).get("total_trades"))
    effective_bet_count = _coerce_float(
        (backtest or {}).get("effective_bet_count")
        or ((backtest or {}).get("statistical_significance") or {}).get("effective_bet_count")
    )
    significance = (backtest or {}).get("statistical_significance") or {}
    significance_enabled = bool(significance.get("enabled", False))
    significance_reason = str(
        significance.get("underpowered_reason") or significance.get("reason") or ""
    ).strip().lower() or None
    significance_observation_count = _coerce_float(significance.get("observation_count"))
    sharpe_ci_lower = _coerce_float(
        ((
            (_get_significance_payload(backtest, "sharpe_ratio") or {}).get("confidence_interval")
            or {}
        ).get("lower"))
    )
    net_profit_pct_ci_lower = _coerce_float(
        ((
            (_get_significance_payload(backtest, "net_profit_pct") or {}).get("confidence_interval")
            or {}
        ).get("lower"))
    )

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
    if gates.get("min_effective_bet_count") is not None:
        required_effective_bets = float(gates.get("min_effective_bet_count"))
        effective_pass = effective_bet_count is not None and effective_bet_count >= required_effective_bets
        checks["effective_bet_count"] = {
            "name": "effective_bet_count",
            "value": effective_bet_count,
            "minimum": required_effective_bets,
            "maximum": None,
            "passed": bool(effective_pass),
            "reason": None if effective_pass else "effective_bet_count_underpowered",
        }
    if gates.get("require_statistical_significance"):
        checks["statistical_significance"] = {
            "name": "statistical_significance",
            "value": significance_enabled,
            "minimum": True,
            "maximum": None,
            "passed": bool(significance_enabled),
            "reason": None if significance_enabled else f"statistical_significance_{significance_reason or 'unavailable'}",
        }
    if gates.get("min_significance_observations") is not None:
        required_observations = float(gates.get("min_significance_observations"))
        observations_pass = (
            significance_observation_count is not None
            and significance_observation_count >= required_observations
        )
        checks["significance_observation_count"] = {
            "name": "significance_observation_count",
            "value": significance_observation_count,
            "minimum": required_observations,
            "maximum": None,
            "passed": bool(observations_pass),
            "reason": None if observations_pass else "statistical_significance_underpowered",
        }
    if gates.get("min_sharpe_ci_lower") is not None:
        checks["sharpe_ci_lower"] = _build_gate_result(
            "sharpe_ci_lower",
            sharpe_ci_lower,
            minimum=gates.get("min_sharpe_ci_lower"),
        )
    if gates.get("min_net_profit_pct_ci_lower") is not None:
        checks["net_profit_pct_ci_lower"] = _build_gate_result(
            "net_profit_pct_ci_lower",
            net_profit_pct_ci_lower,
            minimum=gates.get("min_net_profit_pct_ci_lower"),
        )
    report["checks"] = checks
    report["failed"] = [name for name, payload in checks.items() if not payload["passed"]]
    report["reasons"] = [
        payload.get("reason") or name
        for name, payload in checks.items()
        if not payload["passed"]
    ]
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
        if "pt_mult" in label_space and "sl_mult" in label_space:
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


def _count_completed_trials_for_model_family(study, model_family):
    if not model_family:
        return 0
    count = 0
    for existing_trial in list(getattr(study, "trials", []) or []):
        if getattr(existing_trial, "state", None) != optuna.trial.TrialState.COMPLETE:
            continue
        overrides = dict((getattr(existing_trial, "user_attrs", None) or {}).get("overrides") or {})
        existing_family = str(((overrides.get("model") or {}).get("type") or "")).strip().lower()
        if existing_family == str(model_family).strip().lower():
            count += 1
    return count


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
    _validate_forbidden_search_space_paths(search_space)
    _validate_signal_policy_search_space(search_space)
    _validate_trade_ready_search_space(search_space, automl_config)
    tiers = _classify_search_space(search_space)
    varying_thesis_paths = _find_varying_thesis_paths(tiers)
    requested_evaluation_mode = (
        automl_config.get("evaluation_mode")
        or base_config.get("evaluation_mode")
        or dict(base_config.get("backtest") or {}).get("evaluation_mode")
        or "research_only"
    )
    resolved_mode = resolve_evaluation_mode(
        {
            "evaluation_mode": requested_evaluation_mode,
        }
    )
    automl_config.setdefault("backtest", {})
    automl_config["backtest"]["evaluation_mode"] = resolved_mode.requested_mode
    if varying_thesis_paths:
        joined = ", ".join(varying_thesis_paths)
        if resolved_mode.requested_mode == "trade_ready":
            raise ValueError(
                "Capital-facing/certification AutoML cannot vary thesis parameters. Freeze entries: "
                f"{joined}"
            )
        warnings.warn(
            f"{resolved_mode.requested_mode} AutoML is varying thesis parameters ({joined}); this weakens locked-holdout guarantees.",
            UserWarning,
        )

    full_state_bundle = _build_state_bundle(base_pipeline)
    experiment_manifest = _build_experiment_manifest(base_config, automl_config, full_state_bundle, search_space)
    storage_context = _build_experiment_storage_context(base_config, automl_config, experiment_manifest)
    resume_validation = _validate_resume_manifest(storage_context, experiment_manifest)
    storage_path = Path(storage_context["study_path"])
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_path.as_posix()}"
    study_name = _resolve_study_name(base_config, automl_config)
    tpe_startup_trials = int(automl_config.get("tpe_startup_trials", 25))
    sampler = TPESampler(seed=automl_config.get("seed", 42), n_startup_trials=tpe_startup_trials)
    objective_name = _normalize_objective_name(automl_config.get("objective", "risk_adjusted_after_costs"))

    experiment_dir = Path(storage_context["experiment_dir"])
    experiment_dir.mkdir(parents=True, exist_ok=True)
    write_json(storage_context["manifest_path"], experiment_manifest)
    write_json(
        storage_context["lineage_path"],
        {
            "experiment_id": experiment_manifest["experiment_id"],
            "experiment_family_id": experiment_manifest["experiment_family_id"],
            "study_label": study_name,
            "resume_mode": experiment_manifest["resume_mode"],
            "data_lineage_hash": experiment_manifest["data_lineage_hash"],
            "feature_schema_hash": experiment_manifest["feature_schema_hash"],
            "objective_hash": experiment_manifest["objective_hash"],
            "search_space_hash": experiment_manifest["search_space_hash"],
            "code_revision": experiment_manifest["code_revision"],
            "data_lineage": experiment_manifest.get("data_contract", {}).get("data_lineage") or {},
        },
    )

    holdout_plan = _resolve_holdout_plan(full_state_bundle["raw_data"], automl_config, base_config=base_config)
    search_state_bundle = full_state_bundle
    validation_state_bundle = full_state_bundle
    if holdout_plan["enabled"]:
        search_state_bundle = _build_window_state_bundle(full_state_bundle, end_timestamp=holdout_plan["search_end_timestamp"])
        validation_state_bundle = _build_window_state_bundle(full_state_bundle, end_timestamp=holdout_plan["validation_end_timestamp"])

    capital_evidence_contract = _validate_capital_evidence_contract(
        base_config,
        holdout_plan=holdout_plan,
        base_pipeline=base_pipeline,
    )
    _validate_oos_evidence_preconditions(
        base_config,
        holdout_plan=holdout_plan,
        base_pipeline=base_pipeline,
    )

    enable_pruning = bool(automl_config.get("enable_pruning", True))
    study_kwargs = {
        "direction": "maximize",
        "sampler": sampler,
        "study_name": study_name,
        "storage": storage_url,
        "load_if_exists": bool(resume_validation["load_if_exists"]),
    }
    if enable_pruning:
        study_kwargs["pruner"] = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    study = optuna.create_study(**study_kwargs)
    if hasattr(study, "set_user_attr"):
        study.set_user_attr("experiment_id", experiment_manifest["experiment_id"])
        study.set_user_attr("experiment_family_id", experiment_manifest["experiment_family_id"])
        study.set_user_attr("experiment_manifest", experiment_manifest)
        study.set_user_attr("resume_mode", experiment_manifest["resume_mode"])
        study.set_user_attr("data_lineage_hash", experiment_manifest["data_lineage_hash"])
        study.set_user_attr("feature_schema_hash", experiment_manifest["feature_schema_hash"])
        study.set_user_attr("objective_hash", experiment_manifest["objective_hash"])
        study.set_user_attr("search_space_hash", experiment_manifest["search_space_hash"])
        study.set_user_attr("code_revision", experiment_manifest["code_revision"])
    trial_records = {}
    selection_policy = _resolve_selection_policy(automl_config, evaluation_mode=resolved_mode)

    def objective(trial):
        overrides = _sample_trial_overrides(trial, search_space)
        _validate_trial_overrides(overrides)
        model_family = str(((overrides.get("model") or {}).get("type") or "")).strip().lower()
        max_trials_per_family = selection_policy.get("max_trials_per_model_family", np.inf)
        if model_family and np.isfinite(max_trials_per_family):
            completed_for_family = _count_completed_trials_for_model_family(study, model_family)
            trial.set_user_attr(
                "model_family_precheck",
                {
                    "model_family": model_family,
                    "completed_trials": int(completed_for_family),
                    "max_trials_per_family": int(max_trials_per_family),
                    "passed": bool(completed_for_family < int(max_trials_per_family)),
                },
            )
            if completed_for_family >= int(max_trials_per_family):
                raise optuna.TrialPruned(f"model_family_trial_cap_reached:{model_family}")
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
                evidence_class="search_cv_diagnostic",
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
        trial.set_user_attr("experiment_id", experiment_manifest["experiment_id"])
        trial.set_user_attr("experiment_family_id", experiment_manifest["experiment_family_id"])
        trial.set_user_attr("experiment_manifest", experiment_manifest)
        trial.set_user_attr("resume_mode", experiment_manifest["resume_mode"])
        trial.set_user_attr("data_lineage_hash", experiment_manifest["data_lineage_hash"])
        trial.set_user_attr("feature_schema_hash", experiment_manifest["feature_schema_hash"])
        trial.set_user_attr("objective_hash", experiment_manifest["objective_hash"])
        trial.set_user_attr("search_space_hash", experiment_manifest["search_space_hash"])
        trial.set_user_attr("code_revision", experiment_manifest["code_revision"])
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
                evidence_class="outer_replay",
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

    n_trials = int(automl_config.get("n_trials", 25))
    if n_trials < tpe_startup_trials:
        warnings.warn(
            f"n_trials={n_trials} is below tpe_startup_trials={tpe_startup_trials}; study will remain random-sampling dominated",
            UserWarning,
        )

    study.optimize(
        objective,
        n_trials=n_trials,
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
                evidence_class="search_cv_diagnostic",
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
                    evidence_class="outer_replay",
                )
        except RuntimeError:
            continue
        trial_records[trial.number] = _build_trial_record(overrides, search_record, validation_record)

    selection_report = _build_trial_selection_report(completed_trials, trial_records, objective_name, automl_config)
    selection_policy = _resolve_selection_policy(automl_config, evaluation_mode=resolved_mode)
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

    best_optuna_trial = study.best_trial
    top_trials = _build_top_trial_reports(selection_report)

    if best_trial_report is None:
        validation_contract = _resolve_validation_contract(
            base_config,
            automl_config,
            holdout_enabled=bool(holdout_plan.get("enabled", False)),
        )
        selection_outcome = _build_selection_outcome(selection_report.get("trial_reports"))
        abstention_reasons = list(selection_outcome.get("top_rejection_reasons") or ["no_eligible_trial_under_selection_policy"])
        validation_sources = _resolve_validation_sources(
            {},
            {},
            validation_contract,
            holdout_enabled=bool(holdout_plan.get("enabled", False)),
            replication_enabled=bool((automl_config.get("replication") or {}).get("enabled", False)),
        )
        selection_report["diagnostics"]["promoted_trial"] = None
        selection_report["diagnostics"]["selection_freeze"] = None
        selection_report["diagnostics"]["holdout_access_count"] = 0
        selection_report["diagnostics"]["holdout_evaluated_once"] = False
        selection_report["diagnostics"]["holdout_evaluated_after_freeze"] = False
        selection_report["diagnostics"]["promotion_ready"] = False
        selection_report["diagnostics"]["promotion_reasons"] = abstention_reasons
        selection_report["diagnostics"]["validation_contract"] = validation_contract
        selection_report["diagnostics"]["validation_sources"] = validation_sources

        data_lineage = _json_ready(base_pipeline.state.get("data_lineage") or {})
        policy_profile = _resolve_automl_policy_profile(base_config.get("automl") or {})
        summary_warnings = []
        if policy_profile == "legacy_permissive":
            summary_warnings.append("legacy_permissive_policy_profile_deprecated")

        promotion_eligibility_report = create_promotion_eligibility_report()
        promotion_eligibility_report["blocking_failures"] = list(abstention_reasons)
        oos_evidence = _build_oos_evidence(
            base_config,
            holdout_plan=holdout_plan,
            selection_diagnostics=selection_report["diagnostics"],
            training=None,
            base_pipeline=base_pipeline,
        )
        summary = {
            "study_name": study.study_name,
            "storage": str(storage_path),
            "experiment_id": experiment_manifest["experiment_id"],
            "experiment_family_id": experiment_manifest["experiment_family_id"],
            "experiment_manifest": experiment_manifest,
            "resume_mode": experiment_manifest["resume_mode"],
            "resume_validation": resume_validation,
            "experiment_artifacts": {
                "storage_root": str(storage_context["storage_root"]),
                "experiment_dir": str(storage_context["experiment_dir"]),
                "run_dir": str(storage_context["run_dir"]),
                "study": str(storage_context["study_path"]),
                "manifest": str(storage_context["manifest_path"]),
                "lineage": str(storage_context["lineage_path"]),
                "summary": str(storage_context["summary_path"]),
                "run_id": storage_context.get("run_id"),
            },
            "objective": objective_name,
            "policy_profile": policy_profile,
            "trade_ready_profile": _json_ready((base_config.get("automl") or {}).get("trade_ready_profile") or {}),
            "capital_evidence_contract": _json_ready(copy.deepcopy(capital_evidence_contract)),
            "oos_evidence": _json_ready(oos_evidence),
            "warnings": summary_warnings,
            "selection_metric": selection_report["selection_metric"],
            "selection_mode": selection_report["selection_mode"],
            "selection_outcome": selection_outcome,
            "validation_contract": validation_contract,
            "validation_sources": validation_sources,
            "feature_schema_version": base_config.get("features", {}).get("schema_version"),
            "best_value": None,
            "best_value_raw": None,
            "best_value_penalized": None,
            "optuna_best_value": float(best_optuna_trial.value),
            "best_trial_number": None,
            "optuna_best_trial_number": int(best_optuna_trial.number),
            "best_params": None,
            "best_overrides": None,
            "best_backtest": {},
            "best_training": {},
            "best_overfitting": {},
            "best_objective_diagnostics": None,
            "best_selection_policy": {},
            "selection_freeze": None,
            "validation_holdout": {
                "enabled": bool(holdout_plan.get("enabled", False)),
                "evaluated": False,
                "reason": "selection_abstained_no_eligible_trial",
            },
            "locked_holdout": {
                "enabled": bool(holdout_plan.get("enabled", False)),
                "evaluated_once": False,
                "evaluated_after_freeze": False,
                "reason": "selection_abstained_no_eligible_trial",
            },
            "replication": {
                "enabled": False,
                "reason": "selection_abstained_no_eligible_trial",
            },
            "promotion_ready": False,
            "promotion_reasons": abstention_reasons,
            "promotion_eligibility_report": promotion_eligibility_report,
            "overfitting_diagnostics": selection_report["diagnostics"],
            "data_lineage": data_lineage,
            "top_trials": top_trials,
            "trial_count": len(completed_trials),
        }

        summary = validate_summary_contract(summary)

        write_json(
            storage_context["summary_path"],
            summary,
        )

        _cleanup_optuna_study_resources(study)
        return summary

    best_trial_number = int(best_trial_report["number"])
    best_overrides = copy.deepcopy(best_trial_report["overrides"])
    best_overrides_summary = _json_ready(_clone_value(best_overrides))
    best_trial_report["selection_policy"]["frozen"] = True
    selection_snapshot = _build_selection_snapshot(best_trial_report)
    validation_contract = _resolve_validation_contract(
        base_config,
        automl_config,
        training=best_trial_report.get("training"),
        holdout_enabled=bool(holdout_plan.get("enabled", False)),
    )
    validation_sources = _resolve_validation_sources(
        best_trial_report.get("training"),
        best_trial_report.get("backtest"),
        validation_contract,
        holdout_enabled=bool(holdout_plan.get("enabled", False)),
        replication_enabled=bool((automl_config.get("replication") or {}).get("enabled", False)),
    )

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
        primary_score=_coerce_float(best_trial_report.get("raw_objective_value")),
    )
    replication_report["gate_mode"] = resolve_promotion_gate_mode(selection_policy, "replication")
    replication_report["evidence_class"] = "replication"
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
            passed=bool(replication_report.get("promotion_pass", False)),
            mode=replication_report.get("gate_mode"),
            measured=replication_report.get("pass_rate"),
            threshold={
                "min_coverage": replication_report.get("min_coverage"),
                "min_pass_rate": replication_report.get("min_pass_rate"),
                "min_score": replication_report.get("min_score"),
            },
            reason=None if replication_report.get("promotion_pass", False) else _first_failure_reason(replication_report, "replication_failed"),
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
    stress_realism = evaluate_stress_realism_gate(
        (locked_holdout or {}).get("backtest")
        or validation_holdout.get("backtest")
        or best_trial_report.get("backtest")
        or {},
        policy=selection_policy,
    )
    best_trial_report["selection_policy"]["eligibility_checks"]["stress_realism"] = bool(stress_realism["passed"])
    promotion_eligibility_report = upsert_promotion_gate(
        promotion_eligibility_report,
        group="post_selection",
        name="stress_realism",
        passed=bool(stress_realism["passed"]),
        mode=resolve_promotion_gate_mode(selection_policy, "stress_realism"),
        measured=stress_realism.get("configured_scenarios"),
        threshold=stress_realism.get("required_scenarios"),
        reason=stress_realism.get("reason"),
        details=stress_realism,
    )
    lookahead_guard_gate = _resolve_lookahead_guard_gate(best_trial_report.get("training") or {})
    lookahead_guard = dict(lookahead_guard_gate["details"] or {})
    best_trial_report["selection_policy"]["eligibility_checks"]["lookahead_guard"] = bool(
        lookahead_guard_gate["passed"]
    )
    promotion_eligibility_report = upsert_promotion_gate(
        promotion_eligibility_report,
        group="post_selection",
        name="lookahead_guard",
        passed=bool(lookahead_guard_gate["passed"]),
        mode=resolve_promotion_gate_mode(selection_policy, "lookahead_guard"),
        measured=lookahead_guard.get("checked_timestamps"),
        threshold={
            "mode": lookahead_guard.get("mode"),
            "min_prefix_rows": lookahead_guard.get("min_prefix_rows"),
            "decision_sample_size": lookahead_guard.get("decision_sample_size"),
        },
        reason=lookahead_guard_gate["reason"],
        details=lookahead_guard,
    )
    best_trial_report["selection_policy"] = _update_selection_policy_report(
        best_trial_report["selection_policy"],
        promotion_eligibility_report,
        include_post_selection=True,
    )
    blocking_failures = list(
        (
            best_trial_report["selection_policy"].get("promotion_eligibility_report") or {}
        ).get("blocking_failures") or []
    )
    if blocking_failures:
        best_trial_report["selection_policy"]["promotion_ready"] = False
        best_trial_report["selection_policy"]["promotion_reasons"] = blocking_failures
    best_trial_report["selection_policy"]["holdout_consulted_for_selection"] = False
    selection_report["diagnostics"]["holdout_access_count"] = int(locked_holdout_access_count)
    selection_report["diagnostics"]["holdout_evaluated_once"] = bool(locked_holdout.get("evaluated_once", False))
    selection_report["diagnostics"]["holdout_evaluated_after_freeze"] = bool(
        locked_holdout.get("evaluated_after_freeze", False)
    )
    selection_report["diagnostics"]["execution_realism"] = execution_realism
    selection_report["diagnostics"]["stress_realism"] = stress_realism
    selection_report["diagnostics"]["replication"] = replication_report
    selection_report["diagnostics"]["promotion_ready"] = bool(best_trial_report["selection_policy"]["promotion_ready"])
    selection_report["diagnostics"]["promotion_reasons"] = list(best_trial_report["selection_policy"]["promotion_reasons"])
    selection_report["diagnostics"]["validation_contract"] = validation_contract
    selection_report["diagnostics"]["validation_sources"] = validation_sources

    data_lineage = _json_ready(base_pipeline.state.get("data_lineage") or {})
    policy_profile = _resolve_automl_policy_profile(base_config.get("automl") or {})
    summary_warnings = []
    if policy_profile == "legacy_permissive":
        summary_warnings.append("legacy_permissive_policy_profile_deprecated")

    summary = {
        "study_name": study.study_name,
        "storage": str(storage_path),
        "experiment_id": experiment_manifest["experiment_id"],
        "experiment_family_id": experiment_manifest["experiment_family_id"],
        "experiment_manifest": experiment_manifest,
        "resume_mode": experiment_manifest["resume_mode"],
        "resume_validation": resume_validation,
        "experiment_artifacts": {
            "storage_root": str(storage_context["storage_root"]),
            "experiment_dir": str(storage_context["experiment_dir"]),
            "run_dir": str(storage_context["run_dir"]),
            "study": str(storage_context["study_path"]),
            "manifest": str(storage_context["manifest_path"]),
            "lineage": str(storage_context["lineage_path"]),
            "summary": str(storage_context["summary_path"]),
            "run_id": storage_context.get("run_id"),
        },
        "objective": objective_name,
        "policy_profile": policy_profile,
        "trade_ready_profile": _json_ready((base_config.get("automl") or {}).get("trade_ready_profile") or {}),
        "capital_evidence_contract": _json_ready(copy.deepcopy(capital_evidence_contract)),
        "oos_evidence": _json_ready(
            _build_oos_evidence(
                base_config,
                holdout_plan=holdout_plan,
                selection_diagnostics=selection_report["diagnostics"],
                training=best_trial_report.get("training"),
                base_pipeline=base_pipeline,
            )
        ),
        "warnings": summary_warnings,
        "selection_metric": selection_report["selection_metric"],
        "selection_mode": selection_report["selection_mode"],
        "selection_outcome": _build_selection_outcome(
            selection_report.get("trial_reports"),
            best_trial_report=best_trial_report,
            selection_snapshot=selection_snapshot,
        ),
        "validation_contract": validation_contract,
        "validation_sources": validation_sources,
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
            "validation_sources": validation_sources,
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

    if not bool((summary.get("oos_evidence") or {}).get("evidence_stack_complete", False)):
        promotion_reasons = list(summary.get("promotion_reasons") or [])
        if "oos_evidence_incomplete" not in promotion_reasons:
            promotion_reasons.append("oos_evidence_incomplete")
        summary["promotion_ready"] = False
        summary["promotion_reasons"] = promotion_reasons
        promotion_report = dict(summary.get("promotion_eligibility_report") or {})
        blocking_failures = list(promotion_report.get("blocking_failures") or [])
        if "oos_evidence_incomplete" not in blocking_failures:
            blocking_failures.append("oos_evidence_incomplete")
        promotion_report["blocking_failures"] = blocking_failures
        promotion_report["promotion_ready"] = False
        promotion_report["approved"] = False
        summary["promotion_eligibility_report"] = promotion_report
        capital_contract = dict(summary.get("capital_evidence_contract") or {})
        blocking_reasons = list(capital_contract.get("blocking_reasons") or [])
        for reason in list((summary.get("oos_evidence") or {}).get("blocking_reasons") or []):
            if reason not in blocking_reasons:
                blocking_reasons.append(reason)
        capital_contract["blocking_reasons"] = blocking_reasons
        capital_contract["capital_path_eligible"] = False
        summary["capital_evidence_contract"] = capital_contract
        best_selection_policy = dict(summary.get("best_selection_policy") or {})
        selection_policy_report = dict(best_selection_policy.get("selection_policy") or {})
        if selection_policy_report:
            selection_policy_report["promotion_ready"] = False
            policy_reasons = list(selection_policy_report.get("promotion_reasons") or [])
            if "oos_evidence_incomplete" not in policy_reasons:
                policy_reasons.append("oos_evidence_incomplete")
            selection_policy_report["promotion_reasons"] = policy_reasons
            best_selection_policy["selection_policy"] = selection_policy_report
            summary["best_selection_policy"] = best_selection_policy

    summary = validate_summary_contract(summary)

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

    summary = validate_summary_contract(summary)
    write_json(storage_context["summary_path"], summary)

    _cleanup_optuna_study_resources(study)

    return summary