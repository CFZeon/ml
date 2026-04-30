"""Shared helpers for the repository example scripts."""

import argparse
import copy
from math import isfinite

import pandas as pd


def _copy_value(value):
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return value.copy()
    return copy.deepcopy(value)


def _deep_update_dict(base, overrides):
    for key, value in dict(overrides or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update_dict(base[key], value)
        else:
            base[key] = _copy_value(value)
    return base


def clone_config_with_overrides(base_config, config_overrides=None):
    """Clone a nested example config and apply dict overrides recursively."""

    cloned = _copy_value(base_config)
    if config_overrides:
        _deep_update_dict(cloned, config_overrides)
    return cloned


def build_custom_data_entry(
    name,
    frame,
    *,
    value_columns,
    timestamp_column="timestamp",
    availability_column="available_at",
    allow_exact_matches=None,
    prefix=None,
    max_feature_age=None,
    extra_fields=None,
):
    """Build a point-in-time-safe custom-data config entry for examples."""

    entry = {
        "name": str(name),
        "frame": _copy_value(frame),
        "timestamp_column": str(timestamp_column),
        "availability_column": str(availability_column),
        "value_columns": list(value_columns),
    }
    if allow_exact_matches is not None:
        entry["allow_exact_matches"] = bool(allow_exact_matches)
    if prefix is not None:
        entry["prefix"] = str(prefix)
    if max_feature_age is not None:
        entry["max_feature_age"] = max_feature_age
    return clone_config_with_overrides(entry, extra_fields)


def build_spot_research_config(
    symbol,
    interval,
    start,
    end,
    *,
    indicators,
    context_symbols=None,
    custom_data=None,
    config_overrides=None,
):
    """Build the baseline spot-research example config.

    Copy this into a new script when you want a real-data spot case and then
    change only the nested sections you actually need.
    """

    context_symbols = list(context_symbols or [])
    config = {
        "data": {
            "symbol": str(symbol),
            "interval": str(interval),
            "start": start,
            "end": end,
            "market": "spot",
            "duplicate_policy": "fail",
            "futures_context": {
                "enabled": True,
                "include_recent_stats": True,
                "recent_stats_availability_lag": "period_close",
            },
            "cross_asset_context": {"symbols": context_symbols},
        },
        "universe": build_example_universe_config(
            symbol,
            context_symbols=context_symbols,
            market="spot",
            snapshot_timestamp=start,
        ),
        "indicators": _copy_value(list(indicators)),
        "features": {
            "lags": [1, 3, 6],
            "frac_diff_d": 0.4,
            "rolling_window": 20,
            "squeeze_quantile": 0.2,
            "context_timeframes": ["4h", "1d"],
            "context_missing_policy": {"mode": "preserve_missing", "add_indicator": True},
            "futures_context_ttl": {
                "mark_price": "2h",
                "premium_index": "2h",
                "funding": "12h",
                "recent": "4h",
                "max_stale_rate": 0.05,
                "max_unknown_rate": 0.0,
            },
            "cross_asset_context_ttl": {"max_age": "2h", "max_unknown_rate": 0.0},
        },
        "feature_selection": {"enabled": True, "max_features": 96, "min_mi_threshold": 0.0005},
        "regime": {"method": "hmm"},
        "data_quality": {
            "exclude_flagged_quarantine_rows_from_modeling": True,
            "actions": {
                "ohlc_inconsistency": "drop",
                "duplicate_timestamp": "drop",
                "retrograde_timestamp": "drop",
                "nonpositive_volume": "drop",
                "return_spike": "null",
                "range_spike": "null",
                "quote_volume_inconsistency": "null",
                "trade_count_anomaly": "null",
            },
        },
        "labels": {
            "kind": "triple_barrier",
            "pt_sl": (2.0, 2.0),
            "max_holding": 24,
            "min_return": 0.001,
            "volatility_window": 24,
            "barrier_tie_break": "sl",
        },
        "model": {
            "type": "gbm",
            "cv_method": "cpcv",
            "n_blocks": 4,
            "test_blocks": 2,
            "validation_fraction": 0.2,
            "meta_n_splits": 2,
        },
        "signals": {
            "avg_win": 0.02,
            "avg_loss": 0.02,
            "shrinkage_alpha": 0.5,
            "fraction": 0.5,
            "min_trades_for_kelly": 30,
            "max_kelly_fraction": 0.5,
            "threshold": 0.01,
            "edge_threshold": 0.05,
            "meta_threshold": 0.55,
            "tuning_min_trades": 5,
        },
        "backtest": {
            "equity": 10_000,
            "fee_rate": 0.001,
            "slippage_rate": 0.0002,
            "slippage_model": "sqrt_impact",
            "engine": "vectorbt",
            "evaluation_mode": "research_only",
            "use_open_execution": True,
            "signal_delay_bars": 2,
        },
    }
    if custom_data:
        config["data"]["custom_data"] = _copy_value(list(custom_data))
    return clone_config_with_overrides(config, config_overrides)


def build_futures_research_config(
    symbol,
    interval,
    start,
    end,
    *,
    indicators,
    context_symbols=None,
    config_overrides=None,
):
    """Build the baseline futures-research example config."""

    context_symbols = list(context_symbols or [])
    config = {
        "data": {
            "symbol": str(symbol),
            "interval": str(interval),
            "start": start,
            "end": end,
            "market": "um_futures",
            "duplicate_policy": "fail",
            "futures_context": {
                "enabled": True,
                "include_recent_stats": True,
                "recent_stats_availability_lag": "period_close",
            },
            "cross_asset_context": {"symbols": context_symbols, "market": "um_futures"},
        },
        "universe": build_example_universe_config(
            symbol,
            context_symbols=context_symbols,
            market="um_futures",
            snapshot_timestamp=start,
        ),
        "indicators": _copy_value(list(indicators)),
        "features": {
            "lags": [1, 3, 6],
            "frac_diff_d": 0.4,
            "rolling_window": 24,
            "context_timeframes": ["4h"],
            "context_missing_policy": {"mode": "preserve_missing", "add_indicator": True},
            "futures_context_ttl": {
                "mark_price": "2h",
                "premium_index": "2h",
                "funding": "12h",
                "recent": "4h",
                "max_stale_rate": 0.05,
                "max_unknown_rate": 0.0,
            },
            "cross_asset_context_ttl": {"max_age": "2h", "max_unknown_rate": 0.0},
        },
        "feature_selection": {"enabled": True, "max_features": 64, "min_mi_threshold": 0.0},
        "regime": {"method": "hmm"},
        "data_quality": {
            "exclude_flagged_quarantine_rows_from_modeling": True,
            "actions": {
                "ohlc_inconsistency": "drop",
                "duplicate_timestamp": "drop",
                "retrograde_timestamp": "drop",
                "nonpositive_volume": "drop",
                "return_spike": "null",
                "range_spike": "null",
                "quote_volume_inconsistency": "null",
                "trade_count_anomaly": "null",
            },
        },
        "labels": {
            "kind": "triple_barrier",
            "pt_sl": (1.5, 1.5),
            "max_holding": 12,
            "min_return": 0.0005,
            "volatility_window": 24,
            "barrier_tie_break": "sl",
        },
        "model": {
            "type": "logistic",
            "cv_method": "cpcv",
            "n_blocks": 4,
            "test_blocks": 2,
            "validation_fraction": 0.2,
            "meta_n_splits": 2,
        },
        "signals": {
            "threshold": 0.0,
            "edge_threshold": 0.0,
            "shrinkage_alpha": 0.5,
            "fraction": 0.75,
            "min_trades_for_kelly": 30,
            "max_kelly_fraction": 0.5,
            "meta_threshold": 0.5,
            "profitability_threshold": 0.5,
            "expected_edge_threshold": 0.0,
            "sizing_mode": "expected_utility",
            "tuning_min_trades": 5,
        },
        "backtest": {
            "equity": 10_000,
            "fee_rate": 0.0004,
            "slippage_rate": 0.0002,
            "slippage_model": "sqrt_impact",
            "engine": "pandas",
            "evaluation_mode": "research_only",
            "valuation_price": "mark",
            "apply_funding": True,
            "funding_missing_policy": {"mode": "strict", "expected_interval": "8h", "max_gap_multiplier": 1.25},
            "allow_short": True,
            "leverage": 1.5,
            "use_open_execution": True,
            "signal_delay_bars": 1,
            "futures_account": {
                "enabled": True,
                "margin_mode": "isolated",
                "warning_margin_ratio": 0.8,
                "leverage_brackets_data": {
                    "symbol": str(symbol),
                    "brackets": [
                        {
                            "bracket": 1,
                            "initial_leverage": 20.0,
                            "notional_floor": 0.0,
                            "notional_cap": 50_000.0,
                            "maint_margin_ratio": 0.02,
                            "cum": 0.0,
                        },
                        {
                            "bracket": 2,
                            "initial_leverage": 10.0,
                            "notional_floor": 50_000.0,
                            "notional_cap": 250_000.0,
                            "maint_margin_ratio": 0.04,
                            "cum": 0.0,
                        },
                    ],
                },
            },
        },
    }
    return clone_config_with_overrides(config, config_overrides)


def build_trade_ready_runtime_overrides(*, market="spot"):
    """Build shared fail-closed runtime overrides for trade-ready examples."""

    normalized_market = str(market or "spot").strip().lower()
    backtest_overrides = {"evaluation_mode": "trade_ready"}
    if normalized_market != "spot":
        backtest_overrides.update(
            {
                "apply_funding": True,
                "funding_missing_policy": {
                    "mode": "strict",
                    "expected_interval": "8h",
                    "max_gap_multiplier": 1.25,
                },
            }
        )
    return {
        "data": {
            "gap_policy": "fail",
            "duplicate_policy": "fail",
        },
        "data_quality": {
            "block_on_quarantine": True,
            "exclude_flagged_quarantine_rows_from_modeling": True,
        },
        "signals": {
            "require_paper_verification_for_kelly": True,
            "require_live_calibration_for_kelly": True,
            "uncalibrated_kelly_fraction_cap": 0.25,
            "max_live_calibration_error": 0.25,
        },
        "backtest": backtest_overrides,
    }


def build_research_demo_runtime_overrides(*, market="spot"):
    """Build the explicit research-demo runtime profile."""

    normalized_market = str(market or "spot").strip().lower()
    backtest_overrides = {
        "evaluation_mode": "research_only",
        "execution_profile": "research_surrogate",
    }
    if normalized_market != "spot":
        backtest_overrides.update(
            {
                "apply_funding": True,
                "funding_missing_policy": {
                    "mode": "zero_fill_debug",
                    "expected_interval": "8h",
                    "max_gap_multiplier": 1.5,
                },
            }
        )
    return {"backtest": backtest_overrides}


def build_default_certification_scenario_matrix():
    """Build a conservative default stress matrix for local certification examples."""

    return {
        "downtime": {
            "events": [{"event_type": "downtime", "timestamp": "2024-03-05T12:00:00Z"}],
            "policy": {"downtime_action": "freeze"},
        },
        "stale_mark": {
            "events": [{"event_type": "stale_mark", "timestamp": "2024-03-12T12:00:00Z"}],
            "policy": {"stale_mark_action": "reject"},
        },
        "halt": {
            "events": [{"event_type": "halt", "start": "2024-03-19T12:00:00Z", "end": "2024-03-19T14:00:00Z"}],
            "policy": {},
        },
    }


def build_local_certification_runtime_overrides(*, market="spot"):
    """Build a strict but locally runnable certification profile.

    This profile is intended to sit between the research demo and the fully
    trade-ready operator path. It keeps fail-closed data rules and binding
    validation, while staying small enough for a local certification run.
    """

    overrides = build_trade_ready_runtime_overrides(market=market)
    scenario_matrix = build_default_certification_scenario_matrix()
    local_backtest = {
        "engine": "pandas",
        "evaluation_mode": "local_certification",
        "execution_profile": "local_l1_certification",
        "required_stress_scenarios": list(scenario_matrix),
        "scenario_matrix": scenario_matrix,
        "liquidity_lag_bars": 1,
        "participation_cap": 0.05,
        "min_fill_ratio": 0.25,
        "price_protection_points": 100,
        "execution_policy": {
            "adapter": "nautilus",
            "time_in_force": "IOC",
            "participation_cap": 0.05,
            "min_fill_ratio": 0.25,
            "force_simulation": False,
        },
    }
    return clone_config_with_overrides(
        overrides,
        {
            "backtest": local_backtest,
            "monitoring": {"policy_profile": "local_certification"},
            "data_certification": {"enabled": True},
        },
    )


def build_local_certification_surrogate_runtime_overrides(*, market="spot"):
    """Build the explicit surrogate-backed local certification profile.

    This keeps the stricter local-certification data, validation, and monitoring
    controls, but makes the execution evidence class explicit when Nautilus is
    unavailable on consumer hardware.
    """

    return clone_config_with_overrides(
        build_local_certification_runtime_overrides(market=market),
        {
            "backtest": {
                "execution_profile": "local_surrogate_certification",
                "execution_policy": {
                    "adapter": "bar_surrogate",
                    "time_in_force": "IOC",
                    "participation_cap": 0.05,
                    "min_fill_ratio": 0.25,
                    "force_simulation": False,
                },
            },
        },
    )

def parse_local_certification_args(description):
    """Parse the shared --local-certification flag used by builder-based examples."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--local-certification",
        action="store_true",
        help=(
            "Run this example with the strict local-certification runtime profile. "
            "Uses a real local Nautilus backend when available, otherwise falls back to an explicit surrogate "
            "certification mode that remains paper or pre-capital only."
        ),
    )
    return parser.parse_args()


def prepare_example_runtime_config(
    config,
    *,
    market="spot",
    local_certification=False,
    nautilus_available=True,
    example_name="this example",
):
    """Optionally promote a research example into the strict local-certification runtime profile."""

    resolved = clone_config_with_overrides(config)
    universe_config = dict(resolved.get("universe") or {})
    universe_source = str(universe_config.get("source") or "").strip().lower()
    if universe_source == "synthetic_example_snapshot":
        runtime = dict(resolved.get("example_runtime") or {})
        runtime.setdefault("mode", "research_only")
        runtime.setdefault("risk_level", "research_only")
        synthetic_note = (
            "synthetic universe snapshot is enabled; cross-symbol eligibility, listing age, and liquidity "
            "are example assumptions rather than survivorship-safe historical evidence."
        )
        existing_note = str(runtime.get("note") or "").strip()
        if synthetic_note not in existing_note:
            runtime["note"] = synthetic_note if not existing_note else f"{existing_note} {synthetic_note}"
        resolved["example_runtime"] = runtime
    if not local_certification:
        return resolved

    if not nautilus_available:
        return clone_config_with_overrides(
            resolved,
            clone_config_with_overrides(
                build_local_certification_surrogate_runtime_overrides(market=market),
                {
                    "example_runtime": {
                        "mode": "local_certification_surrogate",
                        "risk_level": "paper_or_pre_capital_only",
                        "note": (
                            "explicit local-certification surrogate is enabled because Nautilus is unavailable; "
                            "execution evidence stays non-event-driven and cannot support live-capital release."
                        ),
                    }
                },
            ),
        )

    return clone_config_with_overrides(
        resolved,
        clone_config_with_overrides(
            build_local_certification_runtime_overrides(market=market),
            {
                "example_runtime": {
                    "mode": "local_certification",
                    "risk_level": "paper_or_pre_capital_only",
                    "note": "strict local certification is enabled; treat results as paper or pre-capital evidence, not a live release.",
                }
            },
        ),
    )


def build_trade_ready_automl_overrides(
    *,
    storage_path,
    study_name,
    profile="certification",
    n_trials=None,
    seed=42,
    objective="risk_adjusted_after_costs",
    validation_fraction=None,
    locked_holdout_fraction=None,
    min_validation_trade_count=None,
    max_trials_per_model_family=None,
    search_space=None,
    extra_automl_fields=None,
):
    """Build a hardened AutoML override block for trade-ready research runs.

    The default `certification` profile is intended to be promotion-safe rather
    than fast. Use `profile="smoke"` only when you explicitly want a reduced-
    power local feedback run.
    """

    profile_name = str(profile or "certification").strip().lower()
    if profile_name in {"smoke", "demo", "reduced_power", "reduced-power"}:
        profile_settings = {
            "name": "smoke",
            "reduced_power": True,
            "n_trials": 4,
            "validation_fraction": 0.20,
            "locked_holdout_fraction": 0.20,
            "min_validation_trade_count": 20,
            "max_trials_per_model_family": 4,
            "minimum_dsr_threshold": 0.30,
            "max_generalization_gap": 0.25,
            "max_param_fragility": 0.20,
            "local_perturbation_limit": 4,
            "replication": {
                "alternate_window_count": 1,
                "min_coverage": 2,
                "min_pass_rate": 1.0,
                "min_rows": 64,
            },
            "deflated_sharpe": {"min_track_record_length": 10},
            "pbo": {"min_block_size": 5},
            "post_selection": {
                "max_candidates": 5,
                "min_overlap_observations": 10,
                "bootstrap_samples": 500,
            },
            "objective_gates": {
                "min_trade_count": 20,
                "min_effective_bet_count": 20,
                "require_statistical_significance": True,
                "min_significance_observations": 32,
                "min_sharpe_ci_lower": None,
            },
        }
    else:
        profile_settings = {
            "name": "certification",
            "reduced_power": False,
            "n_trials": 12,
            "validation_fraction": 0.25,
            "locked_holdout_fraction": 0.25,
            "min_validation_trade_count": 40,
            "max_trials_per_model_family": 8,
            "minimum_dsr_threshold": 0.40,
            "max_generalization_gap": 0.20,
            "max_param_fragility": 0.15,
            "local_perturbation_limit": 8,
            "replication": {
                "alternate_window_count": 2,
                "min_coverage": 3,
                "min_pass_rate": 1.0,
                "min_rows": 128,
            },
            "deflated_sharpe": {"min_track_record_length": 20},
            "pbo": {"min_block_size": 8},
            "post_selection": {
                "max_candidates": 8,
                "min_overlap_observations": 20,
                "bootstrap_samples": 1000,
            },
            "objective_gates": {
                "min_trade_count": 40,
                "min_effective_bet_count": 40,
                "require_statistical_significance": True,
                "min_significance_observations": 64,
                "min_sharpe_ci_lower": 0.0,
            },
        }

    resolved_n_trials = int(profile_settings["n_trials"] if n_trials is None else n_trials)
    resolved_validation_fraction = float(
        profile_settings["validation_fraction"] if validation_fraction is None else validation_fraction
    )
    resolved_locked_holdout_fraction = float(
        profile_settings["locked_holdout_fraction"] if locked_holdout_fraction is None else locked_holdout_fraction
    )
    resolved_min_validation_trade_count = int(
        profile_settings["min_validation_trade_count"]
        if min_validation_trade_count is None
        else min_validation_trade_count
    )
    resolved_max_trials_per_model_family = int(
        profile_settings["max_trials_per_model_family"]
        if max_trials_per_model_family is None
        else max_trials_per_model_family
    )

    constrained_search_space = {
        "features": {
            "lags": {"type": "categorical", "choices": ["1,3,6", "1,4,12"]},
            "frac_diff_d": {"type": "categorical", "choices": [0.4, 0.6]},
            "rolling_window": {"type": "categorical", "choices": [20, 28]},
            "squeeze_quantile": {"type": "categorical", "choices": [0.15, 0.2]},
        },
        "feature_selection": {
            "enabled": {"type": "categorical", "choices": [True]},
            "max_features": {"type": "categorical", "choices": [48, 64]},
            "min_mi_threshold": {"type": "categorical", "choices": [0.0005, 0.001]},
        },
        "labels": {
            "pt_mult": {"type": "categorical", "choices": [1.5, 2.0]},
            "sl_mult": {"type": "categorical", "choices": [1.5, 2.0]},
            "max_holding": {"type": "categorical", "choices": [12, 24]},
            "min_return": {"type": "categorical", "choices": [0.0005, 0.001]},
            "volatility_window": {"type": "categorical", "choices": [24]},
            "barrier_tie_break": {"type": "categorical", "choices": ["sl", "pt"]},
        },
        "regime": {
            "n_regimes": {"type": "categorical", "choices": [2, 3]},
        },
        "model": {
            "type": {"type": "categorical", "choices": ["gbm", "logistic"]},
            "gap": {"type": "categorical", "choices": [24, 48]},
            "validation_fraction": {"type": "categorical", "choices": [resolved_validation_fraction]},
            "meta_n_splits": {"type": "categorical", "choices": [2]},
            "params": {
                "gbm": {
                    "n_estimators": {"type": "categorical", "choices": [100, 200]},
                    "learning_rate": {"type": "categorical", "choices": [0.05, 0.1]},
                    "max_depth": {"type": "categorical", "choices": [2, 3]},
                    "subsample": {"type": "categorical", "choices": [0.85, 1.0]},
                    "min_samples_leaf": {"type": "categorical", "choices": [3, 5]},
                },
                "logistic": {
                    "c": {"type": "categorical", "choices": [0.1, 1.0, 5.0]},
                },
            },
        },
    }
    if not bool(profile_settings["reduced_power"]):
        constrained_search_space["features"] = {
            "lags": {"type": "categorical", "choices": ["1,3,6"]},
            "frac_diff_d": {"type": "categorical", "choices": [0.4]},
            "rolling_window": {"type": "categorical", "choices": [20]},
            "squeeze_quantile": {"type": "categorical", "choices": [0.15]},
        }
        constrained_search_space["feature_selection"] = {
            "enabled": {"type": "categorical", "choices": [True]},
            "max_features": {"type": "categorical", "choices": [48]},
            "min_mi_threshold": {"type": "categorical", "choices": [0.0005]},
        }
        constrained_search_space["labels"] = {
            "pt_mult": {"type": "categorical", "choices": [1.5]},
            "sl_mult": {"type": "categorical", "choices": [1.5]},
            "max_holding": {"type": "categorical", "choices": [24]},
            "min_return": {"type": "categorical", "choices": [0.0005]},
            "volatility_window": {"type": "categorical", "choices": [24]},
            "barrier_tie_break": {"type": "categorical", "choices": ["sl"]},
        }
        constrained_search_space["regime"] = {
            "n_regimes": {"type": "categorical", "choices": [2]},
        }
        constrained_search_space["model"]["gap"] = {"type": "categorical", "choices": [24]}

    profile = {
        "automl": {
            "enabled": True,
            "n_trials": resolved_n_trials,
            "seed": int(seed),
            "validation_fraction": resolved_validation_fraction,
            "minimum_dsr_threshold": float(profile_settings["minimum_dsr_threshold"]),
            "locked_holdout_enabled": True,
            "locked_holdout_fraction": resolved_locked_holdout_fraction,
            "enable_pruning": False,
            "objective": str(objective),
            "study_name": str(study_name),
            "storage": str(storage_path),
            "trade_ready_profile": {
                "name": profile_settings["name"],
                "reduced_power": bool(profile_settings["reduced_power"]),
                "promotion_safe_default": not bool(profile_settings["reduced_power"]),
                "n_trials": resolved_n_trials,
                "validation_fraction": resolved_validation_fraction,
                "locked_holdout_fraction": resolved_locked_holdout_fraction,
                "min_validation_trade_count": resolved_min_validation_trade_count,
                "min_significance_observations": int(profile_settings["objective_gates"]["min_significance_observations"]),
                "replication_min_coverage": int(profile_settings["replication"]["min_coverage"]),
                "replication_alternate_window_count": int(profile_settings["replication"]["alternate_window_count"]),
                "post_selection_bootstrap_samples": int(profile_settings["post_selection"]["bootstrap_samples"]),
                "min_track_record_length": int(profile_settings["deflated_sharpe"]["min_track_record_length"]),
            },
            "validation_contract": {
                "search_ranker": "cpcv",
                "contiguous_validation": "walk_forward_replay",
                "locked_holdout": "single_access_contiguous",
                "replication": "required",
            },
            "selection_policy": {
                "enabled": True,
                "max_generalization_gap": float(profile_settings["max_generalization_gap"]),
                "max_param_fragility": float(profile_settings["max_param_fragility"]),
                "max_complexity_score": 16.0,
                "min_validation_trade_count": resolved_min_validation_trade_count,
                "require_locked_holdout_pass": True,
                "min_locked_holdout_score": 0.0,
                "max_feature_count_ratio": 0.75,
                "max_trials_per_model_family": resolved_max_trials_per_model_family,
                "local_perturbation_limit": int(profile_settings["local_perturbation_limit"]),
                "require_fold_stability_pass": True,
                "required_execution_mode": "event_driven",
                "required_stress_scenarios": ["downtime", "stale_mark", "halt"],
            },
            "replication": {
                "enabled": True,
                "include_symbol_cohorts": True,
                "include_window_cohorts": True,
                "alternate_window_count": int(profile_settings["replication"]["alternate_window_count"]),
                "alternate_window_fraction": 0.5,
                "min_coverage": int(profile_settings["replication"]["min_coverage"]),
                "min_pass_rate": float(profile_settings["replication"]["min_pass_rate"]),
                "min_score": 0.0,
                "min_rows": int(profile_settings["replication"]["min_rows"]),
            },
            "portability_contract": {
                "enabled": True,
                "accepted_kinds": ["symbol", "period"],
                "min_supporting_cohorts": 1,
                "min_passed_supporting_cohorts": 1,
                "require_frozen_universe": True,
            },
            "overfitting_control": {
                "enabled": True,
                "selection_mode": "penalized_ranking",
                "deflated_sharpe": {
                    "enabled": True,
                    "use_effective_trial_count": True,
                    "min_track_record_length": int(profile_settings["deflated_sharpe"]["min_track_record_length"]),
                },
                "pbo": {
                    "enabled": True,
                    "n_blocks": 8,
                    "min_block_size": int(profile_settings["pbo"]["min_block_size"]),
                    "metric": "sharpe_ratio",
                    "overlap_policy": "strict_intersection",
                    "min_overlap_fraction": 0.5,
                },
                "post_selection": {
                    "enabled": True,
                    "require_pass": True,
                    "pass_rule": "spa",
                    "alpha": 0.05,
                    "max_candidates": int(profile_settings["post_selection"]["max_candidates"]),
                    "correlation_threshold": 0.9,
                    "min_overlap_fraction": 0.5,
                    "min_overlap_observations": int(profile_settings["post_selection"]["min_overlap_observations"]),
                    "overlap_policy": "strict_intersection",
                    "bootstrap_samples": int(profile_settings["post_selection"]["bootstrap_samples"]),
                    "random_state": int(seed),
                },
            },
            "objective_gates": {
                "min_trade_count": int(profile_settings["objective_gates"]["min_trade_count"]),
                "min_effective_bet_count": int(profile_settings["objective_gates"]["min_effective_bet_count"]),
                "require_statistical_significance": bool(
                    profile_settings["objective_gates"]["require_statistical_significance"]
                ),
                "min_significance_observations": int(
                    profile_settings["objective_gates"]["min_significance_observations"]
                ),
                "min_sharpe_ci_lower": profile_settings["objective_gates"]["min_sharpe_ci_lower"],
            },
            "search_space": clone_config_with_overrides(
                constrained_search_space,
                search_space,
            ),
        }
    }
    return clone_config_with_overrides(profile, {"automl": extra_automl_fields} if extra_automl_fields else None)


def build_local_certification_automl_overrides(
    *,
    storage_path,
    study_name,
    n_trials=6,
    seed=42,
    extra_automl_fields=None,
):
    """Build a local-certification AutoML profile.

    The profile keeps locked holdout, selection policy, replication, and
    overfitting controls enabled, but uses a smaller search budget than the
    full trade-ready certification example.
    """

    base = build_trade_ready_automl_overrides(
        storage_path=storage_path,
        study_name=study_name,
        profile="certification",
        n_trials=n_trials,
        seed=seed,
        validation_fraction=0.20,
        locked_holdout_fraction=0.20,
        min_validation_trade_count=20,
        max_trials_per_model_family=4,
        extra_automl_fields=extra_automl_fields,
    )
    return clone_config_with_overrides(
        base,
        {
            "automl": {
                "trade_ready_profile": {
                    "name": "local_certification",
                    "reduced_power": False,
                    "promotion_safe_default": True,
                    "n_trials": int(n_trials),
                    "validation_fraction": 0.20,
                    "locked_holdout_fraction": 0.20,
                    "min_validation_trade_count": 20,
                    "min_significance_observations": 48,
                    "replication_min_coverage": 2,
                    "replication_alternate_window_count": 1,
                    "post_selection_bootstrap_samples": 500,
                    "min_track_record_length": 12,
                },
                "replication": {
                    "alternate_window_count": 1,
                    "min_coverage": 2,
                    "min_pass_rate": 1.0,
                    "min_rows": 96,
                },
                "selection_policy": {
                    "min_validation_trade_count": 20,
                },
                "overfitting_control": {
                    "deflated_sharpe": {"min_track_record_length": 12},
                    "pbo": {"min_block_size": 6},
                    "post_selection": {"bootstrap_samples": 500, "min_overlap_observations": 12},
                },
                "objective_gates": {
                    "min_trade_count": 20,
                    "min_effective_bet_count": 20,
                    "require_statistical_significance": True,
                    "min_significance_observations": 48,
                    "min_sharpe_ci_lower": 0.0,
                },
            }
        },
    )


def seed_offline_pipeline_state(
    pipeline,
    raw_data,
    *,
    data=None,
    futures_context=None,
    cross_asset_context=None,
    symbol_filters=None,
    extra_state=None,
):
    """Seed a pipeline with offline state for synthetic or deterministic tests."""

    pipeline.state["raw_data"] = _copy_value(raw_data)
    pipeline.state["data"] = _copy_value(data) if data is not None else _copy_value(raw_data)
    if futures_context is not None:
        pipeline.state["futures_context"] = _copy_value(futures_context)
    if cross_asset_context is not None:
        pipeline.state["cross_asset_context"] = _copy_value(cross_asset_context)
    if symbol_filters is not None:
        pipeline.state["symbol_filters"] = _copy_value(symbol_filters)
    for key, value in dict(extra_state or {}).items():
        pipeline.state[key] = _copy_value(value)
    return pipeline


def _format_metric(value, digits=4, percent=False, money=False):
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not isfinite(numeric):
        return str(value)
    if money:
        return f"${numeric:,.2f}"
    if percent:
        return f"{numeric:.2%}"
    return f"{numeric:.{digits}f}"


def _format_confidence_interval(interval, digits=4, percent=False, money=False):
    if not isinstance(interval, dict):
        return None
    lower = interval.get("lower")
    upper = interval.get("upper")
    confidence_level = interval.get("confidence_level")
    if lower is None or upper is None or confidence_level is None:
        return None
    rendered_lower = _format_metric(lower, digits=digits, percent=percent, money=money)
    rendered_upper = _format_metric(upper, digits=digits, percent=percent, money=money)
    return f"{confidence_level:.0%} [{rendered_lower}, {rendered_upper}]"


def print_section(sep, step, title):
    print(f"\n{sep}\nStep {step} · {title}\n{sep}")


def build_example_universe_config(
    primary_symbol,
    *,
    context_symbols=None,
    market="spot",
    snapshot_timestamp=None,
    min_history_days=30,
    min_liquidity=1_000_000.0,
    requested_symbol_policy="error",
):
    """Build a self-contained historical-universe config for runnable examples.

    Examples that request cross-asset context now need an explicit universe snapshot.
    Keeping the snapshot inline avoids hidden cache dependencies while still exercising
    the same eligibility path as the production pipeline.
    """

    resolved_snapshot = pd.Timestamp(snapshot_timestamp or "2024-01-01", tz="UTC")
    symbols = [str(primary_symbol), *[str(symbol) for symbol in (context_symbols or [])]]
    ordered_symbols = list(dict.fromkeys(symbols))
    base_liquidity = max(float(min_liquidity) * 10.0, 10_000_000.0)

    snapshot_symbols = []
    for index, symbol in enumerate(ordered_symbols):
        snapshot_symbols.append(
            {
                "symbol": symbol,
                "market": market,
                "status": "TRADING",
                "listing_start": "2020-01-01T00:00:00Z",
                "avg_daily_quote_volume": float(base_liquidity - (index * float(min_liquidity))),
            }
        )

    return {
        "market": market,
        "source": "synthetic_example_snapshot",
        "survivorship_safe": False,
        "liquidity_source": "fabricated",
        "snapshots": [
            {
                "snapshot_timestamp": resolved_snapshot.isoformat().replace("+00:00", "Z"),
                "market": market,
                "source": "synthetic_example_snapshot",
                "symbols": snapshot_symbols,
            }
        ],
        "requested_symbol_policy": requested_symbol_policy,
        "min_history_days": int(min_history_days),
        "min_liquidity": float(min_liquidity),
    }


def print_stationarity_summary(stationarity):
    for feature_name in ["close", "close_fracdiff"]:
        stats = stationarity.get(feature_name)
        if not isinstance(stats, dict):
            continue
        print(
            f"  {feature_name:<13}: stationary={stats.get('stationary')}  "
            f"p={_format_metric(stats.get('p_value'), digits=6)}"
        )

    screening = stationarity.get("feature_screening", {}).get("summary", {})
    if screening:
        print(
            "  screened     : "
            f"{screening.get('screened_feature_count', 0)}/{screening.get('total_features', 0)}  "
            f"transformed={screening.get('transformed_features', 0)}  "
            f"dropped={screening.get('dropped_features', 0)}"
        )
        transform_usage = screening.get("transform_usage")
        if transform_usage:
            print(f"  transforms   : {transform_usage}")


def print_regime_summary(regimes):
    if isinstance(regimes, pd.DataFrame):
        print(f"  regime cols  : {list(regimes.columns)}")
        regime_counts = regimes["regime"].value_counts().to_dict() if "regime" in regimes.columns else {}
    else:
        regime_counts = pd.Series(regimes).value_counts().to_dict()
    print(f"  regime counts: {regime_counts}")


def print_label_summary(labels):
    print(f"  label rows   : {len(labels)}")
    integrity_report = dict(getattr(labels, "attrs", {}).get("integrity_report") or {})
    if integrity_report:
        print(
            "  dropped bad  : "
            f"{integrity_report.get('dropped_incomplete_future_windows', 0)}"
        )
    data_quality_report = dict(getattr(labels, "attrs", {}).get("data_quality_report") or {})
    data_quality_summary = dict(data_quality_report.get("summary") or {})
    if data_quality_summary:
        print(
            "  quarantine   : "
            f"rows={data_quality_summary.get('quarantined_rows', 0)}  "
            f"dropped={data_quality_summary.get('dropped_rows', 0)}  "
            f"nulled={data_quality_summary.get('nulled_rows', 0)}  "
            f"excluded={data_quality_summary.get('modeling_excluded_rows', 0)}"
        )
    if "label" in labels.columns:
        print(f"  distribution : {labels['label'].value_counts().to_dict()}")
    if "barrier" in labels.columns:
        print(f"  barriers     : {labels['barrier'].value_counts().to_dict()}")
    if "trend_horizon" in labels.columns:
        print(
            "  trend horizon: "
            f"median={_format_metric(labels['trend_horizon'].median(), digits=2)}  "
            f"max={_format_metric(labels['trend_horizon'].max(), digits=2)}"
        )


def print_alignment_summary(aligned):
    X = aligned["X"]
    y = aligned.get("y")
    print(f"  samples      : {len(X)}")
    print(f"  feature count: {X.shape[1]}")
    if y is not None:
        print(f"  label mix    : {pd.Series(y).value_counts().to_dict()}")


def print_feature_selection_summary(selection):
    report = getattr(selection, "report", {})
    print("  preview mode : supervised MI filtering runs inside each validation split")
    print(
        "  configured   : "
        f"max_features={report.get('max_features') or 'auto'}  "
        f"min_mi={report.get('min_mi_threshold')}"
    )
    family_summary = report.get("input_family_summary", {})
    if family_summary.get("selected_family_counts"):
        print(f"  families     : {family_summary['selected_family_counts']}")


def print_weight_summary(weights):
    print(
        "  weight stats : "
        f"min={_format_metric(weights.min())}  "
        f"max={_format_metric(weights.max())}  "
        f"mean={_format_metric(weights.mean())}"
    )


def print_training_summary(training):
    validation = training.get("validation", {})
    validation_sources = training.get("validation_sources") or {}
    if validation:
        if validation.get("method") == "cpcv":
            print(
                "  validation   : "
                f"cpcv  splits={validation.get('split_count')}  "
                f"n_blocks={validation.get('n_blocks')}  "
                f"test_blocks={validation.get('test_blocks')}  "
                f"embargo={validation.get('embargo_bars')}"
            )
        else:
            print(
                "  validation   : "
                f"walk_forward  splits={validation.get('split_count')}  "
                f"gap={validation.get('gap')}"
            )
    if validation_sources:
        print(
            "  val sources  : "
            f"select={validation_sources.get('selection_metric_source')}  "
            f"diagnostic={validation_sources.get('diagnostic_metric_source')}  "
            f"tradable={validation_sources.get('tradable_metric_source')}"
        )
        print(f"  val gates    : {bool(validation_sources.get('all_required_sources_passed', True))}")

    for metric in training.get("fold_metrics", []):
        split_label = metric.get("split_id") or f"fold {metric.get('fold')}"
        parts = [
            split_label,
            f"acc={_format_metric(metric.get('accuracy'))}",
            f"f1={_format_metric(metric.get('f1_macro'))}",
        ]
        if metric.get("directional_accuracy") is not None:
            parts.append(f"dir_acc={_format_metric(metric.get('directional_accuracy'))}")
        if metric.get("directional_f1_macro") is not None:
            parts.append(f"dir_f1={_format_metric(metric.get('directional_f1_macro'))}")
        if metric.get("log_loss") is not None:
            parts.append(f"log_loss={_format_metric(metric.get('log_loss'))}")
        if metric.get("test_blocks") is not None:
            parts.append(f"test_blocks={metric.get('test_blocks')}")
        print(f"  {'  '.join(parts)}")

    print(f"  avg accuracy : {_format_metric(training.get('avg_accuracy'))}")
    print(f"  avg f1       : {_format_metric(training.get('avg_f1_macro'))}")
    if training.get("avg_directional_accuracy") is not None:
        print(f"  avg dir acc  : {_format_metric(training.get('avg_directional_accuracy'))}")
    if training.get("avg_directional_f1_macro") is not None:
        print(f"  avg dir f1   : {_format_metric(training.get('avg_directional_f1_macro'))}")
    if training.get("avg_log_loss") is not None:
        print(f"  avg log loss : {_format_metric(training.get('avg_log_loss'))}")
    if training.get("avg_brier_score") is not None:
        print(f"  avg brier    : {_format_metric(training.get('avg_brier_score'))}")
    if training.get("avg_calibration_error") is not None:
        print(f"  avg calib    : {_format_metric(training.get('avg_calibration_error'))}")

    feature_selection = training.get("feature_selection", {})
    if feature_selection:
        print(
            "  avg selected : "
            f"{_format_metric(feature_selection.get('avg_selected_features'), digits=2)}"
        )
        if feature_selection.get("selected_families"):
            print(f"  families     : {feature_selection.get('selected_families')}")
        if feature_selection.get("avg_selected_family_counts"):
            print(f"  avg fam cnt  : {feature_selection.get('avg_selected_family_counts')}")
        if feature_selection.get("endogenous_only_selected_all_folds"):
            print("  family mode  : endogenous-only selected in every fold")
    bootstrap = training.get("bootstrap", {})
    if bootstrap:
        print(
            "  bootstrap    : "
            f"used={bootstrap.get('used_in_any_fold')}  "
            f"warnings={bootstrap.get('warning_count', 0)}"
        )
    fallback_scope = training.get("fallback_inference", {})
    if fallback_scope:
        print(
            "  fallback     : "
            f"mode={fallback_scope.get('mode')}  "
            f"safe_rows={fallback_scope.get('aligned_safe_row_count', 0)}  "
            f"last_train_end={fallback_scope.get('last_fold_train_end')}"
        )
    if training.get("last_signal_params"):
        print(f"  tuned signals: {training['last_signal_params']}")

    fold_stability = training.get("fold_stability", {})
    if fold_stability.get("metrics"):
        primary_metric = fold_stability.get("primary_metric")
        primary_stats = fold_stability.get("metrics", {}).get(primary_metric, {})
        if primary_metric and primary_stats:
            print(
                "  stability    : "
                f"{primary_metric} cv={_format_metric(primary_stats.get('cv'), digits=6)}  "
                f"range=[{_format_metric(primary_stats.get('min'), digits=6)}, {_format_metric(primary_stats.get('max'), digits=6)}]"
            )
        if fold_stability.get("worst_fold_sharpe") is not None:
            print(f"  worst sharpe : {_format_metric(fold_stability.get('worst_fold_sharpe'), digits=6)}")
        if fold_stability.get("worst_fold_net_profit_pct") is not None:
            print(f"  worst return : {_format_metric(fold_stability.get('worst_fold_net_profit_pct'), digits=6, percent=True)}")
        if fold_stability.get("max_drawdown_dispersion") is not None:
            print(f"  dd dispersion: {_format_metric(fold_stability.get('max_drawdown_dispersion'), digits=6)}")
        if fold_stability.get("policy_enabled"):
            print(f"  stability ok : {fold_stability.get('passed')}  reasons={fold_stability.get('reasons', [])}")

    block_diag = training.get("feature_block_diagnostics", {})
    if block_diag.get("summary"):
        print("  top blocks   :")
        for block in block_diag["summary"][:5]:
            print(
                "    "
                f"{block['block']}: f1_drop={_format_metric(block.get('avg_f1_drop'), digits=6)}  "
                f"acc_drop={_format_metric(block.get('avg_accuracy_drop'), digits=6)}  "
                f"native={_format_metric(block.get('avg_native_importance'), digits=6)}"
            )

    family_diag = training.get("feature_family_diagnostics", {})
    if family_diag.get("summary"):
        print("  top families :")
        for family in family_diag["summary"][:5]:
            print(
                "    "
                f"{family['family']}: f1_drop={_format_metric(family.get('avg_f1_drop'), digits=6)}  "
                f"acc_drop={_format_metric(family.get('avg_accuracy_drop'), digits=6)}  "
                f"native={_format_metric(family.get('avg_native_importance'), digits=6)}"
            )
    if family_diag.get("bundles"):
        print("  family bundles:")
        for bundle in family_diag["bundles"][:4]:
            print(
                "    "
                f"{bundle['bundle']}: f1_drop={_format_metric(bundle.get('avg_f1_drop_vs_full'), digits=6)}  "
                f"acc_drop={_format_metric(bundle.get('avg_accuracy_drop_vs_full'), digits=6)}  "
                f"families={bundle.get('families')}"
            )


def print_signal_summary(signal_result, allow_short=True):
    signal_source = signal_result.get("signal_source")
    if signal_source is not None:
        print(f"  source       : {signal_source}")
    fallback_scope = signal_result.get("fallback_scope", {})
    if signal_source is not None and signal_source.startswith("post_final_training_fallback"):
        print(
            "  fallback     : "
            f"scored={fallback_scope.get('scored_row_count', fallback_scope.get('aligned_safe_row_count', 0))}  "
            f"excluded={fallback_scope.get('excluded_row_count', 0)}"
        )
    paths = signal_result.get("paths") or []
    if paths:
        long_counts = []
        short_counts = []
        flat_counts = []
        avg_abs_sizes = []
        for path in paths:
            signals = pd.Series(path["signals"], copy=False)
            executable_signals = signals if allow_short else signals.clip(lower=0)
            long_counts.append(float((executable_signals == 1).sum()))
            short_counts.append(float((executable_signals == -1).sum()))
            flat_counts.append(float((executable_signals == 0).sum()))
            continuous = path.get("continuous_signals")
            if continuous is not None:
                continuous_series = pd.Series(continuous, copy=False)
                executable_continuous = continuous_series if allow_short else continuous_series.clip(lower=0.0)
                avg_abs_sizes.append(float(executable_continuous.abs().mean()))

        print(f"  path count   : {len(paths)}")
        print(
            "  avg signal mix: "
            f"long={_format_metric(sum(long_counts) / len(long_counts), digits=2)}  "
            f"short={_format_metric(sum(short_counts) / len(short_counts), digits=2)}  "
            f"flat={_format_metric(sum(flat_counts) / len(flat_counts), digits=2)}"
        )
        if avg_abs_sizes:
            print(f"  avg abs size : {_format_metric(sum(avg_abs_sizes) / len(avg_abs_sizes))}")
        return

    signals = pd.Series(signal_result["signals"], copy=False)
    executable_signals = signals if allow_short else signals.clip(lower=0)
    print(
        "  signal mix   : "
        f"long={int((executable_signals == 1).sum())}  "
        f"short={int((executable_signals == -1).sum())}  "
        f"flat={int((executable_signals == 0).sum())}"
    )
    if not allow_short and int((signals == -1).sum()) > 0:
        print(f"  clipped shorts: {int((signals == -1).sum())}")
    continuous = signal_result.get("continuous_signals")
    if continuous is not None:
        continuous_series = pd.Series(continuous, copy=False)
        executable_continuous = continuous_series if allow_short else continuous_series.clip(lower=0.0)
        print(f"  avg abs size : {_format_metric(executable_continuous.abs().mean())}")


def print_backtest_summary(backtest):
    if backtest.get("path_count"):
        print(
            "  validation   : "
            f"{backtest.get('validation_method')}  "
            f"aggregate={backtest.get('aggregate_mode')}  "
            f"paths={backtest.get('path_count')}"
        )

    runtime_details = []
    if backtest.get("evaluation_mode") is not None:
        runtime_details.append(f"mode={backtest.get('evaluation_mode')}")
    if backtest.get("execution_profile") is not None:
        runtime_details.append(f"profile={backtest.get('execution_profile')}")
    if backtest.get("evidence_class") is not None:
        runtime_details.append(f"evidence={backtest.get('evidence_class')}")
    if runtime_details:
        print(f"  runtime      : {'  '.join(runtime_details)}")

    for label, key, formatter in [
        ("engine", "engine", None),
        ("start equity", "starting_equity", "money"),
        ("end equity", "ending_equity", "money"),
        ("net profit", "net_profit", "money"),
        ("net return", "net_profit_pct", "percent"),
        ("gross profit", "gross_profit", "money"),
        ("gross loss", "gross_loss", "money"),
        ("funding pnl", "funding_pnl", "money"),
        ("fees paid", "fees_paid", "money"),
        ("slippage", "slippage_paid", "money"),
        ("sharpe ratio", "sharpe_ratio", None),
        ("sortino", "sortino_ratio", None),
        ("calmar", "calmar_ratio", None),
        ("CAGR", "cagr", "percent"),
        ("volatility", "annualized_volatility", "percent"),
        ("max drawdown", "max_drawdown", "percent"),
        ("profit factor", "profit_factor", None),
        ("expectancy", "expectancy", "money"),
        ("avg win", "avg_win", "money"),
        ("avg loss", "avg_loss", "money"),
        ("exposure", "exposure_rate", "percent"),
        ("signal delay", "signal_delay_bars", None),
        ("trades", "total_trades", None),
        ("closed trades", "closed_trades", None),
        ("win rate", "win_rate", "percent"),
        ("trade win rt", "trade_win_rate", "percent"),
    ]:
        if key not in backtest:
            continue
        value = backtest.get(key)
        if formatter == "money":
            rendered = _format_metric(value, money=True)
        elif formatter == "percent":
            rendered = _format_metric(value, percent=True)
        else:
            rendered = _format_metric(value)
        print(f"  {label:<12}: {rendered}")

    if "max_drawdown_amount" in backtest:
        print(f"  dd amount    : {_format_metric(backtest.get('max_drawdown_amount'), money=True)}")
    if "max_drawdown_duration_bars" in backtest:
        print(
            "  dd duration  : "
            f"{backtest.get('max_drawdown_duration_bars')} bars "
            f"({backtest.get('max_drawdown_duration')})"
        )
    if backtest.get("account_model") == "futures_margin":
        print(
            "  futures acct : "
            f"mode={backtest.get('futures_margin_mode')}  "
            f"brackets={backtest.get('futures_bracket_count')}"
        )
        print(
            "  liquidations : "
            f"count={backtest.get('liquidation_event_count')}  "
            f"fees={_format_metric(backtest.get('liquidation_fee_paid'), money=True)}"
        )
        print(
            "  margin risk  : "
            f"max_ratio={_format_metric(backtest.get('max_margin_ratio'), digits=6)}  "
            f"warn={backtest.get('bars_above_margin_warning')} bars "
            f"({_format_metric(backtest.get('bars_above_margin_warning_rate'), percent=True)})"
        )
        print(
            "  leverage use : "
            f"avg={_format_metric(backtest.get('avg_realized_leverage'), digits=6)}  "
            f"max={_format_metric(backtest.get('max_realized_leverage'), digits=6)}  "
            f"capped={backtest.get('leverage_cap_adjustments')}"
        )
    funding_coverage = backtest.get("funding_coverage_report") or {}
    if funding_coverage:
        print(
            "  funding cov  : "
            f"status={backtest.get('funding_coverage_status', funding_coverage.get('coverage_status'))}  "
            f"missing={funding_coverage.get('missing_event_count', 0)}  "
            f"pass={bool(funding_coverage.get('promotion_pass', True))}"
        )
        funding_reasons = []
        if funding_coverage.get("coverage_reason"):
            funding_reasons.append(funding_coverage.get("coverage_reason"))
        if funding_coverage.get("fallback_assumption"):
            funding_reasons.append(funding_coverage.get("fallback_assumption"))
        print(f"  funding why  : {funding_reasons if funding_reasons else 'none'}")
    metric_ranges = backtest.get("metric_ranges") or {}
    if metric_ranges:
        for key, label in [("net_profit_pct", "net range"), ("sharpe_ratio", "sharpe rng"), ("max_drawdown", "mdd range")]:
            stats = metric_ranges.get(key)
            if not stats:
                continue
            percent = key in {"net_profit_pct", "max_drawdown"}
            print(
                f"  {label:<12}: "
                f"min={_format_metric(stats.get('min'), percent=percent)}  "
                f"med={_format_metric(stats.get('median'), percent=percent)}  "
                f"max={_format_metric(stats.get('max'), percent=percent)}"
            )

    significance = backtest.get("statistical_significance") or {}
    if significance.get("enabled"):
        print(
            "  stats        : "
            f"{significance.get('method')}  "
            f"samples={significance.get('bootstrap_samples')}  "
            f"ci={significance.get('confidence_level', 0.95):.0%}  "
            f"block={significance.get('mean_block_length')}"
        )
        if significance.get("aggregate_mode") is not None:
            print(
                "  stats agg    : "
                f"{significance.get('aggregate_mode')}  "
                f"paths={significance.get('path_count')}"
            )
        if significance.get("benchmark_sharpe_ratio") is not None:
            print(f"  bench sharpe : {_format_metric(significance.get('benchmark_sharpe_ratio'))}")

        stats_metrics = significance.get("metrics") or {}
        for label, key, formatter in [
            ("sharpe ci", "sharpe_ratio", None),
            ("sortino ci", "sortino_ratio", None),
            ("calmar ci", "calmar_ratio", None),
            ("net ret ci", "net_profit_pct", "percent"),
            ("mdd ci", "max_drawdown", "percent"),
        ]:
            metric = stats_metrics.get(key)
            if not metric:
                continue
            interval = _format_confidence_interval(
                metric.get("confidence_interval"),
                digits=4,
                percent=formatter == "percent",
            )
            if interval is None:
                continue
            suffix = []
            if metric.get("p_value_gt_zero") is not None:
                suffix.append(f"p>0={_format_metric(metric.get('p_value_gt_zero'), digits=6)}")
            if metric.get("p_value_gt_benchmark") is not None:
                suffix.append(f"p>bench={_format_metric(metric.get('p_value_gt_benchmark'), digits=6)}")
            suffix_text = f"  {'  '.join(suffix)}" if suffix else ""
            print(f"  {label:<12}: {interval}{suffix_text}")
    elif significance.get("reason"):
        observation_count = significance.get("observation_count")
        min_observations = significance.get("min_observations")
        detail = ""
        if observation_count is not None and min_observations is not None:
            detail = f", observations={observation_count}/{min_observations}"
        print(f"  stats        : unavailable ({significance.get('reason')}{detail})")


def print_data_certification_summary(certification, *, label="data cert"):
    certification = dict(certification or {})
    if not certification:
        print(f"  {label:<12}: unavailable")
        return

    failed_components = (certification.get("summary") or {}).get("failed_components") or []
    print(
        f"  {label:<12}: "
        f"passed={bool(certification.get('promotion_pass', True))}  "
        f"mode={certification.get('mode')}  "
        f"failed={failed_components if failed_components else 'none'}"
    )
    components = dict(certification.get("components") or {})
    if components:
        print(
            "  data parts  : "
            + "  ".join(
                f"{name}={bool(details.get('promotion_pass', True))}"
                for name, details in components.items()
            )
        )
    reasons = list(certification.get("reasons") or [])
    print(f"  data cert why: {reasons if reasons else 'none'}")


def print_deployment_readiness_summary(report, *, label="deploy ready"):
    report = dict(report or {})
    if not report:
        print(f"  {label:<12}: unavailable")
        return

    failed_components = (report.get("summary") or {}).get("failed_components") or []
    print(
        f"  {label:<12}: "
        f"ready={bool(report.get('ready', False))}  "
        f"action={report.get('operator_action')}  "
        f"failed={failed_components if failed_components else 'none'}"
    )

    components = dict(report.get("components") or {})
    if components:
        print(
            "  deploy parts: "
            + "  ".join(
                f"{name}={bool(details.get('passed', False))}"
                for name, details in components.items()
            )
        )

    reasons = list(report.get("reasons") or [])
    print(f"  deploy why  : {reasons if reasons else 'none'}")

    freshness = dict(components.get("model_freshness") or {})
    if freshness.get("available"):
        print(
            "  model age   : "
            f"age={_format_metric(freshness.get('age_days'), digits=1)}d  "
            f"ttl={_format_metric(freshness.get('max_model_age_days'), digits=1)}d  "
            f"expires={freshness.get('expires_at')}  "
            f"stale={bool(freshness.get('expired', False))}"
        )

    actions = list(report.get("recommended_actions") or [])
    if actions:
        print(f"  next actions : {actions}")


def print_paper_calibration_summary(report, *, label="paper calib"):
    report = dict(report or {})
    if not report:
        print(f"  {label:<12}: unavailable")
        return

    print(
        f"  {label:<12}: "
        f"passed={bool(report.get('passed', False))}  "
        f"mode={report.get('mode')}  "
        f"days={_format_metric(report.get('duration_days'), digits=1)}"
    )
    reasons = list(report.get("reasons") or [])
    print(f"  paper why   : {reasons if reasons else 'none'}")
    observation_summary = dict(report.get("observation_summary") or {})
    if observation_summary:
        print(
            "  paper obs   : "
            f"rows={observation_summary.get('observation_count')}  "
            f"trades={_format_metric(observation_summary.get('weighted_trade_count'), digits=0)}  "
            f"range={observation_summary.get('start_timestamp')} -> {observation_summary.get('end_timestamp')}"
        )


def print_operational_limits_summary(report, *, label="risk limits"):
    report = dict(report or {})
    if not report:
        print(f"  {label:<12}: unavailable")
        return

    print(
        f"  {label:<12}: "
        f"passed={bool(report.get('passed', False))}  "
        f"kill_switch={bool(report.get('kill_switch_ready', False))}  "
        f"triggered={bool(report.get('kill_switch_triggered', False))}"
    )
    print(
        "  drawdown    : "
        f"current={_format_metric(report.get('current_drawdown'), digits=4)}  "
        f"limit={_format_metric(report.get('max_drawdown_limit'), digits=4)}  "
        f"action={report.get('enforced_action')}"
    )
    reasons = list(report.get("reasons") or [])
    print(f"  limits why  : {reasons if reasons else 'none'}")


def print_automl_summary(automl):
    print(f"  study name   : {automl.get('study_name')}")
    if automl.get("evidence_class"):
        print(f"  evidence cls : {automl.get('evidence_class')}")
    oos_evidence = automl.get("oos_evidence") or {}
    if oos_evidence:
        print(
            "  oos evidence : "
            f"class={oos_evidence.get('class')}  "
            f"complete={bool(oos_evidence.get('evidence_stack_complete', False))}"
        )
        oos_reasons = list(oos_evidence.get("blocking_reasons") or [])
        print(f"  oos why      : {oos_reasons if oos_reasons else 'none'}")
    print(f"  objective    : {automl.get('objective')}")
    print(f"  selection    : {automl.get('selection_metric')} ({automl.get('selection_mode')})")
    print(f"  trials       : {automl.get('trial_count')}")
    print(f"  best value   : {_format_metric(automl.get('best_value'))}")
    if automl.get("best_value_raw") is not None:
        print(f"  raw value    : {_format_metric(automl.get('best_value_raw'))}")
    print(f"  best params  : {automl.get('best_params')}")
    if automl.get("best_params") is None:
        top_trials = list(automl.get("top_trials") or [])
        if top_trials:
            rejected = top_trials[0]
            print(
                "  best reject  : "
                f"raw={_format_metric(rejected.get('raw_value'))}  "
                f"selection={_format_metric(rejected.get('value'))}"
            )
            rejection_reasons = list((automl.get("selection_outcome") or {}).get("top_rejection_reasons") or [])
            if not rejection_reasons:
                rejection_reasons = list(((rejected.get("selection_policy") or {}).get("eligibility_reasons") or []))
            print(f"  reject why   : {rejection_reasons if rejection_reasons else 'none'}")
    validation_sources = automl.get("validation_sources") or {}
    if validation_sources:
        print(
            "  val sources  : "
            f"select={validation_sources.get('selection_metric_source')}  "
            f"diagnostic={validation_sources.get('diagnostic_metric_source')}  "
            f"tradable={validation_sources.get('tradable_metric_source')}"
        )
        print(f"  val gates    : {bool(validation_sources.get('all_required_sources_passed', True))}")

    trade_ready_profile = automl.get("trade_ready_profile") or {}
    if trade_ready_profile:
        print(
            "  power profile: "
            f"{trade_ready_profile.get('name')}  "
            f"reduced_power={bool(trade_ready_profile.get('reduced_power', False))}  "
            f"n_trials={trade_ready_profile.get('n_trials')}"
        )

    capital_evidence = automl.get("capital_evidence_contract") or {}
    if capital_evidence:
        print(
            "  capital mode : "
            f"requested={capital_evidence.get('requested_mode')}  "
            f"effective={capital_evidence.get('effective_mode')}  "
            f"eligible={bool(capital_evidence.get('capital_path_eligible', False))}"
        )
        capital_reasons = list(capital_evidence.get("blocking_reasons") or [])
        print(f"  capital why  : {capital_reasons if capital_reasons else 'none'}")

    post_selection = (automl.get("overfitting_diagnostics") or {}).get("post_selection") or {}
    if post_selection:
        print(
            "  post select  : "
            f"enabled={bool(post_selection.get('enabled', False))}  "
            f"passed={bool(post_selection.get('passed', True))}  "
            f"rule={post_selection.get('pass_rule')}"
        )
        white_report = post_selection.get("white_reality_check") or {}
        spa_report = post_selection.get("hansen_spa") or {}
        parts = []
        if white_report.get("enabled"):
            parts.append(f"white_rc_p={_format_metric(white_report.get('p_value'))}")
        if spa_report.get("enabled"):
            parts.append(f"spa_p={_format_metric(spa_report.get('p_value'))}")
        if parts:
            print(f"  post select p: {'  '.join(parts)}")
        if post_selection.get("reason"):
            print(f"  post select r: {post_selection.get('reason')}")

    best_training = automl.get("best_training", {})
    if best_training:
        print(
            "  best train   : "
            f"dir_acc={_format_metric(best_training.get('avg_directional_accuracy'))}  "
            f"acc={_format_metric(best_training.get('avg_accuracy'))}  "
            f"log_loss={_format_metric(best_training.get('avg_log_loss'))}"
        )
        lookahead_guard = best_training.get("lookahead_guard") or {}
        if lookahead_guard.get("enabled"):
            print(
                "  lookahead    : "
                f"passed={bool(lookahead_guard.get('promotion_pass', True))}  "
                f"mode={lookahead_guard.get('mode')}  "
                f"scope={lookahead_guard.get('audit_scope')}  "
                f"checked={lookahead_guard.get('checked_timestamps')}"
            )
            lookahead_reasons = list(lookahead_guard.get("reasons") or [])
            print(f"  lookahead why: {lookahead_reasons if lookahead_reasons else 'none'}")
        data_certification = best_training.get("data_certification") or {}
        if data_certification:
            print_data_certification_summary(data_certification)

    best_backtest = automl.get("best_backtest", {})
    if best_backtest:
        print(
            "  best backtest: "
            f"sharpe={_format_metric(best_backtest.get('sharpe_ratio'))}  "
            f"net={_format_metric(best_backtest.get('net_profit_pct'), percent=True)}  "
            f"mdd={_format_metric(best_backtest.get('max_drawdown'), percent=True)}  "
            f"trades={best_backtest.get('total_trades')}"
        )

    best_objective = automl.get("best_objective_diagnostics") or {}
    if best_objective:
        parts = [
            f"score={_format_metric(best_objective.get('raw_score'))}",
        ]
        if best_objective.get("primary_metric"):
            parts.append(
                f"{best_objective['primary_metric']}={_format_metric(best_objective['components'].get(best_objective['primary_metric']))}"
            )
        if best_objective.get("primary_metric_source"):
            parts.append(f"source={best_objective.get('primary_metric_source')}")
        if best_objective.get("benchmark_reference") not in (None, 0.0):
            parts.append(f"benchmark={_format_metric(best_objective.get('benchmark_reference'))}")
        print(f"  objective det: {'  '.join(parts)}")

        gates = best_objective.get("classification_gates") or {}
        if gates.get("enabled"):
            failed = gates.get("failed") or []
            reasons = gates.get("reasons") or []
            print(
                "  objective gate: "
                f"passed={gates.get('passed')}  "
                f"failed={failed if failed else 'none'}"
            )
            print(f"  objective why: {reasons if reasons else 'none'}")

    best_overfitting = automl.get("best_overfitting", {}).get("deflated_sharpe", {})
    if best_overfitting:
        print(
            "  best DSR     : "
            f"{_format_metric(best_overfitting.get('deflated_sharpe_ratio'))}  "
            f"obs_sr={_format_metric(best_overfitting.get('observed_sharpe_ratio'))}  "
            f"bench_sr={_format_metric(best_overfitting.get('benchmark_sharpe_ratio'))}"
        )

    diagnostics = automl.get("overfitting_diagnostics", {})
    pbo = diagnostics.get("pbo", {})
    if pbo.get("enabled"):
        print(
            "  CPCV PBO     : "
            f"pbo={_format_metric(pbo.get('probability_of_backtest_overfitting'))}  "
            f"splits={pbo.get('split_count')}  "
            f"lambda_med={_format_metric(pbo.get('lambda_median'))}"
        )
    elif pbo:
        print(f"  CPCV PBO     : disabled ({pbo.get('reason')})")

    validation_holdout = automl.get("validation_holdout") or {}
    if validation_holdout.get("enabled"):
        validation_backtest = validation_holdout.get("backtest") or {}
        print(
            "  validation   : "
            f"rows={validation_holdout.get('aligned_validation_rows')}  "
            f"range={validation_holdout.get('start_timestamp')} -> {validation_holdout.get('end_timestamp')}"
        )
        if validation_backtest:
            print(
                "  validation bt: "
                f"sharpe={_format_metric(validation_backtest.get('sharpe_ratio'))}  "
                f"net={_format_metric(validation_backtest.get('net_profit_pct'), percent=True)}  "
                f"trades={validation_backtest.get('total_trades')}"
            )

    locked_holdout = automl.get("locked_holdout") or {}
    if locked_holdout.get("enabled"):
        holdout_backtest = locked_holdout.get("backtest") or {}
        print(
            "  final holdout: "
            f"rows={locked_holdout.get('aligned_holdout_rows')}  "
            f"range={locked_holdout.get('start_timestamp')} -> {locked_holdout.get('end_timestamp')}"
        )
        if holdout_backtest:
            print(
                "  holdout bt   : "
                f"sharpe={_format_metric(holdout_backtest.get('sharpe_ratio'))}  "
                f"net={_format_metric(holdout_backtest.get('net_profit_pct'), percent=True)}  "
                f"trades={holdout_backtest.get('total_trades')}"
            )
        if locked_holdout.get("holdout_warning"):
            print("  holdout warn : Sharpe CI lower bound is below zero")

    evaluation_backtest = (locked_holdout.get("backtest") or validation_holdout.get("backtest") or best_backtest or {})
    if evaluation_backtest:
        stress_matrix = evaluation_backtest.get("stress_matrix") or {}
        execution_evidence = evaluation_backtest.get("execution_evidence") or {}
        funding_coverage = evaluation_backtest.get("funding_coverage_report") or {}
        print(
            "  evaluation   : "
            f"mode={evaluation_backtest.get('evaluation_mode', 'unknown')}  "
            f"stress_ready={bool(evaluation_backtest.get('stress_realism_ready', False))}  "
            f"stress_cases={stress_matrix.get('scenario_names') or 'none'}"
        )
        if execution_evidence:
            print(
                "  execution    : "
                f"class={execution_evidence.get('class')}  "
                f"mode={execution_evidence.get('execution_mode')}  "
                f"ready={bool(execution_evidence.get('promotion_execution_ready', False))}"
            )
            execution_reasons = list(execution_evidence.get("blocking_reasons") or [])
            print(f"  execution why: {execution_reasons if execution_reasons else 'none'}")
        if funding_coverage:
            print(
                "  funding cov  : "
                f"status={evaluation_backtest.get('funding_coverage_status', funding_coverage.get('coverage_status'))}  "
                f"missing={funding_coverage.get('missing_event_count', 0)}  "
                f"pass={bool(funding_coverage.get('promotion_pass', True))}"
            )
            funding_reasons = []
            if funding_coverage.get("coverage_reason"):
                funding_reasons.append(funding_coverage.get("coverage_reason"))
            if funding_coverage.get("fallback_assumption"):
                funding_reasons.append(funding_coverage.get("fallback_assumption"))
            print(f"  funding why  : {funding_reasons if funding_reasons else 'none'}")
        evaluation_significance = dict(evaluation_backtest.get("statistical_significance") or {})
        if evaluation_significance.get("reason"):
            print(
                "  eval stats   : "
                f"unavailable ({evaluation_significance.get('reason')})  "
                f"observations={evaluation_significance.get('observation_count')}/"
                f"{evaluation_significance.get('min_observations')}"
            )

    monitoring_report = (
        evaluation_backtest.get("operational_monitoring")
        or best_backtest.get("operational_monitoring")
        or best_training.get("operational_monitoring")
        or {}
    )
    if monitoring_report:
        monitoring_policy = dict(monitoring_report.get("policy") or {})
        monitoring_reasons = list(monitoring_report.get("reasons") or [])
        monitoring_gate = dict(monitoring_report.get("monitoring_gate_report") or {})
        print(
            "  monitoring   : "
            f"healthy={bool(monitoring_report.get('healthy', True))}  "
            f"profile={monitoring_policy.get('policy_profile', 'custom')}"
        )
        print(f"  monitoring why: {monitoring_reasons if monitoring_reasons else 'none'}")
        if monitoring_gate:
            print(
                "  op envelope : "
                f"pass={bool(monitoring_gate.get('promotion_pass', True))}  "
                f"missing={monitoring_gate.get('missing_metrics') or 'none'}"
            )
            gate_reasons = list(monitoring_gate.get("blocking_reasons") or [])
            print(f"  envelope why: {gate_reasons if gate_reasons else 'none'}")

    replication = automl.get("replication") or {}
    if replication.get("enabled"):
        print(
            "  replication  : "
            f"passed={bool(replication.get('promotion_pass', False))}  "
            f"coverage={replication.get('completed_cohort_count')}/{replication.get('requested_cohort_count')}  "
            f"pass_rate={_format_metric(replication.get('pass_rate'))}  "
            f"min_pass_rate={_format_metric(replication.get('min_pass_rate'))}"
        )
        replication_reasons = list(replication.get("reasons") or [])
        print(f"  replication why: {replication_reasons if replication_reasons else 'none'}")

    print(f"  promotion ok : {bool(automl.get('promotion_ready', False))}")
    promotion_reasons = list(automl.get("promotion_reasons") or [])
    print(f"  promotion why: {promotion_reasons if promotion_reasons else 'none'}")