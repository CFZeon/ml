"""Feature portability, admission, and retirement governance."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .features import ENDOGENOUS_FEATURE_FAMILIES, resolve_feature_family
from .regime import coerce_regime_label_series


_DEFAULT_ADMISSION_POLICY = {
    "enabled": True,
    "min_samples": 48,
    "rolling_window": 48,
    "rolling_step": 24,
    "min_rolling_sign_stability": 0.45,
    "min_regime_sign_stability": 0.34,
    "min_regime_rows": 12,
    "permutation_rounds": 8,
    "min_permutation_importance_gap": -0.01,
    "noise_scale": 0.05,
    "max_perturbation_sensitivity": 0.25,
    "retire_on_reject": True,
    "include_retired_features": False,
}


def _normalize_retired_registry(retired_features):
    if retired_features is None:
        return {}
    if isinstance(retired_features, dict):
        registry = {}
        for column, value in retired_features.items():
            if isinstance(value, dict):
                registry[column] = {
                    "status": value.get("status", "retired"),
                    "reason": value.get("reason", "preconfigured_retired"),
                }
            else:
                registry[column] = {
                    "status": "retired",
                    "reason": str(value or "preconfigured_retired"),
                }
        return registry
    return {
        column: {"status": "retired", "reason": "preconfigured_retired"}
        for column in retired_features
    }


def _infer_hypothesis_tag(column, family):
    lower = str(column or "").lower()
    if any(term in lower for term in ("trend", "momentum", "ret_", "return", "breakout")):
        return "trend"
    if any(term in lower for term in ("vol", "atr", "range", "dispersion", "sigma")):
        return "volatility"
    if any(term in lower for term in ("liquid", "volume", "turnover", "trade", "amihud")):
        return "liquidity"
    if lower.startswith(("ctx_", "ref_", "composite_", "fut_")):
        return "context"
    return family or "general"


def _resolve_portability(column, block_name, family):
    lower = str(column).lower()
    portability_class = "unknown"
    venue_scope = "generic"
    exchange_specific_semantics = False

    if family in ENDOGENOUS_FEATURE_FAMILIES:
        portability_class = "endogenous"
    elif lower.startswith(("composite_", "cross_venue_")):
        portability_class = "cross_venue_composite"
        venue_scope = "multi_venue"
    elif lower.startswith(("ref_", "reference_")) or block_name == "reference_overlay":
        portability_class = "reference_overlay"
        venue_scope = "multi_source_reference"
    elif family == "futures_context" or lower.startswith("fut_"):
        portability_class = "venue_specific"
        venue_scope = "binance_futures"
        exchange_specific_semantics = True
    elif family == "cross_asset":
        portability_class = "venue_specific"
        venue_scope = "binance_cross_asset"
        exchange_specific_semantics = True
    elif family == "custom_exogenous":
        portability_class = "reference_overlay"
        venue_scope = "custom_reference"

    return portability_class, venue_scope, exchange_specific_semantics


def derive_feature_metadata(
    feature_blocks=None,
    feature_families=None,
    columns=None,
    *,
    screening_report=None,
    feature_lineage=None,
    retired_features=None,
):
    feature_blocks = dict(feature_blocks or {})
    feature_families = dict(feature_families or {})
    feature_lineage = dict(feature_lineage or {})
    selected_columns = list(columns) if columns is not None else list(feature_blocks)
    screening_features = dict((screening_report or {}).get("features") or {})
    retired_registry = _normalize_retired_registry(retired_features)
    metadata = {}

    for column in selected_columns:
        block_name = feature_blocks.get(column, "unknown")
        family = feature_families.get(column, resolve_feature_family(block_name))
        portability_class, venue_scope, exchange_specific_semantics = _resolve_portability(column, block_name, family)
        lineage = dict(feature_lineage.get(column) or {})
        transform_chain = list(lineage.get("transform_chain") or ["raw"])
        screening_entry = screening_features.get(column) or {}
        selected_transform = screening_entry.get("selected_transform")
        if selected_transform and selected_transform != "passthrough" and selected_transform not in transform_chain:
            transform_chain.append(selected_transform)
        retirement_entry = retired_registry.get(column, {})
        retirement_status = retirement_entry.get("status", "active")

        metadata[column] = {
            "column": column,
            "block": block_name,
            "family": family,
            "venue_scope": venue_scope,
            "portability_class": portability_class,
            "exchange_specific_semantics": bool(exchange_specific_semantics),
            "hypothesis_tag": _infer_hypothesis_tag(column, family),
            "source_lineage": {
                "source_column": lineage.get("source_column", column),
                "block": lineage.get("block", block_name),
                "family": family,
                "venue_scope": venue_scope,
            },
            "transform_chain": transform_chain,
            "availability": dict(lineage.get("availability") or {}),
            "retirement_status": retirement_status,
            "retired": retirement_status == "retired",
            "retirement_reason": retirement_entry.get("reason"),
        }

    return metadata


def filter_feature_metadata(feature_metadata, columns):
    selected = set(columns)
    return {
        column: dict(metadata)
        for column, metadata in dict(feature_metadata or {}).items()
        if column in selected
    }


def summarize_feature_portability(feature_metadata):
    rows = pd.DataFrame(list(dict(feature_metadata or {}).values()))
    if rows.empty:
        return {
            "feature_count": 0,
            "portability_counts": {},
            "venue_scope_counts": {},
            "exchange_specific_features": 0,
        }

    portability_counts = rows["portability_class"].value_counts().sort_index().to_dict()
    venue_scope_counts = rows["venue_scope"].value_counts().sort_index().to_dict()
    return {
        "feature_count": int(len(rows)),
        "portability_counts": {key: int(value) for key, value in portability_counts.items()},
        "venue_scope_counts": {key: int(value) for key, value in venue_scope_counts.items()},
        "exchange_specific_features": int(rows["exchange_specific_semantics"].sum()),
    }


def evaluate_feature_portability(feature_metadata, top_features=None, family_diagnostics=None, config=None):
    config = dict(config or {})
    top_features = list(top_features or [])
    family_diagnostics = dict(family_diagnostics or {})
    metadata = dict(feature_metadata or {})
    summary = summarize_feature_portability(metadata)

    importance_key = "avg_native_importance"
    total_importance = 0.0
    venue_specific_importance = 0.0
    venue_specific_top_features = 0
    normalized_top_features = []

    for row in top_features:
        feature = row.get("feature")
        importance = float(row.get(importance_key, row.get("native_importance", 0.0)) or 0.0)
        portability = metadata.get(feature, {}).get("portability_class", "unknown")
        total_importance += importance
        if portability == "venue_specific":
            venue_specific_importance += importance
            venue_specific_top_features += 1
        normalized_top_features.append({
            **row,
            "portability_class": portability,
        })

    venue_specific_importance_share = venue_specific_importance / total_importance if total_importance > 0.0 else 0.0
    venue_specific_top_feature_share = venue_specific_top_features / len(top_features) if top_features else 0.0

    bundle_map = {
        row.get("bundle"): row
        for row in family_diagnostics.get("bundles", [])
    }
    endogenous_bundle = bundle_map.get("endogenous_only")
    full_bundle = bundle_map.get("full_context")
    endogenous_replication_gap = None
    if endogenous_bundle is not None and full_bundle is not None:
        endogenous_replication_gap = {
            "accuracy_drop": float(endogenous_bundle.get("avg_accuracy_drop_vs_full", 0.0)),
            "f1_drop": float(endogenous_bundle.get("avg_f1_drop_vs_full", 0.0)),
        }

    max_importance_share = float(config.get("max_venue_specific_importance_share", 0.5))
    max_top_feature_share = float(config.get("max_venue_specific_top_feature_share", 0.5))
    max_ablation_accuracy_drop = float(config.get("max_endogenous_accuracy_drop", 0.02))
    max_ablation_f1_drop = float(config.get("max_endogenous_f1_drop", 0.02))

    reasons = []
    if venue_specific_importance_share > max_importance_share:
        reasons.append("venue_specific_importance_dominates")
    if venue_specific_top_feature_share > max_top_feature_share:
        reasons.append("venue_specific_top_features_dominates")
    if endogenous_replication_gap is not None and (
        endogenous_replication_gap["accuracy_drop"] > max_ablation_accuracy_drop
        or endogenous_replication_gap["f1_drop"] > max_ablation_f1_drop
    ):
        reasons.append("endogenous_ablation_failed")

    promotion_pass = not reasons
    return {
        "promotion_pass": promotion_pass,
        "reasons": reasons,
        "summary": summary,
        "venue_specific_importance_share": round(float(venue_specific_importance_share), 6),
        "venue_specific_top_feature_share": round(float(venue_specific_top_feature_share), 6),
        "endogenous_replication_gap": endogenous_replication_gap,
        "top_features": normalized_top_features,
    }


def _resolve_feature_admission_config(config=None):
    resolved = dict(_DEFAULT_ADMISSION_POLICY)
    if config:
        resolved.update(dict(config))
    return resolved


def _safe_correlation(left, right):
    aligned = pd.concat([pd.Series(left), pd.Series(right)], axis=1).dropna()
    if len(aligned) < 3:
        return None
    if aligned.iloc[:, 0].nunique() <= 1 or aligned.iloc[:, 1].nunique() <= 1:
        return None
    value = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _rolling_sign_stability(series, target, window, step):
    base_corr = _safe_correlation(series, target)
    if base_corr is None:
        return None
    overall_sign = math.copysign(1.0, base_corr) if abs(base_corr) > 1e-9 else 0.0
    checks = []
    for start in range(0, max(1, len(series) - window + 1), max(1, step)):
        stop = start + window
        window_corr = _safe_correlation(series.iloc[start:stop], target.iloc[start:stop])
        if window_corr is None:
            continue
        window_sign = math.copysign(1.0, window_corr) if abs(window_corr) > 1e-9 else 0.0
        checks.append(float(window_sign == overall_sign))
    if not checks:
        return None
    return float(np.mean(checks))


def _regime_sign_stability(series, target, regime_data, min_rows):
    if regime_data is None:
        return None
    regimes = coerce_regime_label_series(regime_data).reindex(series.index)
    base_corr = _safe_correlation(series, target)
    if base_corr is None:
        return None
    overall_sign = math.copysign(1.0, base_corr) if abs(base_corr) > 1e-9 else 0.0
    checks = []
    for label in pd.Series(regimes).dropna().unique():
        mask = regimes == label
        if int(mask.sum()) < int(min_rows):
            continue
        regime_corr = _safe_correlation(series.loc[mask], target.loc[mask])
        if regime_corr is None:
            continue
        regime_sign = math.copysign(1.0, regime_corr) if abs(regime_corr) > 1e-9 else 0.0
        checks.append(float(regime_sign == overall_sign))
    if not checks:
        return None
    return float(np.mean(checks))


def _permutation_importance_gap(series, target, rounds):
    base_corr = _safe_correlation(series, target)
    if base_corr is None:
        return None
    rng = np.random.default_rng(42)
    shuffled_scores = []
    target_array = pd.Series(target).to_numpy(copy=True)
    for _ in range(max(1, int(rounds))):
        permuted = rng.permutation(target_array)
        score = _safe_correlation(series, pd.Series(permuted, index=series.index))
        shuffled_scores.append(abs(score or 0.0))
    return float(abs(base_corr) - float(np.median(shuffled_scores)))


def _perturbation_sensitivity(series, target, noise_scale):
    base_corr = _safe_correlation(series, target)
    if base_corr is None:
        return None
    clean = pd.Series(series).astype(float)
    std = float(clean.std())
    if not np.isfinite(std) or std <= 0.0:
        return 0.0
    rng = np.random.default_rng(7)
    perturbed = clean + rng.normal(0.0, std * float(noise_scale), len(clean))
    perturbed_corr = _safe_correlation(perturbed, target)
    if perturbed_corr is None:
        return None
    return float(abs(perturbed_corr - base_corr))


def _compute_feature_robustness(series, target, regime_data=None, config=None):
    policy = _resolve_feature_admission_config(config)
    aligned = pd.concat([pd.Series(series), pd.Series(target)], axis=1).dropna()
    if len(aligned) < int(policy["min_samples"]):
        return {
            "base_correlation": _safe_correlation(series, target),
            "rolling_sign_stability": None,
            "regime_sign_stability": None,
            "permutation_importance_gap": None,
            "perturbation_sensitivity": None,
            "sample_count": int(len(aligned)),
            "insufficient_samples": True,
        }

    aligned_series = aligned.iloc[:, 0]
    aligned_target = aligned.iloc[:, 1]
    window = min(int(policy["rolling_window"]), len(aligned_series))
    step = max(1, int(policy["rolling_step"]))
    aligned_regimes = None
    if regime_data is not None:
        aligned_regimes = coerce_regime_label_series(regime_data).reindex(aligned.index)

    return {
        "base_correlation": _safe_correlation(aligned_series, aligned_target),
        "rolling_sign_stability": _rolling_sign_stability(aligned_series, aligned_target, window=window, step=step),
        "regime_sign_stability": _regime_sign_stability(
            aligned_series,
            aligned_target,
            aligned_regimes,
            min_rows=policy["min_regime_rows"],
        ),
        "permutation_importance_gap": _permutation_importance_gap(
            aligned_series,
            aligned_target,
            rounds=policy["permutation_rounds"],
        ),
        "perturbation_sensitivity": _perturbation_sensitivity(
            aligned_series,
            aligned_target,
            noise_scale=policy["noise_scale"],
        ),
        "sample_count": int(len(aligned)),
        "insufficient_samples": False,
    }


def apply_feature_retirement(features, feature_blocks=None, feature_families=None, feature_metadata=None, config=None):
    policy = _resolve_feature_admission_config(config)
    metadata = dict(feature_metadata or {})
    registry = _normalize_retired_registry(policy.get("retired_features"))

    if policy.get("include_retired_features", False):
        return features.copy(), dict(feature_blocks or {}), dict(feature_families or {}), metadata, {
            "applied": False,
            "dropped_columns": [],
            "retired_registry": registry,
        }

    retired_columns = {
        column for column, info in metadata.items()
        if info.get("retired")
    }
    retired_columns.update(column for column in registry if column in features.columns)
    kept_columns = [column for column in features.columns if column not in retired_columns]
    filtered_metadata = filter_feature_metadata(metadata, kept_columns)
    filtered_blocks = {
        column: block
        for column, block in dict(feature_blocks or {}).items()
        if column in kept_columns
    }
    filtered_families = {
        column: family
        for column, family in dict(feature_families or {}).items()
        if column in kept_columns
    }
    return (
        features.loc[:, kept_columns].copy(),
        filtered_blocks,
        filtered_families,
        filtered_metadata,
        {
            "applied": bool(retired_columns),
            "dropped_columns": sorted(retired_columns),
            "retired_registry": registry,
        },
    )


def evaluate_feature_admission(features, target, feature_metadata=None, regime_data=None, config=None, candidate_order=None):
    policy = _resolve_feature_admission_config(config)
    metadata = {
        column: dict(value)
        for column, value in dict(feature_metadata or {}).items()
    }
    candidate_columns = [column for column in (candidate_order or list(features.columns)) if column in features.columns]

    if not policy.get("enabled", True):
        for column in candidate_columns:
            metadata.setdefault(column, {"column": column})["admission"] = {
                "decision": "admitted",
                "reasons": [],
                "diagnostics": {},
            }
        return {
            "enabled": False,
            "admitted_columns": candidate_columns,
            "rejected_columns": [],
            "retired_columns": [],
            "feature_metadata": metadata,
            "feature_reports": {},
            "summary": {
                "input_features": int(len(candidate_columns)),
                "admitted_features": int(len(candidate_columns)),
                "rejected_features": 0,
                "retired_features": 0,
            },
            "promotion_pass": True,
        }

    registry = _normalize_retired_registry(policy.get("retired_features"))
    target_series = pd.Series(target).reindex(features.index)
    admitted_columns = []
    rejected_columns = []
    retired_columns = []
    feature_reports = {}

    for column in candidate_columns:
        metadata.setdefault(column, {"column": column})
        meta = metadata[column]
        reasons = []

        retirement_entry = registry.get(column)
        if retirement_entry and not policy.get("include_retired_features", False):
            reasons.append("retired_feature")
            meta["retirement_status"] = retirement_entry.get("status", "retired")
            meta["retired"] = True
            meta["retirement_reason"] = retirement_entry.get("reason", "preconfigured_retired")
            diagnostics = {}
        else:
            diagnostics = _compute_feature_robustness(
                features[column],
                target_series,
                regime_data=regime_data,
                config=policy,
            )
            if not diagnostics.get("insufficient_samples"):
                rolling_stability = diagnostics.get("rolling_sign_stability")
                regime_stability = diagnostics.get("regime_sign_stability")
                permutation_gap = diagnostics.get("permutation_importance_gap")
                perturbation_sensitivity = diagnostics.get("perturbation_sensitivity")

                if rolling_stability is not None and rolling_stability < float(policy["min_rolling_sign_stability"]):
                    reasons.append("rolling_sign_stability_failed")
                if regime_stability is not None and regime_stability < float(policy["min_regime_sign_stability"]):
                    reasons.append("leave_one_regime_out_failed")
                if permutation_gap is not None and permutation_gap < float(policy["min_permutation_importance_gap"]):
                    reasons.append("permutation_sanity_failed")
                if perturbation_sensitivity is not None and perturbation_sensitivity > float(policy["max_perturbation_sensitivity"]):
                    reasons.append("perturbation_sensitivity_failed")

        decision = "admitted" if not reasons else "rejected"
        meta["admission"] = {
            "decision": decision,
            "reasons": reasons,
            "diagnostics": diagnostics,
        }
        if decision == "admitted":
            admitted_columns.append(column)
            if meta.get("retirement_status") is None:
                meta["retirement_status"] = "active"
        else:
            rejected_columns.append(column)
            if policy.get("retire_on_reject", True):
                meta["retired"] = True
                meta["retirement_status"] = "retired"
                if meta.get("retirement_reason") is None:
                    meta["retirement_reason"] = "feature_admission_failed"
                retired_columns.append(column)

        feature_reports[column] = meta["admission"]

    summary = {
        "input_features": int(len(candidate_columns)),
        "admitted_features": int(len(admitted_columns)),
        "rejected_features": int(len(rejected_columns)),
        "retired_features": int(len(set(retired_columns))),
    }
    promotion_pass = bool(admitted_columns)
    return {
        "enabled": True,
        "admitted_columns": admitted_columns,
        "rejected_columns": rejected_columns,
        "retired_columns": sorted(set(retired_columns)),
        "feature_metadata": metadata,
        "feature_reports": feature_reports,
        "summary": summary,
        "promotion_pass": promotion_pass,
    }


def summarize_feature_admission_reports(reports):
    rows = [dict(row or {}) for row in (reports or [])]
    if not rows:
        return {
            "fold_count": 0,
            "avg_admitted_features": 0.0,
            "avg_rejected_features": 0.0,
            "retired_features": [],
            "promotion_pass": True,
        }

    retired = sorted({column for row in rows for column in row.get("retired_columns", [])})
    return {
        "fold_count": int(len(rows)),
        "avg_admitted_features": float(np.mean([row.get("summary", {}).get("admitted_features", 0) for row in rows])),
        "avg_rejected_features": float(np.mean([row.get("summary", {}).get("rejected_features", 0) for row in rows])),
        "retired_features": retired,
        "promotion_pass": all(bool(row.get("promotion_pass", True)) for row in rows),
    }


__all__ = [
    "apply_feature_retirement",
    "derive_feature_metadata",
    "evaluate_feature_admission",
    "evaluate_feature_portability",
    "filter_feature_metadata",
    "summarize_feature_admission_reports",
    "summarize_feature_portability",
]