"""Feature portability metadata and promotion gates."""

from __future__ import annotations

import pandas as pd

from .features import ENDOGENOUS_FEATURE_FAMILIES, resolve_feature_family


def derive_feature_metadata(feature_blocks=None, feature_families=None, columns=None):
    feature_blocks = dict(feature_blocks or {})
    feature_families = dict(feature_families or {})
    selected_columns = list(columns) if columns is not None else list(feature_blocks)
    metadata = {}

    for column in selected_columns:
        block_name = feature_blocks.get(column, "unknown")
        family = feature_families.get(column, resolve_feature_family(block_name))
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

        metadata[column] = {
            "column": column,
            "block": block_name,
            "family": family,
            "venue_scope": venue_scope,
            "portability_class": portability_class,
            "exchange_specific_semantics": bool(exchange_specific_semantics),
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


__all__ = [
    "derive_feature_metadata",
    "evaluate_feature_portability",
    "filter_feature_metadata",
    "summarize_feature_portability",
]