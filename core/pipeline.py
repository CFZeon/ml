"""Stepwise research pipeline abstraction for reusable model workflows."""

import copy
import time
import warnings

import numpy as np
import pandas as pd

from .automl import run_automl_study
from .backtest import kelly_fraction, run_backtest
from .context import (
    _normalize_funding_timestamp_index,
    _resolve_context_missing_policy,
    build_cross_asset_context_feature_block,
    build_futures_context_feature_block,
    build_multi_timeframe_context_feature_block,
    fetch_binance_futures_context,
    fetch_context_symbol_bars,
)
from .data import (
    fetch_binance_bars,
    fetch_binance_futures_contract_spec,
    fetch_binance_symbol_filters,
    join_custom_data,
    load_futures_leverage_brackets,
)
from .data_contracts import (
    build_dataset_bundle_manifest,
    validate_futures_context_bundle,
    validate_market_context_frames,
    validate_market_frame_contract,
    validate_reference_overlay_frame_contract,
)
from .data_quality import check_data_quality
from .execution import resolve_execution_policy
from .feature_governance import (
    apply_feature_retirement,
    derive_feature_metadata,
    evaluate_feature_admission,
    evaluate_feature_portability,
    filter_feature_metadata,
    summarize_feature_admission_reports,
)
from .features import (
    build_feature_set,
    check_stationarity,
    derive_feature_families,
    screen_features_for_stationarity,
    select_features,
    summarize_feature_families,
)
from .indicators import run_indicators
from .labeling import (
    fixed_horizon_labels,
    sample_weights_by_uniqueness,
    trend_scanning_labels,
    triple_barrier_labels,
)
from .models import (
    apply_binary_probability_calibrator,
    build_execution_outcome_frame,
    cpcv_split,
    ConstantProbabilityModel,
    build_meta_feature_frame,
    compute_feature_block_diagnostics,
    compute_feature_family_diagnostics,
    evaluate_model,
    fit_binary_probability_calibrator,
    predict_probability_frame,
    summarize_feature_block_diagnostics,
    summarize_feature_family_diagnostics,
    train_meta_model,
    train_model,
    walk_forward_split,
)
from .monitoring import build_monitoring_report, write_monitoring_artifacts
from .orchestration import run_drift_retraining_cycle as orchestrate_drift_retraining_cycle
from .regime import (
    build_default_regime_feature_set,
    build_regime_ablation_report,
    detect_regime,
    normalize_regime_feature_set,
    summarize_regime_ablation_reports,
)
from .slippage import (
    DepthCurveImpactModel,
    FillAwareCostModel,
    FlatSlippageModel,
    OrderBookImpactModel,
    ProxyImpactModel,
    SquareRootImpactModel,
)
from .reference_data import build_reference_overlay_feature_block, build_reference_validation_bundle
from .signal_decay import build_signal_decay_report
from .universe import (
    build_symbol_lifecycle_frame,
    evaluate_universe_eligibility,
    load_historical_universe_snapshot,
)


def _default_regime_features(pipeline):
    data = pipeline.require("data")
    features_config = pipeline.section("features")
    reference_data = (
        pipeline.state.get("reference_overlay_data")
        if pipeline.state.get("reference_overlay_data") is not None
        else pipeline.state.get("reference_data")
    )
    return build_default_regime_feature_set(
        data,
        base_interval=pipeline.section("data").get("interval", "1h"),
        rolling_window=features_config.get("rolling_window", 20),
        futures_context=pipeline.state.get("futures_context"),
        cross_asset_context=pipeline.state.get("cross_asset_context"),
        reference_data=reference_data,
        context_timeframes=features_config.get("context_timeframes"),
    )


def _default_stationarity_specs(pipeline):
    specs = [{"name": "close", "source": "data", "column": "close"}]
    features = pipeline.state.get("features")
    if features is not None and "close_fracdiff" in features.columns:
        specs.append({"name": "close_fracdiff", "source": "features", "column": "close_fracdiff"})
    return specs


def _manifest_payload(manifest):
    if hasattr(manifest, "to_dict"):
        return manifest.to_dict()
    return dict(manifest or {})


def _extract_frame_manifest_mapping(frames):
    manifests = {}
    for key, frame in dict(frames or {}).items():
        manifest = dict(getattr(frame, "attrs", {}).get("dataset_manifest") or {})
        if manifest:
            manifests[key] = manifest
    return manifests


def _ensure_data_lineage_state(pipeline):
    lineage = dict(pipeline.state.get("data_lineage") or {})
    lineage.setdefault("source_groups", {})
    lineage.setdefault("source_datasets", [])
    lineage.setdefault("active_datasets", {})
    pipeline.state["data_lineage"] = lineage
    return lineage


def _sync_source_datasets(lineage):
    source_datasets = []
    for value in (lineage.get("source_groups") or {}).values():
        if isinstance(value, list):
            source_datasets.extend(_manifest_payload(item) for item in value if item)
            continue
        if isinstance(value, dict) and value.get("contract"):
            source_datasets.append(_manifest_payload(value))
            continue
        if isinstance(value, dict):
            source_datasets.extend(_manifest_payload(item) for item in value.values() if item)
    lineage["source_datasets"] = source_datasets
    return lineage


def _set_lineage_group(pipeline, group_name, manifests):
    if not manifests:
        return _ensure_data_lineage_state(pipeline)

    lineage = _ensure_data_lineage_state(pipeline)
    if isinstance(manifests, list):
        lineage["source_groups"][group_name] = [_manifest_payload(item) for item in manifests if item]
    elif isinstance(manifests, dict) and manifests.get("contract"):
        lineage["source_groups"][group_name] = _manifest_payload(manifests)
    elif isinstance(manifests, dict):
        lineage["source_groups"][group_name] = {
            key: _manifest_payload(value)
            for key, value in manifests.items()
            if value
        }
    return _sync_source_datasets(lineage)


def _refresh_active_dataset_manifests(pipeline, *, data_quality_report=None):
    lineage = _ensure_data_lineage_state(pipeline)
    source_groups = lineage.get("source_groups") or {}
    source_datasets = list(lineage.get("source_datasets") or [])
    market_manifest = source_groups.get("market_data")
    raw_upstreams = [market_manifest] if isinstance(market_manifest, dict) and market_manifest.get("contract") else source_datasets

    validation = {"status": "pass"}
    if data_quality_report is not None:
        validation = {
            "status": data_quality_report.get("status", "pass"),
            "blocking": bool(data_quality_report.get("blocking", False)),
            "data_quality_report": data_quality_report,
        }

    raw_data = pipeline.state.get("raw_data")
    if raw_data is not None:
        lineage["active_datasets"]["raw_data"] = build_dataset_bundle_manifest(
            raw_data,
            name="raw_data",
            upstream_manifests=raw_upstreams,
            validation=validation,
            source={
                "bundle_role": "raw_data",
                "integrity_status": (pipeline.state.get("data_integrity_report") or {}).get("status"),
            },
        )

    data = pipeline.state.get("data")
    if data is not None:
        lineage["active_datasets"]["data"] = build_dataset_bundle_manifest(
            data,
            name="data",
            upstream_manifests=source_datasets,
            validation=validation,
            source={"bundle_role": "data"},
        )

    pipeline.state["data_lineage"] = lineage
    return lineage


def _ensure_pipeline_data_contracts(pipeline, *, include_reference=False):
    config = dict(pipeline.section("data") or {})
    market = config.get("market", "spot")
    interval = config.get("interval", "1h")
    symbol = config.get("symbol", "unknown")

    raw_data = pipeline.state.get("raw_data")
    if raw_data is not None and not dict(getattr(raw_data, "attrs", {}).get("dataset_manifest") or {}):
        validated_raw, market_manifest = validate_market_frame_contract(
            raw_data,
            market=market,
            dataset_name=f"pipeline_{market}_{str(symbol).lower()}_{interval}_bars",
            source={
                "source_name": "pipeline_state",
                "symbol": symbol,
                "market": market,
                "interval": interval,
            },
        )
        pipeline.state["raw_data"] = validated_raw
        _set_lineage_group(pipeline, "market_data", market_manifest)
    elif raw_data is not None:
        _set_lineage_group(
            pipeline,
            "market_data",
            dict(getattr(raw_data, "attrs", {}).get("dataset_manifest") or {}),
        )

    custom_reports = list(pipeline.state.get("custom_data_report") or [])
    custom_manifests = [dict(report.get("dataset_manifest") or {}) for report in custom_reports if report.get("dataset_manifest")]
    if custom_manifests:
        _set_lineage_group(pipeline, "custom_data", custom_manifests)

    futures_context = pipeline.state.get("futures_context")
    if futures_context:
        futures_manifests = _extract_frame_manifest_mapping(futures_context)
        if not futures_manifests:
            validated_context, futures_manifests = validate_futures_context_bundle(
                futures_context,
                symbol=symbol,
                interval=interval,
                source_name="pipeline_state_futures_context",
            )
            pipeline.state["futures_context"] = validated_context
        _set_lineage_group(pipeline, "futures_context", futures_manifests)

    cross_asset_context = pipeline.state.get("cross_asset_context")
    if cross_asset_context:
        cross_asset_manifests = _extract_frame_manifest_mapping(cross_asset_context)
        if not cross_asset_manifests:
            validated_context, cross_asset_manifests = validate_market_context_frames(
                cross_asset_context,
                market=config.get("cross_asset_context", {}).get("market", market),
                interval=interval,
                group_name="cross_asset_context",
            )
            pipeline.state["cross_asset_context"] = validated_context
        _set_lineage_group(pipeline, "cross_asset_context", cross_asset_manifests)

    if include_reference:
        reference_key = "reference_overlay_data" if pipeline.state.get("reference_overlay_data") is not None else "reference_data"
        reference_frame = pipeline.state.get(reference_key)
        if reference_frame is not None:
            reference_manifest = dict(getattr(reference_frame, "attrs", {}).get("dataset_manifest") or {})
            if not reference_manifest:
                validated_reference, reference_manifest = validate_reference_overlay_frame_contract(
                    reference_frame,
                    dataset_name=reference_key,
                    source={"source_name": "pipeline_state_reference_data", "symbol": symbol, "interval": interval},
                )
                pipeline.state[reference_key] = validated_reference
            _set_lineage_group(pipeline, "reference_data", reference_manifest)

    return _refresh_active_dataset_manifests(
        pipeline,
        data_quality_report=pipeline.state.get("data_quality_report"),
    )


def _positive_class_probability(model, X, positive_class=1):
    probabilities = model.predict_proba(X)
    classes = getattr(model, "classes_", None)

    if classes is None and hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model")
        classes = getattr(estimator, "classes_", None)

    if probabilities.ndim == 1:
        return probabilities

    if probabilities.shape[1] == 1:
        if classes is not None and len(classes) == 1 and classes[0] == positive_class:
            return np.ones(len(X))
        return np.zeros(len(X))

    if classes is not None and positive_class in classes:
        positive_index = list(classes).index(positive_class)
        return probabilities[:, positive_index]

    return probabilities[:, -1]


def _resolve_signal_holding_bars(pipeline, signal_config):
    configured = signal_config.get("holding_bars")
    if configured is not None:
        return max(1, int(configured))

    label_config = pipeline.section("labels")
    if label_config.get("kind") == "fixed_horizon":
        return max(1, int(label_config.get("horizon", 1)))
    return max(1, int(label_config.get("max_holding", 1)))


def _resolve_validation_method(model_config):
    method = str(model_config.get("cv_method", "cpcv")).lower()
    aliases = {
        "walk-forward": "walk_forward",
        "walkforward": "walk_forward",
        "wf": "walk_forward",
    }
    method = aliases.get(method, method)
    if method not in {"cpcv", "walk_forward"}:
        raise ValueError(f"Unknown validation method={method!r}. Choose from ['cpcv', 'walk_forward']")
    return method


def _resolve_cpcv_block_count(model_config):
    configured = model_config.get("n_blocks")
    if configured is not None:
        return max(2, int(configured))
    return max(2, int(model_config.get("n_splits", 3)) + 1)


def _resolve_cpcv_test_block_count(model_config):
    n_blocks = _resolve_cpcv_block_count(model_config)
    configured = model_config.get("test_blocks")
    if configured is None:
        return max(1, n_blocks // 2)
    return max(1, min(int(configured), n_blocks - 1))


def _resolve_cpcv_embargo_bars(pipeline, model_config):
    configured = model_config.get("embargo_bars")
    if configured is not None:
        return max(0, int(configured))

    label_config = pipeline.section("labels")
    for key in ("max_holding", "horizon"):
        value = label_config.get(key)
        if value is not None:
            return max(0, int(value))

    return max(0, int(model_config.get("gap", 0)))


def _build_contiguous_test_intervals(index, positions):
    if index is None or len(index) == 0 or positions is None or len(positions) == 0:
        return []

    sorted_positions = np.sort(np.asarray(positions, dtype=int))
    intervals = []
    run_start = sorted_positions[0]
    run_end = sorted_positions[0]

    for position in sorted_positions[1:]:
        if int(position) != int(run_end) + 1:
            intervals.append((index[int(run_start)], index[int(run_end)]))
            run_start = position
        run_end = position

    intervals.append((index[int(run_start)], index[int(run_end)]))
    return intervals


def _iter_validation_splits(pipeline, X):
    model_config = pipeline.section("model")
    validation_method = _resolve_validation_method(model_config)

    if validation_method == "walk_forward":
        for split_number, (train_idx, test_idx) in enumerate(
            walk_forward_split(
                X,
                n_splits=model_config.get("n_splits", 3),
                train_size=model_config.get("train_size"),
                test_size=model_config.get("test_size"),
                gap=model_config.get("gap", 0),
                expanding=model_config.get("expanding", False),
            )
        ):
            yield {
                "fold": split_number,
                "split_id": f"walk_forward_{split_number}",
                "validation_method": validation_method,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "test_intervals": _build_contiguous_test_intervals(X.index, test_idx),
                "metadata": {
                    "gap": int(model_config.get("gap", 0)),
                    "expanding": bool(model_config.get("expanding", False)),
                },
            }
        return

    n_blocks = _resolve_cpcv_block_count(model_config)
    embargo_bars = _resolve_cpcv_embargo_bars(pipeline, model_config)
    test_blocks = _resolve_cpcv_test_block_count(model_config)

    for split_number, (train_idx, test_idx, metadata) in enumerate(
        cpcv_split(
            X,
            n_blocks=n_blocks,
            test_blocks=test_blocks,
            embargo=embargo_bars,
        )
    ):
        split_metadata = dict(metadata)
        split_metadata["embargo_bars"] = int(embargo_bars)
        yield {
            "fold": split_number,
            "split_id": split_metadata.get("split_id", f"cpcv_{split_number}"),
            "validation_method": validation_method,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "test_intervals": _build_contiguous_test_intervals(X.index, test_idx),
            "metadata": split_metadata,
        }


def _resolve_executable_validation_training_config(pipeline, model_config):
    executable_validation = copy.deepcopy(model_config.get("executable_validation") or {})
    enabled = executable_validation.get("enabled")
    if enabled is None:
        enabled = _resolve_validation_method(model_config) == "cpcv"
    if not enabled or bool(model_config.get("_skip_executable_validation", False)):
        return {"enabled": False}

    method = _resolve_validation_method({"cv_method": executable_validation.get("method", "walk_forward")})
    if method != "walk_forward":
        raise ValueError("model.executable_validation.method must resolve to 'walk_forward'")

    replay_model_config = copy.deepcopy(model_config)
    configured_splits = executable_validation.get("n_splits", model_config.get("n_splits"))
    if configured_splits is None:
        configured_splits = max(1, _resolve_cpcv_block_count(model_config) - 1)

    replay_model_config["cv_method"] = method
    replay_model_config["n_splits"] = max(1, int(configured_splits))
    replay_model_config["train_size"] = executable_validation.get("train_size", model_config.get("train_size"))
    replay_model_config["test_size"] = executable_validation.get("test_size", model_config.get("test_size"))
    replay_model_config["gap"] = max(
        0,
        int(
            executable_validation.get(
                "gap",
                model_config.get("gap", _resolve_cpcv_embargo_bars(pipeline, model_config)),
            )
        ),
    )
    replay_model_config["expanding"] = bool(
        executable_validation.get("expanding", model_config.get("expanding", False))
    )
    replay_model_config["_skip_executable_validation"] = True
    replay_model_config["executable_validation"] = {**executable_validation, "enabled": False}
    return {
        "enabled": True,
        "source": "rolling_walk_forward_replay",
        "model_config": replay_model_config,
    }


def _build_executable_validation_training(pipeline):
    model_config = pipeline.section("model")
    executable_validation_config = _resolve_executable_validation_training_config(pipeline, model_config)
    if not executable_validation_config.get("enabled", False):
        return None

    clone_config = copy.deepcopy(pipeline.config)
    clone_config["model"] = executable_validation_config["model_config"]
    monitoring_config = dict(clone_config.get("monitoring") or {})
    monitoring_config["write_reports"] = False
    clone_config["monitoring"] = monitoring_config

    clone = ResearchPipeline(clone_config, steps=[TrainModelsStep])
    clone.state = {
        key: value
        for key, value in pipeline.state.items()
        if key not in {"training", "signals", "backtest", "operational_monitoring"}
    }
    replay_training = clone.run_step("train_models")
    return {
        "enabled": True,
        "source": executable_validation_config["source"],
        "reporting_role": "primary",
        "selection_role": "primary",
        "training": replay_training,
    }


def _resolve_fold_calibration_policy(validation_method):
    if validation_method == "cpcv":
        return {
            "policy_name": "validation_only_or_defaults",
            "allow_cross_fold_borrowing": False,
        }
    return {
        "policy_name": "strictly_earlier_oos_only",
        "allow_cross_fold_borrowing": True,
    }


def _select_causal_prior_trade_outcomes(oos_trade_outcomes, validation_method, calibration_cutoff_timestamp):
    policy = _resolve_fold_calibration_policy(validation_method)
    pooled_outcomes = pd.concat(oos_trade_outcomes).sort_index() if oos_trade_outcomes else pd.DataFrame()
    pooled_row_count = int(len(pooled_outcomes))

    if pooled_outcomes.empty or calibration_cutoff_timestamp is None:
        causal_outcomes = pooled_outcomes.iloc[0:0]
    elif policy["allow_cross_fold_borrowing"]:
        causal_outcomes = pooled_outcomes.loc[pooled_outcomes.index < calibration_cutoff_timestamp]
    else:
        causal_outcomes = pooled_outcomes.iloc[0:0]

    return causal_outcomes, {
        **policy,
        "causal_cutoff_timestamp": calibration_cutoff_timestamp,
        "candidate_trade_rows": pooled_row_count,
        "causal_trade_rows": int(len(causal_outcomes)),
        "dropped_trade_rows": int(pooled_row_count - len(causal_outcomes)),
    }


def _apply_holding_period(event_weights, holding_bars):
    weights = pd.Series(event_weights, copy=False).astype(float)
    if holding_bars <= 1 or weights.empty:
        return weights.clip(-1.0, 1.0)

    held = pd.Series(0.0, index=weights.index)
    for lag in range(holding_bars):
        shifted = weights.shift(lag).fillna(0.0)
        held = held.where(held.abs() >= shifted.abs(), shifted)
    return held.clip(-1.0, 1.0)


def _join_feature_block(features, feature_blocks, block):
    if block is None or block.frame.empty:
        return features, dict(feature_blocks)

    updated_blocks = dict(feature_blocks)
    features = features.join(block.frame)
    for column in block.frame.columns:
        updated_blocks[column] = block.block_name
    return features, updated_blocks


def _resolve_signal_delay_bars(backtest_config):
    configured = backtest_config.get("signal_delay_bars")
    if configured is not None:
        return max(0, int(configured))
    if backtest_config.get("use_open_execution", True):
        return 2
    return 1


def _resolve_backtest_market(pipeline):
    return pipeline.section("data").get("market", "spot")


def _resolve_backtest_valuation_close(pipeline, index):
    raw_data = pipeline.require("raw_data")
    backtest_config = pipeline.section("backtest")
    valuation_series = raw_data["close"].reindex(index)

    if backtest_config.get("valuation_price") == "mark":
        futures_context = pipeline.state.get("futures_context") or {}
        mark_frame = futures_context.get("mark_price")
        if mark_frame is not None and not mark_frame.empty and "mark_close" in mark_frame.columns:
            valuation_series = mark_frame["mark_close"].reindex(index)
    return valuation_series


def _resolve_backtest_execution_prices(pipeline, index):
    raw_data = pipeline.require("raw_data")
    backtest_config = pipeline.section("backtest")
    if backtest_config.get("use_open_execution", True) and "open" in raw_data.columns:
        return raw_data["open"].reindex(index)
    return raw_data["close"].reindex(index)


def _resolve_backtest_slippage_adv_window(pipeline):
    backtest_config = pipeline.section("backtest")
    configured = backtest_config.get("slippage_adv_window")
    if configured is not None:
        return max(1, int(configured))

    for indicator in pipeline.section("indicators") or []:
        if isinstance(indicator, dict):
            kind = indicator.get("kind")
            params = indicator.get("params", {}) or {}
            period = params.get("period", indicator.get("period"))
            if kind == "atr":
                return max(1, int(period)) if period is not None else 14
            continue

        if getattr(indicator, "kind", None) == "atr":
            period = getattr(indicator, "period", None)
            return max(1, int(period)) if period is not None else 14

    return 14


def _resolve_backtest_slippage_model(pipeline):
    backtest_config = pipeline.section("backtest")
    configured = backtest_config.get("slippage_model")
    if configured is None:
        return None

    if isinstance(configured, str):
        aliases = {
            "flat": "flat",
            "proxy": "proxy",
            "sqrt-impact": "sqrt_impact",
            "sqrt_impact": "sqrt_impact",
            "square-root-impact": "sqrt_impact",
            "square_root_impact": "sqrt_impact",
            "orderbook": "orderbook",
            "order_book": "orderbook",
            "depth": "depth_curve",
            "depth_curve": "depth_curve",
            "fill_aware": "fill_aware",
        }
        resolved_name = aliases.get(configured.strip().lower(), configured.strip().lower())
        if resolved_name == "flat":
            return FlatSlippageModel(rate=float(backtest_config.get("slippage_rate", 0.0)))
        if resolved_name == "proxy":
            return ProxyImpactModel(adv_window=_resolve_backtest_slippage_adv_window(pipeline))
        if resolved_name == "sqrt_impact":
            return SquareRootImpactModel(adv_window=_resolve_backtest_slippage_adv_window(pipeline))
        if resolved_name in {"orderbook", "depth_curve"}:
            return OrderBookImpactModel()
        if resolved_name == "fill_aware":
            return FillAwareCostModel(base_model=ProxyImpactModel(adv_window=_resolve_backtest_slippage_adv_window(pipeline)))
        raise ValueError("Unsupported backtest.slippage_model. Choose from ['flat', 'proxy', 'sqrt_impact', 'depth_curve', 'fill_aware']")

    if hasattr(configured, "estimate"):
        return configured
    raise TypeError("backtest.slippage_model must be a supported string alias or implement estimate(...)")


def _resolve_backtest_liquidity_lag_bars(pipeline):
    backtest_config = pipeline.section("backtest")
    configured = backtest_config.get("liquidity_lag_bars")
    if configured is None:
        return 1
    return max(0, int(configured))


def _resolve_pipeline_context_missing_policy(pipeline):
    features_config = pipeline.section("features") or {}
    configured = copy.deepcopy(features_config.get("context_missing_policy"))
    compat_config = dict(pipeline.config.get("compat") or {})
    legacy_missing_semantics = bool(compat_config.get("legacy_missing_semantics", False))

    if legacy_missing_semantics:
        if configured is None:
            configured = {"mode": "zero_fill"}
        elif isinstance(configured, dict) and "mode" not in configured:
            configured["mode"] = "zero_fill"

    return _resolve_context_missing_policy(configured)


_VALID_BACKTEST_EXECUTION_PROFILES = {"research_surrogate", "trade_ready_event_driven"}


def _resolve_backtest_execution_profile(backtest_config):
    evaluation_mode = str(backtest_config.get("evaluation_mode", "research_only")).strip().lower()
    default_profile = "trade_ready_event_driven" if evaluation_mode == "trade_ready" else "research_surrogate"
    execution_profile = str(backtest_config.get("execution_profile", default_profile)).strip().lower()
    if execution_profile not in _VALID_BACKTEST_EXECUTION_PROFILES:
        raise ValueError(
            "backtest.execution_profile must be one of {'research_surrogate', 'trade_ready_event_driven'}"
        )

    research_only_override = bool(backtest_config.get("research_only_override", False))
    if (
        evaluation_mode == "trade_ready"
        and execution_profile != "trade_ready_event_driven"
        and not research_only_override
    ):
        raise RuntimeError(
            "Trade-ready evaluation requires execution_profile='trade_ready_event_driven' or backtest.research_only_override=true"
        )
    return execution_profile


def _resolve_lookahead_guard_config(pipeline):
    features_config = pipeline.section("features") or {}
    builders = list(features_config.get("builders") or [])
    configured = copy.deepcopy(features_config.get("lookahead_guard") or {})
    enabled = bool(configured.get("enabled", True)) and bool(builders)
    default_mode = "blocking" if builders else "advisory"
    mode = str(configured.get("mode", default_mode)).strip().lower()
    if mode not in {"blocking", "advisory"}:
        mode = default_mode
    return {
        "enabled": enabled,
        "mode": mode,
        "decision_sample_size": max(1, int(configured.get("decision_sample_size", 32))),
        "min_prefix_rows": max(1, int(configured.get("min_prefix_rows", 128))),
        "builders_present": bool(builders),
        "builder_count": int(len(builders)),
    }


def _run_pipeline_lookahead_guard(pipeline):
    guard_config = _resolve_lookahead_guard_config(pipeline)
    report = {
        **guard_config,
        "has_bias": False,
        "promotion_pass": True,
        "reasons": [],
        "biased_columns": [],
        "checked_timestamps": 0,
        "requested_timestamps": 0,
        "skipped_timestamps": [],
        "artifact_report": {},
    }
    if not guard_config["enabled"]:
        pipeline.state["lookahead_guard_report"] = report
        return report

    from .lookahead import run_lookahead_analysis

    audit = run_lookahead_analysis(
        pipeline,
        step_names=["build_features"],
        artifact_names=["features"],
        sample_count=guard_config["decision_sample_size"],
        min_prefix_rows=guard_config["min_prefix_rows"],
    )
    artifact_report = dict((audit.get("artifacts") or {}).get("features") or {})
    report.update(
        {
            "has_bias": bool(audit.get("has_bias", False)),
            "promotion_pass": not bool(audit.get("has_bias", False)),
            "reasons": (["lookahead_guard_failed"] if audit.get("has_bias", False) else []),
            "biased_columns": list(artifact_report.get("biased_columns") or []),
            "checked_timestamps": int(audit.get("checked_timestamps", 0)),
            "requested_timestamps": int(audit.get("requested_timestamps", 0)),
            "skipped_timestamps": list(audit.get("skipped_timestamps") or []),
            "artifact_report": artifact_report,
            "audit": audit,
        }
    )
    pipeline.state["lookahead_guard_report"] = report
    return report


def _resolve_backtest_execution_policy(pipeline):
    backtest_config = pipeline.section("backtest") or {}
    execution_profile = _resolve_backtest_execution_profile(backtest_config)
    research_only_override = bool(backtest_config.get("research_only_override", False))
    configured = backtest_config.get("execution_policy")
    if configured is not None:
        resolved_policy = resolve_execution_policy(configured)
    else:
        resolved_policy = resolve_execution_policy(
            {
                "adapter": backtest_config.get("execution_adapter", "bar_surrogate"),
                "order_type": backtest_config.get("order_type", "market"),
                "time_in_force": backtest_config.get("time_in_force", "IOC"),
                "participation_cap": backtest_config.get("participation_cap", 0.10),
                "min_fill_ratio": backtest_config.get("min_fill_ratio", 0.25),
                "action_latency_bars": backtest_config.get("action_latency_bars", 0),
                "max_order_age_bars": backtest_config.get("max_order_age_bars", 1),
                "cancel_replace_bars": backtest_config.get("cancel_replace_bars", 1),
                "force_simulation": backtest_config.get("force_simulation", False),
            }
        )

    if (
        execution_profile == "trade_ready_event_driven"
        and resolved_policy.adapter != "nautilus"
        and not research_only_override
    ):
        raise RuntimeError(
            "Trade-ready evaluation requires a Nautilus execution adapter or backtest.research_only_override=true"
        )
    return resolved_policy.to_dict()


def _resolve_backtest_funding_missing_policy(backtest_config):
    configured = backtest_config.get("funding_missing_policy")
    evaluation_mode = str(backtest_config.get("evaluation_mode", "research_only")).strip().lower()
    policy = {
        "mode": "strict" if evaluation_mode == "trade_ready" else "zero_fill",
        "expected_interval": "8h",
        "max_gap_multiplier": 1.5,
    }
    if isinstance(configured, str):
        policy["mode"] = configured
    elif isinstance(configured, dict):
        policy.update(configured)
    policy["mode"] = str(policy.get("mode", "zero_fill")).strip().lower()
    policy["expected_interval"] = pd.Timedelta(policy.get("expected_interval", "8h"))
    policy["max_gap_multiplier"] = float(policy.get("max_gap_multiplier", 1.5))
    return policy


def _store_backtest_funding_report(pipeline, report):
    if not report:
        return
    context_ttl_report = dict(pipeline.state.get("context_ttl_report") or {})
    context_ttl_report["backtest_funding"] = report
    pipeline.state["context_ttl_report"] = context_ttl_report


def _align_backtest_funding_rates(funding_rates, index, policy):
    if funding_rates is None:
        return None, {
            "enabled": True,
            "scope": "backtest_funding",
            "policy": {
                "mode": policy["mode"],
                "expected_interval": str(policy["expected_interval"]),
                "max_gap_multiplier": float(policy["max_gap_multiplier"]),
            },
            "coverage_reason": "funding_unavailable",
            "promotion_pass": policy["mode"] not in {"strict", "preserve", "preserve_missing"},
        }

    aligned_index = pd.DatetimeIndex(index)
    observed = _normalize_funding_timestamp_index(pd.Series(funding_rates, copy=False)).dropna().astype(float)
    if len(aligned_index) > 0 and not observed.empty:
        observed = observed.loc[(observed.index >= aligned_index[0]) & (observed.index <= aligned_index[-1])]
    aligned = observed.reindex(aligned_index).fillna(0.0).astype(float)
    expected_interval = policy["expected_interval"]
    max_allowed_gap = expected_interval * float(policy["max_gap_multiplier"])

    if observed.empty:
        report = {
            "enabled": True,
            "scope": "backtest_funding",
            "policy": {
                "mode": policy["mode"],
                "expected_interval": str(expected_interval),
                "max_gap_multiplier": float(policy["max_gap_multiplier"]),
            },
            "coverage_reason": "no_observed_funding_events",
            "promotion_pass": policy["mode"] not in {"strict", "preserve", "preserve_missing"},
        }
        return aligned, report

    observed_index = pd.DatetimeIndex(observed.index)
    diffs = pd.Series(observed_index[1:] - observed_index[:-1]) if len(observed_index) > 1 else pd.Series(dtype="timedelta64[ns]")
    max_observed_gap = diffs.max() if not diffs.empty else pd.Timedelta(0)
    internal_gap_count = int((diffs > max_allowed_gap).sum()) if not diffs.empty else 0
    leading_gap = observed_index[0] - aligned_index[0] if len(aligned_index) > 0 else pd.Timedelta(0)
    trailing_gap = aligned_index[-1] - observed_index[-1] if len(aligned_index) > 0 else pd.Timedelta(0)
    off_index_event_count = int(len(observed_index.difference(aligned_index)))

    breach_reasons = []
    if leading_gap > max_allowed_gap:
        breach_reasons.append("leading_coverage_gap")
    if trailing_gap > max_allowed_gap:
        breach_reasons.append("trailing_coverage_gap")
    if internal_gap_count > 0:
        breach_reasons.append("internal_event_gap")
    if off_index_event_count > 0:
        breach_reasons.append("off_index_funding_timestamps")

    report = {
        "enabled": True,
        "scope": "backtest_funding",
        "policy": {
            "mode": policy["mode"],
            "expected_interval": str(expected_interval),
            "max_gap_multiplier": float(policy["max_gap_multiplier"]),
        },
        "observed_event_count": int(len(observed_index)),
        "leading_gap": str(leading_gap),
        "trailing_gap": str(trailing_gap),
        "max_observed_gap": str(max_observed_gap),
        "internal_gap_count": internal_gap_count,
        "off_index_event_count": off_index_event_count,
        "coverage_reason": None if not breach_reasons else ",".join(breach_reasons),
        "promotion_pass": len(breach_reasons) == 0,
    }
    return aligned, report


def _resolve_backtest_funding_rates(pipeline, index):
    backtest_config = pipeline.section("backtest")
    market = _resolve_backtest_market(pipeline)
    if not backtest_config.get("apply_funding", market != "spot"):
        return None

    funding_policy = _resolve_backtest_funding_missing_policy(backtest_config)
    trade_ready_mode = str(backtest_config.get("evaluation_mode", "research_only")).strip().lower() == "trade_ready"
    strict_funding_policy = trade_ready_mode or funding_policy["mode"] in {"strict", "preserve", "preserve_missing"}
    futures_context = pipeline.state.get("futures_context") or {}
    funding_frame = futures_context.get("funding")
    if funding_frame is None or funding_frame.empty or "funding_rate" not in funding_frame.columns:
        report = {
            "enabled": True,
            "scope": "backtest_funding",
            "policy": {
                "mode": funding_policy["mode"],
                "expected_interval": str(funding_policy["expected_interval"]),
                "max_gap_multiplier": float(funding_policy["max_gap_multiplier"]),
            },
            "coverage_reason": "funding_frame_missing",
            "promotion_pass": not strict_funding_policy,
        }
        _store_backtest_funding_report(pipeline, report)
        if strict_funding_policy:
            raise RuntimeError("Funding coverage gate failed: funding_frame_missing")
        return None
    aligned_funding, report = _align_backtest_funding_rates(funding_frame["funding_rate"], index, funding_policy)
    _store_backtest_funding_report(pipeline, report)
    if strict_funding_policy and report is not None and not bool(report.get("promotion_pass", True)):
        raise RuntimeError(f"Funding coverage gate failed: {report.get('coverage_reason', 'unknown')}" )
    return aligned_funding


def _resolve_backtest_futures_account(pipeline):
    market = _resolve_backtest_market(pipeline)
    if market == "spot":
        return None

    backtest_config = pipeline.section("backtest") or {}
    configured = dict(backtest_config.get("futures_account", {}) or {})
    if not configured:
        return None

    if configured.get("contract_spec") is None and pipeline.state.get("futures_contract_spec"):
        configured["contract_spec"] = pipeline.state.get("futures_contract_spec")

    cached_brackets = pipeline.state.get("futures_leverage_brackets")
    if configured.get("leverage_brackets") is None:
        if cached_brackets is not None:
            configured["leverage_brackets"] = cached_brackets
        else:
            leverage_brackets_data = configured.get("leverage_brackets_data")
            leverage_brackets_path = configured.get("leverage_brackets_path")
            use_signed_endpoint = bool(configured.get("use_signed_leverage_brackets", False))
            if leverage_brackets_data is not None or leverage_brackets_path is not None or use_signed_endpoint:
                loaded_brackets = load_futures_leverage_brackets(
                    symbol=pipeline.section("data").get("symbol", "BTCUSDT"),
                    market=market,
                    brackets=leverage_brackets_data,
                    path=leverage_brackets_path,
                    cache_dir=pipeline.section("data").get("cache_dir", ".cache"),
                    use_signed_endpoint=use_signed_endpoint,
                    api_key=configured.get("api_key"),
                    api_secret=configured.get("api_secret"),
                )
                configured["leverage_brackets"] = loaded_brackets
                pipeline.state["futures_leverage_brackets"] = loaded_brackets
    return configured


def _resolve_backtest_runtime_kwargs(pipeline, index):
    backtest_config = pipeline.section("backtest")
    market = _resolve_backtest_market(pipeline)
    raw_data = pipeline.require("raw_data")
    allow_short = backtest_config.get("allow_short")
    if allow_short is None:
        allow_short = market != "spot"

    significance_config = backtest_config.get("significance")
    benchmark_returns = None
    if pipeline.state.get("benchmark_returns") is not None:
        benchmark_returns = pd.Series(pipeline.state["benchmark_returns"], copy=False).reindex(index)
    elif isinstance(significance_config, dict):
        benchmark_returns_column = significance_config.get("benchmark_returns_column")
        benchmark_price_column = significance_config.get("benchmark_price_column")
        for frame_name in ["raw_data", "data"]:
            frame = pipeline.state.get(frame_name)
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                continue
            if benchmark_returns_column and benchmark_returns_column in frame.columns:
                benchmark_returns = pd.Series(frame[benchmark_returns_column], copy=False).reindex(index)
                break
            if benchmark_price_column and benchmark_price_column in frame.columns:
                benchmark_returns = pd.Series(frame[benchmark_price_column], copy=False).reindex(index).pct_change().fillna(0.0)
                break

    futures_account = _resolve_backtest_futures_account(pipeline)

    return {
        "engine": backtest_config.get("engine", "vectorbt"),
        "market": market,
        "leverage": float(backtest_config.get("leverage", 1.0)),
        "allow_short": bool(allow_short),
        "symbol_filters": pipeline.state.get("symbol_filters"),
        "funding_rates": _resolve_backtest_funding_rates(pipeline, index),
        "volume": raw_data["volume"].reindex(index).fillna(0.0) if "volume" in raw_data.columns else None,
        "liquidity_lag_bars": _resolve_backtest_liquidity_lag_bars(pipeline),
        "execution_policy": _resolve_backtest_execution_policy(pipeline),
        "slippage_model": _resolve_backtest_slippage_model(pipeline),
        "orderbook_depth": pipeline.state.get("orderbook_depth"),
        "significance": significance_config,
        "benchmark_returns": benchmark_returns,
        "benchmark_sharpe": significance_config.get("benchmark_sharpe") if isinstance(significance_config, dict) else None,
        "execution_price_policy": backtest_config.get("execution_price_policy", "strict"),
        "execution_price_fill_limit": backtest_config.get("execution_price_fill_limit"),
        "valuation_price_policy": backtest_config.get("valuation_price_policy", "drop_rows"),
        "valuation_price_fill_limit": backtest_config.get("valuation_price_fill_limit"),
        "futures_account": futures_account,
        "futures_contract": pipeline.state.get("futures_contract_spec"),
        "futures_leverage_brackets": pipeline.state.get("futures_leverage_brackets"),
        "symbol_lifecycle": pipeline.state.get("symbol_lifecycle"),
        "symbol_lifecycle_policy": pipeline.state.get("universe_policy"),
        "scenario_schedule": backtest_config.get("scenario_schedule"),
        "scenario_policy": backtest_config.get("scenario_policy"),
        "scenario_matrix": backtest_config.get("scenario_matrix"),
        "evaluation_mode": backtest_config.get("evaluation_mode", "research_only"),
        "required_stress_scenarios": backtest_config.get("required_stress_scenarios"),
    }


def _resolve_monitoring_reference_index(backtest_reports):
    if backtest_reports is None:
        return pd.DatetimeIndex([])
    reports = [backtest_reports] if isinstance(backtest_reports, dict) else list(backtest_reports)
    for report in reports:
        if not isinstance(report, dict):
            continue
        equity_curve = report.get("equity_curve")
        if isinstance(equity_curve, pd.Series) and not equity_curve.empty:
            return pd.DatetimeIndex(equity_curve.index)
        order_ledger = report.get("order_ledger")
        if isinstance(order_ledger, pd.DataFrame) and not order_ledger.empty and "timestamp" in order_ledger.columns:
            return pd.DatetimeIndex(pd.to_datetime(order_ledger["timestamp"], utc=True))
    return pd.DatetimeIndex([])


def _build_pipeline_operational_monitoring(
    pipeline,
    *,
    backtest_reports=None,
    expected_feature_columns=None,
    actual_feature_columns=None,
    signal_decay_report=None,
    inference_latencies_ms=None,
    queue_backlog=None,
    scope="training",
):
    monitoring_config = dict(pipeline.section("monitoring") or {})
    raw_data = pipeline.state.get("raw_data")
    raw_index = raw_data.index if isinstance(raw_data, (pd.DataFrame, pd.Series)) else pd.DatetimeIndex([])
    orderbook_depth = pipeline.state.get("orderbook_depth")
    l2_snapshot_index = orderbook_depth.index if isinstance(orderbook_depth, (pd.DataFrame, pd.Series)) else pd.DatetimeIndex([])
    reference_index = _resolve_monitoring_reference_index(backtest_reports)
    if len(reference_index) == 0:
        reference_index = raw_index

    report = build_monitoring_report(
        data_index=raw_index,
        expected_data_end=monitoring_config.get("expected_data_end", raw_index[-1] if len(raw_index) > 0 else None),
        expected_data_interval=monitoring_config.get("expected_data_interval"),
        max_data_lag=monitoring_config.get("max_data_lag"),
        custom_data_report=pipeline.state.get("custom_data_report"),
        reference_index=reference_index,
        l2_snapshot_index=l2_snapshot_index,
        expected_feature_columns=expected_feature_columns or actual_feature_columns,
        actual_feature_columns=actual_feature_columns,
        backtest_reports=backtest_reports,
        baseline_backtest_report=monitoring_config.get("baseline_backtest_report"),
        signal_decay_report=signal_decay_report,
        baseline_signal_decay_report=monitoring_config.get("baseline_signal_decay_report"),
        inference_latencies_ms=inference_latencies_ms,
        queue_backlog=queue_backlog,
        policy=monitoring_config,
    )
    if monitoring_config.get("write_reports", True):
        symbol = pipeline.section("data").get("symbol", "run")
        timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d%H%M%S")
        report["artifacts"] = write_monitoring_artifacts(
            report,
            monitoring_config.get("root_dir", ".cache/monitoring"),
            run_id=f"{scope}_{symbol}_{timestamp}",
        )
    return report


def _attach_context_ttl_to_operational_monitoring(operational_monitoring, context_ttl_report):
    payload = dict(operational_monitoring or {})
    reports = dict(context_ttl_report or {})
    if not reports:
        return payload

    payload["context_ttl"] = reports
    if not all(bool(report.get("promotion_pass", True)) for report in reports.values()):
        payload["healthy"] = False
        reasons = list(payload.get("reasons", []))
        if "context_ttl_breached" not in reasons:
            reasons.append("context_ttl_breached")
        payload["reasons"] = reasons
    return payload


def _timed_inference_call(latency_store, func, *args, **kwargs):
    started_at = time.perf_counter()
    result = func(*args, **kwargs)
    latency_store.append((time.perf_counter() - started_at) * 1000.0)
    return result


def _resolve_signal_profitability_threshold(training_payload, config):
    avg_loss = float(training_payload.get("last_avg_loss", config.get("avg_loss", 0.02)))
    avg_win = float(training_payload.get("last_avg_win", config.get("avg_win", 0.02)))
    break_even_prob = avg_loss / max(avg_win + avg_loss, 1e-12)
    profitability_threshold = training_payload.get("last_signal_params", {}).get("profitability_threshold")
    if profitability_threshold is None:
        if config.get("sizing_mode", "expected_utility") == "kelly":
            profitability_threshold = training_payload.get("last_signal_params", {}).get(
                "meta_threshold",
                config.get("meta_threshold", 0.55),
            )
        else:
            profitability_threshold = break_even_prob
    return float(break_even_prob), float(profitability_threshold)


def _build_cpcv_path_results(training, pipeline, config, validation_method):
    break_even_prob, profitability_threshold = _resolve_signal_profitability_threshold(training, config)
    holding_bars = _resolve_signal_holding_bars(pipeline, config)
    path_results = []
    for path in training.get("oos_paths", []):
        path_results.append(
            {
                "fold": path.get("fold"),
                "split_id": path.get("split_id"),
                "validation_method": path.get("validation_method", validation_method),
                "train_blocks": path.get("train_blocks"),
                "test_blocks": path.get("test_blocks"),
                "predictions": path["predictions"],
                "primary_probabilities": path["primary_probabilities"],
                "meta_prob": path["meta_prob"],
                "profitability_prob": path.get("profitability_prob", path["meta_prob"]),
                "direction_edge": path["direction_edge"],
                "confidence": path["confidence"],
                "expected_trade_edge": path.get("expected_trade_edge"),
                "expected_trade_utility": path.get("expected_trade_edge"),
                "break_even_profit_prob": break_even_prob,
                "profitability_threshold": profitability_threshold,
                "position_size": path.get("position_size", path["kelly_size"]),
                "kelly_size": path["kelly_size"],
                "event_signals": path["event_signals"],
                "holding_bars": holding_bars,
                "continuous_signals": path["continuous_signals"],
                "signals": path["signals"],
                "avg_win_used": path.get("avg_win_used"),
                "avg_loss_used": path.get("avg_loss_used"),
                "kelly_trade_count": path.get("kelly_trade_count"),
                "used_flat_kelly_fallback": path.get("used_flat_kelly_fallback", False),
                "tuned_params": path.get("signal_params", {}),
                "signal_policy": path.get("signal_policy"),
            }
        )
    return path_results


def _build_signal_result_from_training_payload(training_payload, pipeline, config, fallback_scope, signal_source):
    break_even_prob, profitability_threshold = _resolve_signal_profitability_threshold(training_payload, config)
    return {
        "predictions": training_payload["oos_predictions"],
        "primary_probabilities": training_payload["oos_probabilities"],
        "meta_prob": training_payload["oos_meta_prob"],
        "profitability_prob": training_payload.get("oos_profitability_prob", training_payload["oos_meta_prob"]),
        "direction_edge": training_payload["oos_direction_edge"],
        "confidence": training_payload["oos_confidence"],
        "expected_trade_edge": training_payload.get("oos_expected_trade_edge"),
        "expected_trade_utility": training_payload.get("oos_expected_trade_edge"),
        "break_even_profit_prob": break_even_prob,
        "profitability_threshold": profitability_threshold,
        "position_size": training_payload.get("oos_position_size", training_payload["oos_kelly_size"]),
        "kelly_size": training_payload["oos_kelly_size"],
        "event_signals": training_payload["oos_event_signals"],
        "holding_bars": _resolve_signal_holding_bars(pipeline, config),
        "continuous_signals": training_payload["oos_continuous_signals"],
        "signals": training_payload["oos_signals"],
        "avg_win_used": training_payload.get("last_avg_win", config.get("avg_win", 0.02)),
        "avg_loss_used": training_payload.get("last_avg_loss", config.get("avg_loss", 0.02)),
        "kelly_trade_count": int(training_payload.get("oos_trade_count", 0)),
        "used_flat_kelly_fallback": False,
        "signal_tuning": training_payload.get("signal_tuning", []),
        "signal_policy": training_payload.get("signal_policy"),
        "tuned_params": training_payload.get("last_signal_params", {}),
        "signal_source": signal_source,
        "fallback_scope": fallback_scope,
        "validation_method": training_payload.get("validation", {}).get("method", "walk_forward"),
    }


def _run_path_backtests(pipeline, config, path_entries):
    path_backtests = []
    for path in path_entries:
        positions = path["continuous_signals"] if config.get("use_continuous_positions", True) else path["signals"]
        valuation_close = _resolve_backtest_valuation_close(pipeline, positions.index)
        execution_prices = _resolve_backtest_execution_prices(pipeline, positions.index)
        path_backtest = run_backtest(
            close=valuation_close,
            signals=positions,
            equity=config.get("equity", 10_000.0),
            fee_rate=config.get("fee_rate", 0.001),
            slippage_rate=config.get("slippage_rate", 0.0),
            signal_delay_bars=_resolve_signal_delay_bars(config),
            execution_prices=execution_prices,
            **_resolve_backtest_runtime_kwargs(pipeline, positions.index),
        )
        path_backtests.append(
            {
                "fold": path.get("fold"),
                "split_id": path.get("split_id"),
                "validation_method": path.get("validation_method"),
                "train_blocks": path.get("train_blocks"),
                "test_blocks": path.get("test_blocks"),
                "backtest": path_backtest,
            }
        )
    return path_backtests


def _summarize_path_backtests(path_backtests):
    summary = {
        "validation_method": "cpcv",
        "aggregate_mode": "diagnostic_distribution",
        "path_count": int(len(path_backtests)),
    }
    if not path_backtests:
        return summary

    first_backtest = path_backtests[0].get("backtest", {})
    if first_backtest.get("engine") is not None:
        summary["engine"] = first_backtest.get("engine")

    significance_payloads = []
    for path in path_backtests:
        payload = path.get("backtest", {}).get("statistical_significance")
        if isinstance(payload, dict):
            significance_payloads.append(payload)

    enabled_significance_payloads = [payload for payload in significance_payloads if payload.get("enabled")]
    if enabled_significance_payloads:
        summary["statistical_significance"] = {
            "enabled": False,
            "aggregate_mode": "path_diagnostics_only",
            "path_count": int(len(enabled_significance_payloads)),
            "method": enabled_significance_payloads[0].get("method"),
            "reason": (
                "CPCV path significance is diagnostic only; tradable confidence bounds and p-values "
                "are reported on the executable walk-forward replay."
            ),
            "metrics": {},
        }
    elif significance_payloads:
        summary["statistical_significance"] = {
            "enabled": False,
            "aggregate_mode": "path_diagnostics_only",
            "path_count": int(len(significance_payloads)),
            "reason": significance_payloads[0].get("reason"),
            "method": significance_payloads[0].get("method"),
            "metrics": {},
        }

    numeric_keys = [
        "starting_equity",
        "ending_equity",
        "net_profit",
        "net_profit_pct",
        "gross_profit",
        "gross_loss",
        "funding_pnl",
        "fees_paid",
        "slippage_paid",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "cagr",
        "annualized_volatility",
        "max_drawdown",
        "max_drawdown_amount",
        "profit_factor",
        "expectancy",
        "avg_win",
        "avg_loss",
        "exposure_rate",
        "signal_delay_bars",
        "total_trades",
        "closed_trades",
        "win_rate",
        "trade_win_rate",
    ]
    metrics = {}
    for key in numeric_keys:
        values = []
        for path in path_backtests:
            value = path.get("backtest", {}).get(key)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                values.append(numeric)
        if values:
            metrics[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "max": float(np.max(values)),
            }
    if metrics:
        summary["metrics"] = metrics

    return summary


def _split_train_validation_window(X_train, y_train, sample_weights, model_config):
    validation_size = model_config.get("validation_size")
    validation_fraction = float(model_config.get("validation_fraction", 0.2))
    min_fit_size = int(model_config.get("min_fit_size", 250))
    min_validation_size = int(model_config.get("min_validation_size", 100))

    n_rows = len(X_train)
    if validation_size is None:
        validation_size = int(round(n_rows * validation_fraction))
    validation_size = max(min_validation_size, int(validation_size))
    validation_size = min(validation_size, max(0, n_rows - min_fit_size))

    if validation_size < min_validation_size or (n_rows - validation_size) < min_fit_size:
        return X_train, y_train, sample_weights, None, None, None

    fit_end = n_rows - validation_size
    return (
        X_train.iloc[:fit_end],
        y_train.iloc[:fit_end],
        sample_weights.iloc[:fit_end],
        X_train.iloc[fit_end:],
        y_train.iloc[fit_end:],
        sample_weights.iloc[fit_end:],
    )


def _align_and_drop_invalid_rows(X, y, labels=None, sample_weights=None):
    if X is None or y is None:
        return X, y, labels, sample_weights

    y_series = pd.Series(y, copy=False).reindex(X.index)
    label_frame = labels.reindex(X.index) if labels is not None else None
    weight_series = None
    if sample_weights is not None:
        weight_series = pd.Series(sample_weights, copy=False).reindex(X.index).astype(float)

    mask = X.notna().all(axis=1) & y_series.notna()
    if weight_series is not None:
        mask &= weight_series.notna()

    keep_index = X.index[mask]
    X = X.loc[keep_index]
    y_series = y_series.loc[keep_index]
    if label_frame is not None:
        label_frame = label_frame.loc[keep_index]
    if weight_series is not None:
        weight_series = weight_series.loc[keep_index]
    return X, y_series, label_frame, weight_series


def _drop_all_nan_feature_columns(X, feature_blocks=None, feature_families=None):
    if X is None or X.empty:
        return X, dict(feature_blocks or {}), dict(feature_families or {}), []

    all_nan_columns = [column for column in X.columns if X[column].notna().sum() == 0]
    if not all_nan_columns:
        return X, dict(feature_blocks or {}), dict(feature_families or {}), []

    filtered_X = X.drop(columns=all_nan_columns)
    filtered_blocks = {
        column: block_name
        for column, block_name in dict(feature_blocks or {}).items()
        if column in filtered_X.columns
    }
    filtered_families = {
        column: family_name
        for column, family_name in dict(feature_families or {}).items()
        if column in filtered_X.columns
    }
    return filtered_X, filtered_blocks, filtered_families, all_nan_columns


def _summarize_fold_family_selection(fold_feature_selection):
    if not fold_feature_selection:
        return {
            "selected_families": [],
            "avg_selected_family_counts": {},
            "endogenous_only_selected_any_fold": False,
            "endogenous_only_selected_all_folds": False,
        }

    family_counts = {}
    selected_families = set()
    endogenous_flags = []
    for row in fold_feature_selection:
        family_summary = dict(row.get("family_summary") or {})
        selected_families.update(family_summary.get("selected_families", []))
        endogenous_flags.append(bool(family_summary.get("endogenous_only")))
        for family, count in dict(family_summary.get("selected_family_counts", {})).items():
            family_counts.setdefault(family, []).append(float(count))

    return {
        "selected_families": sorted(selected_families),
        "avg_selected_family_counts": {
            family: round(float(np.mean(counts)), 2)
            for family, counts in sorted(family_counts.items())
        },
        "endogenous_only_selected_any_fold": any(endogenous_flags),
        "endogenous_only_selected_all_folds": all(endogenous_flags),
    }


def _purge_overlapping_training_rows(X, y, labels, cutoff_timestamp, sample_weights=None):
    if cutoff_timestamp is None or labels is None or labels.empty or "t1" not in labels.columns:
        return X, y, labels, sample_weights, 0

    label_ends = pd.to_datetime(labels["t1"], errors="coerce")
    keep_mask = label_ends.notna() & (label_ends < cutoff_timestamp)
    removed = int((~keep_mask).sum())
    if removed == 0:
        return X, y, labels, sample_weights, 0

    keep_index = labels.index[keep_mask]
    X = X.loc[keep_index]
    y = y.loc[keep_index]
    labels = labels.loc[keep_index]
    if sample_weights is not None:
        sample_weights = pd.Series(sample_weights, copy=False).reindex(keep_index).astype(float)
    return X, y, labels, sample_weights, removed


def _purge_training_rows_for_test_intervals(X, y, labels, test_intervals, sample_weights=None):
    if not test_intervals or labels is None or labels.empty or "t1" not in labels.columns:
        return X, y, labels, sample_weights, 0

    label_starts = pd.to_datetime(pd.Index(labels.index), errors="coerce")
    label_ends = pd.to_datetime(labels["t1"], errors="coerce")
    keep_mask = label_starts.notna() & label_ends.notna()

    for interval_start, interval_end in test_intervals:
        overlaps = (label_starts <= interval_end) & (label_ends >= interval_start)
        keep_mask &= ~overlaps

    removed = int((~keep_mask).sum())
    if removed == 0:
        return X, y, labels, sample_weights, 0

    keep_index = labels.index[keep_mask]
    X = X.loc[keep_index]
    y = y.loc[keep_index]
    labels = labels.loc[keep_index]
    if sample_weights is not None:
        sample_weights = pd.Series(sample_weights, copy=False).reindex(keep_index).astype(float)
    return X, y, labels, sample_weights, removed


def _close_window_for_labels(close, labels):
    close = pd.Series(close, copy=False)
    if labels is None or labels.empty or close.empty:
        return close.iloc[0:0]

    start = labels.index.min()
    end = labels.index.max()
    if "t1" in labels.columns and labels["t1"].notna().any():
        end = max(end, pd.DatetimeIndex(labels["t1"].dropna()).max())
    return close.loc[(close.index >= start) & (close.index <= end)]


def _combine_class_balance_weights(y, uniqueness_weights):
    """Combine uniqueness-based sample weights with inverse class-frequency weights.

    Multiplying uniqueness weights by the inverse class frequency corrects for
    class imbalance for all model types (GBM has no class_weight parameter; RF
    and logistic no longer set class_weight='balanced' so this is the sole
    correction).  The result is normalised to unit mean to preserve scale.
    """
    class_counts = y.value_counts()
    n_samples = len(y)
    n_classes = len(class_counts)
    if n_classes == 0:
        return uniqueness_weights.clip(lower=1e-6)
    inv_freq = n_samples / (n_classes * class_counts)
    class_balance = y.map(inv_freq).astype(float)
    combined = uniqueness_weights * class_balance
    mean_weight = combined.mean()
    if mean_weight > 0:
        combined = combined / mean_weight
    return combined.clip(lower=1e-6)


def _compute_fold_sample_weights(labels, close):
    if labels is None or labels.empty:
        return pd.Series(dtype=float, name="sample_weight")

    close_window = _close_window_for_labels(close, labels)
    if close_window.empty:
        return pd.Series(1.0, index=labels.index, name="sample_weight", dtype=float)

    weights = sample_weights_by_uniqueness(labels, close_window)
    return weights.reindex(labels.index).fillna(1.0)


def _build_training_sampling_metadata(labels, close, uniqueness_weights, model_config):
    if labels is None or labels.empty:
        return None

    close_window = _close_window_for_labels(close, labels)
    uniqueness_weights = pd.Series(uniqueness_weights, copy=False).reindex(labels.index).fillna(1.0)
    seq_config = dict(model_config.get("sequential_bootstrap", {}))
    return {
        "labels": labels.copy(),
        "close": close_window,
        "uniqueness_weights": uniqueness_weights,
        "mean_uniqueness": float(uniqueness_weights.mean()) if len(uniqueness_weights) > 0 else None,
        "sequential_bootstrap": {
            "enabled": bool(seq_config.get("enabled", True)),
            "uniqueness_threshold": float(seq_config.get("uniqueness_threshold", 0.90)),
            "n_samples": seq_config.get("n_samples"),
            "random_state": int(seq_config.get("random_state", model_config.get("random_state", 42))),
        },
    }


def _resolve_stationarity_screening_config(pipeline):
    features_config = pipeline.section("features")
    screening_config = dict(pipeline.section("stationarity"))
    screening_config.setdefault("enabled", True)
    screening_config.setdefault("rolling_window", features_config.get("rolling_window", 20))
    screening_config.setdefault("frac_diff_d", features_config.get("frac_diff_d"))
    return screening_config


def _coerce_regime_frame(regimes, column_name="regime"):
    if isinstance(regimes, pd.DataFrame):
        return regimes.copy()
    return pd.DataFrame({column_name: pd.Series(regimes, copy=False)})


class _FoldScopedPipeline:
    """Lightweight pipeline view that scopes 'data' and 'raw_data' to a fold window.

    Passed to user-supplied regime feature builders so that rolling statistics
    are computed only from fold-local data.  This prevents regime feature
    quantile thresholds from being informed by bars outside the fold window.
    """

    _DATA_KEYS = frozenset({"raw_data", "data"})

    def __init__(self, pipeline, windowed_data):
        self._pipeline = pipeline
        self._windowed_data = windowed_data

    def section(self, key):
        return self._pipeline.section(key)

    def require(self, key):
        if key in self._DATA_KEYS:
            return self._windowed_data
        return self._pipeline.require(key)

    @property
    def state(self):
        return self._pipeline.state


def _build_regime_feature_source(pipeline):
    regime_features = pipeline.state.get("regime_features")
    if regime_features is not None:
        return regime_features

    config = pipeline.section("regime")
    builder = config.get("builder") or _default_regime_features
    feature_set = normalize_regime_feature_set(builder(pipeline))
    pipeline.state["regime_features"] = feature_set.frame
    pipeline.state["regime_feature_sources"] = dict(feature_set.source_map)
    pipeline.state["regime_provenance"] = dict(feature_set.provenance)
    return feature_set.frame


def _build_fold_local_regime_frame(pipeline, index, fit_index=None):
    """Build regime columns scoped to a single walk-forward fold.

    Per-fold recomputation guarantees that regime feature rolling statistics
    (e.g. rolling z-scores, trend slopes) and quantile/HMM parameters are
    computed only from data visible within the fold window plus a configurable
    lookback buffer.  The HMM / scaler / quantile thresholds are then fitted
    exclusively on ``fit_index`` (the training portion of the fold).
    """
    config = pipeline.section("regime")
    if not config.get("enabled", True):
        return pd.DataFrame(index=index), {}

    if index is None or len(index) == 0:
        return pd.DataFrame(index=index if index is not None else pd.Index([])), {}

    pipeline.state.pop("_last_regime_details", None)

    # --- Per-fold data scoping: include a lookback buffer so rolling stats
    #     at the fold start are not NaN due to insufficient history, while
    #     still preventing future-bar data from influencing quantile thresholds.
    raw_data = pipeline.state.get("raw_data")
    if raw_data is None:
        raw_data = pipeline.state.get("data")
    lookback = int(config.get("feature_lookback", 80))
    if raw_data is not None and hasattr(raw_data, "index"):
        fold_start_pos = raw_data.index.searchsorted(index[0])
        fold_end_pos = raw_data.index.searchsorted(index[-1], side="right")
        buffer_start = max(0, fold_start_pos - lookback)
        buffered_data = raw_data.iloc[buffer_start:fold_end_pos]
    else:
        buffered_data = raw_data

    builder = config.get("builder") or _default_regime_features
    scoped_pipeline = _FoldScopedPipeline(pipeline, buffered_data)
    feature_set = normalize_regime_feature_set(builder(scoped_pipeline))
    regime_features = feature_set.frame

    if regime_features is None or (hasattr(regime_features, "empty") and regime_features.empty):
        pipeline.state["_last_regime_details"] = {
            "provenance": dict(feature_set.provenance),
            "ablation": {},
        }
        return pd.DataFrame(index=index), {}

    regime_window = regime_features.reindex(index)
    fit_features = regime_features.reindex(fit_index) if fit_index is not None else None
    regimes = detect_regime(
        regime_window,
        n_regimes=config.get("n_regimes", 2),
        method=config.get("method", "hmm"),
        config=config,
        fit_features=fit_features,
    )
    ablation = build_regime_ablation_report(
        {
            "frame": regime_window,
            "source_map": feature_set.source_map,
            "provenance": feature_set.provenance,
        },
        n_regimes=config.get("n_regimes", 2),
        method=config.get("method", "hmm"),
        config=config,
        fit_features=fit_features,
        full_regimes=regimes,
    )
    pipeline.state["_last_regime_details"] = {
        "provenance": dict(feature_set.provenance),
        "ablation": ablation,
    }
    frame = _coerce_regime_frame(regimes, column_name=config.get("column_name", "regime")).reindex(index)
    return frame, {column: "regime" for column in frame.columns}


def _build_execution_trade_outcomes(pipeline, predictions, holding_bars, cutoff_timestamp=None):
    raw_data = pipeline.require("raw_data")
    backtest_config = pipeline.section("backtest")
    full_index = raw_data.index
    runtime_kwargs = _resolve_backtest_runtime_kwargs(pipeline, full_index)
    return build_execution_outcome_frame(
        predictions,
        valuation_prices=_resolve_backtest_valuation_close(pipeline, full_index),
        execution_prices=_resolve_backtest_execution_prices(pipeline, full_index),
        holding_bars=holding_bars,
        signal_delay_bars=_resolve_signal_delay_bars(backtest_config),
        fee_rate=float(backtest_config.get("fee_rate", 0.0)),
        slippage_rate=float(backtest_config.get("slippage_rate", 0.0)),
        funding_rates=runtime_kwargs.get("funding_rates"),
        cutoff_timestamp=cutoff_timestamp,
        equity=float(backtest_config.get("equity", 10_000.0)),
        volume=runtime_kwargs.get("volume"),
        slippage_model=runtime_kwargs.get("slippage_model"),
        orderbook_depth=runtime_kwargs.get("orderbook_depth"),
        liquidity_lag_bars=runtime_kwargs.get("liquidity_lag_bars", 1),
    )


def _resolve_signal_decay_regime_frame(regimes, index):
    if regimes is None:
        return pd.DataFrame(index=index)
    if isinstance(regimes, pd.Series):
        return regimes.reindex(index).to_frame(name=regimes.name or "regime")
    if isinstance(regimes, pd.DataFrame):
        return regimes.reindex(index)
    return pd.DataFrame(index=index)


def _build_signal_decay_segment(
    pipeline,
    *,
    signal_index,
    predictions,
    direction_edge,
    event_signals,
    regime_frame=None,
):
    if signal_index is None or len(signal_index) == 0:
        return None

    valuation_prices = _resolve_backtest_valuation_close(pipeline, signal_index)
    if valuation_prices is None or len(valuation_prices) == 0:
        return None

    execution_prices = _resolve_backtest_execution_prices(pipeline, signal_index)
    runtime_kwargs = _resolve_backtest_runtime_kwargs(pipeline, signal_index)
    backtest_config = pipeline.section("backtest") or {}
    return {
        "predictions": pd.Series(predictions, copy=False).reindex(signal_index),
        "direction_edge": None if direction_edge is None else pd.Series(direction_edge, copy=False).reindex(signal_index),
        "event_signals": pd.Series(event_signals, copy=False).reindex(signal_index),
        "valuation_prices": valuation_prices,
        "execution_prices": execution_prices,
        "runtime_kwargs": runtime_kwargs,
        "fee_rate": float(backtest_config.get("fee_rate", 0.0)),
        "slippage_rate": float(backtest_config.get("slippage_rate", 0.0)),
        "equity": float(backtest_config.get("equity", 10_000.0)),
        "liquidity_lag_bars": runtime_kwargs.get("liquidity_lag_bars", 1),
        "regimes": _resolve_signal_decay_regime_frame(regime_frame, signal_index),
        "cutoff_timestamp": signal_index[-1],
    }


def _build_signal_decay_report_for_segments(pipeline, segments, holding_bars):
    normalized_segments = [segment for segment in list(segments or []) if segment is not None]
    return build_signal_decay_report(
        normalized_segments,
        holding_bars=holding_bars,
        signal_delay_bars=_resolve_signal_delay_bars(pipeline.section("backtest")),
        execution_policy=_resolve_backtest_execution_policy(pipeline),
        config=(pipeline.section("signals") or {}).get("decay"),
    )


def _build_signal_decay_report_from_signal_state(pipeline, signal_state):
    segments = []
    if signal_state.get("paths"):
        for path in signal_state.get("paths", []):
            event_signals = path.get("event_signals")
            if event_signals is None:
                continue
            segment = _build_signal_decay_segment(
                pipeline,
                signal_index=event_signals.index,
                predictions=path.get("predictions"),
                direction_edge=path.get("direction_edge"),
                event_signals=event_signals,
            )
            if segment is not None:
                segments.append(segment)
        holding_bars = int(signal_state.get("paths", [{}])[0].get("holding_bars", 1))
    else:
        event_signals = signal_state.get("event_signals")
        if event_signals is None:
            return {}
        segment = _build_signal_decay_segment(
            pipeline,
            signal_index=event_signals.index,
            predictions=signal_state.get("predictions"),
            direction_edge=signal_state.get("direction_edge"),
            event_signals=event_signals,
            regime_frame=pipeline.state.get("regimes"),
        )
        if segment is not None:
            segments.append(segment)
        holding_bars = int(signal_state.get("holding_bars", 1))

    return _build_signal_decay_report_for_segments(pipeline, segments, holding_bars)


def _count_realized_trades(trade_outcomes):
    if trade_outcomes is None or trade_outcomes.empty:
        return 0

    if "trade_taken" in trade_outcomes.columns:
        trade_taken = pd.to_numeric(trade_outcomes["trade_taken"], errors="coerce").fillna(0)
        return int(trade_taken.astype(bool).sum())

    if "net_trade_return" in trade_outcomes.columns:
        realized_returns = pd.to_numeric(trade_outcomes["net_trade_return"], errors="coerce")
        return int(realized_returns.notna().sum())

    return 0


def _estimate_trade_outcome_stats(trade_outcomes, default_win, default_loss, pooled_trade_outcomes=None, shrinkage_alpha=None):
    if trade_outcomes is None or trade_outcomes.empty or "net_trade_return" not in trade_outcomes.columns:
        return float(default_win), float(default_loss)

    realized_returns = pd.to_numeric(trade_outcomes["net_trade_return"], errors="coerce")
    if "trade_taken" in trade_outcomes.columns:
        trade_mask = pd.to_numeric(trade_outcomes["trade_taken"], errors="coerce").fillna(0).astype(bool)
        realized_returns = realized_returns.loc[trade_mask]
    realized_returns = realized_returns.dropna()
    if realized_returns.empty:
        return float(default_win), float(default_loss)

    wins = realized_returns[realized_returns > 0]
    losses = realized_returns[realized_returns < 0]
    avg_win = float(wins.mean()) if len(wins) > 0 else float(default_win)
    avg_loss = float(losses.abs().mean()) if len(losses) > 0 else float(default_loss)

    if pooled_trade_outcomes is not None and shrinkage_alpha is not None:
        pooled_avg_win, pooled_avg_loss = _estimate_trade_outcome_stats(
            pooled_trade_outcomes,
            default_win,
            default_loss,
        )
        alpha = float(np.clip(shrinkage_alpha, 0.0, 1.0))
        avg_win = alpha * avg_win + (1.0 - alpha) * pooled_avg_win
        avg_loss = alpha * avg_loss + (1.0 - alpha) * pooled_avg_loss

    return avg_win, avg_loss


def _resolve_profitability_target(predictions, y_true, trade_outcomes=None, labels=None):
    if trade_outcomes is not None and not trade_outcomes.empty and "profitable" in trade_outcomes.columns:
        profitable = pd.to_numeric(trade_outcomes["profitable"].reindex(predictions.index), errors="coerce")
        if profitable.dropna().nunique() >= 2:
            return profitable
    if labels is not None:
        trade_outcomes = build_trade_outcome_frame(predictions, labels)
        if not trade_outcomes.empty and "profitable" in trade_outcomes.columns:
            profitable = trade_outcomes["profitable"].reindex(predictions.index).fillna(0).astype(int)
            if profitable.nunique() >= 2:
                return profitable
    return (pd.Series(predictions, index=predictions.index) == pd.Series(y_true, index=predictions.index)).astype(int)


def _average_fold_metric(fold_metrics, key):
    values = [float(metric[key]) for metric in fold_metrics if metric.get(key) is not None and np.isfinite(metric.get(key))]
    if not values:
        return None
    return round(float(np.mean(values)), 4)


def _resolve_validation_stability_policy(pipeline):
    validation_config = pipeline.section("validation") or {}
    configured = dict(validation_config.get("stability_policy", {}) or {})
    if not configured and pipeline.section("model").get("stability_policy"):
        configured = dict(pipeline.section("model").get("stability_policy") or {})

    enabled = bool(configured.get("enabled", bool(configured)))
    return {
        "enabled": enabled,
        "cv_metric": str(configured.get("cv_metric", "directional_accuracy")),
        "max_cv": None if configured.get("max_cv") is None else float(configured.get("max_cv")),
        "min_worst_fold_sharpe": (
            None if configured.get("min_worst_fold_sharpe") is None else float(configured.get("min_worst_fold_sharpe"))
        ),
        "min_worst_fold_net_profit_pct": (
            None
            if configured.get("min_worst_fold_net_profit_pct") is None
            else float(configured.get("min_worst_fold_net_profit_pct"))
        ),
        "max_drawdown_dispersion": (
            None if configured.get("max_drawdown_dispersion") is None else float(configured.get("max_drawdown_dispersion"))
        ),
        "min_fold_count": max(2, int(configured.get("min_fold_count", 2))),
    }


def _compute_fold_metric_stats(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(row.get(key))]
    if not values:
        return None

    array = np.asarray(values, dtype=float)
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    mean = float(np.mean(array))
    cv = None if abs(mean) <= 1e-12 else float(std / abs(mean))
    return {
        "count": int(len(array)),
        "mean": round(mean, 6),
        "std": round(std, 6),
        "median": round(float(np.median(array)), 6),
        "min": round(float(np.min(array)), 6),
        "max": round(float(np.max(array)), 6),
        "cv": None if cv is None else round(cv, 6),
    }


def _build_fold_stability_summary(fold_metrics, fold_backtests, policy=None):
    metric_rows = list(fold_metrics or [])
    backtest_rows = list(fold_backtests or [])
    resolved_policy = dict(policy or {})

    metrics = {}
    for key in [
        "accuracy",
        "f1_macro",
        "directional_accuracy",
        "directional_f1_macro",
        "log_loss",
        "brier_score",
        "calibration_error",
    ]:
        stats = _compute_fold_metric_stats(metric_rows, key)
        if stats is not None:
            metrics[key] = stats

    for key in ["sharpe_ratio", "net_profit_pct", "max_drawdown", "total_trades"]:
        stats = _compute_fold_metric_stats(backtest_rows, key)
        if stats is not None:
            metrics[key] = stats

    primary_metric = resolved_policy.get("cv_metric") or "directional_accuracy"
    if primary_metric not in metrics:
        if "directional_accuracy" in metrics:
            primary_metric = "directional_accuracy"
        elif "accuracy" in metrics:
            primary_metric = "accuracy"
        elif metrics:
            primary_metric = next(iter(metrics))

    reasons = []
    passed = True
    fold_count = int(max(len(metric_rows), len(backtest_rows)))
    if resolved_policy.get("enabled"):
        if fold_count < int(resolved_policy.get("min_fold_count", 2)):
            reasons.append("insufficient_folds")
            passed = False

        max_cv = resolved_policy.get("max_cv")
        primary_stats = metrics.get(primary_metric, {})
        primary_cv = primary_stats.get("cv")
        if max_cv is not None and (primary_cv is None or not np.isfinite(primary_cv) or primary_cv > max_cv):
            reasons.append(f"{primary_metric}_cv_above_limit")
            passed = False

        min_worst_fold_sharpe = resolved_policy.get("min_worst_fold_sharpe")
        worst_fold_sharpe = metrics.get("sharpe_ratio", {}).get("min")
        if min_worst_fold_sharpe is not None and (
            worst_fold_sharpe is None or worst_fold_sharpe < min_worst_fold_sharpe
        ):
            reasons.append("worst_fold_sharpe_below_minimum")
            passed = False

        min_worst_fold_net_profit_pct = resolved_policy.get("min_worst_fold_net_profit_pct")
        worst_fold_net_profit_pct = metrics.get("net_profit_pct", {}).get("min")
        if min_worst_fold_net_profit_pct is not None and (
            worst_fold_net_profit_pct is None or worst_fold_net_profit_pct < min_worst_fold_net_profit_pct
        ):
            reasons.append("worst_fold_net_profit_pct_below_minimum")
            passed = False

        max_drawdown_dispersion = resolved_policy.get("max_drawdown_dispersion")
        drawdown_dispersion = metrics.get("max_drawdown", {}).get("std")
        if max_drawdown_dispersion is not None and (
            drawdown_dispersion is None or drawdown_dispersion > max_drawdown_dispersion
        ):
            reasons.append("max_drawdown_dispersion_above_limit")
            passed = False

    return {
        "enabled": True,
        "policy_enabled": bool(resolved_policy.get("enabled", False)),
        "policy": resolved_policy,
        "fold_count": fold_count,
        "primary_metric": primary_metric,
        "metrics": metrics,
        "passed": passed,
        "reasons": reasons,
        "worst_fold_sharpe": metrics.get("sharpe_ratio", {}).get("min"),
        "worst_fold_net_profit_pct": metrics.get("net_profit_pct", {}).get("min"),
        "max_drawdown_dispersion": metrics.get("max_drawdown", {}).get("std"),
    }


def _position_size_from_profitability(probability, avg_win, avg_loss, signal_config):
    fraction = float(signal_config.get("fraction", 0.5))
    sizing_mode = signal_config.get("sizing_mode", "expected_utility")
    utility_scale = max(float(avg_win), float(avg_loss), 1e-12)
    expected_edge = float(probability) * float(avg_win) - (1.0 - float(probability)) * float(avg_loss)

    if sizing_mode == "kelly":
        size = kelly_fraction(
            prob_win=probability,
            avg_win=avg_win,
            avg_loss=avg_loss,
            fraction=fraction,
        )
    else:
        size = max(0.0, expected_edge) / utility_scale
        size = min(size, 1.0) * fraction

    max_kelly_fraction = signal_config.get("max_kelly_fraction")
    if max_kelly_fraction is not None:
        size = min(float(size), max(0.0, float(max_kelly_fraction)))

    return size, expected_edge


def _calibrate_binary_probability_series(probabilities, target, sample_weight=None, calibrator_config=None):
    probability_series = pd.Series(probabilities, index=target.index if hasattr(target, "index") else None, dtype=float)
    target_series = pd.Series(target, index=probability_series.index if probability_series.index is not None else None)
    mask = probability_series.notna() & target_series.notna()

    if mask.sum() < 30 or target_series.loc[mask].nunique() < 2:
        return probability_series.clip(0.0, 1.0), None

    aligned_weights = None
    if sample_weight is not None:
        aligned_weights = pd.Series(sample_weight, index=target_series.index).loc[mask]

    calibrator = fit_binary_probability_calibrator(
        probability_series.loc[mask].to_numpy(),
        target_series.loc[mask].astype(int),
        sample_weight=aligned_weights,
        model_params=calibrator_config,
    )
    calibrated = pd.Series(
        apply_binary_probability_calibrator(calibrator, probability_series.to_numpy()),
        index=probability_series.index,
        dtype=float,
    ).clip(0.0, 1.0)
    return calibrated, calibrator


def _calibrate_primary_probability_frame(probability_frame, y_true, sample_weight=None, calibrator_config=None):
    frame = probability_frame.copy()
    if 1 not in frame.columns or -1 not in frame.columns:
        return frame, None
    if 0 in frame.columns and frame[0].abs().sum() > 1e-12:
        return frame, None

    target = pd.Series(y_true, index=frame.index)
    binary_mask = target.isin([-1, 1])
    if binary_mask.sum() < 30 or target.loc[binary_mask].nunique() < 2:
        return frame, None

    aligned_weights = None
    if sample_weight is not None:
        aligned_weights = pd.Series(sample_weight, index=target.index).loc[binary_mask]

    calibrator = fit_binary_probability_calibrator(
        frame.loc[binary_mask, 1].to_numpy(),
        target.loc[binary_mask].eq(1).astype(int),
        sample_weight=aligned_weights,
        model_params=calibrator_config,
    )
    calibrated_long = pd.Series(
        apply_binary_probability_calibrator(calibrator, frame[1].to_numpy()),
        index=frame.index,
        dtype=float,
    ).clip(0.0, 1.0)
    frame[1] = calibrated_long
    frame[-1] = (1.0 - calibrated_long).clip(0.0, 1.0)
    if 0 in frame.columns:
        frame[0] = 0.0
    return frame, calibrator


def _apply_primary_probability_calibrator(probability_frame, calibrator):
    if calibrator is None:
        return probability_frame.copy()

    frame = probability_frame.copy()
    calibrated_long = pd.Series(
        apply_binary_probability_calibrator(calibrator, frame[1].to_numpy()),
        index=frame.index,
        dtype=float,
    ).clip(0.0, 1.0)
    frame[1] = calibrated_long
    frame[-1] = (1.0 - calibrated_long).clip(0.0, 1.0)
    if 0 in frame.columns:
        frame[0] = 0.0
    return frame


def _resolve_round_trip_cost_rate(backtest_config, trade_outcomes=None):
    static_round_trip_cost = 2.0 * (
        float(backtest_config.get("fee_rate", 0.001))
        + float(backtest_config.get("slippage_rate", 0.0))
    )
    if trade_outcomes is None or trade_outcomes.empty or "round_trip_cost_rate" not in trade_outcomes.columns:
        return static_round_trip_cost

    realized_costs = pd.to_numeric(trade_outcomes["round_trip_cost_rate"], errors="coerce")
    if "trade_taken" in trade_outcomes.columns:
        trade_mask = pd.to_numeric(trade_outcomes["trade_taken"], errors="coerce").fillna(0).astype(bool)
        realized_costs = realized_costs.loc[trade_mask]
    realized_costs = realized_costs.dropna()
    if realized_costs.empty:
        return static_round_trip_cost
    return float(realized_costs.mean())


def _compute_theory_thresholds(avg_win, avg_loss, backtest_config, signal_config, trade_outcomes=None):
    """Derive signal thresholds from cost math and Kelly break-even probability.

    No search or in-sample optimisation is performed. All thresholds are derived
    from first principles so they are identical whether computed on training data,
    validation data, or live:

    * ``threshold`` — minimum absolute signal size to cover average round-trip
      transaction costs, using realized dynamic slippage when available.
    * ``edge_threshold`` — minimum directional probability edge (|p_long − p_short|)
      before issuing a signal.
    * ``meta_threshold`` / ``profitability_threshold`` — Kelly break-even: the
      minimum probability of a profitable trade needed for positive expected return,
      i.e. ``avg_loss / (avg_win + avg_loss)``.
    """
    round_trip_cost = _resolve_round_trip_cost_rate(backtest_config, trade_outcomes=trade_outcomes)

    break_even_prob = float(avg_loss) / max(float(avg_win) + float(avg_loss), 1e-12)
    meta_threshold = max(0.5, round(break_even_prob, 6))

    threshold = max(round_trip_cost, float(signal_config.get("threshold", round_trip_cost)))
    edge_threshold = max(0.02, float(signal_config.get("edge_threshold", 0.02)))
    fraction = float(signal_config.get("fraction", 0.5))

    params = {
        "threshold": threshold,
        "edge_threshold": edge_threshold,
        "meta_threshold": meta_threshold,
        "profitability_threshold": break_even_prob,
        "fraction": fraction,
    }
    return {"params": params}


def _resolve_signal_policy_mode(signal_config):
    mode = str((signal_config or {}).get("policy_mode", "validation_calibrated")).lower()
    aliases = {
        "theory": "theory_only",
        "manual": "frozen_manual",
        "validation": "validation_calibrated",
    }
    mode = aliases.get(mode, mode)
    valid_modes = {"theory_only", "validation_calibrated", "frozen_manual"}
    if mode not in valid_modes:
        raise ValueError(
            "signals.policy_mode must be one of ['theory_only', 'validation_calibrated', 'frozen_manual']"
        )
    return mode


def _extend_signal_policy_params(params, signal_config):
    extended = dict(params)
    extended["fraction"] = float(signal_config.get("fraction", extended.get("fraction", 0.5)))
    extended["min_trades_for_kelly"] = int(signal_config.get("min_trades_for_kelly", 30))
    extended["max_kelly_fraction"] = float(signal_config.get("max_kelly_fraction", 0.5))
    extended["policy_mode"] = _resolve_signal_policy_mode(signal_config)
    return extended


class SignalPolicyBuilder:
    def __init__(self, signal_config, backtest_config):
        self.signal_config = dict(signal_config or {})
        self.backtest_config = dict(backtest_config or {})
        self.mode = _resolve_signal_policy_mode(self.signal_config)

    def build(self, avg_win, avg_loss, trade_outcomes=None, calibration_context=None):
        calibration_context = dict(calibration_context or {})
        applied_trade_outcomes = None

        if self.mode == "frozen_manual":
            report = self._build_frozen_manual_policy(avg_win, avg_loss)
            source = "frozen_manual"
        elif self.mode == "theory_only":
            report = _compute_theory_thresholds(
                avg_win=avg_win,
                avg_loss=avg_loss,
                backtest_config=self.backtest_config,
                signal_config=self.signal_config,
                trade_outcomes=None,
            )
            source = "static_cost_math"
        else:
            applied_trade_outcomes = trade_outcomes
            report = _compute_theory_thresholds(
                avg_win=avg_win,
                avg_loss=avg_loss,
                backtest_config=self.backtest_config,
                signal_config=self.signal_config,
                trade_outcomes=applied_trade_outcomes,
            )
            source = calibration_context.get("source") or (
                "validation_trade_outcomes"
                if applied_trade_outcomes is not None and not applied_trade_outcomes.empty
                else "validation_unavailable_static_fallback"
            )

        params = _extend_signal_policy_params(report["params"], self.signal_config)
        break_even_prob = float(avg_loss) / max(float(avg_win) + float(avg_loss), 1e-12)
        policy_quality = {
            "mode": self.mode,
            "source": source,
            "avg_win_used": float(avg_win),
            "avg_loss_used": float(avg_loss),
            "break_even_profit_prob": float(break_even_prob),
            "round_trip_cost_rate": float(
                _resolve_round_trip_cost_rate(
                    self.backtest_config,
                    trade_outcomes=applied_trade_outcomes if self.mode == "validation_calibrated" else None,
                )
            ),
            "used_trade_outcomes": bool(applied_trade_outcomes is not None and not applied_trade_outcomes.empty),
            "calibration_rows": int(calibration_context.get("calibration_rows", 0)),
            "kelly_trade_count": calibration_context.get("kelly_trade_count"),
        }
        for key, value in calibration_context.items():
            if key not in policy_quality:
                policy_quality[key] = value

        return {
            "mode": self.mode,
            "params": params,
            "policy_quality": policy_quality,
        }

    def _build_frozen_manual_policy(self, avg_win, avg_loss):
        break_even_prob = float(avg_loss) / max(float(avg_win) + float(avg_loss), 1e-12)
        meta_threshold = float(self.signal_config.get("meta_threshold", max(0.5, round(break_even_prob, 6))))
        profitability_threshold = self.signal_config.get("profitability_threshold")
        if profitability_threshold is None:
            if self.signal_config.get("sizing_mode", "expected_utility") == "kelly":
                profitability_threshold = meta_threshold
            else:
                profitability_threshold = break_even_prob

        return {
            "params": {
                "threshold": float(self.signal_config.get("threshold", 0.03)),
                "edge_threshold": float(self.signal_config.get("edge_threshold", 0.05)),
                "meta_threshold": meta_threshold,
                "profitability_threshold": float(profitability_threshold),
                "fraction": float(self.signal_config.get("fraction", 0.5)),
            }
        }


def _build_signal_state(
    prediction_series,
    probability_frame,
    meta_prob_series,
    signal_config,
    avg_win,
    avg_loss,
    holding_bars,
    kelly_trade_count=None,
):
    direction = prediction_series.apply(lambda value: 1.0 if value > 0 else (-1.0 if value < 0 else 0.0))
    direction_edge = probability_frame[1] - probability_frame[-1]
    confidence = direction_edge.abs().clip(0.0, 1.0)
    profitability_prob = meta_prob_series.clip(0.0, 1.0)
    min_trades_for_kelly = int(signal_config.get("min_trades_for_kelly", 30))
    use_flat_kelly_fallback = (
        signal_config.get("sizing_mode", "expected_utility") == "kelly"
        and kelly_trade_count is not None
        and int(kelly_trade_count) < min_trades_for_kelly
    )
    if use_flat_kelly_fallback:
        warnings.warn(
            (
                "Kelly sizing fallback activated: using flat fractional sizing because only "
                f"{int(kelly_trade_count)} OOS trades are available, below min_trades_for_kelly={min_trades_for_kelly}."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    position_size = pd.Series(index=profitability_prob.index, dtype=float)
    expected_trade_edge = pd.Series(index=profitability_prob.index, dtype=float)
    for timestamp, probability in profitability_prob.items():
        size, edge = _position_size_from_profitability(probability, avg_win, avg_loss, signal_config)
        if use_flat_kelly_fallback:
            size = min(float(signal_config.get("fraction", 0.5)), float(signal_config.get("max_kelly_fraction", 0.5)))
        position_size.loc[timestamp] = size
        expected_trade_edge.loc[timestamp] = edge

    break_even_prob = float(avg_loss) / max(float(avg_win) + float(avg_loss), 1e-12)
    profitability_threshold = signal_config.get("profitability_threshold")
    if profitability_threshold is None:
        if signal_config.get("sizing_mode", "expected_utility") == "kelly":
            profitability_threshold = signal_config.get("meta_threshold", 0.55)
        else:
            profitability_threshold = break_even_prob

    event_signals = direction * position_size
    event_signals = event_signals.where(direction.ne(0.0), 0.0)
    event_signals = event_signals.where(confidence >= signal_config.get("edge_threshold", 0.05), 0.0)
    event_signals = event_signals.where(profitability_prob >= profitability_threshold, 0.0)
    event_signals = event_signals.where(expected_trade_edge >= signal_config.get("expected_edge_threshold", 0.0), 0.0)
    event_signals = event_signals.where(event_signals.abs() >= signal_config.get("threshold", 0.03), 0.0)

    continuous = _apply_holding_period(event_signals, holding_bars)
    signals = continuous.apply(lambda value: 1 if value > 1e-12 else (-1 if value < -1e-12 else 0))
    return {
        "predictions": prediction_series,
        "primary_probabilities": probability_frame,
        "meta_prob": profitability_prob,
        "profitability_prob": profitability_prob,
        "direction_edge": direction_edge,
        "confidence": confidence,
        "expected_trade_edge": expected_trade_edge,
        "expected_trade_utility": expected_trade_edge,
        "break_even_profit_prob": break_even_prob,
        "profitability_threshold": float(profitability_threshold),
        "position_size": position_size,
        "kelly_size": position_size,
        "event_signals": event_signals,
        "holding_bars": holding_bars,
        "continuous_signals": continuous,
        "signals": signals,
        "avg_win_used": avg_win,
        "avg_loss_used": avg_loss,
        "kelly_trade_count": None if kelly_trade_count is None else int(kelly_trade_count),
        "used_flat_kelly_fallback": bool(use_flat_kelly_fallback),
        "tuned_params": {
            "threshold": float(signal_config.get("threshold", 0.03)),
            "edge_threshold": float(signal_config.get("edge_threshold", 0.05)),
            "meta_threshold": float(signal_config.get("meta_threshold", 0.55)),
            "profitability_threshold": float(profitability_threshold),
            "fraction": float(signal_config.get("fraction", 0.5)),
            "min_trades_for_kelly": min_trades_for_kelly,
            "max_kelly_fraction": float(signal_config.get("max_kelly_fraction", 0.5)),
        },
    }


def _build_empty_signal_state(index, signal_config, avg_win, avg_loss, holding_bars):
    empty_probabilities = pd.DataFrame(0.0, index=index, columns=[-1, 0, 1], dtype=float)
    return _build_signal_state(
        pd.Series(index=index, dtype=float),
        empty_probabilities,
        pd.Series(index=index, dtype=float),
        signal_config,
        avg_win,
        avg_loss,
        holding_bars,
    )


def _resolve_fallback_signal_frame(X, training):
    fallback_scope = dict(training.get("fallback_inference") or {})
    fallback_scope.setdefault("mode", "unrestricted")

    if fallback_scope["mode"] != "post_final_training_only":
        fallback_scope["aligned_input_row_count"] = int(len(X))
        fallback_scope["scored_row_count"] = int(len(X))
        fallback_scope["excluded_row_count"] = 0
        return X, fallback_scope

    train_end = fallback_scope.get("last_fold_train_end")
    if train_end is None:
        fit_index = training.get("last_regime_fit_index")
        if fit_index is not None and len(fit_index) > 0:
            train_end = fit_index[-1]
            fallback_scope["last_fold_train_end"] = train_end

    if train_end is None:
        fallback_scope["aligned_input_row_count"] = int(len(X))
        fallback_scope["scored_row_count"] = int(len(X))
        fallback_scope["excluded_row_count"] = 0
        return X, fallback_scope

    safe_X = X.loc[X.index > train_end]
    fallback_scope["aligned_input_row_count"] = int(len(X))
    fallback_scope["scored_row_count"] = int(len(safe_X))
    fallback_scope["excluded_row_count"] = int(len(X) - len(safe_X))
    fallback_scope["aligned_safe_start"] = safe_X.index[0] if len(safe_X) > 0 else None
    fallback_scope["aligned_safe_end"] = safe_X.index[-1] if len(safe_X) > 0 else None
    return safe_X, fallback_scope


def _score_signal_state(backtest, signal_config):
    score = float(backtest.get("net_profit_pct", 0.0))
    score += float(signal_config.get("tuning_weight_sharpe", 0.03)) * float(backtest.get("sharpe_ratio", 0.0))

    profit_factor = backtest.get("profit_factor", 0.0)
    if not np.isfinite(profit_factor):
        profit_factor = signal_config.get("tuning_profit_factor_cap", 5.0)
    score += float(signal_config.get("tuning_weight_profit_factor", 0.02)) * min(
        float(profit_factor),
        float(signal_config.get("tuning_profit_factor_cap", 5.0)),
    )
    score -= float(signal_config.get("tuning_weight_drawdown", 0.5)) * abs(float(backtest.get("max_drawdown", 0.0)))

    min_trades = int(signal_config.get("tuning_min_trades", 3))
    total_trades = int(backtest.get("total_trades", 0))
    if total_trades < min_trades:
        score -= float(signal_config.get("tuning_trade_penalty", 0.01)) * (min_trades - total_trades)
    return score


def _train_inner_meta_model(X_train, y_train, sample_weights, model_config, labels=None, close=None, trade_outcome_builder=None, context_frame=None):
    inner_predictions = []
    inner_probabilities = []
    inner_truth = []
    inner_label_frames = []
    inner_weights = []
    min_train_rows = max(50, min(100, len(X_train) // 3))
    min_test_rows = 20
    binary_primary = model_config.get("binary_primary", True)

    for inner_train_idx, inner_test_idx in walk_forward_split(
        X_train,
        n_splits=model_config.get("meta_n_splits", max(2, min(3, model_config.get("n_splits", 3)))),
        gap=model_config.get("gap", 0),
        expanding=model_config.get("expanding", False),
    ):
        if len(inner_train_idx) < min_train_rows or len(inner_test_idx) < min_test_rows:
            continue

        X_inner_train = X_train.iloc[inner_train_idx]
        y_inner_train = y_train.iloc[inner_train_idx]
        w_inner_train = sample_weights.iloc[inner_train_idx]
        X_inner_test = X_train.iloc[inner_test_idx]
        inner_labels_train = labels.iloc[inner_train_idx] if labels is not None else None

        inner_test_start = X_inner_test.index[0] if len(X_inner_test) > 0 else None
        X_inner_train, y_inner_train, inner_labels_train, w_inner_train, _ = _purge_overlapping_training_rows(
            X_inner_train,
            y_inner_train,
            inner_labels_train,
            cutoff_timestamp=inner_test_start,
            sample_weights=w_inner_train,
        )
        if len(X_inner_train) < min_train_rows:
            continue

        if binary_primary:
            binary_mask = y_inner_train.ne(0)
            X_inner_train = X_inner_train.loc[binary_mask]
            y_inner_train = y_inner_train.loc[binary_mask]
            w_inner_train = w_inner_train.loc[binary_mask]
            if len(X_inner_train) < min_train_rows:
                continue
        w_inner_train = _combine_class_balance_weights(y_inner_train, w_inner_train)

        sampling_metadata = None
        if close is not None:
            sampling_metadata = _build_training_sampling_metadata(
                inner_labels_train.loc[X_inner_train.index] if inner_labels_train is not None else None,
                close,
                sample_weights.loc[X_inner_train.index],
                model_config,
            )

        inner_model = train_model(
            X_inner_train,
            y_inner_train,
            sample_weight=w_inner_train,
            model_type=model_config.get("type", "gbm"),
            model_params=model_config.get("params"),
            sampling_metadata=sampling_metadata,
            emit_warnings=False,
        )

        inner_pred = pd.Series(inner_model.predict(X_inner_test), index=X_inner_test.index)
        inner_prob = predict_probability_frame(inner_model, X_inner_test)
        inner_predictions.append(inner_pred)
        inner_probabilities.append(inner_prob)
        inner_truth.append(y_train.iloc[inner_test_idx])
        if labels is not None:
            inner_label_frames.append(labels.iloc[inner_test_idx])
        inner_weights.append(sample_weights.iloc[inner_test_idx])

    if not inner_predictions:
        return ConstantProbabilityModel(positive_probability=0.5)

    meta_predictions = pd.concat(inner_predictions).sort_index()
    meta_probabilities = pd.concat(inner_probabilities).sort_index()
    meta_truth = pd.concat(inner_truth).sort_index().reindex(meta_predictions.index)
    meta_labels = None
    if inner_label_frames:
        meta_labels = pd.concat(inner_label_frames).sort_index().reindex(meta_predictions.index)
    meta_weights = pd.concat(inner_weights).sort_index().reindex(meta_predictions.index).fillna(1.0)
    meta_trade_outcomes = trade_outcome_builder(meta_predictions) if trade_outcome_builder is not None else None

    meta_context = None
    if context_frame is not None:
        meta_context = context_frame.reindex(meta_predictions.index)

    return train_meta_model(
        meta_predictions,
        meta_probabilities,
        meta_truth,
        labels=meta_labels,
        trade_outcomes=meta_trade_outcomes,
        sample_weight=meta_weights,
        model_params=model_config.get("meta_params"),
        context=meta_context,
    )


class PipelineStep:
    name = "step"

    def run(self, pipeline):
        raise NotImplementedError


class FetchDataStep(PipelineStep):
    name = "fetch_data"

    def run(self, pipeline):
        config = dict(pipeline.section("data"))
        reference_config = dict(pipeline.section("reference_data") or {})
        universe_config = dict(pipeline.section("universe") or {})
        futures_context_config = dict(config.pop("futures_context", {}) or {})
        cross_asset_context_config = dict(config.pop("cross_asset_context", {}) or {})
        custom_data_config = list(config.pop("custom_data", []) or [])
        config.pop("return_report", None)

        market = config.get("market", "spot")
        cache_dir = config.get("cache_dir", ".cache")
        primary_symbol = config.get("symbol", "BTCUSDT")
        context_symbols = list(cross_asset_context_config.get("symbols") or [])

        requested_symbols = [primary_symbol, *context_symbols, *(config.get("symbols") or []), *(universe_config.get("symbols") or [])]
        requested_symbols = list(dict.fromkeys([symbol for symbol in requested_symbols if symbol]))

        universe_snapshot = None
        universe_report = None
        if universe_config or len(requested_symbols) > 1:
            snapshot_timestamp = universe_config.get("snapshot_timestamp", config.get("start"))
            universe_snapshot = load_historical_universe_snapshot(
                snapshot_timestamp=snapshot_timestamp,
                market=universe_config.get("market", market),
                cache_dir=universe_config.get("cache_dir", cache_dir),
                snapshots=universe_config.get("snapshots"),
                path=universe_config.get("path"),
                fetch_if_missing=bool(universe_config.get("fetch_if_missing", False)),
            )
            universe_report = evaluate_universe_eligibility(
                universe_snapshot,
                as_of=snapshot_timestamp,
                requested_symbols=requested_symbols,
                min_history_days=universe_config.get(
                    "min_history_days",
                    universe_config.get("minimum_history_days", 0),
                ),
                min_liquidity=universe_config.get(
                    "min_liquidity",
                    universe_config.get("minimum_liquidity"),
                ),
            )
            ineligible_symbols = dict(universe_report.get("ineligible_symbols", {}))
            if primary_symbol in ineligible_symbols:
                reasons = ", ".join(ineligible_symbols[primary_symbol])
                raise ValueError(
                    f"Primary symbol {primary_symbol!r} is not eligible at the requested universe snapshot: {reasons}"
                )

            requested_symbol_policy = str(universe_config.get("requested_symbol_policy", "error")).lower()
            blocked_context_symbols = {
                symbol: reasons
                for symbol, reasons in ineligible_symbols.items()
                if symbol in context_symbols
            }
            if blocked_context_symbols and requested_symbol_policy == "error":
                blocked_summary = "; ".join(
                    f"{symbol}: {', '.join(reasons)}"
                    for symbol, reasons in blocked_context_symbols.items()
                )
                raise ValueError(
                    "Cross-symbol study requested symbols that are not eligible at the universe snapshot: "
                    f"{blocked_summary}"
                )
            if blocked_context_symbols and requested_symbol_policy in {"drop", "filter"}:
                context_symbols = [symbol for symbol in context_symbols if symbol not in blocked_context_symbols]

            pipeline.state["universe_snapshot"] = universe_snapshot.symbols.copy()
            pipeline.state["universe_snapshot_meta"] = {
                "snapshot_timestamp": universe_snapshot.snapshot_timestamp,
                "market": universe_snapshot.market,
                "source": universe_snapshot.source,
            }
            pipeline.state["eligible_symbols"] = list(universe_report.get("eligible_symbols", []))
            pipeline.state["universe_report"] = universe_report

        market_data, integrity_report = fetch_binance_bars(**config, return_report=True)
        data = market_data.copy()
        custom_data_report = []
        if custom_data_config:
            data, custom_data_report = join_custom_data(data, custom_data_config)

        pipeline.state["raw_data"] = market_data
        pipeline.state["data"] = data.copy()
        pipeline.state["custom_data_report"] = custom_data_report
        pipeline.state["data_integrity_report"] = integrity_report

        try:
            pipeline.state["symbol_filters"] = fetch_binance_symbol_filters(
                symbol=primary_symbol,
                market=market,
                cache_dir=cache_dir,
            )
        except Exception:
            pipeline.state["symbol_filters"] = {}

        if market != "spot":
            try:
                pipeline.state["futures_contract_spec"] = fetch_binance_futures_contract_spec(
                    symbol=primary_symbol,
                    market=market,
                    cache_dir=cache_dir,
                )
            except Exception:
                pipeline.state["futures_contract_spec"] = {}

        if futures_context_config.get("enabled", True):
            pipeline.state["futures_context"] = fetch_binance_futures_context(
                symbol=primary_symbol,
                interval=config.get("interval", "1h"),
                start=config.get("start", "2024-01-01"),
                end=config.get("end", "2024-03-01"),
                cache_dir=futures_context_config.get("cache_dir", cache_dir),
                include_recent_stats=futures_context_config.get("include_recent_stats", True),
                recent_stats_availability_lag=futures_context_config.get("recent_stats_availability_lag", "period_close"),
            )

        lifecycle_events = universe_config.get("lifecycle_events")
        if universe_snapshot is not None or lifecycle_events:
            lifecycle = build_symbol_lifecycle_frame(
                index=market_data.index,
                symbol=primary_symbol,
                snapshot=universe_snapshot,
                events=lifecycle_events,
            )
            if lifecycle is not None:
                pipeline.state["symbol_lifecycle"] = lifecycle
                pipeline.state["universe_policy"] = {
                    "halt_action": universe_config.get("halt_action", "freeze"),
                    "delist_action": universe_config.get("delist_action", "liquidate"),
                }

        if context_symbols:
            pipeline.state["cross_asset_context"] = fetch_context_symbol_bars(
                symbols=context_symbols,
                interval=config.get("interval", "1h"),
                start=config.get("start", "2024-01-01"),
                end=config.get("end", "2024-03-01"),
                cache_dir=cross_asset_context_config.get("cache_dir", cache_dir),
                market=cross_asset_context_config.get("market", market),
            )
            pipeline.state["cross_asset_context_symbols"] = list(context_symbols)

        if reference_config.get("enabled", False):
            reference_bundle = build_reference_validation_bundle(
                market_data,
                market=market,
                symbol=primary_symbol,
                interval=config.get("interval", "1h"),
                start=config.get("start"),
                end=config.get("end"),
                cache_dir=reference_config.get("cache_dir", cache_dir),
                futures_context=pipeline.state.get("futures_context"),
                config=reference_config,
            )
            pipeline.state["reference_integrity_report"] = dict(reference_bundle.get("report") or {})
            pipeline.state["reference_venue_frames"] = dict(reference_bundle.get("venue_frames") or {})
            reference_overlay = reference_bundle.get("overlay")
            if reference_overlay is not None and not pd.DataFrame(reference_overlay).empty:
                pipeline.state["reference_overlay_data"] = pd.DataFrame(reference_overlay).copy()

        market_manifest = dict(getattr(market_data, "attrs", {}).get("dataset_manifest") or {})
        custom_manifests = [
            dict(report.get("dataset_manifest") or {})
            for report in custom_data_report
            if report.get("dataset_manifest")
        ]
        futures_context_manifests = _extract_frame_manifest_mapping(pipeline.state.get("futures_context"))
        cross_asset_manifests = _extract_frame_manifest_mapping(pipeline.state.get("cross_asset_context"))
        if market_manifest:
            _set_lineage_group(pipeline, "market_data", market_manifest)
        if custom_manifests:
            _set_lineage_group(pipeline, "custom_data", custom_manifests)
        if futures_context_manifests:
            _set_lineage_group(pipeline, "futures_context", futures_context_manifests)
        if cross_asset_manifests:
            _set_lineage_group(pipeline, "cross_asset_context", cross_asset_manifests)
        _ensure_pipeline_data_contracts(pipeline)

        return data


class DataQualityStep(PipelineStep):
    name = "check_data_quality"

    def run(self, pipeline):
        _ensure_pipeline_data_contracts(pipeline)
        raw_data = pipeline.require("raw_data")
        data = pipeline.require("data")
        config = pipeline.section("data_quality")
        result = check_data_quality(raw_data, config=config)

        clean_raw = result.clean_frame
        clean_index = clean_raw.index
        pipeline.state["raw_data_original"] = raw_data.copy()
        pipeline.state["raw_data"] = clean_raw
        pipeline.state["data"] = data.reindex(clean_index).copy()
        pipeline.state["data_quality_mask"] = result.quarantine_mask
        pipeline.state["data_quality_report"] = result.report
        _refresh_active_dataset_manifests(pipeline, data_quality_report=result.report)
        if result.report.get("blocking"):
            raise ValueError("Data quality quarantine blocked the run")
        return clean_raw


class IndicatorsStep(PipelineStep):
    name = "run_indicators"

    def run(self, pipeline):
        data = pipeline.require("data")
        indicator_run = run_indicators(data, pipeline.section("indicators"))
        pipeline.state["indicator_run"] = indicator_run
        pipeline.state["data"] = indicator_run.frame
        return indicator_run


class AutoMLStep(PipelineStep):
    name = "run_automl"

    def run(self, pipeline):
        config = pipeline.section("automl")
        if not config.get("enabled"):
            return None

        summary = run_automl_study(
            pipeline,
            pipeline_class=ResearchPipeline,
            trial_step_classes=[
                FeaturesStep,
                RegimeStep,
                LabelsStep,
                AlignDataStep,
                FeatureSelectionStep,
                SampleWeightsStep,
                TrainModelsStep,
                SignalsStep,
                BacktestStep,
            ],
        )

        best_overrides = copy.deepcopy(summary.get("best_overrides", {}))
        for section, values in best_overrides.items():
            current = pipeline.config.get(section, {})
            if isinstance(current, dict) and isinstance(values, dict):
                merged = dict(current)
                merged.update(values)
                pipeline.config[section] = merged
            else:
                pipeline.config[section] = values

        for key in [
            "features",
            "stationarity",
            "regime_features",
            "regimes",
            "labels",
            "X",
            "y",
            "labels_aligned",
            "feature_selection",
            "sample_weights",
            "training",
            "signals",
            "backtest",
        ]:
            pipeline.state.pop(key, None)

        pipeline.state["automl"] = summary
        return summary


class FeaturesStep(PipelineStep):
    name = "build_features"

    def run(self, pipeline):
        _ensure_pipeline_data_contracts(pipeline, include_reference=True)
        data = pipeline.require("data")
        raw_data = pipeline.require("raw_data")
        indicator_run = pipeline.state.get("indicator_run")
        config = pipeline.section("features")
        context_missing_policy = _resolve_pipeline_context_missing_policy(pipeline)
        feature_set = build_feature_set(
            data,
            lags=config.get("lags"),
            frac_diff_d=config.get("frac_diff_d"),
            indicator_run=indicator_run,
            rolling_window=config.get("rolling_window", 20),
            squeeze_quantile=config.get("squeeze_quantile", 0.2),
        )
        features = feature_set.frame
        feature_blocks = dict(feature_set.feature_blocks)
        feature_families = dict(feature_set.feature_families)

        futures_context_block = build_futures_context_feature_block(
            raw_data,
            pipeline.state.get("futures_context"),
            rolling_window=config.get("rolling_window", 20),
            ttl_config=config.get("futures_context_ttl"),
            missing_policy=context_missing_policy,
        )
        features, feature_blocks = _join_feature_block(features, feature_blocks, futures_context_block)
        ttl_report = dict((futures_context_block.metadata or {}).get("ttl_report") or {})
        if ttl_report:
            context_ttl_report = dict(pipeline.state.get("context_ttl_report") or {})
            context_ttl_report["futures_context"] = ttl_report
            pipeline.state["context_ttl_report"] = context_ttl_report

        cross_asset_context_block = build_cross_asset_context_feature_block(
            raw_data,
            pipeline.state.get("cross_asset_context"),
            rolling_window=config.get("rolling_window", 20),
            ttl_config=config.get("cross_asset_context_ttl"),
            missing_policy=context_missing_policy,
        )
        features, feature_blocks = _join_feature_block(features, feature_blocks, cross_asset_context_block)
        ttl_report = dict((cross_asset_context_block.metadata or {}).get("ttl_report") or {})
        if ttl_report:
            context_ttl_report = dict(pipeline.state.get("context_ttl_report") or {})
            context_ttl_report["cross_asset_context"] = ttl_report
            pipeline.state["context_ttl_report"] = context_ttl_report

        multi_timeframe_block = build_multi_timeframe_context_feature_block(
            raw_data,
            base_interval=pipeline.section("data").get("interval", "1h"),
            timeframes=config.get("context_timeframes"),
            rolling_window=config.get("rolling_window", 20),
        )
        features, feature_blocks = _join_feature_block(features, feature_blocks, multi_timeframe_block)

        reference_overlay_block = build_reference_overlay_feature_block(
            raw_data,
            reference_data=(
                pipeline.state.get("reference_overlay_data")
                if pipeline.state.get("reference_overlay_data") is not None
                else pipeline.state.get("reference_data")
            ),
            rolling_window=config.get("rolling_window", 20),
        )
        features, feature_blocks = _join_feature_block(features, feature_blocks, reference_overlay_block)

        for builder in config.get("builders", []):
            built = builder(pipeline, features.copy())
            if built is not None:
                features = built

            feature_blocks = {
                column: feature_blocks.get(column, config.get("custom_block_name", "custom"))
                for column in features.columns
            }

        feature_families = derive_feature_families(feature_blocks, columns=features.columns)

        screening_result = screen_features_for_stationarity(
            features,
            feature_blocks=feature_blocks,
            config=_resolve_stationarity_screening_config(pipeline),
        )
        screening_report = copy.deepcopy(screening_result.report)
        screening_report["mode"] = "global_preview_only"
        screening_report.setdefault("summary", {})["mode"] = "global_preview_only"
        feature_metadata = derive_feature_metadata(
            feature_blocks=feature_blocks,
            feature_families=feature_families,
            columns=features.columns,
            screening_report=screening_report,
            feature_lineage=getattr(feature_set, "feature_lineage", {}),
            retired_features=(pipeline.section("feature_governance") or {}).get("retired_features"),
        )

        pipeline.state["raw_features"] = features
        pipeline.state["feature_blocks_raw"] = feature_blocks
        pipeline.state["feature_families_raw"] = feature_families
        pipeline.state["feature_metadata_raw"] = feature_metadata
        pipeline.state["feature_screening"] = screening_report
        pipeline.state["feature_blocks"] = feature_blocks
        pipeline.state["feature_families"] = feature_families
        pipeline.state["feature_metadata"] = feature_metadata
        pipeline.state["feature_family_summary"] = summarize_feature_families(feature_blocks, columns=features.columns)
        pipeline.state["features"] = features
        return features


class StationarityStep(PipelineStep):
    name = "check_stationarity"

    def run(self, pipeline):
        config = pipeline.section("stationarity")
        specs = config.get("series") or _default_stationarity_specs(pipeline)
        results = {}

        for spec in specs:
            source = pipeline.require(spec.get("source", "data"))
            column = spec["column"]
            series = source[column]
            results[spec.get("name", column)] = check_stationarity(series.dropna())

        if "feature_screening" in pipeline.state:
            results["feature_screening"] = pipeline.state["feature_screening"]

        pipeline.state["stationarity"] = results
        return results


class RegimeStep(PipelineStep):
    name = "detect_regimes"

    def run(self, pipeline):
        config = pipeline.section("regime")
        builder = config.get("builder") or _default_regime_features
        feature_set = normalize_regime_feature_set(builder(pipeline))
        regime_features = feature_set.frame
        regimes = detect_regime(
            regime_features,
            n_regimes=config.get("n_regimes", 2),
            method=config.get("method", "hmm"),
            config=config,
        )
        ablation = build_regime_ablation_report(
            feature_set,
            n_regimes=config.get("n_regimes", 2),
            method=config.get("method", "hmm"),
            config=config,
            full_regimes=regimes,
        )

        pipeline.state["regime_features"] = regime_features
        pipeline.state["regime_feature_sources"] = dict(feature_set.source_map)
        pipeline.state["regime_provenance"] = dict(feature_set.provenance)
        pipeline.state["regimes"] = regimes
        pipeline.state["regime_ablation"] = ablation
        pipeline.state["regime_detection"] = {
            "mode": "global_preview_only",
            "method": config.get("method", "hmm"),
            "columns": list(regimes.columns) if isinstance(regimes, pd.DataFrame) else [config.get("column_name", "regime")],
            "provenance": feature_set.provenance,
            "ablation": ablation,
        }
        return {
            "regime_features": regime_features,
            "regimes": regimes,
            "mode": "global_preview_only",
            "provenance": feature_set.provenance,
            "ablation": ablation,
        }


class LabelsStep(PipelineStep):
    name = "build_labels"

    def run(self, pipeline):
        data = pipeline.require("data")
        raw_data = pipeline.require("raw_data")
        config = pipeline.section("labels")
        backtest_config = pipeline.section("backtest")
        label_kind = config.get("kind", "triple_barrier")
        cost_rate = config.get("cost_rate")
        if cost_rate is None:
            cost_rate = 2.0 * (
                float(backtest_config.get("fee_rate", 0.0))
                + float(backtest_config.get("slippage_rate", 0.0))
            )

        # Resolve execution-aligned entry: anchor label barriers to the actual fill
        # price (open[T+delay] or close[T+delay]) rather than close[T].
        signal_delay = _resolve_signal_delay_bars(backtest_config)
        if backtest_config.get("use_open_execution", True) and "open" in raw_data.columns:
            entry_prices = raw_data["open"]
        else:
            entry_prices = raw_data["close"]
        start_offset = signal_delay

        if label_kind == "triple_barrier":
            volatility_builder = config.get("volatility_builder")
            if volatility_builder is None:
                window = config.get("volatility_window", 24)
                volatility = data["close"].pct_change().rolling(window).std()
            else:
                volatility = volatility_builder(pipeline)

            labels = triple_barrier_labels(
                close=data["close"],
                volatility=volatility,
                high=data.get("high"),
                low=data.get("low"),
                pt_sl=config.get("pt_sl", (2.0, 2.0)),
                max_holding=config.get("max_holding", 24),
                min_return=config.get("min_return", 0.0),
                cost_rate=float(cost_rate),
                barrier_tie_break=config.get("barrier_tie_break", "sl"),
                entry_prices=entry_prices,
                start_offset=start_offset,
            )
        elif label_kind == "fixed_horizon":
            labels = fixed_horizon_labels(
                close=data["close"],
                horizon=config.get("horizon", 5),
                threshold=config.get("threshold", 0.0),
                cost_rate=float(cost_rate),
                entry_prices=entry_prices,
                start_offset=start_offset,
            )
        elif label_kind == "trend_scanning":
            labels = trend_scanning_labels(
                close=data["close"],
                min_horizon=config.get("min_horizon", 8),
                max_horizon=config.get("max_horizon", config.get("max_holding", 48)),
                step=config.get("step", 4),
                min_t_value=config.get("min_t_value", 1.5),
                min_return=config.get("min_return", 0.0),
                cost_rate=float(cost_rate),
                price_transform=config.get("price_transform", "log"),
                entry_prices=entry_prices,
                start_offset=start_offset,
            )
        else:
            raise ValueError(f"Unsupported label kind={label_kind!r}")

        pipeline.state["labels"] = labels
        return labels


class AlignDataStep(PipelineStep):
    name = "align_data"

    def run(self, pipeline):
        features = pipeline.require("features")
        labels = pipeline.require("labels")
        label_column = pipeline.section("labels").get("label_column", "label")
        feature_blocks = pipeline.state.get("feature_blocks", {})
        feature_families = pipeline.state.get("feature_families") or derive_feature_families(feature_blocks)
        feature_metadata = pipeline.state.get("feature_metadata") or derive_feature_metadata(
            feature_blocks=feature_blocks,
            feature_families=feature_families,
        )

        common = features.index.intersection(labels.index)
        X = features.loc[common].copy()
        y = labels.loc[common, label_column].copy()
        labels_aligned = labels.loc[common].copy()

        X, feature_blocks, feature_families, dropped_columns = _drop_all_nan_feature_columns(
            X,
            feature_blocks,
            feature_families,
        )

        mask = X.notna().all(axis=1) & y.notna()
        X = X.loc[mask]
        y = y.loc[mask]
        labels_aligned = labels_aligned.loc[mask]

        X, feature_blocks, feature_families, feature_metadata, retirement_report = apply_feature_retirement(
            X,
            feature_blocks=feature_blocks,
            feature_families=feature_families,
            feature_metadata=feature_metadata,
            config=pipeline.section("feature_governance"),
        )

        pipeline.state["feature_blocks"] = feature_blocks
        pipeline.state["feature_families"] = feature_families
        pipeline.state["feature_metadata"] = filter_feature_metadata(feature_metadata, X.columns)
        pipeline.state["feature_family_summary"] = summarize_feature_families(feature_blocks, columns=X.columns)
        pipeline.state["feature_retirement"] = retirement_report
        if dropped_columns:
            report = dict(pipeline.state.get("feature_screening", {}))
            report.setdefault("alignment", {})["dropped_all_nan_columns"] = dropped_columns
            if retirement_report.get("dropped_columns"):
                report.setdefault("alignment", {})["retired_columns"] = retirement_report.get("dropped_columns")
            pipeline.state["feature_screening"] = report
        elif retirement_report.get("dropped_columns"):
            report = dict(pipeline.state.get("feature_screening", {}))
            report.setdefault("alignment", {})["retired_columns"] = retirement_report.get("dropped_columns")
            pipeline.state["feature_screening"] = report

        pipeline.state["X"] = X
        pipeline.state["y"] = y
        pipeline.state["labels_aligned"] = labels_aligned
        return {"X": X, "y": y, "labels_aligned": labels_aligned}


class FeatureSelectionStep(PipelineStep):
    name = "select_features"

    def run(self, pipeline):
        X = pipeline.require("X")
        config = pipeline.section("feature_selection")

        report = {
            "enabled": bool(config.get("enabled", True)),
            "mode": "fold_local",
            "input_features": int(X.shape[1]),
            "selected_features": int(X.shape[1]),
            "max_features": config.get("max_features"),
            "min_mi_threshold": float(config.get("min_mi_threshold", 0.0)),
            "note": "Supervised feature selection is applied inside each validation split.",
            "input_family_summary": summarize_feature_families(
                pipeline.state.get("feature_blocks", {}),
                columns=X.columns,
            ),
        }
        pipeline.state["feature_selection"] = report
        return type("FeatureSelectionPreview", (), {"report": report})()


class SampleWeightsStep(PipelineStep):
    name = "compute_sample_weights"

    def run(self, pipeline):
        labels_aligned = pipeline.require("labels_aligned")
        raw_data = pipeline.require("raw_data")
        weights = sample_weights_by_uniqueness(labels_aligned, raw_data["close"])
        pipeline.state["sample_weights"] = weights
        return weights


class TrainModelsStep(PipelineStep):
    name = "train_models"

    def run(self, pipeline):
        X = pipeline.require("X")
        y = pipeline.require("y")
        labels_aligned = pipeline.require("labels_aligned")
        raw_data = pipeline.require("raw_data")
        config = pipeline.section("model")
        selection_config = pipeline.section("feature_selection")
        signal_config = pipeline.section("signals")
        backtest_config = pipeline.section("backtest")
        signal_policy_builder = SignalPolicyBuilder(signal_config, backtest_config)
        feature_blocks = pipeline.state.get("feature_blocks", {})
        stationarity_config = _resolve_stationarity_screening_config(pipeline)
        binary_primary = config.get("binary_primary", True)
        holding_bars = _resolve_signal_holding_bars(pipeline, signal_config)
        default_avg_win = float(signal_config.get("avg_win", 0.02))
        default_avg_loss = float(signal_config.get("avg_loss", 0.02))
        shrinkage_alpha = float(signal_config.get("shrinkage_alpha", 0.5))
        default_signal_policy = signal_policy_builder.build(
            default_avg_win,
            default_avg_loss,
            calibration_context={
                "source": "static_defaults",
                "calibration_rows": 0,
                "kelly_trade_count": 0,
            },
        )
        close_all = raw_data["close"]
        validation_method = _resolve_validation_method(config)
        stability_policy = _resolve_validation_stability_policy(pipeline)
        validation_details = {"method": validation_method}
        if validation_method == "cpcv":
            validation_details.update(
                {
                    "n_blocks": _resolve_cpcv_block_count(config),
                    "test_blocks": _resolve_cpcv_test_block_count(config),
                    "embargo_bars": _resolve_cpcv_embargo_bars(pipeline, config),
                }
            )
        else:
            validation_details.update(
                {
                    "gap": int(config.get("gap", 0)),
                    "n_splits": int(config.get("n_splits", 3)),
                }
            )

        fold_metrics = []
        fold_backtests = []
        fold_block_diagnostics = []
        fold_family_diagnostics = []
        fold_feature_governance = []
        fold_feature_selection = []
        fold_signal_policy = []
        fold_stationarity = []
        fold_purging = []
        fold_regime = []
        fold_bootstrap = []
        fold_backtest_reports = []
        inference_latencies_ms = []
        last_model = None
        last_meta = None
        last_primary_calibrator = None
        last_meta_calibrator = None
        last_selected_columns = list(X.columns)
        last_regime_fit_index = X.index[:0]
        last_test_index = X.index[:0]
        last_signal_params = dict(default_signal_policy["params"])
        last_signal_policy = dict(default_signal_policy["policy_quality"])
        last_avg_win = default_avg_win
        last_avg_loss = default_avg_loss
        last_kelly_trade_count = 0
        oos_predictions = []
        oos_probabilities = []
        oos_meta_prob = []
        oos_direction_edge = []
        oos_confidence = []
        oos_kelly_size = []
        oos_expected_trade_edge = []
        oos_event_signals = []
        oos_continuous_signals = []
        oos_signals = []
        oos_trade_outcomes = []
        oos_paths = []
        signal_decay_segments = []

        lookahead_guard_report = _run_pipeline_lookahead_guard(pipeline)
        if lookahead_guard_report.get("enabled", False) and not lookahead_guard_report.get("promotion_pass", True):
            if lookahead_guard_report.get("mode") == "blocking":
                raise RuntimeError("Lookahead guard failed: custom feature builders are not causally safe")

        context_missing_policy = _resolve_pipeline_context_missing_policy(pipeline)
        context_missing_mode = str(context_missing_policy.get("mode", "preserve_missing")).strip().lower()
        trade_ready_mode = str(backtest_config.get("evaluation_mode", "research_only")).strip().lower() == "trade_ready"
        if trade_ready_mode or context_missing_mode in {"preserve", "preserve_missing", "strict"}:
            context_ttl_report = dict(pipeline.state.get("context_ttl_report") or {})
            breached_scopes = [
                scope for scope, report in context_ttl_report.items()
                if not bool((report or {}).get("promotion_pass", True))
            ]
            if breached_scopes:
                joined_scopes = ", ".join(sorted(breached_scopes))
                raise RuntimeError(f"Context integrity gate failed: {joined_scopes}")

        for split in _iter_validation_splits(pipeline, X):
            fold = split["fold"]
            split_id = split["split_id"]
            split_meta = split["metadata"]
            train_idx = split["train_idx"]
            test_idx = split["test_idx"]
            test_intervals = split["test_intervals"]

            X_train_raw = X.iloc[train_idx]
            X_test_raw = X.iloc[test_idx]
            y_train_raw = y.iloc[train_idx]
            y_test_raw = y.iloc[test_idx]
            labels_train_raw = labels_aligned.iloc[train_idx]
            labels_test_raw = labels_aligned.iloc[test_idx]

            X_train, y_train, labels_train, _, outer_purged = _purge_overlapping_training_rows(
                X_train_raw,
                y_train_raw,
                labels_train_raw,
                cutoff_timestamp=None,
            )
            if test_intervals:
                X_train, y_train, labels_train, _, outer_purged = _purge_training_rows_for_test_intervals(
                    X_train_raw,
                    y_train_raw,
                    labels_train_raw,
                    test_intervals,
                )
            if X_train.empty:
                continue

            split_weights = pd.Series(1.0, index=X_train.index, name="sample_weight")
            X_fit, y_fit, _, X_val, y_val, _ = _split_train_validation_window(
                X_train,
                y_train,
                split_weights,
                config,
            )
            labels_fit = labels_train.loc[X_fit.index]
            labels_val = labels_train.loc[X_val.index].copy() if X_val is not None else None

            val_start = X_val.index[0] if X_val is not None and not X_val.empty else None
            X_fit, y_fit, labels_fit, _, inner_purged = _purge_overlapping_training_rows(
                X_fit,
                y_fit,
                labels_fit,
                cutoff_timestamp=val_start,
            )
            if X_fit.empty:
                continue

            fold_window = pd.concat([X_train_raw, X_test_raw]).sort_index()
            fold_screening = screen_features_for_stationarity(
                fold_window,
                feature_blocks=feature_blocks,
                config=stationarity_config,
                fit_features=X_fit,
            )
            fold_feature_blocks = dict(fold_screening.feature_blocks)
            fold_stationarity.append(
                {
                    "fold": fold,
                    "split_id": split_id,
                    "summary": fold_screening.report.get("summary", {}),
                }
            )

            fold_frame = fold_screening.frame
            regime_frame, regime_feature_blocks = _build_fold_local_regime_frame(
                pipeline,
                fold_window.index,
                fit_index=X_fit.index,
            )
            regime_details = dict(pipeline.state.pop("_last_regime_details", {}) or {})
            if not regime_frame.empty:
                fold_frame = fold_frame.join(regime_frame)
                fold_feature_blocks.update(regime_feature_blocks)
            fold_regime.append(
                {
                    "fold": fold,
                    "split_id": split_id,
                    "mode": "fold_local",
                    "columns": list(regime_frame.columns),
                    "available_rows": int(regime_frame.dropna(how="all").shape[0]),
                    "provenance": regime_details.get("provenance", {}),
                    "ablation": regime_details.get("ablation", {}),
                }
            )

            X_fit = fold_frame.reindex(X_fit.index)
            X_val = fold_frame.reindex(X_val.index) if X_val is not None else None
            X_test = fold_frame.reindex(X_test_raw.index)

            X_fit, y_fit, labels_fit, _ = _align_and_drop_invalid_rows(X_fit, y_fit, labels_fit)
            if X_val is not None:
                X_val, y_val, labels_val, _ = _align_and_drop_invalid_rows(X_val, y_val, labels_val)
            X_test, y_test, labels_test, _ = _align_and_drop_invalid_rows(X_test, y_test_raw, labels_test_raw)

            if X_fit.empty or X_test.empty:
                continue
            if X_val is not None and X_val.empty:
                X_val, y_val, labels_val = None, None, None

            w_fit = _compute_fold_sample_weights(labels_fit, close_all).reindex(X_fit.index).fillna(1.0)
            w_val = None
            if X_val is not None and labels_val is not None and not labels_val.empty:
                w_val = _compute_fold_sample_weights(labels_val, close_all).reindex(X_val.index).fillna(1.0)

            fold_purging.append(
                {
                    "fold": fold,
                    "split_id": split_id,
                    "validation_method": validation_method,
                    "outer_purged_rows": int(outer_purged),
                    "inner_purged_rows": int(inner_purged),
                    "embargo_rows": int(split_meta.get("embargo_rows", 0)),
                    "fit_rows": int(len(X_fit)),
                    "validation_rows": int(len(X_val)) if X_val is not None else 0,
                    "test_rows": int(len(X_test)),
                }
            )

            selected_columns = list(X_fit.columns)
            selection_result = None
            if selection_config.get("enabled", True):
                selection_result = select_features(
                    X_fit,
                    y_fit,
                    feature_blocks=fold_feature_blocks,
                    config=selection_config,
                )
                if not selection_result.frame.empty:
                    selected_columns = [column for column in selection_result.frame.columns if column in X_fit.columns]
                    if selected_columns:
                        fold_feature_blocks = selection_result.feature_blocks
                    else:
                        selected_columns = list(X_fit.columns)

            family_summary = summarize_feature_families(fold_feature_blocks, columns=selected_columns)

            X_fit_model = X_fit.loc[:, selected_columns]
            X_test_model = X_test.loc[:, selected_columns]
            X_val_model = X_val.loc[:, selected_columns] if X_val is not None else None

            fold_feature_metadata = filter_feature_metadata(
                pipeline.state.get("feature_metadata") or derive_feature_metadata(fold_feature_blocks, columns=selected_columns),
                selected_columns,
            )
            feature_admission = evaluate_feature_admission(
                X_fit_model,
                y_fit,
                feature_metadata=fold_feature_metadata,
                regime_data=regime_frame.reindex(X_fit_model.index) if not regime_frame.empty else None,
                config=pipeline.section("feature_governance"),
                candidate_order=selected_columns,
            )
            admitted_columns = [column for column in feature_admission.get("admitted_columns", []) if column in X_fit_model.columns]
            if admitted_columns:
                selected_columns = admitted_columns
                family_summary = summarize_feature_families(fold_feature_blocks, columns=selected_columns)
                X_fit_model = X_fit.loc[:, selected_columns]
                X_test_model = X_test.loc[:, selected_columns]
                X_val_model = X_val.loc[:, selected_columns] if X_val is not None else None
                fold_feature_metadata = filter_feature_metadata(feature_admission.get("feature_metadata", {}), selected_columns)

            fold_feature_governance.append(
                {
                    "fold": fold,
                    "split_id": split_id,
                    "summary": feature_admission.get("summary", {}),
                    "admitted_columns": list(feature_admission.get("admitted_columns", [])),
                    "training_columns": list(selected_columns),
                    "rejected_columns": list(feature_admission.get("rejected_columns", [])),
                    "retired_columns": list(feature_admission.get("retired_columns", [])),
                }
            )

            fold_feature_selection.append(
                {
                    "fold": fold,
                    "split_id": split_id,
                    "input_features": int(X_train.shape[1]),
                    "selected_features": int(len(selected_columns)),
                    "top_mi_scores": (selection_result.report.get("top_mi_scores", {}) if selection_result is not None else {}),
                    "family_summary": family_summary,
                    "endogenous_only_selected": family_summary.get("endogenous_only", False),
                }
            )

            X_train_primary = X_fit_model
            y_train_primary = y_fit
            w_train_primary = w_fit
            labels_train_primary = labels_fit.loc[X_fit_model.index]
            if binary_primary:
                binary_mask = y_fit.ne(0)
                X_train_primary = X_fit_model.loc[binary_mask]
                y_train_primary = y_fit.loc[binary_mask]
                w_train_primary = w_fit.loc[binary_mask]
                labels_train_primary = labels_fit.loc[binary_mask]
            w_train_primary = _combine_class_balance_weights(y_train_primary, w_train_primary)

            sampling_metadata = _build_training_sampling_metadata(
                labels_train_primary,
                close_all,
                w_fit.loc[X_train_primary.index],
                config,
            )
            model, bootstrap_report = train_model(
                X_train_primary,
                y_train_primary,
                sample_weight=w_train_primary,
                model_type=config.get("type", "gbm"),
                model_params=config.get("params"),
                sampling_metadata=sampling_metadata,
                return_report=True,
            )
            fold_bootstrap.append(
                {
                    "fold": fold,
                    "split_id": split_id,
                    **bootstrap_report,
                }
            )

            metrics = evaluate_model(model, X_test_model, y_test)
            metrics["fold"] = fold
            metrics["split_id"] = split_id
            metrics["validation_method"] = validation_method
            if split_meta.get("test_blocks") is not None:
                metrics["test_blocks"] = split_meta.get("test_blocks")

            test_primary_preds = pd.Series(
                _timed_inference_call(inference_latencies_ms, model.predict, X_test_model),
                index=X_test_model.index,
            )
            test_primary_probs_raw = _timed_inference_call(
                inference_latencies_ms,
                predict_probability_frame,
                model,
                X_test_model,
            )
            test_primary_probs = test_primary_probs_raw.copy()

            primary_calibrator = None
            meta_calibrator = None
            calibration_cutoff_timestamp = (
                X_val_model.index[0]
                if X_val_model is not None and not X_val_model.empty
                else (X_test_model.index[0] if not X_test_model.empty else None)
            )
            prior_oos_trade_outcomes, calibration_policy = _select_causal_prior_trade_outcomes(
                oos_trade_outcomes,
                validation_method=validation_method,
                calibration_cutoff_timestamp=calibration_cutoff_timestamp,
            )
            prior_oos_trade_count = _count_realized_trades(prior_oos_trade_outcomes)
            # Extract regime / volatility context columns for meta-model enrichment.
            # Cap at 8 columns to avoid the context overshadowing the primary signal.
            _META_CTX_PREFIXES = ("regime", "trend_regime", "volatility_regime", "liquidity_regime")
            _META_CTX_SUFFIXES = ("_atr_pct", "_vol_zscore", "_bw_zscore")
            meta_context_cols = [
                col for col in fold_frame.columns
                if any(col.startswith(p) for p in _META_CTX_PREFIXES)
                or any(col.endswith(s) for s in _META_CTX_SUFFIXES)
            ][:8]
            meta_context_frame = fold_frame[meta_context_cols] if meta_context_cols else None

            meta_model = _train_inner_meta_model(
                X_fit_model,
                y_fit,
                w_fit,
                config,
                labels=labels_fit,
                close=close_all,
                trade_outcome_builder=lambda predictions, cutoff_timestamp=X_train.index[-1]: _build_execution_trade_outcomes(
                    pipeline,
                    predictions,
                    holding_bars=holding_bars,
                    cutoff_timestamp=cutoff_timestamp,
                ),
                context_frame=meta_context_frame.reindex(X_fit_model.index) if meta_context_frame is not None else None,
            )
            fit_primary_preds = pd.Series(
                _timed_inference_call(inference_latencies_ms, model.predict, X_fit_model),
                index=X_fit_model.index,
            )
            fold_avg_win = default_avg_win
            fold_avg_loss = default_avg_loss
            sizing_trade_outcomes = None
            sizing_stats_source = "defaults"
            active_kelly_trade_count = int(prior_oos_trade_count)
            causal_calibration_rows = int(calibration_policy.get("causal_trade_rows", 0))
            signal_policy_context = {
                "source": (
                    "validation_unavailable_static_fallback"
                    if signal_policy_builder.mode == "validation_calibrated"
                    else "static_cost_math"
                ),
                "calibration_rows": 0,
                "kelly_trade_count": int(active_kelly_trade_count),
                "causal_calibration_rows": int(causal_calibration_rows),
                "causal_cutoff_timestamp": calibration_policy.get("causal_cutoff_timestamp"),
                "causal_prior_rows": int(calibration_policy.get("causal_trade_rows", 0)),
                "calibration_policy": calibration_policy.get("policy_name"),
                "cross_fold_borrowing_allowed": bool(calibration_policy.get("allow_cross_fold_borrowing", False)),
            }
            signal_policy_report = signal_policy_builder.build(
                avg_win=fold_avg_win,
                avg_loss=fold_avg_loss,
                trade_outcomes=None,
                calibration_context=signal_policy_context,
            )
            tuned_signal_params = dict(signal_policy_report["params"])
            policy_backtest = None
            if X_val_model is not None and not X_val_model.empty:
                val_primary_preds = pd.Series(
                    _timed_inference_call(inference_latencies_ms, model.predict, X_val_model),
                    index=X_val_model.index,
                )
                val_primary_probs_raw = _timed_inference_call(
                    inference_latencies_ms,
                    predict_probability_frame,
                    model,
                    X_val_model,
                )
                val_primary_probs, primary_calibrator = _calibrate_primary_probability_frame(
                    val_primary_probs_raw,
                    y_val,
                    sample_weight=w_val,
                    calibrator_config=config.get("calibration_params"),
                )

                X_meta_val = build_meta_feature_frame(
                    val_primary_preds,
                    val_primary_probs_raw,
                    context=meta_context_frame.reindex(X_val_model.index) if meta_context_frame is not None else None,
                )
                val_meta_prob_raw = pd.Series(
                    _timed_inference_call(
                        inference_latencies_ms,
                        _positive_class_probability,
                        meta_model,
                        X_meta_val,
                    ),
                    index=X_val_model.index,
                )
                val_trade_outcomes = _build_execution_trade_outcomes(
                    pipeline,
                    val_primary_preds,
                    holding_bars=holding_bars,
                    cutoff_timestamp=X_val_model.index[-1],
                )
                val_profitability_target = _resolve_profitability_target(
                    val_primary_preds,
                    y_val,
                    trade_outcomes=val_trade_outcomes,
                    labels=labels_val,
                )
                val_meta_prob, meta_calibrator = _calibrate_binary_probability_series(
                    val_meta_prob_raw,
                    val_profitability_target,
                    sample_weight=w_val,
                    calibrator_config=config.get("meta_calibration_params"),
                )

                sizing_trade_outcomes = val_trade_outcomes
                validation_trade_count = _count_realized_trades(val_trade_outcomes)
                active_kelly_trade_count = int(prior_oos_trade_count + validation_trade_count)
                causal_calibration_rows = int(calibration_policy.get("causal_trade_rows", 0) + len(val_trade_outcomes))
                fold_avg_win, fold_avg_loss = _estimate_trade_outcome_stats(
                    sizing_trade_outcomes,
                    default_avg_win,
                    default_avg_loss,
                    pooled_trade_outcomes=prior_oos_trade_outcomes if not prior_oos_trade_outcomes.empty else None,
                    shrinkage_alpha=shrinkage_alpha if not prior_oos_trade_outcomes.empty else None,
                )
                sizing_stats_source = "validation_shrunk" if not prior_oos_trade_outcomes.empty else "validation"

                signal_policy_context = {
                    "source": "validation_trade_outcomes",
                    "calibration_rows": int(len(X_val_model)),
                    "kelly_trade_count": int(active_kelly_trade_count),
                    "causal_calibration_rows": int(causal_calibration_rows),
                    "causal_cutoff_timestamp": calibration_policy.get("causal_cutoff_timestamp"),
                    "causal_prior_rows": int(calibration_policy.get("causal_trade_rows", 0)),
                    "calibration_policy": calibration_policy.get("policy_name"),
                    "cross_fold_borrowing_allowed": bool(calibration_policy.get("allow_cross_fold_borrowing", False)),
                }
                signal_policy_report = signal_policy_builder.build(
                    avg_win=fold_avg_win,
                    avg_loss=fold_avg_loss,
                    trade_outcomes=sizing_trade_outcomes if signal_policy_builder.mode == "validation_calibrated" else None,
                    calibration_context=signal_policy_context,
                )
                tuned_signal_params = dict(signal_policy_report["params"])
                # Single diagnostic backtest on validation data. This is reporting only,
                # not a second search loop over policy parameters.
                val_signal_state = _build_signal_state(
                    val_primary_preds,
                    val_primary_probs,
                    val_meta_prob,
                    {**signal_config, **tuned_signal_params},
                    fold_avg_win,
                    fold_avg_loss,
                    holding_bars,
                    kelly_trade_count=active_kelly_trade_count,
                )
                val_close_for_bt = _resolve_backtest_valuation_close(pipeline, X_val_model.index)
                if val_close_for_bt is not None and len(val_close_for_bt) > 0:
                    policy_backtest = run_backtest(
                        close=val_close_for_bt.loc[val_signal_state["continuous_signals"].index],
                        signals=val_signal_state["continuous_signals"],
                        equity=backtest_config.get("equity", 10_000.0),
                        fee_rate=backtest_config.get("fee_rate", 0.001),
                        slippage_rate=backtest_config.get("slippage_rate", 0.0),
                        signal_delay_bars=_resolve_signal_delay_bars(backtest_config),
                        execution_prices=(
                            _resolve_backtest_execution_prices(pipeline, X_val_model.index).loc[
                                val_signal_state["continuous_signals"].index
                            ]
                            if _resolve_backtest_execution_prices(pipeline, X_val_model.index) is not None
                            else None
                        ),
                        **(_resolve_backtest_runtime_kwargs(pipeline, X_val_model.index) or {}),
                    )
            elif not prior_oos_trade_outcomes.empty:
                sizing_trade_outcomes = prior_oos_trade_outcomes
                active_kelly_trade_count = int(prior_oos_trade_count)
                causal_calibration_rows = int(calibration_policy.get("causal_trade_rows", 0))
                fold_avg_win, fold_avg_loss = _estimate_trade_outcome_stats(
                    sizing_trade_outcomes,
                    default_avg_win,
                    default_avg_loss,
                )
                sizing_stats_source = "prior_oos_pooled"
                signal_policy_context = {
                    "source": "prior_oos_trade_outcomes",
                    "calibration_rows": int(len(sizing_trade_outcomes)),
                    "kelly_trade_count": int(active_kelly_trade_count),
                    "causal_calibration_rows": int(causal_calibration_rows),
                    "causal_cutoff_timestamp": calibration_policy.get("causal_cutoff_timestamp"),
                    "causal_prior_rows": int(calibration_policy.get("causal_trade_rows", 0)),
                    "calibration_policy": calibration_policy.get("policy_name"),
                    "cross_fold_borrowing_allowed": bool(calibration_policy.get("allow_cross_fold_borrowing", False)),
                }
                signal_policy_report = signal_policy_builder.build(
                    avg_win=fold_avg_win,
                    avg_loss=fold_avg_loss,
                    trade_outcomes=sizing_trade_outcomes if signal_policy_builder.mode == "validation_calibrated" else None,
                    calibration_context=signal_policy_context,
                )
                tuned_signal_params = dict(signal_policy_report["params"])

            test_primary_probs = _apply_primary_probability_calibrator(test_primary_probs_raw, primary_calibrator)

            block_diagnostics = compute_feature_block_diagnostics(
                model,
                X_train_primary,
                X_test_model,
                y_test,
                feature_blocks=fold_feature_blocks,
                baseline_metrics=metrics,
            )
            fold_block_diagnostics.append(block_diagnostics)
            fold_family_diagnostics.append(
                compute_feature_family_diagnostics(
                    model,
                    X_train_primary,
                    X_test_model,
                    y_test,
                    feature_blocks=fold_feature_blocks,
                    baseline_metrics=metrics,
                )
            )

            X_meta_test = build_meta_feature_frame(
                test_primary_preds,
                test_primary_probs_raw,
                context=meta_context_frame.reindex(X_test_model.index) if meta_context_frame is not None else None,
            )
            meta_prob_test_raw = pd.Series(
                _timed_inference_call(
                    inference_latencies_ms,
                    _positive_class_probability,
                    meta_model,
                    X_meta_test,
                ),
                index=X_test_model.index,
            )
            meta_prob_test = pd.Series(
                apply_binary_probability_calibrator(meta_calibrator, meta_prob_test_raw.to_numpy()),
                index=X_test_model.index,
                dtype=float,
            ).clip(0.0, 1.0) if meta_calibrator is not None else meta_prob_test_raw.clip(0.0, 1.0)

            fold_signal_config = dict(signal_config)
            fold_signal_config.update(tuned_signal_params)
            signal_state = _build_signal_state(
                test_primary_preds,
                test_primary_probs,
                meta_prob_test,
                fold_signal_config,
                fold_avg_win,
                fold_avg_loss,
                holding_bars,
                kelly_trade_count=active_kelly_trade_count,
            )

            fold_backtest = None
            fold_close = _resolve_backtest_valuation_close(pipeline, X_test_model.index)
            fold_execution_prices = _resolve_backtest_execution_prices(pipeline, X_test_model.index)
            fold_runtime_kwargs = _resolve_backtest_runtime_kwargs(pipeline, X_test_model.index) or {}
            if fold_close is not None and len(fold_close) > 0:
                fold_backtest = run_backtest(
                    close=fold_close.loc[signal_state["continuous_signals"].index],
                    signals=signal_state["continuous_signals"],
                    equity=backtest_config.get("equity", 10_000.0),
                    fee_rate=backtest_config.get("fee_rate", 0.001),
                    slippage_rate=backtest_config.get("slippage_rate", 0.0),
                    signal_delay_bars=_resolve_signal_delay_bars(backtest_config),
                    execution_prices=(
                        fold_execution_prices.loc[signal_state["continuous_signals"].index]
                        if fold_execution_prices is not None
                        else None
                    ),
                    **fold_runtime_kwargs,
                )

            fold_metrics.append(metrics)
            if fold_backtest is not None:
                fold_backtest_reports.append(fold_backtest)
                fold_backtests.append(
                    {
                        "fold": fold,
                        "split_id": split_id,
                        "validation_method": validation_method,
                        "net_profit_pct": fold_backtest.get("net_profit_pct"),
                        "sharpe_ratio": fold_backtest.get("sharpe_ratio"),
                        "max_drawdown": fold_backtest.get("max_drawdown"),
                        "total_trades": fold_backtest.get("total_trades"),
                    }
                )
            fold_signal_policy.append(
                {
                    "fold": fold,
                    "split_id": split_id,
                    "mode": signal_policy_report["mode"],
                    "params": tuned_signal_params,
                    "policy_quality": {
                        **signal_policy_report["policy_quality"],
                        "diagnostic_backtest": (
                            {
                                "net_profit_pct": policy_backtest.get("net_profit_pct"),
                                "sharpe_ratio": policy_backtest.get("sharpe_ratio"),
                                "max_drawdown": policy_backtest.get("max_drawdown"),
                                "total_trades": policy_backtest.get("total_trades"),
                            }
                            if policy_backtest is not None
                            else None
                        ),
                    },
                    "backtest": (
                        {
                            "net_profit_pct": policy_backtest.get("net_profit_pct"),
                            "sharpe_ratio": policy_backtest.get("sharpe_ratio"),
                            "max_drawdown": policy_backtest.get("max_drawdown"),
                            "total_trades": policy_backtest.get("total_trades"),
                        }
                        if policy_backtest is not None
                        else None
                    ),
                }
            )
            test_trade_outcomes = _build_execution_trade_outcomes(
                pipeline,
                test_primary_preds,
                holding_bars=holding_bars,
                cutoff_timestamp=X_test_model.index[-1],
            )
            signal_decay_segments.append(
                _build_signal_decay_segment(
                    pipeline,
                    signal_index=X_test_model.index,
                    predictions=test_primary_preds,
                    direction_edge=signal_state["direction_edge"],
                    event_signals=signal_state["event_signals"],
                    regime_frame=regime_frame.reindex(X_test_model.index) if not regime_frame.empty else None,
                )
            )
            oos_paths.append(
                {
                    "fold": fold,
                    "split_id": split_id,
                    "validation_method": validation_method,
                    "train_blocks": split_meta.get("train_blocks"),
                    "test_blocks": split_meta.get("test_blocks"),
                    "predictions": test_primary_preds,
                    "primary_probabilities": test_primary_probs,
                    "meta_prob": meta_prob_test,
                    "profitability_prob": meta_prob_test,
                    "direction_edge": signal_state["direction_edge"],
                    "confidence": signal_state["confidence"],
                    "expected_trade_edge": signal_state["expected_trade_edge"],
                    "position_size": signal_state["position_size"],
                    "kelly_size": signal_state["kelly_size"],
                    "event_signals": signal_state["event_signals"],
                    "continuous_signals": signal_state["continuous_signals"],
                    "signals": signal_state["signals"],
                    "avg_win_used": float(fold_avg_win),
                    "avg_loss_used": float(fold_avg_loss),
                    "kelly_trade_count": signal_state["kelly_trade_count"],
                    "used_flat_kelly_fallback": signal_state["used_flat_kelly_fallback"],
                    "signal_params": {**fold_signal_config},
                    "signal_policy": dict(signal_policy_report["policy_quality"]),
                    "sizing_stats_source": sizing_stats_source,
                    "causal_calibration_rows": int(signal_policy_context.get("causal_calibration_rows", 0)),
                    "causal_cutoff_timestamp": signal_policy_context.get("causal_cutoff_timestamp"),
                    "calibration_policy": signal_policy_context.get("calibration_policy"),
                    "cross_fold_borrowing_allowed": signal_policy_context.get("cross_fold_borrowing_allowed"),
                }
            )
            last_model = model
            last_meta = meta_model
            last_primary_calibrator = primary_calibrator
            last_meta_calibrator = meta_calibrator
            last_selected_columns = list(selected_columns)
            last_regime_fit_index = X_fit.index.copy()
            last_test_index = X_test_model.index.copy()
            last_signal_params = {**fold_signal_config}
            last_signal_policy = dict(signal_policy_report["policy_quality"])
            last_avg_win = fold_avg_win
            last_avg_loss = fold_avg_loss
            last_kelly_trade_count = active_kelly_trade_count
            oos_predictions.append(test_primary_preds)
            oos_probabilities.append(test_primary_probs)
            oos_meta_prob.append(meta_prob_test)
            oos_direction_edge.append(signal_state["direction_edge"])
            oos_confidence.append(signal_state["confidence"])
            oos_kelly_size.append(signal_state["kelly_size"])
            oos_expected_trade_edge.append(signal_state["expected_trade_edge"])
            oos_event_signals.append(signal_state["event_signals"])
            oos_continuous_signals.append(signal_state["continuous_signals"])
            oos_signals.append(signal_state["signals"])
            oos_trade_outcomes.append(test_trade_outcomes)

        if last_model is None or last_meta is None:
            raise RuntimeError("No validation splits were generated; adjust the split configuration.")

        avg_accuracy = _average_fold_metric(fold_metrics, "accuracy")
        avg_f1 = _average_fold_metric(fold_metrics, "f1_macro")
        avg_directional_accuracy = _average_fold_metric(fold_metrics, "directional_accuracy")
        avg_directional_f1 = _average_fold_metric(fold_metrics, "directional_f1_macro")
        avg_log_loss = _average_fold_metric(fold_metrics, "log_loss")
        avg_brier_score = _average_fold_metric(fold_metrics, "brier_score")
        avg_calibration_error = _average_fold_metric(fold_metrics, "calibration_error")
        last_fold_train_end = last_regime_fit_index[-1] if len(last_regime_fit_index) > 0 else None
        aligned_safe_index = X.index[X.index > last_fold_train_end] if last_fold_train_end is not None else X.index[:0]
        if validation_method == "walk_forward":
            oos_predictions = pd.concat(oos_predictions).sort_index()
            oos_probabilities = pd.concat(oos_probabilities).sort_index().reindex(oos_predictions.index)
            oos_meta_prob = pd.concat(oos_meta_prob).sort_index().reindex(oos_predictions.index)
            oos_direction_edge = pd.concat(oos_direction_edge).sort_index().reindex(oos_predictions.index)
            oos_confidence = pd.concat(oos_confidence).sort_index().reindex(oos_predictions.index)
            oos_kelly_size = pd.concat(oos_kelly_size).sort_index().reindex(oos_predictions.index)
            oos_expected_trade_edge = pd.concat(oos_expected_trade_edge).sort_index().reindex(oos_predictions.index)
            oos_event_signals = pd.concat(oos_event_signals).sort_index().reindex(oos_predictions.index)
            oos_continuous_signals = pd.concat(oos_continuous_signals).sort_index().reindex(oos_predictions.index)
            oos_signals = pd.concat(oos_signals).sort_index().reindex(oos_predictions.index)
        else:
            oos_predictions = None
            oos_probabilities = None
            oos_meta_prob = None
            oos_direction_edge = None
            oos_confidence = None
            oos_kelly_size = None
            oos_expected_trade_edge = None
            oos_event_signals = None
            oos_continuous_signals = None
            oos_signals = None
        feature_diagnostics = summarize_feature_block_diagnostics(fold_block_diagnostics)
        feature_family_diagnostics = summarize_feature_family_diagnostics(fold_family_diagnostics)
        feature_family_selection = _summarize_fold_family_selection(fold_feature_selection)
        selected_feature_metadata = filter_feature_metadata(
            pipeline.state.get("feature_metadata") or derive_feature_metadata(pipeline.state.get("feature_blocks", {})),
            last_selected_columns,
        )
        feature_portability_diagnostics = evaluate_feature_portability(
            selected_feature_metadata,
            top_features=feature_diagnostics.get("top_features"),
            family_diagnostics=feature_family_diagnostics,
            config=pipeline.section("feature_governance"),
        )
        reference_integrity_report = dict(pipeline.state.get("reference_integrity_report") or {})
        feature_admission_summary = summarize_feature_admission_reports(fold_feature_governance)
        regime_ablation_summary = summarize_regime_ablation_reports(
            [row.get("ablation") for row in fold_regime]
        )
        signal_decay = _build_signal_decay_report_for_segments(pipeline, signal_decay_segments, holding_bars)
        context_ttl_report = dict(pipeline.state.get("context_ttl_report") or {})
        operational_monitoring = _build_pipeline_operational_monitoring(
            pipeline,
            backtest_reports=fold_backtest_reports,
            expected_feature_columns=last_selected_columns,
            actual_feature_columns=last_selected_columns,
            signal_decay_report=signal_decay,
            inference_latencies_ms=inference_latencies_ms,
            queue_backlog=[0] * len(inference_latencies_ms),
            scope="training",
        )
        if context_ttl_report:
            operational_monitoring = _attach_context_ttl_to_operational_monitoring(
                operational_monitoring,
                context_ttl_report,
            )
        fold_stability = _build_fold_stability_summary(
            fold_metrics,
            fold_backtests,
            policy=stability_policy,
        )
        validation_details["split_count"] = int(len(fold_metrics))
        validation_details["stability_policy"] = stability_policy

        # Estimate avg_win / avg_loss from OOS execution-aligned trade outcomes
        oos_avg_win = None
        oos_avg_loss = None
        oos_trade_count = 0
        try:
            trade_outcomes = pd.concat(oos_trade_outcomes).sort_index() if oos_trade_outcomes else pd.DataFrame()
            if not trade_outcomes.empty:
                oos_trade_count = _count_realized_trades(trade_outcomes)
                oos_avg_win, oos_avg_loss = _estimate_trade_outcome_stats(
                    trade_outcomes,
                    default_avg_win,
                    default_avg_loss,
                )
        except Exception:
            pass

        training = {
            "fold_metrics": fold_metrics,
            "avg_accuracy": avg_accuracy,
            "avg_f1_macro": avg_f1,
            "avg_directional_accuracy": avg_directional_accuracy,
            "avg_directional_f1_macro": avg_directional_f1,
            "avg_log_loss": avg_log_loss,
            "avg_brier_score": avg_brier_score,
            "avg_calibration_error": avg_calibration_error,
            "validation": validation_details,
            "fold_backtests": fold_backtests,
            "fold_stability": fold_stability,
            "headline_metrics": {
                "directional_accuracy": avg_directional_accuracy if avg_directional_accuracy is not None else avg_accuracy,
                "log_loss": avg_log_loss,
                "brier_score": avg_brier_score,
                "calibration_error": avg_calibration_error,
            },
            "last_model": last_model,
            "last_meta": last_meta,
            "last_primary_calibrator": last_primary_calibrator,
            "last_meta_calibrator": last_meta_calibrator,
            "last_selected_columns": last_selected_columns,
            "last_regime_fit_index": last_regime_fit_index,
            "oos_paths": oos_paths,
            "fallback_inference": {
                "mode": "post_final_training_only" if validation_method == "walk_forward" else "last_validation_path_only",
                "feature_selection_fit_scope": "last_fold_train_only",
                "warning_policy": "warn_and_skip_if_empty",
                "last_fold_train_end": last_fold_train_end,
                "last_fold_test_start": last_test_index[0] if len(last_test_index) > 0 else None,
                "last_fold_test_end": last_test_index[-1] if len(last_test_index) > 0 else None,
                "aligned_safe_start": aligned_safe_index[0] if len(aligned_safe_index) > 0 else None,
                "aligned_safe_end": aligned_safe_index[-1] if len(aligned_safe_index) > 0 else None,
                "aligned_safe_row_count": int(len(aligned_safe_index)),
            },
            "last_signal_params": last_signal_params,
            "signal_policy": {
                "mode": _resolve_signal_policy_mode(signal_config),
                "calibration_policy": _resolve_fold_calibration_policy(validation_method),
                "folds": fold_signal_policy,
                "last_policy_quality": last_signal_policy,
                "last_policy_params": last_signal_params,
            },
            "last_avg_win": last_avg_win,
            "last_avg_loss": last_avg_loss,
            "last_kelly_trade_count": int(last_kelly_trade_count),
            "oos_predictions": oos_predictions,
            "oos_probabilities": oos_probabilities,
            "oos_meta_prob": oos_meta_prob,
            "oos_profitability_prob": oos_meta_prob,
            "oos_direction_edge": oos_direction_edge,
            "oos_confidence": oos_confidence,
            "oos_expected_trade_edge": oos_expected_trade_edge,
            "oos_position_size": oos_kelly_size,
            "oos_kelly_size": oos_kelly_size,
            "oos_event_signals": oos_event_signals,
            "oos_continuous_signals": oos_continuous_signals,
            "oos_signals": oos_signals,
            "feature_block_diagnostics": feature_diagnostics,
            "feature_family_diagnostics": feature_family_diagnostics,
            "feature_portability_diagnostics": feature_portability_diagnostics,
            "cross_venue_integrity": reference_integrity_report,
            "signal_decay": signal_decay,
            "feature_governance": {
                "mode": "fold_local",
                "retirement": pipeline.state.get("feature_retirement", {}),
                "admission_summary": feature_admission_summary,
                "folds": fold_feature_governance,
            },
            "lookahead_guard": lookahead_guard_report,
            "operational_monitoring": operational_monitoring,
            "context_ttl_report": context_ttl_report,
            "promotion_gates": {
                "feature_portability": bool(feature_portability_diagnostics.get("promotion_pass", True)),
                "feature_admission": bool(feature_admission_summary.get("promotion_pass", True)),
                "regime_stability": bool(regime_ablation_summary.get("promotion_pass", True)),
                "operational_health": bool(operational_monitoring.get("healthy", True)),
                "cross_venue_integrity": bool(reference_integrity_report.get("promotion_pass", True)),
                "signal_decay": bool(signal_decay.get("promotion_pass", True)),
                "lookahead_guard": bool(lookahead_guard_report.get("promotion_pass", True)),
            },
            "regime": {
                "mode": "fold_local",
                "preview_provenance": pipeline.state.get("regime_provenance"),
                "preview_ablation": pipeline.state.get("regime_ablation"),
                "ablation_summary": regime_ablation_summary,
                "folds": fold_regime,
            },
            "stationarity": {
                "mode": "fold_local",
                "folds": fold_stationarity,
            },
            "feature_selection": {
                "enabled": bool(selection_config.get("enabled", True)),
                "mode": "fold_local",
                "avg_selected_features": round(float(np.mean([row["selected_features"] for row in fold_feature_selection])), 2),
                "selected_families": feature_family_selection["selected_families"],
                "avg_selected_family_counts": feature_family_selection["avg_selected_family_counts"],
                "endogenous_only_selected_any_fold": feature_family_selection["endogenous_only_selected_any_fold"],
                "endogenous_only_selected_all_folds": feature_family_selection["endogenous_only_selected_all_folds"],
                "folds": fold_feature_selection,
            },
            "bootstrap": {
                "model_type": config.get("type", "gbm"),
                "used_in_any_fold": any(row.get("sequential_bootstrap_used", False) for row in fold_bootstrap),
                "warning_count": int(sum(1 for row in fold_bootstrap if row.get("warning"))),
                "folds": fold_bootstrap,
            },
            "purging": fold_purging,
            "signal_tuning": fold_signal_policy,
            "oos_avg_win": oos_avg_win,
            "oos_avg_loss": oos_avg_loss,
            "oos_trade_count": int(oos_trade_count),
        }
        if validation_method == "cpcv":
            training["diagnostic_validation"] = {
                "method": validation_method,
                "aggregate_mode": "path_diagnostics_only",
                "path_count": int(len(oos_paths)),
                "reporting_role": "diagnostic",
            }
            executable_validation = _build_executable_validation_training(pipeline)
            if executable_validation is not None:
                training["executable_validation"] = executable_validation
        pipeline.state["training"] = training
        pipeline.state["operational_monitoring"] = operational_monitoring
        return training


class SignalsStep(PipelineStep):
    name = "generate_signals"

    def run(self, pipeline):
        training = pipeline.require("training")
        config = pipeline.section("signals")
        fallback_scope = training.get("fallback_inference")
        validation_method = training.get("validation", {}).get("method", "walk_forward")
        executable_validation = training.get("executable_validation") or {}
        executable_training = executable_validation.get("training") if executable_validation.get("enabled") else None

        if validation_method == "cpcv" and training.get("oos_paths"):
            path_results = _build_cpcv_path_results(training, pipeline, config, validation_method)

            if executable_training is not None and executable_training.get("oos_continuous_signals") is not None:
                result = _build_signal_result_from_training_payload(
                    executable_training,
                    pipeline,
                    config,
                    executable_training.get("fallback_inference"),
                    signal_source="cpcv_walk_forward_replay",
                )
                result["validation_method"] = validation_method
                result["primary_validation_method"] = executable_training.get("validation", {}).get("method", "walk_forward")
                result["diagnostic_validation"] = {
                    "method": validation_method,
                    "aggregate_mode": "path_diagnostics_only",
                    "path_count": int(len(path_results)),
                    "paths": path_results,
                }
                pipeline.state["signals"] = result
                return result

            result = {
                "validation_method": validation_method,
                "path_count": int(len(path_results)),
                "paths": path_results,
                "signal_tuning": training.get("signal_tuning", []),
                "signal_policy": training.get("signal_policy"),
                "tuned_params": training.get("last_signal_params", {}),
                "signal_source": "cpcv_oos_paths_legacy",
                "fallback_scope": fallback_scope,
                "avg_win_used": training.get("last_avg_win", config.get("avg_win", 0.02)),
                "avg_loss_used": training.get("last_avg_loss", config.get("avg_loss", 0.02)),
                "kelly_trade_count": int(training.get("oos_trade_count", 0)),
                "used_flat_kelly_fallback": False,
            }
            pipeline.state["signals"] = result
            return result

        if training.get("oos_continuous_signals") is not None:
            result = _build_signal_result_from_training_payload(
                training,
                pipeline,
                config,
                fallback_scope,
                signal_source="walk_forward_oos",
            )
            pipeline.state["signals"] = result
            return result

        prediction_series = training.get("oos_predictions")
        probability_frame = training.get("oos_probabilities")
        meta_prob_series = training.get("oos_meta_prob")
        signal_source = "recomputed_from_oos_predictions"
        if prediction_series is None or meta_prob_series is None or probability_frame is None:
            X = pipeline.require("X")
            avg_win = training.get("last_avg_win", config.get("avg_win", 0.02))
            avg_loss = training.get("last_avg_loss", config.get("avg_loss", 0.02))
            holding_bars = _resolve_signal_holding_bars(pipeline, config)
            signal_config = dict(config)
            signal_config.update(training.get("last_signal_params", {}))
            X, fallback_scope = _resolve_fallback_signal_frame(X, training)
            if X.empty:
                warnings.warn(
                    "SignalsStep fallback is restricted to post-final-training rows; no rows remain, returning empty signals.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                result = _build_empty_signal_state(
                    X.index,
                    signal_config,
                    avg_win,
                    avg_loss,
                    holding_bars,
                )
                result["signal_source"] = "post_final_training_fallback_empty"
                result["fallback_scope"] = fallback_scope
                result["signal_policy"] = training.get("signal_policy")
                pipeline.state["signals"] = result
                return result

            selected_columns = training.get("last_selected_columns") or list(X.columns)
            missing_columns = [column for column in selected_columns if column not in X.columns]
            if missing_columns:
                regime_frame, _ = _build_fold_local_regime_frame(
                    pipeline,
                    X.index,
                    fit_index=training.get("last_regime_fit_index"),
                )
                if not regime_frame.empty:
                    X = X.join(regime_frame)
            X_model = X.loc[:, selected_columns]
            model = training["last_model"]
            meta_model = training["last_meta"]
            predictions = pd.Series(model.predict(X_model), index=X_model.index)
            probability_frame_raw = predict_probability_frame(model, X_model)
            probability_frame = _apply_primary_probability_calibrator(
                probability_frame_raw,
                training.get("last_primary_calibrator"),
            )
            _META_CTX_PREFIXES = ("regime", "trend_regime", "volatility_regime", "liquidity_regime")
            _META_CTX_SUFFIXES = ("_atr_pct", "_vol_zscore", "_bw_zscore")
            signals_ctx_cols = [
                col for col in X.columns
                if any(col.startswith(p) for p in _META_CTX_PREFIXES)
                or any(col.endswith(s) for s in _META_CTX_SUFFIXES)
            ][:8]
            signals_context_frame = X[signals_ctx_cols] if signals_ctx_cols else None
            X_meta = build_meta_feature_frame(predictions, probability_frame_raw, context=signals_context_frame)
            meta_prob_raw = pd.Series(_positive_class_probability(meta_model, X_meta), index=X_model.index)
            meta_prob_series = pd.Series(
                apply_binary_probability_calibrator(
                    training.get("last_meta_calibrator"),
                    meta_prob_raw.to_numpy(),
                ),
                index=X_model.index,
                dtype=float,
            ).clip(0.0, 1.0) if training.get("last_meta_calibrator") is not None else meta_prob_raw.clip(0.0, 1.0)
            prediction_series = predictions
            signal_source = "post_final_training_fallback"

        prediction_series = prediction_series.sort_index()
        probability_frame = probability_frame.reindex(prediction_series.index).fillna(0.0)
        meta_prob_series = meta_prob_series.reindex(prediction_series.index)

        avg_win = training.get("last_avg_win", config.get("avg_win", 0.02))
        avg_loss = training.get("last_avg_loss", config.get("avg_loss", 0.02))

        holding_bars = _resolve_signal_holding_bars(pipeline, config)
        signal_config = dict(config)
        signal_config.update(training.get("last_signal_params", {}))
        result = _build_signal_state(
            prediction_series,
            probability_frame,
            meta_prob_series,
            signal_config,
            avg_win,
            avg_loss,
            holding_bars,
            kelly_trade_count=int(training.get("oos_trade_count", 0)),
        )
        result["signal_source"] = signal_source
        result["fallback_scope"] = fallback_scope
        result["signal_policy"] = training.get("signal_policy")
        pipeline.state["signals"] = result
        return result


class BacktestStep(PipelineStep):
    name = "run_backtest"

    def run(self, pipeline):
        signal_state = pipeline.require("signals")
        config = pipeline.section("backtest")

        if signal_state.get("paths") and signal_state.get("continuous_signals") is None:
            path_backtests = _run_path_backtests(pipeline, config, signal_state.get("paths", []))

            backtest = _summarize_path_backtests(path_backtests)
            backtest["paths"] = path_backtests
            backtest["signal_source"] = signal_state.get("signal_source")
            training = pipeline.state.get("training") or {}
            feature_frame = pipeline.state.get("X")
            feature_columns = list(feature_frame.columns) if isinstance(feature_frame, pd.DataFrame) else []
            monitoring_report = _build_pipeline_operational_monitoring(
                pipeline,
                backtest_reports=[row.get("backtest") for row in path_backtests],
                expected_feature_columns=training.get("last_selected_columns") or feature_columns,
                actual_feature_columns=training.get("last_selected_columns") or feature_columns,
                signal_decay_report=(training.get("signal_decay") or {}),
                scope="backtest_paths",
            )
            backtest["operational_monitoring"] = monitoring_report
            backtest["signal_decay"] = dict(training.get("signal_decay") or {})
            if pipeline.state.get("reference_integrity_report") is not None:
                backtest["cross_venue_integrity"] = dict(pipeline.state.get("reference_integrity_report") or {})
            pipeline.state["operational_monitoring"] = monitoring_report
            pipeline.state["backtest"] = backtest
            return backtest

        positions = signal_state["continuous_signals"] if config.get("use_continuous_positions", True) else signal_state["signals"]
        valuation_close = _resolve_backtest_valuation_close(pipeline, positions.index)
        execution_prices = _resolve_backtest_execution_prices(pipeline, positions.index)
        backtest = run_backtest(
            close=valuation_close,
            signals=positions,
            equity=config.get("equity", 10_000.0),
            fee_rate=config.get("fee_rate", 0.001),
            slippage_rate=config.get("slippage_rate", 0.0),
            signal_delay_bars=_resolve_signal_delay_bars(config),
            execution_prices=execution_prices,
            **_resolve_backtest_runtime_kwargs(pipeline, positions.index),
        )
        backtest["validation_method"] = signal_state.get("primary_validation_method", signal_state.get("validation_method", "walk_forward"))
        if signal_state.get("primary_validation_method") is not None and signal_state.get("validation_method") is not None:
            if signal_state.get("primary_validation_method") != signal_state.get("validation_method"):
                backtest["diagnostic_validation_method"] = signal_state.get("validation_method")
        training = pipeline.state.get("training") or {}
        feature_frame = pipeline.state.get("X")
        feature_columns = list(feature_frame.columns) if isinstance(feature_frame, pd.DataFrame) else []
        signal_decay_report = dict(training.get("signal_decay") or {})
        if not signal_decay_report:
            signal_decay_report = _build_signal_decay_report_from_signal_state(pipeline, signal_state)
        monitoring_report = _build_pipeline_operational_monitoring(
            pipeline,
            backtest_reports=[backtest],
            expected_feature_columns=training.get("last_selected_columns") or feature_columns,
            actual_feature_columns=training.get("last_selected_columns") or feature_columns,
            signal_decay_report=signal_decay_report,
            scope="backtest",
        )
        backtest["operational_monitoring"] = monitoring_report
        backtest["signal_decay"] = signal_decay_report
        if pipeline.state.get("reference_integrity_report") is not None:
            backtest["cross_venue_integrity"] = dict(pipeline.state.get("reference_integrity_report") or {})
        diagnostic_paths = (signal_state.get("diagnostic_validation") or {}).get("paths") or []
        if diagnostic_paths:
            path_backtests = _run_path_backtests(pipeline, config, diagnostic_paths)
            backtest["diagnostic_validation"] = {
                "method": (signal_state.get("diagnostic_validation") or {}).get("method", "cpcv"),
                "path_count": int(len(path_backtests)),
                "summary": _summarize_path_backtests(path_backtests),
            }
            backtest["debug"] = {
                "cpcv_path_backtests": path_backtests,
            }
        pipeline.state["operational_monitoring"] = monitoring_report
        pipeline.state["backtest"] = backtest
        return backtest


DEFAULT_STEPS = [
    FetchDataStep,
    DataQualityStep,
    IndicatorsStep,
    AutoMLStep,
    FeaturesStep,
    StationarityStep,
    RegimeStep,
    LabelsStep,
    AlignDataStep,
    FeatureSelectionStep,
    SampleWeightsStep,
    TrainModelsStep,
    SignalsStep,
    BacktestStep,
]


class ResearchPipeline:
    """Reusable stepwise research pipeline backed by config dicts."""

    def __init__(self, config, steps=None):
        self.config = dict(config)
        step_classes = steps or DEFAULT_STEPS
        self.steps = [step() if isinstance(step, type) else step for step in step_classes]
        self.step_map = {step.name: step for step in self.steps}
        self.state = {}
        self.step_results = {}

    def section(self, name):
        return self.config.get(name, {})

    def require(self, key):
        if key not in self.state:
            raise RuntimeError(f"Pipeline state {key!r} is not available yet")
        return self.state[key]

    def run_step(self, name):
        result = self.step_map[name].run(self)
        self.step_results[name] = result
        return result

    def fetch_data(self):
        return self.run_step("fetch_data")

    def check_data_quality(self):
        return self.run_step("check_data_quality")

    def run_indicators(self):
        return self.run_step("run_indicators")

    def build_features(self):
        return self.run_step("build_features")

    def run_automl(self):
        return self.run_step("run_automl")

    def check_stationarity(self):
        return self.run_step("check_stationarity")

    def detect_regimes(self):
        return self.run_step("detect_regimes")

    def build_labels(self):
        return self.run_step("build_labels")

    def align_data(self):
        return self.run_step("align_data")

    def select_features(self):
        return self.run_step("select_features")

    def compute_sample_weights(self):
        return self.run_step("compute_sample_weights")

    def train_models(self):
        return self.run_step("train_models")

    def generate_signals(self):
        return self.run_step("generate_signals")

    def run_backtest(self):
        return self.run_step("run_backtest")

    def run_drift_retraining_cycle(
        self,
        *,
        store,
        reference_features,
        symbol=None,
        current_features=None,
        reference_predictions=None,
        current_predictions=None,
        current_performance=None,
        bars_since_last_retrain=None,
        scheduled_window_open=False,
        train_challenger=None,
        drift_config=None,
        promotion_policy=None,
        current_monitoring_report=None,
        rollback_policy=None,
    ):
        resolved_symbol = symbol or self.section("data").get("symbol")
        if not resolved_symbol:
            raise ValueError("run_drift_retraining_cycle requires a symbol or data.symbol")

        resolved_current_features = current_features
        if resolved_current_features is None:
            resolved_current_features = self.state.get("X")
        if resolved_current_features is None:
            resolved_current_features = self.state.get("features")
        if resolved_current_features is None:
            raise ValueError("run_drift_retraining_cycle requires current_features or populated pipeline state")

        training = dict(self.state.get("training") or {})
        if current_predictions is None:
            current_predictions = training.get("oos_probabilities")
        if current_performance is None:
            backtest = dict(self.state.get("backtest") or {})
            equity_curve = backtest.get("equity_curve")
            if isinstance(equity_curve, pd.Series) and not equity_curve.empty:
                current_performance = equity_curve.pct_change().dropna()
        if current_monitoring_report is None:
            current_monitoring_report = (
                self.state.get("operational_monitoring")
                or (self.state.get("backtest") or {}).get("operational_monitoring")
                or training.get("operational_monitoring")
            )

        result = orchestrate_drift_retraining_cycle(
            store=store,
            symbol=resolved_symbol,
            reference_features=reference_features,
            current_features=resolved_current_features,
            reference_predictions=reference_predictions,
            current_predictions=current_predictions,
            current_performance=current_performance,
            bars_since_last_retrain=bars_since_last_retrain,
            scheduled_window_open=scheduled_window_open,
            train_challenger=train_challenger,
            drift_config=drift_config,
            promotion_policy=promotion_policy,
            current_monitoring_report=current_monitoring_report,
            rollback_policy=rollback_policy,
        )
        self.state["drift_cycle"] = result
        return result

    def run(self):
        for step in self.steps:
            self.run_step(step.name)
        return self.state["backtest"]