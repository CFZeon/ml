"""Drift-triggered retraining orchestration helpers."""

from __future__ import annotations

import math

import pandas as pd

from .drift import DriftMonitor, _resolve_drift_policy, evaluate_drift_guardrails
from .registry.store import evaluate_challenger_promotion


_DEFAULT_CRITICAL_ROLLBACK_REASONS = {
    "context_ttl_breached",
    "custom_data_ttl",
    "drawdown_limit_breached",
    "feature_schema",
    "feature_schema_drift",
    "inference",
    "kill_switch_triggered",
    "l2_snapshot_age",
    "raw_data_freshness",
}


def _evaluate_request_weight_guard(config=None):
    configured = dict((dict(config or {}).get("request_weight_guard") or {}))
    if not configured:
        return {
            "configured": False,
            "allowed": True,
            "reasons": [],
            "limit": None,
            "used": None,
            "remaining": None,
            "reserve_weight": None,
            "retrain_cost": None,
            "retry_after_seconds": None,
        }

    limit = int(max(0, configured.get("limit", 0) or 0))
    used = int(max(0, configured.get("used", 0) or 0))
    retrain_cost = int(max(0, configured.get("retrain_cost", 0) or 0))
    reserve_ratio = float(max(0.0, configured.get("reserve_ratio", 0.1) or 0.0))
    reserve_weight = int(max(0, configured.get("reserve_weight", 0) or 0))
    retry_after_seconds = configured.get("retry_after_seconds")
    if retry_after_seconds is not None:
        retry_after_seconds = float(max(0.0, retry_after_seconds))

    implied_reserve = int(max(reserve_weight, math.ceil(limit * reserve_ratio))) if limit > 0 else reserve_weight
    remaining = None if limit <= 0 else max(0, int(limit - used))
    reasons = []
    allowed = True
    if retry_after_seconds is not None and retry_after_seconds > 0.0:
        allowed = False
        reasons.append("request_weight_retry_after_active")
    if limit > 0 and remaining is not None and remaining < int(retrain_cost + implied_reserve):
        allowed = False
        reasons.append("request_weight_headroom_insufficient")

    return {
        "configured": True,
        "allowed": allowed,
        "reasons": reasons,
        "limit": None if limit <= 0 else int(limit),
        "used": int(used),
        "remaining": remaining,
        "reserve_weight": int(implied_reserve),
        "retrain_cost": int(retrain_cost),
        "retry_after_seconds": retry_after_seconds,
    }


def _window_summary(data):
    if data is None:
        return {"start": None, "end": None, "sample_count": 0}

    index = getattr(data, "index", None)
    if index is None:
        size = int(len(data)) if hasattr(data, "__len__") else 0
        return {"start": None, "end": None, "sample_count": size}

    timeline = pd.DatetimeIndex(index)
    if len(timeline) == 0:
        return {"start": None, "end": None, "sample_count": 0}

    return {
        "start": timeline[0],
        "end": timeline[-1],
        "sample_count": int(len(timeline)),
    }


def _augment_drift_report(
    drift_report,
    *,
    reference_features,
    current_features,
    reference_predictions=None,
    current_predictions=None,
    reference_regimes=None,
    current_regimes=None,
    current_performance=None,
):
    report = dict(drift_report or {})
    report["reference_window"] = _window_summary(reference_features)
    report["current_window"] = _window_summary(current_features)
    report["reference_prediction_window"] = _window_summary(reference_predictions)
    report["current_prediction_window"] = _window_summary(current_predictions)
    report["reference_regime_window"] = _window_summary(reference_regimes)
    report["current_regime_window"] = _window_summary(current_regimes)
    report["performance_window"] = _window_summary(current_performance)
    return report


def _normalize_challenger_payload(payload):
    normalized = dict(payload or {})
    if normalized.get("model") is None:
        raise ValueError("Drift retraining requires challenger payload to include a fitted model")
    if not normalized.get("feature_columns"):
        raise ValueError("Drift retraining requires challenger payload to include feature_columns")
    normalized["feature_columns"] = list(normalized.get("feature_columns") or [])
    normalized["training_summary"] = dict(normalized.get("training_summary") or {})
    normalized["validation_summary"] = dict(normalized.get("validation_summary") or {})
    normalized["metadata"] = dict(normalized.get("metadata") or {})
    normalized["lineage"] = dict(normalized.get("lineage") or {})
    return normalized


def _build_challenger_summary(payload):
    validation_summary = dict(payload.get("validation_summary") or {})
    summary = {
        "sample_count": int(
            payload.get("sample_count")
            or validation_summary.get("sample_count")
            or validation_summary.get("validation_sample_count")
            or 0
        ),
        "selection_value": payload.get("selection_value", validation_summary.get("raw_objective_value")),
        "promotion_ready": payload.get("promotion_ready", validation_summary.get("promotion_ready")),
        "promotion_eligibility_report": payload.get("promotion_eligibility_report"),
    }
    summary.update(dict(payload.get("challenger_summary") or {}))
    return summary


def _evaluate_rollback_action(store, symbol, *, current_monitoring_report=None, operational_limits=None, rollback_policy=None):
    monitoring_report = dict(current_monitoring_report or {})
    limits_report = dict(operational_limits or {})
    policy = dict(rollback_policy or {})
    mode = str(policy.get("mode", "hybrid") or "hybrid").lower()
    critical_reasons = set(policy.get("critical_reasons") or _DEFAULT_CRITICAL_ROLLBACK_REASONS)
    monitoring_reasons = set(monitoring_report.get("reasons") or [])
    monitoring_reasons.update(limits_report.get("reasons") or [])
    monitoring_healthy = bool(monitoring_report.get("healthy", True)) if monitoring_report else True
    limits_healthy = bool(limits_report.get("healthy", True)) if limits_report else True
    recommended = bool(monitoring_report or limits_report) and not (monitoring_healthy and limits_healthy)
    critical = bool(monitoring_reasons & critical_reasons)
    rollback = {
        "recommended": recommended,
        "executed": False,
        "critical": critical,
        "mode": mode,
        "reasons": sorted(monitoring_reasons),
        "restored_version_id": None,
        "status": "not_required",
    }
    if not recommended:
        return rollback

    if mode == "manual":
        rollback["status"] = "manual_review_required"
        return rollback
    if mode == "hybrid" and not critical:
        rollback["status"] = "manual_review_required"
        return rollback

    try:
        restored = store.rollback(symbol)
    except RuntimeError:
        rollback["status"] = "rollback_unavailable"
        rollback["reasons"] = rollback["reasons"] + ["rollback_unavailable"]
        return rollback

    rollback["executed"] = True
    rollback["restored_version_id"] = restored.get("version_id")
    rollback["status"] = "executed"
    return rollback


def run_drift_retraining_cycle(
    *,
    store,
    symbol,
    reference_features,
    current_features,
    reference_predictions=None,
    current_predictions=None,
    reference_regimes=None,
    current_regimes=None,
    current_performance=None,
    bars_since_last_retrain=None,
    scheduled_window_open=False,
    train_challenger=None,
    drift_config=None,
    promotion_policy=None,
    current_monitoring_report=None,
    operational_limits=None,
    rollback_policy=None,
):
    resolved_drift_config = _resolve_drift_policy(drift_config)
    request_weight_guard = _evaluate_request_weight_guard(drift_config)
    champion = store.get_champion(symbol)
    monitor = DriftMonitor(
        reference_features,
        reference_predictions,
        reference_regimes=reference_regimes,
        config=resolved_drift_config,
    )
    drift_report = monitor.check(
        current_features,
        current_predictions=current_predictions,
        current_regimes=current_regimes,
        current_performance=current_performance,
        bars_since_last_retrain=bars_since_last_retrain,
    )
    drift_report = _augment_drift_report(
        drift_report,
        reference_features=reference_features,
        current_features=current_features,
        reference_predictions=reference_predictions,
        current_predictions=current_predictions,
        reference_regimes=reference_regimes,
        current_regimes=current_regimes,
        current_performance=current_performance,
    )
    drift_guardrails = evaluate_drift_guardrails(drift_report, policy=resolved_drift_config)

    if champion is not None:
        store.attach_drift_report(champion["version_id"], drift_report, symbol=symbol)
        if current_monitoring_report is not None:
            store.attach_monitoring_report(champion["version_id"], current_monitoring_report, symbol=symbol)

    result = {
        "symbol": symbol,
        "scheduled_window_open": bool(scheduled_window_open),
        "drift_report": drift_report,
        "drift_guardrails": drift_guardrails,
        "request_weight_guard": request_weight_guard,
        "retrain_status": "not_recommended",
        "candidate_version_id": None,
        "promotion_decision": None,
        "post_retrain_warmup": {
            "required": False,
            "mode": str(resolved_drift_config.get("post_retrain_warmup_mode", "paper") or "paper").lower(),
            "completed": False,
        },
        "rollback": {
            "recommended": False,
            "executed": False,
            "critical": False,
            "mode": str((rollback_policy or {}).get("mode", "hybrid") or "hybrid").lower(),
            "reasons": [],
            "restored_version_id": None,
            "status": "not_required",
        },
    }
    if not drift_guardrails.get("approved", False):
        return result

    if not scheduled_window_open:
        result["retrain_status"] = "scheduled_window_pending"
        return result

    if not request_weight_guard.get("allowed", True):
        result["retrain_status"] = "request_weight_deferred"
        return result

    if train_challenger is None:
        raise ValueError("train_challenger must be provided when a drift-approved retrain window is open")

    challenger_payload = _normalize_challenger_payload(train_challenger())
    candidate_version_id = store.register_version(
        challenger_payload["model"],
        symbol=symbol,
        feature_columns=challenger_payload["feature_columns"],
        metadata=challenger_payload.get("metadata"),
        training_summary=challenger_payload.get("training_summary"),
        validation_summary=challenger_payload.get("validation_summary"),
        locked_holdout=challenger_payload.get("locked_holdout"),
        replication=challenger_payload.get("replication"),
        promotion_eligibility_report=challenger_payload.get("promotion_eligibility_report"),
        lineage=challenger_payload.get("lineage"),
        status="challenger",
        meta_model=challenger_payload.get("meta_model"),
    )
    store.attach_drift_report(candidate_version_id, drift_report, symbol=symbol)
    challenger_monitoring_report = challenger_payload.get("monitoring_report")
    if challenger_monitoring_report is not None:
        store.attach_monitoring_report(candidate_version_id, challenger_monitoring_report, symbol=symbol)

    promotion_decision = evaluate_challenger_promotion(
        _build_challenger_summary(challenger_payload),
        champion_record=champion,
        drift_report=drift_report,
        monitoring_report=challenger_monitoring_report,
        policy=promotion_policy,
    )
    store.record_promotion_decision(candidate_version_id, promotion_decision, symbol=symbol)

    result["candidate_version_id"] = candidate_version_id
    result["promotion_decision"] = promotion_decision
    if promotion_decision.get("approved", False):
        store.promote(candidate_version_id, "champion", symbol=symbol, decision=promotion_decision)
        result["retrain_status"] = "promoted"
        result["post_retrain_warmup"] = {
            "required": bool(resolved_drift_config.get("post_retrain_warmup_required", True)),
            "mode": str(resolved_drift_config.get("post_retrain_warmup_mode", "paper") or "paper").lower(),
            "completed": False,
        }
        return result

    result["retrain_status"] = "challenger_rejected"
    result["rollback"] = _evaluate_rollback_action(
        store,
        symbol,
        current_monitoring_report=current_monitoring_report,
        operational_limits=operational_limits,
        rollback_policy=rollback_policy,
    )
    return result


__all__ = ["run_drift_retraining_cycle"]