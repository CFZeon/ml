"""Drift-triggered retraining orchestration helpers."""

from __future__ import annotations

import math

import pandas as pd

from .drift import DriftMonitor, _resolve_drift_policy, evaluate_drift_guardrails
from .promotion import evaluate_router_stability_gate
from .registry.store import evaluate_challenger_promotion
from .specialists.governance import evaluate_specialist_library_governance
from .storage import read_json


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


def _load_persisted_drift_monitor_state(champion_record):
    record = dict(champion_record or {})
    report_path = record.get("latest_drift_report")
    if not report_path:
        return None
    try:
        persisted = read_json(report_path)
    except Exception:
        return None
    state = dict((persisted or {}).get("drift_monitor_state") or {})
    return state or None


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


def _evaluate_library_review_policy(specialist_library, *, policy=None):
    policy = dict(policy or {})
    specialist_payload = dict(specialist_library or {})
    if not specialist_payload or not list(specialist_payload.get("specialists") or []):
        return {
            "enabled": bool(policy),
            "applicable": False,
            "recommended": False,
            "action": "not_applicable",
            "reasons": [],
            "active_model_ids": [],
            "blocked_model_ids": [],
            "transition_model_ids": [],
            "nonfallback_active_specialist_count": 0,
            "minimum_active_specialists": int(max(0, policy.get("min_active_specialists", 1) or 1)),
            "governance": None,
        }

    governance_policy = dict(policy.get("governance") or {})
    governance = evaluate_specialist_library_governance(specialist_payload, policy=governance_policy)
    model_reports = dict(governance.get("model_reports") or {})
    active_states = {"active", "degraded", "shadow_challenger"}
    fallback_model_id = specialist_payload.get("fallback_model_id")
    selection_contract = dict((specialist_payload.get("metadata") or {}).get("selection_contract") or {})
    active_model_ids = [str(model_id) for model_id in list(selection_contract.get("active_model_ids") or [])]
    nonfallback_active_model_ids = [
        model_id
        for model_id in active_model_ids
        if fallback_model_id in (None, "") or str(model_id) != str(fallback_model_id)
    ]

    blocked_model_ids = sorted(
        model_id
        for model_id, report in model_reports.items()
        if str(report.get("current_state") or "") in active_states and not bool(report.get("approved", False))
    )
    transition_model_ids = sorted(
        {
            str(transition.get("model_id"))
            for transition in list(governance.get("recommended_transitions") or [])
            if str(transition.get("current_state") or "") in active_states
            or str(transition.get("target_state") or "") in {"degraded", "retired"}
        }
    )
    minimum_active_specialists = int(max(0, policy.get("min_active_specialists", 1) or 1))

    reasons = []
    if len(nonfallback_active_model_ids) < minimum_active_specialists:
        reasons.append("active_specialist_coverage_below_minimum")
    if blocked_model_ids:
        reasons.append("blocked_specialists_present")
    if transition_model_ids:
        reasons.append("governance_transitions_recommended")

    recommended = bool(reasons)
    return {
        "enabled": True,
        "applicable": True,
        "recommended": recommended,
        "action": "review_library" if recommended else "hold",
        "reasons": reasons,
        "active_model_ids": active_model_ids,
        "blocked_model_ids": blocked_model_ids,
        "transition_model_ids": transition_model_ids,
        "nonfallback_active_specialist_count": int(len(nonfallback_active_model_ids)),
        "minimum_active_specialists": minimum_active_specialists,
        "governance": governance,
    }


def _evaluate_router_recalibration_policy(router_stability_report, *, policy=None):
    policy = dict(policy or {})
    if not router_stability_report:
        return {
            "enabled": bool(policy),
            "applicable": False,
            "recommended": False,
            "action": "not_applicable",
            "reasons": [],
            "gate": None,
        }

    gate = evaluate_router_stability_gate(
        {"router_stability_report": dict(router_stability_report or {})},
        policy=policy,
    )
    recommended = bool(gate.get("applicable", False) and not gate.get("passed", True) and gate.get("status") == "failed")
    reasons = []
    if gate.get("reason"):
        reasons.append(str(gate.get("reason")))
    if gate.get("failure_class"):
        reasons.append(str(gate.get("failure_class")))

    return {
        "enabled": True,
        "applicable": bool(gate.get("applicable", False)),
        "recommended": recommended,
        "action": "recalibrate_router" if recommended else "hold",
        "reasons": list(dict.fromkeys(reasons)),
        "gate": gate,
    }


def _evaluate_structural_invalidation_policy(drift_report, drift_guardrails, *, policy=None):
    policy = dict(policy or {})
    report = dict(drift_report or {})
    guardrails = dict(drift_guardrails or {})
    recommendation = dict(report.get("recommendation") or {})
    approved = bool(guardrails.get("approved", False))

    structural_reasons = []
    if bool(policy.get("allow_performance_drift", True)) and bool(report.get("performance_drift", False)):
        structural_reasons.append("performance_drift_detected")
    if (
        bool(policy.get("treat_joint_feature_action_drift_as_structural", True))
        and bool(report.get("feature_drift", False))
        and bool(report.get("action_drift", False))
    ):
        structural_reasons.append("joint_feature_action_drift_detected")

    maintenance_reasons = []
    if bool(policy.get("allow_model_ttl_expiry", True)) and bool(recommendation.get("maintenance_refresh_recommended", False)):
        maintenance_reasons.append("model_ttl_expired")

    discovery_reasons = []
    if bool(policy.get("discover_on_regime_drift", True)) and bool(report.get("regime_drift", False)):
        discovery_reasons.append("regime_drift_detected")
    if bool(policy.get("discover_on_feature_drift", True)) and bool(report.get("feature_drift", False)):
        discovery_reasons.append("feature_drift_detected")

    calibration_reasons = []
    if bool(policy.get("recalibrate_on_score_drift", True)) and bool(report.get("score_drift", False)):
        calibration_reasons.append("score_drift_detected")
    if bool(policy.get("recalibrate_on_action_drift", True)) and bool(report.get("action_drift", False)):
        calibration_reasons.append("action_drift_detected")

    observe_reasons = []
    if (
        not structural_reasons
        and not discovery_reasons
        and not calibration_reasons
        and not maintenance_reasons
        and int(report.get("evidence_count") or 0) > 0
    ):
        observe_reasons.append("drift_observe_only")

    retrain_recommended = bool(
        approved
        and bool(guardrails.get("structural_retrain_recommended", recommendation.get("structural_retrain_recommended", False)))
        and structural_reasons
    )
    discover_recommended = bool(
        approved
        and not retrain_recommended
        and bool(guardrails.get("drift_investigation_recommended", recommendation.get("drift_investigation_recommended", False)))
        and discovery_reasons
    )
    recalibrate_recommended = bool(
        approved
        and not retrain_recommended
        and not discover_recommended
        and bool(guardrails.get("recalibration_recommended", recommendation.get("recalibration_recommended", False)))
        and calibration_reasons
    )
    maintenance_refresh_recommended = bool(
        approved
        and not retrain_recommended
        and not discover_recommended
        and not recalibrate_recommended
        and bool(guardrails.get("maintenance_refresh_recommended", recommendation.get("maintenance_refresh_recommended", False)))
        and maintenance_reasons
    )
    observe_recommended = bool(
        approved
        and not retrain_recommended
        and not discover_recommended
        and not recalibrate_recommended
        and not maintenance_refresh_recommended
        and observe_reasons
    )

    return {
        "enabled": True,
        "applicable": approved,
        "retrain_recommended": retrain_recommended,
        "discover_recommended": discover_recommended,
        "recalibrate_recommended": recalibrate_recommended,
        "maintenance_refresh_recommended": maintenance_refresh_recommended,
        "observe_recommended": observe_recommended,
        "action": (
            "retrain"
            if retrain_recommended
            else "discover"
            if discover_recommended
            else "recalibrate"
            if recalibrate_recommended
            else "maintenance_refresh"
            if maintenance_refresh_recommended
            else "observe"
            if observe_recommended
            else "hold"
        ),
        "reasons": list(
            structural_reasons
            if retrain_recommended
            else discovery_reasons
            if discover_recommended
            else calibration_reasons
            if recalibrate_recommended
            else maintenance_reasons
            if maintenance_refresh_recommended
            else observe_reasons
        ),
        "structural_reasons": structural_reasons,
        "discovery_reasons": discovery_reasons,
        "calibration_reasons": calibration_reasons,
        "maintenance_reasons": maintenance_reasons,
        "observe_reasons": observe_reasons,
    }


def _resolve_drift_action_report(*, library_review, router_recalibration, structural_invalidation):
    if bool(library_review.get("recommended", False)):
        return {
            "recommended_action": "reroute",
            "source": "library_review",
            "reasons": list(library_review.get("reasons") or []),
        }
    if bool(router_recalibration.get("recommended", False)):
        return {
            "recommended_action": "recalibrate",
            "source": "router_recalibration",
            "reasons": list(router_recalibration.get("reasons") or []),
        }
    if bool(structural_invalidation.get("discover_recommended", False)):
        return {
            "recommended_action": "discover",
            "source": "structural_invalidation",
            "reasons": list(structural_invalidation.get("discovery_reasons") or []),
        }
    if bool(structural_invalidation.get("recalibrate_recommended", False)):
        return {
            "recommended_action": "recalibrate",
            "source": "structural_invalidation",
            "reasons": list(structural_invalidation.get("calibration_reasons") or []),
        }
    if bool(structural_invalidation.get("maintenance_refresh_recommended", False)):
        return {
            "recommended_action": "maintenance_refresh",
            "source": "structural_invalidation",
            "reasons": list(structural_invalidation.get("maintenance_reasons") or []),
        }
    if bool(structural_invalidation.get("retrain_recommended", False)):
        return {
            "recommended_action": "retrain",
            "source": "structural_invalidation",
            "reasons": list(structural_invalidation.get("structural_reasons") or []),
        }
    if bool(structural_invalidation.get("observe_recommended", False)):
        return {
            "recommended_action": "observe",
            "source": "structural_invalidation",
            "reasons": list(structural_invalidation.get("observe_reasons") or []),
        }
    return {
        "recommended_action": "hold",
        "source": None,
        "reasons": [],
    }


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
    current_router_stability_report=None,
    bars_since_last_retrain=None,
    scheduled_window_open=False,
    train_challenger=None,
    drift_config=None,
    library_review_policy=None,
    router_recalibration_policy=None,
    structural_invalidation_policy=None,
    promotion_policy=None,
    current_monitoring_report=None,
    operational_limits=None,
    rollback_policy=None,
):
    resolved_drift_config = _resolve_drift_policy(drift_config)
    request_weight_guard = _evaluate_request_weight_guard(drift_config)
    champion = store.get_champion(symbol)
    current_specialist_library = None
    if champion is not None:
        try:
            current_specialist_library = store.read_specialist_library(champion["version_id"], symbol=symbol)
        except Exception:
            current_specialist_library = None
    persisted_monitor_state = _load_persisted_drift_monitor_state(champion)
    monitor = DriftMonitor(
        reference_features,
        reference_predictions,
        reference_regimes=reference_regimes,
        config=resolved_drift_config,
        state=persisted_monitor_state,
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
    library_review = _evaluate_library_review_policy(
        current_specialist_library,
        policy=library_review_policy,
    )
    router_recalibration = _evaluate_router_recalibration_policy(
        current_router_stability_report,
        policy=router_recalibration_policy,
    )
    structural_invalidation = _evaluate_structural_invalidation_policy(
        drift_report,
        drift_guardrails,
        policy=structural_invalidation_policy,
    )
    action_report = _resolve_drift_action_report(
        library_review=library_review,
        router_recalibration=router_recalibration,
        structural_invalidation=structural_invalidation,
    )

    if champion is not None:
        store.attach_drift_report(champion["version_id"], drift_report, symbol=symbol)
        if current_monitoring_report is not None:
            store.attach_monitoring_report(champion["version_id"], current_monitoring_report, symbol=symbol)

    result = {
        "symbol": symbol,
        "scheduled_window_open": bool(scheduled_window_open),
        "drift_report": drift_report,
        "drift_guardrails": drift_guardrails,
        "library_review": library_review,
        "router_recalibration": router_recalibration,
        "structural_invalidation": structural_invalidation,
        "action_report": action_report,
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
    if library_review.get("recommended", False):
        result["retrain_status"] = "library_review_recommended"
        return result

    if router_recalibration.get("recommended", False):
        result["retrain_status"] = "router_recalibration_recommended"
        return result

    if not drift_guardrails.get("approved", False):
        return result

    if structural_invalidation.get("maintenance_refresh_recommended", False):
        result["retrain_status"] = "maintenance_refresh_recommended"
        return result

    if not structural_invalidation.get("retrain_recommended", False):
        if structural_invalidation.get("discover_recommended", False):
            result["retrain_status"] = "discovery_recommended"
        elif structural_invalidation.get("recalibrate_recommended", False):
            result["retrain_status"] = "recalibration_recommended"
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
        specialist_library=challenger_payload.get("specialist_library"),
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