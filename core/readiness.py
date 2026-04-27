"""Operator-facing deployment readiness helpers."""

from __future__ import annotations

from pathlib import Path

from .storage import payload_sha256, read_json, write_json


_ACTION_HINTS = {
    "promotion_status": "Promote an approved champion before deployment.",
    "paper_calibration": "Attach a green paper or shadow-live calibration report before capital release.",
    "operational_monitoring": "Refresh operational monitoring and resolve any health breaches before deployment.",
    "drift_status": "Resolve the active drift recommendation before deployment.",
    "backend": "Restore the required execution backend before deployment.",
    "operational_limits": "Verify kill switch and operational risk limits before releasing capital.",
    "rollback": "Keep at least one archived fallback champion available before deployment.",
}

_CAPITAL_RELEASE_STAGES = (
    "research_only",
    "research_certified",
    "paper_verified",
    "micro_capital",
    "scaled_capital",
)


def _dedupe_reasons(values):
    reasons = []
    for value in values or []:
        text = str(value or "").strip()
        if text and text not in reasons:
            reasons.append(text)
    return reasons


def _load_report(path):
    return dict(read_json(path) or {}) if path else {}


def _find_target_record(store, symbol, version_id=None):
    if version_id is None:
        return store.get_champion(symbol)
    for row in store.list_versions(symbol):
        if str(row.get("version_id")) == str(version_id):
            return row
    return None


def _evaluate_promotion_status(record):
    if record is None:
        return {
            "passed": False,
            "version_id": None,
            "current_status": None,
            "promotion_approved": False,
            "reasons": ["version_unavailable"],
        }

    decision = _load_report(record.get("latest_promotion_report"))
    approved = bool(decision.get("approved", record.get("promotion_ready", False)))
    current_status = str(record.get("current_status") or "")
    reasons = []
    if current_status != "champion":
        reasons.append("version_not_champion")
    if not approved:
        reasons.append("promotion_not_approved")
    return {
        "passed": not reasons,
        "version_id": record.get("version_id"),
        "current_status": current_status,
        "promotion_approved": approved,
        "reasons": reasons,
    }


def _evaluate_operational_monitoring(record, monitoring_report=None):
    report = dict(monitoring_report or {})
    if not report and record is not None:
        report = _load_report(record.get("latest_monitoring_report"))
    if not report:
        return {
            "passed": False,
            "available": False,
            "healthy": False,
            "reasons": ["operational_monitoring_unavailable"],
        }

    healthy = bool(report.get("healthy", False))
    reasons = [] if healthy else _dedupe_reasons(report.get("reasons") or ["operational_monitoring_not_healthy"])
    return {
        "passed": healthy,
        "available": True,
        "healthy": healthy,
        "reasons": reasons,
    }


def _evaluate_drift_status(record, drift_cycle=None):
    cycle = dict(drift_cycle or {})
    if cycle:
        guardrails = dict(cycle.get("drift_guardrails") or {})
        approved = bool(guardrails.get("approved", False))
        retrain_status = str(cycle.get("retrain_status") or "")
        reasons = _dedupe_reasons(guardrails.get("reasons") or [])
        if approved and retrain_status != "promoted":
            reasons = _dedupe_reasons(reasons + ["drift_retraining_recommended"])
            if retrain_status == "scheduled_window_pending":
                reasons.append("scheduled_window_pending")
            elif retrain_status == "challenger_rejected":
                reasons.append("challenger_rejected")
        passed = (not approved) or retrain_status == "promoted"
        return {
            "passed": passed,
            "available": True,
            "approved": approved,
            "retrain_status": retrain_status,
            "reasons": reasons,
        }

    drift_report = _load_report((record or {}).get("latest_drift_report"))
    if not drift_report:
        return {
            "passed": False,
            "available": False,
            "approved": None,
            "retrain_status": None,
            "reasons": ["drift_state_unavailable"],
        }

    recommendation = dict(drift_report.get("recommendation") or {})
    recommended = bool(recommendation.get("recommended", False))
    reasons = _dedupe_reasons(recommendation.get("reasons") or [])
    if recommended:
        reasons = _dedupe_reasons(reasons + ["drift_retraining_recommended"])
    return {
        "passed": not recommended,
        "available": True,
        "approved": recommended,
        "retrain_status": None,
        "reasons": reasons,
    }


def _evaluate_backend_status(backend_status=None, policy=None):
    status = dict(backend_status or {})
    config = dict(policy or {})
    required_adapter = config.get("required_adapter", "nautilus")
    adapter = status.get("adapter")
    available = bool(status.get("available", False))
    reasons = []
    if not status:
        reasons.append("backend_status_unavailable")
    if not available:
        reasons.append("backend_unavailable")
    if required_adapter and adapter != required_adapter:
        reasons.append("backend_adapter_mismatch")
    if not status:
        reasons = ["backend_status_unavailable"]
    return {
        "passed": bool(status) and available and adapter == required_adapter,
        "required_adapter": required_adapter,
        "adapter": adapter,
        "available": available,
        "reasons": _dedupe_reasons(reasons),
    }


def _normalize_release_request(release_request=None, policy=None):
    policy = dict(policy or {})
    request = dict(release_request or {})
    requested_stage = str(
        request.get("requested_stage") or policy.get("requested_stage") or "research_certified"
    ).strip().lower()
    if requested_stage not in _CAPITAL_RELEASE_STAGES:
        raise ValueError(
            "requested_stage must be one of " + ", ".join(_CAPITAL_RELEASE_STAGES)
        )

    acknowledged_stages = {
        str(stage).strip().lower()
        for stage in (request.get("acknowledged_stages") or [])
        if str(stage).strip()
    }
    manual_acknowledged = bool(request.get("manual_acknowledged", False) or "micro_capital" in acknowledged_stages)
    scaled_capital_approved = bool(
        request.get("scaled_capital_approved", False) or "scaled_capital" in acknowledged_stages
    )
    return {
        "requested_stage": requested_stage,
        "manual_acknowledged": manual_acknowledged,
        "scaled_capital_approved": scaled_capital_approved,
    }


def _evaluate_operational_limits(operational_limits=None, release_request=None):
    status = dict(operational_limits or {})
    request = dict(release_request or {})
    requested_stage = str(request.get("requested_stage") or "research_certified")
    requires_limits = requested_stage in {"micro_capital", "scaled_capital"}

    if not status:
        return {
            "passed": not requires_limits,
            "available": False,
            "healthy": None,
            "kill_switch_ready": None,
            "reasons": [] if not requires_limits else ["operational_limits_unavailable"],
        }

    healthy = bool(status.get("healthy", False))
    kill_switch_ready = bool(status.get("kill_switch_ready", status.get("kill_switch_enabled", False)))
    reasons = []
    if not healthy:
        reasons.extend(status.get("reasons") or ["operational_limits_not_healthy"])
    if not kill_switch_ready:
        reasons.append("kill_switch_not_ready")
    return {
        "passed": bool(healthy and kill_switch_ready),
        "available": True,
        "healthy": healthy,
        "kill_switch_ready": kill_switch_ready,
        "reasons": _dedupe_reasons(reasons),
        "status": status,
    }


def _resolve_capital_release_stage(components, release_request=None):
    request = dict(release_request or {})
    stage = "research_only"
    blockers = []

    if bool((components.get("promotion_status") or {}).get("passed", False)):
        stage = "research_certified"
    if bool((components.get("paper_calibration") or {}).get("passed", False)):
        stage = "paper_verified"

    requested_stage = str(request.get("requested_stage") or "research_certified")
    if requested_stage in {"paper_verified", "micro_capital", "scaled_capital"} and stage == "research_certified":
        blockers.append("paper_verification_required")

    if requested_stage in {"micro_capital", "scaled_capital"}:
        if not bool(request.get("manual_acknowledged", False)):
            blockers.append("manual_ack_required_for_micro_capital")
        if not bool((components.get("operational_limits") or {}).get("passed", False)):
            blockers.extend((components.get("operational_limits") or {}).get("reasons") or ["operational_limits_not_ready"])
        if stage == "paper_verified" and not blockers:
            stage = "micro_capital"

    if requested_stage == "scaled_capital":
        if not bool(request.get("scaled_capital_approved", False)):
            blockers.append("scaled_capital_approval_required")
        elif stage == "micro_capital" and not blockers:
            stage = "scaled_capital"

    return {
        "stage": stage,
        "requested_stage": requested_stage,
        "blockers": _dedupe_reasons(blockers),
    }


def build_deployment_candidate_id(*, experiment_id, frozen_candidate_hash, execution_profile):
    return payload_sha256(
        {
            "experiment_id": experiment_id,
            "frozen_candidate_hash": frozen_candidate_hash,
            "execution_profile": execution_profile,
        }
    )


def persist_deployment_candidate_artifacts(
    *,
    root_dir,
    deployment_candidate_id,
    paper_metrics=None,
    live_calibration_report=None,
    fill_quality=None,
    readiness=None,
):
    candidate_dir = Path(root_dir) / str(deployment_candidate_id)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    if paper_metrics is not None:
        write_json(candidate_dir / "paper_metrics.json", paper_metrics)
    if live_calibration_report is not None:
        write_json(candidate_dir / "live_calibration_report.json", live_calibration_report)
    if fill_quality is not None:
        write_json(candidate_dir / "fill_quality.json", fill_quality)
    if readiness is not None:
        write_json(candidate_dir / "readiness.json", readiness)
    return {
        "root_dir": str(candidate_dir),
        "paper_metrics": str(candidate_dir / "paper_metrics.json") if paper_metrics is not None else None,
        "live_calibration_report": str(candidate_dir / "live_calibration_report.json")
        if live_calibration_report is not None
        else None,
        "fill_quality": str(candidate_dir / "fill_quality.json") if fill_quality is not None else None,
        "readiness": str(candidate_dir / "readiness.json") if readiness is not None else None,
    }


def build_live_calibration_report(*, certified_expectations=None, paper_metrics=None, policy=None):
    expectations = dict(certified_expectations or {})
    paper = dict(paper_metrics or {})
    policy = dict(policy or {})
    min_duration_days = float(policy.get("min_duration_days", 28.0))
    max_slippage_error = float(policy.get("max_slippage_error", 0.25))
    max_fill_ratio_degradation = float(policy.get("max_fill_ratio_degradation", 0.15))
    duration_days = float(paper.get("duration_days", 0.0) or 0.0)
    data_breaches = int(paper.get("data_breaches", 0) or 0)
    funding_breaches = int(paper.get("funding_breaches", 0) or 0)
    kill_switch_triggers = int(paper.get("kill_switch_triggers", 0) or 0)
    modeled_slippage = paper.get("modeled_slippage_bps")
    realized_slippage = paper.get("realized_slippage_bps")
    modeled_fill_ratio = paper.get("modeled_fill_ratio")
    realized_fill_ratio = paper.get("realized_fill_ratio")
    slippage_error = None
    if modeled_slippage is not None and realized_slippage is not None:
        slippage_error = abs(float(realized_slippage) - float(modeled_slippage))
    fill_ratio_degradation = None
    if modeled_fill_ratio not in (None, 0):
        fill_ratio_degradation = max(0.0, float(modeled_fill_ratio) - float(realized_fill_ratio or 0.0))

    reasons = []
    if duration_days < min_duration_days:
        reasons.append("paper_window_underpowered")
    if data_breaches > 0:
        reasons.append("paper_data_breaches_present")
    if funding_breaches > 0:
        reasons.append("paper_funding_breaches_present")
    if kill_switch_triggers > 0:
        reasons.append("paper_kill_switch_triggered")
    if slippage_error is not None and slippage_error > max_slippage_error:
        reasons.append("slippage_error_above_tolerance")
    if fill_ratio_degradation is not None and fill_ratio_degradation > max_fill_ratio_degradation:
        reasons.append("fill_ratio_degradation_above_tolerance")

    return {
        "mode": str(paper.get("mode", "paper")),
        "passed": not reasons,
        "reasons": reasons,
        "duration_days": duration_days,
        "data_breaches": data_breaches,
        "funding_breaches": funding_breaches,
        "kill_switch_triggers": kill_switch_triggers,
        "certified_expectations": expectations,
        "paper_metrics": paper,
        "slippage_error_bps": slippage_error,
        "fill_ratio_degradation": fill_ratio_degradation,
        "policy": {
            "min_duration_days": min_duration_days,
            "max_slippage_error": max_slippage_error,
            "max_fill_ratio_degradation": max_fill_ratio_degradation,
        },
    }


def _evaluate_paper_calibration(record, paper_report=None, policy=None):
    policy = dict(policy or {})
    require_paper_calibration = bool(policy.get("require_paper_calibration", True))
    if not require_paper_calibration:
        return {
            "passed": True,
            "available": False,
            "mode": None,
            "reasons": [],
        }

    report = dict(paper_report or {})
    if not report and record is not None:
        report = _load_report(record.get("latest_paper_report"))
    if not report:
        return {
            "passed": False,
            "available": False,
            "mode": None,
            "reasons": ["paper_calibration_unavailable"],
        }

    passed = bool(report.get("passed", False))
    reasons = _dedupe_reasons(report.get("reasons") or ([] if passed else ["paper_calibration_failed"]))
    return {
        "passed": passed,
        "available": True,
        "mode": report.get("mode"),
        "duration_days": report.get("duration_days"),
        "reasons": reasons,
        "report": report,
    }


def _evaluate_rollback_status(store, symbol):
    rows = list(store.list_versions(symbol) or [])
    archived_versions = [row for row in rows if str(row.get("current_status") or "") == "archived"]
    champion = store.get_champion(symbol)
    reasons = []
    if champion is None:
        reasons.append("champion_unavailable")
    if not archived_versions:
        reasons.append("rollback_unavailable")
    return {
        "passed": champion is not None and bool(archived_versions),
        "available": bool(archived_versions),
        "archived_version_count": int(len(archived_versions)),
        "current_champion_version_id": None if champion is None else champion.get("version_id"),
        "reasons": reasons,
    }


def build_deployment_readiness_report(
    *,
    store,
    symbol,
    version_id=None,
    monitoring_report=None,
    drift_cycle=None,
    backend_status=None,
    paper_report=None,
    operational_limits=None,
    release_request=None,
    policy=None,
):
    """Build one operator-facing deployment readiness verdict.

    The report fails closed unless promotion status, operational monitoring,
    drift state, backend availability, and rollback readiness all pass.
    """

    record = _find_target_record(store, symbol, version_id=version_id)
    normalized_release_request = _normalize_release_request(release_request=release_request, policy=policy)
    components = {
        "promotion_status": _evaluate_promotion_status(record),
        "paper_calibration": _evaluate_paper_calibration(record, paper_report=paper_report, policy=policy),
        "operational_monitoring": _evaluate_operational_monitoring(record, monitoring_report=monitoring_report),
        "drift_status": _evaluate_drift_status(record, drift_cycle=drift_cycle),
        "backend": _evaluate_backend_status(backend_status=backend_status, policy=policy),
        "operational_limits": _evaluate_operational_limits(
            operational_limits=operational_limits,
            release_request=normalized_release_request,
        ),
        "rollback": _evaluate_rollback_status(store, symbol),
    }
    stage_resolution = _resolve_capital_release_stage(components, release_request=normalized_release_request)
    failed_components = [name for name, component in components.items() if not bool(component.get("passed", False))]
    reasons = _dedupe_reasons(
        reason
        for component in components.values()
        for reason in component.get("reasons") or []
    )
    release_blockers = _dedupe_reasons(list(reasons) + list(stage_resolution.get("blockers") or []))
    recommended_actions = [_ACTION_HINTS[name] for name in failed_components if name in _ACTION_HINTS]
    ready = not failed_components
    capital_release_eligible = bool(
        ready and stage_resolution.get("stage") in {"micro_capital", "scaled_capital"} and not release_blockers
    )
    if capital_release_eligible:
        operator_action = "deploy"
    elif ready and stage_resolution.get("stage") == "paper_verified":
        operator_action = "paper"
    elif bool((components.get("promotion_status") or {}).get("passed", False)):
        operator_action = "certify"
    else:
        operator_action = "hold"
    return {
        "symbol": symbol,
        "version_id": None if record is None else record.get("version_id"),
        "mode": "blocking",
        "ready": ready,
        "capital_release_stage": stage_resolution.get("stage"),
        "requested_stage": stage_resolution.get("requested_stage"),
        "capital_release_eligible": capital_release_eligible,
        "operator_action": operator_action,
        "reasons": reasons,
        "release_blockers": release_blockers,
        "recommended_actions": recommended_actions,
        "summary": {
            "failed_components": failed_components,
            "current_status": None if record is None else record.get("current_status"),
            "capital_release_stage": stage_resolution.get("stage"),
            "requested_stage": stage_resolution.get("requested_stage"),
        },
        "components": components,
    }


__all__ = [
    "build_deployment_candidate_id",
    "build_deployment_readiness_report",
    "build_live_calibration_report",
    "persist_deployment_candidate_artifacts",
]