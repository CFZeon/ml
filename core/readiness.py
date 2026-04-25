"""Operator-facing deployment readiness helpers."""

from __future__ import annotations

from .storage import read_json


_ACTION_HINTS = {
    "promotion_status": "Promote an approved champion before deployment.",
    "operational_monitoring": "Refresh operational monitoring and resolve any health breaches before deployment.",
    "drift_status": "Resolve the active drift recommendation before deployment.",
    "backend": "Restore the required execution backend before deployment.",
    "rollback": "Keep at least one archived fallback champion available before deployment.",
}


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
    policy=None,
):
    """Build one operator-facing deployment readiness verdict.

    The report fails closed unless promotion status, operational monitoring,
    drift state, backend availability, and rollback readiness all pass.
    """

    record = _find_target_record(store, symbol, version_id=version_id)
    components = {
        "promotion_status": _evaluate_promotion_status(record),
        "operational_monitoring": _evaluate_operational_monitoring(record, monitoring_report=monitoring_report),
        "drift_status": _evaluate_drift_status(record, drift_cycle=drift_cycle),
        "backend": _evaluate_backend_status(backend_status=backend_status, policy=policy),
        "rollback": _evaluate_rollback_status(store, symbol),
    }
    failed_components = [name for name, component in components.items() if not bool(component.get("passed", False))]
    reasons = _dedupe_reasons(
        reason
        for component in components.values()
        for reason in component.get("reasons") or []
    )
    recommended_actions = [_ACTION_HINTS[name] for name in failed_components if name in _ACTION_HINTS]
    ready = not failed_components
    return {
        "symbol": symbol,
        "version_id": None if record is None else record.get("version_id"),
        "mode": "blocking",
        "ready": ready,
        "operator_action": "deploy" if ready else "hold",
        "reasons": reasons,
        "recommended_actions": recommended_actions,
        "summary": {
            "failed_components": failed_components,
            "current_status": None if record is None else record.get("current_status"),
        },
        "components": components,
    }


__all__ = ["build_deployment_readiness_report"]