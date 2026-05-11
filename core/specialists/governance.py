"""Specialist certification and degradation policy helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from ..promotion import (
    create_promotion_eligibility_report,
    finalize_promotion_eligibility_report,
    resolve_promotion_gate_mode,
    upsert_promotion_gate,
)
from .contracts import SpecialistHealthContract, SpecialistLibrarySnapshot, SpecialistPerformanceSlice
from .library import (
    apply_specialist_lifecycle_transition,
    build_specialist_selection_contract,
    normalize_specialist_library_snapshot,
)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_string_list(value: Sequence[Any] | None) -> list[str]:
    return [str(item) for item in list(value or [])]


def _normalize_health_contract(contract: SpecialistHealthContract | Mapping[str, Any]) -> SpecialistHealthContract:
    if isinstance(contract, SpecialistHealthContract):
        return SpecialistHealthContract.from_dict(contract.to_dict())
    if not isinstance(contract, Mapping):
        raise TypeError("specialist health contract must be a SpecialistHealthContract or mapping")
    model_id = str(dict(contract).get("model_id", "")).strip()
    if not model_id:
        raise ValueError("specialist health contract must include model_id")
    return SpecialistHealthContract.from_dict(contract)


def _normalize_performance_slices(
    performance_slices: Sequence[SpecialistPerformanceSlice | Mapping[str, Any]] | None,
) -> list[SpecialistPerformanceSlice]:
    normalized = []
    for item in list(performance_slices or []):
        if isinstance(item, SpecialistPerformanceSlice):
            normalized.append(SpecialistPerformanceSlice.from_dict(item.to_dict()))
            continue
        if not isinstance(item, Mapping):
            raise TypeError("specialist performance slices must be SpecialistPerformanceSlice values or mappings")
        model_id = str(dict(item).get("model_id", "")).strip()
        if not model_id:
            raise ValueError("specialist performance slices must include model_id")
        normalized.append(SpecialistPerformanceSlice.from_dict(item))
    return normalized


def _match_failure_flags(failure_flags: Sequence[str] | None, configured_flags: Sequence[str] | None) -> list[str]:
    observed = _coerce_string_list(failure_flags)
    configured = _coerce_string_list(configured_flags)
    if not observed:
        return []
    if not configured or "*" in configured:
        return observed
    allowed = set(configured)
    return [flag for flag in observed if flag in allowed]


def _evaluate_specialist_retirement_policy(
    contract: SpecialistHealthContract,
    *,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy = dict(policy or {})
    minimum_stability_score = _coerce_float(policy.get("min_stability_score"))
    maximum_decay_score = _coerce_float(policy.get("max_decay_score"))
    retiring_failure_flags = list(policy.get("retiring_failure_flags") or [])

    reasons = []
    matched_failure_flags = []
    if retiring_failure_flags:
        matched_failure_flags = _match_failure_flags(contract.failure_flags, retiring_failure_flags)
    if matched_failure_flags:
        reasons.append("retirement_failure_flags_present")
    if minimum_stability_score is not None and contract.stability_score is not None:
        if float(contract.stability_score) < minimum_stability_score:
            reasons.append("retirement_stability_score_below_threshold")
    if maximum_decay_score is not None and contract.decay_score is not None:
        if float(contract.decay_score) > maximum_decay_score:
            reasons.append("retirement_decay_score_above_threshold")

    return {
        "triggered": bool(reasons),
        "reasons": reasons,
        "matched_failure_flags": matched_failure_flags,
        "min_stability_score": minimum_stability_score,
        "max_decay_score": maximum_decay_score,
    }


def _latest_slice_for_model(
    performance_slices: Sequence[SpecialistPerformanceSlice],
    model_id: str,
    *,
    include_training: bool,
) -> SpecialistPerformanceSlice | None:
    matches = [item for item in list(performance_slices or []) if str(item.model_id) == str(model_id)]
    if not include_training:
        matches = [item for item in matches if str(item.split_role) != "training_slice"]
    if not matches:
        return None
    matches.sort(
        key=lambda item: (
            (item.metadata or {}).get("recorded_at") or "",
            (item.metadata or {}).get("window_end") or "",
            str(item.split_role),
        )
    )
    return matches[-1]


def _training_rows_for_model(
    model_id: str,
    performance_slices: Sequence[SpecialistPerformanceSlice],
    *,
    explicit_trained_rows: int | None,
) -> int:
    if explicit_trained_rows is not None:
        return int(explicit_trained_rows)
    training_slice = _latest_slice_for_model(performance_slices, model_id, include_training=True)
    if training_slice is None or str(training_slice.split_role) != "training_slice":
        return 0
    metric_rows = (training_slice.metric_summary or {}).get("trained_rows")
    if metric_rows is not None:
        try:
            return int(metric_rows)
        except (TypeError, ValueError):
            return int(training_slice.row_count)
    return int(training_slice.row_count)


def _replace_snapshot(snapshot: SpecialistLibrarySnapshot, *, metadata: Mapping[str, Any] | None = None) -> SpecialistLibrarySnapshot:
    payload = snapshot.to_dict()
    if metadata is not None:
        payload["metadata"] = dict(metadata or {})
    updated = SpecialistLibrarySnapshot.from_dict(payload)
    updated_payload = updated.to_dict()
    updated_metadata = dict(updated_payload.get("metadata") or {})
    updated_metadata["selection_contract"] = build_specialist_selection_contract(updated)
    updated_payload["metadata"] = updated_metadata
    return SpecialistLibrarySnapshot.from_dict(updated_payload)


def evaluate_specialist_certification_policy(
    health_contract: SpecialistHealthContract | Mapping[str, Any],
    *,
    trained_rows: int | None = None,
    performance_slices: Sequence[SpecialistPerformanceSlice | Mapping[str, Any]] | None = None,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = _normalize_health_contract(health_contract)
    policy = dict(policy or {})
    performance = _normalize_performance_slices(performance_slices)
    minimum_training_rows = int(policy.get("min_training_rows", 0))
    minimum_stability_score = _coerce_float(policy.get("min_stability_score"))
    maximum_decay_score = _coerce_float(policy.get("max_decay_score"))
    blocking_failure_flags = list(policy.get("blocking_failure_flags") or ["*"])
    resolved_trained_rows = _training_rows_for_model(
        contract.model_id,
        performance,
        explicit_trained_rows=trained_rows,
    )

    report = create_promotion_eligibility_report(
        metadata={
            "policy_type": "specialist_certification",
            "model_id": str(contract.model_id),
        }
    )
    report["model_id"] = str(contract.model_id)
    report["policy_type"] = "specialist_certification"
    report["fallback_only"] = bool(contract.fallback_only)

    report = upsert_promotion_gate(
        report,
        group="specialist_certification",
        name="minimum_training_rows",
        passed=bool(resolved_trained_rows >= minimum_training_rows),
        mode=resolve_promotion_gate_mode(policy, "minimum_training_rows"),
        measured=resolved_trained_rows,
        threshold=minimum_training_rows,
        reason=None if resolved_trained_rows >= minimum_training_rows else "minimum_training_rows_not_met",
        details={"model_id": str(contract.model_id)},
    )

    if minimum_stability_score is not None:
        report = upsert_promotion_gate(
            report,
            group="specialist_certification",
            name="stability_score",
            passed=bool(contract.stability_score is not None and float(contract.stability_score) >= minimum_stability_score),
            mode=resolve_promotion_gate_mode(policy, "stability_score"),
            measured=contract.stability_score,
            threshold=minimum_stability_score,
            reason=(
                None
                if contract.stability_score is not None and float(contract.stability_score) >= minimum_stability_score
                else "stability_score_below_threshold"
            ),
            details={"model_id": str(contract.model_id)},
        )

    if maximum_decay_score is not None:
        report = upsert_promotion_gate(
            report,
            group="specialist_certification",
            name="decay_score",
            passed=bool(contract.decay_score is not None and float(contract.decay_score) <= maximum_decay_score),
            mode=resolve_promotion_gate_mode(policy, "decay_score"),
            measured=contract.decay_score,
            threshold=maximum_decay_score,
            reason=(
                None
                if contract.decay_score is not None and float(contract.decay_score) <= maximum_decay_score
                else "decay_score_above_threshold"
            ),
            details={"model_id": str(contract.model_id)},
        )

    matched_failure_flags = _match_failure_flags(contract.failure_flags, blocking_failure_flags)
    report = upsert_promotion_gate(
        report,
        group="specialist_certification",
        name="failure_flags",
        passed=not bool(matched_failure_flags),
        mode=resolve_promotion_gate_mode(policy, "failure_flags"),
        measured=list(contract.failure_flags or []),
        threshold=list(blocking_failure_flags),
        reason=None if not matched_failure_flags else "failure_flags_present",
        details={"matched_failure_flags": matched_failure_flags},
    )

    finalized = finalize_promotion_eligibility_report(report)
    finalized["trained_rows"] = resolved_trained_rows
    finalized["recommended_state"] = "certified" if finalized.get("approved", False) else "candidate"
    return finalized


def evaluate_specialist_degradation_policy(
    health_contract: SpecialistHealthContract | Mapping[str, Any],
    *,
    performance_slices: Sequence[SpecialistPerformanceSlice | Mapping[str, Any]] | None = None,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = _normalize_health_contract(health_contract)
    policy = dict(policy or {})
    performance = _normalize_performance_slices(performance_slices)
    minimum_stability_score = _coerce_float(policy.get("min_stability_score"))
    maximum_decay_score = _coerce_float(policy.get("max_decay_score"))
    minimum_monitoring_rows = int(policy.get("min_monitoring_rows", 0))
    metric_minimums = dict(policy.get("metric_minimums") or {})
    degrading_failure_flags = list(policy.get("degrading_failure_flags") or ["*"])
    latest_monitoring_slice = _latest_slice_for_model(performance, contract.model_id, include_training=False)

    report = create_promotion_eligibility_report(
        metadata={
            "policy_type": "specialist_degradation",
            "model_id": str(contract.model_id),
        }
    )
    report["model_id"] = str(contract.model_id)
    report["policy_type"] = "specialist_degradation"
    report["fallback_only"] = bool(contract.fallback_only)

    if minimum_stability_score is not None:
        report = upsert_promotion_gate(
            report,
            group="specialist_degradation",
            name="stability_score",
            passed=bool(contract.stability_score is not None and float(contract.stability_score) >= minimum_stability_score),
            mode=resolve_promotion_gate_mode(policy, "stability_score"),
            measured=contract.stability_score,
            threshold=minimum_stability_score,
            reason=(
                None
                if contract.stability_score is not None and float(contract.stability_score) >= minimum_stability_score
                else "stability_score_below_threshold"
            ),
            details={"model_id": str(contract.model_id)},
        )

    if maximum_decay_score is not None:
        report = upsert_promotion_gate(
            report,
            group="specialist_degradation",
            name="decay_score",
            passed=bool(contract.decay_score is not None and float(contract.decay_score) <= maximum_decay_score),
            mode=resolve_promotion_gate_mode(policy, "decay_score"),
            measured=contract.decay_score,
            threshold=maximum_decay_score,
            reason=(
                None
                if contract.decay_score is not None and float(contract.decay_score) <= maximum_decay_score
                else "decay_score_above_threshold"
            ),
            details={"model_id": str(contract.model_id)},
        )

    matched_failure_flags = _match_failure_flags(contract.failure_flags, degrading_failure_flags)
    report = upsert_promotion_gate(
        report,
        group="specialist_degradation",
        name="failure_flags",
        passed=not bool(matched_failure_flags),
        mode=resolve_promotion_gate_mode(policy, "failure_flags"),
        measured=list(contract.failure_flags or []),
        threshold=list(degrading_failure_flags),
        reason=None if not matched_failure_flags else "failure_flags_present",
        details={"matched_failure_flags": matched_failure_flags},
    )

    if minimum_monitoring_rows > 0:
        monitoring_rows = 0 if latest_monitoring_slice is None else int(latest_monitoring_slice.row_count)
        report = upsert_promotion_gate(
            report,
            group="specialist_degradation",
            name="minimum_monitoring_rows",
            passed=bool(monitoring_rows >= minimum_monitoring_rows),
            mode=resolve_promotion_gate_mode(policy, "minimum_monitoring_rows"),
            measured=monitoring_rows,
            threshold=minimum_monitoring_rows,
            reason=None if monitoring_rows >= minimum_monitoring_rows else "minimum_monitoring_rows_not_met",
            details={"split_role": None if latest_monitoring_slice is None else latest_monitoring_slice.split_role},
        )

    for metric_name, threshold in metric_minimums.items():
        measured = None if latest_monitoring_slice is None else _coerce_float((latest_monitoring_slice.metric_summary or {}).get(metric_name))
        normalized_threshold = _coerce_float(threshold)
        report = upsert_promotion_gate(
            report,
            group="specialist_degradation",
            name=f"metric::{metric_name}",
            passed=bool(measured is not None and normalized_threshold is not None and measured >= normalized_threshold),
            mode=resolve_promotion_gate_mode(policy, f"metric::{metric_name}"),
            measured=measured,
            threshold=normalized_threshold,
            reason=None if measured is not None and normalized_threshold is not None and measured >= normalized_threshold else f"{metric_name}_below_threshold",
            details={"split_role": None if latest_monitoring_slice is None else latest_monitoring_slice.split_role},
        )

    finalized = finalize_promotion_eligibility_report(report)
    finalized["recommended_state"] = "active" if finalized.get("approved", False) else "degraded"
    finalized["latest_monitoring_split_role"] = None if latest_monitoring_slice is None else str(latest_monitoring_slice.split_role)
    return finalized


def evaluate_specialist_library_governance(snapshot: Any, *, policy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    resolved = normalize_specialist_library_snapshot(snapshot)
    if resolved is None:
        raise ValueError("specialist_library snapshot is required")

    policy = dict(policy or {})
    certification_policy = dict(policy.get("certification") or {})
    degradation_policy = dict(policy.get("degradation") or {})
    retirement_policy = dict(policy.get("retirement") or {})
    replacement_policy = dict(policy.get("replacement") or {})
    selection_contract = build_specialist_selection_contract(resolved)
    performance_by_model = {}
    for performance_slice in list(resolved.performance_slices or []):
        performance_by_model.setdefault(str(performance_slice.model_id), []).append(performance_slice)
    health_by_model = {str(health.model_id): health for health in list(resolved.health or [])}
    specialists_by_model = {str(spec.model_id): spec for spec in list(resolved.specialists or [])}

    model_reports = {}
    recommended_transitions = []
    approved_model_ids = []
    blocked_model_ids = []

    for spec in list(resolved.specialists or []):
        model_id = str(spec.model_id)
        current_state = str(((spec.metadata or {}).get("lifecycle_state") or selection_contract["lifecycle_state_by_model_id"].get(model_id) or "candidate"))
        health_contract = health_by_model.get(model_id)
        if health_contract is None:
            health_contract = SpecialistHealthContract(
                model_id=model_id,
                compatible_regimes=list(spec.compatible_regimes or []),
                fallback_only=bool((spec.metadata or {}).get("fallback_only", False)),
                metadata={"health_missing": True},
            )

        if bool(health_contract.fallback_only):
            model_reports[model_id] = {
                "current_state": current_state,
                "governance_mode": "fallback_exempt",
                "approved": True,
                "recommended_state": current_state,
                "reasons": ["fallback_exempt"],
            }
            approved_model_ids.append(model_id)
            continue

        performance = list(performance_by_model.get(model_id) or [])
        certification_report = None
        degradation_report = None
        retirement_report = None
        recommended_state = current_state
        transition = None
        approved = True

        if current_state in {"candidate", "certified"}:
            certification_report = evaluate_specialist_certification_policy(
                health_contract,
                performance_slices=performance,
                policy=certification_policy,
            )
            approved = bool(certification_report.get("approved", False))
            if current_state == "candidate" and approved:
                recommended_state = "certified"
                transition = {
                    "model_id": model_id,
                    "current_state": current_state,
                    "target_state": recommended_state,
                    "reason": "specialist_certification_passed",
                    "report_type": "certification",
                }
        elif current_state in {"active", "degraded", "shadow_challenger"}:
            degradation_report = evaluate_specialist_degradation_policy(
                health_contract,
                performance_slices=performance,
                policy=degradation_policy,
            )
            approved = bool(degradation_report.get("approved", False))
            if current_state == "active" and not approved:
                recommended_state = "degraded"
                transition = {
                    "model_id": model_id,
                    "current_state": current_state,
                    "target_state": recommended_state,
                    "reason": "specialist_degradation_triggered",
                    "report_type": "degradation",
                }
            elif current_state == "degraded" and approved:
                recommended_state = "active"
                transition = {
                    "model_id": model_id,
                    "current_state": current_state,
                    "target_state": recommended_state,
                    "reason": "specialist_recovery_passed",
                    "report_type": "degradation",
                }
            elif current_state == "shadow_challenger" and approved:
                recommended_state = "active"
                transition = {
                    "model_id": model_id,
                    "current_state": current_state,
                    "target_state": recommended_state,
                    "reason": "shadow_specialist_promoted",
                    "report_type": "degradation",
                }

        if current_state in {"active", "degraded", "shadow_challenger", "certified"}:
            retirement_report = _evaluate_specialist_retirement_policy(
                health_contract,
                policy=retirement_policy,
            )
            if retirement_report.get("triggered", False):
                approved = False
                recommended_state = "retired"
                transition = {
                    "model_id": model_id,
                    "current_state": current_state,
                    "target_state": recommended_state,
                    "reason": "specialist_retirement_triggered",
                    "report_type": "retirement",
                }

        if approved:
            approved_model_ids.append(model_id)
        else:
            blocked_model_ids.append(model_id)
        if transition is not None:
            recommended_transitions.append(transition)

        reasons = []
        if certification_report is not None:
            reasons.extend(list(certification_report.get("reasons") or []))
        if degradation_report is not None:
            reasons.extend(list(degradation_report.get("reasons") or []))
        if retirement_report is not None:
            reasons.extend(list(retirement_report.get("reasons") or []))

        model_reports[model_id] = {
            "current_state": current_state,
            "governance_mode": (
                "certification" if certification_report is not None else "degradation"
            ),
            "approved": approved,
            "recommended_state": recommended_state,
            "reasons": list(dict.fromkeys(reasons or ["approved"])),
            "certification": certification_report,
            "degradation": degradation_report,
            "retirement": retirement_report,
        }

    if replacement_policy:
        preferred_source_states = {
            str(state).strip().lower()
            for state in list(replacement_policy.get("preferred_source_states") or ["certified"])
            if str(state).strip()
        }
        for transition in list(recommended_transitions):
            if str(transition.get("target_state") or "") != "retired":
                continue
            retired_model_id = str(transition.get("model_id") or "")
            retired_spec = specialists_by_model.get(retired_model_id)
            if retired_spec is None:
                continue
            retired_regimes = {str(regime) for regime in list(retired_spec.compatible_regimes or [])}
            replacement_model_id = None
            for candidate_model_id, report in model_reports.items():
                if candidate_model_id == retired_model_id:
                    continue
                if str(report.get("current_state") or "") not in preferred_source_states:
                    continue
                if not bool(report.get("approved", False)):
                    continue
                if any(
                    str(existing.get("model_id") or "") == candidate_model_id
                    for existing in list(recommended_transitions)
                ):
                    continue
                candidate_spec = specialists_by_model.get(candidate_model_id)
                if candidate_spec is None:
                    continue
                candidate_regimes = {str(regime) for regime in list(candidate_spec.compatible_regimes or [])}
                if retired_regimes and candidate_regimes and retired_regimes.isdisjoint(candidate_regimes):
                    continue
                replacement_model_id = candidate_model_id
                break

            if replacement_model_id is None:
                continue

            recommended_transitions.append(
                {
                    "model_id": replacement_model_id,
                    "current_state": model_reports[replacement_model_id]["current_state"],
                    "target_state": "shadow_challenger",
                    "reason": "specialist_replacement_shadow_recommended",
                    "report_type": "replacement",
                    "replaces_model_id": retired_model_id,
                }
            )
            replacement_report = model_reports[replacement_model_id]
            replacement_reasons = list(replacement_report.get("reasons") or [])
            replacement_reasons.append("specialist_replacement_shadow_recommended")
            replacement_report["recommended_state"] = "shadow_challenger"
            replacement_report["replacement_for_model_id"] = retired_model_id
            replacement_report["reasons"] = list(dict.fromkeys(replacement_reasons))

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": policy,
        "model_reports": model_reports,
        "recommended_transitions": recommended_transitions,
        "summary": {
            "model_count": len(model_reports),
            "approved_model_ids": sorted(approved_model_ids),
            "blocked_model_ids": sorted(blocked_model_ids),
            "transition_model_ids": sorted({item["model_id"] for item in recommended_transitions}),
        },
    }


def apply_specialist_governance(snapshot: Any, governance_report: Mapping[str, Any] | None) -> SpecialistLibrarySnapshot:
    resolved = normalize_specialist_library_snapshot(snapshot)
    if resolved is None:
        raise ValueError("specialist_library snapshot is required")
    if governance_report is None or not isinstance(governance_report, Mapping):
        raise ValueError("specialist governance report is required")

    updated = resolved
    for transition in list(governance_report.get("recommended_transitions") or []):
        updated = apply_specialist_lifecycle_transition(
            updated,
            model_id=str(transition.get("model_id")),
            target_state=transition.get("target_state"),
            metadata={
                "governance_reason": transition.get("reason"),
                "governance_report_type": transition.get("report_type"),
                "governance_generated_at": governance_report.get("generated_at"),
            },
        )

    metadata = dict(updated.metadata or {})
    metadata["governance"] = {
        "generated_at": governance_report.get("generated_at"),
        "policy": dict(governance_report.get("policy") or {}),
        "summary": dict(governance_report.get("summary") or {}),
        "model_reports": {
            str(model_id): {
                "current_state": report.get("current_state"),
                "governance_mode": report.get("governance_mode"),
                "approved": bool(report.get("approved", False)),
                "recommended_state": report.get("recommended_state"),
                "reasons": list(report.get("reasons") or []),
            }
            for model_id, report in dict(governance_report.get("model_reports") or {}).items()
        },
    }
    return _replace_snapshot(updated, metadata=metadata)


__all__ = [
    "apply_specialist_governance",
    "evaluate_specialist_certification_policy",
    "evaluate_specialist_degradation_policy",
    "evaluate_specialist_library_governance",
]