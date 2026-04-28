from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any


def _clone_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    return {}


def _clone_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return copy.deepcopy(value)
    if isinstance(value, tuple):
        return [copy.deepcopy(item) for item in value]
    return []


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    return bool(value)


def _as_text(value: Any, default: str | None = None) -> str | None:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _normalize_backtest_payload(payload: Any, *, evidence_class: str | None = None) -> dict[str, Any]:
    normalized = _clone_dict(payload)
    if normalized:
        normalized["evidence_class"] = _as_text(
            normalized.get("evidence_class"),
            evidence_class or "backtest_payload",
        )
    return normalized


def _merge_normalized_backtest(payload: Any, *, evidence_class: str | None = None) -> dict[str, Any]:
    merged = _clone_dict(payload)
    merged.update(_normalize_backtest_payload(payload, evidence_class=evidence_class))
    return merged


@dataclass(slots=True)
class SelectionOutcome:
    status: str = "config_error"
    eligible_trial_count: int = 0
    completed_trial_count: int = 0
    rejected_trial_count: int = 0
    top_rejection_reasons: list[str] = field(default_factory=list)
    selection_freeze: dict[str, Any] | None = None
    promotion_ready: bool = False
    selected_trial_number: int | None = None
    candidate_hash: str | None = None
    holdout_consulted_for_selection: bool = False
    evaluated_after_freeze: bool = False
    error: str | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> SelectionOutcome:
        payload = _clone_dict(payload)
        return cls(
            status=_as_text(payload.get("status"), "config_error") or "config_error",
            eligible_trial_count=_as_int(payload.get("eligible_trial_count"), 0),
            completed_trial_count=_as_int(payload.get("completed_trial_count"), 0),
            rejected_trial_count=_as_int(payload.get("rejected_trial_count"), 0),
            top_rejection_reasons=[str(value) for value in _clone_list(payload.get("top_rejection_reasons"))],
            selection_freeze=_clone_dict(payload.get("selection_freeze")) or None,
            promotion_ready=_as_bool(payload.get("promotion_ready"), False),
            selected_trial_number=(
                _as_int(payload.get("selected_trial_number"))
                if payload.get("selected_trial_number") is not None
                else None
            ),
            candidate_hash=_as_text(payload.get("candidate_hash")),
            holdout_consulted_for_selection=_as_bool(payload.get("holdout_consulted_for_selection"), False),
            evaluated_after_freeze=_as_bool(payload.get("evaluated_after_freeze"), False),
            error=_as_text(payload.get("error")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LockedHoldoutReport:
    enabled: bool = False
    reason: str | None = None
    holdout_warning: bool = False
    evaluated_once: bool = False
    evaluated_after_freeze: bool = False
    access_count: int = 0
    start_timestamp: str | None = None
    end_timestamp: str | None = None
    backtest: dict[str, Any] = field(default_factory=dict)
    evidence_class: str = "locked_holdout"

    @classmethod
    def from_payload(cls, payload: Any) -> LockedHoldoutReport:
        payload = _clone_dict(payload)
        return cls(
            enabled=_as_bool(payload.get("enabled"), False),
            reason=_as_text(payload.get("reason")),
            holdout_warning=_as_bool(payload.get("holdout_warning"), False),
            evaluated_once=_as_bool(payload.get("evaluated_once"), False),
            evaluated_after_freeze=_as_bool(payload.get("evaluated_after_freeze"), False),
            access_count=_as_int(payload.get("access_count"), 0),
            start_timestamp=_as_text(payload.get("start_timestamp")),
            end_timestamp=_as_text(payload.get("end_timestamp")),
            backtest=_normalize_backtest_payload(payload.get("backtest"), evidence_class="locked_holdout"),
            evidence_class=_as_text(payload.get("evidence_class"), "locked_holdout") or "locked_holdout",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReplicationReport:
    enabled: bool = False
    promotion_pass: bool = False
    gate_mode: str | None = None
    pass_rate: float | None = None
    min_pass_rate: float | None = None
    completed_cohort_count: int = 0
    requested_cohort_count: int = 0
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    evidence_class: str = "replication"

    @classmethod
    def from_payload(cls, payload: Any) -> ReplicationReport:
        payload = _clone_dict(payload)
        return cls(
            enabled=_as_bool(payload.get("enabled"), False),
            promotion_pass=_as_bool(payload.get("promotion_pass"), False),
            gate_mode=_as_text(payload.get("gate_mode")),
            pass_rate=payload.get("pass_rate"),
            min_pass_rate=payload.get("min_pass_rate"),
            completed_cohort_count=_as_int(payload.get("completed_cohort_count"), 0),
            requested_cohort_count=_as_int(payload.get("requested_cohort_count"), 0),
            reasons=[str(value) for value in _clone_list(payload.get("reasons"))],
            warnings=[str(value) for value in _clone_list(payload.get("warnings"))],
            evidence_class=_as_text(payload.get("evidence_class"), "replication") or "replication",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PromotionEligibilityReport:
    schema_version: int = 1
    calibration_mode: bool = False
    groups: dict[str, Any] = field(default_factory=dict)
    gate_status: dict[str, Any] = field(default_factory=dict)
    score: dict[str, Any] = field(default_factory=dict)
    raw_failures: list[str] = field(default_factory=list)
    promotion_ready: bool = False
    approved: bool = False
    blocking_failures: list[str] = field(default_factory=list)
    advisory_failures: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: Any) -> PromotionEligibilityReport:
        payload = _clone_dict(payload)
        blocking_failures = [str(value) for value in _clone_list(payload.get("blocking_failures"))]
        promotion_ready = _as_bool(payload.get("promotion_ready"), not blocking_failures)
        approved = _as_bool(payload.get("approved"), not blocking_failures)
        if blocking_failures:
            promotion_ready = False
            approved = False
        return cls(
            schema_version=_as_int(payload.get("schema_version"), 1),
            calibration_mode=_as_bool(payload.get("calibration_mode"), False),
            groups=_clone_dict(payload.get("groups")),
            gate_status=_clone_dict(payload.get("gate_status")),
            score=_clone_dict(payload.get("score")),
            raw_failures=[str(value) for value in _clone_list(payload.get("raw_failures"))],
            promotion_ready=promotion_ready,
            approved=approved,
            blocking_failures=blocking_failures,
            advisory_failures=[str(value) for value in _clone_list(payload.get("advisory_failures"))],
            reasons=[str(value) for value in _clone_list(payload.get("reasons"))],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AutoMLStudySummary:
    evidence_class: str
    best_trial_number: int | None
    best_overrides: dict[str, Any]
    selection_outcome: SelectionOutcome
    locked_holdout: LockedHoldoutReport
    replication: ReplicationReport
    promotion_eligibility_report: PromotionEligibilityReport

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selection_outcome"] = self.selection_outcome.to_dict()
        payload["locked_holdout"] = self.locked_holdout.to_dict()
        payload["replication"] = self.replication.to_dict()
        payload["promotion_eligibility_report"] = self.promotion_eligibility_report.to_dict()
        return payload


def validate_summary_contract(summary: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(summary or {}))

    selection_outcome_payload = _clone_dict(payload.get("selection_outcome"))
    locked_holdout_payload = _clone_dict(payload.get("locked_holdout"))
    replication_payload = _clone_dict(payload.get("replication"))
    promotion_report_payload = _clone_dict(payload.get("promotion_eligibility_report"))

    selection_outcome = SelectionOutcome.from_payload(selection_outcome_payload)
    locked_holdout = LockedHoldoutReport.from_payload(locked_holdout_payload)
    replication = ReplicationReport.from_payload(replication_payload)
    promotion_report = PromotionEligibilityReport.from_payload(promotion_report_payload)

    best_trial_number = payload.get("best_trial_number")
    if best_trial_number is not None:
        best_trial_number = _as_int(best_trial_number)
    payload["best_trial_number"] = best_trial_number

    best_overrides = _clone_dict(payload.get("best_overrides"))
    selection_status = selection_outcome.status
    selected_candidate = best_trial_number is not None or selection_outcome.selected_trial_number is not None
    if selected_candidate and not best_overrides:
        raise ValueError("Selected AutoML summaries must include non-empty best_overrides")
    payload["best_overrides"] = best_overrides

    normalized_selection_outcome = copy.deepcopy(selection_outcome_payload)
    normalized_selection_outcome.update(selection_outcome.to_dict())
    payload["selection_outcome"] = normalized_selection_outcome

    normalized_locked_holdout = copy.deepcopy(locked_holdout_payload)
    normalized_locked_holdout.update(locked_holdout.to_dict())
    normalized_locked_holdout["backtest"] = _merge_normalized_backtest(
        locked_holdout_payload.get("backtest"),
        evidence_class="locked_holdout",
    )
    payload["locked_holdout"] = normalized_locked_holdout

    normalized_replication = copy.deepcopy(replication_payload)
    normalized_replication.update(replication.to_dict())
    payload["replication"] = normalized_replication

    normalized_promotion_report = copy.deepcopy(promotion_report_payload)
    normalized_promotion_report.update(promotion_report.to_dict())
    normalized_promotion_report.setdefault("groups", {})
    normalized_promotion_report.setdefault("gate_status", {})
    normalized_promotion_report.setdefault("score", {})
    normalized_promotion_report.setdefault("raw_failures", [])
    normalized_promotion_report.setdefault("blocking_failures", [])
    normalized_promotion_report.setdefault("advisory_failures", [])
    if not normalized_promotion_report.get("reasons"):
        if normalized_promotion_report["blocking_failures"]:
            normalized_promotion_report["reasons"] = list(normalized_promotion_report["blocking_failures"])
        elif normalized_promotion_report["advisory_failures"]:
            normalized_promotion_report["reasons"] = list(normalized_promotion_report["advisory_failures"])
        elif normalized_promotion_report.get("approved", False):
            normalized_promotion_report["reasons"] = ["approved"]
        else:
            normalized_promotion_report["reasons"] = ["not_approved"]
    payload["promotion_eligibility_report"] = normalized_promotion_report
    payload["evidence_class"] = _as_text(payload.get("evidence_class"), "selection_evidence") or "selection_evidence"

    payload["best_backtest"] = _merge_normalized_backtest(payload.get("best_backtest"), evidence_class="outer_replay")
    validation_holdout = _clone_dict(payload.get("validation_holdout"))
    validation_holdout["evidence_class"] = _as_text(
        validation_holdout.get("evidence_class"),
        "outer_replay",
    ) or "outer_replay"
    validation_holdout["backtest"] = _merge_normalized_backtest(
        validation_holdout.get("backtest"),
        evidence_class="outer_replay",
    )
    payload["validation_holdout"] = validation_holdout

    payload["selection_evidence"] = {
        "evidence_class": "selection_evidence",
        "selection_metric": payload.get("selection_metric"),
        "selection_mode": payload.get("selection_mode"),
        "selection_outcome": selection_outcome.to_dict(),
        "selection_freeze": copy.deepcopy(payload.get("selection_freeze")),
        "best_trial_number": best_trial_number,
        "best_overrides": copy.deepcopy(best_overrides),
        "promotion_ready": _as_bool(payload.get("promotion_ready"), False),
        "promotion_reasons": [str(value) for value in _clone_list(payload.get("promotion_reasons"))],
    }
    payload["validation_replay_evidence"] = {
        "evidence_class": "outer_replay",
        "best_backtest": copy.deepcopy(payload.get("best_backtest")),
        "best_training": _clone_dict(payload.get("best_training")),
        "best_objective_diagnostics": _clone_dict(payload.get("best_objective_diagnostics")),
        "validation_holdout": copy.deepcopy(validation_holdout),
        "validation_sources": _clone_dict(payload.get("validation_sources")),
    }
    payload["locked_holdout_evidence"] = copy.deepcopy(payload["locked_holdout"])
    payload["replication_evidence"] = copy.deepcopy(payload["replication"])
    payload.setdefault("refit_artifact", None)

    if not selected_candidate and selection_status.startswith("selected"):
        raise ValueError("Selection outcome marked as selected without a selected trial number")

    summary_contract = AutoMLStudySummary(
        evidence_class=payload["evidence_class"],
        best_trial_number=best_trial_number,
        best_overrides=copy.deepcopy(best_overrides),
        selection_outcome=selection_outcome,
        locked_holdout=locked_holdout,
        replication=replication,
        promotion_eligibility_report=promotion_report,
    )
    payload["summary_contract"] = summary_contract.to_dict()
    return payload