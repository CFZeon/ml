"""Canonical promotion gate and eligibility report helpers."""

from __future__ import annotations

import copy


_VALID_GATE_MODES = {"blocking", "advisory", "disabled"}


def _coerce_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_promotion_gate_mode(policy=None, gate_name=None, default="blocking"):
    modes = dict((policy or {}).get("gate_modes") or {})
    mode = str(modes.get(gate_name, default)).lower()
    if mode not in _VALID_GATE_MODES:
        return str(default).lower()
    return mode


def create_promotion_eligibility_report(*, calibration_mode=False, score_basis=None, score_value=None, metadata=None):
    report = {
        "schema_version": 1,
        "calibration_mode": bool(calibration_mode),
        "groups": {},
        "gate_status": {},
        "score": {
            "basis": score_basis,
            "value": _coerce_float(score_value),
            "metadata": copy.deepcopy(metadata or {}),
        },
        "raw_failures": [],
        "blocking_failures": [],
        "advisory_failures": [],
        "eligible_before_post_checks": True,
        "promotion_ready": True,
        "approved": True,
        "reasons": ["approved"],
    }
    return report


def evaluate_execution_realism_gate(backtest_summary=None, policy=None):
    backtest_summary = dict(backtest_summary or {})
    policy = dict(policy or {})
    required_execution_mode = str(policy.get("required_execution_mode", "event_driven")).lower()
    execution_mode = str(backtest_summary.get("execution_mode") or "unknown").lower()
    promotion_execution_ready = bool(backtest_summary.get("promotion_execution_ready", False))
    execution_limitations = list(backtest_summary.get("execution_limitations") or [])
    passed = bool(promotion_execution_ready and execution_mode == required_execution_mode)
    return {
        "passed": passed,
        "reason": None if passed else "execution_realism_failed",
        "execution_mode": execution_mode,
        "required_execution_mode": required_execution_mode,
        "promotion_execution_ready": promotion_execution_ready,
        "execution_adapter": backtest_summary.get("execution_adapter"),
        "execution_backend": backtest_summary.get("execution_backend"),
        "execution_limitations": execution_limitations,
        "research_only": not passed,
        "new_symbol_policy": str(policy.get("new_symbol_policy", "conservative_assumptions_only")),
    }


def evaluate_stress_realism_gate(backtest_summary=None, policy=None, regime_aware_summary=None):
    backtest_summary = dict(backtest_summary or {})
    policy = dict(policy or {})
    regime_aware_summary = dict(regime_aware_summary or {})
    evaluation_mode = str(backtest_summary.get("evaluation_mode") or policy.get("evaluation_mode") or "research_only").lower()
    stress_matrix = dict(backtest_summary.get("stress_matrix") or {})
    required_scenarios = [
        str(name) for name in list(
            policy.get("required_stress_scenarios")
            or backtest_summary.get("required_stress_scenarios")
            or ["downtime", "stale_mark", "halt"]
        )
    ]
    required_control_intents = [
        str(name) for name in list(
            policy.get("required_stress_control_intents")
            or backtest_summary.get("required_stress_control_intents")
            or []
        )
    ]
    configured_scenarios = [str(name) for name in list(stress_matrix.get("scenario_names") or [])]
    configured_control_intents = [str(name) for name in list(stress_matrix.get("control_intents") or [])]
    missing_scenarios = [name for name in required_scenarios if name not in configured_scenarios]
    missing_control_intents = [name for name in required_control_intents if name not in configured_control_intents]
    configured = bool(stress_matrix.get("configured", False))

    max_stress_drawdown = _coerce_float(policy.get("max_stress_drawdown"))
    if max_stress_drawdown is None and evaluation_mode == "trade_ready":
        max_stress_drawdown = 0.35
    min_stress_fill_ratio = _coerce_float(policy.get("min_stress_fill_ratio"))
    if min_stress_fill_ratio is None and evaluation_mode == "trade_ready":
        min_stress_fill_ratio = 0.25
    min_stress_trade_count = int(policy.get("min_stress_trade_count", 1 if evaluation_mode == "trade_ready" else 0))
    require_unseen_regime_fallback_bound = bool(policy.get("require_unseen_regime_fallback_bound", False))
    max_unseen_regime_fallback_share = _coerce_float(policy.get("max_unseen_regime_fallback_share"))
    if require_unseen_regime_fallback_bound and max_unseen_regime_fallback_share is None and evaluation_mode == "trade_ready":
        max_unseen_regime_fallback_share = 0.35

    fallback_rows = int(regime_aware_summary.get("fallback_rows") or 0)
    fallback_evidence_rows = int(regime_aware_summary.get("fallback_evidence_rows") or 0)
    fallback_share = None
    if fallback_evidence_rows > 0:
        fallback_share = float(fallback_rows) / float(fallback_evidence_rows)

    missing_metrics = []
    outcome_breaches = []

    worst_drawdown = _coerce_float(stress_matrix.get("worst_max_drawdown"))
    if max_stress_drawdown is not None:
        if worst_drawdown is None:
            missing_metrics.append("worst_max_drawdown")
        elif worst_drawdown < -abs(max_stress_drawdown):
            outcome_breaches.append("stress_drawdown_above_limit")

    worst_fill_ratio = _coerce_float(stress_matrix.get("worst_fill_ratio"))
    if min_stress_fill_ratio is not None:
        if worst_fill_ratio is None:
            missing_metrics.append("worst_fill_ratio")
        elif worst_fill_ratio < float(min_stress_fill_ratio):
            outcome_breaches.append("stress_fill_ratio_below_minimum")

    worst_trade_count = stress_matrix.get("worst_trade_count")
    if min_stress_trade_count > 0:
        if worst_trade_count is None:
            missing_metrics.append("worst_trade_count")
        elif int(worst_trade_count) < int(min_stress_trade_count):
            outcome_breaches.append("stress_trade_count_below_minimum")

    if require_unseen_regime_fallback_bound:
        if not regime_aware_summary:
            missing_metrics.append("unseen_regime_fallback_share")
        elif fallback_share is None:
            missing_metrics.append("unseen_regime_fallback_share")
        elif (
            max_unseen_regime_fallback_share is not None
            and fallback_share > float(max_unseen_regime_fallback_share)
        ):
            outcome_breaches.append("unseen_regime_fallback_share_above_limit")

    if evaluation_mode != "trade_ready":
        passed = False
        reason = "research_only_evaluation"
        failure_class = "evaluation_mode_invalid"
    elif not configured:
        passed = False
        reason = "stress_scenarios_missing"
        failure_class = "scenarios_missing"
    elif missing_scenarios:
        passed = False
        reason = "stress_scenarios_incomplete"
        failure_class = "scenarios_incomplete"
    elif missing_control_intents:
        passed = False
        reason = "stress_control_intents_incomplete"
        failure_class = "scenarios_incomplete"
    elif missing_metrics:
        passed = False
        reason = "stress_metrics_incomplete"
        failure_class = "scenarios_incomplete"
    elif outcome_breaches:
        passed = False
        reason = outcome_breaches[0]
        failure_class = "stressed_outcome_unacceptable"
    else:
        passed = True
        reason = None
        failure_class = None

    return {
        "passed": passed,
        "reason": reason,
        "failure_class": failure_class,
        "evaluation_mode": evaluation_mode,
        "configured": configured,
        "configured_scenarios": configured_scenarios,
        "required_scenarios": required_scenarios,
        "missing_scenarios": missing_scenarios,
        "configured_control_intents": configured_control_intents,
        "required_control_intents": required_control_intents,
        "missing_control_intents": missing_control_intents,
        "missing_metrics": missing_metrics,
        "outcome_breaches": outcome_breaches,
        "worst_net_profit_pct": stress_matrix.get("worst_net_profit_pct"),
        "worst_sharpe_ratio": stress_matrix.get("worst_sharpe_ratio"),
        "worst_max_drawdown": stress_matrix.get("worst_max_drawdown"),
        "worst_fill_ratio": stress_matrix.get("worst_fill_ratio"),
        "worst_trade_count": stress_matrix.get("worst_trade_count"),
        "thresholds": {
            "max_stress_drawdown": max_stress_drawdown,
            "min_stress_fill_ratio": min_stress_fill_ratio,
            "min_stress_trade_count": min_stress_trade_count,
            "require_unseen_regime_fallback_bound": require_unseen_regime_fallback_bound,
            "max_unseen_regime_fallback_share": max_unseen_regime_fallback_share,
        },
        "regime_fallback": {
            "enabled": bool(regime_aware_summary.get("enabled", False)),
            "strategy": regime_aware_summary.get("strategy"),
            "fallback_rows": fallback_rows,
            "fallback_evidence_rows": fallback_evidence_rows,
            "fallback_share": fallback_share,
            "unseen_regimes": list(regime_aware_summary.get("unseen_regimes") or []),
        },
        "research_only": not passed,
    }


def set_promotion_score(report, *, basis, value, metadata=None):
    payload = copy.deepcopy(report or create_promotion_eligibility_report())
    payload["score"] = {
        "basis": basis,
        "value": _coerce_float(value),
        "metadata": copy.deepcopy(metadata or {}),
    }
    return payload


def upsert_promotion_gate(
    report,
    *,
    group,
    name,
    passed,
    mode="blocking",
    measured=None,
    threshold=None,
    reason=None,
    details=None,
):
    payload = copy.deepcopy(report or create_promotion_eligibility_report())
    groups = payload.setdefault("groups", {})
    group_payload = groups.setdefault(group, {"gates": []})
    gate = {
        "name": str(name),
        "group": str(group),
        "passed": bool(passed),
        "mode": str(mode).lower(),
        "measured": copy.deepcopy(measured),
        "threshold": copy.deepcopy(threshold),
        "reason": reason,
        "details": copy.deepcopy(details or {}),
    }
    replaced = False
    updated_gates = []
    for existing_gate in group_payload.get("gates") or []:
        if existing_gate.get("name") == gate["name"]:
            updated_gates.append(gate)
            replaced = True
        else:
            updated_gates.append(existing_gate)
    if not replaced:
        updated_gates.append(gate)
    group_payload["gates"] = updated_gates
    return payload


def finalize_promotion_eligibility_report(report):
    payload = copy.deepcopy(report or create_promotion_eligibility_report())
    calibration_mode = bool(payload.get("calibration_mode", False))
    groups = payload.setdefault("groups", {})
    gate_status = {}
    raw_failures = []
    blocking_failures = []
    advisory_failures = []

    for group_name, group_payload in groups.items():
        normalized_gates = []
        group_raw_failures = []
        group_blocking_failures = []
        group_advisory_failures = []
        group_failed_gate_names = []

        for original_gate in group_payload.get("gates") or []:
            gate = copy.deepcopy(original_gate)
            warnings = list(gate.get("warnings") or [])
            mode = str(gate.get("mode", "blocking")).lower()
            if mode not in _VALID_GATE_MODES:
                warnings.append("unknown_mode_normalized_to_blocking")
                gate["input_mode"] = mode
                mode = "blocking"
            gate["mode"] = mode
            gate["warnings"] = list(dict.fromkeys(warnings))

            failed = not bool(gate.get("passed", True))
            reported_status = str(((gate.get("details") or {}).get("status") or "")).lower()
            if mode == "disabled":
                gate["status"] = "disabled"
            elif failed and reported_status == "unknown":
                gate["status"] = "unknown"
            else:
                gate["status"] = "fail" if failed else "pass"
            gate["enforced"] = bool(mode == "blocking" and not calibration_mode)
            failure_code = gate.get("reason") or gate.get("name")

            if failed and mode != "disabled":
                group_raw_failures.append(failure_code)
                group_failed_gate_names.append(gate["name"])
                if gate["enforced"]:
                    group_blocking_failures.append(failure_code)
                else:
                    group_advisory_failures.append(failure_code)

            normalized_gates.append(gate)
            gate_status[gate["name"]] = gate

        group_payload["gates"] = normalized_gates
        group_payload["failed_gate_names"] = list(dict.fromkeys(group_failed_gate_names))
        group_payload["raw_failures"] = list(dict.fromkeys(group_raw_failures))
        group_payload["blocking_failures"] = list(dict.fromkeys(group_blocking_failures))
        group_payload["advisory_failures"] = list(dict.fromkeys(group_advisory_failures))
        group_payload["passed"] = not group_payload["blocking_failures"]

        raw_failures.extend(group_payload["raw_failures"])
        blocking_failures.extend(group_payload["blocking_failures"])
        advisory_failures.extend(group_payload["advisory_failures"])

    selection_group = groups.get("selection") or {}
    selection_ready = bool(selection_group.get("passed", True))

    pre_registry_group_names = [name for name in groups if name != "registry"]
    promotion_ready = all(bool(groups[name].get("passed", True)) for name in pre_registry_group_names)
    if not pre_registry_group_names:
        promotion_ready = True

    registry_ready = bool((groups.get("registry") or {}).get("passed", True))
    approved = bool(promotion_ready and registry_ready)

    payload["gate_status"] = gate_status
    payload["raw_failures"] = list(dict.fromkeys(raw_failures))
    payload["blocking_failures"] = list(dict.fromkeys(blocking_failures))
    payload["advisory_failures"] = list(dict.fromkeys(advisory_failures))
    payload["eligible_before_post_checks"] = selection_ready
    payload["promotion_ready"] = promotion_ready
    payload["approved"] = approved
    payload["reasons"] = list(payload["blocking_failures"] or (["approved"] if approved else payload["advisory_failures"]))
    if not payload["reasons"]:
        payload["reasons"] = ["not_approved"]
    return payload


def build_promotion_gate_check_map(report):
    return {
        str(name): bool(gate.get("passed", True))
        for name, gate in dict((report or {}).get("gate_status") or {}).items()
    }


def resolve_canonical_promotion_score(*, locked_holdout_report=None, selection_value=None, preference="locked_holdout_first"):
    preference = str(preference or "locked_holdout_first").lower()
    locked_holdout_value = _coerce_float((locked_holdout_report or {}).get("raw_objective_value"))
    selection_value = _coerce_float(selection_value)

    if preference == "selection_value_only":
        if selection_value is not None:
            return {"basis": "selection_value", "value": selection_value}
        return {"basis": None, "value": None}

    if preference == "locked_holdout_only":
        if locked_holdout_value is not None:
            return {"basis": "locked_holdout_raw_objective", "value": locked_holdout_value}
        return {"basis": None, "value": None}

    if locked_holdout_value is not None:
        return {"basis": "locked_holdout_raw_objective", "value": locked_holdout_value}
    if selection_value is not None:
        return {"basis": "selection_value", "value": selection_value}
    return {"basis": None, "value": None}


def get_promotion_score(report):
    return copy.deepcopy((report or {}).get("score") or {})


__all__ = [
    "build_promotion_gate_check_map",
    "create_promotion_eligibility_report",
    "evaluate_execution_realism_gate",
    "evaluate_stress_realism_gate",
    "finalize_promotion_eligibility_report",
    "get_promotion_score",
    "resolve_canonical_promotion_score",
    "resolve_promotion_gate_mode",
    "set_promotion_score",
    "upsert_promotion_gate",
]