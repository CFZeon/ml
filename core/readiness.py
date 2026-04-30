"""Operator-facing deployment readiness helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .storage import payload_sha256, read_json, write_json


_ACTION_HINTS = {
    "promotion_status": "Promote an approved champion before deployment.",
    "model_freshness": "Retrain, validate, and promote a fresh champion before deployment.",
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

_NON_BLOCKING_REASONS = {"approved"}


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


def _coerce_timestamp(value):
    if value is None or value == "":
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


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


def _evaluate_model_freshness(record, policy=None):
    config = dict(policy or {})
    require_model_freshness = bool(config.get("require_model_freshness", True))
    if not require_model_freshness:
        return {
            "passed": True,
            "available": False,
            "expired": False,
            "refresh_due": False,
            "reasons": [],
        }

    if record is None:
        return {
            "passed": False,
            "available": False,
            "expired": None,
            "refresh_due": None,
            "reasons": ["model_freshness_unavailable"],
        }

    promoted_at = _coerce_timestamp(record.get("promoted_at"))
    created_at = _coerce_timestamp(record.get("created_at"))
    anchor_timestamp = promoted_at or created_at
    if anchor_timestamp is None:
        return {
            "passed": False,
            "available": False,
            "expired": None,
            "refresh_due": None,
            "reasons": ["model_freshness_unavailable"],
        }

    max_model_age_days = _coerce_float(config.get("max_model_age_days"))
    if max_model_age_days is None:
        max_model_age_days = 28.0
    if max_model_age_days <= 0.0:
        raise ValueError("max_model_age_days must be positive")

    warn_model_age_days = _coerce_float(config.get("warn_model_age_days"))
    if warn_model_age_days is None:
        warn_model_age_days = min(21.0, max_model_age_days)
    warn_model_age_days = min(max(0.0, warn_model_age_days), max_model_age_days)

    as_of_timestamp = _coerce_timestamp(config.get("as_of_timestamp")) or pd.Timestamp.now(tz="UTC")
    age_days = max(0.0, (as_of_timestamp - anchor_timestamp).total_seconds() / (24.0 * 60.0 * 60.0))
    expires_at = anchor_timestamp + pd.Timedelta(days=max_model_age_days)
    expired = bool(as_of_timestamp >= expires_at)
    refresh_due = bool(age_days >= warn_model_age_days)
    reasons = ["model_expired"] if expired else []

    return {
        "passed": not expired,
        "available": True,
        "anchor_field": "promoted_at" if promoted_at is not None else "created_at",
        "anchor_timestamp": anchor_timestamp.isoformat(),
        "as_of_timestamp": as_of_timestamp.isoformat(),
        "age_days": float(age_days),
        "warn_model_age_days": float(warn_model_age_days),
        "max_model_age_days": float(max_model_age_days),
        "expires_at": expires_at.isoformat(),
        "expired": expired,
        "refresh_due": refresh_due,
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
        reasons = _dedupe_reasons(
            reason for reason in (guardrails.get("reasons") or []) if str(reason or "").strip() not in _NON_BLOCKING_REASONS
        )
        if approved and retrain_status == "promoted":
            reasons = []
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


def _coerce_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_equity_curve(equity_curve):
    if isinstance(equity_curve, pd.Series):
        series = pd.to_numeric(equity_curve, errors="coerce")
    elif equity_curve is None:
        return pd.Series(dtype=float)
    else:
        series = pd.to_numeric(pd.Series(equity_curve, copy=False), errors="coerce")
    return pd.Series(series, copy=False).dropna().astype(float)


def _resolve_drawdown_limit(policy, status):
    for key in ("max_drawdown_limit", "max_drawdown_pct", "drawdown_limit", "max_loss_pct"):
        value = _coerce_float(policy.get(key))
        if value is not None:
            return abs(value)
    for key in ("max_drawdown_limit", "max_drawdown_pct", "drawdown_limit", "max_loss_pct"):
        value = _coerce_float(status.get(key))
        if value is not None:
            return abs(value)
    return 0.10


def build_operational_limits_report(
    *,
    operational_limits=None,
    equity_curve=None,
    current_equity=None,
    peak_equity=None,
    policy=None,
):
    status = dict(operational_limits or {})
    resolved_policy = dict(policy or {})
    series = _coerce_equity_curve(equity_curve if equity_curve is not None else status.get("equity_curve"))
    available = bool(status) or not series.empty or current_equity is not None or peak_equity is not None
    drawdown_limit = _resolve_drawdown_limit(resolved_policy, status)
    if not available:
        return {
            "passed": False,
            "available": False,
            "healthy": None,
            "kill_switch_ready": None,
            "kill_switch_triggered": None,
            "drawdown_breached": None,
            "current_drawdown": None,
            "current_equity": None,
            "peak_equity": None,
            "max_drawdown_limit": float(drawdown_limit),
            "enforced_action": "unavailable",
            "reasons": ["operational_limits_unavailable"],
        }

    resolved_current_equity = _coerce_float(current_equity)
    if resolved_current_equity is None:
        resolved_current_equity = _coerce_float(status.get("current_equity"))
    resolved_peak_equity = _coerce_float(peak_equity)
    if resolved_peak_equity is None:
        resolved_peak_equity = _coerce_float(status.get("peak_equity"))

    if not series.empty:
        if resolved_current_equity is None:
            resolved_current_equity = float(series.iloc[-1])
        if resolved_peak_equity is None:
            resolved_peak_equity = float(series.cummax().iloc[-1])

    current_drawdown = None
    explicit_equity_inputs = equity_curve is not None or current_equity is not None or peak_equity is not None
    if explicit_equity_inputs and resolved_current_equity is not None and resolved_peak_equity and resolved_peak_equity > 0:
        current_drawdown = float(resolved_current_equity / resolved_peak_equity - 1.0)
    if current_drawdown is None:
        for key in ("current_drawdown", "current_drawdown_pct", "drawdown"):
            current_drawdown = _coerce_float(status.get(key))
            if current_drawdown is not None:
                break
    if current_drawdown is None and resolved_current_equity is not None and resolved_peak_equity and resolved_peak_equity > 0:
        current_drawdown = float(resolved_current_equity / resolved_peak_equity - 1.0)

    drawdown_breached = bool(current_drawdown is not None and current_drawdown <= -float(drawdown_limit))
    kill_switch_ready = bool(status.get("kill_switch_ready", status.get("kill_switch_enabled", False)))
    kill_switch_triggered = bool(status.get("kill_switch_triggered", status.get("triggered", False)) or drawdown_breached)

    healthy_flag = status.get("healthy")
    if healthy_flag is None:
        healthy = True
    else:
        healthy = bool(healthy_flag)
    healthy = bool(healthy and not drawdown_breached and not kill_switch_triggered)

    reasons = list(status.get("reasons") or [])
    if drawdown_breached:
        reasons.append("drawdown_limit_breached")
    if kill_switch_triggered:
        reasons.append("kill_switch_triggered")
    if not healthy and not reasons:
        reasons.append("operational_limits_not_healthy")
    if not kill_switch_ready:
        reasons.append("kill_switch_not_ready")

    return {
        "passed": bool(healthy and kill_switch_ready and not kill_switch_triggered and not drawdown_breached),
        "available": True,
        "healthy": healthy,
        "kill_switch_ready": kill_switch_ready,
        "kill_switch_triggered": kill_switch_triggered,
        "drawdown_breached": drawdown_breached,
        "current_drawdown": current_drawdown,
        "current_equity": resolved_current_equity,
        "peak_equity": resolved_peak_equity,
        "max_drawdown_limit": float(drawdown_limit),
        "enforced_action": "flatten_and_hold" if kill_switch_triggered else "monitor",
        "reasons": _dedupe_reasons(reasons),
    }


def _evaluate_operational_limits(operational_limits=None, release_request=None):
    request = dict(release_request or {})
    requested_stage = str(request.get("requested_stage") or "research_certified")
    requires_limits = requested_stage in {"micro_capital", "scaled_capital"}

    status = build_operational_limits_report(operational_limits=operational_limits)
    if not status.get("available", False):
        return {
            "passed": not requires_limits,
            "available": False,
            "healthy": None,
            "kill_switch_ready": None,
            "kill_switch_triggered": None,
            "drawdown_breached": None,
            "current_drawdown": None,
            "max_drawdown_limit": status.get("max_drawdown_limit"),
            "reasons": [] if not requires_limits else list(status.get("reasons") or ["operational_limits_unavailable"]),
        }

    active_breach = bool(status.get("drawdown_breached", False) or status.get("kill_switch_triggered", False))
    passed = bool(status.get("passed", False)) if (requires_limits or active_breach) else True
    return {
        "passed": passed,
        "available": True,
        "healthy": bool(status.get("healthy", False)),
        "kill_switch_ready": bool(status.get("kill_switch_ready", False)),
        "kill_switch_triggered": bool(status.get("kill_switch_triggered", False)),
        "drawdown_breached": bool(status.get("drawdown_breached", False)),
        "current_drawdown": status.get("current_drawdown"),
        "max_drawdown_limit": status.get("max_drawdown_limit"),
        "enforced_action": status.get("enforced_action"),
        "reasons": list(status.get("reasons") or []),
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


def _coerce_paper_observation_frame(paper_observations=None):
    if isinstance(paper_observations, pd.DataFrame):
        return paper_observations.copy()
    if paper_observations is None:
        return pd.DataFrame()
    return pd.DataFrame(list(paper_observations))


def _resolve_observation_timestamps(frame):
    if "timestamp" in frame.columns:
        return pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if "observed_at" in frame.columns:
        return pd.to_datetime(frame["observed_at"], utc=True, errors="coerce")
    if isinstance(frame.index, pd.DatetimeIndex):
        return pd.Series(pd.to_datetime(frame.index, utc=True, errors="coerce"), index=frame.index)
    return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")


def _resolve_observation_counts(frame, *column_names):
    for column_name in column_names:
        if column_name in frame.columns:
            return pd.to_numeric(frame[column_name], errors="coerce").fillna(0.0).clip(lower=0.0)
    return pd.Series(0.0, index=frame.index, dtype=float)


def _weighted_observation_average(frame, weights, column_name):
    if column_name not in frame.columns:
        return None

    values = pd.to_numeric(frame[column_name], errors="coerce")
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return None
    aligned_weights = weights.loc[valid]
    return float((values.loc[valid] * aligned_weights).sum() / aligned_weights.sum())


def _sum_observation_flags(frame, *column_names):
    counts = _resolve_observation_counts(frame, *column_names)
    if counts.empty:
        return 0
    return int(counts.sum())


def _resolve_observation_duration_days(frame, timestamps):
    explicit_duration = _resolve_observation_counts(frame, "duration_days")
    if explicit_duration.sum() > 0:
        return float(explicit_duration.sum())

    valid_timestamps = timestamps.dropna().sort_values()
    if len(valid_timestamps) >= 2:
        deltas = valid_timestamps.diff().dropna()
        inferred_step = deltas.median() if not deltas.empty else pd.Timedelta(0)
        coverage = valid_timestamps.iloc[-1] - valid_timestamps.iloc[0]
        if inferred_step > pd.Timedelta(0):
            coverage += inferred_step
        return float(coverage.total_seconds() / (24.0 * 60.0 * 60.0))
    if len(valid_timestamps) == 1:
        return 1.0
    return 0.0


def build_paper_trading_report(*, certified_expectations=None, paper_observations=None, policy=None):
    expectations = dict(certified_expectations or {})
    policy = dict(policy or {})
    frame = _coerce_paper_observation_frame(paper_observations)
    if frame.empty:
        return {
            "mode": str(policy.get("mode", "paper")),
            "passed": False,
            "reasons": ["paper_observations_unavailable"],
            "duration_days": 0.0,
            "data_breaches": 0,
            "funding_breaches": 0,
            "kill_switch_triggers": 0,
            "certified_expectations": expectations,
            "paper_metrics": {
                "mode": str(policy.get("mode", "paper")),
                "duration_days": 0.0,
                "data_breaches": 0,
                "funding_breaches": 0,
                "kill_switch_triggers": 0,
                "observation_count": 0,
                "weighted_trade_count": 0.0,
                "start_timestamp": None,
                "end_timestamp": None,
            },
            "observation_summary": {
                "observation_count": 0,
                "weighted_trade_count": 0.0,
                "start_timestamp": None,
                "end_timestamp": None,
            },
            "policy": {
                "min_duration_days": float(policy.get("min_duration_days", 28.0)),
                "max_slippage_error": float(policy.get("max_slippage_error", 0.25)),
                "max_fill_ratio_degradation": float(policy.get("max_fill_ratio_degradation", 0.15)),
            },
        }

    timestamps = _resolve_observation_timestamps(frame)
    valid_timestamps = timestamps.notna()
    if valid_timestamps.any():
        frame = frame.loc[valid_timestamps].copy()
        timestamps = timestamps.loc[valid_timestamps]
        frame["_timestamp"] = timestamps
        frame = frame.sort_values("_timestamp")
        timestamps = frame["_timestamp"]

    weights = _resolve_observation_counts(frame, "trade_count", "fill_count", "order_count")
    if float(weights.sum()) <= 0.0:
        weights = pd.Series(1.0, index=frame.index, dtype=float)

    mode = str(
        policy.get("mode")
        or frame.get("mode", pd.Series(dtype=str)).dropna().astype(str).iloc[0]
        if "mode" in frame.columns and not frame.get("mode", pd.Series(dtype=str)).dropna().empty
        else "paper"
    )

    paper_metrics = {
        "mode": mode,
        "duration_days": _resolve_observation_duration_days(frame, timestamps),
        "modeled_slippage_bps": _weighted_observation_average(frame, weights, "modeled_slippage_bps"),
        "realized_slippage_bps": _weighted_observation_average(frame, weights, "realized_slippage_bps"),
        "modeled_fill_ratio": _weighted_observation_average(frame, weights, "modeled_fill_ratio"),
        "realized_fill_ratio": _weighted_observation_average(frame, weights, "realized_fill_ratio"),
        "data_breaches": _sum_observation_flags(frame, "data_breach", "data_breaches"),
        "funding_breaches": _sum_observation_flags(frame, "funding_breach", "funding_breaches"),
        "kill_switch_triggers": _sum_observation_flags(frame, "kill_switch_trigger", "kill_switch_triggers"),
        "observation_count": int(len(frame)),
        "weighted_trade_count": float(weights.sum()),
        "start_timestamp": None if timestamps.dropna().empty else timestamps.dropna().iloc[0].isoformat(),
        "end_timestamp": None if timestamps.dropna().empty else timestamps.dropna().iloc[-1].isoformat(),
    }
    for expectation_key in ("modeled_slippage_bps", "modeled_fill_ratio"):
        if paper_metrics.get(expectation_key) is None and expectations.get(expectation_key) is not None:
            paper_metrics[expectation_key] = float(expectations[expectation_key])

    report = build_live_calibration_report(
        certified_expectations=expectations,
        paper_metrics=paper_metrics,
        policy=policy,
    )
    report["observation_summary"] = {
        "observation_count": int(paper_metrics["observation_count"]),
        "weighted_trade_count": float(paper_metrics["weighted_trade_count"]),
        "start_timestamp": paper_metrics["start_timestamp"],
        "end_timestamp": paper_metrics["end_timestamp"],
    }
    return report


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
    model freshness, drift state, backend availability, and rollback readiness all pass.
    """

    record = _find_target_record(store, symbol, version_id=version_id)
    normalized_release_request = _normalize_release_request(release_request=release_request, policy=policy)
    components = {
        "promotion_status": _evaluate_promotion_status(record),
        "model_freshness": _evaluate_model_freshness(record, policy=policy),
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
        if str(reason or "").strip() not in _NON_BLOCKING_REASONS
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
    elif ready and stage_resolution.get("stage") == "research_certified":
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
    "build_operational_limits_report",
    "build_paper_trading_report",
    "persist_deployment_candidate_artifacts",
]