"""Operations-centric monitoring for research, paper, and registry workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .registry.manifest import build_feature_schema_hash
from .storage import write_json, write_parquet_frame


_DEFAULT_POLICY = {
    "max_data_lag": None,
    "max_custom_ttl_breach_rate": 1.0,
    "max_fallback_assumption_rate": 1.0,
    "max_l2_snapshot_age": None,
    "require_l2_snapshots": False,
    "min_fill_ratio": 0.0,
    "max_fill_ratio_deterioration": np.inf,
    "max_slippage_gap_share": np.inf,
    "max_slippage_drift": np.inf,
    "max_inference_p95_ms": np.inf,
    "max_queue_backlog": np.inf,
    "require_inference_metrics": False,
    "fail_closed_on_schema_drift": True,
    "min_signal_decay_net_edge_at_delay": -np.inf,
    "min_signal_half_life_bars": -np.inf,
    "max_signal_delay_edge_deterioration": np.inf,
    "max_signal_half_life_deterioration": np.inf,
}


def _coerce_index(value):
    if value is None:
        return pd.DatetimeIndex([])
    if isinstance(value, (pd.DataFrame, pd.Series)):
        value = value.index
    return pd.DatetimeIndex(value)


def _coerce_timedelta(value):
    if value in (None, "", False):
        return None
    return pd.Timedelta(value)


def _coerce_float_list(values):
    if values is None:
        return []
    if np.isscalar(values):
        values = [values]
    return [float(value) for value in values if value is not None and np.isfinite(float(value))]


def _safe_ratio(numerator, denominator, default=0.0):
    denominator = float(denominator)
    if denominator <= 0.0 or not np.isfinite(denominator):
        return float(default)
    return float(numerator) / denominator


def _round_metric(value, digits=6):
    if isinstance(value, pd.Timedelta):
        return value
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if not np.isfinite(numeric):
        return numeric
    return round(numeric, digits)


def evaluate_raw_data_freshness(index, *, expected_end=None, expected_interval=None, max_lag=None, source_name="raw_data"):
    observed_index = _coerce_index(index)
    if observed_index.empty:
        return {
            "healthy": True,
            "observed": False,
            "source": source_name,
            "row_count": 0,
            "reason": None,
        }

    inferred_interval = _coerce_timedelta(expected_interval)
    if inferred_interval is None and len(observed_index) > 1:
        inferred_interval = observed_index.to_series().diff().dropna().median()
    observed_end = observed_index[-1]
    expected_end = pd.Timestamp(expected_end) if expected_end is not None else observed_end
    lag = max(expected_end - observed_end, pd.Timedelta(0))
    max_lag = _coerce_timedelta(max_lag)
    healthy = True if max_lag is None else lag <= max_lag
    return {
        "healthy": bool(healthy),
        "observed": True,
        "source": source_name,
        "row_count": int(len(observed_index)),
        "observed_end": observed_end,
        "expected_end": expected_end,
        "expected_interval": inferred_interval,
        "observed_lag": lag,
        "max_allowed_lag": max_lag,
        "reason": None if healthy else "raw_data_freshness_breach",
    }


def evaluate_custom_data_ttl(custom_data_report=None, policy=None):
    policy = {**_DEFAULT_POLICY, **dict(policy or {})}
    reports = list(custom_data_report or [])
    if not reports:
        return {
            "healthy": True,
            "observed": False,
            "dataset_count": 0,
            "ttl_breach_count": 0,
            "reason": None,
            "datasets": [],
        }

    dataset_rows = []
    breach_count = 0
    worst_stale_rate = 0.0
    worst_fallback_rate = 0.0
    for row in reports:
        stale_rate = float(row.get("stale_hit_rate") or 0.0)
        fallback_rate = float(row.get("fallback_assumption_rate") or 0.0)
        dataset_healthy = (
            stale_rate <= float(policy.get("max_custom_ttl_breach_rate", 1.0))
            and fallback_rate <= float(policy.get("max_fallback_assumption_rate", 1.0))
        )
        if not dataset_healthy:
            breach_count += 1
        worst_stale_rate = max(worst_stale_rate, stale_rate)
        worst_fallback_rate = max(worst_fallback_rate, fallback_rate)
        dataset_rows.append(
            {
                "name": row.get("name"),
                "healthy": bool(dataset_healthy),
                "stale_hit_rate": stale_rate,
                "fallback_assumption_rate": fallback_rate,
                "max_feature_age": row.get("max_feature_age"),
                "max_feature_age_observed": row.get("max_feature_age_observed"),
            }
        )

    healthy = breach_count == 0
    return {
        "healthy": bool(healthy),
        "observed": True,
        "dataset_count": int(len(dataset_rows)),
        "ttl_breach_count": int(breach_count),
        "worst_stale_hit_rate": worst_stale_rate,
        "worst_fallback_assumption_rate": worst_fallback_rate,
        "datasets": dataset_rows,
        "reason": None if healthy else "custom_data_ttl_breach",
    }


def evaluate_l2_snapshot_age(reference_index=None, snapshot_index=None, policy=None):
    policy = {**_DEFAULT_POLICY, **dict(policy or {})}
    decisions = _coerce_index(reference_index)
    snapshots = _coerce_index(snapshot_index)
    require_snapshots = bool(policy.get("require_l2_snapshots", False))

    if decisions.empty:
        return {
            "healthy": True,
            "observed": False,
            "configured": not snapshots.empty,
            "reason": None,
        }

    if snapshots.empty:
        return {
            "healthy": not require_snapshots,
            "observed": False,
            "configured": False,
            "decision_count": int(len(decisions)),
            "missing_snapshot_count": int(len(decisions)),
            "max_snapshot_age": None,
            "median_snapshot_age": None,
            "reason": None if not require_snapshots else "l2_snapshot_missing",
        }

    decision_frame = pd.DataFrame({"decision_time": decisions}).sort_values("decision_time")
    snapshot_frame = pd.DataFrame({"snapshot_time": snapshots.unique()}).sort_values("snapshot_time")
    merged = pd.merge_asof(
        decision_frame,
        snapshot_frame,
        left_on="decision_time",
        right_on="snapshot_time",
        direction="backward",
    )
    age = merged["decision_time"] - merged["snapshot_time"]
    missing_mask = merged["snapshot_time"].isna()
    valid_age = age.loc[~missing_mask]
    max_age = _coerce_timedelta(policy.get("max_l2_snapshot_age"))
    breach_mask = pd.Series(False, index=merged.index, dtype=bool)
    if max_age is not None and not valid_age.empty:
        breach_mask.loc[~missing_mask] = valid_age > max_age
    healthy = (not require_snapshots or not missing_mask.any()) and not breach_mask.any()
    return {
        "healthy": bool(healthy),
        "observed": True,
        "configured": True,
        "decision_count": int(len(decisions)),
        "missing_snapshot_count": int(missing_mask.sum()),
        "breach_count": int(breach_mask.sum()),
        "median_snapshot_age": valid_age.median() if not valid_age.empty else None,
        "max_snapshot_age": valid_age.max() if not valid_age.empty else None,
        "max_allowed_age": max_age,
        "reason": None if healthy else "l2_snapshot_age_breach",
    }


def evaluate_feature_schema_health(expected_feature_columns=None, actual_feature_columns=None, policy=None):
    policy = {**_DEFAULT_POLICY, **dict(policy or {})}
    expected = list(expected_feature_columns or [])
    actual = list(actual_feature_columns or [])
    expected_hash = build_feature_schema_hash(expected)
    actual_hash = build_feature_schema_hash(actual)
    missing = [column for column in expected if column not in actual]
    unexpected = [column for column in actual if column not in expected]
    reordered = len(expected) == len(actual) and set(expected) == set(actual) and expected != actual
    healthy = expected == actual and expected_hash == actual_hash
    return {
        "healthy": bool(healthy),
        "observed": bool(expected or actual),
        "expected_feature_count": int(len(expected)),
        "actual_feature_count": int(len(actual)),
        "expected_schema_hash": expected_hash,
        "actual_schema_hash": actual_hash,
        "missing_columns": missing,
        "unexpected_columns": unexpected,
        "reordered_only": bool(reordered and not missing and not unexpected),
        "fail_closed": bool(policy.get("fail_closed_on_schema_drift", True)),
        "reason": None if healthy else "feature_schema_drift",
    }


def evaluate_execution_health(backtest_reports=None, baseline_report=None, policy=None):
    policy = {**_DEFAULT_POLICY, **dict(policy or {})}
    if backtest_reports is None:
        reports = []
    elif isinstance(backtest_reports, dict):
        reports = [backtest_reports]
    else:
        reports = [dict(report or {}) for report in backtest_reports]

    if not reports:
        return {
            "healthy": True,
            "observed": False,
            "run_count": 0,
            "reason": None,
            "runs": [],
        }

    rows = []
    fill_ratios = []
    slippage_paid = []
    expected_slippage = []
    for report in reports:
        execution_cost_report = dict(report.get("execution_cost_report") or {})
        fill_ratio = float(report.get("fill_ratio") or 0.0)
        realized_slippage = float(report.get("slippage_paid") or 0.0)
        expected_cost = float(execution_cost_report.get("total_cost") or 0.0)
        rows.append(
            {
                "fill_ratio": fill_ratio,
                "realized_slippage": realized_slippage,
                "expected_slippage": expected_cost,
                "slippage_gap": realized_slippage - expected_cost,
                "total_trades": int(report.get("total_trades") or 0),
            }
        )
        fill_ratios.append(fill_ratio)
        slippage_paid.append(realized_slippage)
        expected_slippage.append(expected_cost)

    avg_fill_ratio = float(np.mean(fill_ratios)) if fill_ratios else 0.0
    worst_fill_ratio = float(np.min(fill_ratios)) if fill_ratios else 0.0
    avg_slippage_paid = float(np.mean(slippage_paid)) if slippage_paid else 0.0
    avg_expected_slippage = float(np.mean(expected_slippage)) if expected_slippage else 0.0
    avg_slippage_gap = avg_slippage_paid - avg_expected_slippage

    baseline_fill_ratio = None
    baseline_slippage = None
    fill_ratio_deterioration = 0.0
    slippage_drift = 0.0
    if baseline_report is not None:
        baseline_fill_ratio = float((baseline_report or {}).get("fill_ratio") or 0.0)
        baseline_slippage = float((baseline_report or {}).get("slippage_paid") or 0.0)
        fill_ratio_deterioration = max(0.0, baseline_fill_ratio - avg_fill_ratio)
        slippage_drift = avg_slippage_paid - baseline_slippage

    slippage_gap_share = abs(avg_slippage_gap) / avg_expected_slippage if avg_expected_slippage > 0.0 else (0.0 if abs(avg_slippage_gap) <= 1e-12 else float("inf"))

    reasons = []
    if worst_fill_ratio < float(policy.get("min_fill_ratio", 0.0)):
        reasons.append("fill_ratio_below_threshold")
    if fill_ratio_deterioration > float(policy.get("max_fill_ratio_deterioration", np.inf)):
        reasons.append("fill_ratio_deterioration")
    if slippage_gap_share > float(policy.get("max_slippage_gap_share", np.inf)):
        reasons.append("slippage_gap_share")
    if slippage_drift > float(policy.get("max_slippage_drift", np.inf)):
        reasons.append("slippage_drift")

    return {
        "healthy": not reasons,
        "observed": True,
        "run_count": int(len(rows)),
        "avg_fill_ratio": avg_fill_ratio,
        "worst_fill_ratio": worst_fill_ratio,
        "baseline_fill_ratio": baseline_fill_ratio,
        "fill_ratio_deterioration": fill_ratio_deterioration,
        "avg_realized_slippage": avg_slippage_paid,
        "avg_expected_slippage": avg_expected_slippage,
        "avg_slippage_gap": avg_slippage_gap,
        "slippage_gap_share": slippage_gap_share,
        "baseline_realized_slippage": baseline_slippage,
        "slippage_drift": slippage_drift,
        "runs": rows,
        "reason": reasons[0] if reasons else None,
    }


def evaluate_inference_health(latencies_ms=None, queue_backlog=None, policy=None):
    policy = {**_DEFAULT_POLICY, **dict(policy or {})}
    latencies = _coerce_float_list(latencies_ms)
    if queue_backlog is None:
        backlog_values = []
    elif np.isscalar(queue_backlog):
        backlog_values = [float(queue_backlog)]
    else:
        backlog_values = [float(value) for value in queue_backlog if value is not None]

    if not latencies:
        healthy = not bool(policy.get("require_inference_metrics", False))
        return {
            "healthy": bool(healthy),
            "observed": False,
            "queue_backlog_max": int(max(backlog_values)) if backlog_values else 0,
            "reason": None if healthy else "inference_metrics_missing",
        }

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    max_latency = float(np.max(latencies))
    backlog_max = int(max(backlog_values)) if backlog_values else 0
    reasons = []
    if p95 > float(policy.get("max_inference_p95_ms", np.inf)):
        reasons.append("inference_latency_p95")
    if backlog_max > float(policy.get("max_queue_backlog", np.inf)):
        reasons.append("inference_queue_backlog")
    return {
        "healthy": not reasons,
        "observed": True,
        "call_count": int(len(latencies)),
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "max_latency_ms": max_latency,
        "queue_backlog_max": backlog_max,
        "reason": reasons[0] if reasons else None,
    }


def evaluate_signal_decay_health(signal_decay_report=None, baseline_report=None, policy=None):
    policy = {**_DEFAULT_POLICY, **dict(policy or {})}
    report = dict(signal_decay_report or {})
    if not report:
        return {
            "healthy": True,
            "observed": False,
            "reason": None,
        }

    current_net_edge = report.get("net_edge_at_effective_delay")
    current_half_life = report.get("half_life_bars")
    baseline = dict(baseline_report or {})
    baseline_net_edge = baseline.get("net_edge_at_effective_delay")
    baseline_half_life = baseline.get("half_life_bars")

    edge_deterioration = None
    if (
        current_net_edge is not None
        and baseline_net_edge is not None
        and np.isfinite(float(current_net_edge))
        and np.isfinite(float(baseline_net_edge))
    ):
        edge_deterioration = float(baseline_net_edge) - float(current_net_edge)

    half_life_deterioration = None
    if (
        current_half_life is not None
        and baseline_half_life is not None
        and np.isfinite(float(current_half_life))
        and np.isfinite(float(baseline_half_life))
    ):
        half_life_deterioration = float(baseline_half_life) - float(current_half_life)

    reasons = []
    if report.get("gate_mode") == "blocking" and not bool(report.get("promotion_pass", True)):
        reasons.append((report.get("reasons") or ["signal_decay_gate_failed"])[0])
    if current_net_edge is not None and np.isfinite(float(current_net_edge)):
        if float(current_net_edge) < float(policy.get("min_signal_decay_net_edge_at_delay", -np.inf)):
            reasons.append("signal_decay_net_edge_at_delay")
    if current_half_life is not None and np.isfinite(float(current_half_life)):
        if float(current_half_life) < float(policy.get("min_signal_half_life_bars", -np.inf)):
            reasons.append("signal_half_life_below_minimum")
    if edge_deterioration is not None and edge_deterioration > float(policy.get("max_signal_delay_edge_deterioration", np.inf)):
        reasons.append("signal_delay_edge_deterioration")
    if half_life_deterioration is not None and half_life_deterioration > float(policy.get("max_signal_half_life_deterioration", np.inf)):
        reasons.append("signal_half_life_deterioration")

    return {
        "healthy": not reasons,
        "observed": True,
        "trade_count": int(report.get("trade_count", 0)),
        "gate_mode": report.get("gate_mode"),
        "promotion_pass": bool(report.get("promotion_pass", True)),
        "half_life_bars": report.get("half_life_bars"),
        "net_edge_at_effective_delay": current_net_edge,
        "edge_retention_at_effective_delay": report.get("edge_retention_at_effective_delay"),
        "baseline_net_edge_at_effective_delay": baseline_net_edge,
        "baseline_half_life_bars": baseline_half_life,
        "edge_deterioration": edge_deterioration,
        "half_life_deterioration": half_life_deterioration,
        "reason": reasons[0] if reasons else None,
    }


def build_monitoring_report(
    *,
    data_index=None,
    expected_data_end=None,
    expected_data_interval=None,
    max_data_lag=None,
    custom_data_report=None,
    reference_index=None,
    l2_snapshot_index=None,
    expected_feature_columns=None,
    actual_feature_columns=None,
    backtest_reports=None,
    baseline_backtest_report=None,
    signal_decay_report=None,
    baseline_signal_decay_report=None,
    inference_latencies_ms=None,
    queue_backlog=None,
    policy=None,
):
    resolved_policy = {**_DEFAULT_POLICY, **dict(policy or {})}
    if max_data_lag is not None:
        resolved_policy["max_data_lag"] = max_data_lag

    components = {
        "raw_data_freshness": evaluate_raw_data_freshness(
            data_index,
            expected_end=expected_data_end,
            expected_interval=expected_data_interval,
            max_lag=resolved_policy.get("max_data_lag"),
        ),
        "custom_data_ttl": evaluate_custom_data_ttl(custom_data_report, policy=resolved_policy),
        "l2_snapshot_age": evaluate_l2_snapshot_age(reference_index, l2_snapshot_index, policy=resolved_policy),
        "feature_schema": evaluate_feature_schema_health(
            expected_feature_columns,
            actual_feature_columns,
            policy=resolved_policy,
        ),
        "execution_quality": evaluate_execution_health(
            backtest_reports,
            baseline_report=baseline_backtest_report,
            policy=resolved_policy,
        ),
        "signal_decay": evaluate_signal_decay_health(
            signal_decay_report,
            baseline_report=baseline_signal_decay_report,
            policy=resolved_policy,
        ),
        "inference": evaluate_inference_health(
            inference_latencies_ms,
            queue_backlog=queue_backlog,
            policy=resolved_policy,
        ),
    }

    unhealthy_components = [name for name, component in components.items() if not component.get("healthy", True)]
    summary = {
        "healthy_component_count": int(sum(1 for component in components.values() if component.get("healthy", True))),
        "component_count": int(len(components)),
        "worst_fill_ratio": components["execution_quality"].get("worst_fill_ratio"),
        "avg_slippage_gap": components["execution_quality"].get("avg_slippage_gap"),
        "feature_schema_hash": components["feature_schema"].get("actual_schema_hash"),
        "max_inference_p95_ms": components["inference"].get("p95_latency_ms"),
        "signal_half_life_bars": components["signal_decay"].get("half_life_bars"),
        "net_edge_at_effective_delay": components["signal_decay"].get("net_edge_at_effective_delay"),
    }
    return {
        "healthy": not unhealthy_components,
        "policy": resolved_policy,
        "summary": summary,
        "components": components,
        "reasons": unhealthy_components,
    }


def _iter_scalar_metrics(component_name, payload):
    for key, value in dict(payload or {}).items():
        if isinstance(value, (dict, list, pd.DataFrame, pd.Series, pd.Index)):
            continue
        numeric_value = None
        text_value = None
        if isinstance(value, (bool, np.bool_)):
            numeric_value = float(bool(value))
            text_value = str(bool(value)).lower()
        elif isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
            numeric_value = float(value)
        elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
            text_value = str(value)
        elif value is not None:
            text_value = str(value)
        yield {
            "component": component_name,
            "metric": key,
            "value_numeric": numeric_value,
            "value_text": text_value,
        }


def _render_markdown_summary(report):
    lines = ["# Operational Monitoring Summary", ""]
    lines.append(f"Overall health: {'healthy' if report.get('healthy', False) else 'unhealthy'}")
    if report.get("reasons"):
        lines.append(f"Failed components: {', '.join(report['reasons'])}")
    lines.append("")
    for component_name, component in (report.get("components") or {}).items():
        status = "healthy" if component.get("healthy", True) else "unhealthy"
        lines.append(f"## {component_name}")
        lines.append(f"status: {status}")
        for row in _iter_scalar_metrics(component_name, component):
            value = row["value_numeric"] if row["value_numeric"] is not None else row["value_text"]
            lines.append(f"- {row['metric']}: {value}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def write_monitoring_artifacts(report, root_dir, run_id=None):
    resolved_root = Path(root_dir)
    resolved_root.mkdir(parents=True, exist_ok=True)
    run_id = str(run_id or pd.Timestamp.now(tz="UTC").strftime("%Y%m%d%H%M%S"))

    json_path = resolved_root / f"{run_id}.json"
    parquet_path = resolved_root / f"{run_id}.parquet"
    markdown_path = resolved_root / f"{run_id}.md"

    write_json(json_path, report)
    metric_rows = []
    for component_name, component in (report.get("components") or {}).items():
        metric_rows.extend(_iter_scalar_metrics(component_name, component))
    write_parquet_frame(parquet_path, pd.DataFrame(metric_rows))
    markdown_path.write_text(_render_markdown_summary(report), encoding="utf-8")
    return {
        "json": str(json_path),
        "parquet": str(parquet_path),
        "markdown": str(markdown_path),
    }


__all__ = [
    "build_monitoring_report",
    "evaluate_custom_data_ttl",
    "evaluate_execution_health",
    "evaluate_feature_schema_health",
    "evaluate_inference_health",
    "evaluate_l2_snapshot_age",
    "evaluate_raw_data_freshness",
    "evaluate_signal_decay_health",
    "write_monitoring_artifacts",
]