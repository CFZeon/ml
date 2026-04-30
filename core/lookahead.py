"""Lookahead provocation harness based on baseline and prefix-only replays."""

import copy

import numpy as np
import pandas as pd

from .execution import resolve_liquidity_inputs
from .pipeline import ResearchPipeline, _resolve_backtest_execution_prices, _resolve_backtest_runtime_kwargs


DEFAULT_LOOKAHEAD_STEPS = [
    "build_features",
    "detect_regimes",
    "build_labels",
    "align_data",
    "select_features",
    "compute_sample_weights",
    "train_models",
    "generate_signals",
]

DEFAULT_AUDIT_ARTIFACTS = [
    "features",
    "regimes",
    "labels",
    "aligned_labels",
    "oos_probabilities",
    "oos_meta_prob",
    "signals",
    "continuous_signals",
    "execution_prices",
    "execution_volume",
]


def _clone_step_definitions(base_pipeline):
    steps = []
    for step in getattr(base_pipeline, "steps", []):
        steps.append(type(step) if not isinstance(step, type) else step)
    return steps or None


def _slice_temporal_value(value, end_timestamp=None):
    if isinstance(value, pd.DataFrame):
        sliced = value
        if end_timestamp is not None and isinstance(sliced.index, pd.DatetimeIndex):
            sliced = sliced.loc[sliced.index <= end_timestamp]
        return sliced.copy()

    if isinstance(value, pd.Series):
        sliced = value
        if end_timestamp is not None and isinstance(sliced.index, pd.DatetimeIndex):
            sliced = sliced.loc[sliced.index <= end_timestamp]
        return sliced.copy()

    if isinstance(value, dict):
        return {key: _slice_temporal_value(item, end_timestamp=end_timestamp) for key, item in value.items()}
    if isinstance(value, list):
        return [_slice_temporal_value(item, end_timestamp=end_timestamp) for item in value]
    if isinstance(value, tuple):
        return tuple(_slice_temporal_value(item, end_timestamp=end_timestamp) for item in value)
    return copy.deepcopy(value)


def _seed_replay_state(base_state, end_timestamp=None):
    seeded = {}
    for key in [
        "raw_data",
        "data",
        "futures_context",
        "cross_asset_context",
        "cross_asset_context_symbols",
        "symbol_filters",
        "futures_contract_spec",
        "futures_leverage_brackets",
        "orderbook_depth",
        "benchmark_returns",
        "custom_data_report",
        "data_lineage",
        "data_integrity_report",
        "data_quality_mask",
        "data_quality_report",
        "reference_integrity_report",
        "reference_overlay_data",
        "reference_venue_frames",
        "symbol_lifecycle",
        "universe_snapshot",
        "universe_snapshot_meta",
        "eligible_symbols",
        "universe_policy",
        "universe_report",
    ]:
        if key in base_state:
            seeded[key] = _slice_temporal_value(base_state[key], end_timestamp=end_timestamp)
    return seeded


def _resolve_step_sequence(base_pipeline, requested_steps):
    requested_steps = list(requested_steps or DEFAULT_LOOKAHEAD_STEPS)
    if ("raw_data" not in base_pipeline.state or "data" not in base_pipeline.state) and "fetch_data" not in requested_steps:
        requested_steps = ["fetch_data", *requested_steps]
    if (
        base_pipeline.section("indicators")
        and "indicator_run" not in base_pipeline.state
        and "run_indicators" not in requested_steps
    ):
        insert_at = 1 if requested_steps and requested_steps[0] == "fetch_data" else 0
        requested_steps = requested_steps[:insert_at] + ["run_indicators"] + requested_steps[insert_at:]
    return requested_steps


def _prepare_pipeline(base_pipeline, pipeline_class=None, end_timestamp=None):
    pipeline_class = pipeline_class or type(base_pipeline) or ResearchPipeline
    replay_pipeline = pipeline_class(copy.deepcopy(base_pipeline.config), steps=_clone_step_definitions(base_pipeline))
    replay_pipeline.state.update(_seed_replay_state(base_pipeline.state, end_timestamp=end_timestamp))
    return replay_pipeline


def _run_steps(pipeline, step_names):
    for step_name in step_names:
        pipeline.run_step(step_name)
    return pipeline


def _as_audit_frame(value, column_name):
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, pd.Series):
        return value.to_frame(name=value.name or column_name)
    return None


def _concat_path_frames(paths, key, column_name):
    frames = []
    for path in paths or []:
        frame = _as_audit_frame(path.get(key), column_name)
        if frame is not None and not frame.empty:
            frames.append(frame)
    if not frames:
        return None
    return pd.concat(frames).sort_index()


def _first_frame(*values):
    for value in values:
        if isinstance(value, pd.DataFrame) and not value.empty:
            return value.copy()
        if isinstance(value, pd.Series) and not value.empty:
            return value.to_frame(name=value.name or value.name or "value")
    return None


def _extract_audit_artifacts(pipeline, artifact_names=None):
    artifact_names = list(artifact_names or DEFAULT_AUDIT_ARTIFACTS)
    artifacts = {}
    training = pipeline.state.get("training") or {}
    signals = pipeline.state.get("signals") or {}
    signal_paths = signals.get("paths") if isinstance(signals, dict) else None

    direct_artifacts = {
        "features": ("build_features", _as_audit_frame(pipeline.state.get("features"), "features")),
        "regimes": ("detect_regimes", _as_audit_frame(pipeline.state.get("regimes"), "regime")),
        "labels": ("build_labels", _as_audit_frame(pipeline.state.get("labels"), "label")),
        "aligned_labels": ("align_data", _as_audit_frame(pipeline.state.get("labels_aligned"), "label")),
        "oos_probabilities": (
            "train_models",
            _first_frame(
                _as_audit_frame(training.get("oos_probabilities"), "oos_probability"),
                _concat_path_frames(signal_paths, "primary_probabilities", "oos_probability"),
            ),
        ),
        "oos_meta_prob": (
            "train_models",
            _first_frame(
                _as_audit_frame(training.get("oos_meta_prob"), "oos_meta_prob"),
                _concat_path_frames(signal_paths, "meta_prob", "oos_meta_prob"),
            ),
        ),
        "signals": (
            "generate_signals",
            _first_frame(
                _as_audit_frame(signals.get("signals") if isinstance(signals, dict) else None, "signals"),
                _concat_path_frames(signal_paths, "signals", "signals"),
            ),
        ),
        "continuous_signals": (
            "generate_signals",
            _first_frame(
                _as_audit_frame(
                    signals.get("continuous_signals") if isinstance(signals, dict) else None,
                    "continuous_signals",
                ),
                _concat_path_frames(signal_paths, "continuous_signals", "continuous_signals"),
            ),
        ),
    }

    for artifact_name, (stage, frame) in direct_artifacts.items():
        if artifact_name in artifact_names and frame is not None and not frame.empty:
            artifacts[artifact_name] = {"stage": stage, "frame": frame}

    runtime_index = None
    if "continuous_signals" in artifacts:
        runtime_index = artifacts["continuous_signals"]["frame"].index
    elif "signals" in artifacts:
        runtime_index = artifacts["signals"]["frame"].index

    if runtime_index is not None and len(runtime_index) > 0:
        execution_prices = _resolve_backtest_execution_prices(pipeline, runtime_index)
        if "execution_prices" in artifact_names:
            execution_frame = _as_audit_frame(execution_prices, "execution_price")
            if execution_frame is not None and not execution_frame.empty:
                artifacts["execution_prices"] = {"stage": "generate_signals", "frame": execution_frame}

        if "execution_volume" in artifact_names:
            runtime_kwargs = _resolve_backtest_runtime_kwargs(pipeline, runtime_index) or {}
            liquidity_inputs = resolve_liquidity_inputs(
                index=runtime_index,
                volume=runtime_kwargs.get("volume"),
                orderbook_depth=runtime_kwargs.get("orderbook_depth"),
                slippage_model=runtime_kwargs.get("slippage_model"),
                liquidity_lag_bars=runtime_kwargs.get("liquidity_lag_bars", 1),
            )
            volume_frame = _as_audit_frame(liquidity_inputs.get("volume"), "execution_volume")
            if volume_frame is not None and not volume_frame.empty:
                artifacts["execution_volume"] = {"stage": "generate_signals", "frame": volume_frame}

    return artifacts


def _select_decision_timestamps(baseline_pipeline, sample_count=None, decision_timestamps=None):
    if decision_timestamps is not None:
        timestamps = pd.DatetimeIndex(pd.Index(decision_timestamps)).unique().sort_values()
        return list(timestamps)

    candidates = [
        (baseline_pipeline.state.get("signals") or {}).get("signals"),
        (baseline_pipeline.state.get("signals") or {}).get("continuous_signals"),
        (baseline_pipeline.state.get("training") or {}).get("oos_predictions"),
        baseline_pipeline.state.get("labels_aligned"),
        baseline_pipeline.state.get("features"),
    ]
    for candidate in candidates:
        if isinstance(candidate, (pd.Series, pd.DataFrame)) and not candidate.empty:
            index = pd.DatetimeIndex(candidate.index).unique().sort_values()
            if sample_count is None or sample_count >= len(index):
                return list(index)
            sample_positions = np.linspace(0, len(index) - 1, num=max(1, int(sample_count)), dtype=int)
            return list(index[np.unique(sample_positions)])
    return []


def _row_value(frame, timestamp, column):
    if timestamp not in frame.index or column not in frame.columns:
        return None
    value = frame.at[timestamp, column]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _values_match(left, right, atol=1e-12, rtol=1e-9):
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    if pd.isna(left) and pd.isna(right):
        return True
    if isinstance(left, (int, float, np.number)) and isinstance(right, (int, float, np.number)):
        return bool(np.isclose(float(left), float(right), atol=atol, rtol=rtol, equal_nan=True))
    return left == right


def run_lookahead_analysis(
    base_pipeline,
    pipeline_class=None,
    step_names=None,
    artifact_names=None,
    decision_timestamps=None,
    sample_count=None,
    min_prefix_rows=50,
    atol=1e-12,
    rtol=1e-9,
):
    """Run a baseline-plus-prefix replay audit and report mismatches by stage and column."""

    baseline_step_names = _resolve_step_sequence(base_pipeline, step_names)
    baseline_pipeline = _prepare_pipeline(base_pipeline, pipeline_class=pipeline_class)
    _run_steps(baseline_pipeline, baseline_step_names)
    baseline_artifacts = _extract_audit_artifacts(baseline_pipeline, artifact_names=artifact_names)

    audit_timestamps = _select_decision_timestamps(
        baseline_pipeline,
        sample_count=sample_count,
        decision_timestamps=decision_timestamps,
    )
    if not audit_timestamps:
        return {
            "has_bias": False,
            "reason": "no_audit_timestamps",
            "artifacts": {},
            "biased_columns": [],
            "checked_timestamps": 0,
            "skipped_timestamps": [],
        }

    raw_data = baseline_pipeline.require("raw_data")
    prefix_steps = [step for step in baseline_step_names if step != "fetch_data"]
    artifact_reports = {
        name: {
            "stage": payload["stage"],
            "checked_columns": list(payload["frame"].columns),
            "column_reports": {},
            "biased_columns": [],
            "mismatch_count": 0,
            "first_offending_timestamp": None,
        }
        for name, payload in baseline_artifacts.items()
    }
    skipped_timestamps = []

    for timestamp in audit_timestamps:
        raw_position = int(raw_data.index.searchsorted(timestamp, side="right"))
        if raw_position < int(min_prefix_rows):
            skipped_timestamps.append({"timestamp": timestamp.isoformat(), "reason": "insufficient_prefix_rows"})
            continue

        prefix_pipeline = _prepare_pipeline(baseline_pipeline, pipeline_class=pipeline_class, end_timestamp=timestamp)
        try:
            _run_steps(prefix_pipeline, prefix_steps)
        except Exception as exc:
            skipped_timestamps.append({"timestamp": timestamp.isoformat(), "reason": str(exc)})
            continue

        prefix_artifacts = _extract_audit_artifacts(prefix_pipeline, artifact_names=artifact_names)
        for artifact_name, baseline_payload in baseline_artifacts.items():
            baseline_frame = baseline_payload["frame"]
            prefix_frame = (prefix_artifacts.get(artifact_name) or {}).get("frame")
            prefix_frame = prefix_frame if prefix_frame is not None else pd.DataFrame(index=pd.DatetimeIndex([]))
            report = artifact_reports[artifact_name]

            for column in baseline_frame.columns:
                baseline_value = _row_value(baseline_frame, timestamp, column)
                prefix_value = _row_value(prefix_frame, timestamp, column)
                if _values_match(baseline_value, prefix_value, atol=atol, rtol=rtol):
                    continue

                column_report = report["column_reports"].setdefault(
                    column,
                    {
                        "stage": report["stage"],
                        "first_offending_timestamp": timestamp.isoformat(),
                        "mismatch_count": 0,
                        "baseline_value": baseline_value,
                        "prefix_value": prefix_value,
                    },
                )
                column_report["mismatch_count"] += 1
                report["mismatch_count"] += 1
                if report["first_offending_timestamp"] is None:
                    report["first_offending_timestamp"] = timestamp.isoformat()

    biased_columns = []
    for artifact_name, report in artifact_reports.items():
        report["biased_columns"] = sorted(report["column_reports"])
        for column, column_report in report["column_reports"].items():
            biased_columns.append(
                {
                    "artifact": artifact_name,
                    "column": column,
                    "stage": column_report["stage"],
                    "first_offending_timestamp": column_report["first_offending_timestamp"],
                    "mismatch_count": int(column_report["mismatch_count"]),
                }
            )

    return {
        "has_bias": bool(biased_columns),
        "reason": None,
        "artifacts": artifact_reports,
        "biased_columns": sorted(
            biased_columns,
            key=lambda item: (item["first_offending_timestamp"], item["artifact"], item["column"]),
        ),
        "checked_timestamps": int(len(audit_timestamps) - len(skipped_timestamps)),
        "requested_timestamps": int(len(audit_timestamps)),
        "skipped_timestamps": skipped_timestamps,
    }