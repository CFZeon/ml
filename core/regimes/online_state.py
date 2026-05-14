"""Replay regime detector outputs into typed state traces."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..regime import (
    build_regime_observation_contracts,
    build_regime_trace_summary,
    build_regime_transition_contracts,
    normalize_regime_feature_set,
)
from .contracts import RegimeStateContract


def normalize_regime_state_contracts(state_contracts):
    normalized = []
    for contract in list(state_contracts or []):
        if isinstance(contract, RegimeStateContract):
            normalized.append(contract)
            continue
        if isinstance(contract, Mapping):
            normalized.append(RegimeStateContract.from_dict(contract))
            continue
        raise TypeError(f"Unsupported regime state contract payload: {type(contract)!r}")
    return normalized


def slice_regime_state_contracts(state_contracts, index):
    normalized = normalize_regime_state_contracts(state_contracts)
    if index is None:
        return normalized

    target_index = pd.Index(index)
    if target_index.empty or not normalized:
        return []

    by_as_of = {}
    for contract in normalized:
        try:
            key = pd.Timestamp(contract.as_of)
        except Exception:
            key = contract.as_of
        by_as_of[key] = contract

    sliced = []
    for timestamp in target_index:
        try:
            key = pd.Timestamp(timestamp)
        except Exception:
            key = timestamp
        contract = by_as_of.get(key)
        if contract is not None:
            sliced.append(contract)
    return sliced


def build_regime_frame_from_state_contracts(state_contracts, *, index=None, column_name="regime"):
    normalized = normalize_regime_state_contracts(state_contracts)
    rows: dict[Any, dict[str, Any]] = {}
    for contract in normalized:
        row = dict(contract.detector_outputs or {})
        if column_name not in row and contract.label is not None:
            row[column_name] = contract.label
        rows[contract.as_of] = row

    if rows:
        frame = pd.DataFrame.from_dict(rows, orient="index")
    else:
        frame = pd.DataFrame()

    if index is not None:
        frame = frame.reindex(index)
    return frame


def _is_contract_admissible(contract: RegimeStateContract, decision_time: Any) -> tuple[bool, str]:
    unavailable_flag = bool(int((contract.detector_outputs or {}).get("unavailable", 0) or 0))
    if unavailable_flag:
        return False, "unavailable"
    if not bool(contract.warm):
        return False, "warm"
    if contract.available_at is None or decision_time is None:
        return True, "available"

    try:
        available_timestamp = pd.Timestamp(contract.available_at)
        decision_timestamp = pd.Timestamp(decision_time)
        is_admissible = bool(available_timestamp <= decision_timestamp)
    except Exception:
        is_admissible = bool(contract.available_at <= decision_time)
    return (True, "available") if is_admissible else (False, "deferred")


def build_admissible_regime_view(state_contracts, *, index=None, column_name="regime"):
    normalized = normalize_regime_state_contracts(state_contracts)
    target_index = pd.Index(index) if index is not None else pd.Index([contract.as_of for contract in normalized])
    if target_index.empty:
        return pd.DataFrame(index=target_index)

    by_as_of = {}
    for contract in normalized:
        try:
            key = pd.Timestamp(contract.as_of)
        except Exception:
            key = contract.as_of
        by_as_of[key] = contract

    rows: dict[Any, dict[str, Any]] = {}
    for decision_time in target_index:
        try:
            key = pd.Timestamp(decision_time)
        except Exception:
            key = decision_time
        contract = by_as_of.get(key)
        row: dict[str, Any] = {}
        if contract is not None:
            row.update(dict(contract.detector_outputs or {}))
            admissible, state = _is_contract_admissible(contract, decision_time)
            resolved_label = row.get(column_name, contract.label)
            row[column_name] = resolved_label if admissible else np.nan
            if contract.confidence is not None and "regime_confidence" not in row:
                row["regime_confidence"] = float(contract.confidence)
            if contract.confidence_kind is not None and "confidence_kind" not in row:
                row["confidence_kind"] = str(contract.confidence_kind)
            row["available_at"] = contract.available_at
            row["source_available_at"] = contract.source_available_at
            row["recognition_lag_bars"] = contract.recognition_lag_bars
            row.setdefault("warm", bool(contract.warm))
            row["admissible"] = int(admissible)
            row["timing_blocked"] = int(state == "deferred")
            if state != "available":
                row["unavailable"] = int(row.get("unavailable", 1 if state in {"warm", "unavailable"} else 0))
            row["availability_state"] = str(
                contract.metadata.get("availability_state") if state == "available" else state
            )
        rows[decision_time] = row

    frame = pd.DataFrame.from_dict(rows, orient="index") if rows else pd.DataFrame(index=target_index)
    return frame.reindex(target_index)


def replay_regime_detector_trace(
    observations,
    *,
    detector,
    source_map: Mapping[str, str] | None = None,
    provenance: Mapping[str, Any] | None = None,
    fit_observations=None,
    mode: str = "global_preview_only",
    metadata: Mapping[str, Any] | None = None,
):
    normalized = normalize_regime_feature_set({"frame": observations, "source_map": source_map or {}})
    observation_frame = normalized.frame
    replay_metadata = dict(metadata or {})

    fit_frame = observation_frame if fit_observations is None else pd.DataFrame(fit_observations).copy()
    detector.fit(fit_frame)
    runtime_state = detector.initialize(observation_frame)

    observation_contracts = build_regime_observation_contracts(
        observation_frame,
        source_map=normalized.source_map,
        metadata=replay_metadata,
    )
    state_contracts = []
    for observation in observation_contracts:
        runtime_state, state_contract = detector.update(runtime_state, observation)
        state_contracts.append(state_contract)

    state_frame = build_regime_frame_from_state_contracts(
        state_contracts,
        index=observation_frame.index,
        column_name=getattr(detector, "column_name", "regime"),
    )
    detector_manifest = detector.manifest()
    trace_summary = build_regime_trace_summary(
        state_frame,
        mode=mode,
        observation_columns=list(observation_frame.columns),
        provenance=dict(provenance or normalized.provenance),
        metadata={
            **replay_metadata,
            "detector_count": 1,
            "detector_names": [detector_manifest.detector_name],
        },
    )

    return {
        "state_frame": state_frame,
        "observation_contracts": observation_contracts,
        "state_contracts": state_contracts,
        "transition_contracts": build_regime_transition_contracts(state_frame, metadata=replay_metadata),
        "trace_summary": trace_summary,
        "detector_manifests": [detector_manifest],
        "mode": mode,
        "evidence_class": trace_summary.evidence_class,
    }


__all__ = [
    "build_admissible_regime_view",
    "build_regime_frame_from_state_contracts",
    "normalize_regime_state_contracts",
    "replay_regime_detector_trace",
    "slice_regime_state_contracts",
]