"""Replay regime detector outputs into typed state traces."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ..regime import (
    build_regime_observation_contracts,
    build_regime_trace_summary,
    build_regime_transition_contracts,
    normalize_regime_feature_set,
)


def _state_contracts_to_frame(state_contracts, *, index=None, column_name="regime"):
    rows: dict[Any, dict[str, Any]] = {}
    for contract in state_contracts:
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

    state_frame = _state_contracts_to_frame(
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


__all__ = ["replay_regime_detector_trace"]