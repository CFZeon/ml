# Targeted Remediation Plan For Highest-Impact Control Failures

## Status

This document reflects the repository state audited on 2026-04-28.

It intentionally covers only the following findings from [ISSUES.md](ISSUES.md):

1. reduced-power retail fallback disables the strongest selection and holdout controls
2. surrogate execution remains the practical default path
3. missing futures funding is zero-filled outside strict mode
4. monitoring defaults are descriptive rather than binding
5. the practical retail path can still be confused with adversarial out-of-sample evidence

Everything else is out of scope for this document, including broader paper-trading, release-ladder, refit-contamination, and registry-governance work unless those items are required to close the five findings above.

## Institutional Control Baseline

This plan is based on isolated external research from official sources and translates those control families into repo-specific engineering requirements.

| Control family | Official source | Repo implication |
|---|---|---|
| Model governance, validation, monitoring, and controls must be risk-based and commensurate with use | OCC updated model risk management guidance, April 2026 | A capital-facing path cannot silently drop holdout, post-selection, or monitoring controls and still present itself as credible evidence |
| Trustworthy AI requires governance, measurement, and active management rather than passive reporting | NIST AI RMF | Monitoring thresholds, fallback assumptions, and evidence requirements must be explicit, measurable, and fail-closed |
| Exchange timestamps, funding events, and order filters are hard constraints, not soft suggestions | Binance market-data, futures funding, and symbol-filter documentation | UTC handling, discrete funding completeness, and symbol/order constraints must be enforced in capital-facing modes |

## In-Scope Failure Statements

### HF-01: Silent Retail Downgrade Of The Evidence Stack

Current code path: [example_trade_ready_automl.py](example_trade_ready_automl.py) explicitly downgrades to `research_only`, disables `locked_holdout_enabled`, disables `selection_policy`, disables `deflated_sharpe`, disables `pbo`, disables `post_selection`, and disables replication when Nautilus is unavailable and the reduced-power path is chosen.

Why this is institutionally unacceptable:

1. Model governance does not permit a capital-facing path to silently remove effective challenge and still look like a valid decision workflow.
2. A weaker evidence stack must be classified as a different product state, not as a degraded version of the same state.
3. The user-facing path matters more than the best possible path hidden elsewhere in the repo.

### HF-02: Surrogate Execution Is Too Close To The Capital Path

Current code path: [core/backtest.py](core/backtest.py) still emits `conservative_bar_surrogate` with `bar_surrogate_only`, `no_queue_position_model`, `no_event_driven_ack_latency`, and `no_order_book_matching_engine` when no event-driven adapter is available.

Why this is institutionally unacceptable:

1. A bar surrogate is a research simulator, not execution evidence.
2. Queue, acknowledgment latency, matching, and partial-fill realism are not optional for claims about live tradability.
3. If accessible capital-facing evidence depends on event-driven execution, the system must abort when that engine is unavailable.

### HF-03: Missing Funding Is Converted Into Zero Carry

Current code path: [core/pipeline.py](core/pipeline.py) sets funding missingness to `zero_fill` outside strict trade-ready mode, and [core/backtest.py](core/backtest.py) plus [core/models.py](core/models.py) reindex funding with `.fillna(0.0)`.

Why this is institutionally unacceptable:

1. Discrete exchange cash flows are part of PnL and cannot be replaced with neutral values without changing the experiment.
2. A missing funding event is a data-certification breach, not an innocuous interpolation choice.
3. Any sizing logic downstream of biased futures returns is contaminated.

### HF-04: Monitoring Exists But Is Not Operationally Binding

Current code path: [core/monitoring.py](core/monitoring.py) research defaults leave lag, slippage drift, inference latency, queue backlog, and half-life deterioration effectively unconstrained unless the caller overrides them.

Why this is institutionally unacceptable:

1. A metric that cannot block a run is telemetry, not control.
2. A capital path must define admissible operating envelopes before the run starts.
3. Unknown or missing operational state must not be treated as acceptable by default.

### OOS-01: The Practical Retail Path Does Not Prove Adversarial OOS Evidence

Current code path: [core/automl.py](core/automl.py) can build a valid holdout plan, but the accessible reduced-power path disables the strongest protection layers that distinguish a convenient local run from adversarial OOS evidence.

Why this is institutionally unacceptable:

1. The existence of CPCV, purging, embargo, locked holdout, and post-selection inference in the codebase is irrelevant if the runnable path suppresses them.
2. Institutional effective challenge requires an explicit, complete evidence stack, not a partial subset chosen by environment convenience.
3. Capital eligibility must depend on the realized evidence stack, not on the repo's theoretical capability.

## Target Control State

The repository should expose exactly three user-facing evidence states for this scope:

| State | Intended use | Required evidence stack | Allowed execution mode | Capital eligible |
|---|---|---|---|---|
| `research_demo` | idea generation and debugging | optional temporal validation | surrogate allowed | no |
| `local_certification` | retail-accessible strict evidence run | CPCV or purged temporal search, locked holdout, post-selection controls, replication, strict funding and monitoring | event-driven only | not live, but promotable to next gate |
| `trade_ready` | strongest local operational certification | all of `local_certification` plus stronger operational assumptions | event-driven only | yes, subject to broader readiness outside this document |

The system must never transition automatically from one state to another by environment accident.

## Remediation Workstreams

### WS-01: Bind Evidence Class To The Runnable Path

#### Objective

Remove the silent downgrade that currently lets a user run a materially weaker path while thinking they exercised the stronger one.

#### Exact Code Changes

1. In [example_trade_ready_automl.py](example_trade_ready_automl.py), delete the branch that rewrites the profile into `research_only` when Nautilus is unavailable.
2. Replace that branch with a hard abort carrying a structured reason such as:

```python
{
    "status": "aborted",
    "requested_mode": "trade_ready",
    "effective_mode": "none",
    "reason": "event_driven_backend_unavailable",
    "capital_path_eligible": False,
}
```

3. Add a separate, explicit entry point such as `example_local_certification_automl.py` that is discoverable and strict by default.
4. In [example_utils.py](example_utils.py), define immutable profile builders for:
   - `research_demo`
   - `local_certification`
   - `trade_ready`
5. In [core/pipeline.py](core/pipeline.py), add an explicit `evaluation_mode` enum check and reject any unrecognized or downgraded capital-facing mode.
6. In [core/automl.py](core/automl.py), add a `capital_evidence_contract` section to the study summary:

```python
{
    "requested_mode": "local_certification",
    "effective_mode": "local_certification",
    "capital_path_eligible": True,
    "required_controls": {
        "locked_holdout": True,
        "selection_policy": True,
        "post_selection": True,
        "replication": True,
    },
    "observed_controls": {...},
    "blocking_reasons": [],
}
```

7. If any required control is disabled in a capital-facing mode, return structured abstention or abort. Do not continue with a partial study summary.

#### Institutional Translation

This closes the governance defect identified by OCC and NIST: the system must be explicit about what control regime is active, and weaker governance cannot masquerade as stronger governance.

#### Tests To Add Or Update

1. `tests/test_trade_ready_no_silent_research_downgrade.py`
2. `tests/test_local_certification_profile_contract.py`
3. `tests/test_capital_evidence_contract_required_controls.py`

#### Acceptance Criteria

1. No capital-facing example script silently rewrites itself into `research_only`.
2. The study summary always reports `requested_mode`, `effective_mode`, and `capital_path_eligible`.
3. A reduced evidence stack is represented as `research_demo` only.

### WS-02: Make Adversarial OOS Evidence A Hard Contract

#### Objective

Ensure that a retail-accessible strict run proves the full intended OOS control stack rather than merely implying it.

#### Exact Code Changes

1. In [core/automl.py](core/automl.py), add an `evidence_stack` contract with explicit booleans and provenance for:
   - `cpcv_or_purged_temporal_search`
   - `search_stage_embargo`
   - `validation_holdout_gap`
   - `locked_holdout`
   - `post_selection_inference`
   - `replication`
2. Compute a single `evidence_stack_complete` boolean from those controls in certification-capable modes.
3. Require `evidence_stack_complete == True` before any summary may set `promotion_ready=True` or `capital_path_eligible=True`.
4. Add a stable summary payload such as:

```python
{
    "oos_evidence": {
        "class": "adversarial_oos" | "partial_oos" | "search_only",
        "evidence_stack_complete": bool,
        "controls": {...},
        "blocking_reasons": [...],
    }
}
```

5. In [core/automl.py](core/automl.py), if `locked_holdout_enabled` is false in `local_certification` or `trade_ready`, abort before optimization starts rather than continuing and later disclaiming the result.
6. Preserve the holdout plan metadata already generated by `_resolve_holdout_plan(...)`, but upgrade it from descriptive metadata to a precondition.
7. In the example scripts and printed summaries, surface `oos_evidence.class` first, ahead of Sharpe, CAGR, or any backtest metric.

#### Institutional Translation

OCC-style model validation requires clear validation and monitoring controls, and NIST requires measurable governance. The engineering translation is that the OOS control stack must be machine-validated, not inferred by the operator.

#### Tests To Add Or Update

1. `tests/test_certification_requires_complete_oos_stack.py`
2. `tests/test_study_summary_oos_evidence_contract.py`
3. Extend [tests/test_automl_holdout_objective.py](tests/test_automl_holdout_objective.py) so missing holdout or post-selection controls force abstention or abort, not a soft summary.

#### Acceptance Criteria

1. A capital-facing summary cannot exist with `oos_evidence.class != "adversarial_oos"`.
2. The system cannot produce promotable evidence from a partial stack.
3. A user can tell from the summary alone whether the run truly exercised the full OOS discipline.

### WS-03: Separate Surrogate Research Execution From Capital-Facing Execution

#### Objective

Prevent surrogate execution from participating in certification-grade evidence.

#### Exact Code Changes

1. In [core/backtest.py](core/backtest.py), treat `conservative_bar_surrogate` as an explicit evidence class, not only as a limitation note.
2. Add an `execution_evidence` block:

```python
{
    "class": "research_surrogate" | "event_driven_certification",
    "execution_mode": "conservative_bar_surrogate" | "event_driven",
    "promotion_execution_ready": bool,
    "blocking_reasons": [...],
}
```

3. In [core/pipeline.py](core/pipeline.py), for `local_certification` and `trade_ready`:
   - require `execution_mode == "event_driven"`
   - require `promotion_execution_ready == True`
   - otherwise abort before performance metrics are interpreted as certification evidence
4. Keep surrogate execution available only for `research_demo`.
5. Do not allow example scripts to print certification language when `execution_evidence.class == "research_surrogate"`.
6. Surface the currently latent adapter limitations directly in the summary and example output.

#### Institutional Translation

Institutional controls distinguish model signal validity from executable strategy validity. A model cannot be considered deployment-relevant if the fill process used to certify it omits matching, queue, and acknowledgment behavior.

#### Tests To Add Or Update

1. `tests/test_surrogate_execution_blocks_certification.py`
2. `tests/test_execution_evidence_contract.py`
3. `tests/test_example_scripts_do_not_certify_surrogate_runs.py`

#### Acceptance Criteria

1. Surrogate execution is still usable for research.
2. Surrogate execution is never eligible for `local_certification` or `trade_ready` evidence.
3. Certification outputs always declare an event-driven execution class.

### WS-04: Turn Funding Completeness Into A Data-Certification Gate

#### Objective

Stop converting missing futures funding into neutral returns.

#### Exact Code Changes

1. In [core/pipeline.py](core/pipeline.py), extend `_resolve_backtest_funding_missing_policy(...)` so `local_certification` inherits the same strictness as `trade_ready`.
2. Replace current default semantics in capital-facing modes with:

```python
{
    "mode": "strict",
    "expected_interval": "8h",
    "max_gap_multiplier": 1.25,
    "allow_missing_events": False,
}
```

3. In [core/backtest.py](core/backtest.py) and [core/models.py](core/models.py), stop using `.fillna(0.0)` for funding series when the active mode is `local_certification` or `trade_ready`.
4. Introduce a `FundingCoverageReport` artifact with:
   - expected funding timestamps
   - observed timestamps
   - missing event count
   - max consecutive gap
   - coverage ratio
   - `promotion_pass`
5. Make the backtest abort with `funding_coverage_breach` if the report fails.
6. Preserve zero-fill only for `research_demo`, and label it as an explicit fallback assumption.
7. Feed `funding_coverage_report` into the monitoring summary so missing exchange cash flows count as operational fallback usage.

#### Institutional Translation

Official exchange documentation defines funding as discrete, timestamped cash flows. Replacing absent events with zeros changes realized economics and violates the measurement discipline expected by institutional model governance.

#### Tests To Add Or Update

1. `tests/test_local_certification_funding_strict.py`
2. `tests/test_funding_zero_fill_forbidden_in_capital_modes.py`
3. `tests/test_funding_coverage_report_contract.py`

#### Acceptance Criteria

1. Capital-facing futures runs fail when funding coverage is incomplete.
2. The summary always states whether funding coverage was strict, fallback, or not applicable.
3. Kelly sizing cannot consume futures results generated from zero-filled missing funding in certification-capable modes.

### WS-05: Make Monitoring And Fallback Assumptions Binding

#### Objective

Convert operational telemetry into hard admissibility gates for capital-facing runs.

#### Exact Code Changes

1. In [core/monitoring.py](core/monitoring.py), add a `local_certification` profile instead of inheriting empty `research` defaults.
2. Set finite default thresholds for capital-facing profiles, including at minimum:
   - `max_data_lag`
   - `max_l2_snapshot_age`
   - `max_fill_ratio_deterioration`
   - `max_slippage_gap_share`
   - `max_slippage_drift`
   - `max_inference_p95_ms`
   - `max_queue_backlog`
   - `min_signal_decay_net_edge_at_delay`
   - `max_fallback_assumption_rate = 0.0`
3. In [core/pipeline.py](core/pipeline.py), bind `resolve_monitoring_policy(...)` to `evaluation_mode` so a caller cannot accidentally inherit `research` tolerances in `local_certification`.
4. Add a `monitoring_gate_report` with:
   - configured thresholds
   - measured values
   - missing metrics
   - `promotion_pass`
   - blocking reasons
5. Require all mandatory metrics to be present in capital-facing modes. Missing metrics are breaches, not warnings.
6. Propagate any fallback assumption, including surrogate execution or funding fallback, into `max_fallback_assumption_rate` accounting.

#### Institutional Translation

NIST's govern-measure-manage structure requires controls to be measurable and actionable. A control without a threshold and failure consequence is not a control.

#### Tests To Add Or Update

1. `tests/test_local_certification_monitoring_defaults.py`
2. `tests/test_monitoring_gate_blocks_missing_metrics.py`
3. `tests/test_fallback_assumption_rate_blocks_capital_modes.py`

#### Acceptance Criteria

1. Monitoring thresholds are finite and profile-specific in all capital-facing modes.
2. Missing operational metrics block certification.
3. The summary explicitly states whether the run stayed inside the admissible operating envelope.

## Delivery Sequence

1. Implement WS-01 first. Without path classification and no-silent-downgrade behavior, every later control is still easy to bypass.
2. Implement WS-02 next. The repository needs a machine-checkable OOS evidence contract before any performance number is meaningful.
3. Implement WS-03 and WS-04 together. Execution realism and funding completeness are the largest sources of false live confidence in the current path.
4. Implement WS-05 last within this scope. Monitoring only matters once the certified path and data assumptions are already fail-closed.

## Definition Of Done For This Scope

The five in-scope findings are remediated only when all of the following are true simultaneously:

1. No capital-facing entry point can silently fall back to `research_only`.
2. `local_certification` and `trade_ready` runs require a complete adversarial OOS evidence stack and declare that fact in the summary contract.
3. Surrogate execution is impossible to treat as certification evidence.
4. Futures funding completeness is strict and machine-validated in capital-facing modes.
5. Monitoring and fallback assumptions are profile-bound, finite, and blocking.

Until then, the correct classification for the accessible retail workflow remains `research_demo`, regardless of the quality of any single backtest metric.
