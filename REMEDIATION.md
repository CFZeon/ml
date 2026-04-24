# REMEDIATION PLAN

## 1) Objective

This plan remediates the highest-risk findings identified in the audit so the repository defaults are safe for serious research and cannot be mistaken for production-grade trading readiness.

Target state:

1. Promotion and post-selection controls are binding by default.
2. Execution realism defaults are conservative and fail closed in trade-ready mode.
3. Lookahead/leakage provocation is mandatory for custom feature builders.
4. Context/funding missingness is represented as unknown state, not zero alpha.
5. Objective gates and minimum evidence thresholds are materially stricter.

Non-goals for this cycle:

1. Building a full exchange-native matching engine in this repository.
2. Supporting legacy permissive behavior without explicit opt-in.

---

## 2) Issue-to-Workstream Mapping

| Issue | Severity | Workstream |
|---|---|---|
| Promotion-safety controls non-binding by default | High | WS-01 Governance Defaults |
| Optimistic execution defaults and surrogate ambiguity | High | WS-02 Execution Hardening |
| Lookahead provocation not automatic for custom builders | Medium-High | WS-03 Leak Guardrails |
| Context outage handling can collapse into zero state | Medium | WS-04 Context/Funding Semantics |
| Objective gates too lenient for capital decisions | Medium | WS-05 Objective Gate Tightening |

---

## 3) Implementation Principles

1. Fail closed in trade-ready mode.
2. Separate research and trade-ready behavior explicitly in configuration and output metadata.
3. Make unsafe legacy behavior explicit and opt-in, never implicit.
4. Every control must have tests that prove both pass and fail paths.
5. All promotion claims must be tied to concrete gate outcomes, not summary strings.

---

## 4) Workstreams

## WS-01 Governance Defaults (AutoML and Promotion)

### Root cause

Current defaults still allow permissive selection and post-selection behavior unless hardened fields are explicitly configured.

### Code changes

1. `core/automl.py` in `_resolve_selection_policy`:
   1. Change defaults:
      1. `require_locked_holdout_pass` -> `True`
      2. `require_fold_stability_pass` -> `True`
      3. Set explicit blocking `gate_modes` defaults for:
         1. `locked_holdout`
         2. `locked_holdout_gap`
         3. `replication`
         4. `execution_realism`
         5. `stress_realism`
         6. `param_fragility`
2. `core/automl.py` in `_resolve_overfitting_control`:
   1. Set `post_selection.require_pass` default to `True`.
3. `core/automl.py` in final trial promotion update path:
   1. Add explicit hard-stop path if any blocking gate fails before summary return.
   2. Keep advisory gate behavior visible but non-blocking.
4. `core/promotion.py`:
   1. Keep schema stable, but enforce unknown mode normalization to `blocking` with a warning token in `gate_status`.

### Config/schema changes

1. Add top-level profile key:
   1. `automl.policy_profile`: `legacy_permissive` or `hardened_default`
2. Default to `hardened_default`.
3. If profile is `legacy_permissive`, emit deprecation warning in summary payload.

### Tests

1. Add/update `tests/test_promotion_gate_binding.py`:
   1. Verify locked holdout fail blocks promotion under default profile.
   2. Verify post-selection fail blocks promotion under default profile.
   3. Verify advisory mode still records failure but does not block.
2. Add `tests/test_selection_policy_defaults.py`:
   1. Assert default policy fields are hardened.

### Acceptance criteria

1. A trial cannot be `promotion_ready=True` if locked holdout or post-selection fails under default config.
2. Summary must list blocking failures in deterministic order.

---

## WS-02 Execution Hardening and Trade-Ready Separation

### Root cause

Execution defaults and metadata can be interpreted as deployable despite bar-surrogate constraints.

### Code changes

1. `core/execution/policies.py`:
   1. Tighten defaults:
      1. `participation_cap`: `1.0` -> `0.10`
      2. `min_fill_ratio`: `0.0` -> `0.25`
      3. `max_order_age_bars`: keep conservative but ensure non-trivial partial fill handling.
2. `core/pipeline.py` in `_resolve_backtest_execution_policy`:
   1. If `backtest.evaluation_mode == trade_ready` and adapter is not `nautilus`, raise `RuntimeError` unless explicit `research_only_override=true`.
3. `core/backtest.py`:
   1. In evaluation metadata attachment, include explicit field:
      1. `trade_ready_blockers` list.
   2. If `evaluation_mode=trade_ready` and `promotion_execution_ready=False`, append blocking reason and set `research_only=True`.
4. `core/automl.py`:
   1. Ensure execution realism gate reads from locked holdout backtest first, then validation, then trial backtest (existing behavior retained) and blocks by default under WS-01.

### Config/schema changes

1. Add `backtest.execution_profile`: `research_surrogate` or `trade_ready_event_driven`.
2. Validate profile compatibility against `evaluation_mode`.

### Tests

1. Update/add `tests/test_execution_partial_fills.py`:
   1. Confirm stricter defaults reflected in policy.
2. Add `tests/test_trade_ready_execution_fail_closed.py`:
   1. `trade_ready + non-nautilus` must fail closed.
3. Add `tests/test_execution_adapter_parity.py` assertions:
   1. Verify trade-ready blockers populate metadata.

### Acceptance criteria

1. Trade-ready mode cannot complete using surrogate-only path without explicit research override.
2. Promotion output must include deterministic execution realism blocker when backend is non-event-driven.

---

## WS-03 Mandatory Lookahead Provocation for Custom Builders

### Root cause

Custom feature builders can inject future information unless users manually run lookahead analysis.

### Code changes

1. `core/pipeline.py` in `build_features` flow:
   1. Detect presence of `features.builders`.
   2. After features are built and before training steps, auto-run `run_lookahead_analysis` for artifact `features`.
2. Add new guard config section:
   1. `features.lookahead_guard`:
      1. `enabled` default `true`
      2. `mode`: `blocking` or `advisory` (default `blocking` when builders are present)
      3. `decision_sample_size` default 32
      4. `min_prefix_rows` default 128
3. Store report in `pipeline.state["lookahead_guard_report"]` and propagate into training summary/promotion gates.
4. Add promotion gate in `core/automl.py` post-selection group:
   1. `name=lookahead_guard`
   2. blocking under default gate modes.

### Tests

1. Extend `tests/test_lookahead_provocation.py`:
   1. Ensure custom future-shifted builder fails automatically in blocking mode.
   2. Ensure advisory mode records failure but allows pipeline completion.
2. Add `tests/test_pipeline_lookahead_guard_wiring.py`:
   1. Verify report appears in training summary and promotion eligibility.

### Acceptance criteria

1. Any builder-induced leak detected by guard prevents promotion under default profile.
2. Guard report is present in artifacts and summary without manual user calls.

---

## WS-04 Context and Funding Missingness Semantics

### Root cause

Default behavior can collapse unknown/stale context into tradable zero state.

### Code changes

1. `core/context.py` in `_resolve_context_missing_policy`:
   1. Default from `zero_fill` to `preserve_missing`.
   2. Default `add_indicator=true`.
   3. Default `max_unknown_rate=0.0`.
2. `core/context.py` in context block builders:
   1. Remove unconditional terminal `fillna(0.0)` for context feature frames.
   2. Preserve NaN and indicators; only zero-fill if explicit `mode=zero_fill`.
3. `core/pipeline.py` training pre-check:
   1. Enforce context TTL/report gate for trade-ready mode regardless of missing policy string.
4. `core/pipeline.py` funding alignment helper:
   1. Require strict funding coverage in trade-ready mode.
   2. If coverage breach, block backtest/promotion.

### Config/schema changes

1. Add explicit `features.context_missing_policy.mode` default migration:
   1. Existing configs without mode are upgraded to `preserve_missing` unless `compat.legacy_missing_semantics=true`.
2. Add `backtest.funding_missing_policy.mode` default to `strict` in trade-ready profile.

### Tests

1. Add `tests/test_context_missing_policy_defaults.py`:
   1. Verify preserve-missing defaults.
2. Add `tests/test_context_ttl_gate_binding.py`:
   1. Unknown-rate breach blocks trade-ready training.
3. Add `tests/test_funding_coverage_gate.py`:
   1. Funding gaps fail closed in trade-ready mode.

### Acceptance criteria

1. Missing/stale context cannot silently become zero-valued signal state under defaults.
2. Funding gaps cannot proceed as zero carry in trade-ready runs.

---

## WS-05 Objective Gate Tightening and Evidence Thresholds

### Root cause

Default objective gates are below institutional confidence thresholds.

### Code changes

1. `core/automl.py` in `_resolve_objective_gates`:
   1. Tighten defaults:
      1. `min_directional_accuracy`: `0.45` -> `0.52`
      2. `max_log_loss`: `1.0` -> `0.78`
      3. `max_calibration_error`: `0.35` -> `0.15`
      4. `min_trade_count`: `5` -> `30`
2. Add optional lower-bound gate support:
   1. `objective_gates.min_sharpe_ci_lower`.
   2. `objective_gates.min_net_profit_pct_ci_lower`.
3. Wire confidence-interval checks from backtest significance payload into objective gate report.

### Tests

1. Add `tests/test_objective_gate_thresholds.py`:
   1. Assert tightened defaults.
2. Add `tests/test_objective_gate_confidence_bounds.py`:
   1. Fails when lower CI bound is below threshold.

### Acceptance criteria

1. Candidates with weak evidence cannot pass objective gates by default.
2. Objective gate report includes all failed checks and measured values.

---

## 5) Rollout Plan

## Phase 0 (Preparation)

1. Add migration warnings and compatibility flags.
2. Freeze baseline metrics on existing examples/tests for comparison.

## Phase 1 (Governance and gating)

1. Ship WS-01 and WS-05 first.
2. Update example defaults so user entry points are hardened.

## Phase 2 (Execution and causality)

1. Ship WS-02 and WS-04.
2. Run targeted regression matrix across spot/futures research modes.

## Phase 3 (Leak guard automation)

1. Ship WS-03.
2. Enforce guard in CI for builder-enabled pipelines.

## Phase 4 (Cleanup and deprecation)

1. Mark permissive flags deprecated.
2. Set removal date for legacy compatibility flags.

---

## 6) CI/Test Matrix to Add to Pipeline

1. `python -m pytest tests/test_promotion_gate_binding.py -q`
2. `python -m pytest tests/test_execution_partial_fills.py -q`
3. `python -m pytest tests/test_lookahead_provocation.py -q`
4. `python -m pytest tests/test_context_ttl_gate_binding.py -q`
5. `python -m pytest tests/test_funding_coverage_gate.py -q`
6. `python -m pytest tests/test_objective_gate_thresholds.py -q`
7. `python -m pytest tests/test_objective_gate_confidence_bounds.py -q`

Add one smoke profile for each mode:

1. research-only surrogate profile must pass and report `research_only=true`.
2. trade-ready profile without event-driven backend must fail closed.
3. trade-ready profile with backend and complete context/funding must produce promotable candidate only when all gates pass.

---

## 7) Documentation Changes Required

1. `README.md`:
   1. Add explicit section: "Research-safe vs Trade-ready-safe behavior".
2. `HOW_TO_USE.md`:
   1. Add migration notes for hardened defaults.
3. `example_automl.py`:
   1. Keep as demo; print explicit non-promotion-safe disclaimer.
4. `example_trade_ready_automl.py`:
   1. Keep as hardened path; document fail-closed expectations.

---

## 8) Definition of Done

This remediation is complete only when all are true:

1. Default AutoML and promotion path is hardened and blocking for critical gates.
2. Trade-ready mode fails closed without event-driven execution backend.
3. Custom builder lookahead guard runs automatically and blocks leaks.
4. Context/funding unknown states are preserved and threshold-gated.
5. Objective gates are tightened and confidence-bound aware.
6. New tests exist for each control and all pass in CI.
7. Docs explain mode separation and migration flags clearly.

---

## 9) Recommended Execution Order (Engineering)

1. WS-01 Governance Defaults
2. WS-05 Objective Gate Tightening
3. WS-04 Context/Funding Semantics
4. WS-02 Execution Hardening
5. WS-03 Mandatory Leak Guard

Rationale:

1. Governance defaults stop the highest risk immediately.
2. Objective evidence tightening reduces false positives early.
3. Data semantics fix removes hidden alpha leakage from missingness.
4. Execution hardening prevents research-mode confusion in deployment claims.
5. Leak guard automation closes remaining custom-feature loopholes.

---

## 10) Ownership and Tracking Template

For each workstream, track:

1. Owner
2. PR list
3. Test additions
4. Backward-compatibility flags introduced
5. Migration notes written
6. Residual risk after merge

Suggested status values:

1. `planned`
2. `in_progress`
3. `in_review`
4. `merged_pending_rollout`
5. `rolled_out`
6. verified