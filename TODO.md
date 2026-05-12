# TODO

## Purpose

This file replaces the prior phase-only backlog with the current execution plan
for the 2026-05-11 adversarial audit issues.

The scope of this plan is narrow and binding:

1. Convert each audited issue into a concrete implementation workstream.
2. Keep research and promotion paths causally aligned.
3. Use fail-closed semantics in capital-facing modes.
4. Stay realistic for a retail operator on consumer hardware.

## Global Delivery Rules

1. Fix the controlling abstraction, not only the downstream report.
2. Treat `unknown` and `insufficient_evidence` as blocking in capital-facing paths.
3. Keep one authoritative execution path for research evidence and promotion evidence.
4. Prefer deterministic, CPU-cheap runtime logic over additional model complexity.
5. Add or extend focused regression tests alongside each workstream.

## Implementation Order

Implement the issues in this order:

1. Issue 1: Bind router decisions into executed PnL.
2. Issue 3: Make sparse-evidence router gates fail closed.
3. Issue 2: Treat warm and unavailable regime states as safe-mode inputs.
4. Issue 4: Expand lookahead certification to the full causal surface.
5. Issue 5: Treat specialist fallback share as concentration risk.
6. Issue 6: Replace ordinal regime IDs with stable semantic contracts.
7. Issue 7: Upgrade drift monitoring from mean-probability KL to decision-aware drift.

This order is intentional:

1. Router-to-PnL binding is the foundation for any honest routing gate.
2. Sparse-evidence and warm-state handling must be corrected before routing evidence is trusted.
3. Lookahead certification must then cover the entire capital path.
4. Fallback-share and semantic-regime work should build on the corrected routing/runtime surface.
5. Drift upgrades should consume the hardened runtime semantics instead of locking in the current coarse contracts.

## Issue 1: Bind Router Decisions Into Executed PnL

Status: not started

### Objective

Make routing decisions part of the executed backtest path instead of a replay-only diagnostic attached after returns are computed.

### Controlling Files

1. `core/pipeline.py`
2. `core/backtest.py`
3. `core/routing/router.py`
4. `core/routing/diagnostics.py`
5. `core/promotion.py`

### Existing Regression Surfaces

1. `tests/test_router_trace_replay.py`
2. `tests/test_router_hard_switch_weighted.py`
3. `tests/test_promotion_gate_binding.py`

### Implementation Steps

1. Extend `core/pipeline.py::_resolve_backtest_runtime_kwargs(...)` so the runtime path can pass the active router and specialist library, not only `regime_states`.
2. Add a single execution-bound router path inside `core/backtest.py::run_backtest(...)` that:
   - consumes `router`, `specialist_library`, and aligned `regime_states`
   - chooses the active model at each decision timestamp
   - applies the selected model's requested position to the executed position stream
3. Keep `replay_router_trace(...)` as an audit helper, but stop using replay-only trace summaries as the primary evidence for routing behavior when executable routing objects are available.
4. Replace the current purely hypothetical routing-overhead payload with an explicit split:
   - `diagnostic_router_switching_cost_report` for replay-only studies
   - `realized_router_switching_cost_report` when routing is actually executed in the backtest path
5. Update promotion and backtest summaries so routing gates read the execution-bound routing payload first and downgrade to diagnostic-only status when routing was not executed.
6. Ensure the router decision trace and executed equity curve share the same timestamps and state transitions.

### Test Plan

1. Extend `tests/test_router_trace_replay.py` to prove that the bound execution path and replay trace remain consistent on the same regime stream.
2. Add a focused backtest test showing that enabling the router changes both the decision trace and the resulting equity curve.
3. Extend `tests/test_promotion_gate_binding.py` to ensure routing claims are blocking only when execution-bound routing evidence exists.

### Exit Criteria

1. Disabling the router changes the equity curve when the routed path would have selected different specialists.
2. Routing switch costs can affect realized PnL.
3. Promotion cannot treat replay-only routing summaries as equivalent to executed routing evidence.

## Issue 3: Make Sparse-Evidence Router Gates Fail Closed

Status: not started

### Objective

Change router-stability governance so thin evidence resolves to `unknown` or blocked instead of `passed`.

### Controlling Files

1. `core/promotion.py`
2. `core/automl.py`
3. `core/pipeline.py`

### Existing Regression Surfaces

1. `tests/test_promotion_gate_binding.py`
2. `tests/test_router_trace_replay.py`

### Implementation Steps

1. Refactor router-stability gate outputs in `core/promotion.py` from boolean-only semantics to a tri-state contract:
   - `passed`
   - `failed`
   - `unknown`
2. Treat `decision_count < min_router_decision_count` as `unknown`, not pass.
3. Keep `not_applicable` reserved for cases where routing is explicitly disabled or absent by policy.
4. Thread the tri-state result into `core/automl.py` selection summaries, promotion eligibility payloads, and user-facing reports.
5. In capital-facing modes, treat `unknown` router-stability evidence as blocking.
6. In research-only mode, surface `unknown` as advisory without promoting it to `passed_with_warning`.

### Test Plan

1. Extend `tests/test_promotion_gate_binding.py` with a case where sparse routing evidence blocks trade-ready promotion.
2. Add a case where routing disabled by config reports `not_applicable` rather than `unknown`.
3. Add a case where sufficient routing evidence passes without semantic changes to the existing stable path.

### Exit Criteria

1. Thin routing evidence never produces a green gate in capital-facing modes.
2. User-facing summaries distinguish `unknown` from `passed`.
3. Promotion behavior matches the gate semantics exactly.

## Issue 2: Treat Warm And Unavailable Regime States As Safe-Mode Inputs

Status: not started

### Objective

Prevent the router from treating detector warmup, missing observations, or unavailable state as ordinary routing inputs.

### Controlling Files

1. `core/regimes/detectors.py`
2. `core/routing/router.py`
3. `core/pipeline.py`
4. `core/backtest.py`

### Existing Regression Surfaces

1. `tests/test_regime_online_detection.py`
2. `tests/test_regime_detectors.py`
3. `tests/test_router_hard_switch_weighted.py`

### Implementation Steps

1. Standardize regime-state availability semantics in the detector/runtime contract:
   - `known`
   - `warm`
   - `unavailable`
2. Update `core/routing/router.py` so candidate scoring first checks regime-state availability before compatibility and confidence weighting.
3. Add a config-controlled safe-mode routing policy, for example:
   - `fallback_only`
   - `no_trade`
4. Replace deterministic model-id tie resolution in warm or unavailable states with an explicit safe-priority resolution path.
5. Surface warm-state decision counts, unavailable-state decision counts, and fallback/no-trade counts in the router trace summary and backtest payloads.

### Test Plan

1. Extend `tests/test_regime_online_detection.py` to assert explicit warm and unavailable state contracts.
2. Extend `tests/test_router_hard_switch_weighted.py` to verify safe-mode behavior under warm and unavailable states.
3. Add a regression proving warm-state routing cannot trigger an uncertified specialist switch.

### Exit Criteria

1. Warm or unavailable states never route implicitly by model-id ordering.
2. The chosen safe-mode policy is explicit in both config and runtime output.
3. Router traces quantify the share of decisions made under degraded regime-state availability.

## Issue 4: Expand Lookahead Certification To The Full Causal Surface

Status: not started

### Objective

Extend the lookahead guard so it certifies the full capital-relevant pipeline surface rather than only the pre-training feature artifact.

### Controlling Files

1. `core/pipeline.py`
2. `core/lookahead.py`
3. `core/regime.py`
4. `core/regime_training.py`
5. `core/labeling.py`

### Existing Regression Surfaces

1. `tests/test_global_lookahead_guard_default.py`
2. `tests/test_pipeline_lookahead_guard_wiring.py`
3. `tests/test_lookahead_provocation.py`
4. `tests/test_regime_leakage_controls.py`

### Implementation Steps

1. Expand the default lookahead guard scope in `core/pipeline.py::_resolve_lookahead_guard_config(...)` to include, at minimum:
   - regime observation or regime state artifacts
   - labels
   - signal outputs
   - sizing inputs derived from trade outcomes or pooled OOS history
2. Add stage-level coverage reporting so the guard records:
   - requested stages
   - available stages
   - skipped stages
   - blocking reason when a required stage is unavailable
3. Reuse fold-local helpers during lookahead replay rather than global preview artifacts wherever a stage is trained or generated fold-locally.
4. Teach `core/lookahead.py` to compare deterministic sampled prefixes across multiple artifact types without materializing duplicate full-frame copies.
5. In capital-facing modes, treat missing required stage coverage as blocking.

### Test Plan

1. Extend `tests/test_pipeline_lookahead_guard_wiring.py` to assert the widened default stage set.
2. Extend `tests/test_lookahead_provocation.py` with at least one violation in regime, label, or signal surfaces that the old guard would have missed.
3. Extend `tests/test_regime_leakage_controls.py` to ensure fold-local regime artifacts participate in lookahead certification.

### Exit Criteria

1. A passed lookahead report means the audited stage set actually covers the capital path used for selection and backtesting.
2. Missing audit stages block capital-facing promotion.
3. The guard remains cheap enough to run on sampled fold prefixes on consumer hardware.

## Issue 5: Treat Specialist Fallback Share As Concentration Risk

Status: not started

### Objective

Make high fallback share a binding governance metric instead of a passive report field.

### Controlling Files

1. `core/regime_training.py`
2. `core/promotion.py`
3. `core/pipeline.py`
4. `core/automl.py`

### Existing Regression Surfaces

1. `tests/test_regime_aware_training.py`
2. `tests/test_automl_regime_aware_training.py`
3. `tests/test_regime_coverage_gate.py`

### Implementation Steps

1. Make unseen-regime fallback share a first-class summary field in training, validation, and post-selection outputs.
2. In capital-facing modes, default `require_unseen_regime_fallback_bound = true` unless explicitly overridden by a stricter profile.
3. Compute fallback share at multiple levels:
   - per fold
   - aggregate validation or holdout window
   - by unseen-regime label when available
4. Add candidate classification fields that distinguish:
   - `specialist_effective`
   - `specialist_degraded_to_fallback`
   - `generalist_only`
5. Bind promotion readiness to the fallback-share gate rather than only surfacing fallback rows in diagnostics.

### Test Plan

1. Extend `tests/test_regime_aware_training.py` to assert stable fallback-share summaries.
2. Extend `tests/test_automl_regime_aware_training.py` so specialist trials surface fallback-share metrics in their trial summaries.
3. Extend `tests/test_regime_coverage_gate.py` with a case where high fallback share blocks capital-facing promotion.

### Exit Criteria

1. A specialist candidate with excessive fallback usage cannot become promotion-ready.
2. Reports show whether the deployed object is effectively specialist or effectively fallback-driven.
3. Fallback-share gating is on by default in capital-facing modes.

## Issue 6: Replace Ordinal Regime IDs With Stable Semantic Contracts

Status: not started

### Objective

Stop keying live specialist compatibility and routing logic to unstable ordinal latent-state ids.

### Controlling Files

1. `core/regime.py`
2. `core/regimes/detectors.py`
3. `core/specialists/library.py`
4. `core/routing/router.py`
5. `core/pipeline.py`

### Existing Regression Surfaces

1. `tests/test_regime_compatibility_replay.py`
2. `tests/test_regime_hmm_filtered.py`
3. `tests/test_regime_detectors.py`
4. `tests/test_regime_aware_training.py`

### Implementation Steps

1. Define a semantic regime contract that is versioned and persisted in detector manifests.
2. Update native detectors to emit semantic labels directly where possible.
3. For filtered-HMM or other latent detectors, add an explicit semantic mapping layer based on observable characteristics rather than relying on sorted latent indices alone.
4. Update specialist compatibility and router policy logic to match on semantic labels plus detector schema version, not raw integer ids.
5. Persist the mapping in pipeline summaries, registry metadata, and specialist-library snapshots so a retrain does not silently rebind specialist meaning.
6. Keep latent ids available for research diagnostics, but make them secondary to the semantic label in capital-facing paths.

### Test Plan

1. Extend `tests/test_regime_compatibility_replay.py` to assert semantic compatibility survives retraining when market-state meaning is unchanged.
2. Extend `tests/test_regime_hmm_filtered.py` to assert semantic mapping is persisted and exposed even when latent ordering changes.
3. Extend `tests/test_regime_aware_training.py` so specialist compatibility keys off semantic labels rather than raw ids.

### Exit Criteria

1. Specialist compatibility remains stable across retrains when the same semantic market state recurs.
2. Detector manifests expose raw model details and semantic mapping details.
3. Capital-facing routing no longer depends on bare ordinal regime ids.

## Issue 7: Upgrade Drift Monitoring From Mean-Probability KL To Decision-Aware Drift

Status: not started

### Objective

Replace coarse mean-probability divergence with a multi-signal drift report that can distinguish input drift, score drift, action drift, and realized performance drift.

### Controlling Files

1. `core/drift.py`
2. `core/orchestration.py`
3. `core/monitoring.py`
4. `core/readiness.py`

### Existing Regression Surfaces

1. `tests/test_drift_monitoring.py`
2. `tests/test_drift_retraining_workflow.py`

### Implementation Steps

1. Replace `_kl_divergence(...)` as the primary score-drift metric with fixed-bin or bounded divergence metrics over:
   - predicted class distribution
   - confidence or maximum probability
   - class margin or direction edge
   - action rate or abstain rate
2. Keep feature-drift metrics separate from score-drift metrics in the public report.
3. Add action-drift reporting based on executed signal behavior, not only raw prediction probabilities.
4. Add calibration-drift or realized-outcome drift fields when labels or trade outcomes are available.
5. Update orchestration and maintenance reports so drift can recommend distinct actions such as:
   - observe
   - recalibrate
   - reroute
   - retrain
6. Normalize default thresholds using bar frequency and realized trade frequency where practical so 5m, 1h, and 4h systems do not share the same raw sample assumptions.

### Test Plan

1. Extend `tests/test_drift_monitoring.py` with cases that isolate input drift, score drift, action drift, and performance drift.
2. Extend `tests/test_drift_retraining_workflow.py` so maintenance actions differ based on which drift family fired.
3. Add a regression showing that benign probability-mean movement without action or performance change does not force retraining.

### Exit Criteria

1. Drift reports explain which layer moved and why the chosen maintenance action was recommended.
2. Retraining is no longer triggered primarily by mean-probability movement.
3. The drift stack remains lightweight enough for scheduled retail operation on consumer hardware.

## Cross-Cutting Validation Pass

Run these focused checks after each completed workstream and again after the full stack lands:

1. `tests/test_router_trace_replay.py`
2. `tests/test_router_hard_switch_weighted.py`
3. `tests/test_promotion_gate_binding.py`
4. `tests/test_regime_online_detection.py`
5. `tests/test_regime_detectors.py`
6. `tests/test_regime_leakage_controls.py`
7. `tests/test_global_lookahead_guard_default.py`
8. `tests/test_pipeline_lookahead_guard_wiring.py`
9. `tests/test_lookahead_provocation.py`
10. `tests/test_regime_aware_training.py`
11. `tests/test_automl_regime_aware_training.py`
12. `tests/test_regime_coverage_gate.py`
13. `tests/test_regime_compatibility_replay.py`
14. `tests/test_drift_monitoring.py`
15. `tests/test_drift_retraining_workflow.py`

## Definition Of Done

This plan is complete only when:

1. Routing claims are backed by executed routing paths, not replay-only summaries.
2. Warm, unavailable, sparse-evidence, and fallback-heavy states are fail-closed in capital-facing modes.
3. Lookahead certification covers the entire capital-relevant causal surface.
4. Specialist compatibility is stable across retrains because regime semantics are explicit and versioned.
5. Drift reports distinguish input, score, action, and realized-outcome degradation.
6. The resulting controls remain deterministic and feasible on consumer hardware.
