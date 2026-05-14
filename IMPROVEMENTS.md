# Improvements Plan

Date: 2026-05-14

This document converts the 2026-05-14 adversarial audit into implementation
workstreams. It supersedes the 2026-05-13 draft because that version covered
adjacent weaknesses but not the seven highest-severity failures now confirmed in
the current code paths.

## Coverage Map

The implementation program below covers all seven problems found in the audit:

1. Regime-recognition lag is erased in router replay and backtest binding.
2. Specialist and meta-model training use a different regime surface than live inference.
3. Research mode still permits preview-only regime narratives and a too-narrow lookahead guard.
4. Router switching cost is reported but does not bind executed PnL.
5. Router health scoring is largely placeholder logic rather than executable evidence.
6. Drift adaptation is a static-baseline vote counter that confuses noise, policy shift, and real decay.
7. Regime taxonomy and routing compatibility remain window-relative and unstable across refits.

## Institutional Reference Set

- PRA SS1/23, Model risk management principles for banks.
- BCBS 239, Principles for effective risk data aggregation and risk reporting.
- SEC Rule 15c3-5 market access control release and fact sheet.
- NIST AI RMF 1.0.
- NIST AI RMF Playbook.

## Program Rules

- Executable paths must be keyed by decision-time admissibility, not by row order or retrospective alignment.
- Training, replay, backtest, and promotion evidence must share one causal regime surface.
- Adaptive controls must bind executed exposure and executed PnL, not just diagnostics.
- Research outputs must be downgraded when the full causal surface has not been audited.
- Maintenance refresh, recalibration, challenger training, and champion replacement are separate actions.

## 1. Preserve Regime Recognition Lag Through Router Replay And Backtest Binding

### Audit Problem

The router and backtest stack can erase regime-recognition lag. The current flow
allows regime states to align positionally with the target index and then binds
router decisions back onto the same row, even when `available_at` implies the
state was not known until a later bar.

### Institutional Practice Synthesis

Research basis: PRA SS1/23 Principles 3, 4, and 5; SEC Rule 15c3-5; NIST AI RMF.

- PRA requires model development, implementation, and use to match actual operating conditions rather than optimistic reconstruction.
- PRA validation expectations imply that timing assumptions must be causally correct and replayable.
- SEC 15c3-5 requires automated controls to act before exposure is taken, not after the fact.
- NIST AI RMF treats traceability and validity as operational properties that must survive deployment and replay.

### Technical Implementation Plan

- Replace positional regime alignment in the router/backtest path with a decision-time admissibility join.
- Introduce a single helper such as `build_admissible_router_regime_trace(regime_states, decision_index)` that:
  - resolves the latest regime state whose `available_at <= decision_time`
  - marks rows as `timing_blocked`, `warm`, or `unavailable` when no admissible state exists
  - preserves staleness metrics such as bars-since-recognition
- Refactor router replay so `router.select(...)` is called with `timestamp=decision_time`, not with the regime state's own `available_at` substituted as the decision clock.
- Refactor signal binding so routed signals are keyed by decision timestamp, not by list position in the replay trace.
- Remove or hard-gate the exact-length positional mode in backtests. Keep it only as an explicit debug-only path that cannot feed promotion or capital-facing evidence.
- Thread the same admissible regime trace through:
  - signal generation
  - router diagnostics
  - backtest replay
  - promotion evidence
- Add explicit transition-latency diagnostics:
  - timing-blocked row share
  - stale-state share
  - mean and max recognition lag
  - safe-mode rows caused by timing only

### Validation And Exit Criteria

- A regime state with `available_at > decision_time` never influences the executed row.
- Signal generation, router replay, and backtest replay produce identical routing decisions when given the same admissible trace.
- Tail-path losses increase measurably in synthetic delayed-recognition tests, proving the old optimistic path is gone.
- Promotion reports include timing-specific fallback and block rates.

## 2. Unify Specialist And Meta Training With The Same Admissible Regime Surface Used At Inference

### Audit Problem

Specialist training and inner meta-model training currently use preview-aligned
regime frames, while test-time inference uses admissible delayed regime
contracts. That means the specialist stack is trained on a cleaner partition than
the one that actually exists at execution time.

### Institutional Practice Synthesis

Research basis: PRA SS1/23 Principles 3 and 4; NIST AI RMF; NIST AI RMF Playbook.

- PRA expects model use to remain consistent with development assumptions and operating constraints.
- PRA validation must test implementation correctness, not just conceptual intent.
- NIST AI RMF and the Playbook both emphasize that measurement and management should be anchored to the same real operating context.

### Technical Implementation Plan

- Introduce two explicitly named regime surfaces:
  - `preview_regime_surface` for offline diagnostics only
  - `admissible_regime_surface` for any executable, calibration, or promotion-facing path
- Refactor specialist training so regime-conditioned training rows are labeled by the admissible regime visible at the original decision timestamp, not by contemporaneous preview labels.
- Refactor `_train_inner_meta_model(...)` so regime-aware context and filtering use the same admissible regime contracts that will exist at inference.
- Remove direct use of preview-aligned `fit_regime_view` and equivalent label-only frames from executable training paths.
- When no admissible regime exists for a row, force one of three explicit policies:
  - drop row from specialist training
  - map row to `unknown_regime`
  - route row to fallback/generalist evidence only
- Persist regime-surface lineage in every model artifact:
  - surface type
  - admissibility policy
  - timing-blocked row count
  - unknown-regime row count
- Add a strict policy gate for promotion and capital-facing runs:
  - fail closed if regime-aware training was built on preview-only rows

### Validation And Exit Criteria

- Specialist training, meta training, validation, and inference all declare the same regime-surface type in lineage.
- Re-running training with preview labels disabled does not change artifact admissibility status; it only changes performance if leakage was previously present.
- Unknown-regime and timing-blocked rows are measured and surfaced rather than silently absorbed.
- Promotion evidence cannot be created from preview-trained specialist artifacts.

## 3. Tighten Research Evidence Classes And Broaden The Lookahead Guard

### Audit Problem

The default lookahead guard still audits only `build_features` in many research
paths. Regime detection, admissible regime construction, router replay inputs,
and regime-conditioned adaptation can therefore remain unaudited while the
research stack still reports regime-aware performance.

### Institutional Practice Synthesis

Research basis: PRA SS1/23 Principle 4; BCBS 239; NIST AI RMF; NIST AI RMF Playbook.

- PRA validation requires evidence that reflects how the system is actually used.
- BCBS 239 requires risk reporting that stays accurate, complete, and decision-useful under stress, not just in average conditions.
- NIST AI RMF distinguishes measurement from management and expects coverage gaps to be visible, documented, and acted on.

### Technical Implementation Plan

- Expand the default lookahead guard to the full causal surface whenever any of the following are enabled:
  - regime detection
  - regime-aware modeling
  - specialist routing
  - regime-conditioned feature adaptation
  - trade-ready or promotion-facing evaluation
- Add audit artifacts beyond `features`:
  - regime state contracts
  - admissible regime view
  - regime-conditioned transformed features
  - router decision inputs
  - executed routed signals
  - execution prices and execution volume
- Replace the current binary research/evaluation split with explicit evidence classes:
  - `preview_only`
  - `causal_research`
  - `promotion_eligible`
  - `capital_facing`
- Make summary artifacts inherit the weakest uncovered surface. If regime or routing timing was not audited, the entire regime-aware result stays non-promotable.
- Add a coverage report that lists:
  - requested stages
  - audited stages
  - missing stages
  - missing artifacts
  - resulting evidence downgrade
- Extend lookahead regression tests so they provoke failures in:
  - regime admissibility
  - regime-conditioned scaling and masking
  - router replay and signal binding

### Validation And Exit Criteria

- Any regime-aware or router-aware run without full causal-surface audit is automatically marked non-promotable.
- Lookahead coverage explicitly includes regime and routing artifacts whenever those layers are enabled.
- Research summaries can no longer present preview-only regime outputs as if they were executable evidence.
- Backtest and training artifacts carry evidence-class lineage end to end.

## 4. Bind Router Switching Cost And Adaptation Friction To Executed PnL

### Audit Problem

Router switching cost is currently advisory reporting. It is estimated and
attached to the summary, but it does not reduce the equity curve or any reported
performance metric.

### Institutional Practice Synthesis

Research basis: PRA SS1/23 Principles 3 and 4; SEC Rule 15c3-5; BCBS 239.

- PRA expects model use and validation to include the actual operating frictions that determine whether a strategy is viable.
- SEC-style market-access controls emphasize that exposure-changing actions must be controlled before market entry, including the operational consequences of rapid switching.
- BCBS 239 requires stress-relevant risk reporting; reporting switching cost separately while excluding it from PnL is not decision-useful.

### Technical Implementation Plan

- Introduce a router execution-friction model applied inside the backtest engines, not after summary generation.
- Compute executed router turnover from the actual executed weight path and charge it directly to the equity curve.
- Support at least three binding cost components:
  - flat cost per switch event
  - proportional cost per executed allocation turnover unit
  - optional activation-latency penalty for switching into a new specialist
- Apply the same binding cost logic in:
  - pandas backtest path
  - vectorbt backtest path
  - event-driven execution replay path when available
- Add stress scenarios for regime flapping and delayed recognition so switching-cost sensitivity becomes part of promotion evidence.
- Replace advisory-only router stability reports with two layers:
  - descriptive stability diagnostics
  - cost-adjusted realized routing burden

### Validation And Exit Criteria

- `router_switching_cost_estimate` and realized PnL move together because cost is now charged before metrics are computed.
- A high-switch strategy can fail promotion on net economic value even if its gross signal quality remains strong.
- Router path summaries report both gross and post-switch-cost outcomes.
- Flapping stress scenarios visibly penalize the executed equity curve rather than only adding narrative warnings.

## 5. Replace Placeholder Specialist Health Scoring With Executable Runtime Evidence

### Audit Problem

The router claims to use specialist stability, decay, and failure controls, but
default health contracts are largely unbound placeholders. In practice, routing
is either forced into fallback mode or relies on neutral defaults such as
`missing_health_score` instead of measured evidence.

### Institutional Practice Synthesis

Research basis: PRA SS1/23 Principles 2, 3, and 5; SEC Rule 15c3-5; NIST AI RMF.

- PRA governance requires that important controls have owners, evidence, and real operating effect.
- PRA mitigants should reduce model risk in production, not merely decorate summaries.
- SEC 15c3-5 is relevant by analogy because exposure-changing controls must remain under direct control and cannot rely on downstream assumptions.
- NIST AI RMF emphasizes accountability, monitoring, and enforceable operational policy.

### Technical Implementation Plan

- Split specialist routing inputs into three explicit contracts:
  - `SpecialistRegistrationContract` for static library membership
  - `SpecialistEligibilityContract` for bar-local executable permission
  - `SpecialistHealthEvidenceContract` for measured decay, stability, and failure evidence
- Remove `missing_health_score` from executable and promotion-facing modes.
- In executable modes, unresolved health must map to one of two explicit outcomes:
  - fallback-only routing
  - no-trade safe mode
- Populate specialist health evidence from persisted measured sources only:
  - out-of-sample slice performance
  - calibration freshness
  - decay diagnostics
  - runtime failures
  - minimum sample sufficiency
- Thread the same health-evidence trace through runtime, replay, backtest, and promotion artifacts.
- Require lifecycle state and executable permission to be independently logged. A specialist can exist in the library without being executable.
- Add explicit lineage to every health decision:
  - evidence window
  - sample count
  - metric basis
  - last refresh time
  - policy version that resolved eligibility

### Validation And Exit Criteria

- No specialist receives executed exposure in executable modes without a resolved eligibility decision backed by measured evidence.
- Runtime routing, replay routing, and backtest routing agree on which specialists were eligible on each bar.
- Placeholder `None` health values are impossible to interpret as positive routing scores in executable paths.
- Health-driven blocks are reported by cause, lineage, and affected-row share.

## 6. Redesign Drift Monitoring Into Separate Maintenance, Investigation, Recalibration, And Retrain Actions

### Audit Problem

The current drift monitor freezes a static reference window, counts several drift
flags as one evidence total, and can treat TTL expiry plus one drift vote as
sufficient retraining logic. That mixes market turbulence, policy shift,
taxonomy noise, and real edge decay into the same action channel.

### Institutional Practice Synthesis

Research basis: PRA SS1/23 Principles 3, 4, and 5; NIST AI RMF; NIST AI RMF Playbook.

- Institutional practice separates monitoring, recalibration, challenger refresh, and champion replacement.
- PRA highlights the control burden created by dynamic recalibration and model change; frequent small changes must not accumulate into untested material drift.
- NIST AI RMF and the Playbook emphasize ongoing monitoring plus policy-driven responses that distinguish measurement from management.

### Technical Implementation Plan

- Replace the single `should_retrain` recommendation with four separate actions:
  - `maintenance_refresh_recommended`
  - `drift_investigation_recommended`
  - `recalibration_recommended`
  - `structural_retrain_recommended`
- Reclassify TTL expiry as a maintenance trigger only. TTL can authorize challenger refresh or shadow evaluation, but never automatic champion replacement.
- Move from a single frozen reference window to a governed reference bank containing:
  - last training reference
  - recent rolling reference
  - regime-conditioned reference slices
  - event-conditioned reference slices when available
- Detect real degradation on executed-path outcomes, not only on feature and prediction distributions:
  - realized post-cost performance
  - abstain-rate shift relative to expected policy
  - router fallback burden
  - specialist eligibility collapse
- Require persistence and economic materiality before structural retraining:
  - multi-window confirmation
  - minimum adverse economic impact
  - cooldown-aware escalation policy
- Keep router recalibration, library review, challenger training, and champion promotion as separate branches in the orchestration report.
- Suppress regime-drift escalation when taxonomy instability is unresolved.

### Validation And Exit Criteria

- TTL expiry alone can no longer replace the champion.
- Drift reports separate maintenance, recalibration, investigation, and retraining decisions in persisted artifacts.
- Executed-path degradation, not raw distribution noise alone, becomes the binding trigger for structural retraining.
- Taxonomy noise no longer appears as market drift until remapping has completed.

## 7. Stabilize Regime Taxonomy And Routing Compatibility Across Refits

### Audit Problem

The regime detector still produces a taxonomy that is heavily window-relative.
Explicit regimes depend on fit-window quantiles, and the filtered HMM depends on
within-window ordering plus similarity heuristics. Specialists are therefore
bound to routing keys that can drift even when the underlying economics have not.

### Institutional Practice Synthesis

Research basis: PRA SS1/23 Principles 3, 4, and 5; NIST AI RMF; BCBS 239.

- PRA expects model outputs used in decisions to remain interpretable, monitored, and challengeable across model change.
- NIST AI RMF requires traceability and transparency for how changing models map inputs to decisions over time.
- BCBS 239 implies that state definitions used in risk reporting must remain coherent enough to support timely and accurate decisions under stress.

### Technical Implementation Plan

- Decouple routing identity from human semantic label.
- Introduce a versioned taxonomy registry that persists:
  - canonical regime id
  - semantic label
  - state signature summary
  - similarity score to prior deployed states
  - remap reason
  - mapping confidence
- Refactor specialist compatibility so executable routing binds to canonical regime ids or regime families, not ephemeral semantic strings.
- Add a mandatory remap step during retraining:
  - compare new detector states to the currently deployed taxonomy
  - preserve canonical ids when similarity exceeds the governed threshold
  - declare `new_regime_family` explicitly when similarity is insufficient
- Add neighboring-window stability diagnostics to detector training:
  - adjacent-refit agreement
  - state-signature distance
  - unresolved new-family rate
  - compatibility break count
  - remap churn rate
- Prevent regime drift logic and specialist routing changes from reacting to semantic-label renames before canonical remapping completes.
- Add a safe-mode policy for genuinely unseen canonical regimes:
  - fallback-only
  - no-trade
  - shadow-only evaluation

### Validation And Exit Criteria

- Routine retrains do not silently change routing keys when the underlying state family is materially unchanged.
- New or unresolved states are explicit and auditable rather than hidden inside fallback behavior.
- Drift metrics distinguish true distribution change from taxonomy remap noise.
- Specialist compatibility remains stable across routine refits unless a governed taxonomy change is declared.

## Recommended Implementation Order

1. Preserve regime recognition lag through router replay and backtest binding.
2. Unify specialist and meta training with the same admissible regime surface used at inference.
3. Tighten research evidence classes and broaden the lookahead guard.
4. Replace placeholder specialist health scoring with executable runtime evidence.
5. Bind router switching cost and adaptation friction to executed PnL.
6. Stabilize regime taxonomy and routing compatibility across refits.
7. Redesign drift monitoring into separate maintenance, investigation, recalibration, and retrain actions.

## Definition Of Done For This Remediation Program

- No executable or promotion-facing path consumes regime information earlier than its admissible availability contract allows.
- Specialist training and inference use the same regime admissibility rules.
- Research outputs are downgraded automatically when the full causal surface has not been audited.
- Router exposure changes and router costs bind executed PnL rather than only diagnostic summaries.
- Specialist eligibility depends on measured runtime evidence, not placeholder health defaults.
- Drift policy distinguishes maintenance from structural deterioration and does not treat taxonomy noise as market drift.
- Regime routing keys remain stable across routine detector refits, or changes are explicitly remapped, versioned, and governed.