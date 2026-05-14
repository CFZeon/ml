# Improvements Plan

Date: 2026-05-13

This document converts the hostile audit of the regime, routing, specialist, and
drift stack into implementation workstreams. It supersedes older draft sections
that mixed unrelated improvements with the audited failures. Every section below
maps directly to a specific problem identified in the audit.

## Institutional Reference Set

- PRA SS1/23, Model risk management principles for banks.
- BCBS 239, Principles for effective risk data aggregation and risk reporting.
- SEC Rule 15c3-5 market access control fact sheet.
- NIST AI RMF 1.0.
- NIST AI RMF Playbook.
- BIS FSI Occasional Paper No 24, Managing explanations: how regulators can address AI explainability.

## Program Rules

- Capital-facing paths must fail closed when causal timing, calibration, lineage, or runtime eligibility cannot be proven.
- Preview and research artifacts must never be reusable as promotion or deployment evidence without an explicit admissibility transition.
- Router metrics, specialist metrics, and drift metrics must describe the executed path, not a reconstructed narrative path.
- Any adaptive control that can change exposure must expose stable identifiers, persisted state, and replayable decision evidence.

## 1. Enforce Causal Regime Availability In Specialist And Meta Paths

### Audit Problem

The regime-aware specialist path and the inner meta-model path can consume raw
aligned regime labels directly at inference time even though the regime contract
supports delayed availability via `available_at` and `recognition_lag_bars`.
This means the adaptive stack can be evaluated using regime information earlier
than the framework's own causal contract allows.

### Institutional Practice Synthesis

Research basis: PRA SS1/23 Principles 3 and 4, NIST AI RMF, SEC Rule 15c3-5.

- PRA Principle 3 requires model development, implementation, and use to be controlled in a way that matches actual operating conditions rather than optimistic assumptions.
- PRA Principle 4 requires validation of conceptual soundness and implementation, which includes whether timing assumptions are causal and reproducible.
- NIST AI RMF treats validity, reliability, and traceability as lifecycle properties, not post hoc labels.
- SEC 15c3-5 is not a model-risk text, but it is directly relevant to trading controls: pre-trade controls must operate before exposure is taken, not as a retrospective supervisory story.

### Technical Implementation Plan

- Create a single admissible regime-input abstraction for all inference consumers:
  - primary specialist routing
  - inner meta-model training
  - validation-time specialist surface construction
  - test-time specialist routing
- Refactor the pipeline so these consumers receive `RegimeStateContract` rows or a derived admissible view keyed by `available_at`, never a raw aligned label frame.
- Add an explicit helper such as `build_admissible_regime_view(index, regime_states, cutoff_mode)` that:
  - drops states whose `available_at` is later than the decision timestamp
  - preserves `warm`, `unavailable`, and lagged states
  - surfaces fallback causes when no admissible state exists
- Remove any direct use of contemporaneous `regime_data=test_regime_view` or equivalent aligned label frames in specialist inference and inner meta-model training.
- Add a strict policy flag for capital-facing and promotion-facing runs:
  - `causal_regime_only=true`
  - hard failure if the caller attempts to pass label-only regime frames without availability metadata
- Persist regime timing evidence per fold:
  - same-bar recognition rate
  - lagged-recognition rate
  - unavailable-state rate
  - fallback rows due to timing only
- Extend lookahead tests so they explicitly provoke regime timing violations in specialist inference and meta-model training, not only in the main signal path.

### Validation And Exit Criteria

- Specialist inference, inner meta-model training, validation, and test routing all consume the same admissible regime contract.
- A regime state with `available_at > decision_time` is never visible to an executable or promotion-facing path.
- Timing-only fallback rates are reported and auditable.
- Prefix replay and causal replay agree on specialist eligibility whenever the same admissible regime states are supplied.

## 2. Stabilize Regime Taxonomy Before Using It As A Routing Key

### Audit Problem

The HMM detector constructs semantic labels from within-fit relative state means,
and those labels are then used as exact compatibility keys for specialist
selection and routing. The resulting taxonomy can be internally coherent within a
fit while remaining unstable across retrains or neighboring refits.

### Institutional Practice Synthesis

Research basis: PRA SS1/23 Principles 3, 4, and 5, BIS FSI 24, NIST AI RMF.

- PRA model development and validation expectations imply that a model taxonomy used for decisions must be stable enough to be interpreted, monitored, and challenged across model changes.
- PRA mitigants require controls when model outputs can change for reasons unrelated to the underlying risk.
- BIS FSI 24 emphasizes that governance, validation, monitoring, documentation, and independent review become harder when labels and explanations are unstable or misleading.
- NIST AI RMF requires traceability and transparency around how a system maps inputs to decisions over time.

### Technical Implementation Plan

- Decouple `routing identity` from `human semantic label`.
- Introduce a detector-owned `canonical_regime_id` that is stable across retrains whenever the underlying state remains sufficiently similar.
- Maintain a regime mapping artifact in detector manifests containing:
  - canonical state id
  - semantic label
  - state signature summary
  - similarity score to prior deployed states
  - mapping confidence
  - remap reason
- Replace exact string compatibility on `compatible_regimes=[semantic_label]` with compatibility on canonical ids or explicit regime families.
- Add a regime remapping step at retrain time:
  - compare new state signatures to the deployed detector's signatures
  - preserve canonical ids when similarity exceeds threshold
  - mark genuinely new states as `new_regime_family` instead of silently overloading old names
- Prevent drift logic from using raw label-name changes as evidence until the remapping step has run.
- Add taxonomy-stability diagnostics to every detector training report:
  - neighboring-refit agreement
  - state-signature distance
  - remap rate
  - unresolved new-state rate
  - compatibility break count
- Require specialist libraries to bind to canonical regime ids or stable regime-family identifiers, not ephemeral semantic names.

### Validation And Exit Criteria

- Retraining with neighboring windows does not silently change routing keys when state economics are materially unchanged.
- Unmapped states are explicit and auditable rather than hidden inside fallback behavior.
- Regime drift metrics distinguish true state-distribution change from label-renaming noise.
- Specialist compatibility survives routine detector refits unless a real taxonomy change is declared.

## 3. Replace Placeholder Router Health With Runtime Eligibility Evidence

### Audit Problem

The router claims to score specialists with stability, decay, and failure signals,
but the default specialist health contracts are initialized with `None` or empty
values. In default operation, routing can therefore appear health-aware while in
practice relying mostly on regime compatibility plus a neutral missing-health
default.

### Institutional Practice Synthesis

Research basis: PRA SS1/23 Principles 2, 3, and 5, SEC Rule 15c3-5, NIST AI RMF.

- PRA governance requires that important model controls have identified owners and actual operating effect, not placeholder governance artifacts.
- PRA mitigants are meant to reduce model risk in production, not decorate research summaries.
- SEC 15c3-5 is relevant here because exposure-changing controls must remain under direct operator control before execution and cannot be delegated to assumptions about downstream safety.
- NIST AI RMF requires accountability, monitoring, and enforceable operational controls across the lifecycle.

### Technical Implementation Plan

- Split the current router inputs into three distinct contracts:
  - `SpecialistRegistrationContract`: static library membership and metadata
  - `SpecialistEligibilityContract`: bar-local executable eligibility
  - `SpecialistHealthEvidenceContract`: measured performance and degradation evidence
- Remove `missing_health_score` as a default execution-time neutralizer for capital-facing modes.
- In executable modes, unresolved health must produce one of two outcomes:
  - specialist blocked and fallback-only routing
  - explicit `unknown_health_safe_mode` if policy permits no-trade
- Generate bar-local eligibility from persisted evidence and runtime monitors:
  - lifecycle state
  - certification state
  - compatibility with the admissible regime id
  - calibration freshness
  - performance decay thresholds
  - failure flags
  - warm-up requirements
- Thread the same eligibility trace through:
  - runtime routing
  - replay diagnostics
  - backtest summaries
  - promotion evidence
- Require specialist health manifests to contain measured values with lineage:
  - evidence window
  - sample count
  - metric basis
  - last refresh time
  - owner policy that marked the specialist active, degraded, shadow, or blocked
- Add a containment mode for incomplete health programs:
  - `router_execution_policy=fallback_only_until_health_binding`

### Validation And Exit Criteria

- No specialist receives non-zero executed exposure in capital-facing modes without a resolved eligibility decision.
- Replay, backtest, and runtime all agree on which specialists were executable on each bar.
- Health-driven blocks are reported by cause and sample lineage.
- Placeholder `None` health values are impossible to interpret as a positive routing score in executable modes.

## 4. Replace Shared Meta-Model Reuse With Specialist-Specific Edge Validation

### Audit Problem

One meta-model can be trained on out-of-fold primary bundle outputs and then
reused to estimate expected edge for fallback and specialist models separately.
That lets the router compare specialist surfaces using a meta model that was not
trained or validated on each specialist's own prediction distribution.

### Institutional Practice Synthesis

Research basis: PRA SS1/23 Principles 3 and 4, BIS FSI 24, NIST AI RMF.

- PRA conceptual soundness requires that a model be used only in ways consistent with its development assumptions and validation evidence.
- PRA independent validation requires scrutiny of implementation choices that can create hidden extrapolation or misuse of model outputs.
- BIS FSI 24 warns that advanced-model overlays can produce misleading confidence or explanations when applied outside the conditions that justify them.
- NIST AI RMF Measure and Manage functions imply that uncertainty, validity, and monitoring must follow the actual deployed scoring path.

### Technical Implementation Plan

- Immediate containment:
  - disable specialist-specific expected-edge ranking based on a shared bundle-level meta model
  - route only on admissible regime compatibility and certified health until specialist-specific edge evidence exists
- Medium-term replacement:
  - train a separate meta edge model per executable scoring surface, or
  - train a single multi-surface model with an explicit `surface_id` and prove by holdout validation that it is stable across surfaces
- Require any specialist edge model to be trained from that specialist's own out-of-fold predictions, probabilities, and realized outcomes.
- Add per-surface calibration diagnostics:
  - Brier score
  - expected calibration error
  - profitability calibration by decile
  - abstain calibration
  - minimum trade count
- For counterfactual specialist evaluation, use a separately governed shadow-evaluation program rather than treating synthetic cross-surface meta outputs as deployable edge estimates.
- Persist per-specialist meta lineage in artifacts:
  - training folds
  - feature schema
  - outcome basis
  - calibration date
  - admissible operating range
- Hard-block Kelly sizing or expected-utility sizing when the relevant specialist edge model lacks per-surface validation.

### Validation And Exit Criteria

- No specialist ranking in executable routing depends on a meta model validated only on another surface.
- Each deployed edge model has surface-specific calibration and profitability evidence.
- Kelly sizing is disabled automatically when per-surface trade counts or calibration quality are insufficient.
- Shadow specialist edge estimates are labeled as non-executable unless they pass the same validation program as live surfaces.

## 5. Prevent Candidate Specialists From Entering Executable Research Paths

### Audit Problem

New specialists default into candidate state, yet the router can still fall back to
candidate models when no active or certified models exist. That means research and
backtest performance can depend on specialists that have not passed the library's
own lifecycle controls.

### Institutional Practice Synthesis

Research basis: PRA SS1/23 Principles 2, 3, 4, and 5, SEC Rule 15c3-5, NIST AI RMF.

- PRA governance and validation principles require explicit control over model changes and production use, especially when model components change exposure.
- Candidate, shadow, and active states must have different operating permissions; otherwise lifecycle governance is cosmetic.
- SEC market-access practice is a useful analogue: controls must remain under the operator's control and be applied before orders are exposed to the market.
- NIST AI RMF requires documented policies for when systems can move from development into operational use.

### Technical Implementation Plan

- Separate `research evaluation`, `shadow evaluation`, and `executable routing` permissions.
- Change router candidate selection policy for executable modes to:
  - use only `active_model_ids`
  - optionally allow `certified_model_ids` if explicitly configured
  - never auto-include `candidate_model_ids`
  - never auto-include `degraded_model_ids` unless the safe-mode policy explicitly authorizes a fallback exception
- Add a `shadow_specialist_surfaces` path so candidate specialists can still be evaluated counterfactually without contributing to executed exposure.
- Require a formal transition artifact before a specialist becomes executable:
  - promotion decision
  - calibration evidence
  - minimum sample evidence
  - regime compatibility evidence
  - rollback target
- Persist lifecycle transitions with timestamps and approver policy lineage.
- Add pipeline assertions so promotion-facing or capital-facing summaries cannot claim performance from candidate specialists in the executed path.

### Validation And Exit Criteria

- Executed backtests and live routing never allocate to candidate specialists unless an explicit research override is enabled.
- Shadow candidate performance is reported separately from executable performance.
- Promotion review can reconstruct exactly when a specialist moved from candidate to executable status.
- Removal of candidate exposure changes executed-path metrics whenever older runs were benefitting from unapproved specialists.

## 6. Tighten Drift Governance So Retraining Is Not Calendar-Driven Or Taxonomy-Driven Noise

### Audit Problem

The drift workflow can approve retraining on model TTL expiry even without drift
evidence, while regime drift is partly computed from categorical regime label and
transition distributions that may themselves be unstable when the detector
taxonomy changes.

### Institutional Practice Synthesis

Research basis: PRA SS1/23 Principles 3, 4, and 5, NIST AI RMF, BIS FSI 24.

- Institutional model-risk practice distinguishes monitoring, recalibration, challenger training, and production replacement; those are not interchangeable actions.
- PRA mitigants and validation principles imply that automatic model replacement should not be driven by maintenance timers alone.
- NIST AI RMF Manage and Govern functions emphasize ongoing monitoring, policy-driven response, and documented decision thresholds.
- BIS FSI 24 highlights the governance burden created when complex model outputs are unstable or hard to interpret across changes.

### Technical Implementation Plan

- Split current `should_retrain` into three separate recommendations:
  - `maintenance_refresh_recommended`
  - `drift_investigation_recommended`
  - `structural_retrain_recommended`
- Reclassify TTL expiry as a maintenance trigger only. TTL alone may schedule challenger training or recalibration review, but it must not authorize automatic promotion.
- Gate structural retraining on durable evidence:
  - persistent multi-window drift evidence, or
  - a critical-event override with explicit operator acknowledgement
- Refactor regime drift so it operates on stable regime identities or detector score distributions, not raw semantic labels alone.
- Add a `taxonomy_instability_guard` that suppresses regime-distribution drift escalation until remapping and stability checks have completed.
- Require challenger replacement to satisfy both:
  - post-selection evidence better than the champion on admissible holdout criteria
  - post-retrain warm-up or shadow evidence if triggered by maintenance only
- Persist separate drift channels and actions:
  - feature drift
  - score drift
  - action drift
  - stable-regime drift
  - performance drift
  - maintenance TTL
- Add a formal retrain decision report with action lineage:
  - why a challenger was trained
  - why recalibration was sufficient or insufficient
  - whether taxonomy instability was present
  - why promotion was or was not allowed

### Validation And Exit Criteria

- TTL expiry alone cannot produce automatic champion replacement.
- Regime drift alarms do not fire on pure label remapping events.
- Maintenance refreshes, recalibrations, challenger trainings, and promotions are visibly distinct actions in logs and registry artifacts.
- Historical drift decisions can be replayed and explained without relying on unstable semantic label names.

## 7. Replace Same-Fold Specialist Uplift And Mean-Path Narratives With Locked-Holdout, Cost-Aware Evidence

### Audit Problem

Specialist incremental evidence is currently framed as same-fold accuracy lift
versus the fallback generalist, and aggregate CPCV or path summaries can average
away the exact tail-path failures that matter most in live trading. This can make
the adaptive architecture look stronger than the tradable evidence supports.

### Institutional Practice Synthesis

Research basis: PRA SS1/23 Principle 4, BCBS 239, NIST AI RMF, BIS FSI 24.

- PRA independent validation requires evidence that is relevant to the model's actual use, not only internally favorable summary statistics.
- BCBS 239 requires risk reporting that is accurate, timely, and decision-useful under stress; tail-path blindness is not acceptable risk reporting.
- NIST AI RMF Measure and Manage functions support outcome-based evaluation, uncertainty visibility, and escalation when aggregate summaries hide important failure modes.
- BIS FSI 24 reinforces that persuasive narratives and explanations are not substitutes for reliable validation evidence.

### Technical Implementation Plan

- Replace same-fold specialist uplift as the default evidence class with a layered evidence stack:
  - locked-holdout post-cost uplift versus champion fallback
  - worst-fold and worst-path degradation
  - transition-segment behavior
  - fallback dependence under unseen or unavailable regimes
  - calibration and abstain quality
  - significance or stability intervals where sample size permits
- Treat accuracy lift as a secondary descriptive metric, never as a promotion control.
- Expand CPCV and path summaries to report tail-sensitive statistics in addition to means:
  - minimum
  - lower decile or lower quartile
  - conditional worst-regime performance
  - maximum switch-cost burden
  - maximum fallback share
  - maximum eligible-specialist collapse
- Add promotion blockers keyed to adverse tails rather than only average uplift.
- Require specialist evidence to be traced to admissible rows only:
  - no unavailable regime rows
  - no blocked-by-health rows
  - no candidate-only execution rows
- Add a dedicated `adaptive_value_report` artifact covering:
  - economic incremental value after costs
  - robustness across paths and regimes
  - dependency on fallback behavior
  - sensitivity to taxonomy remaps and recalibration

### Validation And Exit Criteria

- A specialist architecture that improves mean accuracy but worsens tail economic outcomes fails promotion.
- Promotion evidence exposes worst-path, worst-regime, and worst-transition behavior alongside central tendencies.
- Same-fold uplift is clearly labeled descriptive and non-evidentiary.
- Adaptive-value reports can explain whether specialist benefit remains after costs, fallback, and adverse-path analysis.

## Recommended Implementation Order

1. Enforce causal regime availability in specialist and meta paths.
2. Stabilize regime taxonomy before using it as a routing key.
3. Replace placeholder router health with runtime eligibility evidence.
4. Prevent candidate specialists from entering executable research paths.
5. Replace shared meta-model reuse with specialist-specific edge validation.
6. Tighten drift governance so retraining is not calendar-driven or taxonomy-driven noise.
7. Replace same-fold specialist uplift and mean-path narratives with locked-holdout, cost-aware evidence.

## Definition Of Done For This Audit Remediation Program

- No executable or promotion-facing path consumes regime information earlier than its admissible availability contract allows.
- Specialist routing keys remain stable across routine detector refits, or changes are explicitly remapped and governed.
- Router exposure changes depend only on resolved runtime eligibility and measured health evidence.
- No executable allocation depends on a candidate specialist or a meta edge surface that lacks surface-specific validation.
- Drift policy distinguishes maintenance from structural deterioration and does not treat taxonomy noise as market drift.
- Promotion evidence is locked-holdout, cost-aware, tail-sensitive, and tied to admissible executed rows.