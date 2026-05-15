# Improvements Plan

Date: 2026-05-14

This document converts the 2026-05-14 adversarial audit into implementation
workstreams. It supersedes the earlier 2026-05-14 draft because that version
covered adjacent weaknesses but did not map cleanly to the eight failures now
confirmed in the executable routing, drift, validation, and backtest paths.

## Coverage Map

The implementation program below covers all eight problems found in the audit:

1. Reported specialist edge is not always the executable specialist edge.
2. Stale regime states can still steer router decisions.
3. Drift governance authorizes adaptation on model age alone.
4. Regime identity continuity is heuristic and unstable across refits.
5. Specialist formation and certification are under-regularized.
6. Research execution defaults are looser than executable policy defaults.
7. Preview-only regime artifacts can still shape research conclusions.
8. Adaptive-value evidence is descriptive rather than independent.

## Institutional Reference Set

- PRA SS1/23, Model risk management principles for banks.
- PRA PS6/23, Model risk management principles for banks.
- ESMA MiFID II Article 17, Algorithmic trading.
- NIST AI RMF 1.0.
- NIST AI RMF Playbook, especially Measure and Manage.

## Program Rules

- Governance-facing evidence must be produced on the same executable surface that would control capital.
- Decision-time admissibility, freshness, and fallback policy must bind training replay, validation, backtest, and promotion.
- Unconfigured defaults must be conservative and centrally resolved.
- Exploratory artifacts may exist, but they must be explicitly tagged and barred from automated selection or promotion.
- Maintenance refresh, recalibration, challenger training, and champion replacement are separate actions with separate evidence thresholds.

## 1. Bind Specialist Evidence To Executable Router Eligibility

### Audit Problem

`core/regime_training.py` can still report specialist performance from direct
bundle inference paths that bypass the executable router constraints enforced in
`core/routing/router.py`. That means a specialist can appear valuable in
validation even when the deployed router would block it because health,
compatibility, or fallback policy is unresolved.

### Institutional Practice Synthesis

Research basis: PRA SS1/23 Principles 3, 4, and 5; NIST AI RMF Playbook
MEASURE 1.3, 2.3, 2.5, and 2.9; ESMA MiFID II Article 17.

- PRA expects model development, implementation, use, validation, and risk mitigants to apply to the actual decision path, not to an easier proxy path.
- NIST expects deployment-context evaluation, independent assessment, and documented limitations when measured performance differs from operating conditions.
- MiFID II Article 17 expects algorithmic trading controls, thresholds, monitoring, and records for the effective system actually used in trading.

### Technical Implementation Plan

- Refactor `RegimeAwareModelBundle.predict_with_probability_report()` and all certification or summary paths in `core/regime_training.py` so governance metrics are generated through a router-mediated evaluation harness rather than direct specialist calls.
- Build a fold-local executable evaluation surface that combines:
  - admissible regime states from `core/regimes/online_state.py`
  - router eligibility and safe-mode policy from `core/routing/router.py`
  - point-in-time specialist health contracts from `core/specialists/contracts.py`
  - the same fallback and no-trade behavior used in backtest and inference
- Emit two distinct evidence classes for specialist bundles:
  - `research_direct_model_skill` for descriptive research only
  - `executable_routed_skill` for certification, promotion, and user-facing summaries
- Make `executable_routed_skill` the only metric family allowed into `core/promotion.py`, `core/readiness.py`, and any trade-ready experiment summaries.
- Treat missing health evidence, unresolved compatibility, stale regime state, and router safe mode as first-class blockers that reduce executable coverage instead of silently disappearing from metrics.
- Persist a per-decision router trace alongside every specialist evaluation artifact so later reviewers can reconstruct why a specialist was selected, blocked, downgraded, or replaced by fallback.
- Add regression tests proving that a specialist cannot be certified, promoted, or cited as trade-ready if the reported gain exists only when bypassing router constraints.

### Validation And Exit Criteria

- Every specialist certification report contains routed coverage, blocked-decision counts, fallback share, and a replayable router trace.
- No promotion-facing metric can be produced from direct specialist inference alone.
- A negative test with missing health evidence or incompatible routing state prevents specialist certification even when the raw specialist model scores well.

## 2. Fail Safe On Stale Regime State Availability

### Audit Problem

`core/routing/diagnostics.py` carries forward admissible regime states after the
regime evidence is stale, and `core/routing/router.py` does not reliably enter a
safe mode when regime availability is expired. The router can therefore keep
steering exposure using stale state labels rather than fresh admissible evidence.

### Institutional Practice Synthesis

Research basis: ESMA MiFID II Article 17; NIST AI RMF Playbook MEASURE 2.6,
MEASURE 3.1, MANAGE 2.4, and MANAGE 4.1; PRA SS1/23 Principles 3 and 5.

- MiFID II Article 17 requires resilient systems, appropriate thresholds and limits, proper testing, continuous monitoring, and business continuity controls for algorithmic trading.
- NIST expects AI systems to fail safely when operating beyond their knowledge limits and to include monitoring, override, deactivation, and incident-response plans.
- PRA expects model limitations and mitigating controls to bind actual use rather than remain descriptive footnotes.

### Technical Implementation Plan

- Extend regime contracts in `core/regime.py` and `core/regimes/online_state.py` to include explicit freshness metadata:
  - `available_at`
  - `expires_at`
  - `max_age`
  - `freshness_state` in `{fresh, delayed, stale, unavailable}`
- Replace indefinite carry-forward behavior in `build_admissible_router_regime_trace()` and related backtest alignment helpers with bounded carry-forward that stops at `expires_at`.
- Change `_classify_regime_availability()` and `_resolve_safe_mode_state()` in `core/routing/router.py` so stale regime inputs are treated as a control breach rather than as still-usable routing evidence.
- Add mode-specific stale-input policy:
  - research: fallback-only or shadow-only
  - paper/live-capital modes: fallback-only or no-trade
- Record regime age, stale-state counts, and time-since-last-fresh-regime in router traces, diagnostics, and backtest summaries.
- Add a router-level kill switch that forces safe mode when stale or unavailable regime coverage exceeds governed thresholds over a rolling window.
- Extend the backtest binding in `core/backtest.py` so stale regime control behavior changes executed exposure, not just attached diagnostics.

### Validation And Exit Criteria

- Router traces never classify expired regime states as valid specialist-routing inputs.
- A simulated regime feed outage pushes the router into the configured safe mode within one freshness interval.
- Backtest summaries report stale and unavailable regime exposure shares, and those shares materially affect executed routing outcomes.

## 3. Separate Maintenance Refresh From Evidence-Triggered Adaptation

### Audit Problem

`core/drift.py` exposes `min_ttl_drift_signals` but does not enforce it in the
recommendation and approval path. TTL expiry alone can therefore authorize a
maintenance refresh, and the current guardrail language blurs routine hygiene,
recalibration, investigation, and genuine retraining triggers.

### Institutional Practice Synthesis

Research basis: PRA PS6/23 dynamic recalibration feedback; PRA SS1/23 Principles
2 through 5; NIST AI RMF Playbook MEASURE 3.1, MANAGE 1.3, and MANAGE 4.1.

- PRA explicitly calls out the risk that a sequence of small recalibrations can accumulate into a material model change without being properly tested and governed.
- PRA model governance expects model changes, monitoring, validation, escalation, and mitigants to remain distinct rather than collapse into one undifferentiated refresh action.
- NIST expects documented risk responses, post-deployment monitoring, change management, and clear response procedures when measured behavior departs from intended use.

### Technical Implementation Plan

- Redesign the action vocabulary in `core/drift.py` and `core/orchestration.py` into separate channels:
  - `maintenance_refresh`
  - `investigation`
  - `recalibration_review`
  - `challenger_training`
  - `structural_invalidation`
  - `promotion_review`
- Enforce `min_ttl_drift_signals` as a hard precondition for any action that claims adaptive need rather than simple maintenance hygiene.
- Make TTL expiry sufficient only for a scheduled maintenance check, never for a statement that the deployed model has deteriorated or that recalibration is justified.
- Add evidence-gating rules that require minimum sample counts, persistence across time, and cross-channel confirmation before escalation from maintenance to investigation or retraining.
- Split drift evidence by source in `DriftMonitor.check()` and `_resolve_drift_recommendation_channels()`:
  - input or feature drift
  - regime drift
  - prediction drift
  - realized performance drift
  - taxonomy or mapping instability
  - structural break evidence
- Require every non-maintenance action in `run_drift_retraining_cycle()` to emit an evidence bundle showing which channels fired, for how long, on what sample base, and under which cooldown and minimum-data policies.
- Rewrite the current tests that encode TTL-only approval as acceptable drift escalation behavior. TTL-only outcomes should remain allowed only for maintenance bookkeeping.

### Validation And Exit Criteria

- TTL expiry without corroborating drift evidence yields `maintenance_refresh` only.
- No recalibration, challenger training, or structural invalidation action can occur without explicit evidence and minimum-data support.
- Drift reports show action class, triggering evidence, persistence window, cooldown state, and last independent validation date.

## 4. Stabilize Regime Identity Across Refits With Governed Canonical Mapping

### Audit Problem

`core/regimes/detectors.py` still uses window-relative heuristics and similarity
matching to map refit HMM states into canonical regime labels. That leaves
specialist compatibility, drift interpretation, and routing continuity exposed to
semantic churn even when the underlying market state family may not have changed.

### Institutional Practice Synthesis

Research basis: PRA SS1/23 Principles 1, 3, and 4; PRA PS6/23 model change and
documentation feedback; NIST AI RMF and Playbook MEASURE 2.9.

- PRA expects model inventory, model change documentation, independent validation, and interpretability of decision-relevant outputs across model updates.
- PRA feedback on dynamic recalibration and model changes makes clear that repeated small changes cannot be allowed to drift into material output changes without governed review.
- NIST requires traceable, interpretable, documented output behavior so changing state definitions do not silently alter downstream decisions.

### Technical Implementation Plan

- Introduce a versioned taxonomy registry in `core/regime.py` or `core/storage.py` that persists for every regime state:
  - canonical regime id
  - regime family id
  - semantic label
  - state signature summary
  - similarity to prior deployed state
  - mapping confidence
  - remap reason
  - predecessor and successor lineage
- Refactor `_build_hmm_canonical_state_map()` and related helpers in `core/regimes/detectors.py` so semantic names never serve as the durable routing identity.
- Require each retrain to perform an explicit remap step against the currently deployed taxonomy before specialist compatibility or drift scoring is computed.
- Add unresolved outcomes for weak matches:
  - `same_family_low_confidence`
  - `new_regime_family`
  - `needs_shadow_period`
- Prevent unresolved or newly created regime families from immediately inheriting specialist eligibility without governed compatibility review.
- Add adjacent-window and neighboring-refit stability diagnostics:
  - remap churn rate
  - compatibility break count
  - signature distance to prior family
  - unresolved family rate
  - routing-key change count
- Teach drift logic to separate market drift from taxonomy remap noise until canonical mapping completes.

### Validation And Exit Criteria

- Routine detector refits preserve canonical routing identity when the state family is materially unchanged.
- Unresolved or new regime families are explicit, versioned, and auditable rather than hidden inside fallback behavior.
- Drift reports distinguish taxonomy churn from genuine market-state change.

## 5. Harden Specialist Formation, Validation, And Certification Thresholds

### Audit Problem

`core/regime_training.py` can still form regime specialists with very limited
support, using raw row-count thresholds and basic class-count checks. That is not
enough for sparse, overlapping, or short-lived regimes, and it creates a direct
path to overfit specialists with misleading certification summaries.

### Institutional Practice Synthesis

Research basis: PRA SS1/23 Principles 1, 3, and 4; PRA PS6/23 proportionality
and model tiering feedback; NIST AI RMF Playbook MEASURE 2.5 and MANAGE 1.1.

- PRA expects model tiering and proportional validation intensity based on model complexity, use, and risk, not on a single crude threshold.
- PRA validation expectations imply that models used in material decisions need evidence that is appropriate to their data support and deployment impact.
- NIST expects deployed systems to be shown valid and reliable for conditions similar to deployment, and to document where that generalization does not hold.

### Technical Implementation Plan

- Replace `min_samples_per_regime` as the primary gate with an effective sufficiency contract in `core/regime_training.py` and `core/specialists/governance.py` that includes:
  - uniqueness-weighted effective sample size
  - per-class support after abstain filtering
  - number of independent regime episodes
  - routed decision count
  - expected executed trade count under the active router
  - feature coverage and missingness stability
- Introduce specialist lifecycle states in `core/specialists/library.py`:
  - `research_only`
  - `shadow`
  - `executable`
  - `degraded`
  - `retired`
- Require sparse regimes to merge into a parent family, defer to a pooled model, or remain in shadow mode instead of fitting a fully independent executable specialist.
- Add robustness checks across adjacent windows, stressed slices, and small taxonomy perturbations before a specialist can enter executable status.
- Require certification reports to disclose effective sample metrics, routed coverage, execution count, turnover, failure envelope, and reasons for any governance overrides.
- Tighten promotion gates so specialist certification depends on executable routed evidence rather than on raw specialist model fit.
- Add negative tests proving that single-episode or low-coverage regimes cannot become executable specialists by passing only a row-count threshold.

### Validation And Exit Criteria

- No specialist reaches executable status without meeting effective sample, routed coverage, and robustness thresholds.
- Sparse regimes are merged, shadowed, or retired rather than overfit into executable specialists.
- Specialist certification artifacts expose why a specialist is trusted, not just its score.

## 6. Make Research Execution Defaults Conservative And Executable

### Audit Problem

`core/pipeline.py` still resolves certain backtest execution parameters more
permissively than the defaults defined in `core/execution/policies.py`, including
`participation_cap=1.0` and `min_fill_ratio=0.0` when config is absent. That
means research-mode results can underprice switching friction, partial fills, and
crisis execution failure relative to the project's own execution policy surface.

### Institutional Practice Synthesis

Research basis: ESMA MiFID II Article 17; PRA SS1/23 Principles 3 and 5; NIST AI
RMF Playbook MEASURE 2.3 and 2.6.

- MiFID II Article 17 expects trading systems to operate with effective thresholds, limits, testing, and monitoring, not with silent fallbacks to permissive behavior.
- PRA expects model use to reflect implementation constraints and model risk mitigants rather than assuming away hard-to-execute conditions.
- NIST expects systems to be demonstrated as valid in deployment-like conditions and to fail safely when operating beyond supported assumptions.

### Technical Implementation Plan

- Remove permissive inline execution defaults from `core/pipeline.py`. All execution-policy resolution should flow through `resolve_execution_policy()` in `core/execution/policies.py`.
- Define explicit profiles for `research_exploratory`, `research_promotion_candidate`, `paper`, and `capital_facing` modes, with monotonic conservatism rules so a looser mode cannot masquerade as a stricter one.
- Make the unconfigured fallback profile at least as conservative as the current centrally defined execution defaults.
- Bind participation limits, fill ratios, switching costs, crisis widening, delay, and stale-routing penalties to executed fills and executed PnL in `core/backtest.py`, not only to attached diagnostics.
- Require any intentionally relaxed assumption to be explicit in config and visibly watermark all downstream reports as exploratory and non-promotion-eligible.
- Add consistency tests proving `core/pipeline.py` cannot silently override stricter defaults from `core/execution/policies.py`.
- Extend backtest outputs to show how much PnL changed because of switching friction, partial-fill shortfall, and crisis execution constraints.

### Validation And Exit Criteria

- Unconfigured backtests inherit the central execution-policy defaults.
- Relaxed assumptions never appear in promotion-ready summaries without an explicit exploratory label.
- Changing execution-friction assumptions changes executed PnL and routed trade counts, not only a side diagnostic.

## 7. Quarantine Preview-Only Regime Artifacts From Research Selection

### Audit Problem

`core/pipeline.py` blocks preview-only regime artifacts only in capital-facing
modes. The same preview-only or retrospective regime narratives can still shape
research conclusions, leaderboard ordering, and human selection even when those
artifacts are not admissible on the executable causal surface.

### Institutional Practice Synthesis

Research basis: PRA SS1/23 Principles 3 and 4; NIST AI RMF Playbook MEASURE 2.1,
2.3, 2.5, and MANAGE 1.1.

- PRA expects model use and independent validation to focus on the actual decision context, not on results that depend on unavailable information.
- NIST expects TEVV artifacts, deployment-context limitations, and go or no-go decisions to be documented clearly enough that exploratory evidence cannot be confused with executable evidence.
- Good governance practice requires descriptive diagnostics to remain visible but clearly barred from automated selection and promotion claims.

### Technical Implementation Plan

- Introduce an explicit evidence-class schema in `core/evaluation_modes.py`, `core/pipeline.py`, and `core/automl.py`:
  - `executable`
  - `replay_audited`
  - `preview_only`
  - `retrospective_exploratory`
- Prevent `preview_only` and `retrospective_exploratory` artifacts from entering:
  - objective functions
  - leaderboard scores
  - promotion manifests
  - readiness checks
  - automated configuration selection
- Split research reporting into separate causal and exploratory sections so diagnostic regime narratives remain available without contaminating selection decisions.
- Extend the lookahead guard so preview-only regime data cannot leak into regime coverage summaries, adaptive-value summaries, threshold selection, or feature-selection shortcuts.
- Stamp every report, manifest, and chart with its evidence class and fail CI when a promotion-facing artifact includes preview-only rows.
- Add tests that attempt to route preview-only artifacts into selection paths and assert hard failure.

### Validation And Exit Criteria

- Preview-only artifacts remain accessible for diagnostics but cannot influence automated selection or promotion decisions.
- Research reports visibly separate executable evidence from retrospective narrative.
- CI fails if preview-only evidence is aggregated into any promotion-facing metric.

## 8. Require Independent Adaptive-Value Evidence Before Claims Or Promotion

### Audit Problem

Adaptive-value and coverage summaries in `core/pipeline.py` and `core/automl.py`
are still largely descriptive. They can be built from fallback rows, same-fold
degradations, or other non-independent slices rather than from locked,
out-of-time, executable evidence showing that adaptation improves decisions under
the actual router and execution surface.

### Institutional Practice Synthesis

Research basis: PRA SS1/23 Principle 4; PRA PS6/23 review and effective challenge
themes; NIST AI RMF Playbook MEASURE 1.3, 2.3, 2.5, 2.13, and MEASURE 4.2.

- PRA expects independent validation and effective challenge of materially important performance claims.
- NIST expects independent assessors, deployment-context evaluation, documented TEVV methods, and measurable improvement or decline tracking over time.
- Institutional practice treats counterfactual or adaptive benefit claims as governed evidence only when they are demonstrated on cohorts that were not used to invent the claim.

### Technical Implementation Plan

- Redefine adaptive-value reporting in `core/automl.py`, `core/pipeline.py`, and `core/promotion.py` around locked, out-of-time, router-mediated evaluation cohorts.
- Require all adaptive-value claims to compare executable surfaces on the same decision set:
  - routed adaptive policy
  - governed fallback baseline
  - no-trade or broad-market baseline where relevant
- Exclude same-fold degradations, preview-only rows, and fallback-only descriptive slices from promotion-facing adaptive-value summaries unless they are explicitly labeled descriptive and non-independent.
- Add holdout registry support in `core/storage.py` or `core/promotion.py` so each adaptive-value claim records:
  - cohort lock date
  - cohort generation rule
  - evidence class
  - comparator policy
  - execution policy profile
  - routed coverage
- Require stability checks on the claimed delta using post-selection evidence only, such as fold-consistency thresholds or bootstrap uncertainty intervals on the locked holdout.
- Downgrade adaptive-value claims automatically to exploratory status when the independent executable cohort fails, even if descriptive same-fold diagnostics still look positive.
- Add tests ensuring promotion summaries cannot cite adaptive benefit unless the required independent cohort artifact exists and passes the configured governance threshold.

### Validation And Exit Criteria

- Every adaptive-value claim points to an independent, locked, executable cohort artifact.
- Same-fold or fallback-only descriptive gains can never appear as promotion evidence.
- Promotion summaries disclose comparator policy, routed coverage, and evidence class for every adaptive-value claim.

## Recommended Implementation Order

1. Fail safe on stale regime state availability.
2. Bind specialist evidence to executable router eligibility.
3. Harden specialist formation, validation, and certification thresholds.
4. Stabilize regime identity across refits with governed canonical mapping.
5. Separate maintenance refresh from evidence-triggered adaptation.
6. Make research execution defaults conservative and executable.
7. Quarantine preview-only regime artifacts from research selection.
8. Require independent adaptive-value evidence before claims or promotion.

## Definition Of Done For This Remediation Program

- No training, validation, backtest, or promotion path can bypass the same router, admissibility, freshness, and fallback constraints that would control execution.
- Expired regime evidence forces governed safe-mode behavior instead of silently steering exposure.
- Drift monitoring distinguishes routine maintenance from evidence-backed adaptation and records the justification for each action.
- Regime identities remain stable across routine refits, or changes are explicitly remapped, versioned, and governed.
- Specialists can only become executable when their routed evidence, data sufficiency, and robustness support that decision.
- Unconfigured execution assumptions are conservative and materially bind executed PnL.
- Preview-only and retrospective artifacts remain visible for diagnostics but cannot influence automated selection or promotion.
- Adaptive-value claims are independently evidenced on locked executable cohorts or are explicitly downgraded to exploratory status.