# Improvements Plan

Date: 2026-05-13

This document converts the adversarial audit into implementation workstreams.
It is not a feature wishlist. Each section corresponds to a specific failure mode
that can invalidate research conclusions or amplify losses in production.

## Institutional Reference Set

- BCBS 239, Principles for effective risk data aggregation and risk reporting: https://www.bis.org/publ/bcbs239.htm
- SEC Rule 15c3-5 fact sheet, market access controls: https://www.sec.gov/news/press/2010/2010-210.htm
- NIST AI RMF overview: https://www.nist.gov/itl/ai-risk-management-framework
- NIST AI RMF Playbook: https://airc.nist.gov/AI_RMF_Knowledge_Base/Playbook/
- NIST AI RMF FAQs: https://www.nist.gov/itl/ai-risk-management-framework/ai-risk-management-framework-faqs
- BIS FSI Occasional Paper No 24, Managing explanations: how regulators can address AI explainability: https://www.bis.org/fsi/fsipapers24.htm

## Working Rules For This Plan

- Control metrics must describe the executed path, not a proxy path.
- Every adaptive control must carry explicit data lineage, timing, and state persistence.
- Capital-facing modes must fail closed when the framework cannot prove causal inputs, calibrated uncertainty, or operational headroom.
- Preview, research, validation, and deployment artifacts must remain explicitly separated.

## 1. Bind Router Stability To Executed Exposure

### Audit Problem

The current weighted router can block a model switch via hysteresis or cooldown while
still allocating execution weight to the blocked challenger. Router stability metrics
count only selected-model switches, and router switching costs are attached as
hypothetical diagnostics rather than realized costs.

### Institutional Practice Synthesis

Research basis: SEC Rule 15c3-5, NIST AI RMF, NIST AI RMF Playbook.

- SEC 15c3-5 requires automated controls to apply before market access and remain under direct, exclusive control of the operator.
- NIST AI RMF and the Playbook emphasize that monitoring and risk treatment must reflect the actual deployed behavior across the lifecycle.
- A routing-control framework is not institution-grade if the supervisory metric describes one decision path while execution follows another.

### Technical Implementation Plan

- Split router semantics into two explicit modes:
  - `selection_only`: exactly one model may receive non-zero executed weight.
  - `mixture_allocation`: multiple models may receive weight, but mixture eligibility must be determined before weights are generated.
- Refactor `core/routing/router.py` so hysteresis, persistence, cooldown, and safe-mode controls operate on the executed candidate set, not only on `selected_model_id`.
- In `WeightedRouter`, zero out any candidate that is blocked by hysteresis, cooldown, lifecycle state, health state, or compatibility constraints before softmax normalization.
- Extend `RoutingDecisionContract` to carry:
  - `executed_candidate_ids`
  - `executed_weight_turnover`
  - `executed_weight_l1_change`
  - `effective_model_count`
  - `allocation_control_reason`
- Replace `router_switch_count` as the primary stability measure with a richer executed-exposure report:
  - discrete switch count
  - weight turnover
  - concentration drift
  - effective candidate count
  - blocked-allocation count
- In `core/pipeline.py`, keep `_route_signal_state_with_router(...)` and router diagnostics on the same contract and same decision object so the bound path and the reported path cannot diverge.
- In `core/backtest.py`, charge router costs from realized weight deltas and routed exposure turnover, not only from model-id switches.
- Promote router stability gates to read executed-allocation metrics first, with legacy switch-count metrics retained only for backward-compatible reporting.

### Validation And Exit Criteria

- A `hysteresis_hold` or `cooldown_hold` decision implies zero executed weight to the blocked challenger in `selection_only` mode.
- `mixture_allocation` mode reports non-zero allocation turnover whenever weights change even if `selected_model_id` stays constant.
- Backtest PnL changes when router weight churn changes.
- Promotion and stress gates fail on allocation churn even when model-id switch rate is low.

## 2. Make Specialist Execution Honor Health, Lifecycle, And True Regime Eligibility

### Audit Problem

The executed routing path ignores dynamic specialist health updates and evaluates all
specialist surfaces on all rows using the same downstream meta path. That creates a
correlated ensemble disguised as a governed specialist architecture.

### Institutional Practice Synthesis

Research basis: BIS FSI 24, NIST AI RMF, SEC Rule 15c3-5.

- BIS FSI 24 ties acceptable advanced-model deployment to governance, documentation, validation, deployment, monitoring, and independent review.
- NIST AI RMF requires accountability and traceability over who can act, when, and under what constraints.
- SEC market-access practice rejects reliance on customer assurances in place of operator-controlled gating; the analogous rule here is that runtime eligibility cannot be inferred from static specialist registration alone.

### Technical Implementation Plan

- Thread `specialist_health_trace` through `_route_signal_state_with_router(...)` so execution and replay consume the same time-local health payload.
- Introduce a runtime `SpecialistEligibilityContract` derived per bar from:
  - compatible regime label
  - lifecycle state
  - certification status
  - calibration freshness
  - health failure flags
  - decay and stability thresholds
- Refactor `_build_specialist_signal_surfaces(...)` to support row-level masking:
  - incompatible or degraded specialists emit `NaN` or explicit `no_trade`, not live predictions.
  - fallback generalist fills only the rows where no specialist remains eligible.
- Separate `specialist prediction availability` from `router candidate availability` so allocation cannot silently blend disallowed specialists.
- Add per-specialist meta-context support or per-specialist calibration identifiers so the meta layer no longer homogenizes all specialists by default.
- Store eligibility traces in backtest artifacts and registry manifests so promotion review can audit why a specialist was routable on each segment.

### Validation And Exit Criteria

- Execution and replay produce identical routed allocations when fed the same health trace.
- A degraded or failed specialist never contributes non-zero execution weight.
- Fallback usage is measurable by cause: unseen regime, health failure, lifecycle block, calibration expiry, or missing regime state.
- Specialist performance reports are computed only on eligible rows.

## 3. Remove Synthetic Confidence And Enforce Real Availability Timing

### Audit Problem

Several regime paths fabricate certainty by converting labels into probability 1.0 and
defaulting `available_at` to the same timestamp as `as_of`. That overstates confidence
and makes transitions visible earlier than a live system could know them.

### Institutional Practice Synthesis

Research basis: NIST AI RMF, NIST AI RMF Playbook, BIS FSI 24.

- NIST AI RMF treats validity, reliability, transparency, and accountability as lifecycle properties, not labels applied after the fact.
- The Playbook emphasizes measurement and documentation of uncertainty rather than implicit certainty.
- BIS FSI 24 explicitly warns that explanation and uncertainty tooling can be unstable or misleading if it does not correspond to how the model actually works.

### Technical Implementation Plan

- Redesign `RegimeStateContract` generation so label-only outputs no longer auto-populate `probabilities={label: 1.0}` or `confidence=1.0`.
- Add explicit fields:
  - `confidence_kind` with values such as `posterior`, `calibrated_score`, `heuristic`, `unsupported`
  - `recognition_lag_bars`
  - `source_available_at`
  - `availability_reason`
- Require every detector to declare whether same-bar availability is valid. If not explicitly declared, shift `available_at` by the minimum causal recognition lag.
- Restrict router scoring to calibrated or posterior confidence kinds. Heuristic confidence may be reported but must not drive allocation.
- Add detector-level calibration support for confidence-bearing detectors, including reliability summaries stored in manifests.
- Make `build_regime_state_contracts(...)` preserve `confidence=None` when the upstream state frame lacks probability or calibrated confidence outputs.
- Add hard assertions in capital-facing modes that `available_at >= as_of` and same-bar recognition is justified by detector contract, not by default.

### Validation And Exit Criteria

- No path fabricates confidence from labels alone.
- Same-bar regime recognition occurs only when explicitly declared by the detector contract.
- Router behavior changes when confidence is unavailable instead of silently assuming certainty.
- Backtest alignment reports surface recognition lag and unavailable-state rates.

## 4. Replace Persistence-Only Regime Stability Gating With Incremental Predictive Evidence

### Audit Problem

The current regime stability gate can pass a contextual regime taxonomy simply because it
switches less often than an endogenous baseline. That rewards stickiness, not usefulness.

### Institutional Practice Synthesis

Research basis: NIST AI RMF, BIS FSI 24.

- Institutional model governance does not accept internal consistency as a substitute for outcome validity.
- Explainability and review requirements imply that a contextual regime must provide measurable incremental value over simpler alternatives.
- Stable narratives that do not improve decisions are still model risk.

### Technical Implementation Plan

- Replace the current `persistence improvement` gate with a multi-metric incremental-evidence gate.
- Add contextual-regime evaluation metrics that compare enriched regimes against an endogenous-only baseline on:
  - transition-aware predictive lift
  - locked-holdout incremental utility after costs
  - unseen-regime fallback rate
  - post-transition degradation
  - label entropy floor and minimum dwell distribution
  - boundary reproducibility under neighboring refits
- Demote pure persistence to a secondary diagnostic rather than a promotion control.
- Store full regime-gating evidence under a dedicated artifact block such as `regime_incremental_evidence`.
- Make promotion contingent on the enriched regime either:
  - improving decision quality after costs, or
  - reducing catastrophic fallback/degradation without hiding risk in no-trade behavior.
- Add a null benchmark that tests whether the contextual regime merely partitions past performance without improving out-of-sample routing or thresholding.

### Validation And Exit Criteria

- A stickier regime that adds no predictive or execution benefit fails promotion.
- Regime gate reports include both stability diagnostics and incremental outcome evidence.
- Contextual regimes cannot be promoted on persistence alone.

## 5. Make Cross-Venue And Cross-Asset Context Freshness Fail Closed

### Audit Problem

Cross-asset context can forward-fill indefinitely when no TTL is configured, and reference
overlays rely on backward alignment without a mandatory freshness policy. That permits stale
state to masquerade as valid regime context.

### Institutional Practice Synthesis

Research basis: BCBS 239, NIST AI RMF, NIST AI RMF FAQs.

- BCBS 239 emphasizes risk data that can be aggregated fully, quickly, and accurately under stress.
- Timeliness, completeness, and adaptability are core data-control expectations for institution-grade risk systems.
- NIST AI RMF extends those expectations across deployment and monitoring: trustworthy systems must remain reliable and resilient in operation, not only in development.

### Technical Implementation Plan

- Change `core/context.py` and `core/reference_data.py` defaults so cross-asset, futures, and reference overlays require an explicit TTL policy in capital-facing modes.
- Remove indefinite forward-fill as the default for cross-asset leader data. The default should be `preserve_missing` plus an unknown indicator unless a narrower policy is explicitly configured.
- Add source-timestamp, feature-age, and freshness-state columns to every context block.
- Introduce a unified `ContextFreshnessContract` for all non-local data sources:
  - source timestamp
  - availability timestamp
  - age at decision time
  - TTL status
  - fill method used
  - data-quality flags
- In capital-facing modes, stale contextual features must either:
  - be dropped and force fallback behavior, or
  - block signal generation if the feature family is required for the selected path.
- Extend scenario testing with stale-feed and partial-venue-lag events for futures context, reference overlays, and leader-basket symbols.
- Persist freshness diagnostics into research artifacts, promotion evidence, and deployment readiness reports.

### Validation And Exit Criteria

- No contextual feature can be used without a measurable freshness state.
- Capital-facing runs fail closed when required context is stale beyond policy.
- Stress tests include stale cross-venue and stale leader-asset scenarios.
- Promotion evidence includes stale-hit and unknown-hit rates for each context source.

## 6. Persist Drift State Across Operating Cycles

### Audit Problem

The drift monitor is recreated for each retraining cycle, so ADWIN and related evidence are
window-local rather than persistent. That makes drift decisions sensitive to arbitrary cycle
boundaries and can both miss structural breaks and trigger noisy retrains.

### Institutional Practice Synthesis

Research basis: NIST AI RMF, NIST AI RMF Playbook, BIS FSI 24.

- Institutional monitoring expects deployment-time behavior to be reviewed continuously, not as isolated snapshots.
- Advanced AI governance requires monitoring, validation, and independent review to track the same model through time.
- A drift process that resets hidden state at each review boundary is not operational monitoring; it is repeated re-sampling.

### Technical Implementation Plan

- Persist `DriftMonitor` state and detector internals alongside the champion artifact in the registry.
- Add a `DriftStateSnapshot` artifact containing:
  - detector family state
  - reference/current window lineage
  - last approved drift evidence
  - cooldown state
  - last recalibration and retrain timestamps
- Make `run_drift_retraining_cycle(...)` load prior detector state, update it with the new slice, and persist the updated state after evaluation.
- Distinguish three layers explicitly in stored state:
  - distribution baselines
  - streaming detector state
  - decision-policy state
- Add overlap-aware confirmation logic so structural retraining requires either:
  - persistent evidence across sequential windows, or
  - a critical-event override.
- Preserve separate state for score drift, action drift, regime drift, and performance drift so one family cannot implicitly overwrite another.
- Add deterministic replay tooling that can recompute a historical drift decision from persisted state and the exact input window.

### Validation And Exit Criteria

- Replaying the same chronological monitoring stream yields the same retrain decision regardless of review window batching.
- Detector state survives process restarts.
- Drift decisions can be reconstructed from persisted artifacts without hidden in-memory state.

## 7. Hard-Separate Global Preview From Admissible Evidence

### Audit Problem

The pipeline still supports a `global_preview_only` regime path that fits detectors on the same
window it later narrates. That is acceptable for inspection, but dangerous if preview artifacts
bleed into validation, backtest, or promotion evidence.

### Institutional Practice Synthesis

Research basis: NIST AI RMF, NIST AI RMF Playbook, BIS FSI 24.

- Lifecycle governance requires clear separation between development, validation, deployment, and monitoring artifacts.
- Explainability and review lose value when a retrospective research object is reused as if it were live evidence.
- Institution-grade review demands that non-causal preview outputs be clearly labeled, isolated, and barred from promotion decisions.

### Technical Implementation Plan

- Split regime outputs into two explicit namespaces:
  - `preview_regime_*` for full-sample or exploratory traces
  - `evidence_regime_*` for fold-local, holdout, or deployment-admissible traces
- Prevent `global_preview_only` artifacts from populating any state that can later feed:
  - backtest runtime kwargs
  - training summaries used for promotion
  - registry manifests used for capital-facing review
- Add a `evidence_class` field to every regime artifact with allowed values such as `preview_only`, `fold_local_oos`, `locked_holdout`, `live_monitoring`.
- In capital-facing modes, reject any run where a preview-only regime artifact is attached to backtest or promotion evidence.
- Update pipeline serialization and summary builders to preserve the distinction end to end.
- Add tests that deliberately attempt to reuse preview artifacts in validation and assert hard failure.

### Validation And Exit Criteria

- Preview artifacts are impossible to pass into capital-facing backtests and promotion gates.
- Research UI can still inspect preview traces, but summaries clearly mark them as non-evidentiary.
- Registry manifests store only admissible evidence classes for champion/challenger review.

## 8. Define A Real Consumer-Hardware And Reduced-Power Operating Envelope

### Audit Problem

The framework has a `reduced_power` notion, but no explicit CPU, RAM, storage, dataset-size,
or latency envelope proving the stack is safe and deterministic on consumer hardware. There is
no standards-based basis to claim retail suitability without that envelope.

### Institutional Practice Synthesis

Research basis: SEC Rule 15c3-5, NIST AI RMF, NIST AI RMF FAQs.

- SEC market-access controls assume the operator maintains direct control, pre-trade controls, and regular review of effectiveness.
- NIST AI RMF is lifecycle-wide and scalable, but it still requires systems to remain valid, reliable, safe, secure, resilient, and documented in deployment.
- A consumer-hardware claim is a deployment claim; it needs measured capacity, latency, failure-mode handling, and clear stage-gating rather than informal reassurance.

### Technical Implementation Plan

- Define explicit deployment profiles in config and readiness policy:
  - `research_workstation`
  - `consumer_laptop`
  - `mini_server`
  - `reduced_power_research_only`
- For each profile, declare hard budgets for:
  - peak RSS memory
  - model load latency
  - per-bar inference latency
  - drift-cycle latency
  - storage footprint
  - maximum supported symbol and timeframe set
- Instrument the pipeline and runtime with resource telemetry and persist it into readiness reports.
- Extend readiness gating so any profile breach blocks progression above `research_certified` unless a narrower validated profile is explicitly selected.
- Add a deterministic replay benchmark suite that runs representative workflows on the declared low-spec profile and records:
  - throughput
  - latency tail percentiles
  - memory spikes
  - restart recovery time
  - degraded-mode behavior
- Define failover behavior for low-resource conditions:
  - forced fallback to simpler detectors
  - reduced candidate set
  - disabled mixture routing
  - research-only downgrade
- Publish a supported-operating-envelope matrix in user-facing docs and release criteria.

### Validation And Exit Criteria

- Every capital-facing mode is tied to a validated hardware profile.
- `reduced_power` remains non-capital-eligible unless a dedicated validation program is added.
- Retail suitability claims are backed by reproducible replay benchmarks and readiness reports.

## Recommended Implementation Order

1. Bind router stability to executed exposure.
2. Remove synthetic confidence and enforce real availability timing.
3. Make context freshness fail closed.
4. Persist drift state across cycles.
5. Hard-separate preview from admissible evidence.
6. Make specialist execution honor health and eligibility.
7. Replace persistence-only regime gating.
8. Define the consumer-hardware operating envelope.

## Definition Of Done For The Program

- No capital-facing path depends on fabricated probabilities, same-bar regime availability by default, or preview-only artifacts.
- Router, specialist, and drift controls report the executed path and can be replayed deterministically.
- Contextual data sources are freshness-aware, point-in-time safe, and stress-tested.
- Promotion, stress, and readiness gates evaluate the same artifacts that live runtime uses.
- Consumer-hardware claims are tied to measured resource envelopes and enforced stage gates.