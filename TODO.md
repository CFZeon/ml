# TODO

## Purpose

This file tracks the remaining upgrade phases for the regime-orchestrated
framework redesign.

Source of truth for the detailed architecture remains `UPGRADES.md`. This file
is the execution-oriented checklist derived from that plan.

## Current Status

- Phase 0 is complete.
- Phase 0 delivered additive contracts, compatibility facades, optional
  registry payload support, and example-level Phase 0 summaries.
- Phase 1 is complete.
- Phase 2 is complete.
- Phase 2 Slice 1 delivered typed feature-adaptation contracts, an identity
   adapter seam in `TrainModelsStep`, additive training metadata, and focused
   runtime and pipeline tests.
- Phase 2 Slice 2 delivered a frozen global and per-regime scaling bank,
   explicit fail-closed guards for `model.regime_aware.strategy == "feature"`,
   and fallback-inference parity through the stored final adapter.
- Phase 2 Slice 3 delivered frozen per-regime masking, a composite
   `scale -> mask` adapter runtime, fold-local policy-disabled column gating,
   and fallback-inference parity for masking through the stored final adapter.
- Phase 2 Slice 4 delivered a dedicated feature-strategy adapter, bounded
   `interaction_budget` generation, and frozen adapter reuse in
   `RegimeAwareModelBundle` fit and inference paths.
- Phase 2 Slice 5 delivered feature-adaptation config validation, AutoML
   training-summary propagation, explicit post-selection refit surfacing, and
   user-facing training summary output in the experiment runner and example
   helpers.
- Phase 3 Slice 1 delivered a dedicated specialist-library runtime,
   candidate-state snapshot generation in specialist training outputs,
   registry-backed artifact ref enrichment, and additive lifecycle transition
   events plus active-library reads without mutating immutable manifests.
- Phase 5 is complete.
- Phase 6 Slice 1 delivered bundle-centric AutoML search-space support,
   bundle lineage propagation through study summaries and registry metadata,
   and trade-ready thesis-space freeze coverage for `orchestration.bundle`.
- Phase 6 Slice 2 delivered a config-driven orchestration-bundle AutoML
   example, user-facing runner summaries for selected bundle lineage and router
   stability, and focused config/runner regression coverage.
- Phase 6 Slice 3 delivered orchestration-first example-guide cleanup,
   explicit detector/router entrypoints in the docs, and a downstream
   maintenance banner on the drift review example.
- Phase 7 Slice 1 delivered an additive specialist-library review policy in
   the drift maintenance path, allowing degraded active libraries to return an
   explicit `library_review_recommended` outcome before retraining.
- Default runtime behavior has not been switched to specialist-library or router
  execution yet.
- The next required implementation slice is Phase 7 Slice 2.

## Phase Checklist

### Phase 0: Contracts and Compatibility Scaffolding

Status: completed

Delivered:

1. Added typed regime, specialist, and router contracts.
2. Kept `core/regime.py` and `core/regime_training.py` as public facades.
3. Added optional registry manifest support for Phase 0 payload sections.
4. Re-exported the new contracts from `core`.
5. Updated examples to print Phase 0 summaries where applicable.

Completion criteria met:

1. Contracts serialize to JSON-ready dicts and round-trip cleanly.
2. Registry manifests accept both legacy and Phase 0 payload shapes.
3. No router behavior or specialist-library behavior is enabled by default.

### Phase 1: Online Regime Detection Layer

Status: completed

Objective:

Move regime detection behind the new detector interfaces and make regime traces
causal, replayable, and explicit.

Delivered:

1. Canonical regime observation building in `core/regimes/observations.py`.
2. Canonical replay runtime in `core/regimes/online_state.py`.
3. Compatibility, native score-based, and filtered-HMM detectors in
   `core/regimes/detectors.py`.
4. Config normalization for native-primary and compatibility detector shapes.
5. Detector manifests, replay metadata, and trace summaries surfaced in
   `pipeline.state["regime_detection"]`.
6. Example propagation for native detector and filtered-HMM paths plus focused
   replay/runtime tests.

Implementation targets:

1. Port current default regime feature construction into
   `core/regimes/observations.py`.
2. Implement volatility, trend, liquidity, and break detectors.
3. Move HMM handling behind a filtered online interface.
4. Add detector ensemble and confidence logic.

Acceptance criteria met:

1. Regime traces are reproducible under walk-forward replay.
2. No detector uses future data.
3. Detector disagreement and warm-up states are visible in outputs.

### Phase 2: Feature Adaptation Layer

Status: completed

Objective:

Insert an explicit feature adaptation step between regime detection and model
training.

Delivered so far:

1. Added `core/feature_adaptation/` with `FeaturePolicyContract`,
   `BaseFeatureAdapter`, and the Slice 1 identity runtime.
2. Routed fold-local fit/validation/test frames through the feature-adaptation
   seam in `TrainModelsStep` without changing model behavior.
3. Added additive JSON-safe `training["feature_adaptation"]` and
   `pipeline.state["feature_adaptation"]` payloads.
4. Added focused coverage in `tests/test_feature_adaptation_runtime.py` and
   extended `tests/test_automl_regime_aware_training.py`.

Implementation targets:

1. Implement regime-conditioned scaling and masks.
2. Log applied feature policy per fold and per backtest step.
3. Add deterministic fallback behavior under low confidence and sparse support.

Acceptance criteria:

1. Feature transforms are prefix-invariant at cutoffs.
2. Per-regime masks do not leak future occupancy.
3. Fallback behavior is explicit and tested.

Immediate next slice:

1. Start Phase 5 by replaying detector and router state causally through the
   validation and holdout path instead of relying on pooled backtest summaries.
2. Reuse the canonical router replay diagnostics from Phase 4 rather than
   reimplementing routing logic inside validation-specific code.

### Phase 3: Specialist Library

Status: completed 2026-05-09

Objective:

Replace the single-model assumption with a specialist library plus fallback
generalist.

Delivered so far:

1. Added `core/specialists/library.py` with specialist snapshot normalization,
   selection-contract construction, registry-status projection, artifact-ref
   attachment, and fail-closed lifecycle transition validation.
2. Updated specialist training and post-selection refit outputs to surface
   `specialist_library` in `training`, pipeline state, and refit artifacts.
3. Extended `LocalRegistryStore` to auto-persist specialist libraries from
   specialist training summaries, enrich them with registry artifact refs, and
   expose `read_specialist_library(...)` / `get_active_specialist_library(...)`.
4. Added additive `record_specialist_lifecycle_transition(...)` event storage
   so runtime lifecycle state can evolve without rewriting immutable version
   manifests.
5. Added `core/specialists/health.py` plus append-only
   `attach_specialist_health_update(...)` registry storage so specialist health
   and regime performance history replay into runtime library reads without
   mutating immutable manifests.
6. Added `core/specialists/governance.py` with explicit certification and
   degradation policy helpers, advisory/blocking gate support, and in-memory
   lifecycle application over specialist snapshots.
7. Tightened the Phase 3 lifecycle runtime so registry status is tracked as
   library metadata rather than overwriting per-specialist states, and champion
   promotion now auto-activates only the fallback generalist plus already
   certified specialists through append-only lifecycle events.

Acceptance criteria:

1. One symbol can carry multiple certified specialists plus fallback.
2. Each specialist records compatible regimes and failure flags.
3. Rollback and retirement work without mutating history.

### Phase 4: Router Implementation

Status: completed 2026-05-09

Objective:

Introduce explicit router policies and anti-flapping controls.

Implementation targets:

1. Implement hard-switch and weighted router policies.
2. Add hysteresis, persistence, cooldown, and score diagnostics.
3. Persist routing traces in backtests and live-like replay.

Delivered so far:

1. Added `core/routing/router.py` with deterministic `HardSwitchRouter` and
   `WeightedRouter` implementations plus `build_router(...)` factory support.
2. Added `core/routing/diagnostics.py` with deterministic
   `replay_router_trace(...)` summaries over regime-state streams.
3. Added focused regressions for hard-switch anti-flapping behavior, weighted
   allocation normalization, factory/config routing, and replay determinism.
4. Threaded optional router trace summaries through `run_backtest(...)` so
   switch counts, route reasons, blocked-switch reasons, and decision traces
   are exposed in user-facing backtest summaries without changing execution
   math.
5. Added optional router switching-cost reporting to backtest summaries as an
   explicit hypothetical routing-overhead estimate rather than silently
   treating it as realized execution cost.

Acceptance criteria:

1. Router decisions are deterministic under replay.
2. Switching costs appear in backtest summaries.
3. Route reasons and blocked-switch reasons are visible.

### Phase 5: Validation and Backtest Overhaul

Status: completed 2026-05-09

Objective:

Refactor evaluation to replay the full adaptive loop instead of relying on
pooled aggregate metrics.

Delivered so far:

1. Extended `run_backtest(...)` with additive `regime_segment_report` and
   `transition_segment_report` payloads driven by aligned `regime_states`,
   without changing execution math or requiring a router runtime.
2. Updated CPCV diagnostic path replay so fold-local regime states are carried
   into per-path backtests and surfaced in
   `diagnostic_validation["summary"]` as path-level regime and transition
   diagnostics instead of pooled aggregate metrics alone.
3. Hardened fold-local regime joins in `core/pipeline.py` so sparse regime
   metadata flags do not invalidate otherwise usable CPCV fit/test rows.
4. Added delayed-recognition and transition-lag diagnostics to
   `transition_segment_report`, reusing `signal_delay_bars` as the simulated
   recognition window and surfacing the aggregated fields through CPCV
   `diagnostic_validation["summary"]`.
5. Added fold-local unseen-regime degradation reporting, including fallback
   exposure shares, affected-fold summaries, and additive backtest surfacing
   via `unseen_regime_degradation_report`.
6. Added router stability and over-switching diagnostics plus a promotion gate,
   using replay-only router trace metrics and switching-cost estimates without
   changing routing decisions or executable PnL.

Implementation targets:

1. Add regime-segmented and transition-segmented metrics.
2. Add delayed recognition simulation.
3. Add unseen-regime degradation reports.
4. Add router stability and over-switching gates.

Acceptance criteria:

1. Validation summaries no longer rely on pooled aggregate metrics alone.
2. The holdout path replays detector plus router causally.
3. Unseen regime behavior is observable and gated.

### Phase 6: AutoML and Experiment Redesign

Status: completed 2026-05-11

Objective:

Make studies explicit about regime detectors, specialists, and routing.

Delivered so far:

1. Added `orchestration.bundle` as a first-class AutoML search-space path in
   `core/automl.py`, allowing studies to compare detector, specialist, and
   router bundles through the existing override contract instead of only leaf
   model hyperparameters.
2. Preserved `best_bundle_lineage` and per-trial bundle lineage through study
   summaries, best-trial diagnostics, and registry lineage/metadata outputs.
3. Hardened bundle sampling so leaf-level search overrides deep-merge into the
   sampled bundle rather than overwriting detector or router lineage.
4. Added focused regression coverage proving trade-ready profiles reject
   `orchestration.bundle` as thesis variation and that AutoML summaries surface
   bundle lineage and normalized bundle overrides.
5. Added `configs/btc_regime_bundle_automl.yaml` and
   `example_regime_bundle_automl.py` so users can run bundle-centric AutoML
   from the shared config-driven workflow instead of constructing nested
   overrides by hand.
6. Updated the config-driven experiment runner to print selected bundle lineage
   and router-stability diagnostics when AutoML chooses an orchestration
   bundle.
7. Added focused config-loader and runner-output regressions for the new
   orchestration-bundle user path.
8. Validated the new config through `run.py --config ... --quick --validate-only`.
9. Reordered the user-facing docs so `example_regime_orchestration.py` and
   `example_regime_bundle_automl.py` are surfaced as primary orchestration
   entrypoints ahead of maintenance flows.
10. Reframed `example_drift_retraining_cycle.py` as a downstream maintenance
    and drift-review example in the docs and in the script banner itself.
11. Validated the updated maintenance example in quick mode after the entrypoint
    banner change.

Implementation targets:

1. Redesign the AutoML search space around detector, specialist, and router
   bundles.
2. Add new configs and example entrypoints.
3. De-emphasize retraining-focused examples.

Acceptance criteria:

1. AutoML can compare regime bundles, not only model hyperparameters.
2. Experiment manifests capture detector and router lineage.
3. Examples produce routing and regime diagnostics by default.

### Phase 7: Maintenance and Governance

Status: completed 2026-05-11

Objective:

Refactor drift handling away from immediate retraining.

Delivered so far:

1. Added an additive `library_review` policy report to the drift maintenance
   flow, using the active champion specialist library plus existing specialist
   governance rules to recommend explicit review before retraining.
2. Surfaced `library_review_recommended` as a first-class drift-cycle outcome
   when the active specialist library is degraded but drift evidence does not
   justify challenger training.
3. Added explicit specialist retirement rules plus certified shadow
   replacement recommendations in `core/specialists/governance.py`, so
   terminally unhealthy specialists can move to `retired` and governed backups
   can move to `shadow_challenger` through existing lifecycle transitions.
4. Added an additive `router_recalibration` maintenance path in
   `core/orchestration.py`, reusing the existing router-stability gate to
   recommend router recalibration before challenger training when switching
   behavior is unstable.
5. Added an additive `structural_invalidation` policy split plus top-level
   `action_report`, so approved drift evidence now resolves to explicit
   `reroute`, `recalibrate`, `discover`, or `retrain` actions instead of
   treating all approved drift as immediate retraining.
6. Reserved challenger training for structural invalidation cases such as
   model TTL expiry, performance drift, or joint feature/prediction shift,
   while non-structural regime changes now surface `discovery_recommended`
   without opening a retraining window.
7. Tightened empty runtime specialist-library handling so missing specialist
   libraries do not produce false-positive library-review recommendations.
8. Added focused governance and drift workflow regressions covering
   retirement-plus-replacement, router recalibration, non-structural discovery,
   and preserved TTL-driven retraining.
9. Validated the complete Phase 7 slice with focused maintenance and
   governance suites.

Implementation targets:

1. Add a library review policy.
2. Add specialist retirement and replacement rules.
3. Add a router recalibration path.
4. Reserve retraining for structural invalidation cases.

Acceptance criteria:

1. Drift action reports distinguish reroute, recalibrate, discover, and
   retrain.
2. Retraining is no longer the default response to short-term degradation.

## Required Order

Implement the remaining phases in this sequence:

1. Phase 2: Feature adaptation layer
2. Phase 3: Specialist library
3. Phase 4: Router implementation
4. Phase 5: Validation and backtest overhaul
5. Phase 6: AutoML and experiment redesign
6. Phase 7: Maintenance and governance

## Definition Of Done

The upgrade backlog is complete only when:

1. Regime detection is causal, online, replayable, and explicit.
2. Feature adaptation is a first-class runtime layer with deterministic
   fallback policies.
3. Each symbol can run a specialist library plus fallback generalist.
4. Router decisions are explicit, deterministic, and backtest-replayable.
5. Validation and holdout paths replay detector, adaptation, and routing
   causally.
6. AutoML compares full regime bundles rather than only model hyperparameters.
7. Drift handling prefers reroute, recalibrate, discover, and retire before
   retrain.