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
- Default runtime behavior has not been switched to specialist-library or router
  execution yet.
- The next required implementation phase is Phase 1.

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

Status: pending

Objective:

Move regime detection behind the new detector interfaces and make regime traces
causal, replayable, and explicit.

Implementation targets:

1. Port current default regime feature construction into
   `core/regimes/observations.py`.
2. Implement volatility, trend, liquidity, and break detectors.
3. Move HMM handling behind a filtered online interface.
4. Add detector ensemble and confidence logic.

Acceptance criteria:

1. Regime traces are reproducible under walk-forward replay.
2. No detector uses future data.
3. Detector disagreement and warm-up states are visible in outputs.

Immediate next slice:

1. Extract regime observation building into `core/regimes/observations.py`.
2. Add concrete detector classes and manifests in `core/regimes/detectors.py`.
3. Rewire the `detect_regimes` flow through the new runtime modules while
   keeping `core/regime.py` as the compatibility facade.
4. Add walk-forward replay tests for causal detector output.

### Phase 2: Feature Adaptation Layer

Status: pending

Objective:

Insert an explicit feature adaptation step between regime detection and model
training.

Implementation targets:

1. Implement regime-conditioned scaling and masks.
2. Log applied feature policy per fold and per backtest step.
3. Add deterministic fallback behavior under low confidence and sparse support.

Acceptance criteria:

1. Feature transforms are prefix-invariant at cutoffs.
2. Per-regime masks do not leak future occupancy.
3. Fallback behavior is explicit and tested.

### Phase 3: Specialist Library

Status: pending

Objective:

Replace the single-model assumption with a specialist library plus fallback
generalist.

Implementation targets:

1. Add specialist metadata and health tracking.
2. Persist historical regime performance by model.
3. Extend registry manifests to support multiple active specialists per symbol.
4. Implement the specialist lifecycle state machine.

Acceptance criteria:

1. One symbol can carry multiple certified specialists plus fallback.
2. Each specialist records compatible regimes and failure flags.
3. Rollback and retirement work without mutating history.

### Phase 4: Router Implementation

Status: pending

Objective:

Introduce explicit router policies and anti-flapping controls.

Implementation targets:

1. Implement hard-switch and weighted router policies.
2. Add hysteresis, persistence, cooldown, and score diagnostics.
3. Persist routing traces in backtests and live-like replay.

Acceptance criteria:

1. Router decisions are deterministic under replay.
2. Switching costs appear in backtest summaries.
3. Route reasons and blocked-switch reasons are visible.

### Phase 5: Validation and Backtest Overhaul

Status: pending

Objective:

Refactor evaluation to replay the full adaptive loop instead of relying on
pooled aggregate metrics.

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

Status: pending

Objective:

Make studies explicit about regime detectors, specialists, and routing.

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

Status: pending

Objective:

Refactor drift handling away from immediate retraining.

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

1. Phase 1: Online regime detection layer
2. Phase 2: Feature adaptation layer
3. Phase 3: Specialist library
4. Phase 4: Router implementation
5. Phase 5: Validation and backtest overhaul
6. Phase 6: AutoML and experiment redesign
7. Phase 7: Maintenance and governance

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