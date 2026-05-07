# Regime-Orchestrated AutoML Framework Upgrade Plan

## Purpose

This plan redesigns the framework from a retraining-centric system into a regime-orchestrated research and execution stack.

The target state is not "retrain one model faster." The target state is:

- detect market regimes causally and online
- maintain a library of specialist models
- route capital between specialists using an explicit router
- retrain only when regime structure or model coverage genuinely breaks

This plan is anchored to the current codebase rather than a greenfield rewrite. The existing repository already has useful starting points:

- `core/regime.py` builds causal regime features and already supports explicit and HMM-style regime detection
- `core/regime_training.py` already exposes regime-aware training helpers
- `core/pipeline.py` already has a regime-aware training switch under `model.regime_aware`
- `core/automl.py` already searches regime parameters and applies promotion gates
- `core/drift.py`, `core/monitoring.py`, `core/backtest.py`, and `core/registry/` already provide monitoring, simulation, and artifact primitives

What is missing is the architectural shift that treats regime state, model specialization, and routing as first-class research objects.

## Problem Statement

The current framework still centers the lifecycle around a primary model that is retrained when metrics drift. That is insufficient for crypto markets because:

- the data-generating process is non-stationary and regime-dependent
- one model can be locally strong and globally fragile
- aggregated validation can hide regime-specific failure
- frequent retraining is often reactive noise-fitting rather than genuine adaptation
- regime shifts and structural breaks require routing, not only recalibration

The framework must therefore move from one evolving model to a persistent specialist library with explicit causal regime inference and explicit routing.

## Non-Negotiable Invariants

These constraints must remain binding through the refactor.

### Causality and leakage control

- No full-sample clustering over the complete dataset.
- No future-aware regime labeling.
- No HMM smoothing for live or backtest decisions. Only filtered state probabilities are allowed at decision time.
- Any clustering-based detector must fit on the training prefix only and infer online on later data.
- Any detector that requires state prototypes or thresholds must freeze them per fold before validation and holdout.
- Feature adaptation must use only the regime state available at the same timestamp.
- Router performance priors must update only after outcome maturity and execution delay.
- Validation, holdout, and promotion must preserve the existing purging and embargo discipline.

### Temporal correctness

- Regime state must be stamped with `as_of` and `available_at` timestamps.
- Regime probabilities, feature transforms, and routing decisions must all respect publication delays and availability lags.
- Backtests must simulate delayed regime recognition, switching latency, and switching costs.

### Statistical defensibility

- Aggregate metrics are insufficient for admissibility.
- Performance must be reported by regime, by transition, and on unseen or weakly covered regimes.
- Drift monitors must distinguish temporary turbulence from persistent structural change.
- Retraining must be a governed maintenance decision, not the default reaction to short-horizon degradation.

### Product constraints that remain fixed

- One deployed model library per symbol.
- Shared timeframe research across symbols remains valid.
- Spot and futures stay separated where venue assumptions differ.
- Custom data joins remain point-in-time safe.
- The backtest and execution engine boundary remains adapter-based.
- All workflows remain config-driven and reproducible.

## Target System Architecture

The framework should be refactored around the following runtime flow:

```text
Market Data
  -> Regime Observation Builder
  -> Regime Detection Layer
  -> Regime-Aware Feature Adaptation
  -> Specialist Model Library
  -> Router / Selection Engine
  -> Execution and Backtesting
  -> Evaluation, Diagnostics, and Monitoring
```

The architecture also needs three distinct control loops:

1. Fast loop: per-bar or per-event regime inference and routing.
2. Medium loop: delayed performance attribution, router recalibration, and specialist health updates.
3. Slow loop: specialist discovery, retirement, replacement, and controlled retraining.

This separation is critical. The current design implicitly collapses all three loops into retraining. The upgraded system must not do that.

## Current-to-Target Gap Summary

The current repo has regime features and regime-aware training hooks, but it does not yet have:

- a first-class online regime state contract
- a detector ensemble abstraction
- regime-conditioned feature policies as a separate layer
- a specialist model library with compatibility metadata
- a router with hysteresis, persistence logic, and cooldowns
- regime-transition-aware backtest replay and diagnostics
- a maintenance policy that prefers routing and specialist replacement over constant retraining

The plan below closes those gaps in a staged way.

## Refactored Module Structure

The recommended target structure is:

```text
core/
  regimes/
    __init__.py
    contracts.py
    observations.py
    detectors.py
    ensemble.py
    online_state.py
    transitions.py
    breaks.py
    diagnostics.py
    validation.py
  feature_adaptation/
    __init__.py
    contracts.py
    scaling.py
    gating.py
    selection.py
    diagnostics.py
  specialists/
    __init__.py
    contracts.py
    training.py
    calibration.py
    library.py
    health.py
    retirement.py
  routing/
    __init__.py
    contracts.py
    scorer.py
    hysteresis.py
    policies.py
    router.py
    diagnostics.py
  validation/
    __init__.py
    regime_walk_forward.py
    transition_metrics.py
    unseen_regime.py
    regime_reports.py
  backtest_regime_trace.py
  backtest_routing_trace.py
  backtest_switching_costs.py
  orchestration_regime_library.py
  orchestration_router_maintenance.py
```

Existing files should be repurposed rather than abandoned:

- `core/regime.py` becomes a compatibility facade and high-level entrypoint over `core/regimes/`
- `core/regime_training.py` becomes a compatibility facade over `core/specialists/training.py`
- `core/drift.py` is split into regime drift, feature drift, model decay, and structural break monitors
- `core/pipeline.py` becomes the canonical orchestration point for new regime and router steps
- `core/automl.py` becomes a regime-centric study runner rather than a retraining-centric tuner
- `core/registry/` is extended to store specialist-library manifests and router artifacts

## Core Contracts and Interfaces

The refactor should begin by introducing explicit contracts. Without them, the redesign will sprawl into implicit state and ad hoc dictionaries.

### Regime contracts

```python
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import pandas as pd

@dataclass(frozen=True)
class RegimeObservation:
    as_of: pd.Timestamp
    available_at: pd.Timestamp
    values: Mapping[str, float]
    source_map: Mapping[str, str]
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class RegimeState:
    as_of: pd.Timestamp
    available_at: pd.Timestamp
    label: str
    probabilities: Mapping[str, float]
    confidence: float
    detector_outputs: Mapping[str, Any]
    warm: bool
    transition_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

class BaseRegimeDetector(Protocol):
    def fit(self, observations: pd.DataFrame) -> "BaseRegimeDetector": ...
    def initialize(self, observations: pd.DataFrame) -> Any: ...
    def update(self, state: Any, observation: RegimeObservation) -> tuple[Any, RegimeState]: ...
    def min_history(self) -> int: ...
```

Detector rules:

- `fit(...)` may only consume the training prefix of a fold.
- `update(...)` must emit the next decision-eligible state using only past and current information.
- any latent-state model must emit filtered probabilities, never smoothed probabilities.

### Feature adaptation contracts

```python
@dataclass(frozen=True)
class FeaturePolicy:
    feature_columns: Sequence[str]
    disabled_columns: Sequence[str]
    scaling_policy: str
    selector_version: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

class BaseFeatureAdapter(Protocol):
    def fit(self, X: pd.DataFrame, regime_history: pd.DataFrame) -> "BaseFeatureAdapter": ...
    def transform(self, X: pd.DataFrame, regime_state: RegimeState) -> tuple[pd.DataFrame, FeaturePolicy]: ...
```

Feature adaptation rules:

- scalers and selectors must be fit on the training slice only
- regime-conditioned transforms must fall back deterministically when regime confidence is low or regime-specific samples are too sparse
- the applied policy must be logged for every prediction step in backtest and live paths

### Specialist model contracts

```python
@dataclass(frozen=True)
class SpecialistSpec:
    model_id: str
    symbol: str
    timeframe: str
    compatible_regimes: Sequence[str]
    estimator_family: str
    feature_policy_id: str
    calibration_id: str | None
    training_window: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class SpecialistHealth:
    model_id: str
    compatible_regimes: Sequence[str]
    last_calibrated_at: pd.Timestamp | None
    stability_score: float
    decay_score: float
    failure_flags: Sequence[str]
    fallback_only: bool = False
```

The specialist library must support:

- multiple specialists per symbol
- explicit regime compatibility mappings
- performance history by regime and by transition
- calibration lineage
- champion and challenger status per specialist family
- clean retirement and rollback

### Router contracts

```python
@dataclass(frozen=True)
class RoutingDecision:
    as_of: pd.Timestamp
    available_at: pd.Timestamp
    selected_model_id: str | None
    weights: Mapping[str, float]
    regime_label: str
    regime_confidence: float
    route_reason: str
    hysteresis_applied: bool
    cooldown_active: bool
    candidate_scores: Mapping[str, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)

class BaseRouter(Protocol):
    def initialize(self, specialists: Sequence[SpecialistSpec]) -> Any: ...
    def select(
        self,
        state: Any,
        regime_state: RegimeState,
        specialist_health: Sequence[SpecialistHealth],
        timestamp: pd.Timestamp,
    ) -> tuple[Any, RoutingDecision]: ...
```

The router must be explicit, deterministic, serializable, and backtest-replayable.

## Regime Detection Subsystem

### Objectives

The regime subsystem must provide:

- discrete labels
- soft probabilities
- confidence scores
- transition events
- detector provenance

### Supported detector families

The initial implementation should support these causal detector families behind one interface:

1. Volatility state detector.
2. Trend versus mean-reversion detector.
3. Online change-point detector.
4. Hidden Markov detector with filtered posterior only.
5. Microstructure and liquidity state detector.
6. Ensemble regime combiner.

### Detector design details

#### Volatility and trend detectors

- Reuse and expand the causal features already built in `core/regime.py`.
- Use rolling and exponentially weighted statistics only.
- Emit interpretable state probabilities such as `low_vol`, `normal_vol`, `high_vol`, `trend_up`, `trend_down`, `mean_reverting`.

#### Change-point detection

- Add online CUSUM and optionally Bayesian online change-point detection.
- Structural break signals should be emitted separately from standard regimes.
- Break detectors should not directly force retraining. They should first influence routing confidence and health penalties.

#### HMM detectors

- Fit only on train folds.
- Infer forward using filtered state probabilities.
- Persist transition matrix, emission parameters, warm-up length, and online state.
- Explicitly forbid smoothed posterior usage in backtest and live paths.

#### Clustering-style detectors

- If clustering is used, support only prefix-fit centroids or prototypes.
- No fitting on the full sample.
- No recomputing centroids using future validation or holdout data.

#### Detector ensemble

- Support majority vote, weighted vote, and learned but fixed-weight fusion.
- Ensemble outputs must emit detector-level evidence for diagnostics.
- Disagreement between detectors should reduce confidence, not silently collapse into a label.

### Outputs to persist

Each timestep should persist:

- `regime_label`
- `regime_probabilities`
- `regime_confidence`
- `detector_outputs`
- `transition_flag`
- `break_score`
- `available_at`

## Regime-Aware Feature Layer

This layer separates regime inference from model training. It must sit between `build_feature_set(...)` and model prediction.

### Responsibilities

- volatility normalization
- regime-conditioned scaling
- dynamic feature masking and disabling
- regime-specific feature subsets
- compatibility checks between regime state and specialist requirements

### Required feature policies

1. Global baseline transform.
2. Regime-conditioned scaler bank.
3. Regime-specific feature masks.
4. Confidence-aware fallback policy when regime posteriors are diffuse.
5. Sparse-regime fallback policy when a detector has insufficient support.

### Strict causal rules

- feature transforms may use the current regime state only if that state is itself decision-available
- regime-conditioned selectors must be trained on train-prefix samples only
- scaling must never borrow future regime occupancy statistics
- the applied transform policy must be included in the prediction artifact

## Specialist Model Architecture

### Design goal

Replace the single evolving model with a library of specialists that each have explicit scope and failure boundaries.

### Specialist categories to support initially

- trend-following specialist
- mean-reversion specialist
- breakout specialist
- low-volatility carry or drift specialist
- fallback generalist

The fallback generalist is required. The framework must never assume perfect regime coverage.

### Specialist metadata requirements

Each specialist artifact must track:

- `model_id`
- `symbol`
- `timeframe`
- `compatible_regimes`
- `incompatible_regimes`
- `training_window`
- `validation_window`
- `feature_schema_version`
- `feature_policy_id`
- `detector_bundle_id`
- `router_compatibility_version`
- `historical_regime_performance`
- `transition_performance`
- `stability_metrics`
- `failure_conditions`
- `last_valid_calibration_period`
- `decay_report`
- `artifact_location`

### Specialist lifecycle

Each specialist should move through these states:

1. Candidate.
2. Certified specialist.
3. Active.
4. Shadow challenger.
5. Degraded.
6. Retired.

Transitions between states must be driven by explicit evidence, not overwritten in place.

### Training policy

- Specialist discovery is performed offline on train folds.
- Specialists may train on regime-filtered samples or regime-conditioned features.
- Every specialist must produce a compatibility report and minimum-support report.
- When a regime slice is too small, the system must fail closed or fall back to a generalist rather than hallucinate coverage.

## Router and Selection Engine

The router becomes a core decision system, not a helper.

### Router modes

Support at least these policies:

1. Hard switch.
2. Confidence-weighted blend.
3. Regime posterior weighted blend.
4. Robustness-prioritized switch.

### Router score components

The router score for a candidate specialist should combine:

- regime compatibility score
- active regime posterior mass
- specialist historical performance in the same regime
- transition robustness when entering the regime
- stability penalty
- decay penalty
- stale calibration penalty
- switching cost penalty
- low-confidence penalty

The exact scoring formula can vary by router policy, but all terms must be explicit and logged.

### Anti-flapping controls

The router must implement:

- persistence requirements before switching
- hysteresis margins between incumbent and challenger
- cooldown windows after a switch
- optional minimum confidence before leaving fallback mode

### Router state

The router needs explicit online state storage containing:

- current active model
- time since last switch
- pending challenger streak
- most recent matured performance updates
- current cooldown state

No hidden module-level state is allowed.

### Routing diagnostics

Each routing decision must expose:

- selected model
- candidate scores
- regime input and confidence
- whether hysteresis blocked a switch
- whether cooldown blocked a switch
- whether the fallback generalist was used
- the exact route reason

## Regime-Aware Validation Redesign

Validation must stop optimizing or judging models on pooled averages alone.

### Required validation geometry

- fold-local regime detector fitting
- fold-local regime-conditioned feature adaptation
- fold-local specialist training
- out-of-sample router replay on the next slice
- purging and embargo when labels overlap
- locked holdout that also replays regime detection and routing causally

### Required metrics

Every study summary must report:

- Sharpe by regime
- drawdown by regime
- hit rate by regime
- turnover by regime
- switching frequency
- switching cost burden
- transition-period performance
- unseen-regime degradation
- fallback usage rate
- regime-confidence calibration
- model-stability dispersion by regime

### Overfitting checks

Add explicit gates for:

- regime coverage concentration
- per-regime performance collapse behind a strong aggregate score
- transition fragility
- unseen-regime failure
- router over-switching
- detector instability across adjacent folds

## Backtesting Refactor

The backtester must simulate the actual adaptive system, not a simplified labeling overlay.

### Required backtest events

For each timestep, the backtest should replay:

1. observation availability
2. regime state update
3. feature adaptation
4. routing decision
5. signal generation from the active specialist or blend
6. execution with switching cost and latency
7. delayed attribution once outcomes mature

### Backtest outputs to add

- active model timeline
- routing decision timeline
- regime timeline
- regime confidence timeline
- switch count and switch reasons
- switching cost PnL impact
- fallback generalist usage
- detection lag metrics
- delayed recognition versus realized regime transition metrics

### Realism requirements

- regime detection lag must be configurable and reflected in decision availability
- switching may incur idle bars, spread penalties, or warm-up penalties when appropriate
- no decision may use a regime state that was not available at the time

## Drift, Regime Change, and Structural Break Monitoring

The monitoring layer must explicitly separate different failure modes.

### Distinct monitors to implement

1. Regime drift: occupancy change, transition matrix drift, posterior confidence erosion.
2. Feature drift: PSI, KS, and schema drift within regime slices.
3. Model decay: per-regime and transition-conditioned deterioration.
4. Structural breaks: persistent change-point or break-score elevation.

### Action policy

The system should not jump from any drift signal directly to full retraining.

Action precedence should be:

1. reduce router confidence or increase fallback usage
2. deactivate a degraded specialist
3. recalibrate router priors or model calibration layer
4. trigger specialist discovery for uncovered regimes
5. retrain only if regime taxonomy or feature-model compatibility is no longer valid

### Explicit retraining triggers

Retraining should require at least one of these conditions:

- a persistent new regime appears that has no specialist support
- all compatible specialists for an important regime are degraded
- structural break evidence persists past cooldown and minimum-sample thresholds
- feature schema or exchange structure changes invalidate current specialists
- calibration validity expires across the active library

## AutoML and Experimentation Changes

AutoML must become regime-centric.

### Search focus shift

Instead of mainly searching frequent retraining-friendly knobs, studies should search:

- detector bundle choice
- detector hyperparameters constrained to causal variants
- feature adaptation policy
- specialist family set
- specialist-regime compatibility templates
- router policy and hysteresis parameters
- calibration policy

The search space must remain constrained. Unbounded combinatorics across detectors, specialists, and routing policies will destroy interpretability and statistical power.

### New config structure

The target configuration shape should look like this:

```yaml
regime:
  detectors:
    - name: vol_trend
      type: volatility_trend_hybrid
      warmup_bars: 240
      params:
        vol_window: 48
        trend_window: 96
    - name: liquidity
      type: liquidity_state
      warmup_bars: 120
      params:
        window: 48
  ensemble:
    type: weighted_vote
    confidence_floor: 0.55
    disagreement_penalty: 0.25

feature_adaptation:
  scaling:
    mode: regime_conditioned
    fallback: global
  selection:
    mode: per_regime_mask
    min_regime_samples: 128
  disable_incompatible_features: true

model_library:
  fallback_model: global_generalist
  specialists:
    - model_id: trend_model
      estimator: gbm
      compatible_regimes: [trend_up_low_vol, trend_up_high_vol]
    - model_id: mean_reversion_model
      estimator: logistic
      compatible_regimes: [range_low_vol, range_normal_vol]
    - model_id: breakout_model
      estimator: rf
      compatible_regimes: [transition_high_vol, panic_high_vol]

router:
  type: confidence_weighted
  hysteresis_margin: 0.08
  min_persistence_bars: 4
  cooldown_bars: 8
  fallback_confidence_floor: 0.45

maintenance:
  retraining_policy: structural_only
  min_new_regime_support: 256
  break_persistence_bars: 24
  library_review_cooldown_bars: 168
```

### Example experiments to add

- `configs/btc_regime_router.yaml`
- `configs/btc_regime_specialists.yaml`
- `configs/btc_futures_regime_router.yaml`
- `example_regime_orchestration.py`
- `example_router_diagnostics.py`
- `example_regime_transition_backtest.py`

## Diagnostics and Explainability

The upgraded system must expose its adaptive logic explicitly.

### Runtime diagnostics

Expose at each timestep:

- current regime label
- current regime probabilities
- regime confidence
- active specialist model
- routing weights
- routing reason
- cooldown status
- transition flags
- break and drift scores

### Study-level diagnostics

Each experiment summary should include:

- regime occupancy timeline
- regime transition matrix
- model activation timeline
- switch reason counts
- per-regime specialist leaderboard
- specialist failure analysis by regime
- unseen-regime fallback analysis
- detector disagreement summary
- calibration decay summary

### Storage requirements

Persist these as first-class artifacts beside the usual summary outputs:

- regime trace frame
- routing trace frame
- specialist health report
- detector bundle manifest
- router manifest
- regime-segmented metrics report

## Migration Strategy

The migration must be staged. A full flag-day rewrite would create too much breakage across examples, tests, and operator flows.

### Phase 0: Contracts and compatibility scaffolding

Add new contracts and facades without breaking the current public API.

- add `core/regimes/contracts.py`, `core/specialists/contracts.py`, `core/routing/contracts.py`
- keep `core/regime.py` and `core/regime_training.py` as forwarding facades
- add compatibility serialization in `core/registry/`
- add trace dataclasses without changing current examples yet

Acceptance criteria:

- no existing example breaks
- current regime-aware training still runs
- new contracts serialize cleanly

### Phase 1: Online regime detection layer

Implement the new detector interfaces and move current regime builders behind them.

- port current default regime feature construction into `core/regimes/observations.py`
- implement volatility, trend, liquidity, and break detectors
- move HMM handling behind a filtered online interface
- add detector ensemble and confidence logic

Acceptance criteria:

- regime traces are reproducible under walk-forward replay
- no detector uses future data
- detector disagreement and warm-up states are visible in outputs

### Phase 2: Feature adaptation layer

Insert an explicit feature adaptation step between regime detection and model training.

- implement regime-conditioned scaling and masks
- log applied feature policy per fold and per backtest step
- add deterministic fallback behavior under low confidence and sparse support

Acceptance criteria:

- feature transforms are prefix-invariant at cutoffs
- per-regime masks do not leak future occupancy
- fallback behavior is explicit and tested

### Phase 3: Specialist library

Replace the single-model assumption with a specialist library and fallback generalist.

- add specialist metadata and health tracking
- persist historical regime performance by model
- extend registry manifests to support multiple active specialists per symbol
- implement specialist lifecycle state machine

Acceptance criteria:

- one symbol can carry multiple certified specialists plus fallback
- each specialist records compatible regimes and failure flags
- rollback and retirement work without mutating history

### Phase 4: Router implementation

Introduce router policies and anti-flapping controls.

- implement hard-switch and weighted router policies
- add hysteresis, persistence, cooldown, and score diagnostics
- persist routing traces in backtests and live-like replay

Acceptance criteria:

- router decisions are deterministic under replay
- switching costs appear in backtest summaries
- route reasons and blocked-switch reasons are visible

### Phase 5: Validation and backtest overhaul

Refactor evaluation to replay the full adaptive loop.

- add regime-segmented and transition-segmented metrics
- add delayed recognition simulation
- add unseen-regime degradation reports
- add router stability and over-switching gates

Acceptance criteria:

- validation summaries no longer rely on pooled aggregate metrics alone
- the holdout path replays detector plus router causally
- unseen regime behavior is observable and gated

### Phase 6: AutoML and experiment redesign

Make studies explicit about regime detectors, specialists, and routing.

- redesign AutoML search space around detector, specialist, and router bundles
- add new configs and example entrypoints
- de-emphasize retraining-focused examples

Acceptance criteria:

- AutoML can compare regime bundles, not only model hyperparameters
- experiment manifests capture detector and router lineage
- examples produce routing and regime diagnostics by default

### Phase 7: Maintenance and governance

Refactor drift handling away from immediate retraining.

- add library review policy
- add specialist retirement and replacement rules
- add router recalibration path
- reserve retraining for structural invalidation cases

Acceptance criteria:

- drift action reports distinguish reroute, recalibrate, discover, and retrain
- retraining is no longer the default response to short-term degradation

## File-Level Refactor Map

The first code pass after this document should target these files.

### Files to introduce

- `core/regimes/contracts.py`
- `core/regimes/observations.py`
- `core/regimes/detectors.py`
- `core/regimes/ensemble.py`
- `core/regimes/online_state.py`
- `core/routing/contracts.py`
- `core/routing/router.py`
- `core/routing/hysteresis.py`
- `core/specialists/contracts.py`
- `core/specialists/library.py`
- `core/specialists/health.py`
- `core/validation/regime_walk_forward.py`
- `core/validation/transition_metrics.py`
- `core/backtest_regime_trace.py`
- `core/backtest_routing_trace.py`
- `core/backtest_switching_costs.py`
- `core/orchestration_regime_library.py`
- `core/orchestration_router_maintenance.py`

### Files to refactor heavily

- `core/pipeline.py`
- `core/automl.py`
- `core/backtest.py`
- `core/drift.py`
- `core/monitoring.py`
- `core/registry/` manifests and promotion surfaces
- `example_automl.py`
- `example_drift_retraining_cycle.py`
- `configs/btc_regime_aware.yaml`

### Files to keep as compatibility facades

- `core/regime.py`
- `core/regime_training.py`

## Pipeline Changes Required in `core/pipeline.py`

The current stepwise pipeline should be upgraded to this conceptual order:

1. Fetch and validate market and context data.
2. Build regime observations.
3. Run fold-local detector fit and online regime replay.
4. Build raw features.
5. Apply regime-aware feature adaptation.
6. Train specialist library.
7. Replay router decisions out of sample.
8. Generate signals from the active specialist or blend.
9. Backtest with switching and detection lag.
10. Persist regime, routing, and specialist diagnostics.

New pipeline steps should include:

- `BuildRegimeObservationsStep`
- `DetectRegimeStateStep`
- `AdaptFeaturesByRegimeStep`
- `TrainSpecialistLibraryStep`
- `RouteSpecialistStep`
- `EvaluateRegimeSegmentsStep`
- `WriteRoutingDiagnosticsStep`

## Specialist Model Lifecycle Policy

The specialist library should use a governed lifecycle.

### Admission

- minimum regime support
- minimum transition support
- minimum stability score
- no blocked leakage findings
- acceptable calibration quality

### Active usage

- routed only when regime compatibility and health thresholds are satisfied
- can be blended if router policy allows

### Degradation

- marked degraded when per-regime or transition-conditioned performance collapses
- degradation should first reduce routing weight before retirement

### Retirement

- retire when persistent degradation, invalid schema, or structural incompatibility is confirmed
- retain immutable artifacts for reproducibility

## Example Experiment Flow

The standard research flow after the refactor should look like this:

1. Load market data and point-in-time-safe custom data.
2. Build causal regime observations from instrument, market, and cross-asset sources.
3. Fit detector bundle on the training prefix.
4. Replay detector updates online through validation or holdout.
5. Build raw features and apply regime-aware adaptation using decision-available regime state.
6. Train fallback generalist plus regime-compatible specialists.
7. Run the router online with hysteresis and cooldown.
8. Generate signals from the active specialist or weighted ensemble.
9. Execute the backtest with switching latency and switching costs.
10. Attribute outcomes back to the specialist, regime, and transition once labels mature.
11. Produce regime-segmented diagnostics and governance reports.

## Test Plan

The redesign should be accompanied by new regression tests from the first code pass.

### Core contract tests

- `tests/test_regime_detector_online_contract.py`
- `tests/test_regime_state_temporal_availability.py`
- `tests/test_feature_adaptation_prefix_invariance.py`
- `tests/test_specialist_library_metadata_contract.py`
- `tests/test_router_contract.py`

### Behavior tests

- `tests/test_router_hysteresis.py`
- `tests/test_router_cooldown.py`
- `tests/test_router_weighted_blend.py`
- `tests/test_unseen_regime_fallback.py`
- `tests/test_regime_transition_backtest_trace.py`
- `tests/test_structural_break_does_not_use_future_data.py`

### Governance tests

- `tests/test_regime_segmented_promotion_gates.py`
- `tests/test_router_over_switching_gate.py`
- `tests/test_specialist_retirement_policy.py`
- `tests/test_retraining_requires_structural_trigger.py`

## Recommended Implementation Order

The first implementation sequence after this planning document should be:

1. Add contracts and compatibility facades.
2. Extract regime observation and detector modules.
3. Insert explicit feature adaptation layer.
4. Implement specialist library metadata and persistence.
5. Implement router state, hysteresis, and diagnostics.
6. Refactor backtest replay for regime and routing traces.
7. Redesign AutoML config and experiment entrypoints.
8. Refactor drift and maintenance policy away from default retraining.

This order minimizes breakage because it establishes contracts before changing orchestration logic.

## Final Design Principles

The upgraded framework should behave according to these principles:

- regime inference is a live decision input, not an offline annotation
- model selection is a first-class governed subsystem
- specialist coverage is preferable to constant retraining
- drift response should escalate gradually from routing to recalibration to discovery to retraining
- every adaptive decision must be replayable and explainable
- unexplained aggregate success is not admissible evidence

If implemented in this sequence, the framework will stop treating regime change as a nuisance around a single model and start treating it as the central organizing fact of crypto market research.