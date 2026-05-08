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

Phase 0 is the compatibility foundation. Its purpose is not to introduce routing behavior yet. Its purpose is to make the codebase structurally ready for later phases by adding typed contracts and additive module boundaries while keeping today's pipeline, examples, configs, and artifacts working.

The governing rule for this phase is simple: additive structure only, no default behavior change.

#### Phase 0 objectives

Phase 0 must accomplish five things:

1. Introduce explicit typed contracts for regime state, specialist metadata, and routing decisions.
2. Keep the current public import surface working through compatibility facades.
3. Preserve current config-driven workflows and example entrypoints.
4. Make the new contracts serializable and safe to store in manifests and summaries.
5. Create one canonical place for future phases to extend rather than widening the current ad hoc dict surfaces.

#### Phase 0 delivery rules

- Do not change current training, backtest, or AutoML defaults.
- Do not require `router`, `model_library`, or detector bundles for existing examples to run.
- Do not remove or rename current exports from `core/__init__.py`, `core/regime.py`, or `core/regime_training.py`.
- Do not introduce hidden runtime state inside contract or facade modules.
- All new contracts must be plain dataclass-style payloads or enums that can round-trip through JSON-ready dicts.
- New registry and summary fields must be optional and fail-open for historical artifacts, but fail-typed for newly written artifacts.

#### Phase 0 scope boundaries

Included in Phase 0:

- contract dataclasses and protocols
- package skeletons for `core/regimes/`, `core/specialists/`, and `core/routing/`
- compatibility facades in existing modules
- serialization helpers and manifest extensions
- public export updates in `core/__init__.py`
- narrow tests for contract shape, serialization, and backward compatibility

Explicitly out of scope for Phase 0:

- live router scoring or switching logic
- specialist-library training and retirement behavior
- regime detector ensemble implementation
- backtest routing replay
- promotion gating on router artifacts
- AutoML search over detector or router bundles

If any of those behaviors are required to satisfy a code change, that change belongs to Phase 1 or later, not Phase 0.

#### Files to add in Phase 0

Add these files even if some only contain contracts and no business logic yet:

- `core/regimes/__init__.py`
- `core/regimes/contracts.py`
- `core/specialists/__init__.py`
- `core/specialists/contracts.py`
- `core/routing/__init__.py`
- `core/routing/contracts.py`

These files are intentionally contract-first. They should not depend on the current training implementation beyond shared primitive types such as pandas timestamps, enums, and JSON-ready metadata.

#### Files to modify in Phase 0

- `core/regime.py`
- `core/regime_training.py`
- `core/__init__.py`
- `core/registry/` manifest helpers and versioned metadata emitters
- `core/pipeline.py` only where current explicit state keys need typed aliases or summaries
- `tests/` for new compatibility and serialization coverage

#### Contract deliverables

##### Regime contracts

`core/regimes/contracts.py` should define the canonical regime payloads used by later phases.

Minimum deliverables:

- `RegimeObservationContract`
- `RegimeStateContract`
- `RegimeTransitionContract`
- `RegimeDetectorManifest`
- `RegimeTraceSummary`
- `BaseRegimeDetector` protocol

Required field expectations:

- all contracts carry `schema_version`
- all time-aware contracts carry `as_of` and `available_at` when applicable
- all contracts carry a freeform `metadata` mapping for forward-compatible fields
- all contracts expose `to_dict()` and `from_dict()` or equivalent normalization helpers

Compatibility mapping to current code:

- `RegimeObservationContract` must align with current `pipeline.state["regime_observations"]`
- `RegimeStateContract` must align with current `pipeline.state["regime_state_frame"]` and `pipeline.state["regime_detection"]`
- `RegimeTraceSummary` must be able to summarize current preview and fold-local regime reporting without requiring the future router

##### Specialist contracts

`core/specialists/contracts.py` should define the canonical metadata for regime specialists, even before a full specialist-library implementation exists.

Minimum deliverables:

- `SpecialistSpec`
- `SpecialistArtifactRef`
- `SpecialistHealthContract`
- `SpecialistPerformanceSlice`
- `SpecialistLifecycleState` enum
- `SpecialistLibrarySnapshot`

Required field expectations:

- `SpecialistSpec` must capture compatible regimes, feature policy references, estimator family, symbol, timeframe, and training-window lineage
- `SpecialistHealthContract` must capture stability, decay, calibration freshness, and failure flags
- `SpecialistLibrarySnapshot` must support one fallback generalist plus zero or more specialists without implying routing logic yet

Compatibility mapping to current code:

- `RegimeAwareModelBundle` remains the current runtime carrier in `core/regime_training.py`
- Phase 0 should add adapters that can summarize a `RegimeAwareModelBundle` into `SpecialistLibrarySnapshot` semantics without changing prediction behavior

##### Router contracts

`core/routing/contracts.py` should define the decision and state payloads that later router implementations must emit.

Minimum deliverables:

- `RoutingDecisionContract`
- `RoutingScoreComponent`
- `RouterStateSnapshot`
- `RouterManifest`
- `BaseRouter` protocol

Required field expectations:

- routing decisions must support hard switch and weighted selection fields simultaneously
- score payloads must be decomposable by named component
- router state must be serializable without requiring a live model object
- manifests must be versioned and additive so historical summaries can omit them safely

Compatibility mapping to current code:

- there is no existing router module to preserve, so router contracts are additive
- compatibility in this case means the new router package can be imported and serialized without changing any current caller behavior

#### Compatibility facade plan

##### `core/regime.py`

`core/regime.py` remains the user-facing regime facade in Phase 0.

Its role after Phase 0:

- continue exporting current helpers such as `RegimeFeatureSet`, `build_default_regime_feature_set`, `detect_regime`, and ablation helpers
- re-export new regime contracts from `core/regimes/contracts.py`
- add small normalization helpers that convert existing preview/fold-local dicts into typed contract instances
- keep all current function signatures intact

No caller should need to import `core.regimes.contracts` directly unless they want the new typed layer.

##### `core/regime_training.py`

`core/regime_training.py` remains the current training facade.

Its role after Phase 0:

- continue exporting `RegimeAwareFeatureFrame`, `RegimeAwareModelBundle`, `summarize_regime_coverage`, `train_regime_aware_model`, and `train_regime_aware_walk_forward`
- add adapter helpers that derive `SpecialistSpec`, `SpecialistHealthContract`, and `SpecialistLibrarySnapshot` summaries from the current bundle and training reports
- keep current training entrypoints dict-compatible so existing tests and examples remain unchanged

The crucial rule is that Phase 0 must not force the training path to consume the new specialist contracts before the specialist-library implementation exists.

##### `core/__init__.py`

`core/__init__.py` is the public compatibility boundary and must be treated as part of the Phase 0 deliverable.

Changes required:

- preserve every currently exported symbol
- add exports for the new regime, specialist, and router contracts
- keep import order stable enough that user code importing `from core import ...` does not break on optional package initialization
- avoid circular imports by keeping contract modules dependency-light

#### Pipeline touch points allowed in Phase 0

`core/pipeline.py` should only receive the smallest compatibility edits required to expose typed aliases for the new layers.

Allowed Phase 0 touches:

- store typed-compatible summaries beside existing dict state
- attach contract metadata to `regime_detection`, `training["regime"]`, or future routing placeholders
- keep `build_regime_observations()` and `detect_regimes()` outputs backward-compatible while making them convertible to the new contracts

Disallowed Phase 0 touches:

- introducing real router selection in `TrainModelsStep` or `SignalsStep`
- changing signal generation defaults
- changing backtest behavior

#### Registry and serialization plan

Phase 0 must extend artifact and manifest handling before later phases write richer objects.

Required work in `core/registry/` and adjacent storage helpers:

- add JSON-ready serialization helpers for all new contracts
- add optional manifest sections for:
  - `regime_contracts`
  - `specialist_library`
  - `router_manifest`
- include `schema_version` and `contract_type` in stored payloads
- ensure historical manifests without these sections still load cleanly
- ensure newly written manifests can be validated structurally without knowing future Phase 1+ fields

Compatibility rule:

- Phase 0 readers must accept old manifests
- Phase 0 writers may emit the new optional sections
- no capital-facing logic may require the new sections yet

#### Detailed implementation order inside Phase 0

Implement Phase 0 in this exact sequence:

1. Create `__init__.py` and `contracts.py` files under `core/regimes/`, `core/specialists/`, and `core/routing/`.
2. Add dataclasses, enums, and protocols with `to_dict()` and `from_dict()` support.
3. Add lightweight normalization helpers for current dict-based pipeline state.
4. Update `core/regime.py` to re-export regime contracts and provide regime contract adapters.
5. Update `core/regime_training.py` to expose specialist snapshot adapters while preserving existing training entrypoints.
6. Update `core/__init__.py` to re-export the new contract symbols.
7. Extend registry and storage manifest helpers with optional contract blocks.
8. Add narrow compatibility tests before any later-phase logic is introduced.

This order matters because it guarantees that the public facade is preserved before registry writers and downstream code begin seeing the new payloads.

#### Test plan for Phase 0

Add these focused tests:

- `tests/test_regime_contracts.py`
- `tests/test_specialist_contracts.py`
- `tests/test_router_contracts.py`
- `tests/test_contract_serialization.py`
- `tests/test_regime_facade_compatibility.py`
- `tests/test_regime_training_facade_compatibility.py`
- `tests/test_core_public_exports.py`
- `tests/test_registry_manifest_contract_compatibility.py`

The minimum behavior each test group must cover:

- contracts instantiate with required fields only
- contracts round-trip through `to_dict()` / `from_dict()`
- old facade imports still resolve
- current `ResearchPipeline.detect_regimes()` and regime-aware training outputs can be summarized into new contract forms
- old manifests load without contract sections
- new manifests write optional contract sections without breaking old readers

#### User-facing compatibility matrix

Phase 0 is only complete if this matrix remains true.

Must remain unchanged for users:

- root examples and `run.py --config ...`
- current YAML sections such as `regime` and `model.regime_aware`
- imports from `core`, `core.regime`, and `core.regime_training`
- current training summary keys consumed by tests and examples
- current preview regime outputs under `pipeline.detect_regimes()`

May be added in Phase 0 but must be optional:

- new contract classes exported from `core`
- optional contract summaries in manifests and pipeline state
- additive helper functions for contract normalization

#### Risks and failure modes to prevent in Phase 0

- circular imports between `core/__init__.py`, facade modules, and contract modules
- contract modules importing heavy training or backtest logic
- accidental renames of legacy exports
- writing required manifest fields too early and breaking historical loads
- mixing typed contracts and live model objects in the same serialized payload

The simplest guardrail is to keep contract modules dependency-light and serialization-focused.

#### Phase 0 acceptance criteria

Phase 0 is complete only when all of these are true:

- no existing example or config path requires changes
- current regime preview and regime-aware training tests still pass unchanged
- `core/regime.py` and `core/regime_training.py` remain valid public facades
- new regime, specialist, and router contracts can be imported from `core`
- all new contracts serialize to JSON-ready dicts and round-trip cleanly
- registry manifests accept both legacy and Phase 0 payload shapes
- no router behavior or specialist-library behavior has been activated by default

If any default runtime behavior changes during this phase, Phase 0 has expanded past its intended scope and should be split before continuing.

### Phase 1: Online regime detection layer

Phase 1 is the first runtime-bearing phase. It replaces the current ad hoc
`detect_regime(...)` execution path with a typed causal detector runtime while
preserving the current public facade and config-driven workflows.

The governing rule for this phase is simple: one canonical observation builder,
one canonical replay engine, and no detector may inspect future rows.

#### Phase 1 objectives

Phase 1 must accomplish five things:

1. Extract regime observation construction into a dedicated runtime module and
   make it the canonical source for both global preview and fold-local replay.
2. Implement concrete detector classes behind `BaseRegimeDetector` for the
   first supported causal families: volatility, trend, liquidity, break, and
   filtered HMM.
3. Add an online replay engine that executes `fit -> initialize -> update`
   causally and materializes a typed regime state trace.
4. Add detector-bundle fusion so the runtime can emit confidence,
   disagreement, warm-up state, and primary-label outputs from one place.
5. Rewire `core/pipeline.py` and `core/regime.py` so all preview and fold-local
   regime outputs are produced through the new runtime without breaking current
   examples, tests, or public imports.

#### Phase 1 delivery rules

- No detector may consume rows beyond the current observation timestamp.
- Filtered HMM probabilities are allowed; smoothed posteriors are forbidden in
  preview, training, holdout, and backtest paths.
- Legacy config keys such as `regime.method`, `regime.n_regimes`,
  `regime.builder`, and `regime.column_name` must continue to work.
- Current pipeline aliases must remain populated:
  - `pipeline.state["regime_observations"]`
  - `pipeline.state["regime_state_frame"]`
  - `pipeline.state["regimes"]`
  - `pipeline.state["regime_detection"]`
- Warm-up rows and detector disagreement must be explicit in outputs rather
  than hidden behind default labels.
- Detector manifests and trace summaries may be added to state and summaries,
  but feature adaptation, specialist selection, and router behavior remain out
  of scope for this phase.

#### Phase 1 scope boundaries

Included in Phase 1:

- `core/regimes/observations.py` for canonical observation building
- concrete detector implementations under `core/regimes/`
- an online replay engine for global preview and fold-local inference
- detector ensemble and confidence fusion logic
- native runtime support for both legacy and detector-bundle config shapes
- detector diagnostics, manifests, and typed trace summaries
- replay determinism and no-lookahead test coverage

Explicitly out of scope for Phase 1:

- regime-conditioned scaling, masking, or selector banks
- specialist-library persistence and lifecycle behavior
- router scoring, hysteresis, cooldown, and switching
- routing-aware backtest replay and switching-cost accounting
- AutoML search across detector-specialist-router bundles

If a code change requires per-regime feature policies or routing behavior to be
complete, that work belongs to Phase 2 or later, not Phase 1.

#### Files to add in Phase 1

Add these runtime modules even if some begin as thin wrappers around current
logic:

- `core/regimes/observations.py`
- `core/regimes/detectors.py`
- `core/regimes/ensemble.py`
- `core/regimes/online_state.py`
- `core/regimes/diagnostics.py`
- `core/regimes/validation.py`

Expected responsibilities:

- `observations.py`: canonical observation-frame construction and normalization
- `detectors.py`: concrete detector classes and fit/update helpers
- `ensemble.py`: detector fusion, disagreement, and confidence rules
- `online_state.py`: replay engine from observations to typed state trace
- `diagnostics.py`: detector manifests, disagreement summaries, warm-up stats,
  and trace-materialization helpers
- `validation.py`: replay determinism checks and fold-local no-leakage helpers

#### Files to modify in Phase 1

- `core/regime.py`
- `core/pipeline.py`
- `core/__init__.py`
- `experiments/config.py`
- regime-focused tests under `tests/`

Phase 1 should not require broad edits to examples or root config entrypoints.
Existing examples must continue to run unless they explicitly opt into richer
Phase 1 detector diagnostics.

#### Runtime architecture plan

##### Observation layer

`core/regimes/observations.py` becomes the canonical regime observation source.

Required responsibilities:

- move the current default builder behavior out of `core/pipeline.py`
- preserve `RegimeFeatureSet` semantics: `frame`, `source_map`, and
  provenance summaries
- support both full-preview and fold-local observation windows
- reuse the current point-in-time-safe context joins from:
  - futures context
  - reference overlays
  - cross-asset context
  - multi-timeframe context
- keep observation columns stable for the same config and source data

Required public helper surface:

- one function for global preview observation construction
- one function for fold-local buffered observation construction
- one normalization helper that returns a `RegimeFeatureSet`

The current `_build_regime_observation_feature_set(...)` path in
`core/pipeline.py` should become a thin delegating wrapper, not the owning
implementation.

##### Detector execution contract

Each detector implementation must conform to `BaseRegimeDetector` and be usable
in a causal replay loop.

Operational contract:

1. `fit(observations)` consumes only the training prefix or configured fit
   window.
2. `initialize(observations)` creates online runtime state using only fitted
   parameters and any allowed warm-up prefix.
3. `update(state, observation)` consumes exactly one decision-eligible
   `RegimeObservationContract` and emits the next `RegimeStateContract`.
4. `manifest()` returns frozen detector metadata including parameters,
   warm-up requirements, fit-window lineage, and schema version.

The replay engine in `core/regimes/online_state.py` should own iteration over
observation rows so detectors never need ad hoc batch semantics in pipeline
code.

##### Concrete detectors to implement first

The first implementation wave should support these concrete detectors:

1. `VolatilityStateDetector`
   - fit threshold bands from train-prefix volatility features
   - use causal features such as `ewm_vol_*`, `vol_cluster_ratio_*`,
     `shock_score_*`, and related range features
   - emit interpretable labels such as `low_vol`, `normal_vol`, `high_vol`,
     and optional `shock`

2. `TrendStateDetector`
   - use causal trend and mean-reversion features such as `trend_z_*`,
     `mean_reversion_gap_*`, drawdown, and directional return features
   - emit labels such as `trend_up`, `trend_down`, `range`, and
     `mean_reverting`

3. `LiquidityStateDetector`
   - use quote-volume, turnover, illiquidity, and trade-count features where
     available
   - emit `liquid`, `normal_liquidity`, `illiquid`, and
     `liquidity_shock`-style states

4. `BreakStateDetector`
   - wrap causal break and shock evidence such as online CUSUM-like scores
   - emit a break-aware label or explicit break-score payload without requiring
     structural retraining behavior yet

5. `FilteredHMMDetector`
   - fit only on the training prefix
   - freeze scaler and HMM parameters after fit
   - emit filtered forward probabilities only
   - explicitly forbid smoothing in replay and summaries

The migration-safe default should be a compatibility detector that reproduces
the current explicit regime behavior behind the new detector protocol before
additional detector families become default.

##### Detector bundle and ensemble logic

`core/regimes/ensemble.py` must support both single-detector and multi-detector
execution.

Minimum fusion responsibilities:

- weighted vote over detector-level label probabilities
- primary-detector fallback when the bundle is configured but fusion is
  underdetermined
- confidence floors and disagreement penalties
- explicit `warm` state when one or more detectors are not yet decision-ready
- detector-level evidence retained in diagnostics rather than collapsed away

Required aggregated outputs:

- `regime`
- `regime_confidence`
- `warm`
- `detector_disagreement`
- per-detector label and confidence fields
- per-detector probability payloads when available

The ensemble must reduce confidence when detectors disagree. It must not hide
detector disagreement behind a final label without diagnostics.

##### State trace materialization

`core/regimes/online_state.py` should materialize a replayed detector trace into
both typed contracts and DataFrame outputs.

Required outputs for each replay:

- ordered `RegimeObservationContract` sequence or equivalent trace summary
- ordered `RegimeStateContract` sequence
- optional `RegimeTransitionContract` sequence
- a state DataFrame aligned to the requested index
- detector manifests and replay metadata
- warm-up counts, disagreement counts, and available-row counts

The DataFrame representation must continue to satisfy current pipeline callers,
especially `pipeline.state["regimes"]` and fold-local training paths.

#### Config compatibility and migration plan

Phase 1 should make the runtime understand the new detector-bundle shape
natively while preserving the old shape.

Required compatibility rules:

- legacy shape:
  - `regime.method`
  - `regime.n_regimes`
  - `regime.builder`
  - `regime.column_name`
- new shape:
  - `regime.detectors`
  - `regime.ensemble`

Runtime policy:

1. If `regime.detectors` is present, execute the native detector bundle.
2. If only legacy keys are present, normalize them into a single-detector
   bundle internally.
3. Preserve `raw_config` fidelity in config loaders so old and new configs can
   both round-trip through experiment summaries.

`experiments/config.py` should remain the compatibility boundary for mixed old
and new regime config shapes until later phases remove the legacy projection.

#### Pipeline integration plan

The controlling integration seam remains `core/pipeline.py`.

Required Phase 1 pipeline changes:

1. `BuildRegimeObservationsStep`
   - delegate observation construction to `core/regimes/observations.py`
   - keep current state keys and provenance aliases intact

2. `RegimeStep`
   - replace direct `detect_regime(...)` execution with the online replay
     engine
   - store detector manifests, replay details, disagreement summaries, and warm
     counts inside `pipeline.state["regime_detection"]`
   - continue returning backward-compatible fields:
     - `regime_observations`
     - `regime_state_frame`
     - `regimes`
     - `mode`
     - `provenance`

3. `_build_fold_local_regime_frame(...)`
   - use the same observation builder and replay engine as global preview
   - ensure buffered observation windows are used only for rolling feature
     stabilization, not for future-aware threshold fitting
   - ensure detector fit is restricted to `fit_index`

4. `_build_regime_state_frame(...)`
   - become a compatibility wrapper over the new runtime rather than owning the
     logic directly

The most important invariant is that global preview and fold-local replay must
share the same observation-to-state implementation path.

#### Diagnostics and artifact plan

Phase 1 must make detector behavior inspectable before Phase 2 introduces
feature adaptation.

Add these diagnostics to pipeline state and summaries:

- detector manifest list
- observation column list
- state column list
- available row count
- warm-up row count
- detector disagreement summary
- primary detector name
- ensemble policy name
- transition count
- trace summary contract

Add these optional artifact payloads where natural:

- `regime_detection["detector_manifests"]`
- `regime_detection["ensemble"]`
- `regime_detection["trace_summary"]`
- `regime_detection["detector_outputs"]`

These additions must be additive and must not replace current legacy summary
keys consumed by tests or examples.

#### Detailed implementation order inside Phase 1

Implement Phase 1 in this exact sequence:

1. Move the current observation-building logic into
   `core/regimes/observations.py` without changing behavior.
2. Add one compatibility detector that reproduces the current explicit regime
   method behind `BaseRegimeDetector`.
3. Add the replay engine in `core/regimes/online_state.py` and route a
   single-detector preview path through it.
4. Add detector manifests, trace summaries, and diagnostics helpers.
5. Implement volatility, trend, liquidity, and break detectors using the
   canonical observation frame.
6. Implement the filtered HMM detector with explicit no-smoothing guards.
7. Add ensemble fusion and detector disagreement logic.
8. Rewire fold-local regime construction and training-time replay through the
   same runtime.
9. Update config normalization so new detector-bundle config sections are
   handled natively while legacy configs still run.
10. Add focused replay, no-lookahead, and determinism tests.

This order matters because it creates a low-risk bridge from the current
single-path behavior into the multi-detector runtime before any detector bundle
becomes authoritative.

#### Concrete implementation slices for Phase 1

Execute Phase 1 in the following slices. Each slice must end with a narrow test
or replay validation before the next slice begins.

##### Slice 1: Canonical observation runtime extraction

Goal:

Create `core/regimes/observations.py` as the owning module for preview and
fold-local observation construction while preserving current behavior.

Implementation scope:

- move pipeline-facing default observation construction behind the new module
- centralize reference-data resolution and fold-local buffered observation
  scoping
- keep `RegimeFeatureSet` and provenance behavior unchanged
- make `core/pipeline.py` delegate rather than own the observation-building
  logic

Validation slice:

- `tests/test_regime_layer_ablation.py`
- one new focused observation-runtime test file

##### Slice 2: Single-detector compatibility replay runtime

Goal:

Add the online replay engine and one compatibility detector that reproduces the
current explicit regime path behind `BaseRegimeDetector`.

Implementation scope:

- add `core/regimes/online_state.py`
- add a compatibility detector in `core/regimes/detectors.py`
- route single-detector preview through replay
- emit manifests and trace summaries without changing output keys

Validation slice:

- existing regime preview tests
- new replay determinism test for the compatibility detector

##### Slice 3: Native detector family rollout

Goal:

Introduce the first concrete detector families using the canonical observation
frame.

Implementation scope:

- add four native `BaseRegimeDetector` implementations in
  `core/regimes/detectors.py`:
  - `TrendRegimeDetector`
  - `VolatilityRegimeDetector`
  - `LiquidityRegimeDetector`
  - `BreakRegimeDetector`
- factor shared detector helpers so native detectors do not duplicate:
  - observation-frame normalization
  - schema locking at fit time
  - lexical / provenance-aware column selection
  - threshold fitting on the training prefix only
  - neutral fallback emission when evidence is unavailable
- extend `core/regimes/online_state.py` only as needed so the existing replay
  engine can run any one native detector without introducing multi-detector
  fusion yet
- update `core/pipeline.py` to resolve a selected detector instance and route
  single-detector replay through the same preview / fold-local seam added in
  Slice 2
- extend `experiments/config.py` so one native detector can be configured
  natively while explicitly rejecting multi-detector fusion requests until
  Slice 5
- keep detector outputs, manifests, and trace summaries additive; do not remove
  or rename current legacy keys in `pipeline.state["regime_detection"]`

Required invariants:

- every detector may inspect the full fit prefix during `fit(...)`, but no
  detector may inspect validation or test rows when replaying a fold-local
  trace
- detector schema must freeze at fit time; selected columns, source counts, and
  threshold parameters must not change mid-replay
- detector `update(...)` must be row-causal; if a detector requires prior state
  such as a previous score, that state must be carried explicitly in the replay
  state object rather than recomputed from future rows
- `state_frame.index` must match the requested replay index exactly, including
  warm-up rows that emit neutral or partial outputs
- native detector rollout must be additive only: legacy `method: explicit`
  behavior stays unchanged unless config explicitly selects a native detector
- Slice 3 must not introduce detector fusion, weighted voting, disagreement
  penalties, or any bundled authority over the final composite label; those are
  reserved for Slice 5

Detector contract for this slice:

Each native detector should emit a `RegimeStateContract` whose
`detector_outputs` contain at minimum:

- `score`: scalar detector score for the current row
- detector-specific threshold fields such as `lower_threshold`,
  `upper_threshold`, or `trigger_threshold`
- `selected_columns`: frozen fit-time column list or a serialized equivalent in
  metadata
- `selected_column_count`
- `warm`: explicit readiness flag
- one typed state column, for example `trend_regime`, `volatility_regime`,
  `liquidity_regime`, or `structural_break_regime`
- `regime`: detector-local label used by the existing replay surface when a
  single detector is authoritative

Detector manifests must include at minimum:

- `detector_name`
- `detector_type`
- fit-time params actually used after defaults are resolved
- `warmup_bars`
- fit-window start, end, and row count
- selected-column summary and source breakdown in `metadata`

Proposed detector semantics:

- trend detector:
  - score source selection should default to columns whose names contain one of
    `trend`, `ret_`, `return`, `momentum`, or `slope`
  - default exclusions should include `vol`, `volume`, `liquid`, `break`, and
    `shock`
  - fit should compute frozen lower and upper quantile thresholds from the fit
    prefix score series
  - replay should emit `trend_regime` in `{-1, 0, 1}` representing bearish,
    neutral, and bullish trend states
- volatility detector:
  - score source selection should default to columns matching `vol`, `range`,
    `atr`, `dispersion`, `cluster`, `drawdown`, or `shock`
  - replay should emit `volatility_regime` in `{-1, 0, 1}` representing calm,
    neutral, and stressed volatility states
  - fit must freeze quantile thresholds from the training prefix only
- liquidity detector:
  - primary score should aggregate `liquid`, `volume`, `turnover`, and `trade`
    features
  - illiquidity penalties should subtract `illiquid` and `amihud` features if
    present
  - replay should emit `liquidity_regime` in `{-1, 0, 1}` representing
    illiquid, neutral, and liquid states
  - any inversion rule must be stored in the manifest and never inferred later
- break detector:
  - score source selection should default to `break`, `shock`, `jump`,
    `crash`, and `drawdown`
  - this detector may also maintain explicit replay state for prior volatility
    or prior break score if a first-difference acceleration term is used
  - replay should emit `structural_break_regime` in `{0, 1}` with `1`
    reserved for an active break / shock state
  - thresholding should be upper-tail only by default; there is no need for a
    symmetric lower threshold in Slice 3

Config and routing rules for Slice 3:

- accepted config surface for the first rollout should be the existing
  `regime.detectors` list, but Slice 3 should allow exactly one enabled native
  detector to be authoritative at runtime
- if more than one non-compatibility detector is enabled, fail fast with a
  precise error stating that detector fusion is deferred to Slice 5
- if `regime.method` is set and `regime.detectors` is absent, preserve current
  Slice 2 behavior unchanged
- if a single native detector is configured, `core/pipeline.py` should resolve
  it through a detector factory instead of the legacy `detect_regime(...)`
  dispatcher
- `experiments/config.py` should normalize one-detector native configs without
  collapsing them back into legacy `method` aliases
- config normalization must reject ambiguous combinations such as both
  `method: explicit` and a single native detector marked `primary: true`

Hidden dependency that must be addressed in Slice 3:

The current ablation report in `core/regime.py` still calls
`detect_regime(...)` internally. If Slice 3 introduces a native detector as the
authoritative state producer, the ablation report must stop silently switching
back to the legacy path.

Plan for that dependency:

- add a detector-aware ablation helper that replays the same detector over:
  - the full canonical observation frame
  - the endogenous-only observation subset
- freeze schema separately for each replayed subset so contextual columns are
  not accidentally carried into the endogenous baseline
- compute stability, agreement, and gate decisions from replayed outputs, not
  from a fallback call to `detect_regime(...)`
- keep the public ablation payload shape unchanged where possible so existing
  summaries remain readable

Recommended implementation order inside Slice 3:

1. refactor shared fit / update helpers in `core/regimes/detectors.py` so the
   compatibility detector and native detectors share schema locking and scalar
   coercion behavior
2. implement `TrendRegimeDetector` first because it is the lowest-risk
   three-state detector and provides the template for quantile-threshold
   detectors
3. implement `VolatilityRegimeDetector` and `LiquidityRegimeDetector` on the
   same helper surface
4. implement `BreakRegimeDetector` last because it is the only detector in this
   slice that may require explicit replay state beyond frozen thresholds
5. add a detector factory / resolver in `core/regimes/detectors.py` or a small
   adjacent module, then route `core/pipeline.py` through it for the
   single-native-detector case only
6. migrate ablation reporting so native-detector preview does not fall back to
   legacy `detect_regime(...)`
7. add config normalization guards in `experiments/config.py`
8. add focused tests and only then consider wiring one example config for
   preview inspection

Planned file-level changes:

- `core/regimes/detectors.py`
  - keep `ExplicitCompatibilityRegimeDetector`
  - add native detector classes
  - add shared score-selection and threshold-fit helpers
  - add a factory such as `build_regime_detector(spec, config)`
- `core/regimes/online_state.py`
  - keep replay ownership centralized here
  - add only the minimal metadata plumbing needed for detector-specific warm-up
    diagnostics and detector-local outputs
- `core/pipeline.py`
  - replace the current `if method == "explicit"` branch with detector
    resolution logic that can select either the compatibility detector or one
    native detector
  - keep the returned `regime_state_frame` and additive metadata keys stable
- `core/regime.py`
  - either adapt `build_regime_ablation_report(...)` to accept a detector
    callback / replay hook or add a detector-aware sibling helper and route the
    pipeline through it
- `experiments/config.py`
  - normalize one-detector native config
  - reject unsupported multi-detector authority before Slice 5
- tests:
  - add `tests/test_regime_detectors.py`
  - extend `tests/test_regime_layer_ablation.py`
  - extend config-runtime compatibility coverage if native detector config is
    accepted through entrypoints

Validation slice:

- `tests/test_regime_detectors.py`
- `tests/test_regime_layer_ablation.py`
- focused config-runtime compatibility coverage if a native detector becomes
  user-selectable through config entrypoints

Minimum validation behaviors:

- replaying the same fit prefix twice yields identical detector thresholds,
  manifests, and state frames
- extending the dataset beyond a cutoff does not change detector outputs on the
  already-observed prefix
- fold-local replay never fits on rows outside `fit_index`
- detectors with no selected columns emit explicit neutral / unavailable states
  instead of crashing or silently borrowing unrelated columns
- native-detector ablation compares detector replay against an endogenous-only
  replay of the same detector, not against the legacy dispatcher
- single-native-detector config routes correctly, while multi-detector
  authority is rejected with a deterministic error message

Suggested narrow validation commands while implementing Slice 3:

- `g:/N/repos/ml/.venv/Scripts/python.exe -m pytest tests/test_regime_detectors.py`
- `g:/N/repos/ml/.venv/Scripts/python.exe -m pytest tests/test_regime_layer_ablation.py`
- `g:/N/repos/ml/.venv/Scripts/python.exe -m pytest tests/test_regime_detectors.py tests/test_regime_layer_ablation.py tests/test_regime_observation_runtime.py`

Acceptance criteria:

- the repo contains four native detector classes with deterministic fit / update
  behavior on canonical observation frames
- one native detector can be selected as the authoritative replayed detector
  without changing legacy explicit behavior for configs that do not opt in
- detector manifests expose fit-time schema and thresholds clearly enough to
  debug why a detector emitted a given state
- ablation reporting remains causally correct when a native detector is used
- the slice ends without any ensemble fusion, weighted vote, or multi-detector
  final label logic being introduced early

Explicit non-goals for Slice 3:

- no filtered HMM work
- no multi-detector fusion or disagreement scoring
- no router integration
- no example or docs churn beyond the minimum needed to exercise one native
  detector path after tests pass

##### Slice 4: Filtered HMM runtime

Goal:

Support HMM-based regime inference under explicit filtered-only replay rules.

Implementation scope:

- implement filtered HMM fit and update path
- freeze fitted parameters per fold
- explicitly block smoothed posterior usage

Validation slice:

- HMM replay test confirming filtered outputs only
- regression coverage for fold-local prefix fitting

##### Slice 5: Detector ensemble and confidence fusion

Goal:

Fuse multiple detectors into one typed regime trace with disagreement and warm
diagnostics.

Implementation scope:

- add `core/regimes/ensemble.py`
- implement weighted vote, primary-detector fallback, and disagreement penalty
- expose bundle manifests and detector-level evidence in summaries

Validation slice:

- ensemble determinism tests
- config compatibility tests for `regime.detectors` plus `regime.ensemble`

##### Slice 6: Fold-local runtime unification

Goal:

Ensure training-time fold-local regime construction uses the same replay engine
as global preview.

Implementation scope:

- route `_build_fold_local_regime_frame(...)` through the canonical runtime
- restrict detector fit to `fit_index`
- keep buffered observation windows only for rolling feature stabilization

Validation slice:

- fold-local regime tests
- regime-aware training regression slice

#### Concrete validation matrix for Phase 1

At minimum, each slice should validate one of these behaviors directly:

1. identical inputs and config yield identical observation frames
2. buffered fold-local observation windows stop at the fold boundary
3. detector fit windows never include validation or test rows
4. replayed regime traces align exactly to the requested index
5. warm-up and disagreement states are visible in the emitted trace
6. filtered HMM outputs never include smoothed posterior fields
7. legacy configs and detector-bundle configs both resolve successfully

#### Test plan for Phase 1

Add these focused tests:

- `tests/test_regime_observation_runtime.py`
- `tests/test_regime_detectors.py`
- `tests/test_regime_hmm_filtered.py`
- `tests/test_regime_ensemble.py`
- `tests/test_regime_replay_determinism.py`
- `tests/test_regime_config_runtime_compatibility.py`

Extend these existing suites:

- `tests/test_regime_layer_ablation.py`
- `tests/test_experiment_config.py`
- any pipeline slice that currently exercises fold-local regime behavior

Minimum behavior each Phase 1 test group must cover:

- observation frames are stable for identical input windows and config
- fold-local observation buffering does not alter fit-prefix causality
- explicit detectors fit only on the training prefix
- filtered HMM replay never exposes smoothed probabilities
- replaying the same detector bundle twice yields identical outputs
- detector disagreement is visible in outputs and trace summaries
- warm-up rows are explicit and not silently relabeled as fully confident
- legacy `regime.method` configs and new `regime.detectors` configs both run

#### Risks and failure modes to prevent in Phase 1

- silently changing label taxonomy for current explicit preview paths
- fitting detector thresholds on the full replay window instead of the fit
  prefix
- allowing HMM smoothing or future-scaled normalization in evaluation paths
- dropping warm-up rows and thereby misaligning state frames with input index
- exploding state-column width with inconsistent per-detector probability names
- diverging global preview and fold-local regime logic into separate code paths

The main guardrail is to make replay the only owner of regime-state production
and treat all preview and fold-local consumers as wrappers around that runtime.

#### Phase 1 acceptance criteria

Phase 1 is complete only when all of these are true:

- current root examples and legacy regime configs still run
- current `pipeline.detect_regimes()` callers still receive backward-compatible
  keys
- both global preview and fold-local training paths use the same observation
  and replay runtime
- regime traces are reproducible under walk-forward replay
- no detector uses future rows beyond the configured fit prefix
- detector disagreement and warm-up states are visible in outputs
- filtered HMM execution uses forward probabilities only
- detector manifests and trace summaries are available in pipeline state and
  summaries

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