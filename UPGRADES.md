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
- No HMM smoothing for live or backtest decisions. Only filtered state
  probabilities are allowed at decision time.
- Any clustering-based detector must fit on the training prefix only and infer
  online on later data.
- Any detector that requires state prototypes or thresholds must freeze them
  per fold before validation and holdout.
- Feature adaptation must use only the regime state available at the same
  timestamp.
- Router performance priors must update only after outcome maturity and
  execution delay.
- Validation, holdout, and promotion must preserve the existing purging and
  embargo discipline.

### Temporal correctness

- Regime state must be stamped with `as_of` and `available_at` timestamps.
- Regime probabilities, feature transforms, and routing decisions must all
  respect publication delays and availability lags.
- Backtests must simulate delayed regime recognition, switching latency, and
  switching costs.

### Statistical defensibility

- Aggregate metrics are insufficient for admissibility.
- Performance must be reported by regime, by transition, and on unseen or
  weakly covered regimes.
- Drift monitors must distinguish temporary turbulence from persistent
  structural change.
- Retraining must be a governed maintenance decision, not the default reaction
  to short-horizon degradation.

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

This separation is critical. Phase 1 established the fast regime-inference loop,
but the feature-adaptation, specialist, router, and maintenance loops are still
missing as first-class runtime layers.

## Current-to-Target Gap Summary

Implemented through Phase 1:

- typed regime, specialist, and router contracts
- canonical regime observation construction in `core/regimes/observations.py`
- canonical replay runtime in `core/regimes/online_state.py`
- compatibility, native score-based, and filtered-HMM detectors in
  `core/regimes/detectors.py`
- additive detector manifests and trace summaries in
  `pipeline.state["regime_detection"]`

Still missing for the target architecture:

- first-class feature-adaptation contracts, manifests, and replay artifacts
- fold-local regime-conditioned scaling and masking fitted on `X_fit` only
- specialist-library persistence and lifecycle management
- router scoring, hysteresis, and switching traces
- validation and backtest replay across the full
  `detector -> adaptation -> router` loop
- maintenance policy that prefers reroute and recalibration over retrain

The remaining phases below close those gaps without reopening the Phase 1
detector seam.

## Refactored Module Structure

Current realized structure after Phase 1:

```text
core/
  regimes/
    __init__.py
    contracts.py
    observations.py
    detectors.py
    online_state.py
  pipeline.py
  regime.py
  regime_training.py
```

Target additions for the remaining phases:

```text
core/
  feature_adaptation/
    __init__.py
    contracts.py
    runtime.py
    scaling.py
    masking.py
    diagnostics.py
  specialists/
    training.py
    library.py
    health.py
  routing/
    scorer.py
    hysteresis.py
    router.py
    diagnostics.py
  validation/
    regime_walk_forward.py
    transition_metrics.py
    unseen_regime.py
```

Existing files should be repurposed rather than abandoned:

- `core/regime.py` remains the compatibility facade and high-level entrypoint
  over `core/regimes/`
- `core/regime_training.py` remains the compatibility bridge until feature
  adaptation and specialist training are extracted cleanly
- `core/pipeline.py` remains the canonical orchestration point for detector,
  adaptation, training, and backtest steps
- `core/automl.py` should shift from retraining-centric search to bundle-
  centric search only after the runtime layers exist
- `core/drift.py` must eventually split regime drift, feature drift, model
  decay, and structural-break actions

## Core Contracts and Interfaces

The redesign must continue to rely on explicit runtime contracts rather than ad
hoc dictionaries. Phase 0 already introduced the regime, specialist, and router
contract families. The next contract family to add is the feature-adaptation
layer.

Required Phase 2 contract surface:

```python
@dataclass(frozen=True)
class FeaturePolicyContract:
    policy_id: str
    feature_columns: list[str]
    disabled_columns: list[str]
    generated_columns: list[str]
    regime_column: str
    scaling_mode: str
    fallback_mode: str
    sparse_regimes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseFeatureAdapter(Protocol):
    def fit(
        self,
        X: pd.DataFrame,
        regime_frame: pd.DataFrame,
        feature_metadata: Mapping[str, Any] | None = None,
    ) -> "BaseFeatureAdapter": ...

    def transform(
        self,
        X: pd.DataFrame,
        regime_frame: pd.DataFrame,
    ) -> tuple[pd.DataFrame, FeaturePolicyContract]: ...

    def manifest(self) -> dict[str, Any]: ...
```

Adapter rules:

- `fit(...)` may inspect only the fold-local training prefix
- `transform(...)` must reuse the frozen fit-time schema, masks, and scaling
  statistics
- low-confidence, warm, or sparse-regime rows must fall back deterministically
- generated feature lineage must be explicit enough to feed existing
  governance and portability diagnostics

## Regime Detection Subsystem

Phase 1 now owns the regime runtime and should not be reopened in later phases
except through additive integration points.

The regime subsystem already provides:

- discrete labels
- soft probabilities when a detector supports them
- confidence and warm-state signaling
- transition events
- detector provenance, manifests, and trace summaries

Supported detector families after Phase 1:

1. explicit compatibility replay
2. native trend, volatility, liquidity, and break detectors
3. filtered HMM replay with filtered-only posterior output

Later phases may consume these outputs, but they should not reintroduce
parallel regime-state implementations in pipeline code.


### Detector design details
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

- add `FilteredHMMDetector` to `core/regimes/detectors.py` behind the same
  `BaseRegimeDetector` protocol used by Slice 2 and Slice 3 detectors
- fit the HMM only on the training prefix and freeze all replay-time artifacts:
  - selected observation schema and column order
  - scaler parameters
  - fitted HMM parameters
  - state remap / label-order mapping
  - any warm-up or burn-in metadata
- replace the current batch `GaussianHMM.fit(...); predict(...)` path with a
  causal filtered forward pass in replay
- route a single authoritative `filtered_hmm` detector through the existing
  replay seam in `core/regimes/online_state.py`
- migrate legacy `regime.method: hmm` preview and fold-local paths onto the
  same filtered detector runtime so pipeline code no longer owns a second HMM
  implementation path
- expose HMM-specific manifests and diagnostics without changing current public
  pipeline keys

Required invariants:

- the HMM may fit on the full fit prefix, but replay must never inspect rows
  beyond the current observation timestamp
- scaler statistics must be frozen from the fit prefix only; no replay-time
  re-centering, re-scaling, or normalization on future rows is allowed
- state ordering must be deterministic and frozen at fit time so labels do not
  drift within a fold or change merely because future replay rows were added
- filtered probabilities must be computed incrementally from prior filtered
  state and the current emission likelihood only; no backward pass is allowed
- `state_frame.index` must match the requested replay index exactly, including
  rows that are marked unavailable, warm, or degenerate
- Slice 4 must not introduce detector fusion, smoothing-based diagnostics,
  adaptive online re-estimation, or live/backtest divergence in HMM logic

Filtered HMM detector contract:

`FilteredHMMDetector` should own four responsibilities:

1. schema resolution
   - default to all numeric observation columns unless explicit `columns` are
     configured
   - support explicit exclusion lists and fail fast when the effective schema
     is empty unless neutral fallback is explicitly allowed
   - freeze selected column order at fit time
2. fit-time artifact capture
   - freeze scaler mean and scale
   - freeze fitted `GaussianHMM` parameters
   - freeze state remap from raw latent states to ordered output labels
   - record fit-window lineage and model hyperparameters in the manifest
3. replay-time filtering
   - transform each new observation with the frozen scaler
   - compute emission log-likelihood for that single observation
   - advance one forward-filter step from the previous filtered state
   - normalize probabilities in log space to prevent underflow
4. typed output emission
   - emit `regime`, `regime_confidence`, and per-state probability fields
   - expose `warm` explicitly
   - include detector-local evidence in `detector_outputs` and metadata

Required HMM replay outputs per row:

- `regime`: argmax of the remapped filtered state probabilities
- `regime_confidence`: max filtered probability after remap
- stable probability field names such as `prob_state_0`, `prob_state_1`, ...
- `warm`: explicit readiness flag
- `selected_column_count`
- optional diagnostic fields such as `log_evidence` or `degenerate_fallback`
  only if they are additive and stable

No-smoothing enforcement rules:

- replay code must not call `GaussianHMM.predict(...)`,
  `GaussianHMM.predict_proba(...)`, `GaussianHMM.decode(...)`, or
  `GaussianHMM.score_samples(...)` on the replay sequence
- replay must compute filtered probabilities from frozen parameters and the
  current row only
- manifests and summaries must mark the posterior mode explicitly as
  `filtered`
- no smoothed posterior fields may appear in `state_frame`, detector outputs,
  manifests, summaries, or example output

Algorithm plan for Slice 4:

- fit stage:
  - clean the fit prefix and align it to the frozen selected schema
  - fit a frozen scaler on the fit prefix only
  - fit `GaussianHMM` on the scaled fit prefix
  - extract and freeze:
    - `startprob_`
    - `transmat_`
    - emission parameters (`means_`, `covars_`, covariance type)
    - state ordering / remap
- replay stage:
  - initialize the prior from the frozen start probabilities
  - for each row, compute one-step emission log-likelihood under the frozen
    Gaussian emissions
  - run a log-space forward recursion using the frozen transition matrix
  - normalize with `logsumexp`
  - remap latent-state probabilities into the frozen ordered label space
  - emit `RegimeStateContract` from the filtered probabilities

State remap policy for deterministic labels:

The raw HMM state IDs returned by EM are not stable enough for user-facing
labels. Slice 4 must freeze an ordering policy at fit time and persist it in
the manifest.

Recommended policy:

- compute a deterministic scalar summary for each latent state from the fitted
  emission means in scaled space
- sort states by that summary and map raw state IDs to ordered output states
- persist that mapping in the manifest metadata
- apply the same remap consistently to:
  - `regime`
  - per-state probability fields
  - ablation replays
  - fold-local replays

Numerical stability and fallback requirements:

- forward recursion must run in log space
- covariance handling must be explicit; do not silently approximate unsupported
  covariance types
- if the fit prefix is too short for the requested state count, collapse to a
  deterministic one-state fallback instead of raising unpredictably
- if HMM fit fails numerically, replay must emit an explicit fallback state and
  mark the manifest / detector outputs with the failure mode
- if replay rows contain NaNs across the selected schema, emit an unavailable or
  warm row rather than fabricating a confident state

Covariance-type policy for the first rollout:

- Slice 4 must support `diag` covariance because it is the current default
- if `full` covariance is implemented, it must be tested independently from
  `diag`
- if `spherical` or `tied` covariance are not implemented in replay, fail fast
  with a precise config/runtime error instead of silently coercing them

Warm-up policy:

- HMM replay should be decision-ready from the first valid filtered row once a
  fit has succeeded, unless the config explicitly sets `warmup_bars`
- if `warmup_bars` is configured, emit filtered probabilities but keep `warm`
  false until that replay count is reached
- warm-up rows must remain present in the state frame and trace summary

Config and routing rules for Slice 4:

- add native detector support for `type: filtered_hmm`
- support detector params such as:
  - `n_regimes` / `state_count`
  - `covariance_type`
  - `n_iter`
  - `tol`
  - `random_state`
  - `columns`
  - `exclude_columns`
  - `warmup_bars`
- preserve Slice 3 policy that only one native detector may be authoritative at
  runtime before Slice 5
- route legacy `regime.method: hmm` through the filtered HMM detector runtime
  instead of the old batch `_detect_hmm_regime(...)` path
- if both `regime.method: hmm` and a native primary detector are configured,
  keep the existing validation rule and reject the config as ambiguous

Hidden dependencies that must be addressed in Slice 4:

1. `core/regime.py` still contains `_detect_hmm_regime(...)`, which currently
   fits a batch HMM and calls `predict(...)` over the replay window.
   Slice 4 must ensure authoritative pipeline and ablation paths stop using
   that implementation.
2. Slice 3 made ablation detector-aware, but legacy `method: hmm` configs will
   still bypass that unless the pipeline converts them into a detector-backed
   replay path.
3. State ordering for HMMs is more fragile than the score-detector families.
   The plan must freeze and persist remap logic or fold-local comparisons will
   be unstable.

Plan for those dependencies:

- extend `build_regime_detector(...)` to construct `FilteredHMMDetector`
- add a compatibility translation so legacy `method: hmm` resolves to the same
  filtered detector used by native `filtered_hmm` specs
- ensure `build_regime_ablation_report(...)` receives a detector spec for HMM
  paths so the endogenous baseline is replayed through the same filtered logic
- either convert `_detect_hmm_regime(...)` into a thin compatibility façade
  over filtered replay or remove it from every authoritative runtime path by the
  end of the slice

Recommended implementation order inside Slice 4:

1. add HMM-specific helper functions in `core/regimes/detectors.py` for:
   - schema resolution
   - emission log-likelihood computation
   - log-space forward filtering
   - deterministic state remap
2. implement `FilteredHMMDetector.fit(...)`, `initialize(...)`, `update(...)`,
   and `manifest(...)`
3. extend the detector factory and canonical type mapping so
   `filtered_hmm` becomes a supported replay detector
4. route a single `filtered_hmm` detector spec through `core/pipeline.py`
5. migrate legacy `method: hmm` routing onto the same detector-backed replay
   path
6. ensure detector-aware ablation uses the filtered HMM replay path for both
   full and endogenous-only baselines
7. add focused tests that prove replay is filtered-only and prefix-invariant
8. only after those tests pass, consider exposing one example or config that
   opts into `filtered_hmm`

Planned file-level changes:

- `core/regimes/detectors.py`
  - add `FilteredHMMDetector`
  - add HMM-specific fit and forward-filter helpers
  - extend detector factory and type canonicalization
- `core/regimes/online_state.py`
  - keep replay ownership centralized here
  - add only the minimal metadata plumbing needed for per-state probabilities,
    confidence, and posterior-mode reporting
- `core/pipeline.py`
  - route both native `filtered_hmm` specs and legacy `method: hmm` through the
    same replay surface
  - keep additive output keys stable
- `core/regime.py`
  - stop relying on batch `_detect_hmm_regime(...)` in authoritative preview,
    ablation, and fold-local paths
  - keep public compatibility behavior only if it delegates to the filtered
    runtime
- `experiments/config.py`
  - preserve current config-validation rules
  - allow `filtered_hmm` native detector specs without collapsing them into a
    second HMM runtime
- tests:
  - add `tests/test_regime_hmm_filtered.py`
  - extend `tests/test_regime_layer_ablation.py`
  - extend config/runtime compatibility coverage if legacy `method: hmm`
    translation changes

Validation slice:

- `tests/test_regime_hmm_filtered.py`
- `tests/test_regime_layer_ablation.py`
- focused config/runtime compatibility coverage if legacy `method: hmm` is
  rerouted through the filtered detector runtime

Minimum validation behaviors:

- replaying the same filtered HMM twice with the same fit prefix yields
  identical manifests, remap metadata, and state frames
- extending the replay horizon does not change filtered outputs on the already
  observed prefix
- replay does not invoke `predict`, `predict_proba`, `decode`, or
  `score_samples` on the replay sequence
- legacy `method: hmm` preview and native `filtered_hmm` preview both produce
  filtered-only outputs through the same replay seam
- fold-local fit windows never include validation or test rows
- HMM ablation replays use the same filtered detector logic for both full and
  endogenous-only baselines
- degenerate or short fit windows fall back deterministically instead of
  raising unstable numerical errors

Suggested narrow validation commands while implementing Slice 4:

- `g:/N/repos/ml/.venv/Scripts/python.exe -m pytest tests/test_regime_hmm_filtered.py`
- `g:/N/repos/ml/.venv/Scripts/python.exe -m pytest tests/test_regime_hmm_filtered.py tests/test_regime_layer_ablation.py`
- `g:/N/repos/ml/.venv/Scripts/python.exe -m pytest tests/test_regime_hmm_filtered.py tests/test_regime_detectors.py tests/test_regime_layer_ablation.py tests/test_experiment_config.py tests/test_regime_compatibility_replay.py tests/test_regime_observation_runtime.py`

Acceptance criteria:

- the repo contains a `FilteredHMMDetector` that emits forward-filtered state
  probabilities only
- authoritative HMM preview and fold-local paths no longer depend on the old
  batch `predict(...)` regime path
- legacy `regime.method: hmm` and native `filtered_hmm` configs both resolve to
  the same causal replay runtime
- manifests clearly expose frozen schema, fit window, covariance policy, and
  state-remap metadata
- HMM replay is prefix-invariant, deterministic, and explicit about fallback /
  degenerate conditions
- no smoothed posterior fields appear anywhere in outputs or summaries

Explicit non-goals for Slice 4:

- no detector ensemble or disagreement fusion yet
- no smoothing-based retrospective diagnostics
- no online HMM re-fitting or adaptive parameter updates during replay
- no new latent-state model families beyond the filtered Gaussian HMM runtime

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

Phase 2 is the next implementation phase. Phase 1 made regime detection causal
and replayable; Phase 2 must make feature usage causal and replayable under the
same fold-local rules.

The governing rule for this phase is simple: every adapted feature seen by the
model must come from a policy fit on `X_fit` and `fit_regime_view` only, then
reused unchanged for validation, test, refit, and inference.

#### Phase 2 objectives

Phase 2 must accomplish five things:

1. Add a typed feature-adaptation runtime and make it the owning
   implementation for regime-conditioned scaling, masks, and bounded
   interaction expansion.
2. Integrate the adapter into `TrainModelsStep` so each validation fold runs
   `fit -> transform(train/val/test)` with a frozen schema before supervised
   feature selection and model fit.
3. Reuse the new runtime inside `core/regime_training.py` so the current
   `build_regime_aware_feature_frame(...)` path becomes a compatibility
   wrapper instead of a second implementation.
4. Persist per-fold policy manifests, fallback reasons, disabled columns,
   generated columns, and regime sample counts in training output and
   post-selection refit artifacts.
5. Keep default behavior additive: configs without `feature_adaptation` or
   with `enabled: false` must continue to behave like the current pass-through
   path.

#### Phase 2 delivery rules

- Adapter fit may inspect only `X_fit`, `fit_regime_view`, and feature
  metadata derived from the same fold.
- Validation and test rows must be transformed using frozen fit-time
  statistics, masks, generated-column order, and fallback thresholds.
- Adapted schemas must be prefix-invariant: extending the dataset may not
  change transformed outputs on an already-observed prefix.
- Low-confidence, warm, missing, or sparse-regime rows must follow an explicit
  fallback policy; no implicit borrowing from future occupancy is allowed.
- New generated columns must either be stationarity-safe by construction or
  pass a post-adaptation screening step fit on `X_fit` only.
- Feature-adaptation diagnostics must be additive to current training
  summaries; do not remove existing `feature_selection`,
  `feature_governance`, or `regime` payloads.

#### Phase 2 scope boundaries

Included in Phase 2:

- typed feature policy and adapter contracts
- fold-local regime-conditioned scaling
- per-regime feature masking and `disable_incompatible_features` logic
- bounded regime interaction features driven by config
- deterministic fallback for low-confidence and sparse-regime states
- policy manifests, diagnostics, and artifact propagation
- compatibility migration for `build_regime_aware_feature_frame(...)` and
  `model.regime_aware.strategy == "feature"`

Explicitly out of scope for Phase 2:

- specialist-library persistence or certification logic
- router scoring, hysteresis, or switching traces
- regime-conditioned specialist selection
- full adaptive backtest replay beyond transformed feature reuse
- AutoML search across adaptation bundles beyond config passthrough

#### Files to add in Phase 2

Add these runtime modules:

- `core/feature_adaptation/__init__.py`
- `core/feature_adaptation/contracts.py`
- `core/feature_adaptation/runtime.py`
- `core/feature_adaptation/scaling.py`
- `core/feature_adaptation/masking.py`
- `core/feature_adaptation/diagnostics.py`

Expected responsibilities:

- `contracts.py`: `FeaturePolicyContract`, adapter manifest payloads, and
  typed summaries
- `runtime.py`: adapter factory, fold-local fit/transform orchestration, and
  identity fallback
- `scaling.py`: global and regime-conditioned scaler-bank fitting with frozen
  statistics
- `masking.py`: per-regime masks, confidence fallback, sparse-regime fallback,
  and generated-column registry
- `diagnostics.py`: policy summaries, row-count diagnostics, and schema
  lineage augmentation

#### Files to modify in Phase 2

- `core/pipeline.py`
- `core/regime_training.py`
- `core/features.py`
- `core/feature_governance.py`
- `core/__init__.py`
- `experiments/config.py`
- training and manifest consumers that summarize fold diagnostics
- feature- and regime-focused tests under `tests/`

`core/automl.py` should only be touched if additive training-summary fields
must be carried through study manifests or promotion reports.

#### Runtime architecture plan

##### Owning adaptation seam

The owning Phase 2 seam is the fold loop in `TrainModelsStep`, immediately
after fold-local regime construction and row alignment, and before supervised
feature selection.

Required fold order:

1. build `fold_frame` from screened base features plus the fold-local
   `regime_frame`
2. align `X_fit`, `X_val`, and `X_test` to valid rows only
3. fit the feature adapter on `X_fit` plus `fit_regime_view` only
4. transform fit, validation, and test frames through the frozen adapter
5. run any post-adaptation stationarity guard on the adapted fit slice only
   and project the frozen adapted schema into validation and test
6. run supervised selection and feature-admission governance on the adapted
   feature frame
7. train the model bundle on the adapted selected columns

The feature-adaptation runtime must not be implemented as a global preview step
that fits on the full sample. The authoritative fit happens inside the
fold-local training loop.

##### Initial policy set

The first rollout should support four policy families behind one adapter
surface:

1. `identity`
   - no-op transform used when `feature_adaptation` is absent or disabled
   - produces a manifest and policy summary so the runtime shape is stable

2. `regime_conditioned` scaling
   - fit a global scaler on `X_fit`
   - fit per-regime scaler banks only for regimes with sufficient support
   - select the per-row scaler using the current regime label only
   - fall back to the global scaler when the row is warm, low-confidence,
     missing, or in a sparse regime

3. `per_regime_mask`
   - derive a fit-time active-column mask per regime using `X_fit` only
   - disable columns that are degenerate, unavailable, or below configured
     support thresholds within that regime
   - preserve one frozen union schema across fit, validation, and test by
     reindexing masked columns to zero rather than dropping them per row

4. bounded interaction expansion
   - migrate the current `build_regime_aware_feature_frame(...)` interaction
     behavior behind the adapter
   - cap interaction generation with the configured `interaction_budget`
   - log generated column lineage explicitly so downstream governance and
     summaries can distinguish raw, scaled, and regime-conditioned features

##### Fallback and confidence rules

The adapter must resolve one explicit fallback decision for every row.

Required fallback inputs:

- `regime`
- `regime_confidence` when available
- `warm`
- configured `min_regime_samples`
- configured fallback mode such as `global`, `identity`, or `disable`

Required fallback behavior:

- warm or unavailable regime rows use the global or identity policy
- low-confidence rows use the configured fallback policy even if a regime label
  exists
- sparse regimes may reuse the global scaler and global mask, but that choice
  must be explicit in the policy metadata
- no fallback path may inspect future regime occupancy or future model
  performance

##### Governance and lineage plan

Phase 2 must extend, not replace, the current feature-governance stack.

Required integration points:

- preserve `feature_blocks`, `feature_families`, and `feature_metadata` for
  adapted columns
- append adaptation lineage such as `regime_scale:<label>`,
  `regime_mask:<label>`, or `regime_interaction:<label>` to the transform
  chain for generated columns
- run `evaluate_feature_admission(...)` on the adapted selected columns rather
  than on the raw screened frame
- keep retirement behavior deterministic when a column is disabled globally by
  the adapter versus retired by governance

##### Compatibility and migration plan

The current repo already contains a partial regime-aware feature path in
`core/regime_training.py`. Phase 2 must unify that code with the new adapter
runtime rather than maintain two feature-conditioning implementations.

Required migration rules:

1. `feature_adaptation` becomes the authoritative config surface for runtime
   feature conditioning.
2. If `feature_adaptation` is absent or disabled, the pipeline runs the
   identity adapter and preserves current behavior.
3. `build_regime_aware_feature_frame(...)` becomes a thin compatibility
   wrapper over the new adapter with a preset that enables bounded regime
   interactions.
4. `train_regime_aware_model(..., strategy="feature")` must use the same
   adapter runtime for both fit and inference paths.
5. Existing `feature_selection` and `feature_governance` config sections stay
   valid and continue to run after adaptation.

The key invariant is that the repo ends Phase 2 with one feature-adaptation
implementation, not separate pipeline and regime-training branches.

##### Diagnostics and artifact plan

Phase 2 must make the adapted feature policy inspectable.

Add these diagnostics to training state and summaries:

- adapter type and policy id
- per-fold disabled-column count
- per-fold generated-column count
- per-fold sparse-regime counts
- per-fold fallback-row counts by reason
- fit-time regime sample counts used to build scaler banks and masks
- adapted schema column list for the selected model input
- policy manifest for the final refit path

Recommended additive payloads:

- `pipeline.state["feature_adaptation"]`
- `training["feature_adaptation"]`
- `post_selection_refit["feature_adaptation"]`
- `training["selection_freeze"]["feature_policy"]` or an equivalent final
  manifest slot

These payloads must be JSON-safe so they can flow into current experiment,
promotion, and artifact summaries.

#### Detailed implementation order inside Phase 2

Implement Phase 2 in this exact sequence:

1. Completed: add feature-adaptation contracts, an identity adapter, and a
  no-op runtime integration seam inside `TrainModelsStep`.
2. Completed on 2026-05-08: implement regime-conditioned scaler-bank fitting,
   explicit fail-closed guards for `model.regime_aware.strategy == "feature"`,
   and fallback inference parity through the stored final adapter.
3. Implement per-regime masks, sparse-regime fallback, and disabled-column
   diagnostics.
4. Migrate bounded regime interactions behind the adapter and retire the
   duplicate feature-conditioning logic in `core/regime_training.py`.
5. Thread manifests and per-fold diagnostics into training, refit, and summary
   payloads.
6. Add config normalization and validation for `feature_adaptation`.
7. Add focused tests for prefix invariance, sparse fallback, compatibility, and
   JSON-safe diagnostics.

This order matters because it lands the integration seam first, then the
stateful transforms, and only then the compatibility migration and artifact
plumbing.

#### Concrete implementation slices for Phase 2

Execute Phase 2 in the following slices. Each slice must end with a narrow test
or replay validation before the next slice begins.

##### Slice 1: Contracts and identity adapter seam

Status:

Completed on 2026-05-08.

Goal:

Add `core/feature_adaptation/` with typed contracts and integrate an identity
adapter into the fold-local training loop without changing current model
behavior.

Implementation scope:

- add `FeaturePolicyContract` and an identity adapter
- add runtime helpers that return unchanged frames plus policy metadata
- integrate the no-op adapter into `TrainModelsStep`
- persist additive per-fold adaptation diagnostics even when the adapter is a
  no-op

Validation slice:

- new focused runtime test for identity adaptation
- one pipeline regression proving no-op equivalence on existing configs

Delivered:

- added `core/feature_adaptation/__init__.py`
- added `core/feature_adaptation/contracts.py`
- added `core/feature_adaptation/runtime.py` with an identity adapter and
  batch split helper
- routed fold-local `X_fit` / `X_val` / `X_test` through the feature-
  adaptation seam in `TrainModelsStep`
- exposed additive JSON-safe feature-adaptation summaries in training state
- added `tests/test_feature_adaptation_runtime.py`
- extended `tests/test_automl_regime_aware_training.py`

##### Slice 2: Regime-conditioned scaling bank

Status:

Completed on 2026-05-08.

Binding decision for this slice:

- Slice 2 must fail closed for `model.regime_aware.strategy == "feature"`
  whenever non-identity scaling is requested, unless the exact same frozen
  scaling policy is also available on the inference path through the same
  adapter runtime.
- The planned implementation for Slice 2 is to block that path explicitly and
  defer feature-bundle integration to Slice 4.

Goal:

Support frozen global and per-regime scaling fitted on `X_fit` only.

Implementation scope:

- implement a first real adapter-backed transform in
  `core/feature_adaptation/scaling.py` behind the Slice 1 runtime seam
- add one mandatory global scaler bank plus optional per-regime scaler banks
  fitted on `X_fit` only
- keep transformed column order identical to the fit-time feature order; this
  slice must not add, drop, or reorder columns
- use `fit_regime_view` only to decide which rows qualify for regime-local
  scaler fitting
- use the current-row regime state only to assign rows to a regime scaler or a
  fallback path at transform time
- emit policy manifests and fold diagnostics showing which scaler banks were
  fitted, which regimes were skipped, and why rows fell back to the global or
  identity policy

Required invariants:

- extending validation or test horizons must not change scaled outputs on an
  already-observed prefix
- the transformed frame must preserve the exact fit-time index, column names,
  column order, and width; Slice 2 is not allowed to widen or shrink schema
- no mean, scale, variance, or support statistic may consume validation or test
  rows
- scaler-bank fitting must use only `X_fit` plus `fit_regime_view` aligned to
  the same index
- transform-time scaler assignment must use only row-local regime information:
  label, confidence, warm/unavailable flags, and configured fallback thresholds
- this slice must not silently train on scaled inputs and infer on unscaled
  inputs for regime-aware feature bundles
- this slice must not introduce masking, interaction generation, or schema
  freeze logic that belongs to Slice 3 and Slice 4

Scaling target selection rules:

- scale continuous numeric columns only
- preserve excluded columns unchanged when they represent discrete regime state
  or binary availability markers, including:
  - `regime`
  - any column ending with `_regime`
  - `warm`
  - `unavailable`
  - `degenerate_fallback`
  - `selected_column_count`
- keep continuous regime diagnostics such as `score`, `regime_confidence`,
  `prob_state_*`, and `log_evidence` eligible for scaling
- record both scaled and excluded column sets in the manifest so later slices
  can reason about them deterministically

Fit-time algorithm plan:

1. Resolve the frozen scaling schema from `X_fit.columns`.
2. Partition columns into:
   - continuous numeric columns eligible for scaling
   - passthrough columns that remain unchanged
3. Fit one mandatory global scaler on eligible `X_fit` columns.
4. Build a regime-eligibility mask from `fit_regime_view` using only fit rows:
   - non-null regime label
   - row is decision-eligible if warm/unavailable flags exist
   - row meets configured confidence floor if `regime_confidence` exists
5. Group eligible rows by regime label and fit a per-regime scaler only when
   the group satisfies `min_regime_samples`.
6. Record skipped regimes with explicit reasons such as:
   - `insufficient_rows`
   - `missing_regime_labels`
   - `confidence_below_floor`
   - `no_eligible_scaling_columns`
7. Freeze the bank using fit-time column order and per-regime sample counts.

Transform-time algorithm plan:

1. Reindex incoming frames to the frozen fit-time column order.
2. Compute the global-scaled frame once for all eligible columns.
3. If `scaling.mode != regime_conditioned`, return the global-scaled frame.
4. If `scaling.mode == regime_conditioned`, derive one scaler assignment per
   row using only the current row’s regime payload.
5. For rows assigned to a fitted regime bank, overwrite the global-scaled
   values with the regime-scaled values for the eligible columns.
6. Leave excluded passthrough columns unchanged.
7. Emit per-row fallback counts by reason:
   - `missing_regime`
   - `warm_or_unavailable`
   - `confidence_below_floor`
   - `sparse_regime`
   - `missing_regime_bank`
   - `identity_fallback_configured`

Scaler-bank policy decisions for the first rollout:

- use a standard z-score bank for Slice 2: fit mean and standard deviation on
  `X_fit` only
- do not add robust scaling, EWM scaling, whitening, PCA, or mixed scaler-bank
  families in Slice 2
- treat zero-variance columns as scale `1.0` and record them explicitly as
  constant columns in the manifest
- fit statistics on finite values only; any missing values encountered during
  transform should map to a zero-centered standardized output where possible so
  the transformed frame remains model-safe
- keep raw scaler arrays in the runtime object, but expose only JSON-safe
  summarized manifests in pipeline state and training summaries

Fallback policy plan:

- always fit a global scaler bank even when regime-conditioned scaling is
  enabled; global is the default safe fallback
- support `fallback: global` as the canonical Slice 2 fallback mode
- preserve `fallback: identity` as an explicit opt-out path for rows that must
  not consume the global bank
- derive `min_regime_samples` using the narrowest available config in this
  order:
  1. `feature_adaptation.scaling.min_regime_samples`
  2. `feature_adaptation.selection.min_regime_samples`
  3. `model.regime_aware.min_samples_per_regime`
  4. default `40`
- if no per-regime bank exists for a row’s label, fall back deterministically;
  never approximate by borrowing another regime’s scaler

Hidden dependency that must be resolved before implementation:

`RegimeAwareModelBundle` still owns prediction-time feature transformation for
`strategy="feature"` in `core/regime_training.py`. Slice 2 must not create a
train/inference mismatch where the pipeline trains on scaled features but the
bundle predicts on unscaled features.

Resolution for Slice 2:

- do not thread the frozen scaling bank into the regime-aware feature bundle in
  this slice
- reject non-identity scaling for `model.regime_aware.strategy == "feature"`
  with a precise config/runtime error that explains the path is deferred until
  Slice 4 runtime unification

The unsafe option is to scale only inside the training fold loop and leave the
bundle inference path untouched. That option remains forbidden.

Recommended file-level changes:

- `core/feature_adaptation/scaling.py`
  - add frozen scaler-stat dataclasses or equivalent private structs
  - add pure fit/transform helpers for global and per-regime banks
  - add column-resolution helpers for eligible versus passthrough columns
- `core/feature_adaptation/runtime.py`
  - extend `build_feature_adapter(...)` to return a scaling adapter when
    `scaling.mode == regime_conditioned`
  - keep Slice 1 deferred markers for selection, masking, and interaction
    sections not implemented yet
- `core/feature_adaptation/contracts.py`
  - extend policy metadata only if needed for scaler-bank summaries
  - avoid widening the public contract surface unless the scaling manifest
    genuinely requires it
- `core/pipeline.py`
  - preserve the same fold-local integration seam added in Slice 1
  - add fold diagnostics for fitted banks, fallback reasons, constant columns,
    and excluded scaling columns
- `core/regime_training.py`
  - only if required to enforce the fail-closed guard for
    `strategy="feature"`
  - do not add a scaling bridge here in Slice 2
- tests:
  - add `tests/test_feature_adaptation_scaling.py`
  - extend `tests/test_feature_adaptation_runtime.py`
  - extend `tests/test_automl_regime_aware_training.py`
  - add a focused config/runtime guard test for the blocked feature-strategy
    scaling path

Validation slice:

- prefix-invariance test for transformed prefixes
- no-lookahead test proving validation and test rows do not alter fit-time
  scaling statistics
- deterministic fallback test for sparse and low-confidence regimes
- fail-closed test proving non-identity scaling is rejected for
  `strategy="feature"` in Slice 2

Minimum validation behaviors:

- replaying the same fit prefix twice yields identical global and regime-bank
  manifests
- extending the dataset beyond a cutoff does not change scaled outputs on the
  already-observed prefix
- no validation or test row can change fit-time means or scales
- excluded regime label columns remain unchanged after transformation
- zero-variance columns do not produce NaN or inf outputs
- rows without a fitted regime bank fall back deterministically and record the
  exact reason
- if `strategy="feature"` requests non-identity scaling, the config/runtime is
  blocked explicitly with a deterministic error

Suggested narrow validation commands while implementing Slice 2:

- `g:/N/repos/ml/.venv/Scripts/python.exe -m pytest tests/test_feature_adaptation_scaling.py`
- `g:/N/repos/ml/.venv/Scripts/python.exe -m pytest tests/test_feature_adaptation_runtime.py tests/test_feature_adaptation_scaling.py`
- `g:/N/repos/ml/.venv/Scripts/python.exe -m pytest tests/test_feature_adaptation_runtime.py tests/test_feature_adaptation_scaling.py tests/test_automl_regime_aware_training.py`

Acceptance criteria:

- the repo contains a frozen global/per-regime scaling bank that fits on
  `X_fit` only and transforms validation/test rows without mutating schema
- transform-time scaler assignment uses only row-local regime state and frozen
  fit-time banks
- training summaries expose JSON-safe scaler-bank diagnostics and fallback
  counts
- continuous regime diagnostics can be scaled while discrete regime-state
  columns remain unchanged
- prefix invariance and no-lookahead behavior are covered by focused tests
- the slice resolves the regime-aware feature-bundle train/inference symmetry
  problem explicitly by blocking the unsafe feature-strategy scaling path
- no config path can train on scaled inputs and infer on unscaled inputs after
  Slice 2 lands

Explicit non-goals for Slice 2:

- no per-regime masking yet
- no disable-incompatible-features behavior yet
- no interaction generation migration yet
- no schema-width changes or new generated columns
- no bridge of non-identity scaling into `RegimeAwareModelBundle` yet
- no specialist-library or router integration

##### Slice 3: Per-regime masking and sparse-support fallback

Status:

Completed on 2026-05-08. Delivered frozen per-regime masks, a composite
`scale -> mask` adapter runtime, fold-local disabled-column gating, and
fallback-inference reuse through the stored adapter. Conservative scope held:
no interaction generation, config normalization, or
`core/regime_training.py` feature-bundle migration in this slice.

Binding decisions for this slice:

- Slice 3 must preserve the Slice 2 fail-closed rule for
  `model.regime_aware.strategy == "feature"` whenever any non-identity
  `selection.mode` or `disable_incompatible_features` behavior is requested.
  The authoritative feature-bundle inference reuse path still lands in Slice
  4.
- Slice 3 must reuse the same stored adapter object on the fallback inference
  path. Do not introduce a second masking code path in `SignalsStep`.
- Slice 3 must keep transformed frames schema-preserving. Masking is zero-fill
  only; no per-row drop, reorder, or width change is allowed.
- `disabled_columns` in this slice are fold-local policy-disabled columns, not
  retired features. Slice 3 must not mutate retirement registries or rewrite
  global `pipeline.state["feature_metadata"]` to represent adaptation-level
  disables.

Goal:

Add frozen per-regime active-column masks, deterministic sparse-support
fallback, and explicit policy-disabled column semantics on top of the Slice 2
scaling runtime.

Owning architecture decision:

- `build_feature_adapter(...)` should evolve from a single-stage selector into
  an ordered composite adapter surface.
- Required stage order for Slice 3:
  1. identity or scaling stage
  2. masking stage
- The composite adapter must still expose one `fit(...)`, one `transform(...)`,
  and one JSON-safe manifest so `TrainModelsStep` and `SignalsStep` fallback
  continue to reuse exactly one authoritative runtime object.

Implementation scope:

- add `core/feature_adaptation/masking.py`
- fit a mandatory global mask plus optional per-regime masks on post-scaling
  `X_fit` only
- derive maskable versus passthrough columns using the frozen fit-time schema,
  `fit_regime_view`, and fit-local feature metadata
- support `selection.mode: per_regime_mask` with `fallback: global` as the
  canonical fallback and `fallback: identity` as an explicit opt-out
- support `disable_incompatible_features: true` by deriving fold-local
  globally disabled columns and excluding them from downstream model-input
  candidate columns
- preserve the frozen union schema in transformed frames by zero-filling
  masked or disabled columns rather than dropping them
- emit policy metadata and fold summaries for:
  - active columns by regime and globally
  - disabled columns and disable reasons
  - sparse-regime and low-confidence fallback counts
  - masked-row and masked-cell totals

Required invariants:

- mask fitting may inspect only the post-scaling adapted `X_fit`,
  `fit_regime_view`, and feature metadata derived from the same fold
- no support, variance, activity, or disabled-column statistic may consume
  validation or test rows
- extending validation or test horizons must not change masked outputs on an
  already-observed prefix
- all columns present in `fit_regime_view` must remain passthrough and
  unmasked
- any column listed in `policy.disabled_columns` must remain present in the
  transformed frame but must be excluded from downstream model candidate
  columns
- when feature selection is disabled, policy-disabled columns must still be
  removed from `selected_columns` before model fit
- sparse, missing, low-confidence, warm, or unavailable rows must follow one
  deterministic fallback mask; no borrowing of another regime’s mask is
  allowed
- Slice 3 must not introduce interaction generation, post-adaptation schema
  widening, or feature-bundle runtime migration

Mask candidate resolution rules:

- start from the frozen post-scaling fit schema
- passthrough columns for Slice 3 are:
  - every column present in `fit_regime_view`
  - `regime`
  - any column ending with `_regime`
  - `warm`
  - `unavailable`
  - `degenerate_fallback`
  - `selected_column_count`
  - any non-numeric column
- mask candidates are numeric columns in the frozen schema not in the
  passthrough set
- already-retired features are out of scope because they should already be
  absent before adaptation runs
- `disable_incompatible_features` may apply only to mask candidates;
  passthrough columns can never be policy-disabled in Slice 3

Fit-time algorithm plan:

1. Run Stage 1 identity or scaling on `X_fit` using the Slice 2 runtime.
2. Freeze the union schema from the Stage 1 output column order.
3. Partition Stage 1 columns into passthrough versus mask candidates.
4. Compute a global support summary on mask candidates using fit rows only:
   - finite row count
   - active row count using `abs(value) > activity_epsilon`
   - active share
   - standard deviation or variance
5. Derive a global active mask using deterministic thresholds:
   - `finite_count >= min_feature_rows`
   - `active_share >= min_active_share`
   - `std > min_variance`
6. Build the regime-eligibility mask from `fit_regime_view` using the same
   fit-only gates as Slice 2 scaling:
   - non-null regime label
   - warm or unavailable eligibility
   - confidence floor when `regime_confidence` exists
7. For each eligible regime with at least `min_regime_samples`, fit a
   per-regime active mask over the globally active candidate columns using the
   same support thresholds.
8. Record skipped regimes with explicit reasons:
   - `insufficient_rows`
   - `missing_regime_labels`
   - `confidence_below_floor`
   - `no_global_active_columns`
9. If `disable_incompatible_features` is enabled, derive global disabled
   columns and reasons:
   - `below_global_support_threshold`
   - `globally_constant`
   - `inactive_in_all_supported_regimes`
10. Freeze the mask bank with:
    - union schema
    - global active columns
    - per-regime active columns
    - globally disabled columns and reasons
    - thresholds and regime sample counts

Transform-time algorithm plan:

1. Reindex incoming frames to the frozen post-scaling union schema.
2. Run Stage 1 identity or scaling first.
3. Apply the global disable pass, zero-filling `policy.disabled_columns` for
   all rows when enabled.
4. If `selection.mode != per_regime_mask`, return the globally-disabled Stage
   1 frame plus policy metadata.
5. If `selection.mode == per_regime_mask`, resolve one mask assignment per row
   using only:
   - current-row regime label
   - `warm`
   - `unavailable`
   - `regime_confidence`
   - frozen `min_regime_samples` and confidence thresholds
6. For rows assigned to a fitted regime mask, zero-fill candidate columns not
   active in that regime.
7. For fallback rows, apply either:
   - `fallback: global` -> zero-fill columns not in the frozen global active
     mask
   - `fallback: identity` -> keep Stage 1 values for non-disabled candidate
     columns
8. Preserve passthrough columns unchanged for all rows.
9. Emit diagnostics for:
   - fallback rows by reason
   - mask assignment counts by regime
   - total masked cells
   - disabled columns by reason
   - active candidate counts by regime and globally

Mask policy decisions for the first rollout:

- canonical `selection.mode` values for Slice 3:
  - `identity`
  - `per_regime_mask`
- canonical `selection.fallback` values for Slice 3:
  - `global`
  - `identity`
- do not add soft or probabilistic masks, top-k masks, weighted gating, or
  learned per-regime selectors in Slice 3
- compute support using deterministic thresholds, not supervised feature
  importance or downstream model scores
- keep support thresholds configurable but narrow:
  - `selection.min_feature_rows`
  - `selection.min_active_share`
  - `selection.min_variance`
  - `selection.activity_epsilon`
  - `selection.min_regime_samples`
- resolve `selection.min_regime_samples` using the same precedence order as
  Slice 2 scaling
- preserve `disable_incompatible_features` as a boolean only; do not add a
  semantic incompatibility DSL in Slice 3

Compatibility and governance rules:

- adaptation-disabled columns must remain distinct from governance-retired
  columns
- `AlignDataStep` retirement remains the only place that globally removes
  retired features from `pipeline.state["X"]`
- `TrainModelsStep` should seed candidate `selected_columns` from
  `fit_policy.feature_columns - fit_policy.disabled_columns`
- `feature_selection` and `evaluate_feature_admission(...)` must run on the
  adaptation-eligible candidate set, not on policy-disabled columns
- training summaries must report both:
  - `feature_adaptation.disabled_columns`
  - `feature_governance.retired_columns` and rejected columns
  without conflating them

Hidden dependencies that must be resolved before implementation:

1. Slice 2 persisted the final adapter and reuses it in `SignalsStep`
   fallback. Slice 3 must extend that same runtime object instead of
   introducing a masking-only inference shim.
2. `model.regime_aware.strategy == "feature"` still owns a separate
   feature-bundle inference path in `RegimeAwareModelBundle`. Non-identity
   masking would recreate the same train/inference mismatch that Slice 2
   blocked for scaling.

Resolution for Slice 3:

- continue to reject `selection.mode != identity` or
  `disable_incompatible_features: true` for `strategy="feature"`
- limit Slice 3 runtime enablement to non-regime-aware models and
  `strategy="specialist"` until Slice 4 unifies the feature-bundle path

Recommended file-level changes:

- `core/feature_adaptation/masking.py`
  - add frozen mask-bank dataclasses and pure fit/transform helpers
  - add candidate-resolution helpers for passthrough versus maskable columns
  - add disabled-column reason summarizers
- `core/feature_adaptation/runtime.py`
  - introduce a staged or composite adapter returned by
    `build_feature_adapter(...)`
  - preserve current identity and scaling manifests while adding stage-level
    mask metadata
  - extend runtime support validation to fail closed for feature-strategy
    masking or disable requests
- `core/feature_adaptation/contracts.py`
  - keep the public contract unchanged unless a new top-level field is truly
    unavoidable
  - prefer storing mask diagnostics inside `FeaturePolicyContract.metadata`
- `core/pipeline.py`
  - preserve the existing fold-local adaptation seam
  - exclude policy-disabled columns from downstream model candidate selection
    even when feature selection is disabled
  - surface mask-bank diagnostics in fold and final summaries
- `core/regime_training.py`
  - only to extend the fail-closed guard for feature-strategy masking or
    disable requests
  - do not migrate feature-bundle runtime here in Slice 3
- tests:
  - add `tests/test_feature_adaptation_masking.py`
  - extend `tests/test_feature_adaptation_runtime.py`
  - extend `tests/test_automl_regime_aware_training.py`
  - extend `tests/test_regime_leakage_controls.py`
  - optionally extend `tests/test_feature_admission_policy.py` for
    disabled-versus-retired separation

Validation slice:

- deterministic schema-freeze test proving fit, validation, and test frames
  keep identical column order and width under masking
- prefix-invariance test for masked prefixes
- sparse-regime fallback test proving global or identity mask fallback is
  deterministic
- passthrough-column test proving regime-state columns remain unchanged
- policy-disabled-column test proving disabled columns remain present in
  transformed frames but are excluded from model candidate columns
- fail-closed test for `strategy="feature"` masking or disable requests
- fallback inference test proving `SignalsStep` reapplies the final mask
  adapter on post-training rows

Minimum validation behaviors:

- extending the dataset beyond a cutoff does not change masked outputs on the
  already-observed prefix
- no validation or test row can change fit-time active column sets
- sparse regimes do not borrow another regime’s mask
- passthrough regime-state columns remain unchanged after transform
- globally disabled columns appear in the policy manifest with explicit
  reasons
- feature selection disabled still prevents policy-disabled columns from
  entering model fit
- fallback inference uses the same mask policy as training for the final
  stored adapter
- `strategy="feature"` requests non-identity masking or disable flags are
  blocked explicitly

Suggested narrow validation commands while implementing Slice 3:

- `g:/N/repos/ml/.venv/Scripts/python.exe -m pytest tests/test_feature_adaptation_masking.py`
- `g:/N/repos/ml/.venv/Scripts/python.exe -m pytest tests/test_feature_adaptation_runtime.py tests/test_feature_adaptation_masking.py`
- `g:/N/repos/ml/.venv/Scripts/python.exe -m pytest tests/test_regime_leakage_controls.py::RegimeLeakageControlsTest::test_signals_fallback_reapplies_final_feature_adapter tests/test_feature_adaptation_masking.py`
- `g:/N/repos/ml/.venv/Scripts/python.exe -m pytest tests/test_feature_adaptation_runtime.py tests/test_feature_adaptation_masking.py tests/test_automl_regime_aware_training.py`

Acceptance criteria:

- the runtime can fit one authoritative `scale -> mask` adapter on `X_fit`
  only and reuse it unchanged for validation, test, and fallback inference
- per-regime masks and globally disabled columns are persisted in JSON-safe
  manifests and fold summaries
- transformed frames preserve the frozen union schema while rows receive
  deterministic zero-filled masks
- sparse, missing, warm, unavailable, and low-confidence rows follow an
  explicit fallback mask policy
- policy-disabled columns are excluded from model candidate inputs without
  being misreported as retired features
- no config path can train with masked inputs and infer with unmasked inputs
  after Slice 3 lands

Explicit non-goals for Slice 3:

- no bounded interaction generation yet
- no feature-bundle runtime migration yet
- no semantic incompatibility DSL
- no learned or soft masks
- no config normalization beyond narrow runtime validation
- no specialist-library or router work

##### Slice 4: Interaction budget and regime-training migration

Status:

Completed on 2026-05-09. Delivered a dedicated feature-strategy adapter in
`core/feature_adaptation/feature_strategy.py`, routed
`build_regime_aware_feature_frame(...)` through that adapter, reused the same
frozen adapter in `RegimeAwareModelBundle` inference, and threaded resolved
feature-adaptation config into both primary and inner regime-aware training
paths. Scope stayed narrow: no pre-selection pipeline semantics were widened.

Goal:

Move the existing regime-aware feature engineering helper behind the new
adapter runtime and eliminate duplicate feature-conditioning logic.

Implementation scope:

- route `build_regime_aware_feature_frame(...)` through the adapter
- support bounded interaction generation using `interaction_budget`
- reuse the same adapter in `RegimeAwareModelBundle` fit and inference paths
- preserve current `strategy="feature"` behavior while making the runtime
  authoritative

Validation slice:

- `tests/test_regime_aware_training.py`
- focused adapter test for generated-column lineage and interaction caps

##### Slice 5: Diagnostics, manifests, and refit propagation

Status:

Completed on 2026-05-09. Delivered feature-adaptation config validation,
AutoML training-summary propagation, explicit post-selection refit surfacing,
and user-facing policy summaries in `experiments/runner.py` and
`example_utils.py`.

Goal:

Persist adaptation diagnostics through training summaries, post-selection
refits, and user-facing experiment artifacts.

Implementation scope:

- add JSON-safe fold diagnostics and final manifests
- carry feature-adaptation summaries through training and refit artifacts
- add config validation for `feature_adaptation`
- ensure examples and experiment summaries can show the active policy

Validation slice:

- manifest serialization tests
- training-summary regression tests
- focused config-compatibility coverage

#### Concrete validation matrix for Phase 2

At minimum, each slice should validate one of these behaviors directly:

1. disabled or absent `feature_adaptation` yields a no-op transform with stable
   manifests
2. extending the dataset does not change adapted outputs on an already-observed
   prefix
3. no adaptation statistic is fit on validation or test rows
4. sparse-regime rows fall back deterministically and record the reason
5. masked columns do not change output schema across fit, validation, and test
6. generated interaction columns respect the configured budget
7. regime-aware feature strategy and pipeline training share the same adapter
   runtime

#### Test plan for Phase 2

Add these focused tests:

- `tests/test_feature_adaptation_runtime.py`
- `tests/test_feature_adaptation_scaling.py`
- `tests/test_feature_adaptation_masking.py`

Extend these existing suites:

- `tests/test_regime_aware_training.py`
- `tests/test_feature_admission_policy.py`
- `tests/test_experiment_config.py`
- any training-summary or experiment-manifest slice that asserts JSON-safe
  payloads

Minimum behavior each Phase 2 test group must cover:

- identity adapter preserves current behavior when the feature-adaptation layer
  is disabled
- regime-conditioned scaling is prefix-invariant and fold-local
- per-regime masks do not leak future occupancy
- generated columns preserve explicit lineage and deterministic ordering
- low-confidence and warm rows fall back to the configured policy
- training and inference paths use the same adapter manifest
- diagnostics remain JSON-safe and additive in training output

#### Risks and failure modes to prevent in Phase 2

- fitting scaler banks or masks on the combined fit-plus-test window
- allowing adapted schemas to drift between fit and inference because masks are
  applied by dropping columns instead of reindexing a frozen union schema
- generating regime interactions outside the configured budget and silently
  exploding feature width
- reintroducing a second feature-conditioning implementation in
  `core/regime_training.py`
- treating low-confidence rows as fully regime-qualified instead of invoking
  fallback policy
- bypassing stationarity discipline for newly generated columns

The main guardrail is to make one adapter runtime own every regime-conditioned
feature transform, then reuse that runtime everywhere else.

#### Phase 2 acceptance criteria

Phase 2 is complete only when all of these are true:

- the repo contains a typed feature-adaptation runtime with identity,
  regime-conditioned scaling, masking, and bounded interaction support
- fold-local training fits the adapter on `X_fit` only and reuses frozen policy
  artifacts for validation and test
- `build_regime_aware_feature_frame(...)` and the pipeline share the same
  authoritative adapter implementation
- low-confidence, warm, and sparse-regime fallbacks are explicit and recorded
- training and refit artifacts persist JSON-safe feature-policy manifests and
  diagnostics
- adapted feature schemas are deterministic and prefix-invariant
- newly generated columns do not bypass stationarity and governance checks

Explicit non-goals for Phase 2:

- no specialist-library certification or routing logic yet
- no router-aware switching-cost replay
- no detector-fusion redesign beyond consuming the existing regime trace
- no broad AutoML search redesign beyond carrying the new diagnostics forward

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