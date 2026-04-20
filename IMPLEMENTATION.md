# P0 Implementation Plan

## Purpose

This document turns the P0 findings from IMPROVEMENTS.md into an implementation plan for the repository.

The plan is designed around the following user decisions:

- Write the plan in IMPLEMENTATION.md only.
- Use a staged execution-research hardening path before full NautilusTrader dependency.
- Use a validation-library-backed data contract design.
- Include both spot and futures in first-wave cross-venue validation.
- Aim for institutional standards for promotion governance.

For this plan, "institutional standards" means the end-state is fail-closed promotion governance. A candidate may be trained, evaluated, and stored, but it is not eligible for champion promotion if mandatory integrity or replication gates fail. Because several new gates do not yet have baseline thresholds in this repository, the rollout should include a short calibration period where distributions are measured and thresholds are frozen before hard enforcement is enabled.

## End-State Requirements

Champion promotion must require all of the following:

- Data contract pass for every input dataset used in training, validation, holdout, and backtest.
- Feature admission pass.
- Feature portability pass.
- Regime stability pass.
- Operational health pass.
- Cross-venue integrity pass.
- Signal half-life and delay compatibility pass.
- Replication cohort pass across non-primary out-of-sample cohorts.
- Existing post-selection and overfitting controls remain in force.

The system must fail closed on unknown schema drift, missing provenance, missing reference validation where required, and promotion-metric mismatches.

## Recommended Technical Defaults

### Data contracts

- Use Pandera as the first validation layer for tabular dataset contracts.
- Keep repository-owned manifest and lineage logic in plain Python dataclasses or typed dicts rather than outsourcing provenance to a separate platform.
- Persist contract hash, schema version, semantic version, source fingerprint, availability timestamp policy, and validation result with every derived artifact.

### Cross-venue validation

- Spot first-wave reference venues: Binance plus Coinbase and Kraken.
- Futures first-wave reference venues: Binance futures plus Bybit and OKX, with explicit handling for mark price, index price, and basis.
- Treat reference data as validation and overlay infrastructure, not as a trading venue abstraction.

### Signal decay

- Default measurement should be edge decay by forward horizon and by execution delay.
- Report both gross and net decay curves.
- Gate promotion on whether usable edge survives the configured execution delay and minimum holding horizon.

### Execution realism

- Preserve the current backtest engine as the default research engine.
- Harden it with event-time semantics, queue heuristics, latency, and stricter fill-state modeling.
- Preserve the Nautilus adapter boundary so the repository can later swap promotion-grade execution to Nautilus-backed runs without rewriting strategy logic.

### Replication governance

- Promotion should not rely on a single symbol and timeframe holdout alone.
- Require a configurable replication cohort spanning other symbols, adjacent regimes, and held-out historical windows.
- Persist the full replication matrix in the registry for every promotable artifact.

## Concrete Repo Issues

## Issue P0-1: Governance Gates Are Computed But Not Binding

### Problem

The pipeline already computes important promotion diagnostics, but the final promotion path does not consistently bind them into a single fail-closed eligibility decision. This creates a false-confidence path where the repository appears to enforce governance more strongly than it actually does.

### Current touchpoints

- core/pipeline.py
- core/feature_governance.py
- core/regime.py
- core/monitoring.py
- core/automl.py
- core/registry/store.py
- tests/test_feature_admission_policy.py
- tests/test_feature_portability_gates.py
- tests/test_regime_layer_ablation.py
- tests/test_operations_monitoring.py

### Target state

There is one canonical promotion-eligibility object with:

- Gate name
- Status
- Thresholds
- Measured value
- Blocking or advisory severity
- Human-readable rejection reason

core/automl.py and core/registry/store.py must consume the same object and make the same promotion decision from it.

### Implementation tasks

1. Define a canonical PromotionEligibilityReport schema and use it everywhere.
2. Bind feature_portability, feature_admission, regime_stability, and operational_health as promotion blockers.
3. Remove any mismatched comparison path where challenger selection metrics and champion holdout metrics are compared on different bases.
4. Persist rejection reasons into manifests and registry summaries.
5. Add config-driven thresholds and explicit advisory-versus-blocking modes.
6. Add a temporary calibration mode that logs values without promotion until thresholds are frozen.

### Acceptance criteria

- A candidate that fails portability, regime stability, or operational health cannot be promoted.
- The same candidate receives the same decision in automl selection and registry promotion.
- Rejection reasons are persisted and visible in training artifacts.
- No promotion path bypasses the canonical gate object.

## Issue P0-2: Data Layer Lacks Contracts, Lineage, And Fail-Closed Drift Handling

### Problem

The repository has useful data quality checks, but it does not yet behave like a data-as-code system. There is no first-class contract layer that declares what a dataset is allowed to look like, what semantics its columns carry, and which upstream changes are breaking versus tolerable.

### Current touchpoints

- core/data.py
- core/data_quality.py
- core/storage.py
- core/context.py
- core/reference_data.py
- tests/test_data_fetch_integrity.py
- tests/test_data_quality_quarantine.py
- tests/test_safe_artifact_storage.py

### Target state

Every input dataset has:

- A contract with required columns, dtypes, nullability, allowed duplicates policy, timezone policy, semantic field definitions, and availability timestamp policy.
- A source manifest with source name, endpoint, extraction window, checksum or fingerprint, schema version, and contract hash.
- A validation result that can hard-stop downstream training or backtesting.

### Implementation tasks

1. Introduce dataset contracts using Pandera schemas for:
   - OHLCV spot bars
   - Futures bars and funding context
   - Custom tabular data
   - Reference overlay data
2. Introduce a repository-owned DatasetManifest structure capturing lineage and source metadata.
3. Hash the active contract and persist it with all derived artifacts and registry versions.
4. Split data validation outcomes into pass, quarantine, and fail-closed categories.
5. Promote unknown schema drift, timezone drift, and missing availability metadata from warnings to blocking failures.
6. Ensure custom data joins require explicit availability timestamps and point-in-time-safe alignment.
7. Extend storage artifacts so raw inputs, validation reports, and contract fingerprints are reproducible.

### Acceptance criteria

- Any upstream schema change that affects required columns or semantics blocks training until contracts are updated.
- All artifacts can identify the exact input contract and source fingerprint used.
- Custom data without availability timestamps cannot silently enter model features.
- Data quality quarantine can block rather than merely annotate runs.

## Issue P0-3: Single-Venue Truth Is Treated As Market Truth

### Problem

The repository still assumes Binance is the effective truth source for both spot and futures. That is not adequate for a research stack whose conclusions are supposed to survive venue anomalies, exchange outages, wick errors, stale feeds, or venue-local microstructure pathologies.

### Current touchpoints

- core/data.py
- core/context.py
- core/reference_data.py
- core/data_quality.py
- core/scenarios.py
- tests/test_exchange_failure_scenarios.py
- tests/test_historical_universe_selection.py
- tests/test_derivatives_context_pipeline.py

### Target state

The system maintains a venue-normalized reference layer used for:

- Data sanity checks
- Feature overlays
- Futures mark and basis validation
- Venue divergence diagnostics
- Promotion gating where coverage is sufficient

### Implementation tasks

1. Introduce a ReferenceConnector abstraction that fetches normalized reference data without changing the existing trading venue abstractions.
2. Build canonical spot reference snapshots for overlapping symbols from Binance, Coinbase, and Kraken.
3. Build futures reference snapshots and diagnostics for Binance futures, Bybit, and OKX with explicit mark-price, index-price, and basis fields.
4. Add symbol mapping and contract metadata normalization for reference venues.
5. Extend data-quality checks with:
   - Sustained venue divergence detection
   - Stale-reference detection
   - Exchange outage or freeze detection
   - Futures mark-versus-trade divergence checks
6. Feed the validated composite reference layer into existing reference overlay feature plumbing.
7. Make cross-venue integrity a promotion gate once baseline thresholds are calibrated.

### Acceptance criteria

- Single-venue anomalies can be identified against reference venues rather than treated as market truth.
- Reference coverage gaps are explicit and cannot silently masquerade as clean data.
- Futures validation distinguishes trade price, mark price, index price, and basis.
- Promotion can be blocked on severe reference divergence when configured coverage is present.

## Issue P0-4: Signal Half-Life And Delay Compatibility Are Not Measured

### Problem

The system can produce signals and backtests, but it does not yet quantify how quickly a signal decays or whether the edge still exists after the execution delay implied by the backtest and sizing assumptions. This is a major gap because a signal that dies in a few bars cannot be evaluated the same way as a slower-moving signal.

### Current touchpoints

- core/models.py
- core/backtest.py
- core/pipeline.py
- core/monitoring.py
- tests/test_signal_profitability_sizing.py

### Target state

Every trained signal has a decay report that includes:

- Gross edge by forward horizon
- Net edge by forward horizon after costs
- Edge by execution delay
- Estimated half-life or effective decay horizon
- Regime-specific decay summary where sample size permits

### Implementation tasks

1. Add a signal-decay computation module that evaluates predictions across forward horizons.
2. Measure decay separately for:
   - Raw classifier scores or probabilities
   - Thresholded trade signals
   - Net realized trade outcomes
3. Compute delay compatibility against configured signal_delay_bars and order-aging assumptions.
4. Persist signal half-life and edge-at-delay in training summaries, backtest summaries, and registry manifests.
5. Add promotion thresholds such as:
   - Minimum net edge at expected delay
   - Minimum effective half-life relative to intended holding period
6. Extend monitoring to detect live or rolling-walk-forward decay deterioration.

### Acceptance criteria

- Every promotable model has a stored decay report.
- A signal whose edge disappears before the configured execution delay cannot be promoted.
- Backtest outputs distinguish gross edge and net usable edge.
- Delay compatibility is tied to execution assumptions instead of being a separate vanity metric.

## Issue P0-5: Execution Realism Is Better Than Before But Still Surrogate

### Problem

The current execution path is directionally better than naive full-fill backtesting, but it is still mostly a deterministic bar-volume surrogate rather than a sufficiently realistic event-driven execution model. Without harder execution semantics, short-horizon performance can remain overstated.

### Current touchpoints

- core/backtest.py
- core/execution/intents.py
- core/execution/policies.py
- core/execution/liquidity.py
- core/execution/costs.py
- core/execution/nautilus_adapter.py
- core/slippage.py
- tests/test_execution_adapter_parity.py
- tests/test_execution_partial_fills.py
- tests/test_microstructure_cost_models.py
- tests/test_causal_liquidity_inputs.py

### Target state

The default research engine preserves current ergonomics but uses stricter execution semantics:

- Explicit order lifecycle
- Submission and cancel latency
- Queue-position heuristic for passive fills
- Size-aware partial fills
- Price protection and venue constraints
- No same-bar execution shortcuts that leak unavailable information

### Implementation tasks

1. Refactor the backtest execution core around an explicit order-state machine.
2. Add configurable submission latency and cancel latency.
3. Add probabilistic or heuristic queue-position modeling for passive fills.
4. Add data-granularity-aware fill models so bar-only inputs are explicitly conservative.
5. Tighten bar timestamp and bar-processing semantics to prevent look-ahead-like execution artifacts.
6. Unify slippage, spread, participation, and liquidity-consumption logic under a single execution policy contract.
7. Ensure futures execution remains consistent with funding, mark-price, liquidation, and margin assumptions.
8. Preserve the Nautilus adapter boundary and define a future promotion-grade execution hook, but do not require immediate Nautilus adoption.

### Acceptance criteria

- Partial fills, queue effects, age-outs, and delayed commands are represented in the default engine.
- Execution metrics clearly expose fill ratio, residual quantity, cancellation reason, and average delay.
- Tight-stop and short-horizon strategies degrade appropriately under stricter execution.
- Existing parity tests for abundant-liquidity cases still pass where intended.

## Issue P0-6: Out-Of-Sample Governance Is Too Local

### Problem

The repository has useful holdout discipline, but promotion remains too dependent on success in the primary local study. That is not enough to defend against a lucky symbol, timeframe, or period-specific effect.

### Current touchpoints

- core/automl.py
- core/pipeline.py
- core/registry/store.py
- core/registry/manifest.py
- core/universe.py
- tests/test_automl_holdout_objective.py
- tests/test_post_selection_inference.py
- tests/test_local_registry_flow.py

### Target state

Promotion is based on a replication cohort, not just the primary target. The cohort should include adjacent but non-identical out-of-sample tests such as:

- Same signal family on other symbols
- Shared timeframe analysis on non-primary symbols
- Different historical windows
- Regime-specific slices

### Implementation tasks

1. Add a ReplicationValidator stage after primary candidate selection.
2. Define a configuration for replication cohorts by symbol, timeframe, regime, and period.
3. Compute cohort-level summaries such as pass rate, median performance, tail performance, and consistency after costs.
4. Persist replication results in registry manifests and promotion reports.
5. Require minimum cohort coverage before a model is promotion-eligible.
6. Prevent a candidate from being promoted solely because one symbol and period look attractive.

### Acceptance criteria

- A candidate can fail promotion even when its primary holdout looks good if replication is weak.
- Registry versions store full replication evidence.
- Promotion decisions are explainable in terms of primary and replication outcomes.
- The validation layer remains time-aware and leakage-safe.

## Delivery Sequence

The implementation should be delivered in the following order.

## Phase 0: Policy Freeze And Baseline Calibration

### Goal

Define the governance objects, thresholds, and artifact schemas before hard behavioral changes start.

### Work

1. Freeze canonical schemas for:
   - PromotionEligibilityReport
   - DatasetManifest
   - CrossVenueIntegrityReport
   - SignalDecayReport
   - ReplicationReport
2. Add config surfaces for gate modes, reference venues, decay thresholds, and replication cohorts.
3. Decide which thresholds need calibration from observed history rather than immediate hard-coding.

### Deliverables

- Schema definitions
- Config defaults
- Migration notes for existing artifacts

## Phase 1: Bind Existing Governance Gates

### Why first

This closes the largest false-confidence gap with the least architectural risk and leverages code the repository already has.

### Main files

- core/automl.py
- core/pipeline.py
- core/registry/store.py

### Tests

- Extend tests/test_feature_admission_policy.py
- Extend tests/test_feature_portability_gates.py
- Extend tests/test_regime_layer_ablation.py
- Extend tests/test_operations_monitoring.py
- Add tests/test_promotion_gate_binding.py

### Exit criteria

- One canonical eligibility object is enforced in both automl and registry promotion.

## Phase 2: Data Contracts, Lineage, And Hard Validation

### Why second

Everything else depends on being able to trust the input datasets and provenance.

### Main files

- core/data.py
- core/data_quality.py
- core/storage.py
- core/context.py

### Tests

- Extend tests/test_data_fetch_integrity.py
- Extend tests/test_data_quality_quarantine.py
- Extend tests/test_safe_artifact_storage.py
- Add tests/test_data_contracts.py

### Exit criteria

- Training or backtesting cannot proceed with unknown breaking schema drift or missing lineage.

## Phase 3: Cross-Venue Reference Validation For Spot And Futures

### Why third

Once contracts exist, the next weakness is venue-local truth. The repository already has reference-overlay hooks, so this phase can land without re-architecting the entire pipeline.

### Main files

- core/data.py
- core/reference_data.py
- core/context.py
- core/data_quality.py
- core/scenarios.py

### Tests

- Extend tests/test_exchange_failure_scenarios.py
- Extend tests/test_derivatives_context_pipeline.py
- Add tests/test_cross_venue_reference_validation.py
- Add tests/test_futures_mark_index_validation.py

### Exit criteria

- Spot and futures inputs can be checked against reference venues and related diagnostics are persisted.

## Phase 4: Signal Half-Life And Delay Compatibility

### Why fourth

This is the first phase that directly tells the system whether a signal survives its own execution assumptions.

### Main files

- core/models.py
- core/backtest.py
- core/pipeline.py
- core/monitoring.py

### Tests

- Extend tests/test_signal_profitability_sizing.py
- Add tests/test_signal_half_life.py
- Add tests/test_signal_delay_compatibility.py

### Exit criteria

- Every promotable model has a decay report and delay compatibility check.

## Phase 5: Execution Engine Hardening

### Why fifth

By this point, signal governance will know what edge should survive; execution hardening then tests whether the research engine is still crediting impossible fills.

### Main files

- core/backtest.py
- core/execution/intents.py
- core/execution/policies.py
- core/execution/liquidity.py
- core/execution/costs.py
- core/execution/nautilus_adapter.py
- core/slippage.py

### Tests

- Extend tests/test_execution_partial_fills.py
- Extend tests/test_execution_adapter_parity.py
- Extend tests/test_microstructure_cost_models.py
- Extend tests/test_causal_liquidity_inputs.py
- Add tests/test_execution_queue_latency.py
- Add tests/test_bar_timestamp_execution_semantics.py

### Exit criteria

- The default engine supports explicit latency, queue heuristics, and stricter partial-fill behavior without breaking existing research ergonomics.

## Phase 6: Replication Cohorts And Promotion Governance

### Why last

This phase ties together the previous ones and raises promotion from local validation to broader out-of-sample evidence.

### Main files

- core/automl.py
- core/pipeline.py
- core/registry/store.py
- core/registry/manifest.py
- core/universe.py

### Tests

- Extend tests/test_automl_holdout_objective.py
- Extend tests/test_post_selection_inference.py
- Extend tests/test_local_registry_flow.py
- Add tests/test_replication_promotion_policy.py

### Exit criteria

- Promotion requires both local success and cohort replication success.

## Threshold Rollout Strategy

To avoid arbitrary first-pass thresholds while still moving toward fail-closed governance:

1. Introduce new reports with measurement-only mode.
2. Collect distributions from historical runs and representative examples.
3. Freeze thresholds in config.
4. Turn advisory checks into blocking promotion gates.
5. Require threshold changes to be versioned and documented.

This should be short-lived. The purpose is calibration, not a permanent excuse for soft controls.

## Proposed New Test Modules

- tests/test_promotion_gate_binding.py
- tests/test_data_contracts.py
- tests/test_cross_venue_reference_validation.py
- tests/test_futures_mark_index_validation.py
- tests/test_signal_half_life.py
- tests/test_signal_delay_compatibility.py
- tests/test_execution_queue_latency.py
- tests/test_bar_timestamp_execution_semantics.py
- tests/test_replication_promotion_policy.py

## Proposed Dependency Changes

- Add pandera to requirements.txt for dataframe contract validation.

Optional later additions, not required for the first wave:

- pydantic if config and manifest validation becomes too complex for dataclasses alone
- venue-specific client libraries if plain REST integration becomes too brittle

## Definition Of Done

The P0 program is complete when all of the following are true:

- Promotion uses one canonical fail-closed eligibility object.
- Dataset contracts and lineage are first-class and persisted.
- Spot and futures data can be sanity-checked against reference venues.
- Every signal has a stored decay report and delay compatibility result.
- The default research execution engine enforces materially stricter fill realism.
- Promotion requires replication beyond the primary local study.
- The registry stores enough evidence to explain and reproduce every promotion decision.

## Non-Goals For This Wave

- Full replacement of the research engine with NautilusTrader.
- A custom multi-venue execution engine for live trading.
- Exhaustive exchange coverage beyond the first reference set.
- Solving every P1 or P2 issue before P0 promotion integrity is fixed.

## Recommended Immediate Start

Start with Phase 1 and Phase 2 in the same branch, but land them as separate merges:

- Merge 1: canonical promotion gate binding
- Merge 2: data contracts and lineage

That sequence closes the largest governance illusion first and then hardens the trust boundary around the data that drives every later decision.