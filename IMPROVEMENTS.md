# IMPROVEMENTS - Detailed Remediation Plan

This document replaces the earlier adversarial audit memo with implementation-ready remediation plans.

## Locked Assumptions

- Output target: this file is now the detailed plan document.
- Granularity: every item is specified as an implementation-ready work package with architecture, steps, file touchpoints, tests, and acceptance criteria.
- Delivery model: phased delivery is required.
- Product scope: v1 remains backtesting-only, but the architecture must leave a clean path to paper/live readiness.
- Execution path: migrate to an existing event-driven engine immediately, anchored on NautilusTrader. Do not build a custom execution engine.
- Market scope: Binance remains primary, but reference and multi-exchange overlays should be designed as generic abstractions.
- L2 data: likely available soon, so the design should support order-book-aware simulation natively.
- Registry: local filesystem-first, no paid services.
- Drift stack: River and Evidently are acceptable.
- Statistical rigor: include family-wise post-selection tests such as White Reality Check and Hansen SPA.
- Change style: prefer architectural cleanup when needed rather than local patchwork.
- Compute budget: must run on consumer hardware.

## Phase Model

### V1 - Backtesting Hardening

- Freeze evaluation protocol before further model search work.
- Migrate execution simulation to NautilusTrader-backed research backtests.
- Remove causal leakage from liquidity, validation, and stage boundaries.
- Add post-selection inference, lookahead provocation, data quality, and artifact safety.

### V1.5 - Paper Readiness

- Add local registry, drift monitoring, operations-centric monitoring, and portability gates.
- Add reference/multi-exchange overlays and stronger regime/context handling.

### V2 - Live Governance Readiness

- Add scenario-based venue failure realism, champion/challenger promotion, rollback, and live-facing governance.

## Recommended Execution Order

1. P0-1 Locked holdout isolation
2. P0-2 Signal-policy decontamination
3. P0-4 Causal liquidity inputs
4. P1-6 NautilusTrader execution migration
5. P1-7 Microstructure-aware cost stack
6. P0-3 Lookahead provocation harness
7. P0-5 Family-wise post-selection tests
8. P1-9 Data-quality quarantine
9. P1-13 Cross-stage embargoes
10. P2-15 Safe artifact persistence
11. P1-10 Historical universe control
12. P1-8 Venue-portability governance
13. P1-11 Regime-layer redesign
14. P1-12 Feature admission and retirement
15. P2-14 Local registry plus drift workflows
16. P2-17 Operations-centric monitoring
17. P2-16 Venue failure realism

---

## P0-1. Make The Locked Holdout Actually Locked

**Phase**: V1  
**Decision**: Convert selection into a strict four-stage flow: search, validation, selection freeze, final locked holdout. The locked holdout must be touched once per study, after a single candidate is frozen.

### Architecture

```
Search Window -> Validation Window -> Selection Freeze -> Final Locked Holdout
```

- Search window: CPCV or walk-forward used only for trial generation and pruning.
- Validation window: used for ranking and eligibility gates.
- Selection freeze: one immutable candidate snapshot is created before holdout access.
- Locked holdout: single post-selection evaluation only.

### Steps

1. Refactor `core/automl.py` so no code inside the per-candidate promotion loop can call `_evaluate_locked_holdout(...)`.
2. Introduce a `SelectionSnapshot` or equivalent immutable payload containing:
	- frozen overrides
	- validation metrics
	- eligibility results
	- trial hash / config hash
	- selection timestamp
3. Create a dedicated `finalize_selection()` path that:
	- chooses the winning candidate from search and validation only
	- persists the frozen snapshot into the study summary
	- evaluates the locked holdout exactly once
4. Add explicit diagnostics:
	- `holdout_access_count`
	- `frozen_candidate_hash`
	- `holdout_evaluated_once`
	- `holdout_evaluated_after_freeze`
5. Make holdout evaluation side-effect free: it must not mutate ranking, reorder candidates, or trigger re-selection.
6. Persist the frozen candidate manifest before holdout execution so any later rerun can prove the holdout did not affect selection.

### Files To Modify Or Create

- `core/automl.py`
- `tests/test_automl_holdout_objective.py`
- `tests/test_locked_holdout_selection_freeze.py` (new)

### Tests

- Verify no more than one locked-holdout evaluation occurs per study.
- Verify the winning candidate is chosen before holdout evaluation.
- Verify changing holdout metrics does not change which candidate is selected.
- Verify the study summary records the frozen candidate hash and access count.

### Acceptance Criteria

- Locked holdout access count equals 1 for a completed study.
- Candidate ranking uses only search and validation evidence.
- Holdout evaluation cannot change the selected candidate.
- The study report proves post-selection holdout access.

---

## P0-2. Remove Signal-Policy Knobs From AutoML Search

**Phase**: V1  
**Decision**: Separate model search from signal policy. AutoML may search feature, label, and model families, but not threshold, abstain, aggressiveness, or sizing knobs.

### Architecture

```
Model Search -> Validation Metrics -> Deterministic SignalPolicyBuilder -> Backtest
```

- Model search decides features, labels, model family, and calibrators.
- `SignalPolicyBuilder` derives thresholds from cost math, trade statistics, and fixed governance rules.
- Optional policy calibration is allowed only as a deterministic post-search step, never as an Optuna dimension inside the same study.

### Steps

1. Remove `signals.threshold`, `signals.fraction`, `signals.edge_threshold`, `signals.meta_threshold`, and similar policy knobs from `DEFAULT_AUTOML_SEARCH_SPACE` in `core/automl.py`.
2. Create a `SignalPolicyBuilder` in `core/pipeline.py` or a new `core/signal_policy.py` that resolves:
	- action threshold
	- edge threshold
	- abstain threshold
	- Kelly cap / flat fallback
	from explicit formulas and policy config.
3. Split signal policy modes into:
	- `theory_only`
	- `validation_calibrated`
	- `frozen_manual`
4. If `validation_calibrated` is enabled, calibrate on the validation slice only and record every derived parameter in the study manifest.
5. Report model quality and policy quality separately so thresholding is not mistaken for predictive power.
6. Add a hard rule that policy calibration cannot open a second hyperparameter search loop inside model AutoML.

### Files To Modify Or Create

- `core/automl.py`
- `core/pipeline.py`
- `core/signal_policy.py` (new, recommended)
- `example_automl.py`
- `tests/test_automl_signal_policy_decontamination.py` (new)

### Tests

- Verify the default AutoML search space no longer includes signal-policy knobs.
- Verify policy parameters are deterministically reconstructed from the frozen config.
- Verify changing policy mode does not alter the frozen model-search candidate list.
- Verify the study summary reports policy parameters separately from model params.

### Acceptance Criteria

- AutoML no longer searches signal-policy thresholds or Kelly fractions.
- Signal policy is fully reconstructible from recorded config and validation statistics.
- Model ranking is not confounded by searched execution or sizing aggressiveness.

---

## P0-3. Add A Lookahead-Provocation Harness

**Phase**: V1  
**Decision**: Add a differential replay harness that compares a full-run pipeline against prefix-only replays at sampled decision timestamps.

### Architecture

```
Baseline Full Run
		  |
Prefix Replay Runner
		  |
Column Comparator -> Bias Report
```

- The comparator audits features, labels, predictions, signals, and execution inputs.
- Any difference at a decision timestamp indicates future contamination or non-causal recursion.

### Steps

1. Create `core/research_integrity.py` or `core/lookahead.py` with a replay harness that:
	- runs the pipeline on the full dataset
	- reruns the pipeline on truncated prefixes ending at sampled timestamps
	- compares audited columns at each decision timestamp
2. Audit at minimum:
	- engineered features
	- regime labels
	- aligned labels
	- model probabilities
	- signals
	- execution prices and liquidity inputs
3. Add configurable sampling so the harness can run on consumer hardware:
	- full replay for small fixtures
	- sampled decision timestamps for large datasets
4. Emit a machine-readable report listing:
	- biased columns
	- first offending timestamp
	- mismatch count
	- stage where mismatch originated
5. Add a regression fixture with an intentionally biased feature so the harness must fail.

### Files To Modify Or Create

- `core/lookahead.py` or `core/research_integrity.py` (new)
- `tests/test_lookahead_provocation.py` (new)
- `README.md`

### Tests

- A deliberately future-shifted feature must be detected.
- A clean causal feature set must pass.
- Prefix replay and full replay must match for audited columns at sampled timestamps.
- The report must identify the first offending stage and column.

### Acceptance Criteria

- The repo has a deterministic lookahead-provocation harness, not just ad hoc leakage tests.
- Newly added indicators and feature builders can be audited without manual inspection.
- Bias reports are reproducible and machine-readable.

---

## P0-4. Remove Ex-Post Liquidity From Cost Estimation

**Phase**: V1  
**Decision**: All liquidity inputs must be aligned to what is knowable at order submission time. For open-bar execution, same-bar realized volume is forbidden.

### Architecture

```
Decision Time -> Known Liquidity State -> Order Submission -> Fill Simulation
```

- Bar-mode fallback uses lagged ADV and lagged realized volume proxies.
- L2-aware mode uses order-book snapshots timestamped at or before order submission.
- Event-driven mode uses NautilusTrader market depth or synthetic queue models.

### Steps

1. Create a `LiquidityInputResolver` in `core/backtest.py` or a new `core/execution/liquidity.py`.
2. For bar-based execution:
	- if execution is next-open, shift volume-based inputs by one full bar
	- compute ADV from prior bars only
	- forbid zero-lag access to realized bar volume
3. For L2 mode:
	- require order-book snapshots timestamped no later than execution submission time
	- reject snapshots that occur after the simulated order timestamp
4. Add diagnostics:
	- `liquidity_lag_bars`
	- `ex_post_liquidity_rows`
	- `liquidity_source`
5. Integrate these rules into the NautilusTrader adapter so event-driven execution uses the correct state by construction.

### Files To Modify Or Create

- `core/backtest.py`
- `core/pipeline.py`
- `core/execution/liquidity.py` (new, recommended)
- `tests/test_signal_execution_alignment.py`
- `tests/test_causal_liquidity_inputs.py` (new)

### Tests

- Verify open-bar execution cannot read same-bar realized volume.
- Verify lagged ADV matches only prior bars.
- Verify post-timestamp L2 snapshots are rejected.
- Verify diagnostics count any ex-post rows as zero in valid runs.

### Acceptance Criteria

- Same-bar realized volume is no longer used for open-bar execution costs.
- Liquidity provenance is explicit in backtest outputs.
- Event-driven and bar-based paths share the same causality rules.

---

## P0-5. Add Family-Wise Post-Selection Inference

**Phase**: V1  
**Decision**: Add White Reality Check and Hansen SPA as post-selection tests on the surviving candidate set, with compute controls suitable for consumer hardware.

### Architecture

```
Candidate Return Archive -> Aligned Return Matrix -> Stationary/Block Bootstrap
												 |                         |
										White Reality Check        Hansen SPA
```

### Steps

1. Create `core/post_selection_tests.py` or `core/stat_tests.py`.
2. Persist validation-stage return paths for the top `N` eligible candidates after de-duplication by correlation cluster. Default `N` should be modest, e.g. 8 to 16.
3. Build an aligned candidate return matrix with explicit overlap rules, reusing the repo’s existing overlap diagnostics where possible.
4. Implement:
	- White Reality Check
	- Hansen SPA
	- optional stepdown / max-stat variant for stronger control
5. Add bootstrap settings compatible with consumer hardware:
	- stationary or moving block bootstrap
	- cached aligned return matrices
	- deterministic random seeds
6. Report raw and adjusted p-values in the study summary and optionally gate promotion for paper/live readiness.

### Files To Modify Or Create

- `core/stat_tests.py` (new)
- `core/automl.py`
- `tests/test_post_selection_inference.py` (new)

### Tests

- Synthetic noise candidates must fail the family-wise tests.
- A clearly superior synthetic candidate must pass under controlled settings.
- The aligned return matrix must honor overlap policies.
- Runtime must stay bounded for the configured candidate cap.

### Acceptance Criteria

- The study summary reports White RC and SPA results.
- Promotion can be configured to require passing post-selection inference.
- Candidate caps and caching keep runtime feasible on consumer hardware.

---

## P1-6. Replace Full-Fill Optimism With Event-Driven Execution Simulation

**Phase**: V1  
**Decision**: NautilusTrader becomes the canonical execution-simulation engine for research backtests. The existing pandas/vectorbt path remains only as a regression baseline until decommissioned.

### Architecture

```
Signals -> OrderIntent -> NautilusTrader Adapter -> Fill / PartialFill / Cancel Events -> Position State
```

### Steps

1. Create a new `core/execution/` package with:
	- `intents.py` for order intents
	- `adapter.py` for the execution interface
	- `nautilus_adapter.py` for NautilusTrader-backed simulation
	- `policies.py` for order submission rules
2. Refactor `BacktestStep` so it emits order intents, not immediately filled target weights.
3. Add execution policy config:
	- participation caps
	- time in force
	- cancel/replace windows
	- minimum fill ratio
	- max order age
4. Model partial fills and stale orders explicitly.
5. Keep the current execution contract only as a parity harness while the Nautilus path is validated.

### Files To Modify Or Create

- `core/pipeline.py`
- `core/backtest.py`
- `core/execution/__init__.py` (new)
- `core/execution/intents.py` (new)
- `core/execution/adapter.py` (new)
- `core/execution/nautilus_adapter.py` (new)
- `tests/test_execution_partial_fills.py` (new)
- `tests/test_execution_adapter_parity.py` (new)

### Tests

- Orders larger than available depth or participation cap must fill partially.
- Aged or unfilled orders must cancel according to policy.
- Legacy and Nautilus paths must match on simple deterministic fixtures.
- Backtests must report fill ratio, unfilled notional, and cancel counts.

### Acceptance Criteria

- Full immediate fills are no longer assumed by default.
- NautilusTrader is the primary research execution simulator.
- Backtest summaries expose fill realism diagnostics.

---

## P1-7. Replace The Current Impact Model With A Tiered Microstructure Cost Stack

**Phase**: V1  
**Decision**: Move from a single square-root slippage heuristic to a tiered cost stack that supports proxy mode, L2-aware mode, and Nautilus fill-aware mode.

### Architecture

```
CostModel
├── ProxyImpactModel        (lagged ADV, spread proxy, participation, stress)
├── DepthCurveImpactModel   (L2 snapshot, imbalance, queue proxy)
└── FillAwareCostModel      (Nautilus executed fills)
```

### Steps

1. Move slippage logic into `core/execution/costs.py` or refactor `core/slippage.py` into an execution-cost module.
2. Add proxy-model components:
	- spread proxy
	- lagged ADV / participation
	- volatility / stress multiplier
	- adverse selection penalty
3. Add L2 depth-curve mode using generic depth inputs:
	- bid/ask depth by level
	- imbalance
	- snapshot age
	- queue proxy
4. Add fill-aware post-trade attribution in the Nautilus path so realized cost is calculated from fill events, not just ex-ante estimates.
5. Add cost stress sweeps to show margin of safety under worse spreads and thinner books.

### Files To Modify Or Create

- `core/slippage.py` or `core/execution/costs.py`
- `core/backtest.py`
- `core/execution/nautilus_adapter.py`
- `tests/test_microstructure_cost_models.py` (new)

### Tests

- Proxy cost must increase with participation, spread, and volatility.
- L2 cost must react to thinner displayed depth and adverse imbalance.
- Fill-aware realized cost must equal the sum of fill-event costs on deterministic fixtures.
- Stress sweeps must show monotonic cost deterioration.

### Acceptance Criteria

- Cost modeling supports at least proxy and L2-aware modes.
- Nautilus fill events become the canonical source of realized execution cost.
- The repo can run without L2, but degrades honestly rather than pretending bar volume is enough.

---

## P1-8. Add Venue-Portability Governance For Exchange-Specific Features

**Phase**: V1.5  
**Decision**: Features derived from Binance-only microstructure must be tagged, audited, and promotion-gated. Generic reference overlays should exist, but the abstraction should not hard-code specific venues.

### Architecture

```
Feature Provenance
├── endogenous
├── venue_specific
├── reference_overlay
└── cross_venue_composite
```

### Steps

1. Extend feature metadata to include:
	- source venue scope
	- portability class
	- dependence on exchange-specific semantics
2. Add generic reference-overlay adapters in `core/context.py` or a new `core/reference_data.py` for:
	- cross-venue price checks
	- reference volume or breadth proxies
	- composite funding / basis overlays
3. Add an ablation runner comparing:
	- endogenous-only features
	- venue-specific features
	- reference-overlay features
4. Gate promotion if a strategy’s importance is dominated by unvalidated venue-specific features and the ablation fails to replicate edge.
5. Surface portability diagnostics in study reports.

### Files To Modify Or Create

- `core/features.py`
- `core/context.py`
- `core/reference_data.py` (new)
- `core/feature_governance.py` (new)
- `tests/test_feature_portability_gates.py` (new)

### Tests

- Venue-specific features must be tagged automatically.
- Promotion gate must fail when edge disappears under portability ablation.
- Generic overlays must integrate without hard-coding one venue name into the feature interface.

### Acceptance Criteria

- Exchange-specific features are first-class governed objects, not anonymous columns.
- Promotion reports quantify venue-specific dependence.
- Generic reference-overlay hooks exist for future multi-exchange data.

---

## P1-9. Add A Data-Quality And Bad-Print Quarantine Layer

**Phase**: V1  
**Decision**: Build a dedicated pre-feature data quality layer that can flag, quarantine, drop, or null suspicious candles before labels and features are created.

### Architecture

```
Raw Bars -> DataQualityReport -> Clean Bars + Quarantine Mask -> Features / Labels
```

### Steps

1. Create `core/data_quality.py`.
2. Implement checks for:
	- OHLC consistency
	- duplicate or retrograde timestamps
	- extreme return and range spikes relative to local robust volatility
	- zero or negative volume anomalies
	- quote/base volume inconsistencies
	- trade-count anomalies
3. Support configurable actions per anomaly type:
	- `flag`
	- `null`
	- `drop`
	- `winsorize` for explicitly allowed cases
4. Feed the quality mask into labeling and feature builders so quarantined rows cannot silently dominate training.
5. Persist structured quality reports into pipeline state and experiment outputs.

### Files To Modify Or Create

- `core/data_quality.py` (new)
- `core/data.py`
- `core/pipeline.py`
- `tests/test_data_quality_quarantine.py` (new)

### Tests

- Synthetic bad prints must be detected and quarantined.
- Quarantined rows must not enter feature or label generation when policy forbids it.
- Quality reports must count anomalies by type and action.

### Acceptance Criteria

- The pipeline has an explicit data quality layer before feature engineering.
- Anomalous candles cannot silently influence labels or model fitting.
- Every run records a structured quality report.

---

## P1-10. Add Historical Universe Snapshots To Remove Survivorship Bias In Cross-Symbol Research

**Phase**: V1.5  
**Decision**: Cross-symbol studies must use historical eligibility snapshots, not today’s surviving symbol list.

### Architecture

```
ExchangeInfo Snapshots -> HistoricalUniverseStore -> Eligibility Policy -> Cross-Symbol Study
```

### Steps

1. Create `core/universe.py`.
2. Persist exchange metadata snapshots over time in Parquet or JSON manifests.
3. Build eligibility rules using:
	- listing start
	- delisting end
	- trading status windows
	- minimum history
	- minimum liquidity
4. Require all cross-symbol experiments to request a universe snapshot keyed to the training start timestamp.
5. Add policies for test-period handling:
	- drop symbol at delisting
	- freeze signal generation
	- liquidate on halt / delist event

### Files To Modify Or Create

- `core/universe.py` (new)
- `core/data.py`
- `core/pipeline.py`
- `tests/test_historical_universe_selection.py` (new)

### Tests

- Synthetic delisted symbols must disappear from later universe snapshots.
- Cross-symbol studies must reject using symbols not eligible at study start.
- Delisting events must trigger configured handling in backtests.

### Acceptance Criteria

- Cross-symbol research uses time-varying universes.
- Delisting and trading-status events are modeled explicitly.
- Survivorship bias is no longer implicit in future multi-symbol studies.

---

## P1-11. Redesign The Regime Layer So It Is Not Just Price Echo

**Phase**: V1.5  
**Decision**: Split regime detection into instrument-state and market-state layers. Market-state should rely on context and reference overlays, not just the same instrument’s price history.

### Architecture

```
Regime Layer
├── instrument_state
├── market_state
└── cross_asset_state
```

### Steps

1. Move regime logic out of `core/models.py` into `core/regime.py`.
2. Define separate inputs for:
	- instrument-local state
	- reference-market state
	- cross-asset state
3. Add regime feature provenance so the study report can show how much of regime detection came from endogenous vs contextual inputs.
4. Add regime ablation tests:
	- endogenous-only regime layer
	- context-enriched regime layer
5. Require regime-conditioned models or thresholds to demonstrate stability improvement, not just point-performance improvement.

### Files To Modify Or Create

- `core/regime.py` (new)
- `core/models.py`
- `core/pipeline.py`
- `core/context.py`
- `tests/test_regime_layer_ablation.py` (new)

### Tests

- Endogenous-only and context-aware regime builders must produce distinct provenance reports.
- Regime-conditioned thresholds must show stability or be rejected by policy.
- Fold-local fitting rules must still hold after refactor.

### Acceptance Criteria

- Regime detection is no longer dominated by the target instrument alone.
- The repo can explain where regime signals came from.
- Regime-aware behavior must improve stability, not just in-sample fit.

---

## P1-12. Add Feature Admission, Robustness, And Retirement Governance

**Phase**: V1.5  
**Decision**: Stationarity remains necessary but becomes only one admission check. Features must also pass robustness and portability gates.

### Architecture

```
FeatureAdmissionPolicy
├── stationarity
├── provenance
├── regime stability
├── perturbation stability
└── portability / ablation checks
```

### Steps

1. Create `core/feature_governance.py`.
2. Require each feature family to emit metadata:
	- hypothesis or rationale tag
	- source lineage
	- transform chain
	- venue scope
3. Add robustness checks:
	- rolling sign stability
	- leave-one-regime-out importance stability
	- null or permutation importance sanity check
	- small perturbation sensitivity
4. Add retirement logic so brittle features can be marked deprecated and excluded from future studies automatically.
5. Surface admission and retirement decisions in experiment manifests.

### Files To Modify Or Create

- `core/feature_governance.py` (new)
- `core/features.py`
- `core/automl.py`
- `tests/test_feature_admission_policy.py` (new)

### Tests

- Features that pass ADF but fail stability checks must be rejected.
- Retired features must not enter future AutoML studies unless explicitly re-enabled.
- Governance metadata must survive experiment serialization.

### Acceptance Criteria

- ADF is no longer treated as a sufficient feature-quality check.
- Feature admission and retirement are explicit, testable steps.
- Study outputs explain why a feature family was admitted or rejected.

---

## P1-13. Add Cross-Stage Temporal Embargoes

**Phase**: V1  
**Decision**: Search, validation, and locked holdout must be separated by explicit stage-level embargo gaps derived from label horizon and execution delay.

### Architecture

```
Search -> Search/Validation Gap -> Validation -> Validation/Holdout Gap -> Locked Holdout
```

### Steps

1. Extend the holdout plan in `core/automl.py` to include:
	- `search_validation_gap_bars`
	- `validation_holdout_gap_bars`
2. Default gap size should be the maximum of:
	- label horizon / max holding
	- signal delay bars
	- configured embargo
3. Ensure aligned split helpers drop gap rows rather than silently merging contiguous windows.
4. Report dropped-gap rows in the study summary.
5. Ensure CPCV inside the search window still operates independently of the outer-stage gaps.

### Files To Modify Or Create

- `core/automl.py`
- `core/pipeline.py`
- `tests/test_cross_stage_embargo.py` (new)

### Tests

- Validation and holdout windows must start after the configured stage gaps.
- Gap rows must not appear in train or test slices.
- Default gap must widen automatically when label horizon increases.

### Acceptance Criteria

- Search, validation, and holdout windows are no longer contiguous by default.
- Stage-gap sizes are explicit and reported.
- Barrier-based labels cannot bleed across stage boundaries by construction.

---

## P2-14. Add A Local Registry And Drift-Governed Promotion Flow

**Phase**: V1.5  
**Decision**: Build a local filesystem-first registry with immutable versions, batch drift reports, streaming drift hooks, and offline champion/challenger promotion.

### Architecture

```
Registry
├── version manifest
├── feature schema
├── metrics and lineage
├── drift reports
└── promotion status (candidate/champion/challenger/archived)
```

### Steps

1. Create `core/registry/` with:
	- manifest schema
	- immutable version directories
	- status tags
	- lineage metadata
2. Create `core/drift.py` with:
	- Evidently batch drift on features and predictions
	- River ADWIN hooks for streaming probability or realized-edge drift
3. Define retraining guardrails:
	- minimum samples
	- cooldown windows
	- no retrain on drift alone without sufficient evidence
4. Add offline champion/challenger evaluation using paper-style replay before promotion is allowed.
5. Store drift and promotion decisions in the registry manifest.

### Files To Modify Or Create

- `core/registry/__init__.py` (new)
- `core/registry/manifest.py` (new)
- `core/registry/store.py` (new)
- `core/drift.py` (new)
- `core/automl.py`
- `tests/test_local_registry_flow.py` (new)
- `tests/test_drift_monitoring.py` (new)

### Tests

- Version manifests must be immutable once created.
- Drift alerts must not trigger promotion without minimum sample thresholds.
- Challenger rollback to the previous champion must be possible.

### Acceptance Criteria

- The repo has a local registry with immutable versions.
- Drift reports and promotion decisions are first-class artifacts.
- Champion/challenger flows are supported offline without paid services.

---

## P2-15. Remove Unsafe Persistent Pickle Usage

**Phase**: V1  
**Decision**: Remove pickle from persistent registry and cache paths. Use Parquet/JSON for data artifacts and `skops` or an equivalent safer format for sklearn-compatible models.

### Architecture

```
Persistent Storage
├── Parquet / Feather for tables
├── JSON / YAML for metadata
├── skops for sklearn model artifacts
└── SHA-256 manifest verification
```

### Steps

1. Replace persistent DataFrame/object caches in `core/data.py` and `core/context.py` with:
	- Parquet for tabular frames
	- JSON for small metadata payloads
2. Replace deployable model storage in `core/models.py` with:
	- `skops` where supported
	- explicit manifest plus hashes
	- fallback to retraining recipe storage if a model type cannot be safely serialized
3. Add artifact hash verification before load.
4. Add schema compatibility checks for feature order and required columns.
5. Restrict any remaining pickle usage to short-lived, trusted, explicitly marked developer caches only if absolutely unavoidable.

### Files To Modify Or Create

- `core/models.py`
- `core/data.py`
- `core/context.py`
- `requirements.txt`
- `tests/test_safe_artifact_storage.py` (new)

### Tests

- Tampered artifacts must fail hash verification.
- Feature schema mismatches must fail closed.
- Persistent market and context caches must no longer write pickle files.

### Acceptance Criteria

- Persistent artifact storage no longer relies on raw pickle.
- Every load path validates artifact integrity and schema compatibility.
- Data and model artifacts are portable and inspectable.

---

## P2-16. Add Venue Failure Realism And Stress Scenarios

**Phase**: V2  
**Decision**: Add a scenario layer that injects venue outages, stale marks, halts, leverage changes, and other exchange-state failures into backtests.

### Architecture

```
ScenarioEventSchedule -> Nautilus Backtest Adapter -> Stress Results
```

### Steps

1. Create `core/scenarios.py` for exchange-state events:
	- downtime windows
	- stale mark windows
	- trading halts
	- leverage bracket changes
	- forced deleveraging analogs
2. Integrate scenario schedules into the Nautilus research adapter.
3. Add strategy responses:
	- kill switch
	- stale data reject
	- forced liquidation policy
	- halt liquidation / freeze behavior
4. Add scenario matrix sweeps so results include best-case, base-case, and stressed-case outcomes.

### Files To Modify Or Create

- `core/scenarios.py` (new)
- `core/execution/nautilus_adapter.py`
- `core/backtest.py`
- `tests/test_exchange_failure_scenarios.py` (new)

### Tests

- Downtime events must suppress fills.
- Stale marks must trigger configured rejection or warning behavior.
- Halt events must force the configured position-handling path.

### Acceptance Criteria

- Venue failures can be injected into research backtests.
- Results report stressed behavior separately from base-case performance.
- Strategies can no longer ignore exchange-state discontinuities.

---

## P2-17. Add Operations-Centric Monitoring

**Phase**: V1.5  
**Decision**: Monitoring must extend beyond research metrics to freshness, schema, fill quality, slippage drift, and inference latency.

### Architecture

```
RunMonitor
├── data freshness
├── schema checks
├── fill quality drift
├── realized slippage drift
└── inference latency / backlog
```

### Steps

1. Create `core/monitoring.py`.
2. Track at minimum:
	- raw-data freshness
	- custom-data TTL breaches
	- L2 snapshot age
	- feature schema drift
	- realized vs expected slippage
	- fill ratio deterioration
	- inference latency and queue backlog
3. Emit local JSON and Parquet reports plus human-readable markdown summaries.
4. Integrate monitoring artifacts with the local registry and drift reports.
5. Add policy hooks so paper/live readiness can require healthy operational metrics before promotion.

### Files To Modify Or Create

- `core/monitoring.py` (new)
- `core/pipeline.py`
- `core/registry/store.py`
- `tests/test_operations_monitoring.py` (new)

### Tests

- Freshness breaches must be detected and reported.
- Schema drift must fail closed.
- Fill-quality and slippage drift must be measurable on replayed runs.

### Acceptance Criteria

- Monitoring artifacts cover operational failure modes, not just research metrics.
- Local reports are produced without paid services.
- Promotion workflows can consume operational health checks.

---

## Exit Criteria By Phase

### V1 Exit Criteria

- Locked holdout is touched once after candidate freeze.
- AutoML no longer searches signal-policy knobs.
- Liquidity inputs are causal.
- NautilusTrader is the primary research execution simulator.
- Cost modeling supports proxy and L2-aware paths.
- The repo has a lookahead-provocation harness.
- White RC and Hansen SPA are implemented with consumer-hardware defaults.
- Data quality and cross-stage embargo layers are active.
- Persistent pickle usage is removed from deployable paths.

### V1.5 Exit Criteria

- Venue portability and historical universe controls are active.
- Regime and feature governance layers are explicit and auditable.
- Local registry and drift workflows exist.
- Operations-centric monitoring artifacts are emitted.

### V2 Exit Criteria

- Venue failure scenarios are part of standard research validation.
- Champion/challenger and rollback workflows are operationally defensible.

## Final Note

The critical sequencing rule is simple: do not spend time improving alpha discovery before the evaluation and execution layers stop overstating alpha. The first meaningful milestone is not “better models.” It is “the current models stop lying.”