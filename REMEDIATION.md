# Remediation Plan To Raise The Minimum Bar For Risking Money

## Status

This document supersedes earlier remediation notes and reflects the repo state audited on 2026-04-28.

It is not a claim that the repository has no good controls. It already has meaningful components, including CPCV, walk-forward replay, lookahead auditing, statistical significance tooling, replication hooks, and a more realistic execution stack than most retail codebases. The problem is that those controls are not yet bound tightly enough on the user-facing path to justify risking money.

Current blocking evidence:

1. The easy path is still explicitly research-only in [example_automl.py](example_automl.py).
2. The demo workflow still rebuilds and re-backtests after winner selection, which contaminates the presentation of final evidence.
3. The targeted AutoML holdout and promotion suite still fails in material places in [tests/test_automl_holdout_objective.py](tests/test_automl_holdout_objective.py).
4. Research-mode defaults remain materially more permissive than a capital-facing workflow should allow.

This plan is therefore organized around one question only:

What must change for a retail operator on consumer hardware to be able to risk small amounts of money without confusing research output for deployment evidence?

The answer is not a single code fix. It is a combination of:

1. stricter code contracts
2. cleaner evidence separation
3. accessible local certification tooling
4. more binding execution realism
5. a pre-capital paper and micro-capital release ladder

Until those are complete, the repo remains research-only.
## External Research Applied

This remediation plan incorporates isolated external research from the following sources and translates it into repo-specific engineering actions:

1. Hudson & Thames / mlfinlab cross-validation documentation:
   - purging removes training samples whose information overlaps test labels
   - embargo removes nearby rows that can still leak across the train/test boundary
   - CPCV is useful because it produces a distribution of backtest paths, not a single flattering path
2. Hudson & Thames / mlfinlab backtest statistics documentation:
   - DSR is specifically meant to adjust Sharpe-based evidence for non-normal returns, track-record length, and multiple testing
   - minimum track record length and bet concentration matter for deciding whether observed Sharpe is actionable or just lucky
3. scikit-learn nested CV guidance:
   - using the same data for tuning and evaluation is optimistic by construction
   - parameter search must live inside an inner loop and performance estimation must happen on an outer untouched loop
4. scikit-learn TimeSeriesSplit documentation:
   - `gap` is an exclusion boundary between train and test, but it is not a full replacement for purging overlapping labels
   - ownership of boundary logic should be singular, not split across multiple masking layers
5. NautilusTrader backtesting and order documentation:
   - consumer-hardware local event-driven backtesting is feasible with low-level `BacktestEngine` or high-level `BacktestNode`
   - event ordering, queue position, liquidity consumption, price protection, timestamp convention, and fill models must be configured explicitly
   - bar data can be used conservatively, but only when bar timestamps and execution semantics are handled correctly
6. QuantStart transaction-cost guidance:
   - flat costs are not enough for capital decisions
   - slippage, latency, liquidity, market impact, and order-type behavior materially change live outcomes

The practical implication is straightforward: the repo already points in the right direction, but must stop mixing research convenience with capital-facing claims.
## What "Safe Enough To Risk Money" Means Here

In this repository, "safe enough to risk money" does not mean production-grade institutional deployment.

It means all of the following are true before even micro-capital is allowed:

1. There is a clean separation between model selection evidence, locked-holdout evidence, refit artifacts, paper evidence, and live evidence.
2. The system can abstain cleanly when no candidate is good enough.
3. Every capital-facing run fails closed on market-data gaps, duplicate conflicts, quarantine breaches, stale context, and missing funding or reference coverage.
4. The local certification path is accessible on consumer hardware and does not require the user to fall back to the research-only demo.
5. The execution engine used for certification is event-driven and configured to match the actual data granularity.
6. The model is selected using time-series-compatible inner validation and judged using untouched outer evidence.
7. Promotion gates include DSR or equivalent multiple-testing correction, minimum track-record logic, effective-bet thresholds, and replication or portability checks.
8. Paper or shadow trading demonstrates that realized fill quality, slippage, and operational behavior are not materially worse than the certified assumptions.
9. Capital is released in stages, with explicit loss caps, notional caps, and operator acknowledgements.
10. The entire capital-facing path is green in focused regression tests and exposes evidence artifacts that a user can inspect without reading source code.

If any one of those is false, the correct outcome is `research_only` or `paper_only`, not live money.

## Current Blocking Issues And Why They Matter

### Issue A: Post-selection backtest contamination in the demo workflow

In [example_automl.py](example_automl.py), the workflow runs AutoML and then rebuilds the canonical workflow with the selected config. In [core/pipeline.py](core/pipeline.py), `AutoMLStep.run(...)` applies `best_overrides` to the live pipeline config and clears downstream state, which means subsequent training, signal generation, and backtesting are re-run under the winning specification.

Why this matters:

1. A user can mistake a post-selection refit for final out-of-sample evidence.
2. Even if locked holdout or CPCV existed internally, the presented "final" backtest is no longer untouched evidence.
3. This is exactly the kind of workflow confusion that causes retail operators to over-trust a lucky model.
### Issue B: The safe path is not the accessible path

[example_automl.py](example_automl.py) is explicit research-only. It disables locked holdout, selection-policy gating, and overfitting controls. The harder path in [example_trade_ready_automl.py](example_trade_ready_automl.py) fails closed without Nautilus.

Why this matters:

1. The codebase may contain strong controls, but if the easiest runnable path bypasses them, the effective product behavior is still unsafe.
2. A retail operator is far more likely to run the demo script than to assemble a separate event-driven certification environment.
3. The minimum bar must be raised on the path a user can actually execute.

### Issue C: The AutoML summary and promotion contract is not reliable yet

The focused test slice still fails in [tests/test_automl_holdout_objective.py](tests/test_automl_holdout_objective.py):

1. selected `best_overrides` unexpectedly returns `None`
2. `locked_holdout["holdout_warning"]` is missing
3. `replication["promotion_pass"]` is missing

Why this matters:

1. A capital-facing system cannot have ambiguous selection outcomes.
2. Missing summary fields mean users and downstream tooling cannot tell whether promotion was denied, abstained, or silently degraded.
3. Promotion logic must be schema-bound and test-bound, not best-effort.

### Issue D: Research defaults are too permissive for capital-facing decisions

Current research defaults still allow looser data and execution behavior, including warning-only gap handling and permissive fill assumptions.

Why this matters:

1. The model can pass because the simulator was generous, not because the edge is robust.
2. Weak defaults are acceptable for exploration, but not for any path that leads toward money.
3. The repo must distinguish exploratory convenience from deployment evidence at the configuration level.
## Remediation Principles

These rules should govern every change below:

1. Evidence classes must be explicit and non-interchangeable.
2. Promotion must be binding and allowed to abstain.
3. The local capital-facing path must be stricter than the research path and equally easy to discover.
4. One component owns each temporal boundary.
5. Event-driven execution evidence outranks bar-surrogate evidence.
6. A passing backtest is not enough; paper evidence is mandatory before live money.
7. Any failure in lineage, summary schema, data certification, or execution realism must fail closed.

## Remediation Roadmap

The work is organized into eight workstreams. `P0` means money must not be risked until the workstream is complete.

| ID | Workstream | Priority | Outcome |
|---|---|---|---|
| WS-01 | Separate evidence classes and remove post-selection contamination | P0 | No demo or summary can present refit output as untouched OOS evidence |
| WS-02 | Build an accessible local certification path | P0 | A consumer-hardware user can run a strict local certification workflow |
| WS-03 | Repair and harden the AutoML summary contract | P0 | Promotion, holdout, and abstention reporting become deterministic and schema-valid |
| WS-04 | Tighten capital-facing data and config defaults | P0 | Capital-facing runs fail closed on data and stale-state defects |
| WS-05 | Bind validation architecture and statistical gating | P1 | Inner search, outer replay, holdout, and replication have distinct roles and gates |
| WS-06 | Upgrade execution realism using the local event-driven path | P0 | Fill quality, latency, queue, and timestamp semantics are materially closer to live trading |
| WS-07 | Add paper and shadow-live calibration gates | P0 | Backtest assumptions must be confirmed against realized paper behavior |
| WS-08 | Add a capital-release ladder and operator controls | P0 | Live money is released gradually, with explicit caps and rollback gates |

## WS-01: Separate Evidence Classes And Remove Post-Selection Contamination
### Objective

Make it impossible for the repo to present a post-selection refit as if it were final certification evidence.

### Exact Code Changes

1. In [core/pipeline.py](core/pipeline.py), stop mutating `pipeline.config` inside `AutoMLStep.run(...)` by default.
2. Replace the current implicit mutation with an explicit two-step process:

```python
summary = pipeline.run_automl()
refit = pipeline.refit_selected_candidate(summary)  # explicit, optional
```

3. Introduce a new explicit step or method such as `RefitSelectedCandidateStep` that:
   - takes a frozen `selection_snapshot`
   - applies `best_overrides`
   - rebuilds features and trains a final model
   - emits artifact metadata with `evidence_class = "post_selection_refit"`
4. Extend the AutoML summary in [core/automl.py](core/automl.py) so it returns separate top-level sections:

```python
{
    "selection_evidence": {...},
    "validation_replay_evidence": {...},
    "locked_holdout_evidence": {...},
    "replication_evidence": {...},
    "refit_artifact": None,
}
```

5. When a refit is later produced, do not write it back into the original AutoML summary as if it were evidence. Persist it as a sibling artifact.
6. In [example_automl.py](example_automl.py):
   - either stop after printing the AutoML evidence summary
   - or keep the rebuild path, but rename that stage to `Research Refit Artifact` and print a hard warning that it is not untouched OOS evidence
7. Add a hard `summary["evidence_class"]` or equivalent label to every backtest payload:
   - `search_cv_diagnostic`
   - `outer_replay`
   - `locked_holdout`
   - `replication`
   - `post_selection_refit`
   - `paper_shadow`
   - `live_runtime`

### Tests To Add Or Update

1. Add `tests/test_evidence_class_separation.py`:
   - `run_automl()` must not mutate the pipeline into a final backtest state by default
   - `refit_selected_candidate()` must mark outputs as `post_selection_refit`
2. Extend [tests/test_automl_holdout_objective.py](tests/test_automl_holdout_objective.py):
   - final selected evidence and post-selection refit must remain separate
3. Add `tests/test_example_automl_research_warning.py`:
   - the demo script must emit a non-promotable warning if it performs a refit

### Acceptance Criteria

1. A user cannot obtain a "final backtest" from the demo path without the output being labeled as a refit artifact.
2. The only promotable evidence classes are outer replay, locked holdout, replication, and later paper or live calibration artifacts.
3. No capital-facing report may use refit PnL as certification evidence.

## WS-02: Build An Accessible Local Certification Path
### Objective

Create a strict local workflow that a retail user can actually run on consumer hardware, without falling back to the research-only demo.

### External Research Translation

NautilusTrader documentation makes clear that local event-driven backtesting is feasible with either:

1. `BacktestNode` for repeated configuration-driven runs
2. `BacktestEngine.reset()` for parameter sweeps on the same data
3. streaming data ingestion when memory is limited

That means the repo does not need a remote or hidden "real Nautilus backend" to raise the bar. It needs a documented local certification path that uses Nautilus locally when installed.

### Exact Code Changes

1. Add a new user-facing script such as `example_local_certification_automl.py`.
2. Add a new helper in [example_utils.py](example_utils.py):
   - `build_local_certification_runtime_overrides(...)`
   - `build_local_certification_automl_overrides(...)`
3. Define a new evaluation mode distinct from both `research_only` and fully live-ready `trade_ready`, for example:
   - `evaluation_mode = "local_certification"`
4. The local certification profile must enable all of the following by default:
   - `locked_holdout_enabled = True`
   - `selection_policy.enabled = True`
   - `overfitting_control.enabled = True`
   - `objective = "risk_adjusted_after_costs"`
   - fail-closed data integrity
   - fail-closed data-quality quarantine
   - binding validation contract
   - binding replication
5. If Nautilus is installed locally, use it directly through a generated backtest configuration.
6. If Nautilus is not installed:
   - do not silently downgrade to the research demo
   - print a deterministic message that local certification requires Nautilus local installation
   - keep the research demo available, but separate
7. Add installation instructions for a local certification environment to [README.md](README.md) and [HOW_TO_USE.md](HOW_TO_USE.md).
8. Make the README show three distinct entry points:
   - `example_automl.py` = research demo only
   - `example_local_certification_automl.py` = local capital-facing certification
   - `example_trade_ready_automl.py` = stricter promotion path with the full event-driven stack and operations integration
### Tests To Add Or Update

1. Add `tests/test_local_certification_profile.py`:
   - confirms locked holdout, selection policy, overfitting controls, and fail-closed data policies are on
2. Add `tests/test_local_certification_requires_nautilus_or_aborts.py`:
   - no silent downgrade to the research demo
3. Add `tests/test_example_entrypoint_classification.py`:
   - each example script advertises exactly one evidence class and one risk level

### Acceptance Criteria

1. A retail user no longer has to choose between an unsafe demo and an inaccessible hardened path.
2. The local certification path is discoverable from the top-level docs.
3. The research demo can no longer be confused with a promotable workflow.

## WS-03: Repair And Harden The AutoML Summary Contract

### Objective

Make summary outputs deterministic, typed, and impossible to persist in a half-broken shape.

### Exact Code Changes

1. Introduce typed summary models in a dedicated module, for example `core/automl_contracts.py`.
2. At minimum define:
   - `SelectionOutcome`
   - `LockedHoldoutReport`
   - `ReplicationReport`
   - `PromotionEligibilityReport`
   - `AutoMLStudySummary`
3. Use either:
   - standard-library `dataclasses` plus explicit validation functions
   - or a runtime schema library such as Pydantic if the maintainers are willing to add the dependency
4. Ensure every capital-facing field has explicit required semantics:

```python
class SelectionOutcome:
    status: Literal["selected", "abstained", "config_error"]
    best_trial_number: int | None
    best_overrides: dict[str, Any]
    abstention_reasons: list[str]
```
5. Never rely on raw nested dictionaries assembled piecemeal near the end of [core/automl.py](core/automl.py).
6. Convert intermediate reports to typed objects first, then serialize them once at the end.
7. Add a single `validate_summary_contract(summary)` step before:
   - returning the summary
   - writing artifacts to disk
   - storing registry metadata
8. Replace late `RuntimeError` paths caused by policy rejection with structured abstention:
   - `status = "abstained"`
   - `best_overrides = {}`
   - `promotion_ready = False`
   - detailed abstention reasons preserved
9. Where a selected candidate exists, `best_overrides` must always be present and non-empty.
10. Preserve explicit booleans for fields already implied by the code, including:
   - `locked_holdout.holdout_warning`
   - `replication.promotion_pass`
   - `selection_policy.eligible`
   - `promotion_ready`

### Tests To Add Or Update

1. Fix the current failures in [tests/test_automl_holdout_objective.py](tests/test_automl_holdout_objective.py).
2. Add `tests/test_automl_summary_contract.py`:
   - required nested fields exist in selected, abstained, and configuration-error cases
3. Add `tests/test_automl_summary_serialization.py`:
   - artifact round-tripping preserves the same schema and required fields
4. Add `tests/test_automl_abstention_contract.py`:
   - no eligible candidate returns abstention instead of a crash

### Acceptance Criteria

1. The focused AutoML summary suite is green.
2. There is no path where promotion-related fields simply disappear from the returned summary.
3. Downstream tooling can treat the summary contract as stable.

## WS-04: Tighten Capital-Facing Data And Config Defaults

### Objective

Make all capital-facing profiles fail closed on data integrity and stale-state problems.

### Exact Code Changes

1. Introduce a clear profile split in [example_utils.py](example_utils.py):
   - `research_demo`
   - `local_certification`
   - `trade_ready`
2. Do not reuse permissive research defaults for certification profiles.
3. For all certification-capable profiles, enforce by default:
   - `data.gap_policy = "fail"`
   - `data.duplicate_policy = "fail"`
   - `data_quality.block_on_quarantine = True`
   - strict funding missing policy where applicable
   - reference validation or explicit absence declaration
4. Add a top-level `data_certification` artifact that summarizes:
   - gaps
   - duplicates
   - quarantine anomalies
   - stale context rates
   - funding coverage
   - reference coverage
   - whether the run is promotable from a data-quality perspective
5. Make promotion impossible when `data_certification.promotion_pass` is false.
6. Distinguish between research-only tolerances and certification tolerances in one place, not spread across helper functions.
7. Preserve missingness as missingness. Do not convert unknown state into tradable zeros.
### Tests To Add Or Update

1. Add `tests/test_certification_data_defaults.py`:
   - certification profiles fail closed on gaps and quarantine
2. Add `tests/test_data_certification_gate_binding.py`:
   - promotion must fail when data certification fails
3. Add `tests/test_missing_state_not_silently_zero_filled.py`:
   - stale or unavailable context must remain explicitly unknown

### Acceptance Criteria

1. No capital-facing run can proceed with warning-only gap handling.
2. Unknown state can no longer masquerade as economically valid state.
3. Data certification is visible in every capital-facing summary.

## WS-05: Bind Validation Architecture And Statistical Gating

### Objective

Separate search ranking from tradable evidence and make the promotion gates match what the external literature recommends.

### External Research Translation

1. Nested evaluation is required because tuning and evaluation on the same slice is optimistic.
2. Time-series validation needs gap ownership and purging or embargo where labels overlap.
3. DSR, minimum track record, and effective-bet logic are not optional once many trials are being searched.

### Exact Code Changes

1. Make the validation contract explicit and mandatory in all certification profiles.
2. The contract should have four distinct layers:
   - inner search ranking on CPCV or equivalent purged temporal splits
   - outer contiguous replay for tradable behavior on the search window
   - locked holdout consulted exactly once after freeze
   - replication on alternate windows and, when data exists, nearby symbols or venues
3. Selection must not be based solely on a single metric. Use a penalized ranking that incorporates:
   - cost-aware objective
   - trade count or effective bets
   - generalization gap
   - fragility
   - DSR or equivalent multiple-testing penalty
4. Default objective for certification-capable profiles must be `risk_adjusted_after_costs`, not raw Sharpe.
5. Expand gating so certification requires all of the following, not just one:
   - minimum trade or effective-bet count
   - non-negative or configured Sharpe confidence lower bound
   - DSR above threshold
   - minimum track record satisfied
   - replication pass rate satisfied
   - no blocking promotion gate failures
6. Restrict search space by evidence budget:
   - broad thesis search is only allowed in research mode
   - certification mode may tune only within a frozen thesis family unless the user explicitly chooses a wider research study
7. Preserve current good behavior where feature selection and stationarity transforms are fit on training slices only.
### Tests To Add Or Update

1. Extend the validation-source tests so the summary must say which evidence source drove selection.
2. Add `tests/test_min_track_record_gate.py`.
3. Add `tests/test_certification_requires_cost_aware_objective.py`.
4. Add `tests/test_replication_and_holdout_are_binding.py`.
5. Add `tests/test_thesis_space_disabled_in_certification.py` if not already enforced on the current branch.

### Acceptance Criteria

1. Search ranking, replay evidence, holdout evidence, and replication evidence remain distinct in both code and artifacts.
2. A candidate can no longer be promoted on raw Sharpe alone.
3. Certification mode cannot search thesis-level economic assumptions without the user explicitly opting into research.

## WS-06: Upgrade Execution Realism Using The Local Event-Driven Path

### Objective

Make certification use an event-driven engine with settings appropriate to the available data granularity.

### External Research Translation

Nautilus documentation provides a practical path forward:

1. use `BacktestNode` or `BacktestEngine` locally
2. set book type to match data granularity
3. for L1 or bar-only data, use fill models and conservative queue assumptions
4. align bar timestamps correctly to avoid execution timing bias
5. use `liquidity_consumption`, queue position, and price protection where appropriate

### Exact Code Changes

1. Add an execution-profile matrix in [core/pipeline.py](core/pipeline.py) or a dedicated execution config module:

| Profile | Intended Use | Minimum Data | Promotable |
|---|---|---|---|
| `research_surrogate` | idea screening | bars | no |
| `local_l1_certification` | local certification | quotes or bars plus conservative fill model | yes, up to paper-only gate |
| `local_l2_certification` | stronger local certification | L2 plus trades | yes |
| `trade_ready_event_driven` | strongest path | event-driven venue model | yes |

2. For bar-based certification paths:
   - ensure execution timestamps represent bar close, not open
   - if source data is indexed by open time, add a conversion layer for the execution engine only
   - keep feature-generation indexing unchanged if it already assumes open-time semantics causally
3. For L1 or bar-based certification paths, set conservative defaults:
   - `participation_cap <= 0.05`
   - `min_fill_ratio >= 0.25`
   - `liquidity_lag_bars >= 1`
   - non-zero slippage or probabilistic fill degradation
   - `price_protection_points` configured
4. For quote or trade data paths, enable:
   - `trade_execution = True` when trades are present
   - `queue_position = True` when queue modeling is available
   - `liquidity_consumption = True` to prevent duplicate use of displayed liquidity
5. For bar-only backtests, use adaptive bar ordering or a conservative fixed policy and label the result as bar-resolution evidence.
6. Add explicit order-type policies to capital-facing examples:
   - `IOC` or `FOK` for aggressive entries unless passive logic is intentional
   - `post_only` for maker-only assumptions
   - `reduce_only` for exits where supported
7. Add execution-quality artifacts with at least:
   - modeled fill ratio
   - modeled slippage
   - percent of volume-constrained orders
   - queue-related non-fills
   - latency model assumptions
### Tests To Add Or Update

1. Add `tests/test_bar_timestamp_execution_contract.py`.
2. Add `tests/test_local_certification_execution_defaults.py`.
3. Add `tests/test_queue_position_and_liquidity_consumption_binding.py`.
4. Add `tests/test_price_protection_gate.py`.

### Acceptance Criteria

1. Certification evidence never runs through the same permissive execution assumptions as the research demo.
2. The execution profile used is visible in summaries and examples.
3. Bar-data certification is explicitly labeled as lower-confidence than quote or order-book certification.

## WS-07: Add Paper And Shadow-Live Calibration Gates

### Objective

Require paper or shadow trading evidence before even micro-capital is released.

### Why Code Fixes Alone Are Not Enough

No backtest engine, however careful, can perfectly reproduce:

1. broker or venue rejects
2. throttling
3. symbol-level precision mismatches
4. order acknowledgements and cancels under load
5. real fill degradation under current network and routing conditions

That means the minimum bar must include paper evidence.

### Exact Code Changes

1. Add a first-class `paper_shadow_report` artifact to the readiness pipeline.
2. For each paper window, record:
   - submitted orders
   - accepted orders
   - rejected or denied orders
   - fill ratio
   - realized slippage vs modeled slippage
   - latency distribution
   - data gaps or stale-state breaches during runtime
   - PnL divergence vs certified expectation
3. Add a `paper_calibration` section to deployment readiness, with blocking thresholds such as:
   - `paper_days >= 30`
   - `effective_bets >= 100` across locked holdout plus paper, or `MinTRL` satisfied, whichever is stricter
   - `realized_slippage <= modeled_slippage * 1.20`
   - `realized_fill_ratio >= modeled_fill_ratio - 0.15`
   - `order_reject_rate <= 0.01`
   - zero unresolved data-certification breaches during paper
4. Promotion to live must fail when paper calibration is missing or red.
5. Persist paper calibration alongside model registry artifacts so the deploy decision is auditable.
### Tests To Add Or Update

1. Add `tests/test_paper_calibration_gate.py`.
2. Add `tests/test_readiness_requires_paper_shadow_report.py`.
3. Add `tests/test_slippage_drift_blocks_live_release.py`.

### Acceptance Criteria

1. No code path can move directly from certified backtest to live capital without paper evidence.
2. Paper calibration metrics are surfaced in deployment readiness.
3. The deploy decision becomes evidence-driven rather than narrative-driven.

## WS-08: Add A Capital-Release Ladder And Operator Controls

### Objective

Release capital gradually and conservatively instead of treating a passing certification run as permission for full-size deployment.

### Exact Code Changes

1. Implement explicit release stages in readiness artifacts:
   - `research_only`
   - `local_certified`
   - `paper_verified`
   - `micro_capital`
   - `scaled_live`
2. Add hard stage-gating rules:
   - `research_only -> local_certified`: all P0 workstreams complete, certification suite green
   - `local_certified -> paper_verified`: paper shadow report green
   - `paper_verified -> micro_capital`: operator acknowledgement plus live risk caps configured
   - `micro_capital -> scaled_live`: additional live calibration period green
3. Introduce conservative default live caps for `micro_capital`:
   - `max_gross_exposure = 0.5% of account equity`
   - `max_daily_loss = 0.25% of account equity`
   - `max_position_notional = min(0.5% of equity, 1% of trailing bar volume * close, venue minimum notional * 5)`
   - `max_open_orders_per_symbol` fixed and small
   - `max_strategy_instances = 1` by default for first live release
4. Require explicit kill-switch configuration:
   - daily loss halt
   - repeated reject halt
   - repeated stale-data halt
   - live slippage drift halt
   - operator manual disable
5. Add rollback readiness checks:
   - previous champion artifact available
   - deployable previous config available
   - operator can revert within one command or one config switch
### Tests To Add Or Update

1. Add `tests/test_release_ladder_gate_order.py`.
2. Add `tests/test_micro_capital_defaults.py`.
3. Add `tests/test_kill_switches_are_blocking.py`.
4. Add `tests/test_rollback_readiness_required.py`.

### Acceptance Criteria

1. Passing certification does not directly authorize scaled live trading.
2. The repo has explicit default behavior for micro-capital only.
3. All live scaling decisions are reversible and auditably gated.

## Immediate P0 Edits To Make First

These are the first changes that should happen before broader refactoring:

1. Fix the failing summary tests in [tests/test_automl_holdout_objective.py](tests/test_automl_holdout_objective.py).
2. Stop `AutoMLStep.run(...)` from silently rewriting the pipeline into a post-selection backtest state.
3. Rename or relabel [example_automl.py](example_automl.py) so users cannot confuse it with a promotable workflow.
4. Add `example_local_certification_automl.py` and surface it in [README.md](README.md).
5. Bind capital-facing profiles to fail-closed data defaults.
6. Require paper evidence before any live release stage can be green.

If those six edits are not done, nothing else in this document materially raises the minimum bar for risking money.

## Suggested Delivery Sequence

### Phase 0: Remove False Confidence

Complete WS-01, WS-02, WS-03, and WS-04 first.

Expected outcome:

1. the user-facing path is no longer misleading
2. the AutoML contract is reliable
3. local certification is discoverable
4. data defaults are appropriately strict

### Phase 1: Make Certification Technically Credible

Complete WS-05 and WS-06 next.

Expected outcome:

1. evidence sources are cleanly separated
2. certification is based on cost-aware, event-driven evidence rather than a bar surrogate alone

### Phase 2: Make Capital Release Operationally Credible

Complete WS-07 and WS-08 last.

Expected outcome:

1. paper evidence becomes binding
2. live money is released conservatively and reversibly
## Definition Of Done

This repo reaches the minimum bar for risking small amounts of money only when all of the following are true simultaneously:

1. The focused AutoML summary and promotion test suite is green.
2. The repo has a local certification example that is strict by default and easy to discover.
3. The demo path is unambiguously labeled as research-only and does not present post-selection refits as certification evidence.
4. Capital-facing profiles fail closed on gaps, duplicates, quarantine breaches, stale context, and missing funding or reference coverage.
5. Certification evidence is generated through an event-driven execution path appropriate to the available data granularity.
6. Paper or shadow-live calibration is green and bound into readiness.
7. Live release is limited to micro-capital first, with documented kill switches and rollback readiness.

Even then, the correct interpretation is still conservative:

The repo would be good enough to risk small money carefully, not good enough to skip paper trading, not good enough to skip micro-capital ramping, and not good enough to assume the model will survive regime change without ongoing monitoring.