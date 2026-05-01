# Regime Adaptation And Stress Control Plan

## Purpose

This document replaces the previous remediation backlog with a focused implementation plan for the three highest-leverage gaps that still remain in the research-to-promotion path:

1. Wire the regime-aware trainer into the AutoML trial loop.
2. Enforce hard regime-coverage gates so no single market state dominates admissibility.
3. Bind stressed scenario outcomes into promotion eligibility so failure under crash, drought, or unseen-regime conditions blocks deployment.

The intent is not to create parallel infrastructure. The intent is to harden the current pipeline, AutoML, and promotion surfaces so regime adaptation and stress realism are enforced where the system already decides whether a candidate is admissible.

## Institutional Research Basis

This plan is based on a small, isolated review of institutional guidance that is directly relevant to capital-facing model governance:

- NIST AI RMF: risk controls must be governed, measured, managed, and traceable. Unknown or unmeasured behavior should not be presented as trustworthy behavior.
- BIS BCBS 239: risk data and reports must be accurate, complete, timely, and adaptable under stress. Aggregations that hide concentrations or omit unstable slices are governance failures, not cosmetic defects.
- ESMA MiFID II Article 17: algorithmic trading systems must have effective controls, thresholds, testing, monitoring, and continuity arrangements that prevent disorderly behavior under extreme conditions.

For this repository, those principles translate into five implementation rules:

- Promotion-relevant evidence must be computed on the same causal timeline as inference.
- Regime diversity must be measured explicitly, not inferred from average performance.
- Specialist behavior under unseen regimes must degrade to a controlled fallback, never to silent extrapolation.
- Scenario stress must evaluate outcomes, not only whether a scenario name was configured.
- Missing or incomplete regime and stress evidence must fail closed in capital-facing paths.

## Delivery Rules

- Make the smallest change at the controlling abstraction point.
- Keep regime logic fold-local and inference-aligned.
- Prefer one canonical report per concern over multiple partially overlapping summaries.
- Use tri-state evidence semantics where relevant: `passed`, `failed`, `unknown`.
- Add regression tests before or alongside each behavior change.
- Work strictly in this order: item 1, then item 2, then item 3.

## Implementation Order

The work must be delivered in this sequence:

1. Regime-aware AutoML integration.
2. Regime-coverage admissibility gates.
3. Stress realism binding.

This order matters because item 2 depends on item 1 exposing regime-aware training artifacts, and item 3 should consume the same promotion-report structure rather than introducing a separate side channel.

## Item 1: Wire Regime-Aware Training Into AutoML

### Problem

The repository now contains a standalone regime-aware trainer, but AutoML trial execution still runs through the legacy training path. That means regime-aware feature conditioning and regime-specialist behavior can exist in isolation while candidate selection still optimizes non-regime-aware training summaries.

### Target Outcome

- AutoML trials can evaluate regime-aware candidates natively.
- Regime labels used during training are exactly the labels available at inference time on each fold.
- Trial summaries expose whether the candidate used a single regime-conditioned model or regime specialists.
- Specialist models must expose fallback behavior for unseen regimes, and that behavior must be visible in trial diagnostics.

### Best-Practice Design Requirements

- The regime signal is an input, not an ex post annotation.
- Model comparison must be apples-to-apples: same split geometry, same embargoes, same objective path, same holdout discipline.
- Specialist routing must be deterministic and auditable.
- Fallback usage under unseen regimes must be measured as a first-class control statistic.

### Files To Modify

- `core/pipeline.py`
- `core/regime_training.py`
- `core/automl.py`
- `core/__init__.py` only if exports change further
- `example_automl.py` and related examples only if configuration examples must surface the new mode

### Code Changes

#### 1. Add a single regime-aware training switch in the pipeline

In `core/pipeline.py`:

- Extend `TrainModelsStep` to recognize a `model.regime_aware` section.
- Supported strategies:
  - `feature`: add regime-conditioned features into one primary model.
  - `specialist`: fit specialist models by regime with a mandatory global fallback.
- The regime source for each fold must be the existing fold-local `regime_frame`, not any global preview artifact.
- Preserve the existing walk-forward, CPCV, purging, and calibration path.

#### 2. Reuse the standalone trainer instead of duplicating logic

In `core/regime_training.py`:

- Keep `train_regime_aware_model(...)` and `train_regime_aware_walk_forward(...)` as the canonical implementation.
- Add summary fields needed by AutoML and promotion:
  - `strategy`
  - `trained_regimes`
  - `trained_rows_by_regime`
  - `fallback_rows`
  - `unseen_regimes`
  - `coverage_summary`
  - `regime_alignment = "inference_aligned_input"`
- Ensure the specialist path records when a regime is skipped because of insufficient samples or single-class slices.

#### 3. Make AutoML search regime-aware candidates explicitly

In `core/automl.py`:

- Extend the search space to support `model.regime_aware.enabled` and `model.regime_aware.strategy`.
- Keep the search surface tight. Do not let AutoML explode into unconstrained combinations of strategy plus model family plus regime count.
- Thread the chosen regime-aware config into the executed pipeline overrides.
- Record regime-aware trial metadata in each trial report so later gate logic can inspect it without reparsing nested training payloads.

#### 4. Preserve current behavior as the default unless explicitly enabled

- Existing runs that do not set `model.regime_aware.enabled` should continue to use the legacy path.
- Trade-ready or certification examples may opt in later, but this item should not silently switch all existing examples to specialist training.

### Tests To Add Or Modify

- Add `tests/test_automl_regime_aware_training.py`
  - `test_automl_feature_strategy_executes_regime_aware_training`
  - `test_automl_specialist_strategy_records_fallback_usage`
  - `test_regime_aware_trials_preserve_fold_local_regime_alignment`
- Extend `tests/test_regime_aware_training.py`
  - assert the pipeline-facing summary shape needed by AutoML is stable
- Add or extend a narrow AutoML dummy-pipeline test
  - assert trial reports include regime-aware metadata without requiring the full study path

### Acceptance Criteria

- AutoML can run a regime-aware feature strategy trial and a specialist strategy trial.
- Trial summaries expose fallback usage and unseen-regime handling.
- No regime-aware trial can train on regime labels that were not available on the fold-local inference timeline.

## Item 2: Add Hard Regime-Coverage Gates To Candidate Acceptance

### Problem

Even with regime-aware training, AutoML can still overfit to the dominant regime if admissibility only looks at aggregate objective values. A candidate that works only in a long bull or high-liquidity period can still look excellent when evaluated on pooled averages.

### Target Outcome

- Candidate admissibility must fail when training or validation regime coverage is too concentrated.
- Coverage failures must be explicit in the promotion report, not buried in diagnostics.
- Missing coverage evidence must resolve to `unknown`, and `unknown` must block capital-facing paths.

### Best-Practice Design Requirements

- Regime diversity is a governance control, not a descriptive statistic.
- Coverage must be measured on train and validation slices separately.
- The gate must be deterministic and threshold-driven.
- Thresholds must be configurable but binding by default in the hardened policy profile.

### Files To Modify

- `core/regime_training.py`
- `core/pipeline.py`
- `core/automl.py`
- `core/promotion.py` only if gate normalization needs new metadata
- `tests/test_selection_policy_defaults.py`

### Code Changes

#### 1. Standardize a regime-coverage report contract

In `core/regime_training.py`:

- Promote `summarize_regime_coverage(...)` to the canonical coverage contract.
- Ensure it emits:
  - `status = "passed" | "failed" | "unknown"`
  - `promotion_pass`
  - `distinct_regimes`
  - `dominant_regime`
  - `dominant_share`
  - `regime_distribution`
  - `reasons`
- Treat missing regime data or an empty slice as `unknown`, not `passed`.

#### 2. Persist fold-local coverage into training summaries

In `core/pipeline.py`:

- For each fold, compute regime coverage on:
  - fit window
  - validation window when present
  - test window for diagnostics
- Aggregate fold coverage into a top-level `training["regime"]["coverage_summary"]` payload.
- Add fold-level reasons for:
  - insufficient distinct regimes
  - dominant regime above threshold
  - specialist regime sample shortfall

#### 3. Bind coverage into the hardened AutoML selection policy

In `core/automl.py`:

- Add a new gate: `regime_coverage`.
- Resolve it through the same `_resolve_evidence_gate(...)` path used for portability and regime stability.
- Add a hardened default gate mode:
  - `regime_coverage = "blocking"`
- Add default thresholds in the selection policy, for example:
  - `min_distinct_regimes = 2`
  - `max_dominant_regime_share = 0.80`
  - optional per-fold minimum sample threshold for specialist mode
- Use the new coverage summary to populate `eligibility_checks`, the selection gate report, and the promotion eligibility report.

#### 4. Keep research and certification semantics distinct

- Research-only mode may surface `unknown` coverage as advisory.
- Capital-facing modes must treat `unknown` and `failed` coverage as blocking.

### Tests To Add Or Modify

- Add `tests/test_regime_coverage_gate.py`
  - `test_dominant_regime_blocks_selection`
  - `test_missing_regime_coverage_is_unknown_and_blocking_in_trade_ready`
  - `test_balanced_regime_coverage_passes_gate`
- Extend `tests/test_selection_policy_defaults.py`
  - assert `regime_coverage` is blocking under the hardened default profile
- Extend an AutoML gate test such as `tests/test_automl_holdout_objective.py`
  - assert the new gate appears in `promotion_eligibility_report.gate_status`

### Acceptance Criteria

- A dominant-regime trial cannot become promotion-eligible in a capital-facing path.
- Missing coverage evidence is surfaced as `unknown`, not silently green.
- The gate is visible in both selection checks and the canonical promotion report.

## Item 3: Bind Stress Results Into Promotion Eligibility

### Problem

The current stress realism gate checks whether required scenarios were configured, but it does not judge whether the candidate remains operationally acceptable under those scenarios. Configuration alone is not a meaningful control for flash crashes, liquidity droughts, stale marks, halts, or unseen-regime fallback behavior.

### Target Outcome

- Promotion eligibility must depend on stressed outcomes, not only scenario presence.
- Stress scenarios must include regime-relevant failure modes:
  - unseen-regime fallback pressure
  - low-liquidity or fill drought conditions
  - sudden volatility or crash conditions
- Candidates that breach configured stress thresholds must fail promotion.

### Best-Practice Design Requirements

- Stress tests must be outcome-based and threshold-bound.
- Scenario evidence must be reproducible and attached to the promotion report.
- The gate must fail closed when required stressed metrics are missing.
- Operational degradation under stress must be distinguished from ordinary performance variability.

### Files To Modify

- `core/scenarios.py`
- `core/backtest.py`
- `core/promotion.py`
- `core/automl.py`
- `example_trade_ready_automl.py` and similar scripts only if default required scenarios must be expanded

### Code Changes

#### 1. Enrich the scenario matrix summary contract

In `core/backtest.py` and `core/scenarios.py`:

- Expand `stress_matrix` summaries to expose scenario outcome metrics needed for gating:
  - `worst_net_profit_pct`
  - `worst_sharpe_ratio`
  - `worst_max_drawdown`
  - `worst_fill_ratio` when execution evidence is available
  - per-scenario outcome summaries
- Add support for tagging scenarios by control intent, not just name:
  - `regime_transition`
  - `liquidity_drought`
  - `volatility_spike`
  - `venue_failure`

#### 2. Make the stress realism gate outcome-aware

In `core/promotion.py`:

- Extend `evaluate_stress_realism_gate(...)` to fail when any required stress metric breaches policy thresholds.
- Add policy thresholds such as:
  - `max_stress_drawdown`
  - `min_stress_fill_ratio`
  - `min_stress_trade_count`
  - `require_unseen_regime_fallback_bound`
  - `max_unseen_regime_fallback_share`
- Distinguish three failure classes:
  - scenarios missing
  - scenarios incomplete
  - stressed outcome unacceptable

#### 3. Bind fallback behavior from regime-aware trials into the stress gate

In `core/automl.py`:

- When a best trial is regime-aware, feed specialist fallback diagnostics into the post-selection stress evaluation.
- Treat excessive fallback under unseen-regime scenarios as a blocking stress failure.
- Persist the enriched stress gate details into `selection_report["diagnostics"]["stress_realism"]` and the canonical promotion report.

#### 4. Harden default required scenarios for capital-facing paths

- Keep the current required base scenarios.
- Add capital-facing defaults for:
  - crash or volatility spike
  - liquidity drought
  - unseen-regime fallback pressure when regime-aware specialists are used

### Tests To Add Or Modify

- Extend `tests/test_promotion_gate_binding.py`
  - `test_stress_realism_blocks_unacceptable_drawdown_even_when_scenarios_exist`
  - `test_stress_realism_blocks_low_fill_ratio_drought`
  - `test_stress_realism_blocks_unseen_regime_fallback_breach`
- Extend `tests/test_exchange_failure_scenarios.py`
  - assert the stress matrix summary exposes the new stressed outcome fields
- Add `tests/test_stress_realism_thresholds.py`
  - verify missing scenario names and breached metrics fail for different reasons

### Acceptance Criteria

- A capital-facing candidate cannot pass promotion only because required scenario names were present.
- Stress reports expose breached thresholds and the exact failing scenario class.
- Regime-aware specialist fallback under unseen regimes is promotion-relevant, not diagnostic-only.

## Focused Validation Order

Run targeted test batches after each item rather than waiting for the full suite.

### After Item 1

- `python -m pytest tests/test_regime_aware_training.py tests/test_automl_regime_aware_training.py`

### After Item 2

- `python -m pytest tests/test_regime_coverage_gate.py tests/test_selection_policy_defaults.py tests/test_automl_holdout_objective.py`

### After Item 3

- `python -m pytest tests/test_promotion_gate_binding.py tests/test_exchange_failure_scenarios.py tests/test_stress_realism_thresholds.py`

### Final Slice

- `python -m pytest tests/test_regime_aware_training.py tests/test_automl_regime_aware_training.py tests/test_regime_coverage_gate.py tests/test_promotion_gate_binding.py tests/test_exchange_failure_scenarios.py tests/test_automl_holdout_objective.py`

## Definition Of Done

These three issues are materially addressed only when all of the following are true:

- AutoML can execute regime-aware trials through the canonical training path.
- Regime-aware trial reports expose fallback behavior and inference-aligned regime usage.
- Candidate admissibility fails when regime coverage is missing or overly concentrated in capital-facing modes.
- Stress realism evaluates stressed outcomes, not only scenario presence.
- Unseen-regime fallback, liquidity droughts, and crash behavior can each block promotion when thresholds are breached.
- The canonical promotion report records these failures explicitly and fail-closed.