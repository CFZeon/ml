# TODO

## Purpose

This file is the current remediation plan as of 2026-05-04.
It replaces the older backlog with a priority-ordered checklist based on the
verified live defects from the code audit, not on stale findings that have
already been fixed.

The ordering is by expected research-integrity impact for this repository:

1. Can the system admit a strategy using evidence that is not causally valid?
2. Can a strategy pass promotion without complete regime or stress evidence?
3. Can the research path produce optimistic execution or data assumptions that
    survive into operator-facing summaries?
4. Can the runtime or deployment path silently weaken controls that a retail
    trader on consumer hardware would assume are binding?

## Institutional Guidance Distilled Into Repository Rules

The implementation order below is grounded in a small isolated review of
institutional governance guidance that is relevant to this codebase.

1. NIST AI RMF: trustworthiness claims must be measurable, governed, and tied
    to documented evidence rather than inferred from missing or partial reports.
2. BCBS 239: risk evidence used for decisions must be complete, accurate,
    timely, and stress-aware; incomplete aggregation is a control failure.
3. MiFID II Article 17 style controls: algorithmic systems need explicit
    testing, thresholds, monitoring, and fail-safe behavior under stressed or
    disorderly conditions.

For this repository, those principles become concrete engineering rules:

1. Promotion-relevant evidence must be generated on the same causal timeline as
    inference.
2. Unknown regime, stress, execution, or data integrity evidence must not be
    treated as passing evidence in capital-facing paths.
3. Stress realism must evaluate stressed outcomes, not only scenario presence.
4. Research defaults may be advisory, but any optimistic fallback must be
    explicit, visible, and blocked from promotion by default.
5. Consumer-hardware constraints matter: controls must remain deterministic,
    bounded in runtime, and usable without institutional infrastructure.

## Execution Rules

1. Work one issue at a time.
2. Fix the controlling abstraction, not downstream symptoms.
3. Run the narrowest test slice after each substantive change.
4. Keep research-only flexibility, but make capital-facing paths fail closed.
5. Prefer deterministic defaults and bounded compute for retail hardware.

## Priority Checklist

### P0.1 Wire regime-aware training into the canonical pipeline

Status: pending
Research-integrity impact: highest

Problem:
`core/regime_training.py` contains a regime-aware training implementation, but
`TrainModelsStep` in `core/pipeline.py` still uses the legacy path. AutoML can
therefore report regime-aware configuration without the canonical training loop
actually using the regime-aware trainer.

Implementation targets:

1. Add a single `model.regime_aware` switch in `TrainModelsStep`.
2. Route fold-local regime data into the regime-aware trainer.
3. Preserve the existing split, purge, calibration, and backtest flow.
4. Surface regime-aware metadata in the training summary:
    `enabled`, `strategy`, `folds`, `coverage_summary`, `fallback_rows`,
    `unseen_regimes`, `trained_regimes`, `trained_rows_by_regime`.

Acceptance slice:

1. `tests/test_automl_regime_aware_training.py`

### P0.2 Add hard regime-coverage admissibility gates

Status: pending
Research-integrity impact: highest

Problem:
Even with regime-aware training available, the selection path in
`core/automl.py` does not currently block candidates that are dominated by a
single regime or missing regime coverage evidence.

Implementation targets:

1. Standardize regime coverage summaries as the canonical contract.
2. Persist fit, validation, and test coverage into the training payload.
3. Add `regime_coverage` to hardened AutoML gate defaults.
4. Treat missing coverage as `unknown` and blocking in capital-facing modes.

Acceptance slice:

1. `tests/test_regime_coverage_gate.py`

### P0.3 Bind stressed outcomes into promotion eligibility

Status: pending
Research-integrity impact: highest

Problem:
`core/promotion.py` can evaluate stressed outcomes, but the backtest stress
summary in `core/backtest.py` does not yet emit the full metric and control
intent contract that the gate expects. AutoML also does not pass regime-aware
fallback diagnostics into the stress gate.

Implementation targets:

1. Expand stress summaries with `worst_max_drawdown`, `worst_fill_ratio`,
    `worst_trade_count`, per-scenario control tags, and control intents.
2. Ensure capital-facing backtests produce promotion-ready stress payloads.
3. Thread regime-aware fallback evidence into stress evaluation when relevant.

Acceptance slice:

1. `tests/test_exchange_failure_scenarios.py`
2. `tests/test_promotion_gate_binding.py`
3. `tests/test_stress_realism_thresholds.py`

### P1.1 Fail closed on optimistic backtest fallbacks

Status: pending
Research-integrity impact: high

Problem:
The public backtest API can still fall back to same-bar close execution and can
quietly switch from VectorBT to the pandas adapter when VectorBT is missing.

Implementation targets:

1. Make same-bar execution fallback opt-in only.
2. Require explicit engine parity or fail closed in capital-facing modes.
3. Stamp engine usage and fallback assumptions into all backtest summaries.

### P1.2 Tighten research data integrity defaults

Status: pending
Research-integrity impact: high

Problem:
Gap handling and anomaly quarantine are still permissive enough to let exchange
artifacts flow into features, labels, and research summaries.

Implementation targets:

1. Tighten research defaults from silent permissiveness toward explicit
    quarantine or dropped windows for continuity-sensitive logic.
2. Ensure labeling and rolling features cannot span missing-bar discontinuities.
3. Carry quarantine lineage into downstream summaries.

### P1.3 Decouple calibration fitting from model selection

Status: pending
Research-integrity impact: high

Problem:
Primary and meta calibration parameters are still tuned inside the same
selection loop used for model ranking.

Implementation targets:

1. Keep model-family search and probability calibration on separate evidence
    slices.
2. Preserve retail-hardware feasibility by reusing fold-local validation data
    with explicit calibration sub-splits rather than expensive nested studies.

### P1.4 Add sample-qualified evaluation metrics

Status: pending
Research-integrity impact: medium-high

Problem:
Scalar metrics such as profit factor, Brier score, and Calmar are still easier
to over-interpret than they should be on short crypto samples.

Implementation targets:

1. Add Brier decomposition.
2. Suppress or downgrade metrics with insufficient realized trade counts.
3. Surface trade-level and portfolio-level risk summaries side by side.

### P2.1 Wire feature lag reach into split geometry

Status: pending
Research-integrity impact: medium

Problem:
Lag-aware embargo support exists in the splitters, but the canonical pipeline
does not yet feed feature-derived `max_lag` into split construction.

Implementation targets:

1. Compute effective lag reach from the feature graph.
2. Extend walk-forward and CPCV split calls with that lag reach.
3. Fail when embargo or gap is shorter than the feature reach.

### P2.2 Bound fractional differentiation for short samples

Status: pending
Research-integrity impact: medium

Problem:
Fractional differentiation can still consume too much history on short samples
and collapse usable rows.

Implementation targets:

1. Add `max_lag` and minimum retained-sample controls.
2. Report retention ratios per transformed feature.
3. Reject transforms that leave too little usable data.

### P2.3 Harden drift and deployment controls for retail operation

Status: completed
Research-integrity impact: medium

Problem:
Cooldowns remain static in bar counts, retrained models can go live without a
burn-in stage, and research monitoring defaults are still too permissive.

Implementation targets:

1. Normalize cooldown and minimum-sample rules by interval and trade rate.
2. Add post-retrain shadow or paper warm-up before capital-facing sizing.
3. Replace effectively infinite research monitoring defaults with visible
    advisory bounds.

### P2.4 Add retail-safe operational guards

Status: completed
Research-integrity impact: medium

Problem:
Rate-limit-aware retraining orchestration and mandatory symbol-filter evidence
are still incomplete for retail deployment constraints.

Implementation targets:

1. Add Binance request-weight-aware retraining scheduling.
2. Make symbol-filter availability mandatory for paper and trade-ready sizing.
3. Treat missing venue constraints as unknown execution evidence, not success.

## Implementation Order For This Run

1. P0.1 regime-aware training integration
2. P0.2 regime-coverage admissibility gate
3. P0.3 stress realism binding
4. P1.1 optimistic backtest fallback hardening
5. P1.2 research data integrity tightening
6. P1.3 calibration-selection separation
7. P1.4 sample-qualified metric hardening
8. P2.1 lag-aware split wiring
9. P2.2 fractional differentiation bounds
10. P2.3 drift and warm-up hardening
11. P2.4 retail operational guardrails

## Definition Of Done

The backlog is complete only when:

1. The regime-aware trainer is used by the canonical training path.
2. Regime coverage and stressed outcomes can block promotion.
3. Capital-facing runs fail closed on optimistic execution and missing evidence.
4. Research metrics and data-integrity summaries make their limitations explicit.
5. Retail deployment paths remain deterministic, rate-aware, and operationally
    bounded on consumer hardware.