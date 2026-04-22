# IMPROVEMENTS - Remediation Implementation Program

## Scope

- This file replaces the prior gap-ranking memo with an execution plan.
- The plan is severity-ranked and implementation-oriented.
- Every item includes code targets, tests, acceptance criteria, and sequencing constraints.
- The operating assumption is a retail trader on consumer hardware, so every fix must remain CPU-friendly, deterministic, and bounded in memory/runtime.

## Execution Rule

- Work the list top to bottom.
- For each item: finish research, resolve user-facing design questions, implement, validate, and document before moving on.
- Do not promote realism through reporting alone. Any safeguard that matters must bind selection, backtesting, or deployment behavior.

## P0.1 - Separate CPCV Diagnostics From Executable Performance

### Why this is first

- The current CPCV flow can make a non-executable distribution of path results look like one tradable strategy.
- This contaminates optimization, reporting, and any downstream capital-allocation decision.

### Objective

- Keep CPCV as an overfitting and robustness diagnostic.
- Stop presenting averaged CPCV path results as executable portfolio performance.
- Ensure that any metric used for deployment, promotion, or Kelly sizing comes from one causal chronological evaluation path.

### Required implementation changes

1. Training-output contract hardening
	- Update [core/pipeline.py](core/pipeline.py) so `validation.method == "cpcv"` produces a diagnostic object, not an implied tradable OOS stream.
	- Preserve per-path predictions, probabilities, signals, and backtests under a clearly named diagnostic namespace such as `cpcv_paths` or `cpcv_diagnostics`.
	- Remove or deprecate any field name that implies a single combined OOS series when the source is CPCV.

2. Backtest summary semantics
	- Replace `_summarize_path_backtests()` with a distribution summary that reports `median`, `p10`, `p90`, `min`, `max`, and path count for each metric.
	- Remove averaged p-values and averaged confidence intervals entirely.
	- Keep per-path significance results as raw path diagnostics only.

3. Executable evaluation path
	- Add an explicit executable evaluation mode for promotion and deployment, likely one of:
	  - locked walk-forward validation slice
	  - locked post-search holdout slice
	  - explicit rolling-forward replay after final model freeze
	- Require AutoML selection and registry promotion to resolve to that executable path for any metric that drives capital or promotion.

4. Reporting and UX cleanup
	- Mark CPCV outputs as `diagnostic_only` in [core/automl.py](core/automl.py), [core/pipeline.py](core/pipeline.py), and any example summaries.
	- Ensure markdown summaries and artifacts never call the CPCV mean a tradable Sharpe, tradable net profit, or production-equivalent result.

### Code surfaces

- [core/pipeline.py](core/pipeline.py)
- [core/automl.py](core/automl.py)
- [core/backtest.py](core/backtest.py) if significance payload shapes need adjustment
- [example_automl.py](example_automl.py) and any example summary utilities that currently print headline CPCV performance

### Test plan

- Add `tests/test_cpcv_executable_separation.py`
  - Verify CPCV outputs are tagged `diagnostic_only`.
  - Verify no single combined OOS signal series is emitted for CPCV unless explicitly produced by a causal replay step.
  - Verify path summaries report quantiles and path counts, not means of p-values or confidence intervals.
- Extend [tests/test_cpcv_validation.py](tests/test_cpcv_validation.py)
  - Assert CPCV backtest summary uses distribution metrics.
  - Assert promotion-facing metrics come from the executable evaluation slice, not CPCV path aggregation.
- Add regression tests in [tests/test_automl_holdout_objective.py](tests/test_automl_holdout_objective.py)
  - Verify AutoML objective ranking changes when CPCV diagnostics and executable holdout disagree.

### Acceptance criteria

- CPCV results are clearly diagnostic and cannot be mistaken for one tradable path.
- No averaged p-value or averaged confidence interval survives in public outputs.
- AutoML and registry decisions use one causal executable evaluation source.

## P0.2 - Remove Cross-Fold Leakage From Sizing And Signal Policy Calibration

### Why this is second

- Even a good model-selection stack is invalid if fold-level sizing uses future fold outcomes.
- The current leakage is subtle and therefore dangerous: it can survive ordinary accuracy checks.

### Objective

- Ensure fold-local Kelly inputs, profitability thresholds, and policy calibration only use information available before the evaluated decision time.
- Preserve useful calibration while eliminating fold-order contamination.

### Required implementation changes

1. Fold-state redesign
	- Refactor [core/pipeline.py](core/pipeline.py) so fold calibration state is keyed by timestamp ordering, not loop order.
	- Under CPCV, disallow borrowing from previously processed folds unless their timestamps are strictly earlier than the current fold’s calibration window.

2. Calibration source policy
	- Define one explicit policy per validation method:
	  - walk-forward: prior realized OOS outcomes allowed if strictly earlier in time
	  - CPCV: no cross-path borrowing by default; use validation subset only or static conservative defaults
	- Make that policy visible in training artifacts.

3. Kelly and signal-policy safeguards
	- Force fallback to conservative static sizing when causal trade-outcome evidence is unavailable.
	- Persist why the fallback triggered and how many trades informed the active sizing inputs.

4. Documentation of causal provenance
	- Add provenance fields such as `sizing_stats_source`, `causal_calibration_rows`, and `causal_cutoff_timestamp` to each fold artifact.

### Code surfaces

- [core/pipeline.py](core/pipeline.py)
- [core/backtest.py](core/backtest.py) only if outcome-frame metadata needs to expand
- [core/models.py](core/models.py) if fold metadata helpers are promoted there

### Test plan

- Add `tests/test_fold_local_sizing_causality.py`
  - Verify CPCV folds cannot consume outcomes from future timestamps.
  - Verify walk-forward folds may only consume strictly earlier realized outcomes.
  - Verify fallback sizing is used when no causal calibration data exists.
- Extend [tests/test_signal_profitability_sizing.py](tests/test_signal_profitability_sizing.py)
  - Assert reported `kelly_trade_count` and calibration provenance match the allowed evidence set.
- Add a synthetic adversarial test
  - Create folds where future outcomes would obviously improve sizing and confirm the guarded implementation refuses to use them.

### Acceptance criteria

- Fold-level signal policy and Kelly sizing are causal under both walk-forward and CPCV.
- Calibration provenance is explicit and testable.
- When causal evidence is unavailable, the system degrades conservatively rather than silently borrowing future data.

## P0.3 - Make Execution Realism Binding, Not Cosmetic

### Why this is third

- Incorrect fills corrupt every downstream statistic, even if validation and leakage controls are perfect.
- The current adapter boundary advertises more realism than the engine actually provides.

### Objective

- Either integrate a true event-driven execution simulator or make the bar-based surrogate explicitly conservative and promotion-limited.
- Prevent the system from claiming production-grade realism when it does not have it.

### Required implementation changes

1. Execution-mode contract
	- Introduce explicit execution modes such as `bar_surrogate`, `conservative_bar_surrogate`, and `event_driven`.
	- Attach the mode to every backtest, training summary, registry artifact, and promotion decision.

2. Conservative surrogate hardening
	- Tighten [core/backtest.py](core/backtest.py) to support stricter latency, participation caps, stale-book rejection, order expiry, and asymmetric adverse selection defaults.
	- Treat unsupported semantics as blockers for promotion when the strategy depends on them.

3. Promotion gating
	- Add a blocking gate so models depending on unsupported execution semantics cannot become `promotion_ready`.
	- Surface this in [core/automl.py](core/automl.py), [core/promotion.py](core/promotion.py), and [core/registry/store.py](core/registry/store.py).

4. Optional event-driven path
	- If Nautilus integration is chosen, implement a real execution adapter behind [core/execution/nautilus_adapter.py](core/execution/nautilus_adapter.py) and route supported scenarios through it.
	- If not, keep the adapter honest: it must report that it is a surrogate and what venue mechanics are missing.

### Code surfaces

- [core/backtest.py](core/backtest.py)
- [core/execution/nautilus_adapter.py](core/execution/nautilus_adapter.py)
- [core/automl.py](core/automl.py)
- [core/promotion.py](core/promotion.py)
- [core/registry/store.py](core/registry/store.py)

### Test plan

- Add `tests/test_execution_realism_gate.py`
  - Verify unsupported execution assumptions block promotion.
  - Verify surrogate mode is recorded in artifacts and summaries.
- Extend [tests/test_execution_partial_fills.py](tests/test_execution_partial_fills.py)
  - Cover stricter latency, expiry, and participation-cap behavior.
- Extend [tests/test_execution_adapter_parity.py](tests/test_execution_adapter_parity.py)
  - Ensure parity tests only apply where the surrogate is expected to match the legacy bar model.

### Acceptance criteria

- Every reported backtest declares its execution realism tier.
- Unsupported execution assumptions cannot silently pass as production-ready.
- Promotion behavior changes when execution realism is insufficient.

## P1.4 - Add Market-Integrity And Multi-Venue Truth Controls

### Objective

- Reduce the chance that the model learns exchange-local manipulation, stale prints, or non-portable liquidity artifacts.

### Required implementation changes

1. Market-integrity feature and quarantine layer
	- Extend [core/data_quality.py](core/data_quality.py) with crypto-specific integrity checks:
	  - abnormal trade-count-to-volume ratios
	  - suspicious quote-volume disagreement clusters
	  - abrupt venue-local price divergence versus reference overlays
	  - repeated zero-spread or mechanically flat prints during active trading hours
	- Support blocking, advisory, and quarantine-only modes.

2. Cross-venue truth construction
	- Replace simple median-based overlays in [core/reference_data.py](core/reference_data.py) with a configurable composite policy:
	  - median
	  - liquidity-weighted median/mean
	  - strict quorum
	- Track missing-venue share and reference disagreement over time.

3. Selection and promotion enforcement
	- Bind cross-venue integrity and market-integrity failures into selection eligibility, not just reporting.

### Code surfaces

- [core/data_quality.py](core/data_quality.py)
- [core/reference_data.py](core/reference_data.py)
- [core/pipeline.py](core/pipeline.py)
- [core/automl.py](core/automl.py)

### Test plan

- Add `tests/test_market_integrity_controls.py`
  - Verify suspicious venue-local anomalies trigger configured actions.
  - Verify cross-venue composite behavior changes under different quorum and weighting policies.
- Extend [tests/test_cross_venue_reference_validation.py](tests/test_cross_venue_reference_validation.py)
  - Cover blocking behavior for partial coverage and severe divergence.

### Acceptance criteria

- Venue-local anomalies can block, quarantine, or downgrade research runs.
- Cross-venue validation is configurable and enforceable.
- Promotion can no longer ignore failed market-integrity evidence.

## P1.5 - Add TTL And Stale-State Enforcement For Context Features

### Objective

- Stop old futures, reference, or custom-data states from being silently forward-filled into current decisions.

### Required implementation changes

1. TTL policy layer
	- Add explicit freshness and TTL controls for futures context, reference overlays, and custom datasets.
	- Fail closed or null out stale features after the configured age budget.

2. Alignment metadata
	- Persist feature ages, stale-row counts, and fallback rates into pipeline state and monitoring artifacts.

3. Selection gating
	- Block feature blocks with excessive stale usage from model selection or promotion.

### Code surfaces

- [core/context.py](core/context.py)
- [core/reference_data.py](core/reference_data.py)
- [core/data.py](core/data.py)
- [core/monitoring.py](core/monitoring.py)
- [core/pipeline.py](core/pipeline.py)

### Test plan

- Add `tests/test_context_ttl_enforcement.py`
  - Verify stale futures context is nulled or blocked after TTL.
  - Verify monitoring captures stale usage rates.
- Extend [tests/test_derivatives_context_pipeline.py](tests/test_derivatives_context_pipeline.py)
  - Cover TTL expiry and stale-mark behavior.

### Acceptance criteria

- No context feature can remain implicitly current forever.
- Staleness is visible in artifacts and enforceable in selection.

## P1.6 - Operationalize Drift-Triggered Retraining And Rollback

### Objective

- Turn drift detection from an advisory report into an operational control loop with bounded risk.

### Required implementation changes

1. Drift artifact generation
	- Run [core/drift.py](core/drift.py) as part of the normal research or promotion lifecycle where appropriate.
	- Persist comparable reference windows, evidence counts, and cooldown state.

2. Retraining workflow
	- Add an orchestrated retrain path that:
	  - triggers on drift or schedule
	  - re-runs evaluation and promotion gates
	  - preserves champion/challenger state
	  - refuses automatic promotion without passing executable holdout criteria

3. Rollback and live-safe controls
	- Extend registry logic so operational degradation can freeze promotion or recommend rollback to the last healthy champion.

### Code surfaces

- [core/drift.py](core/drift.py)
- [core/registry/store.py](core/registry/store.py)
- [core/monitoring.py](core/monitoring.py)
- orchestration entry points or new scheduled-job helpers if added

### Test plan

- Add `tests/test_drift_retraining_workflow.py`
  - Verify drift evidence can trigger retraining recommendation.
  - Verify cooldown and minimum-sample gates work.
  - Verify rollback recommendation occurs when challenger fails promotion after drift.
- Extend [tests/test_drift_monitoring.py](tests/test_drift_monitoring.py)
  - Cover persisted reference-window and evidence metadata.

### Acceptance criteria

- Drift signals can feed a deterministic retrain decision.
- Champion/challenger and rollback behavior is test-covered.
- Drift handling is no longer advisory-only.

## P1.7 - Harden Selection Against Point-Estimate Noise And Metric Mismatch

### Objective

- Reduce the chance that AutoML and registry promotion still reward noisy point estimates or compare mismatched scores.

### Required implementation changes

1. Objective hardening
	- Make confidence-lower-bound or equivalent uncertainty-aware scoring the default for trading-first objectives.
	- Require explicit opt-out for point-estimate-first selection.

2. Registry metric alignment
	- Ensure challenger and champion are always compared on the same score basis.
	- Remove any validation-vs-holdout apples-to-oranges comparison path.

3. Governance binding
	- Bind all already-computed portability, regime-stability, and cross-venue gates into actual eligibility and promotion logic.

### Code surfaces

- [core/automl.py](core/automl.py)
- [core/promotion.py](core/promotion.py)
- [core/registry/store.py](core/registry/store.py)
- [core/registry/manifest.py](core/registry/manifest.py)

### Test plan

- Add `tests/test_selection_score_alignment.py`
  - Verify champion and challenger comparisons use the same score basis.
  - Verify lower-bound selection is default for trading objectives.
  - Verify portability and regime-stability failures block promotion when configured as blocking.
- Extend [tests/test_local_registry_flow.py](tests/test_local_registry_flow.py)
  - Cover score-basis persistence and comparable promotion decisions.

### Acceptance criteria

- Selection defaults are uncertainty-aware for trading objectives.
- Registry promotion compares like with like.
- Computed governance gates are actually enforced.

## Suggested Working Order

1. P0.1
2. P0.2
3. P0.3
4. P1.4
5. P1.5
6. P1.6
7. P1.7

## Done Definition For The Whole Program

- Diagnostic outputs are clearly separated from executable ones.
- No fold-level calibration path can see future information.
- Execution realism tier is explicit and binding.
- Market-integrity, cross-venue truth, staleness, and drift signals can block promotion rather than merely decorate reports.
- Registry comparisons are score-aligned and uncertainty-aware.
- Every remediation item lands with dedicated regression tests.