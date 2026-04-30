# Remediation Plan

## Purpose

This document turns the findings in `ISSUES.md` into an implementation plan.

The goal is not a broad rewrite. The goal is to close the specific fail-open paths that can inflate research results or create false certification language while preserving the current architecture where possible.

This plan is written against the code that exists in this workspace now. It does **not** assume the presence of a separate `core/research/` package.

## Delivery Principles

- Fix controls at the point where the pipeline decides admissibility, not only in reporting.
- Prefer fail-closed behavior for certification and promotion paths.
- For research-only paths, distinguish `unknown`, `advisory`, and `passed`; do not silently coerce `unknown` into `passed`.
- Keep migration explicit when tightening defaults would break existing demos.
- Add regression tests before or alongside each behavior change.
- Update example entrypoints so the safest behavior is the easiest path to run.

## Workstream Order

Implement the work in this order:

1. Close the certification bypasses.
2. Tighten point-in-time feature availability.
3. Make labels reject incomplete future paths.
4. Make data-quality quarantine behavior explicit and harder to misuse.
5. Make missing-evidence governance fail closed.
6. Remove research-mode funding leniency as a silent default.
7. Replace synthetic example-universe assumptions with explicit demo-only semantics.
8. Remove the close-fill footgun from the raw backtest API.

This order matters because items 1, 2, 3, and 5 directly affect whether any existing result should be trusted.

## Workstream 1: Re-enable And Enforce Lookahead Certification

### Problem

The operator-facing trade-ready example disables `features.lookahead_guard.enabled`, but the AutoML promotion stack still treats a missing or disabled lookahead report as a pass.

### Target Outcome

- Trade-ready and local-certification entrypoints must always run a pre-training feature-surface lookahead replay.
- Promotion eligibility must treat `disabled`, `missing`, and `not_run` as failures for capital-facing paths.
- Research-only paths may allow advisory behavior, but the summary must show `not_run` distinctly from `passed`.

### Files To Modify

- `example_trade_ready_automl.py`
- `example_local_certification_automl.py`
- `example_utils.py`
- `core/pipeline.py`
- `core/automl.py`
- `core/promotion.py` if gate-report normalization must distinguish `unknown` from `passed`

### Code Changes

#### 1. Remove the explicit disable in the trade-ready example

In `example_trade_ready_automl.py`:

- Delete the override that sets `features.lookahead_guard.enabled = False`.
- Keep `mode = "blocking"`.
- Keep the small decision sample size for smoke runs only if the audit remains enabled.

#### 2. Normalize runtime defaults in shared example builders

In `example_utils.py`:

- Extend `build_trade_ready_runtime_overrides(...)` to inject a blocking lookahead guard by default.
- Extend local-certification runtime helpers the same way if they do not already do so.
- If smoke profiles use reduced-power settings, only reduce sample size or prefix length. Never disable the guard.

#### 3. Make the pipeline emit explicit status

In `core/pipeline.py`:

- Change `_run_pipeline_lookahead_guard(...)` so the report contains a stable status field such as:
  - `status = "passed"`
  - `status = "failed"`
  - `status = "disabled"`
  - `status = "skipped"`
- When no eligible step exists, use `skipped`, not an implicit pass.
- When the guard is disabled in a capital-facing path, either:
  - raise immediately, or
  - emit `promotion_pass = False` with reason `lookahead_guard_disabled`.

#### 4. Make AutoML treat missing lookahead evidence as failure in capital-facing modes

In `core/automl.py`:

- Replace `lookahead_guard.get("promotion_pass", True)` with stricter logic.
- Add a helper such as `_resolve_lookahead_gate_status(training_summary, selection_policy, evaluation_mode)` that returns:
  - `passed`
  - `failed`
  - `missing`
  - `disabled`
- In capital-facing selection and post-selection promotion gates, `missing` and `disabled` must fail.
- In research-only mode, `missing` may remain advisory, but the reason must be attached explicitly.

### Tests To Add Or Modify

- Update `tests/test_lookahead_provocation.py`
  - Add a test that the trade-ready configuration built by `example_trade_ready_automl.py` has the guard enabled.
  - Add a test that smoke mode keeps the guard enabled.
- Update `tests/test_global_lookahead_guard_default.py`
  - Add assertions for capital-facing default behavior.
- Update `tests/test_pipeline_lookahead_guard_wiring.py`
  - Add a case where `lookahead_guard = {}` or `enabled = False` and assert that promotion fails under default trade-ready selection policy.
- Add a new test file if needed: `tests/test_trade_ready_lookahead_guard_required.py`
  - Build the trade-ready config.
  - Assert the final config cannot mark lookahead as passed when the guard is disabled or absent.

### Example Validation

- `example_trade_ready_automl.py --smoke`
  - Summary must show lookahead enabled and checked.
- `example_local_certification_automl.py`
  - Summary must show lookahead enabled and checked.

### Acceptance Criteria

- No capital-facing example can produce a promotion-eligible report with `lookahead_guard.enabled = False`.
- The summary contract distinguishes `not_run` from `passed`.

## Workstream 2: Make Custom Data Joins Strictly Point-In-Time By Default

### Problem

`CustomDataset.default_allow_exact_matches = True` for explicit availability columns. Because market bars are indexed by Binance kline open time, exact-timestamp joins are unsafe for many external datasets that are only known at or after bar close or event publication.

### Target Outcome

- Exact-timestamp matches are opt-in, not default.
- The custom-data contract forces the user to declare whether exact availability is causal.
- Join reports explicitly state whether exact matching was used and why.

### Files To Modify

- `core/data.py`
- `core/data_contracts.py` if the contract should validate exact-match semantics
- `example_custom_data.py`
- `example_test_case_template.py`
- `HOW_TO_USE.md`
- `README.md`

### Code Changes

#### 1. Tighten dataset defaults

In `core/data.py`:

- Change `CustomDataset.default_allow_exact_matches` default from `True` to `False`.
- In `load_custom_dataset(...)`, change the default for explicit availability columns to `False` unless the caller explicitly declares an exact-match-safe contract.

Suggested new contract fields:

- `allow_exact_matches`
- `availability_semantics`
  - `"strictly_before_decision"`
  - `"exact_timestamp_is_tradeable"`
  - `"assumed_from_event_time"`

#### 2. Fail when semantics are ambiguous

In `load_custom_dataset(...)` and/or contract validation:

- If `availability_column` is explicit but `allow_exact_matches` is omitted, default to `False` and record a warning or manifest flag.
- If `assume_event_time_is_available_time = True`, continue to force `allow_exact_matches = False`.

#### 3. Improve reports

In `join_custom_dataset(...)`:

- Add `availability_semantics` and `exact_match_policy_source` to the join report.
- Record how many rows matched exactly at the decision timestamp.

### Tests To Add Or Modify

- Extend `tests/test_data_backtest_adapter.py`
  - Existing coverage already checks assumed vs explicit reports.
  - Add behavior tests showing that explicit availability timestamps do **not** join on exact match unless the caller opts in.
  - Add a case where `allow_exact_matches=True` is explicitly set and verify the same-timestamp join occurs.
- Add a dedicated test file if the existing file becomes too mixed, e.g. `tests/test_custom_data_point_in_time.py`
  - `test_explicit_availability_defaults_to_no_exact_match`
  - `test_exact_match_requires_explicit_opt_in`
  - `test_join_report_counts_exact_matches`

### Example Validation

- Update `example_custom_data.py` to show both:
  - the safe default path with `allow_exact_matches=False`
  - an explicitly opted-in path for genuinely decision-time-safe feeds
- Update the template in `example_test_case_template.py` so new users copy the safe default.

### Acceptance Criteria

- No custom feed with an explicit availability column joins on equal timestamps unless the config opts in explicitly.
- The join report makes equal-timestamp joins auditable.

## Workstream 3: Reject Incomplete Future Paths In Triple-Barrier Labeling

### Problem

`triple_barrier_labels()` does not verify that the entire forward `high`, `low`, and `close` window is present. Missing values silently convert PT/SL checks into false negatives.

### Target Outcome

- Label generation must either drop incomplete forward windows or mark them explicitly invalid.
- Research metrics must not count labels built from broken future paths.

### Files To Modify

- `core/labeling.py`
- `core/pipeline.py`
- `core/data.py`
- `README.md` or `HOW_TO_USE.md` for labeling assumptions

### Code Changes

#### 1. Add forward-window completeness policy

In `core/labeling.py`:

- Add a parameter to `triple_barrier_labels(...)`, for example:
  - `missing_future_policy="drop"`
- Before barrier evaluation, check whether the forward `close`, `high`, and `low` slice is fully non-null.
- If incomplete:
  - under `drop`, skip the label row entirely
  - under `flag`, emit a row with `label = NaN` or `barrier = "invalid_future_window"`
- Do not allow silent continuation.

#### 2. Thread policy through the pipeline

In `core/pipeline.py`:

- Extend `LabelsStep` to pass a label completeness policy from config.
- Add summary metrics for:
  - `dropped_for_incomplete_future_window`
  - `invalid_future_window_count`

#### 3. Tighten research defaults

In the shared config builders in `example_utils.py` if needed:

- Set label completeness policy to `drop` by default.

### Tests To Add Or Modify

- Add `tests/test_labeling_future_window_integrity.py`
  - `test_triple_barrier_drops_rows_with_nan_in_forward_high_low_close`
  - `test_time_barrier_not_emitted_when_forward_window_is_incomplete`
  - `test_complete_forward_window_still_produces_expected_label`
- Extend `tests/test_data_quality_quarantine.py`
  - Add a pipeline-level case where a missing bar or nullified anomalous row prevents label creation downstream.

### Example Validation

- `example.py`
  - print or expose dropped-label counts in the label summary.
- `example_custom_data.py`
  - verify aligned sample count shrinks when future windows are incomplete.

### Acceptance Criteria

- Triple-barrier labels never silently use incomplete forward windows.
- Label summaries report how many candidate entries were discarded due to future-path incompleteness.

## Workstream 4: Make Data-Quality Quarantine Harder To Ignore

### Problem

Quarantine behavior is advisory by default for many anomalies. That allows flagged glitch rows to remain tradable in research pipelines.

### Target Outcome

- The pipeline should distinguish between `drop`, `null`, `flag-advisory`, and `flag-blocking` clearly.
- Research configs should not silently learn from quarantined rows unless the caller explicitly chooses that behavior.

### Files To Modify

- `core/data_quality.py`
- `core/pipeline.py`
- `example_utils.py`
- `README.md` and `HOW_TO_USE.md`

### Code Changes

#### 1. Add quarantine disposition to the cleaned frame and report

In `core/data_quality.py`:

- Expand the report to include per-row or aggregated counts by disposition:
  - `dropped`
  - `nulled`
  - `flagged_only`
- Add a `quarantine_severity` classification so downstream code can choose stricter handling.

#### 2. Add a pipeline option to exclude flagged-only rows from modeling

In `core/pipeline.py`:

- Introduce a config such as `exclude_flagged_quarantine_rows_from_modeling`.
- When enabled, remove `flagged_only` rows from `raw_data`, `data`, features, and labels, even if they were not physically dropped in the data-quality step.
- Set this to `True` for local-certification and trade-ready modes.
- Consider setting it to `True` by default for research examples too, unless the example explicitly demonstrates anomaly tolerance.

#### 3. Tighten example defaults

In `example_utils.py`:

- Add explicit data-quality defaults for research examples rather than relying on library defaults.
- Recommended baseline:
  - `ohlc_inconsistency = drop`
  - `duplicate_timestamp = drop`
  - `retrograde_timestamp = drop`
  - `nonpositive_volume = drop`
  - `return_spike = null` or `drop`
  - `range_spike = null` or `drop`
  - `quote_volume_inconsistency = null`
  - `trade_count_anomaly = null`

### Tests To Add Or Modify

- Extend `tests/test_data_quality_quarantine.py`
  - add a pipeline-level test for `exclude_flagged_quarantine_rows_from_modeling=True`
  - add a test that `flag` no longer means “still used without distinction” when that option is enabled
  - add a report-contract test for disposition counts

### Example Validation

- Update example summaries to print:
  - quarantined rows
  - dropped rows
  - nulled rows
  - modeling-excluded rows

### Acceptance Criteria

- A user cannot mistake `flagged` rows for clean rows in either summaries or downstream state.
- Capital-facing modes never train on flagged-only quarantine rows.

## Workstream 5: Make Governance Gates Fail Closed On Missing Evidence

### Problem

Several governance summaries return `promotion_pass=True` when there is no evidence rather than positive evidence. Selection logic then uses `get(..., True)` and promotes candidates under uncertainty.

### Target Outcome

- Missing evidence is represented as `unknown`, not `passed`.
- Capital-facing selection policy treats `unknown` as blocking failure.
- Research mode may keep `unknown` advisory, but never silently green.

### Files To Modify

- `core/feature_governance.py`
- `core/regime.py`
- `core/signal_decay.py`
- `core/automl.py`
- `core/automl_contracts.py` if serialized gate contracts need tri-state support
- `core/promotion.py` if promotion report helpers assume boolean-only gates

### Code Changes

#### 1. Introduce tri-state gate semantics

Standardize gate payloads to include:

- `status = "passed" | "failed" | "unknown"`
- `promotion_pass = True/False` only after mode-specific resolution
- `reasons` for both `failed` and `unknown`

#### 2. Change summary helpers

In `core/feature_governance.py`:

- `evaluate_feature_portability(...)`
  - if `top_features` is empty or total importance is zero, return `status = "unknown"`, not pass
- `summarize_feature_admission_reports(...)`
  - if `reports` is empty, return `status = "unknown"`

In `core/regime.py`:

- `summarize_regime_ablation_reports(...)`
  - if no required rows exist, return `status = "unknown"`

In `core/signal_decay.py`:

- preserve the low-sample advisory, but mark insufficient evidence as `unknown` rather than default positive evidence

#### 3. Resolve gates centrally in AutoML selection

In `core/automl.py`:

- Add a single helper to resolve gate status based on evaluation mode and selection policy.
- Replace all `get(..., True)` patterns for promotion-relevant gates.
- Ensure top-trial reports expose `passed`, `failed`, and `unknown` distinctly.

### Tests To Add Or Modify

- Extend `tests/test_selection_policy_defaults.py`
  - assert that trade-ready defaults resolve key evidence gates as blocking
- Add `tests/test_missing_evidence_gate_resolution.py`
  - verify missing portability evidence blocks certification
  - verify missing regime ablation evidence blocks certification
  - verify low-sample signal-decay evidence is `unknown`, not `passed`
- Extend any existing AutoML selection tests that assert promotion-ready behavior

### Example Validation

- `print_automl_summary(...)` in `example_utils.py`
  - print `unknown` gates separately from failed gates
- `example_trade_ready_automl.py`
  - make the interpretation text warn that `unknown` evidence is disqualifying for promotion

### Acceptance Criteria

- A capital-facing run cannot become promotion-ready with missing portability, regime, lookahead, or decay evidence.

## Workstream 6: Replace Silent Research Funding Zero-Fill With Explicit Unknown-Carry Modes

### Problem

Research futures backtests silently zero-fill missing funding events. That treats unknown carry as free carry.

### Target Outcome

- Research runs should default to either strict funding coverage or explicit exclusion of incomplete windows.
- If a fallback mode remains, it must be opt-in and visibly labeled as non-comparable to strict results.

### Files To Modify

- `core/pipeline.py`
- `core/backtest.py`
- `example_utils.py`
- `example_futures.py`
- `example_active_futures.py`
- `example_trade_ready_automl.py` and `example_local_certification_automl.py` only if messaging must change

### Code Changes

#### 1. Change research default policy

In `core/pipeline.py::_resolve_backtest_funding_missing_policy(...)`:

- Change the default research mode from `zero_fill` to either:
  - `strict`, or
  - `preserve_missing`

Recommended path:

- `research_only` default becomes `preserve_missing`
- `preserve_missing` means incomplete funding windows are not forced to zero; instead the run is flagged underpowered or incomplete

#### 2. Adjust runtime funding normalization

In `core/backtest.py::_normalize_runtime_funding_rates(...)`:

- Support a mode where missing funding propagates as missing evidence rather than zero PnL.
- If a backtest cannot compute funding cleanly for a futures path, surface:
  - `funding_coverage_status = "incomplete"`
  - `promotion_pass = False`
  - optional research-only advisory that the PnL is not certification-grade

#### 3. Keep zero-fill only as an explicit debug mode

- If zero-fill remains supported, rename it to something unmistakably unsafe such as `zero_fill_debug` or `research_demo_zero_fill`.
- Remove it from default example builders.

### Tests To Add Or Modify

- Update `tests/test_funding_zero_fill_forbidden_in_capital_modes.py`
  - change research-default expectation if the default mode changes
- Extend `tests/test_funding_coverage_gate.py`
  - add cases for `preserve_missing`
  - add cases for incomplete research summaries
- Add `tests/test_futures_research_funding_policy.py`
  - verify default research futures configs do not silently zero-fill

### Example Validation

- `example_futures.py`
  - summary must show whether funding was strict, incomplete, or debug-fallback
- `example_active_futures.py`
  - same requirement

### Acceptance Criteria

- No default futures example silently converts missing funding events into zero carry.

## Workstream 7: Replace Synthetic Example Universe Assumptions With Explicit Snapshot Modes

### Problem

`build_example_universe_config(...)` fabricates all symbols as tradable since 2020 with made-up liquidity. That is acceptable for offline demos only, but not as a default research assumption.

### Target Outcome

- Example configs must declare when they are using synthetic universe assumptions.
- Certification-capable paths must refuse synthetic snapshots.
- Research summaries must label synthetic universe assumptions as non-survivorship-safe.

### Files To Modify

- `example_utils.py`
- `core/universe.py`
- `core/pipeline.py` if universe metadata is surfaced there
- `tests/test_demo_universe_not_allowed_in_certification.py`
- `tests/test_historical_universe_selection.py`
- `README.md`

### Code Changes

#### 1. Mark synthetic snapshots explicitly

In `example_utils.py::build_example_universe_config(...)`:

- Add metadata like:
  - `source = "synthetic_example_snapshot"`
  - `survivorship_safe = False`
  - `liquidity_source = "fabricated"`

#### 2. Tighten certification gating

In `core/universe.py` or the portability contract path:

- Treat `synthetic_example_snapshot` the same way current tests treat live-default or non-frozen snapshots for certification.
- Add explicit rejection reason such as `synthetic_universe_snapshot_not_allowed`.

#### 3. Improve summaries

- Surface universe provenance in example summaries and AutoML promotion diagnostics.

### Tests To Add Or Modify

- Extend `tests/test_demo_universe_not_allowed_in_certification.py`
  - assert synthetic example snapshots fail portability/certification contracts
- Extend `tests/test_historical_universe_selection.py`
  - ensure real/frozen snapshots still pass where appropriate

### Example Validation

- All example scripts that use `build_example_universe_config(...)`
  - print a warning that the snapshot is synthetic unless the user provides a real snapshot

### Acceptance Criteria

- No certification or trade-ready path can rely on the synthetic example universe snapshot.

## Workstream 8: Remove Same-Bar Close Execution As An Easy Backtest Footgun

### Problem

The low-level `run_backtest(...)` API falls back to executing on `close` if `execution_prices` is omitted. The example builders usually avoid this, but the primitive is still easy to misuse.

### Target Outcome

- Raw backtest callers must choose execution semantics explicitly.
- If they do not, the API should either warn loudly or fail in strict modes.

### Files To Modify

- `core/backtest.py`
- `core/pipeline.py`
- `tests/test_data_backtest_adapter.py`
- `README.md` or `HOW_TO_USE.md`

### Code Changes

#### 1. Add explicit execution default policy

In `core/backtest.py::run_backtest(...)`:

- Introduce an argument like `execution_price_source` or `require_explicit_execution_prices`.
- For direct callers:
  - certification/local-certification/trade-ready: require explicit execution prices
  - research-only: either require explicit execution prices or emit a `same_bar_execution_fallback` warning in the result

#### 2. Ensure pipeline helpers stay explicit

In `core/pipeline.py`:

- Keep `_resolve_backtest_execution_prices(...)` as the canonical execution-price resolver.
- Ensure all internal pipeline backtest calls always pass execution prices.

### Tests To Add Or Modify

- Extend `tests/test_data_backtest_adapter.py`
  - add a test that capital-facing modes reject missing `execution_prices`
  - add a test that research-only direct calls emit a visible warning if fallback is allowed

### Example Validation

- No shipped example should rely on implicit close execution.

### Acceptance Criteria

- Same-bar execution can no longer happen accidentally in any capital-facing or example-driven path.

## Cross-Cutting Test Plan

Run focused tests after each workstream instead of relying only on the full suite.

### Focused Test Batches

- Lookahead and gate semantics
  - `python -m pytest tests/test_lookahead_provocation.py tests/test_global_lookahead_guard_default.py tests/test_pipeline_lookahead_guard_wiring.py tests/test_selection_policy_defaults.py`

- Data quality, PIT joins, and labeling integrity
  - `python -m pytest tests/test_data_quality_quarantine.py tests/test_data_backtest_adapter.py tests/test_labeling_future_window_integrity.py`

- Funding and futures research behavior
  - `python -m pytest tests/test_funding_coverage_gate.py tests/test_funding_coverage_report_contract.py tests/test_funding_zero_fill_forbidden_in_capital_modes.py tests/test_local_certification_funding_strict.py`

- Universe and certification contracts
  - `python -m pytest tests/test_demo_universe_not_allowed_in_certification.py tests/test_historical_universe_selection.py`

### End-To-End Example Checks

- `python example_automl.py`
  - verify research-only summary now reports unknown evidence explicitly
- `python example_local_certification_automl.py`
  - verify blocking guardrails remain enabled
- `python example_trade_ready_automl.py --smoke`
  - verify lookahead is enabled and the run cannot present a false pass
- `python example_custom_data.py`
  - verify safe custom-data join behavior is shown
- `python example_futures.py`
  - verify funding coverage is strict or explicitly incomplete, never silently zero-filled by default

## Rollout Strategy

### Phase 1: Certification Integrity

Deliver together:

- Workstream 1
- Workstream 5

Reason:

- These determine whether a capital-facing run can falsely present itself as admissible.

### Phase 2: Causal Data Surface

Deliver together:

- Workstream 2
- Workstream 3
- Workstream 4

Reason:

- These determine whether the model is learning on causally valid inputs and valid labels.

### Phase 3: Futures And Universe Realism

Deliver together:

- Workstream 6
- Workstream 7
- Workstream 8

Reason:

- These are the remaining realism and misuse-hardening tasks once the causal and certification core is fixed.

## Definition Of Done

The issues in `ISSUES.md` should be considered materially addressed only when all of the following are true:

- Trade-ready and local-certification entrypoints cannot bypass lookahead certification.
- Missing evidence in promotion-relevant gates is surfaced as `unknown` and blocks capital-facing promotion.
- Custom data joins are strict PIT by default and exact matches require explicit opt-in.
- Triple-barrier labels reject incomplete forward paths.
- Flagged anomaly rows are either excluded from modeling or visibly retained by explicit user choice.
- Default futures research examples no longer silently zero-fill unknown funding carry.
- Synthetic example universe snapshots are labeled as demo-only and rejected by certification paths.
- Raw backtest callers cannot accidentally rely on implicit same-bar close fills in capital-facing modes.

## Suggested First PR

The first PR should be intentionally small and high-leverage:

1. Re-enable the lookahead guard in `example_trade_ready_automl.py`.
2. Change `core/automl.py` so missing or disabled lookahead evidence fails trade-ready promotion.
3. Add the focused tests for that behavior.

That PR closes the most serious false-certification hole with the smallest blast radius.