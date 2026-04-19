# GAPS - Current-State Priority Backlog

This document is the current verified gap backlog for the repository after reconciling:

- the current codebase,
- the passing test suite,
- repository memory notes, and
- external professional standards.

It intentionally does not repeat older backlog items that are already implemented. The goal is to capture the remaining gaps that still matter for research integrity, execution realism, and production readiness.

## Scope and Method

- The priority order below reflects the latest verified audit, not older historical gap lists.
- Each item includes the professional standard, the concrete repo evidence, a detailed implementation plan, acceptance criteria, and a test plan.
- The highest priority items are the ones most likely to distort results or create false confidence in the research outputs.

## Professional Standards Used

- Nested model selection and selection-bias control:
  https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
- Time-aware validation and explicit train-test gaps:
  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- Sequential bootstrapping for overlapping financial labels:
  https://hudsonthames.org/bagging-in-financial-machine-learning-sequential-bootstrapping-python/
- Lookahead-bias detection patterns for trading systems:
  https://www.freqtrade.io/en/stable/lookahead-analysis/
- Point-in-time joins and historical feature reconstruction:
  https://docs.feast.dev/getting-started/concepts/point-in-time-joins
- Exchange order filters and symbol constraints:
  https://developers.binance.com/docs/binance-spot-api-docs/filters
- Data and prediction drift monitoring:
  https://docs.evidentlyai.com/metrics/explainer_drift
- Streaming drift detection with ADWIN:
  https://riverml.xyz/dev/api/drift/ADWIN/
- Model registry lifecycle, versioning, immutability, and archive semantics:
  https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-models?view=azureml-api-2
- Serialization safety warning for pickle:
  https://docs.python.org/3/library/pickle.html

## Already Implemented and Not Repeated Here

- Deflated Sharpe Ratio and CPCV-style overfitting diagnostics are already integrated into AutoML selection reporting.
- A three-stage search, validation, and locked-holdout flow already exists.
- CPCV is already available and is the default validation path in current repo memory.
- Fold-local regime fitting, fold-local stationarity transforms, and fallback-signal-window hardening are already in place.
- Stationary-bootstrap significance reporting already exists in the backtest layer.
- Dynamic slippage models already exist.
- Kelly sizing decontamination is already in place.

The gaps below start from that hardened baseline.

## Priority Order

### P0 - Research Integrity and Execution Validity

1. AutoML search governance is still too permissive relative to the outer validation evidence.
2. The default AutoML objective is still classification-first instead of trading-first.
3. Sequential bootstrapping exists but is not wired into RandomForest training.
4. Execution semantics still diverge between the vectorbt and pandas backtest paths.
5. Binance execution-rule coverage is still partial.
6. Missing execution and valuation prices can still be backfilled from the future.

### P1 - Data Hygiene and Robustness

7. Missing-candle handling is warning-only and the downloader lacks retry, backoff, and strict completeness controls.
8. Point-in-time custom joins still rely too heavily on caller discipline.
9. The feature stack is still too dominated by endogenous OHLCV algebra and lacks stronger causal feature governance.
10. Trial-return alignment for overfitting diagnostics still zero-fills missing periods.
11. Fold stability is still descriptive only and is not enforced as a selection gate.

### P2 - Futures Realism and Operating Model

12. Futures research still lacks liquidation, margin, and leverage-tier realism.
13. Deployment governance is still incomplete: no implemented drift-monitoring loop, no local registry workflow, no champion-challenger promotion flow, and raw pickle persistence is still present.

---

## P0-1. AutoML Search Governance Beyond Outer Validation

**Why this is P0**

- The repo already separates search, validation, and locked holdout, which is the correct direction.
- The remaining gap is that the search space is still flexible enough that Optuna can win by finding a fragile configuration that barely survives the current ranking logic.
- Professional model-selection practice does not stop at nested validation. It also constrains search freedom, penalizes fragility, and rejects winners that are too sensitive to small perturbations.

**Professional standard**

- Scikit-learn's nested CV guidance is explicit that parameter tuning and performance estimation must be separated because the selection loop is itself a source of bias.
- In professional research stacks, the selected configuration must also survive complexity controls, generalization-gap checks, and simple parameter-perturbation tests before it is considered promotable.

**Repo evidence**

- `core/automl.py::compute_objective_value()` can still rank trials on a single scalar objective with no separate fragility penalty.
- `core/automl.py::_sample_trial_overrides()` can vary features, labels, model params, and signal params simultaneously.
- `core/automl.py::_build_trial_selection_report()` already uses validation metrics and DSR, but it does not yet reject overly fragile or overly complex winners.

**Repo touchpoints**

- `core/automl.py`
- `example_automl.py`
- `tests/test_automl_holdout_objective.py`

**Implementation plan**

1. Add an explicit `automl.selection_policy` block.
   - Required keys: `max_generalization_gap`, `max_param_fragility`, `max_complexity_score`, `require_locked_holdout_pass`, `min_locked_holdout_score`, `min_validation_trade_count`, `max_feature_count_ratio`, `max_trials_per_model_family`.
   - Keep defaults conservative rather than permissive.

2. Add a deterministic `trial_complexity_score`.
   - Compute it from selected feature count, lag count, label horizon, regime count, tree depth, estimator count, and any enabled meta-calibration layers.
   - Store it in each trial report.
   - Reject or down-rank trials that exceed the configured complexity budget.

3. Add an explicit `generalization_gap` report.
   - Record `search_raw_objective_value - validation_raw_objective_value`.
   - Record `validation_raw_objective_value - locked_holdout_raw_objective_value` when the locked holdout exists.
   - If either gap exceeds policy thresholds, mark the trial ineligible for promotion.

4. Add a winner-fragility pass after study completion.
   - Re-run the winning configuration with small local perturbations around the most important parameters.
   - At minimum perturb: threshold, feature cap, label horizon, and top model hyperparameters.
   - Compute the local dispersion of objective values and store `param_fragility_score`.

5. Change the final ranking policy.
   - Rank eligible trials only.
   - Eligibility should require: DSR gate pass, validation trade-count floor, complexity pass, generalization-gap pass, and locked-holdout pass when enabled.
   - Only after eligibility should the scalar objective decide the ordering.

6. Expose all of the above in the final AutoML summary.
   - The summary should show not just the winner but why it survived the gating stack.

**Acceptance criteria**

- A trial with the best validation score but an excessive search-to-validation gap is not selected.
- A trial with an excessive local fragility score is not selected.
- The study summary exposes complexity, gap, and fragility diagnostics for the top candidates.

**Test plan**

- Extend `tests/test_automl_holdout_objective.py` with a scenario where a high-search-score but fragile trial loses to a slightly weaker but stable trial.
- Add a test that a trial exceeding `max_complexity_score` is excluded from the eligible ranking set.
- Add a test that the final summary contains `generalization_gap`, `trial_complexity_score`, and `param_fragility_score`.

---

## P0-2. Make the Default Objective Trading-First Rather Than Classification-First

**Why this is P0**

- The product requirements already say to optimize trading-aware objectives, not raw accuracy alone.
- The current default remains `accuracy_first`, which biases the search toward classifiers that may look statistically tidy while failing after costs and execution rules.

**Professional standard**

- In deployed trading research, classification metrics are usually constraints or diagnostics.
- The primary selection objective is normally after-cost trading performance, risk-adjusted return, or a robust variant such as Sharpe, Calmar, or benchmark-relative performance.

**Repo evidence**

- `core/automl.py::_normalize_objective_name()` still defaults to `accuracy_first`.
- `core/automl.py::compute_objective_value()` gives the composite default heavy weight on directional accuracy and supporting classification metrics.
- `example_automl.py` still uses `objective: "accuracy_first"`.

**Repo touchpoints**

- `core/automl.py`
- `example_automl.py`
- `example_utils.py`
- `tests/test_automl_holdout_objective.py`

**Implementation plan**

1. Introduce a new default objective name such as `risk_adjusted_after_costs`.
   - Base it on validation-window backtest output, not classification output.
   - Make the default score a function of after-cost Sharpe or Calmar, with a drawdown penalty and a turnover penalty.

2. Relegate classification metrics to gating constraints.
   - Example gates: `min_directional_accuracy`, `max_log_loss`, `max_calibration_error`, `min_trade_count`.
   - If a candidate fails the gates, it should be ineligible even if its backtest score looks good.

3. Add benchmark-awareness to objective evaluation.
   - Use existing benchmark plumbing where available.
   - Support objective variants such as `benchmark_excess_sharpe` or `net_profit_pct_vs_benchmark`.

4. Add significance-awareness to objective evaluation.
   - If stationary-bootstrap significance exists, allow selection to use the lower confidence bound instead of the point estimate.
   - Example: use the Sharpe lower bound or a penalty based on wide uncertainty.

5. Preserve explicit opt-in to classification-first objectives.
   - Researchers should still be able to use `directional_accuracy` or `accuracy_first`, but it should be a conscious override, not the default.

6. Update examples and docs.
   - The examples should demonstrate a trading-first default and classification metrics as supporting diagnostics.

**Acceptance criteria**

- If no objective is specified, AutoML selects on an after-cost trading objective.
- Classification metrics remain visible but no longer silently define the default search target.
- The default example config uses the new trading-first selection path.

**Test plan**

- Add a test that the default objective resolves to the new trading-first mode.
- Add a test where the more accurate model loses because its after-cost validation backtest is worse.
- Add a test confirming that an explicit `accuracy_first` override still works.

---

## P0-3. Wire Sequential Bootstrapping Into RandomForest Training

**Why this is P0**

- This gap directly undermines any ensemble training on overlapping triple-barrier labels.
- The repo already has uniqueness weights and a sequential-bootstrap implementation, so the remaining problem is not theory, it is missing integration.

**Professional standard**

- Financial labels are not IID when event windows overlap.
- Hudson and Thames' sequential-bootstrapping guidance is explicit: bagged estimators should draw higher-uniqueness samples more often, because standard bootstrap inflates effective sample size and out-of-bag optimism.

**Repo evidence**

- `core/labeling.py` contains `sequential_bootstrap()`.
- `core/models.py::train_model()` still builds sklearn `RandomForestClassifier` and calls `fit()` directly.
- No custom bootstrap path or concurrency-aware subsampling is used when `model_type == "rf"`.

**Repo touchpoints**

- `core/models.py`
- `core/labeling.py`
- `core/pipeline.py`
- `tests/test_regime_leakage_controls.py`
- `tests/test_automl_holdout_objective.py`

**Implementation plan**

1. Extend the training API.
   - `train_model()` should accept label-event metadata or precomputed uniqueness diagnostics in addition to `sample_weight`.
   - Do not make RandomForest responsible for discovering concurrency on its own.

2. Add a concurrency decision rule.
   - Compute or pass `mean_uniqueness` for the training slice.
   - If `model_type == "rf"` and uniqueness is below a configured threshold, activate sequential bootstrapping.

3. Implement an explicit bootstrap-index path.
   - Generate bootstrap indices with `sequential_bootstrap()`.
   - Train RF on the resampled rows with `bootstrap=False` to avoid double bootstrapping.
   - Continue to pass sample weights on top of the resampled training slice.

4. Fix determinism.
   - Update `sequential_bootstrap()` to use `np.random.default_rng(random_state)` instead of global RNG state.
   - Make the seed explicit in config and in the training report.

5. Emit warnings when the researcher is using RF on high-concurrency labels without sequential bootstrap.
   - This should be loud in both runtime logs and the training summary.

6. Leave boosted models on uniqueness weights only.
   - For GBM, keep the current sample-weight approach unless a more principled bagging-like subsampling path is introduced later.

**Acceptance criteria**

- RandomForest training with concurrent labels no longer uses vanilla sklearn bootstrapping by default.
- Sequential-bootstrap runs are reproducible with a fixed seed.
- The training summary reports whether sequential bootstrapping was used and on what uniqueness basis.

**Test plan**

- Add a test that sequential bootstrap produces deterministic index sequences under a fixed seed.
- Add a test that RF training uses the sequential-bootstrap branch when uniqueness is below threshold.
- Add a test that RF emits a warning when sequential bootstrap is disabled on highly concurrent labels.

---

## P0-4. Unify Execution Semantics Across vectorbt and pandas Backtests

**Why this is P0**

- The same strategy should not be accepted or rejected depending on which engine executes it.
- Right now the vectorbt path enforces more of the execution contract than the pandas path does.

**Professional standard**

- Trading research systems should have a single execution specification and then multiple engine adapters.
- Freqtrade's lookahead-analysis guidance is instructive here: the platform forces market-order semantics in that mode precisely to avoid false positives caused by execution-path ambiguity.

**Repo evidence**

- `core/backtest.py::_run_vectorbt_backtest()` uses `tick_size`, `step_size`, `max_qty`, and `min_notional`.
- `core/backtest.py::_run_pandas_backtest()` ignores symbol filters and applies `position * returns` as a continuous target-weight process.
- The two engines therefore do not simulate the same executable strategy.

**Repo touchpoints**

- `core/backtest.py`
- `core/pipeline.py`
- `tests/test_data_backtest_adapter.py`

**Implementation plan**

1. Introduce a shared execution-contract layer.
   - Build a pure function that converts target signals into executable order intents after applying delay, leverage, side constraints, tick rounding, size rounding, and notional checks.
   - Both engines must consume this normalized representation.

2. Split portfolio construction from order validation.
   - Order validation should happen before the engine choice.
   - The engine should only differ in how it replays already-normalized orders.

3. Replace the pandas path's continuous-weight shortcut with a normalized event replay path.
   - The pandas engine should replay fills, flips, exits, and blocked orders from the shared order tape.
   - If that is not possible for a given config, fail closed rather than silently switching semantics.

4. Add a parity mode.
   - For simple scenarios, vectorbt and pandas should produce the same closed-trade count, nearly the same ending equity, and the same blocked-order decisions.

5. Expose blocked or adjusted orders in the backtest summary.
   - Researchers need to see how much of the original signal stream was not executable.

**Acceptance criteria**

- For the same inputs, both engines apply the same rounding, rejection, and delay rules.
- Parity tests pass for simple deterministic scenarios.
- The pandas engine no longer silently ignores rules that the vectorbt engine enforces.

**Test plan**

- Extend `tests/test_data_backtest_adapter.py` with engine-parity scenarios.
- Add a test where a small order is rejected in both engines for the same min-notional violation.
- Add a test where flip behavior and trade counts match between engines for a simple signal sequence.

---

## P0-5. Expand Binance Rule Coverage Beyond the Current Partial Filter Set

**Why this is P0**

- Even a backtest with correct delay and slippage is still misleading if it allows orders Binance would reject.
- The current filter support is only a subset of the actual exchange rule set.

**Professional standard**

- Binance defines symbol, exchange, and asset filters including price increments, lot sizes, min and max notional, market-lot constraints, side-sensitive percent-price limits, and position limits.
- Research systems should either model those rules or clearly fail if the requested simulation exceeds the modelled envelope.

**Repo evidence**

- `core/data.py::_parse_symbol_filters()` currently parses only a subset of filters.
- `core/backtest.py::_run_vectorbt_backtest()` consumes `tick_size`, `step_size`, `max_qty`, and `min_notional`.
- The pandas path consumes none of them.

**Repo touchpoints**

- `core/data.py`
- `core/backtest.py`
- `core/pipeline.py`
- `tests/test_data_backtest_adapter.py`

**Implementation plan**

1. Expand the filter parser.
   - Add support for `MARKET_LOT_SIZE`, `PERCENT_PRICE`, `PERCENT_PRICE_BY_SIDE`, `NOTIONAL`, `MAX_POSITION`, and the flags controlling market-order applicability.
   - Preserve unknown filters in raw form so the backtest layer can at least report unsupported cases.

2. Build a centralized order validator.
   - Inputs: side, order type, price, quantity, current position, weighted average price proxy, symbol filters.
   - Outputs: accepted, adjusted, or rejected plus the rejection reason.

3. Use the validator in both position sizing and backtesting.
   - Sizing should clamp to valid executable sizes.
   - Backtest execution should refuse fills that violate the same validator.

4. Add rejection accounting.
   - Report counts and notional share for blocked orders.
   - Report which filter caused the rejection most often.

5. Separate spot and futures assumptions.
   - Do not assume spot filters are enough for futures.
   - Keep the validator market-aware.

**Acceptance criteria**

- The repo supports all materially relevant Binance order filters for research-phase realism.
- Rejected and adjusted orders are visible in the output report.
- Sizing and execution use the same validation logic.

**Test plan**

- Add tests for `PERCENT_PRICE`, `MARKET_LOT_SIZE`, `NOTIONAL`, and `MAX_POSITION` behavior.
- Add a test that the same signal gets clipped or rejected before both execution engines.
- Add a test that rejection reasons are recorded in the backtest summary.

---

## P0-6. Remove Future-Backfill From Execution and Valuation Price Normalization

**Why this is P0**

- Any backfill from the future is a causal violation.
- It only takes a few missing leading bars to contaminate a backtest's execution price path.

**Professional standard**

- Lookahead-bias guidance in trading systems is clear: future values must never be used to construct current signals, features, or execution prices.
- A missing price should be dropped, forward-filled under explicit policy, or cause the relevant period to be excluded. It should not be backfilled from the future.

**Repo evidence**

- `core/backtest.py::_normalize_price_series()` currently does `.replace(0.0, np.nan).ffill().bfill()`.
- `core/pipeline.py::_resolve_backtest_valuation_close()` also uses `.ffill().bfill()` when mark-price valuation is requested.

**Repo touchpoints**

- `core/backtest.py`
- `core/pipeline.py`
- `tests/test_data_backtest_adapter.py`
- `tests/test_derivatives_context_pipeline.py`

**Implementation plan**

1. Replace implicit bfill with explicit causal fill policies.
   - Add config values such as `strict`, `ffill`, `drop_rows`, and `ffill_with_limit`.
   - Default to `strict` for execution prices and `drop_rows` or `ffill_with_limit` for valuation prices.

2. Treat leading missing values as invalid, not recoverable.
   - If the first executable bar has no valid execution price, drop it or stop the simulation with a clear error.

3. Distinguish execution-price policy from valuation-price policy.
   - Execution prices should be stricter than valuation prices.
   - Mark-price valuation may be forward-filled if the exchange semantics justify it, but not backfilled.

4. Add diagnostics.
   - Report how many rows were dropped, forward-filled, or invalidated by missing-price policy.

5. Apply the same policy to all execution-relevant price series.
   - Spot close, futures mark, and next-open execution prices should all follow the same causal rules.

**Acceptance criteria**

- No execution or valuation price path uses future data to fill missing values.
- Leading missing prices cause row drops or hard failures, not future backfill.
- The backtest summary exposes fill-policy actions.

**Test plan**

- Add a test where missing leading execution prices no longer get filled from future bars.
- Add a test where missing mark prices under `strict` mode abort valuation-path construction.
- Add a test confirming that forward-fill still works causally when explicitly enabled.

---

## P1-7. Add Strict Missing-Candle Handling and Downloader Retry/Backoff Policies

**Why this is P1**

- Missing candles corrupt rolling features, label horizons, and realized execution assumptions.
- The current downloader detects incomplete windows but only warns after a single refresh attempt.

**Professional standard**

- Production-grade market-data ingestion uses bounded retries, exponential backoff on transient failures, and explicit completeness policies.
- Exchange-aware systems also need clear handling for 404 archive gaps, 429 limits, and partial-day outages.

**Repo evidence**

- `core/data.py::_download_archive()` does a single `session.get()` call and raises on failure.
- `core/data.py::_load_period()` warns when a refreshed window still has missing candles, but it does not escalate or annotate downstream consumers.

**Repo touchpoints**

- `core/data.py`
- `core/pipeline.py`
- `tests/test_data_backtest_adapter.py`

**Implementation plan**

1. Add a reusable HTTP retry policy.
   - Retry on transient 5xx errors, timeouts, and 429-style rate-limit responses.
   - Use exponential backoff with jitter and a small max retry count.

2. Add completeness reporting to the fetch layer.
   - Record expected rows, observed rows, missing segments, retry count, and final integrity status.
   - Return or store the report instead of only printing warnings.

3. Add a `data.gap_policy` config.
   - Supported values: `fail`, `warn`, `flag`, `drop_windows`.
   - `fail` should stop the pipeline when unresolved gaps remain in requested windows.

4. Surface unresolved gaps into the pipeline state.
   - Feature builders and backtests should be able to inspect a structured integrity report.

5. Preserve exchange-friendly download behavior.
   - Keep the existing monthly-then-daily fallback strategy.
   - Add retry and completeness semantics on top of it rather than replacing it.

**Acceptance criteria**

- Transient download errors are retried with bounded backoff.
- Unresolved gaps are visible as structured integrity output.
- The pipeline can be configured to fail closed when requested research windows are incomplete.

**Test plan**

- Add a mocked HTTP test showing retry on timeout or 5xx before success.
- Add a test that unresolved gaps produce a structured integrity report.
- Add a test that `gap_policy: fail` stops the pipeline when missing windows remain.

---

## P1-8. Make Point-in-Time Custom Joins Fail Closed by Default

**Why this is P1**

- Custom datasets are one of the highest-leakage surfaces in the repo.
- The current join logic is good when the caller supplies true availability timestamps, but the defaults are still too permissive.

**Professional standard**

- Feast's point-in-time join model is explicit: historical retrieval should reconstruct the world as of each entity timestamp by scanning backward from the decision timestamp within a defined TTL window.
- In practice, feature stores require explicit availability semantics rather than assuming event time equals publish time.

**Repo evidence**

- `core/data.py::load_custom_dataset()` defaults `availability_column` to `timestamp_column`.
- `core/data.py::join_custom_dataset()` accepts `allow_exact_matches=True` and has no mandatory TTL or maximum age requirement.

**Repo touchpoints**

- `core/data.py`
- `example_custom_data.py`
- `tests/test_data_backtest_adapter.py`

**Implementation plan**

1. Make availability semantics explicit.
   - Require `availability_column` unless the caller explicitly opts into `assume_event_time_is_available_time: true`.
   - Emit a strong warning or reject the dataset when this explicit opt-in is missing.

2. Add TTL and freshness controls.
   - Support `max_feature_age` or `ttl` in the dataset config.
   - If no feature row is available within TTL, leave the joined columns null and record a stale-join event.

3. Add join auditing.
   - Report coverage, stale-hit count, median feature age, max feature age, and count of rows using the fallback assumption that event time equals availability time.

4. Tighten exact-match behavior.
   - Keep exact matches allowed only when the dataset uses an explicit availability column.
   - If the caller relies on the event-time fallback, default exact matches to disabled.

5. Document the contract in the example.
   - `example_custom_data.py` should demonstrate an explicit `available_at` field and a TTL.

**Acceptance criteria**

- The repo no longer silently assumes publish-time availability when the caller omits it.
- Historical joins enforce a TTL or maximum-age policy.
- Join audit reports expose stale and fallback usage rates.

**Test plan**

- Extend the existing point-in-time join test to assert TTL handling.
- Add a test that omitting `availability_column` without explicit opt-in raises or warns.
- Add a test that exact-match behavior changes based on explicit availability semantics.

---

## P1-9. Add Causal Feature Governance and Reduce Endogenous Feature Dominance

**Why this is P1**

- The feature layer is broad, but it is still dominated by transformations of the same OHLCV tape.
- That is acceptable for a baseline, but not as the default operating posture of a regime-aware crypto research system.

**Professional standard**

- Mature trading research systems track feature provenance, compare feature families by ablation, and explicitly distinguish endogenous price transforms from external or structurally causal inputs.
- Point-in-time safe exogenous data, derivatives context, and cross-asset context should be first-class sources rather than optional afterthoughts.

**Repo evidence**

- `core/features.py::build_feature_set()` still centers the default feature bundle on price-volume blocks and indicator-derived algebra.
- `core/context.py` already provides futures and cross-asset context, but these families are not yet governed as first-class selection groups.
- The pipeline summary tracks feature-block diagnostics, but it does not yet enforce family-level diversity or family-level ablation.

**Repo touchpoints**

- `core/features.py`
- `core/context.py`
- `core/pipeline.py`
- `example_custom_data.py`
- `tests/test_derivatives_context_pipeline.py`

**Implementation plan**

1. Promote feature-family metadata to a hard contract.
   - Every feature should belong to a family such as `endogenous_price`, `indicator`, `futures_context`, `cross_asset`, or `custom_exogenous`.
   - Preserve this metadata after feature selection.

2. Add family-level ablation reporting.
   - Evaluate baseline bundles such as `endogenous_only`, `endogenous_plus_futures`, `endogenous_plus_cross_asset`, and `full_context`.
   - Store performance deltas by family bundle.

3. Add family quotas or family penalties in AutoML.
   - Prevent the search from drifting into an unbounded soup of endogenous features without any external information.
   - At minimum, record when the selected model is effectively endogenous-only.

4. Make futures and cross-asset context easier to opt into.
   - Add clearer defaults and examples for funding, basis, mark-price, open-interest-derived, and cross-symbol context features.

5. Tighten custom-data onboarding.
   - Provide one strong example of a point-in-time safe custom feature family and include it in the research templates.

**Acceptance criteria**

- The pipeline reports which feature families contributed to the selected model.
- Researchers can compare endogenous-only and context-aware variants without rewriting the pipeline.
- The feature summary can tell whether performance depends entirely on self-referential OHLCV transforms.

**Test plan**

- Extend derivatives-context tests to validate family metadata survives alignment and selection.
- Add a test that family-ablation reporting is populated when multiple families are present.
- Add a test that the pipeline can explicitly identify an endogenous-only selected feature set.

---

## P1-10. Make Trial-Return Alignment for PBO and Related Diagnostics Strictly Observed-Only

**Why this is P1**

- Overfitting diagnostics should be conservative by construction.
- Synthetic zero returns inserted into missing periods make the path comparison more benign than the real evidence warrants.

**Professional standard**

- Path-level diagnostics should compare observed outcomes, not fabricated returns.
- When different trials cover different observed return windows, the analysis should either use the strict overlap or explicitly down-weight or reject insufficiently overlapping comparisons.

**Repo evidence**

- `core/automl.py::_align_trial_return_series()` reindexes trial returns to a common index and fills missing values with `0.0`.
- `core/automl.py::compute_cpcv_pbo()` again replaces missing values with `0.0` before scoring paths.

**Repo touchpoints**

- `core/automl.py`
- `tests/test_automl_holdout_objective.py`

**Implementation plan**

1. Replace union-plus-zero-fill alignment with explicit overlap policies.
   - Support `strict_intersection`, `pairwise_overlap`, and `min_overlap_fraction`.
   - Default to strict intersection for PBO-style comparisons.

2. Preserve missingness as missingness.
   - Do not coerce absent return periods into flat performance.
   - Carry a coverage mask alongside each aligned frame.

3. Reject weak comparisons.
   - If overlap between trials or paths is below a configured threshold, exclude the comparison and record why.

4. Report overlap diagnostics.
   - Add overlap counts, overlap fractions, and the number of excluded trial-pair comparisons to the overfitting summary.

5. Keep a deliberate fallback mode only for debugging.
   - If zero-fill is retained at all, it should be an explicit opt-in debug mode, never the default.

**Acceptance criteria**

- PBO and related path diagnostics no longer insert zero returns for missing periods by default.
- The summary exposes how much true overlap existed across the compared paths.
- Low-overlap comparisons are excluded rather than silently sanitized.

**Test plan**

- Add a test where staggered trial return series no longer produce zero-filled overlaps.
- Add a test that low-overlap comparisons are excluded and reported.
- Add a regression test proving the new strict mode is more conservative than the old zero-fill behavior.

---

## P1-11. Enforce Fold Stability as a Selection Gate, Not Just a Descriptive Report

**Why this is P1**

- Mean OOS performance alone is not enough.
- A strategy that alternates between excellent and broken across folds should not be promoted just because its average looks acceptable.

**Professional standard**

- Time-series validation should report both central tendency and dispersion.
- Professional research workflows usually reject unstable signals by thresholding fold dispersion, downside fold behavior, or minimum acceptable worst-fold performance.

**Repo evidence**

- `core/pipeline.py` stores `fold_metrics` and fold averages but does not compute standard deviation, coefficient of variation, or minimum-fold acceptance checks.
- AutoML selection currently has no fold-stability gate.

**Repo touchpoints**

- `core/pipeline.py`
- `core/automl.py`
- `example_utils.py`
- `tests/test_automl_holdout_objective.py`

**Implementation plan**

1. Compute fold-stability diagnostics in the training step.
   - For key metrics, compute mean, std, min, max, median, and coefficient of variation.
   - Include a per-fold validation backtest metric such as Sharpe or net profit.

2. Add an explicit `validation.stability_policy` config.
   - Supported gates: `max_cv`, `min_worst_fold_sharpe`, `min_worst_fold_net_profit_pct`, `max_drawdown_dispersion`.

3. Surface stability into AutoML ranking.
   - Trials failing stability gates become ineligible.
   - Trials passing all other gates but failing stability should still be visible in the report as rejected.

4. Show stability in summaries and examples.
   - The example reporting should print the range and dispersion, not only the average.

**Acceptance criteria**

- Fold dispersion statistics are present in the training summary.
- AutoML can reject unstable trials via policy.
- Example output highlights unstable winners rather than hiding them inside averages.

**Test plan**

- Add a test where a high-average but unstable trial is rejected by the stability policy.
- Add a test verifying coefficient-of-variation and worst-fold metrics are computed correctly.
- Add a test that the reporting layer displays stability diagnostics.

---

## P2-12. Add Futures Margin, Liquidation, and Leverage-Tier Realism

**Why this is P2**

- The repo already models funding and can use mark-price valuation.
- The remaining gap is the absence of liquidation mechanics, maintenance-margin thresholds, and leverage-tier constraints, which are central to futures realism.

**Professional standard**

- Futures research must separate spot and futures assumptions and must model mark-price driven equity changes, leverage tiers, maintenance margin, and liquidation events.
- A backtest that permits leverage without liquidation logic is structurally optimistic.

**Repo evidence**

- `core/backtest.py` supports funding and leverage but has no liquidation path.
- `core/context.py` already fetches funding and mark-price-related context.
- No maintenance-margin or leverage-bracket loader exists in the repo.

**Repo touchpoints**

- `core/backtest.py`
- `core/context.py`
- `core/data.py`
- `tests/test_derivatives_context_pipeline.py`
- `example_futures.py`

**Implementation plan**

1. Add a futures account model.
   - Distinguish isolated and cross-margin modes.
   - Track position notional, initial margin, maintenance margin, unrealized PnL on mark price, and margin ratio over time.

2. Add contract-spec and leverage-tier data adapters.
   - Load or cache the exchange metadata needed for contract size, leverage brackets, and maintenance-margin schedules.
   - Keep this separate from spot filters.

3. Add liquidation logic.
   - If margin ratio breaches maintenance requirements, close the position at the simulated liquidation price or at the next available executable mark-based proxy.
   - Apply liquidation fees or penalties explicitly.

4. Report futures-only metrics.
   - Number of liquidation events, max margin ratio, time spent above warning thresholds, funding paid and received, realized leverage usage.

5. Keep the execution adapter market-aware.
   - Spot and futures should not share the same execution-risk assumptions.

**Acceptance criteria**

- Levered futures backtests can produce liquidation events.
- Mark-price-based equity and maintenance-margin calculations are visible in the results.
- Spot mode remains unaffected by futures-specific machinery.

**Test plan**

- Add a deterministic liquidation test where a large adverse move forces position closure.
- Add a test that isolated and cross-margin assumptions produce different outcomes on the same price path.
- Extend futures examples or tests to verify that liquidation reporting fields are populated.

---

## P2-13. Implement Deployment Governance: Drift Monitoring, Local Registry, Promotion Flow, and Safe Artifacts

**Why this is P2**

- The repo is already beyond toy-research scaffolding, but it still lacks the operating model needed to retrain, compare, promote, and roll back models safely.
- This is the largest remaining gap between research code and a controlled production workflow.

**Professional standard**

- Evidently-style drift monitoring uses explicit reference versus current distributions and dataset-level drift rules.
- River's ADWIN gives a streaming detector with clear significance and window semantics.
- Model registries use immutable versions, mutable tags and descriptions, archive states, and explicit lineage from run to artifact.
- Python's own documentation warns that pickle is not secure and should only be unpickled from trusted sources.

**Repo evidence**

- `core/models.py::save_model()` and `load_model()` still use raw pickle.
- There is no registry module, no version manifest, no champion-challenger lifecycle, and no rollback workflow.
- No drift-monitoring module exists in the codebase.

**Repo touchpoints**

- `core/models.py`
- `core/automl.py`
- `core/pipeline.py`
- `requirements.txt`
- `tests/test_automl_holdout_objective.py`

**Implementation plan**

1. Replace raw artifact persistence with a local registry layout.
   - Add a registry folder structure such as `artifacts/registry/<symbol>/<model_name>/<version>/`.
   - Store: model artifact, config snapshot, feature schema, label spec, dataset window, metrics summary, lineage metadata, and a manifest file.

2. Make model versions immutable.
   - Only descriptions, tags, and status should be mutable after registration.
   - Promotion should create or update a status tag such as `candidate`, `champion`, `archived`, or `rejected`.

3. Add champion-challenger promotion logic.
   - Promotion should require locked-holdout pass, stability pass, and registry metadata completeness.
   - Challenger deployment should preserve the previous champion for rollback.

4. Add drift monitoring primitives.
   - Batch drift: feature drift and prediction drift using KS, PSI, Wasserstein, or JS-based tests.
   - Streaming drift: ADWIN on calibrated probabilities, realized edge, or trade-win process.
   - Retraining triggers should require minimum sample thresholds and cooldown windows.

5. Add safe artifact verification.
   - Replace raw pickle persistence with `joblib` as an immediate step, plus artifact hashing.
   - Keep a path open for `skops` or another safer format later.
   - Verify hash integrity before loading.

6. Add schema validation at load time.
   - The registry should store the exact ordered feature schema and training-time feature-family metadata.
   - Inference should fail closed on schema mismatch.

7. Integrate registry writes with AutoML outputs.
   - The best selected trial should be registrable as a versioned artifact with its evaluation lineage, not just a returned Python object.

**Acceptance criteria**

- Models can be registered with immutable versioned metadata and archived without deletion.
- The repo supports champion and challenger status assignments with rollback capability.
- Drift metrics and retraining triggers are stored in a structured report.
- Raw pickle-only persistence is removed from the main model-storage path.

**Test plan**

- Add a registry test that creates a model version and verifies manifest completeness.
- Add a rollback test where a challenger is archived and the previous champion remains loadable.
- Add drift tests covering both batch drift and ADWIN-triggered alerts.
- Add a hash-verification test proving tampered artifacts are rejected at load time.

---

## Recommended Execution Order

1. Complete P0-4, P0-5, and P0-6 together as a single execution-integrity workstream.
2. Complete P0-1 and P0-2 together as a single AutoML-selection workstream.
3. Complete P0-3 immediately after the selection workstream so ensemble results are not overstated.
4. Complete P1-7 and P1-8 before expanding the feature graph further.
5. Complete P1-10 and P1-11 before trusting any new AutoML ranking changes.
6. Complete P2-12 before treating futures results as deployment-grade.
7. Complete P2-13 before any scheduled or drift-triggered retraining workflow is treated as production-capable.

## Exit Condition for Calling the Research Stack "Professionally Defensible"

The repository should only be considered professionally defensible for research-to-deployment promotion when all of the following are true:

- AutoML defaults to a trading-first objective and rejects fragile winners.
- RandomForest training is concurrency-aware.
- Backtest engines share one execution contract.
- Binance constraints are centrally validated.
- No future-backfill remains in price normalization.
- Data completeness and point-in-time joins fail closed by default.
- Overfitting diagnostics use observed-only overlaps.
- Fold stability gates are enforced.
- Futures leverage can liquidate.
- Models are versioned, drift-monitored, and safely loadable.
