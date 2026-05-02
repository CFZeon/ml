# Remediation Plans

Each section below corresponds to a numbered finding from ISSUES.md Audit #2 (2026-05-02)
and, where still open, Audit #1 (2026-04-30). Plans are ordered by the section grouping
used in that audit. Each plan states the root cause, the exact code change required, and
the acceptance test that confirms the fix.

---

## Section 1 — Data Integrity and Preprocessing

---

### 1.1 ADF stationarity screening on the full dataset (Critical)

**Root cause.**
`core/features.py::screen_features_for_stationarity()` and `check_stationarity()` are
called on the entire feature `DataFrame` before any train/test split. The ADF test
decision — and the choice of which transform from `DEFAULT_STATIONARITY_TRANSFORM_ORDER`
to apply — is therefore influenced by the full time series including OOS observations.
This makes the selected fractional differentiation order `d` a function of the holdout set.

**Fix.**
1. Add a new function `fit_stationarity_transforms(series, significance, transform_order)
   -> dict` to `core/features.py`. It runs the ADF search on a **training window only**
   and returns a `TransformSpec` dataclass: `{column, transform_name, params}`.
2. Add a complementary `apply_stationarity_transforms(frame, specs) -> DataFrame` that
   applies pre-fitted specs to any window (train or test) without re-running ADF.
3. Modify `screen_features_for_stationarity()` to accept an optional `fit_window`
   parameter. When provided, ADF search uses only `series.iloc[:fit_window]`. The
   returned screening report stores the fitted specs so downstream callers can apply them
   to OOS data reproducibly.
4. In `core/pipeline.py::FeatureScreeningStep`, pass `fit_window = len(train_idx)` when
   screening runs inside a fold context, and `fit_window = None` (existing behaviour) for
   the exploratory full-series pass that precedes split creation.
5. In `core/automl.py` trial execution, call `fit_stationarity_transforms` on the
   training slice only, persist the specs in the trial artifact, and call
   `apply_stationarity_transforms` on both train and test slices.

**Acceptance test.**
Write a test that creates a synthetic non-stationary series, splits it 70/30, calls
`fit_stationarity_transforms` on the 70% window, applies the returned spec to the 30%
window, and asserts that: (a) the fitted `d` was not computed from the full series, and
(b) the same spec is applied consistently across folds in a walk-forward setting.
Add to `tests/test_stationarity_fold_isolation.py`.

---

### 1.2 `DEFAULT_STATIONARITY_TRANSFORM_ORDER` sequential fallback is full-sample model selection (Critical)

**Root cause.**
The ordered fallback `log_diff → pct_change → diff → zscore → frac_diff` searches for
the first transform that produces ADF p < 0.05 on the entire series. This is a discrete
hyperparameter search where the target variable is the OOS stationarity outcome, not the
training-window stationarity. The winning transform may not preserve memory properties
or be reproducible on a different data window.

**Fix.**
1. Remove the implicit sequential search from `screen_features_for_stationarity`. Instead,
   require the caller to specify either a single transform or an explicit ordered list via
   config key `features.stationarity_transforms`.
2. Introduce a new config field `features.stationarity_search_policy`:
   - `"train_window_only"` (new default): ADF search runs only on the training window and
     the result is recorded as a fold artifact.
   - `"fixed"`: the transform is taken from `features.stationarity_transforms[column]`
     without any ADF search; useful when the researcher has already determined `d`
     externally.
3. Emit a `WARNING` log and set `leakage_risk: True` in the screening report whenever
   `stationarity_search_policy` is absent or the fit window is `None`, so the risk is
   visible in all output summaries rather than silent.
4. Update the default pipeline config in `example_utils.py` to set
   `stationarity_search_policy: "train_window_only"`.

**Acceptance test.**
Parametrised test over two policies: `"fixed"` and `"train_window_only"`. Verify that the
`leakage_risk` flag is absent for both, that the selected transform is recorded in the
trial artifact, and that re-running with different data bounds but the same fixed spec
produces the identical column values for the overlapping range.

---

### 1.3 Volatility series for triple-barrier labels computed on full series (High)

**Root cause.**
`triple_barrier_labels()` accepts a pre-computed `volatility` series. In
`core/pipeline.py` the volatility is typically an ATR or rolling std computed over the
entire `data` frame before any fold split. A bar inside the training fold therefore uses
volatility that was calculated with knowledge of future bars in the test set.

**Fix.**
1. In `core/pipeline.py::LabelingStep`, compute volatility **inside** the label
   computation loop for each fold rather than once globally:
   ```python
   # Before: compute globally
   volatility = data["close"].pct_change().rolling(vol_window).std()

   # After: compute only up to the current fold's training boundary
   def _fold_volatility(close, train_end_idx, window):
       vol = close.iloc[:train_end_idx].pct_change().rolling(window, min_periods=window//2).std()
       # extend forward-fill into test window (no future info used, only last training value)
       return vol.reindex(close.index).ffill()
   ```
2. Expose `volatility_fit_boundary` as a parameter on `triple_barrier_labels()`. When
   set, the function computes an internal rolling std using only the observations up to
   that boundary index and forward-fills the rest. When `None`, behaviour is unchanged for
   backward compatibility but emits a `UserWarning`.
3. In `core/automl.py` trial execution, pass `volatility_fit_boundary = train_end_idx`
   at each CV fold so the label function is explicitly fold-aware.

**Acceptance test.**
Construct a price series with a known volatility regime change at the fold boundary.
Assert that training-set labels do not change when the test-set is replaced with random
data, proving no future observations affect training labels.

---

### 1.4 `check_data_quality()` defaults four anomaly types to `"flag"` (High)

**Root cause.**
`core/data_quality.py::check_data_quality()` uses `default_actions` where
`return_spike`, `range_spike`, `quote_volume_inconsistency`, and `trade_count_anomaly`
are all `"flag"`. Flagged rows remain in the DataFrame and enter feature/label
computation, creating a training set that includes Binance-specific artefacts
(exchange glitches, wash-trade bursts) the model cannot generalise from.

**Fix.**
1. Add a new `modeling_exclusion_policy` config key with valid values:
   - `"none"`: keep current behaviour (flags only, rows included).
   - `"exclude_from_labels"` (new default): flagged rows are removed from the label
     computation input. Features for these bars can still be computed (they may be
     preceding bars for a future observation), but these bars cannot be the `t0` of a
     label.
   - `"exclude_from_features"`: flagged rows are also excluded from the feature matrix
     passed to model training (but not from the lookback window used to compute lagged
     features for unflagged bars).
2. Modify `DataQualityResult.modeling_exclusion_mask` (already defined in the dataclass)
   to be populated according to `modeling_exclusion_policy`.
3. In `core/pipeline.py::LabelingStep`, apply `modeling_exclusion_mask` to drop flagged
   bars from the label input before calling `triple_barrier_labels()`.
4. Add a `block_on_quarantine_minimum_severity` config key: rows where
   `ohlc_inconsistency`, `duplicate_timestamp`, or `retrograde_timestamp` are already
   dropped; for the four softer anomalies, introduce a configurable `spike_threshold`
   multiplier (`default: 8.0`) that can be tightened.
5. Update the research default to `modeling_exclusion_policy: "exclude_from_labels"` and
   certification default to `"exclude_from_features"`.

**Acceptance test.**
Inject a synthetic return spike row into a clean DataFrame. Run `check_data_quality` with
`modeling_exclusion_policy="exclude_from_labels"`. Assert the flagged bar does not appear
as a `t0` index value in the resulting label set.

---

### 1.5 Binance kline intra-bar path unknown for barrier tie-breaks (High)

**Root cause.**
Binance Vision kline CSVs only provide open, high, low, close per bar. When both
`high >= upper` and `low <= lower` in the same bar, `triple_barrier_labels()` resolves the
tie using `barrier_tie_break` (default `"sl"`). The assumption is not measured; it is an
arbitrary choice. Including `barrier_tie_break` as a searchable hyperparameter allows
Optuna to pick whichever assumption inflates the objective.

**Fix.**
1. Remove `barrier_tie_break` from `DEFAULT_AUTOML_SEARCH_SPACE` in `core/automl.py` and
   from `_THESIS_SPACE_PATHS`.
2. Add `barrier_tie_break` as a fixed, documented config field under `labels`:
   ```yaml
   labels:
     barrier_tie_break: "conservative"   # new option; see below
   ```
3. Introduce a new option `"conservative"`: when both barriers are hit in the same bar,
   record the outcome as `label=0` (abstain) with `barrier="tie"`. This is the only
   honest treatment given no path information. The bar is still a valid observation for
   the time barrier, but the direction is unknown.
4. Add a `tie_count` field to the `integrity_report` attached to the label DataFrame's
   `.attrs`, so the researcher can see what fraction of samples are affected.
5. Keep `"sl"` and `"pt"` as valid values for backward compatibility but emit a
   `UserWarning` describing the assumption risk.

**Acceptance test.**
Construct a price series with a synthetic bar where `high > upper` and `low < lower`.
Assert that under `"conservative"`, the resulting label is `0` and `barrier == "tie"`.
Assert that under `"sl"`, the label is `-1`. Assert the `tie_count` field is non-zero in
both cases.

---

### 1.6 Missing-candle gap policy defaults to `"warn"` in research mode (Medium)

**Root cause.**
`core/data.py` accepts `gap_policy` values from `{"fail", "warn", "flag", "drop_windows"}`.
The research pipeline uses `"warn"`, which leaves index gaps intact. Rolling feature
computations (ATR, MACD, Bollinger) treat discontinuous bars as adjacent, biasing
volatility estimates downward and masking drawdown risk.

**Fix.**
1. Change the research default from `"warn"` to `"flag"`.
2. Add a post-fetch step in `core/pipeline.py::FetchStep` that checks for gaps in the
   DatetimeIndex using `_interval_timedelta(interval)` from `data.py` and:
   - Adds a boolean column `"gap_preceding"` to the raw data frame (True when the
     interval from the previous bar exceeds 1.5× the nominal interval).
   - Stores a `gap_report` in the pipeline state with gap count, total missing bars, and
     the timestamp of each gap start.
3. Any rolling computation that crosses a gap boundary must reset its window. Implement
   `_gap_aware_rolling(series, window, gap_mask, min_periods)` in `core/features.py` that
   sets rolling values to NaN at the first bar inside each gap (and for `window` bars
   thereafter) rather than treating the gap as an ordinary step.
4. In the Sharpe annualisation, replace the median-interval approach with the nominal
   interval derived from the configured `interval` string: parse the interval to
   `pd.Timedelta` and divide `_SECONDS_PER_YEAR` by the nominal seconds. This eliminates
   the gap-inflation bias in the annualisation factor.

**Acceptance test.**
Construct a 1h series with a 6-bar gap at the midpoint. Assert that (a) ATR values in
the 6 bars following the gap are NaN (not carried from before the gap), and (b) the
Sharpe annualisation uses exactly 8,766 periods per year (365.25 × 24) regardless of gap
count.

---

### 1.7 `CustomDataset.default_allow_exact_matches` timestamp leakage (Medium)

**Root cause.**
`core/data.py::join_custom_data()` merges external data onto market bars. When
`allow_exact_matches=True` (the default for many callers), a custom feature timestamped
`T` is available at bar `T`, which is only valid if the feature was published before bar
`T` opened. For coarse daily or delayed feeds this is almost never true.

**Fix.**
1. Change `CustomDataset.default_allow_exact_matches = False` as the class-level default.
   This is a one-line change to the dataclass definition.
2. Require all callers that need exact-match semantics to explicitly pass
   `allow_exact_matches=True` with a required string argument `availability_rationale`
   that is stored in the dataset manifest (providing an audit trail).
3. Add an `availability_lag` field to `CustomDataset` (type `pd.Timedelta | None`). When
   set, the join uses `asof` merge with `tolerance=availability_lag` and shifts the
   external index backward by `availability_lag` before joining, approximating the actual
   publication delay.
4. In `core/data_contracts.py::validate_custom_source_contract()`, add a validation that
   raises if `availability_is_assumed=True` and `allow_exact_matches=True` simultaneously,
   unless `availability_rationale` is explicitly provided.

**Acceptance test.**
Build a custom dataset with daily values timestamped at midnight. Join it onto an hourly
bar series with `allow_exact_matches=False`. Assert that bars on day `D` use the value
from day `D-1`, not day `D`.

---

## Section 2 — Feature Engineering

---

### 2.1 Regime labels from full data contaminate fold feature engineering (Critical)

**Root cause.**
`core/regime.py::build_instrument_regime_state()` is called on the full `base_data`
frame. Any downstream feature that conditions on regime membership (e.g. regime-specific
z-scores, regime interaction features) carries implicit future information into every
training fold.

**Fix.**
1. Add a `boundary_idx: int | None` parameter to `build_instrument_regime_state()`. When
   set, CUSUM, rolling statistics, and any unsupervised clustering (KMeans, HMM) are
   **fitted** using only `base_data.iloc[:boundary_idx]` and then **applied** (without
   re-fitting) to the full range.
2. In `core/pipeline.py`, the regime step must be called inside each fold loop with
   `boundary_idx = train_end_idx`. Store the fitted regime model (cluster centres or HMM
   params) in the fold artifact so the same fitted model is reused for test prediction.
3. For the non-fold exploratory regime pass (used to build the regime feature block before
   splits are defined), use a rolling/online version: at each bar `t`, use only
   `base_data.iloc[:t]` to produce the regime label for bar `t`. This is equivalent to an
   expanding-window regime fit and requires no future data.
   - Implement `build_instrument_regime_state_online(base_data, ...)` in `core/regime.py`
     using an expanding window via pandas `expanding().apply()` or a vectorised equivalent
     for the rolling statistics.
4. Update `core/automl.py` trial execution to call the online variant during the
   pre-split regime construction phase.

**Acceptance test.**
Create a synthetic price series with a known regime change at bar 500. Fit regimes online
(expanding) and assert that regime labels at bars 1–499 do not change when bars 500–1000
are removed from the input, proving no future data is used.

---

### 2.2 Fractional differentiation burn-in too long for short windows (High)

**Root cause.**
`core/features.py::fractional_diff()` computes the weight vector until `|w| < 1e-5`.
For `d=0.2`, this produces a weight vector of roughly 500–1000 terms, meaning the first
valid output bar is at index 500+. On a 500-bar fold that leaves near-zero valid rows.

**Fix.**
1. Add a `max_lag: int | None` parameter to `fractional_diff()`. When set, truncate the
   weight vector at `min(natural_width, max_lag)` regardless of the threshold criterion.
   This mirrors the "fixed-width window" approach from López de Prado chapter 5.
2. Add a `min_valid_rows: int` guard: after computing the output, if the non-NaN row
   count is below `min_valid_rows`, raise `ValueError` with a message suggesting the
   caller increase data length or reduce `max_lag`.
3. In `core/pipeline.py::FeatureScreeningStep`, add a `fracdiff_max_lag` config key
   (default `200`) that is passed to `fractional_diff()`. Expose it via the features
   config block.
4. Report `burn_in_rows` and `valid_rows` in the stationarity screening report so
   researchers can see the actual output yield before models are trained.
5. During the ADF search in `fit_stationarity_transforms`, skip `frac_diff` if
   `valid_rows < min_valid_rows` and fall back to `diff` instead of proceeding with a
   near-empty series.

**Acceptance test.**
Call `fractional_diff(series, d=0.2, max_lag=50)`. Assert the first valid output is at
index 50, not at the natural convergence point (~600). Assert that `valid_rows` in the
returned report equals `len(series) - 50`.

---

### 2.3 Cross-asset context features use same-venue data, correlated missingness (High)

**Root cause.**
`core/context.py::build_cross_asset_context_feature_block()` fetches ETHUSDT bars from
Binance to use as context for BTCUSDT. Both symbols are on the same exchange, so outages
create simultaneous NaNs. The model learns that absent context = normal conditions.

**Fix.**
1. Add a `cross_asset_source_policy` config key under `data.cross_asset_context`:
   - `"same_venue"`: current behaviour, with an explicit warning in the output report.
   - `"multi_venue"` (recommended): each context symbol must have at least one
     non-Binance price series (e.g., from a public websocket or alternative data source)
     joined alongside the Binance series.
2. Add a `correlated_missingness_check` to `core/data_quality.py`: given two binary
   missingness masks, compute the phi coefficient (Matthews correlation) and report it. If
   phi > 0.5 between the primary symbol and any context symbol, emit a `WARNING` and set
   `correlated_missingness: True` in the data quality report.
3. Add an `outage_imputation_policy` config key:
   - `"drop_bar"`: bars where the context is missing are excluded from training labels.
   - `"last_valid"`: forward-fill context values up to `max_context_lag` bars; beyond
     that treat as missing and apply drop_bar.
   - `"explicit_missing_feature"`: add a binary indicator column `ctx_{symbol}_missing`
     so the model can explicitly learn the outage pattern without carrying spurious values.
4. Default to `"explicit_missing_feature"` in the feature config.

**Acceptance test.**
Construct a primary series and a context series where 10 bars are simultaneously NaN.
Assert that (a) phi coefficient > 0.5 triggers the warning, and (b) with
`outage_imputation_policy="explicit_missing_feature"`, the output DataFrame contains a
`ctx_ETH_missing` column with value `1.0` at those bars.

---

### 2.4 Rolling features produce startup artifacts at fold boundaries (Medium)

**Root cause.**
`_rolling_zscore()`, ATR, MACD, and Bollinger Band computations use pandas `.rolling()`
with `min_periods` that allows output before the window is full. The first `window - 1`
values are qualitatively different (single-observation statistics) and create
distribution shift between folds that start at different positions.

**Fix.**
1. Add a utility function `_rolling_with_warmup_nans(series, window, func)` in
   `core/features.py` that always sets the first `window - 1` output values to NaN
   regardless of `min_periods`, enforcing a proper burn-in.
2. Replace all internal rolling calls that use `min_periods < window` with this function,
   or add explicit NaN setting:
   ```python
   result = series.rolling(window).mean()
   result.iloc[:window - 1] = np.nan
   ```
3. In the feature screening step, add a `startup_nan_share` diagnostic per column: the
   fraction of NaN values attributable to the rolling burn-in period. Report this in the
   feature admission report so researchers see the true usable length per feature.
4. Enforce that the training fold length is at least `max_window_size * 2` to ensure
   at least half the training rows are fully initialised rolling values. Raise a
   `ValueError` if this condition is not met, rather than silently training on a fold
   dominated by startup NaNs.

**Acceptance test.**
Compute `_rolling_zscore(series, window=20)`. Assert that `result.iloc[:19].isna().all()`
is True. Assert that `result.iloc[20:].isna().sum() == 0` for a clean series.

---

### 2.5 Feature lags longer than embargo carry OOS information into training (Medium)

**Root cause.**
The search space includes lag configs such as `"1,6,12"`. When the CV embargo is 12 bars
and a lag of 12 is selected, the last 12 training bars have their lag features computed
from what would, under a shorter embargo, be OOS territory. The embargo removes rows from
the **start** of the test set but does not check whether training features contain
look-forwards via lags.

**Fix.**
1. In `core/models.py::cpcv_split()` and `walk_forward_split()`, add a
   `max_lag: int = 0` parameter. When non-zero, extend the embargo by `max_lag` additional
   rows so the effective embargo is `gap + max_lag`. This ensures that no lagged feature
   in the training set reaches into what would be within `gap` bars of the test set.
2. In `core/pipeline.py` and `core/automl.py`, compute `max_lag` from the lag config
   before calling the split function:
   ```python
   lags = [int(x) for x in features_config.get("lags", "1").split(",")]
   max_lag = max(lags)
   train_idx, test_idx = cpcv_split(X, ..., embargo=gap, max_lag=max_lag)
   ```
3. Add a `lag_embargo_extension` field to the split metadata dict so the value is
   auditable in the run summary.
4. Emit a `WARNING` in the feature governance report if any lag exceeds the configured
   embargo: `"lag {L} exceeds embargo {G}; effective embargo extended to {G+L}"`.

**Acceptance test.**
Split a DataFrame with `gap=12`, lags `[1, 12]`, and `max_lag=12`. Assert that the last
row of every training split has index at least `12 + 12 = 24` bars before the first row
of the corresponding test split. Assert that `lag_embargo_extension == 12` in the split
metadata.

---

### 2.6 z-score NaN values propagate silently into model training (Low)

**Root cause.**
`_rolling_zscore()` in `core/features.py` returns NaN for the first `window` bars.
If the model's training fold starts at bar 0, these NaN values enter `X_train`. Scikit-
learn tree models skip NaN-containing rows silently (or raise, depending on the model),
creating inconsistent sample counts between folds.

**Fix.**
1. Apply the `_rolling_with_warmup_nans` fix from Issue 2.4, which makes the NaN range
   explicit and consistent.
2. In `core/models.py::train_model()`, add a pre-fit assertion:
   ```python
   nan_count = int(pd.DataFrame(X).isna().any(axis=1).sum())
   if nan_count > 0:
       raise ValueError(f"train_model received {nan_count} rows with NaN features. "
                        "Check rolling window burn-in and fold boundary alignment.")
   ```
3. In `core/pipeline.py::AlignmentStep`, after joining features and labels, assert
   `X.isna().sum().sum() == 0` before proceeding to training.

**Acceptance test.**
Attempt to call `train_model` with a DataFrame containing NaN values. Assert a
`ValueError` is raised. Then assert that, after applying the warmup-NaN fix and dropping
the leading NaN rows during fold alignment, training succeeds with zero NaN assertions.

---

## Section 3 — AutoML Process

---

### 3.1 Label hyperparameters in the search space contaminate the holdout (Critical)

**Root cause.**
`DEFAULT_AUTOML_SEARCH_SPACE` includes `pt_mult`, `sl_mult`, `max_holding`,
`volatility_window`, and `barrier_tie_break` under the `labels` section. Each Optuna
trial with different label params generates a **different label set** and evaluates it
against a shared OOS partition. The partition is therefore used to select label
definitions, not just model hyperparameters. The "locked holdout" is no longer unseen
after the first trial.

**Fix.**
1. Create a new concept: the **thesis contract**. A thesis contract is a frozen,
   pre-committed label definition that must be set by the researcher before the AutoML
   study begins, not searched over.
2. Move all `labels.*` keys from `DEFAULT_AUTOML_SEARCH_SPACE` into a separate config
   block: `automl.thesis_contract.labels`. Document that these values must be fixed before
   a study begins.
3. In `core/automl.py::_validate_trade_ready_search_space()`, extend the existing
   `_find_varying_thesis_paths()` check to also run in **research** and
   **local_certification** modes, not only in `trade_ready` mode. In research mode emit a
   `WARNING`; in certification and trade-ready modes raise `ValueError`.
4. The `_THESIS_SPACE_PATHS` set already correctly identifies these paths. The only change
   needed is calling `_find_varying_thesis_paths` in the trial runner regardless of
   evaluation mode and surfacing the result in the trial report.
5. Add a `thesis_locked` boolean to the experiment manifest. Set it to `True` when no
   thesis-space paths vary. The promotion gate should require `thesis_locked == True`.

**Acceptance test.**
Configure an AutoML study with `labels.pt_mult` as a varying search space entry. Assert
that: (a) in research mode, the manifest records `thesis_locked: False` and a warning is
logged; (b) in certification mode, a `ValueError` is raised before the study starts; (c)
when `pt_mult` is removed from the search space, `thesis_locked: True` and the study runs.

---

### 3.2 `validation_fraction` in the search space inflates apparent OOS score (Critical)

**Root cause.**
`model.validation_fraction` is in `DEFAULT_AUTOML_SEARCH_SPACE` with choices
`[0.15, 0.2, 0.25, 0.3]`. Varying the fraction over which the validation score is
computed is equivalent to sliding a window over the data to find the most favourable
evaluation sub-period. Optuna's TPE sampler will converge on whichever fraction makes
the selected objective easiest to maximise for the given seed and data.

**Fix.**
1. Remove `model.validation_fraction` from `DEFAULT_AUTOML_SEARCH_SPACE` entirely.
2. Make `validation_fraction` a fixed config field under `model`: one value only,
   established before the study starts.
3. Move it to the thesis contract (see 3.1 fix). Like label parameters, the validation
   set size is a thesis-level commitment: the researcher declares upfront what fraction of
   data is held out for model selection.
4. Add a config key `model.validation_fraction_policy`:
   - `"fixed"`: required, a single float.
   - `"canonical"`: use the default `0.2` automatically. Emit an info log.
5. Remove `model.validation_fraction` from `_THESIS_SPACE_PATHS` and add it to a new
   `_FORBIDDEN_SEARCH_PATHS` set that causes an immediate error if found in the search
   space in any evaluation mode.

**Acceptance test.**
Attempt to add `model.validation_fraction` to the search space. Assert that
`run_automl_study` raises a `ValueError` indicating a forbidden search path, regardless
of evaluation mode.

---

### 3.3 Post-selection inference does not fully correct for multiple comparisons (High)

**Root cause.**
`core/stat_tests.py::select_post_selection_candidates()` deduplicates correlated trials
then runs a stationary bootstrap p-value test on each remaining candidate. The candidates
were selected from the same dataset, so their return series are generated under the same
data-generating process. The Benjamini–Hochberg procedure applied per candidate does not
account for the fact that the p-values were maximised over a large search space.

**Fix.**
1. Implement the **Deflated Sharpe Ratio (DSR)** test from Bailey & López de Prado (2014)
   in `core/stat_tests.py::compute_deflated_sharpe_ratio()`:
   ```
   DSR = SR* × sqrt(T) / (1 - SR* × skew + SR*² × (kurtosis - 1) / 6)^0.5
   ```
   where `SR*` is the Sharpe of the selected strategy and the deflation accounts for the
   number of independent trials tested.
2. Add `n_independent_trials` estimation: use the correlation matrix of the trial return
   frame to compute the effective number of independent strategies via the eigenvalue
   method (number of eigenvalues > mean eigenvalue).
3. Report `deflated_sharpe_ratio`, `effective_trials`, and `multiple_testing_corrected_p`
   in the post-selection inference output alongside the existing bootstrap p-value.
4. Add a promotion gate `min_deflated_sharpe_ratio` (default `0.0` in research,
   configurable in certification). If DSR ≤ 0, the trial does not pass post-selection,
   regardless of the raw Sharpe or bootstrap p-value.

**Acceptance test.**
Generate 50 independent random-walk return series (no signal). Run
`compute_deflated_sharpe_ratio` on the best one. Assert that the resulting DSR is close to
zero and that `multiple_testing_corrected_p > 0.05` for the null hypothesis.

---

### 3.4 Seed-dependent reproducibility misinterpreted as independent replication (High)

**Root cause.**
`automl_config.get("seed", 42)` produces a deterministic Optuna search trajectory. A
re-run with the same seed on the same data window is not an independent replication but
an exact replay.

**Fix.**
1. Add a `replication_seed` field to the experiment manifest. When `automl.seed` is
   explicitly set to a fixed value, set `replication_seed: "fixed_{seed}"` and emit a
   `WARNING`: *"Fixed seed produces a deterministic search trajectory. Re-running with the
   same seed and data is a replay, not an independent replication."*
2. Add a `replication` config block (already referenced in `automl.py`) with a field
   `n_independent_seeds: int` (default `1`). When `n_independent_seeds > 1`, run the
   study once per seed from a generated list of independent seeds, store each run's best
   trial result, and report the cross-seed variance of the objective value. High variance
   indicates the result is seed-sensitive (noise-fitting).
3. Add a `seed_sensitivity_cv` metric to the AutoML summary: the coefficient of variation
   of the best trial objective across seeds. Gate promotion on `seed_sensitivity_cv < 0.3`
   in certification mode.

**Acceptance test.**
Run an AutoML study twice with `seed=42` and `seed=123`. Assert the two best trial numbers
differ. Assert `seed_sensitivity_cv` is computed and stored in the summary. Assert that
running with `seed=42` twice produces identical trial sequences (confirming determinism).

---

### 3.5 `meta_n_splits` in search space biases meta-model data volume (High)

**Root cause.**
`DEFAULT_AUTOML_SEARCH_SPACE` includes `model.meta_n_splits` with choices `[2, 3]`.
Fewer splits give the meta-model more training data per fold, increasing its ability to
overfit. Optuna can select a split count that produces better validation metrics not
because the model is better, but because it has more data.

**Fix.**
1. Remove `model.meta_n_splits` from `DEFAULT_AUTOML_SEARCH_SPACE`.
2. Fix `meta_n_splits` in the thesis contract (alongside label and validation params).
   Default to `3` in all profiles.
3. Add an explicit note in the config documentation: meta-model split count controls data
   volume available for meta-model training. Changing it during search is equivalent to
   data snooping on meta-label quality.

**Acceptance test.**
Assert that `DEFAULT_AUTOML_SEARCH_SPACE["model"]` does not contain a key
`"meta_n_splits"`. Assert a study configured with `meta_n_splits` in the search space
raises a `ValueError` from the thesis-contract validation.

---

### 3.6 Calibration hyperparameters co-optimised with model selection (Medium)

**Root cause.**
`model.calibration_params.c` is in the search space and is optimised jointly with model
selection. Calibration quality cannot be reliably measured on the same data used to
select the model because the calibration score is already maximised as part of the Optuna
objective.

**Fix.**
1. Split model training into two phases in `core/models.py::train_model()`:
   - **Phase 1**: select model hyperparameters on the CV folds.
   - **Phase 2**: fit calibration on a dedicated calibration split that was not used in
     Phase 1. Implement via a 3-way split: `train / calibration / validation` rather than
     `train / validation`.
2. Remove `calibration_params.c` from the search space. Expose it as a post-selection
   config field applied after the best model is selected.
3. Add a config key `model.calibration_split_fraction` (default `0.15`). The final
   `1 - validation_fraction - calibration_split_fraction` of non-holdout data is used for
   training, the `calibration_split_fraction` portion immediately after for calibration,
   and the `validation_fraction` at the end for selection scoring.

**Acceptance test.**
Assert that the calibration data rows are temporally after all training rows and before
the first validation row. Assert that changing `calibration_params.c` does not change the
trial selection score (because calibration is applied post-selection, not during search).

---

### 3.7 `n_trials` too small for TPE warm-up (Medium)

**Root cause.**
Optuna's TPE sampler uses the first 25 trials (by default) as random sampling before
switching to tree-structure exploitation. Studies with fewer than 25 trials are entirely
random, providing no benefit from the TPE optimisation.

**Fix.**
1. Add a guard in `core/automl.py::run_automl_study()`:
   ```python
   n_trials = automl_config.get("n_trials", 0)
   tpe_startup = automl_config.get("tpe_startup_trials", 25)
   if n_trials < tpe_startup:
       warnings.warn(
           f"n_trials={n_trials} < tpe_startup_trials={tpe_startup}. "
           "The entire study will be random sampling. "
           "Set n_trials >= 25 for TPE to provide meaningful search guidance."
       )
   ```
2. Add `tpe_startup_trials` to the Optuna `TPESampler` constructor:
   ```python
   sampler = TPESampler(seed=seed, n_startup_trials=tpe_startup)
   ```
3. In the research profile, enforce a minimum of `n_trials=30`. In certification enforce
   `n_trials >= 50`. In trade-ready enforce `n_trials >= 100`. Raise if below minimum.
4. Report `tpe_startup_trials` and `tpe_exploitation_trials` (= `n_trials - startup`) in
   the study summary.

**Acceptance test.**
Configure an AutoML study with `n_trials=10` and `tpe_startup_trials=25`. Assert a
`UserWarning` is raised containing "random sampling". Assert the study still runs to
completion (warning only, not a hard error in research mode). Assert a `ValueError` in
certification mode.

---

## Section 4 — Backtesting Methodology

---

### 4.1 Kelly sizing uses in-sample win probability (Critical)

**Root cause.**
`core/backtest.py::kelly_fraction(prob_win, avg_win, avg_loss)` is called with statistics
derived from the **same backtest** whose performance is being measured. The IS win rate
is always higher than the OOS rate for any marginally-above-chance classifier. Using IS
Kelly inflates the equity curve.

**Fix.**
1. Add a `kelly_source` config key to the backtest section:
   - `"model_probability"` (new default): use the model's calibrated probability output
     `p(long)` or `p(short)` directly as the Kelly numerator, rather than the IS win rate.
   - `"oos_estimate"`: use an externally-supplied OOS win rate (from a prior walk-forward
     fold); intended for paper trading, not research.
   - `"is_observed"`: current behaviour; emits a `UserWarning` and sets a
     `kelly_bias_risk: True` flag in the backtest summary.
2. Implement `kelly_fraction_from_probability(p, b, fraction)` in `core/backtest.py`:
   ```python
   def kelly_fraction_from_probability(p, b=1.0, fraction=0.5):
       """Kelly fraction from a single calibrated probability estimate.
       p: probability of the favourable outcome (from model.predict_proba)
       b: net odds (avg_win / avg_loss; default 1.0 for symmetric payoff)
       """
       q = 1.0 - p
       k = (p * b - q) / b
       return max(0.0, min(k, 1.0)) * fraction
   ```
3. In `run_backtest()`, when `kelly_source == "model_probability"`, pass the per-bar
   calibrated probability series and use `kelly_fraction_from_probability` at each bar
   rather than computing a single IS fraction for the whole window.
4. Document that even calibrated probability-based Kelly sizing requires a shrinkage
   factor of at least 0.5 (half-Kelly) to account for estimation error. Cap the maximum
   fraction at 0.25 in research mode.

**Acceptance test.**
Build a synthetic test where IS win rate = 0.60 but OOS calibrated probability = 0.51.
Assert that `kelly_fraction_from_probability(0.51, b=1.0, fraction=0.5) < 0.01`, while
`kelly_fraction(0.60, avg_win=1.0, avg_loss=1.0, fraction=0.5) ≈ 0.10`. Assert the
backtest equity curve is lower under `"model_probability"` Kelly than under `"is_observed"`
Kelly on the same trade set.

---

### 4.2 Same-bar execution fallback reachable in research mode (Critical)

**Root cause.**
`core/backtest.py::_resolve_execution_price_input()` returns `close` with a warning dict
when `execution_prices` is `None` and the mode is not capital-facing. The warning is
silent (stored in a dict field) and does not block the backtest. Callers that omit
`execution_prices` produce same-bar fill results that overstate performance.

**Fix.**
1. Change the fallback behaviour in non-capital-facing modes from silent warning to an
   explicit **opt-in**. Add a config key `backtest.allow_same_bar_fill_fallback: bool`
   (default `False`).
2. When `execution_prices is None` and `allow_same_bar_fill_fallback` is `False`, raise
   `ValueError("execution_prices must be provided. To use same-bar close fallback "
   "explicitly, set backtest.allow_same_bar_fill_fallback=True.")`.
3. When `allow_same_bar_fill_fallback` is `True`, the current warning dict is produced AND
   an additional printed warning is emitted to stderr: `"WARNING: same-bar close fill is
   not replicable in live trading."`.
4. Update all example configs in `example_utils.py` to provide an explicit
   `execution_prices` based on `open.shift(-signal_delay_bars)` rather than relying on
   the fallback.
5. Add `same_bar_fill_used: bool` to the backtest summary's top-level fields so it
   appears in every report.

**Acceptance test.**
Call `run_backtest` without `execution_prices` and without `allow_same_bar_fill_fallback`.
Assert `ValueError` is raised. Call again with `allow_same_bar_fill_fallback=True`. Assert
`same_bar_fill_used == True` in the returned summary.

---

### 4.3 Signal delay as static bar offset ignores wall-clock latency (High)

**Root cause.**
`signal_delay_bars=2` is a static integer offset. On 1h bars this assumes a 2-hour
execution window, which is far too generous for live crypto trading (typical REST API
round-trip is under 500ms). On 5m bars, 2 bars = 10 minutes — catastrophically large.
There is no per-interval validation that the delay is calibrated to realistic latency.

**Fix.**
1. Add a `signal_delay_calibration` config block:
   ```yaml
   signal_delay_calibration:
     min_delay_bars: 1
     max_realistic_delay_ms: 2000   # 2 seconds
     warn_if_delay_exceeds_ms: 5000
   ```
2. In `core/pipeline.py`, compute the wall-clock equivalent of `signal_delay_bars` as
   `delay_bars * interval_timedelta_seconds * 1000` ms. If this exceeds
   `warn_if_delay_exceeds_ms`, emit a warning. If it exceeds a hard cap
   (`max_realistic_delay_ms * 10`), raise in certification mode.
3. Add a `signal_delay_wall_clock_ms` field to the backtest summary so latency
   assumptions are visible in every report.
4. Document in the config schema that `signal_delay_bars=1` is the minimum credible value
   for any live system using bar-open execution, and that `signal_delay_bars=2` is
   appropriate only for intervals of 1h or longer.

**Acceptance test.**
Configure `signal_delay_bars=2` with `interval="5m"`. Assert the computed
`signal_delay_wall_clock_ms = 600_000` (10 minutes) triggers a warning in research mode
and a `ValueError` in certification mode where `max_realistic_delay_ms=2000`.

---

### 4.4 `_infer_periods_per_year()` uses median inter-bar interval (High)

**Root cause.**
`core/backtest.py::_infer_periods_per_year()` computes the median of `index.diff()`.
When an index contains gaps, the median still equals the nominal interval, but the actual
number of bars per year is lower due to missing candles. The Sharpe ratio is therefore
annualised using the nominal period count rather than the realised bar count.

**Fix.**
1. Replace `_infer_periods_per_year()` with `_nominal_periods_per_year(interval_str)` in
   `core/backtest.py`:
   ```python
   def _nominal_periods_per_year(interval_str):
       td = _interval_timedelta(interval_str)  # from data.py
       if td is None or td.total_seconds() <= 0:
           return 0.0
       return _SECONDS_PER_YEAR / td.total_seconds()
   ```
2. Require callers to pass `interval_str` explicitly to `run_backtest()`. Add it as a
   required parameter (with a fallback to the index-inference method that emits a
   deprecation warning when `interval_str` is not provided).
3. In `core/pipeline.py::BacktestStep`, pass `interval=pipeline.section("data")["interval"]`
   to `run_backtest()`.
4. Compute and report `realised_bar_count` in the backtest summary: the actual number of
   non-NaN return bars. The Sharpe annualisation uses the nominal rate; the completeness
   report shows `realised_bar_count / nominal_bar_count` as a coverage ratio.

**Acceptance test.**
Construct a 1h index with 10% of bars removed (gaps). Assert that
`_nominal_periods_per_year("1h") == 8766.0` regardless of gap count, and that
`realised_bar_count` in the backtest summary is lower than the nominal bar count for the
same period.

---

### 4.5 VectorBT fallback to pandas adapter is silent (High)

**Root cause.**
`core/backtest.py` imports VectorBT with `except ImportError: vbt = None`. When VectorBT
is unavailable, `run_backtest()` falls back to the pandas adapter without any visible
indicator that the two engines produce different results. A researcher who concludes that
a strategy is profitable based on pandas adapter results cannot be confident those results
hold under a different engine.

**Fix.**
1. Add a config key `backtest.engine_policy`:
   - `"auto"`: use VectorBT if available, pandas adapter otherwise; emit a
     `UserWarning` when falling back.
   - `"vectorbt"`: require VectorBT; raise if unavailable.
   - `"pandas"`: explicitly use the pandas adapter; no warning.
2. Add `engine_used` and `engine_fallback_occurred` to the backtest summary.
3. Add a promotion gate `require_vectorbt_for_certification: bool` (default `False`,
   `True` in local_certification and trade_ready profiles). When `True`, a run that used
   the pandas adapter cannot pass the promotion check.
4. In `example_utils.py`, change the research example default to
   `engine_policy: "auto"` and add a note explaining the difference.

**Acceptance test.**
Run the same pipeline configuration twice: once with VectorBT mocked as unavailable
(monkeypatching `vbt = None`) and once with it available. Assert that
`engine_fallback_occurred == True` in the first run. Assert `engine_used == "vectorbt"` in
the second. Assert the metric values differ, confirming they are not using the same code
path.

---

### 4.6 Stress scenario fill ratios not calibrated to historical market regimes (Medium)

**Root cause.**
`evaluate_stress_realism_gate()` evaluates `worst_fill_ratio` over the explicitly
configured stress scenarios. These scenarios are synthetic (downtime, stale mark, halt)
and do not include market-regime-driven low-liquidity periods based on historical data.

**Fix.**
1. Add a `historical_stress_scenarios` config block under `backtest.scenario_matrix`:
   ```yaml
   historical_stress_scenarios:
     enabled: true
     lookback_start: "2021-01-01"
     lookback_end: "2024-12-31"
     quantile_threshold: 0.05   # bottom 5% liquidity periods
   ```
2. Implement `build_historical_liquidity_stress_schedule(volume_series, quantile_threshold)`
   in `core/scenarios.py` that identifies historical bars where volume was below the
   `quantile_threshold` percentile and returns a scenario schedule marking those bars as
   "low_liquidity_stress".
3. When `historical_stress_scenarios.enabled`, automatically include these periods in the
   scenario matrix alongside synthetic scenarios. Report `historical_stress_fill_ratio`
   separately in the stress summary.
4. Add a gate `min_historical_stress_fill_ratio` (default `0.15` in certification) that
   must be satisfied independently from the synthetic scenario fill ratio.

**Acceptance test.**
Construct a volume series with 5% of bars having volume below the 5th percentile. Assert
`build_historical_liquidity_stress_schedule` identifies those bars. Assert the stress
gate fails when `worst_fill_ratio` on those bars is below `min_historical_stress_fill_ratio`.

---

### 4.7 `min_observations` gate counts abstain trades toward significance floor (Medium)

**Root cause.**
The significance floor `min_observations=64` in certification mode is checked against all
outcome rows, including abstain (`label=0`) outcomes. A strategy with 40 abstains and 24
directional trades appears to meet the floor while having only 24 actual signal samples.

**Fix.**
1. Add a `min_directional_observations` gate alongside `min_observations` in the
   significance config (default: `min_directional_observations = min_observations / 2`).
2. In `core/backtest.py`, compute `directional_trade_count` as the count of rows where
   `signal != 0`, and report it in the backtest summary separately from `trade_count`.
3. Gate promotion on `directional_trade_count >= min_directional_observations`.
4. Report `abstain_rate` (fraction of bars where the signal was 0) in the summary so the
   user can see when the no-trade class is dominating.

**Acceptance test.**
Generate a signal series with 40 abstains and 24 directional signals. Assert
`directional_trade_count == 24`. Assert the significance gate fails when
`min_directional_observations=32` even though total `outcome_count == 64`.

---

## Section 5 — Evaluation Metrics

---

### 5.1 Sharpe ratio computed on bar-level returns, not trade-level (Critical)

**Root cause.**
The equity curve in `run_backtest()` is computed bar by bar. For a strategy that holds a
position for 48 bars and trades infrequently, 90%+ of bars have a zero return (or nearly
zero, because the position is flat). The standard deviation of this series is very small,
producing an artificially inflated annualised Sharpe ratio.

**Fix.**
1. Add a `sharpe_computation_method` config key:
   - `"trade_level"` (new default): compute returns at entry and exit of each trade; the
     Sharpe denominator is the std of trade-level returns, annualised by
     `sqrt(periods_per_year / avg_holding_bars)`.
   - `"bar_level"`: current behaviour; emits a warning when `avg_holding_bars > 4`.
2. Implement `compute_trade_level_sharpe(trade_returns, periods_per_year, avg_holding)` in
   `core/backtest.py`:
   ```python
   n_trades_per_year = periods_per_year / avg_holding
   sr = trade_returns.mean() / trade_returns.std(ddof=1) * np.sqrt(n_trades_per_year)
   ```
3. Report both `sharpe_ratio_bar_level` and `sharpe_ratio_trade_level` in the backtest
   summary so researchers can see the difference.
4. Use `sharpe_ratio_trade_level` as the primary metric in the AutoML objective function
   when the objective is `"risk_adjusted_after_costs"` or `"sharpe_ratio"`.

**Acceptance test.**
Generate a strategy that holds every position for 48 bars with a constant 0.1% per-trade
return. Assert `sharpe_ratio_bar_level > 10` (inflated). Assert
`sharpe_ratio_trade_level ≈ 0.0` (flat, no variance in returns). Assert they differ by at
least 5×.

---

### 5.2 `profit_factor` reported without minimum trade count context (High)

**Root cause.**
The backtest summary includes `profit_factor` as a stand-alone metric. With fewer than 30
directional trades, a profit factor above 1.0 is consistent with random noise at a 95%
confidence level. No context is reported alongside the metric.

**Fix.**
1. Add `profit_factor_significance_met` (bool) to the backtest summary. Set it to `True`
   only when `directional_trade_count >= 30`.
2. Add `profit_factor_95ci` tuple: bootstrap the profit factor distribution over the trade
   list using `n_resamples=1000` and report the 5th–95th percentile interval. A strategy
   with a wide CI is unlikely to be real signal.
3. Add a gate `min_profit_factor_significance_trades` (default `30`) to the certification
   profile. Fail promotion when the trade count is below this, regardless of the profit
   factor value.

**Acceptance test.**
Run a backtest with 5 winning and 2 losing trades. Assert
`profit_factor_significance_met == False`. Assert the 95% CI for profit factor includes
values below 1.0. Assert the promotion gate fails.

---

### 5.3 Maximum drawdown not mark-to-market during open positions (High)

**Root cause.**
The equity curve in `run_backtest()` marks positions to close at each bar but may not
fully reflect intrabar adverse excursions during an open position with a long holding
period. The reported `max_drawdown` is the peak-to-trough of the daily mark-to-market
equity, not the peak-to-trough of the open-position path.

**Fix.**
1. Add `mark_to_market_policy` config key:
   - `"bar_close"`: current behaviour.
   - `"intrabar_high_low"`: at each bar, if a long position is open, mark equity to
     `min(close, low)` for drawdown purposes; if a short position is open, mark to
     `max(close, high)`. This gives a worst-case intrabar equity path.
2. In `run_backtest()`, compute `equity_low_path` (worst-case equity at each bar, using
   low prices for long positions) and report `max_drawdown_intrabar` from this series.
3. Report both `max_drawdown` (close-based) and `max_drawdown_intrabar` (high/low-based)
   in the summary. Use `max_drawdown_intrabar` for the Calmar ratio when available.

**Acceptance test.**
Construct a scenario where a long position experiences a 10% intrabar adverse low but
closes at only 2% below entry. Assert `max_drawdown < 0.03` but
`max_drawdown_intrabar ≈ 0.10`.

---

### 5.4 Calmar ratio unreliable over short backtests (Medium)

**Root cause.**
Calmar ratio = annualised return / max drawdown. On a 3–6 month backtest, max drawdown
is path-dependent and understates the true long-run tail risk.

**Fix.**
1. Add a `calmar_min_bars` gate (default `504`, equivalent to approximately 21 trading
   days × 24 hours for a 1h series; roughly 3 months of 24/7 crypto trading).
2. Set `calmar_ratio_reliable` to `False` in the backtest summary when
   `realised_bar_count < calmar_min_bars` or `directional_trade_count < 30`.
3. Add a `calmar_ratio_90ci` bootstrapped interval, analogous to the profit factor CI.
4. Replace the raw Calmar ratio in the AutoML objective with the lower bound of the
   bootstrapped 90% CI when `calmar_ratio_reliable == False`, effectively penalising
   short backtests.

**Acceptance test.**
Run a 100-bar backtest. Assert `calmar_ratio_reliable == False`. Assert the Calmar CI
lower bound is used in the objective score rather than the raw point estimate.

---

### 5.5 Brier score reported without decomposition (Medium)

**Root cause.**
A scalar Brier score cannot distinguish a well-calibrated uncertain model from a
poorly-calibrated but sharp model. Decomposition into reliability (calibration) and
resolution (sharpness) is standard practice (Murphy 1973).

**Fix.**
1. Implement `decompose_brier_score(y_true, y_prob, n_bins=10)` in `core/models.py`:
   - Bin predictions into `n_bins` equal-width probability bins.
   - **Reliability**: mean squared error of observed frequency vs. predicted probability
     per bin (weighted by bin size).
   - **Resolution**: variance of observed frequencies across bins (higher is better).
   - **Uncertainty**: variance of the overall base rate (constant for a given dataset).
   - Verify: `brier_score = reliability - resolution + uncertainty`.
2. Add `brier_reliability`, `brier_resolution`, `brier_uncertainty` to the evaluation
   output from `evaluate_model()` in `core/models.py`.
3. Add a calibration gate `max_brier_reliability` (default `0.05` in certification) to
   the promotion check. A model with `brier_reliability > 0.05` is poorly calibrated
   and Kelly sizing will be incorrect.

**Acceptance test.**
Construct a model that always predicts `p=0.5`. Assert `brier_resolution == 0.0` (no
discriminatory power). Construct a perfectly calibrated model. Assert
`brier_reliability ≈ 0.0`. Verify the decomposition identity holds within `1e-10`.

---

## Section 6 — Out-of-Sample and Robustness

---

### 6.1 CPCV OOS estimates are statistically dependent (Critical)

**Root cause.**
`core/models.py::cpcv_split()` generates all combinations of `test_block_count` from
`n_blocks` blocks. The same training samples appear in multiple splits. The resulting OOS
metrics are correlated and their average does not equal the expected value of a single
independent test.

**Fix.**
1. Add a `deflation_correction: bool` parameter to `cpcv_split()` and the CPCV evaluation
   loop. When `True`, apply the López de Prado CPCV variance correction:
   ```
   Var_corrected = (1/N_test_paths) × Var_sample + ((N_paths - 1) / N_paths) × Cov_sample
   ```
   where `N_paths` is the number of CPCV paths and `Cov_sample` is estimated from the
   cross-path covariance of path returns.
2. Report `cpcv_path_return_correlation` in the validation metrics. A high average
   cross-path correlation indicates that the CPCV paths are providing less independent
   information than the number of paths suggests.
3. Add a `cpcv_effective_paths` metric: approximate the effective number of independent
   paths using the eigenvalue method on the path return correlation matrix.
4. In the AutoML objective, use `cpcv_effective_paths` to weight the aggregate OOS score:
   when `cpcv_effective_paths` is close to 1, the aggregate score has low statistical
   power and should be penalised (e.g., halved).

**Acceptance test.**
Run CPCV on a pure random-walk series with `n_blocks=5`, `test_block_count=2`. Assert
that `cpcv_path_return_correlation` is close to 0 (random walk, paths should be
uncorrelated). Run on a trend series and assert correlation is higher, demonstrating the
metric captures shared structure.

---

### 6.2 Locked holdout contaminated by full-data regime/stationarity/feature construction (High)

**Root cause.**
The locked holdout is defined as the final chronological slice of the dataset. However,
before the holdout test is run, the full dataset is used for regime detection,
stationarity transform selection, and feature schema construction. The holdout's
distributional properties therefore influence the feature pipeline, making the holdout
score overoptimistic.

**Fix.**
1. Add a `holdout_wall` to the pipeline: all full-data computations must complete before
   a designated `holdout_start_timestamp`. The holdout data is then fully excised from
   the feature engineering phase.
2. Implement a `hold_out_barrier(data, holdout_start)` helper in `core/pipeline.py` that:
   - Returns `train_data = data.loc[:holdout_start - 1 bar]` for feature/regime/stationarity
     construction.
   - Returns `holdout_data = data.loc[holdout_start:]` only for final evaluation.
3. Modify `core/regime.py::build_instrument_regime_state()` to accept `boundary_idx` (as
   per Issue 2.1) and call it with `len(train_data)` when building features for the
   holdout evaluation.
4. Modify `screen_features_for_stationarity()` to use `fit_window = len(train_data)` as
   required by Issue 1.1.
5. Add a `holdout_contamination_risk` flag to the experiment manifest. Set it to `True`
   if any of the above barriers are not applied. Gate promotion on
   `holdout_contamination_risk == False`.

**Acceptance test.**
Remove the holdout period's data entirely from the pipeline inputs. Assert that the
feature values for the pre-holdout period are identical whether or not the holdout data
was included. This confirms no future holdout information leaked into the feature pipeline.

---

### 6.3 No hyperparameter sensitivity analysis (High)

**Root cause.**
The pipeline reports the best hyperparameter configuration but does not measure whether
small perturbations produce comparable scores. A fragile model with a narrow performance
peak is indistinguishable from a robust one in the current output.

**Fix.**
1. Add a `sensitivity_analysis` config block to the AutoML section:
   ```yaml
   sensitivity_analysis:
     enabled: true
     n_perturbations: 10
     perturbation_scale: 0.1   # ±10% of each continuous param
   ```
2. After selecting the best trial, implement
   `compute_hyperparameter_sensitivity(best_params, eval_fn, n_perturbations, scale)`
   in `core/automl.py`:
   - For each hyperparameter, generate `n_perturbations` neighbours by perturbing the
     value by ±`scale`.
   - Evaluate the objective at each neighbour (using the same train/test split, no
     refitting from scratch).
   - Compute the sensitivity as: `std(neighbour_objectives) / |best_objective|`.
3. Report `param_sensitivity` per hyperparameter and `overall_sensitivity_index`
   (maximum sensitivity across all params) in the study summary.
4. Add a gate `max_sensitivity_index` (default `0.5` in certification). A model that
   degrades by more than 50% with a 10% parameter perturbation is fragile and should not
   be promoted.

**Acceptance test.**
Fit a GBM with `learning_rate=0.1`. Perturb to `0.09` and `0.11`. Assert the sensitivity
index is computed and stored. Assert a model that collapses completely under ±10%
perturbation exceeds `max_sensitivity_index=0.5` and fails the gate.

---

### 6.4 Regime ablation reports pass with zero ablations (Medium)

**Root cause.**
`core/regime.py::summarize_regime_ablation_reports()` returns `promotion_pass=True` when
no required ablations fail. When there are zero ablation reports (none were run), the
function still returns `promotion_pass=True`, interpreting absence of evidence as evidence
of absence.

**Fix.**
1. Change the logic in `summarize_regime_ablation_reports()` to:
   ```python
   if not ablation_reports:
       return {
           "promotion_pass": False,
           "reason": "no_ablation_reports",
           "required_ablations_met": False,
           ...
       }
   ```
2. Add a `min_required_ablation_reports` config key (default `1` in research,
   `n_regimes` in certification). When the number of completed ablation reports is below
   this threshold, set `promotion_pass=False` with reason `"insufficient_ablation_coverage"`.
3. In the AutoML trial runner, ensure that at least one regime ablation report is
   generated per trial when `n_regimes >= 2`. Add this as a required trial step.

**Acceptance test.**
Call `summarize_regime_ablation_reports([])`. Assert `promotion_pass == False` and
`reason == "no_ablation_reports"`. Call with one report that passes. Assert
`promotion_pass == True`.

---

### 6.5 Sequential bootstrap uniqueness threshold not calibrated to label concurrency (Medium)

**Root cause.**
`core/models.py::_resolve_rf_sampling_report()` activates sequential bootstrap when
`mean_uniqueness < 0.90`. For a 24-bar holding period on 1h data, average label
concurrency is roughly `1/24 = 0.042`, giving uniqueness close to `1 - 0.042 = 0.96`,
above the threshold. Standard bootstrap is used even though labels clearly overlap.

**Fix.**
1. Replace the uniqueness threshold with a **maximum_holding_period** heuristic:
   ```python
   def _should_use_sequential_bootstrap(max_holding, mean_uniqueness, uniqueness_threshold):
       # If max_holding > 1, labels necessarily overlap. Use sequential bootstrap.
       if max_holding > 1:
           return True
       return mean_uniqueness < uniqueness_threshold
   ```
2. Add `max_holding` as a parameter to `_resolve_rf_sampling_report()`, sourced from the
   label config.
3. Change `uniqueness_threshold` default from `0.90` to `0.70` — a more conservative
   value that triggers sequential bootstrap earlier. Expose it via
   `model.sequential_bootstrap_uniqueness_threshold` in the config.
4. Report `sequential_bootstrap_activated` in the sampling report and include it in the
   trial artifact so it is auditable.

**Acceptance test.**
Configure `max_holding=24`. Assert `_should_use_sequential_bootstrap(24, 0.95, 0.90)`
returns `True` (override by holding period). Assert that `sequential_bootstrap_used: True`
appears in the training report.

---

## Section 7 — Deployment Realism

---

### 7.1 ADWIN drift detection too slow for infrequent trading (Critical)

**Root cause.**
`core/drift.py::ADWINDetector(delta=0.002)` requires a statistically significant shift in
the stream mean before triggering. For a strategy that trades 3 times per day on 1h data,
ADWIN processes performance observations at the trade rate, not the bar rate. At
`delta=0.002`, detection requires hundreds of samples. A structural break visible to a
human in 5–10 bars may take weeks of trading to reach statistical significance.

**Fix.**
1. Add a parallel **feature-based early-warning system** to `DriftMonitor` that operates
   at the bar rate, not the trade rate:
   - Compute PSI and KS test on the most recent 48 bars of each feature vs. the reference
     window at every bar (not just when a new trade occurs).
   - Trigger a "soft drift alert" when PSI > 0.1 on 3 or more features simultaneously.
     Soft alerts do not trigger retraining but increment a drift pressure counter.
   - When the drift pressure counter reaches `max_pressure_bars` (default 24), trigger
     retraining even if ADWIN has not yet reached significance.
2. Add `regime_change_detector` as a supplemental detector: use the CUSUM score from
   `core/regime.py::_online_cusum_score()` on the price series. When CUSUM exceeds
   `cusum_threshold` (default `5.0`), add a "structural break" alert to the drift report.
3. Report `drift_pressure_bars` and `regime_cusum_score` in `DriftMonitor.check()`.
4. Add `drift_detection_lag_estimate` to the drift report: estimated bars between regime
   change onset and ADWIN trigger, based on the configured `delta` and the estimated effect
   size from the CUSUM score.

**Acceptance test.**
Construct a performance stream that degrades by 0.05 per step after bar 50. Assert that
the feature-based early-warning system fires within 24 bars of bar 50. Assert that
`drift_pressure_bars` reaches `max_pressure_bars=24` before ADWIN fires, demonstrating
the early-warning system is faster.

---

### 7.2 Cooldown and min-sample thresholds not interval-normalised (High)

**Root cause.**
`DriftMonitor.config` has `cooldown_bars=500` and `min_samples=200` as fixed bar counts.
The semantic meaning of these values differs by orders of magnitude across trading
intervals (1h vs. 5m).

**Fix.**
1. Add an `interval` parameter to `DriftMonitor.__init__()`, and add
   `cooldown_calendar_days` and `min_samples_calendar_days` as alternative config keys:
   ```python
   if "cooldown_calendar_days" in config:
       td = _interval_timedelta(interval)
       bars_per_day = 86400 / td.total_seconds()
       config["cooldown_bars"] = int(config["cooldown_calendar_days"] * bars_per_day)
   ```
2. In `core/pipeline.py`, always pass the configured interval to `DriftMonitor`.
3. Change the config defaults to:
   ```python
   "cooldown_calendar_days": 14,   # 2 weeks
   "min_samples_calendar_days": 7, # 1 week
   ```
   These translate to 336 bars on 1h data and 2016 bars on 5m data — meaningful and
   comparable windows regardless of interval.
4. Report `cooldown_bars` and `min_samples` in the drift report after interval
   normalisation so the researcher can verify the effective thresholds.

**Acceptance test.**
Construct a `DriftMonitor` with `interval="5m"` and `cooldown_calendar_days=7`. Assert
`cooldown_bars = 7 * 24 * 12 = 2016`. Repeat with `interval="1h"`. Assert
`cooldown_bars = 7 * 24 = 168`.

---

### 7.3 No model warm-up or burn-in period after retraining (High)

**Root cause.**
After a drift-triggered retrain, the new model begins issuing signals on the next bar.
The calibration was fit on historical data; the new model's probability outputs are not
yet verified against the live distribution.

**Fix.**
1. Add a `post_retrain_policy` config block:
   ```yaml
   post_retrain_policy:
     mode: "shadow"         # or "active", "ramped"
     shadow_bars: 24        # bars in shadow mode before going live
     ramp_bars: 12          # bars to linearly scale up Kelly fraction
     min_oos_trades: 5      # minimum live trades before full activation
   ```
2. Implement `PostRetrainShadowController` in `core/drift.py`:
   - In `"shadow"` mode, the new model generates predictions internally but the old model
     continues to generate live signals. The shadow model's predictions are logged for
     comparison.
   - After `shadow_bars` bars, switch to the new model if its shadow performance (vs. old)
     meets the gate `min_shadow_directional_accuracy >= 0.45`.
   - In `"ramped"` mode, linearly increase the Kelly fraction from 0 to the configured
     maximum over `ramp_bars` bars.
3. Report `model_warmup_mode`, `warmup_bars_remaining`, and `shadow_accuracy` in the live
   inference output.

**Acceptance test.**
Simulate a retrain event. Assert that the old model continues generating signals for
`shadow_bars=24` bars. Assert the new model activates only after its shadow performance
meets the gate. Assert `warmup_bars_remaining` decrements to 0 after 24 bars.

---

### 7.4 Research monitoring profile sets all limits to infinity (High)

**Root cause.**
`core/monitoring.py::_POLICY_PROFILES["research"]` is an empty dict. All limits
inherit `_DEFAULT_POLICY` values of `None` or `np.inf`. A researcher using the research
profile will never see a monitoring failure, even if the live data is completely degraded.

**Fix.**
1. Replace the empty research profile dict with a set of permissive but non-infinite
   defaults:
   ```python
   "research": {
       "max_data_lag": "48h",
       "max_custom_ttl_breach_rate": 0.5,
       "max_fallback_assumption_rate": 0.5,
       "min_fill_ratio": 0.05,
       "max_slippage_gap_share": 0.95,
       "fail_closed_on_schema_drift": True,
       "required_components": [
           "raw_data_freshness",
           "feature_schema",
       ],
   }
   ```
2. Add a `monitoring_profile_upgrade_path` to the deployment readiness report: show the
   delta between the current profile's limits and the trade_ready profile's limits so
   researchers can see what gets tighter.
3. Add a gate in `core/readiness.py`: if `policy_profile == "research"` and the
   deployment readiness is being evaluated, emit a `WARNING` and set
   `monitoring_adequate_for_deployment: False`.

**Acceptance test.**
Create a monitoring report with `max_data_lag=72h` (exceeds research default of 48h).
Assert the report flags `data_freshness: FAIL` under the research profile. Assert the same
report passes under a looser custom profile. Assert `monitoring_adequate_for_deployment`
is `False` when using the research profile in a readiness evaluation.

---

### 7.5 Champion/challenger comparison uses different OOS periods (Medium)

**Root cause.**
`core/registry/__init__.py::evaluate_challenger_promotion()` compares the challenger's
submitted evidence against threshold gates. There is no requirement that both the champion
and challenger were evaluated on the same OOS date range.

**Fix.**
1. Add an `oos_evaluation_period` field to the model registry entry at promotion time:
   ```json
   { "oos_start": "2024-01-01", "oos_end": "2024-07-01" }
   ```
2. In `evaluate_challenger_promotion()`, check whether the challenger's `oos_evaluation_period`
   overlaps with the champion's. If overlap < 50% of the champion's period, set
   `comparable_evaluation_period: False` and require manual review (emit a `UserWarning`
   and set a gate).
3. Add an optional `force_shared_oos_period: bool` flag. When `True`, the challenger
   evaluation must re-run using the champion's exact OOS period dates. Implement this as
   a pipeline parameter that re-fetches and re-evaluates data for the canonical period.
4. Report `overlap_fraction` and `comparable_evaluation_period` in the promotion decision
   report.

**Acceptance test.**
Register a champion with `oos_end="2024-07-01"`. Submit a challenger evaluated on
`2025-01-01 – 2025-06-01`. Assert `overlap_fraction == 0.0`. Assert
`comparable_evaluation_period: False`. Assert promotion is blocked when
`force_shared_oos_period=True`.

---

### 7.6 Pickle fallback when skops is unavailable (Low)

**Root cause.**
`core/models.py` falls back to pickle serialisation when skops is not installed. Pickle
deserialisation executes arbitrary Python code and is a security risk when loading from
untrusted sources. On consumer hardware with API keys in environment variables, this is
a genuine attack vector.

**Fix.**
1. Change the serialisation fallback chain:
   - **First choice**: skops (safe, restricted deserialisation).
   - **Second choice**: joblib (safer than pickle for sklearn models; supports
     compression).
   - **Fallback of last resort**: pickle with a `UserWarning` message: *"Pickle
     serialisation is being used. Never load model artifacts from untrusted sources.
     Install skops for safe deserialisation."*
2. Add a `require_safe_serialisation: bool` config key (default `False` in research,
   `True` in certification). When `True`, raise `ImportError` if neither skops nor a safe
   alternative is available.
3. In model load paths, add an integrity check: store a SHA-256 hash of the serialised
   artifact at save time and verify it at load time. This prevents tampering even if
   pickle is used.
4. Add skops to `requirements.txt` as a non-optional dependency.

**Acceptance test.**
Mock skops as unavailable. Assert a `UserWarning` is raised mentioning pickle. Assert that
when `require_safe_serialisation=True`, a `RuntimeError` is raised rather than silently
falling back to pickle. Assert that SHA-256 verification detects a tampered artifact.

---

## Section 8 — Structural Failure Modes

The structural failure modes listed in Audit #2 Section 8 are consequences of the issues
addressed above. The following table cross-references each failure mode to its primary
remediation plan.

| Failure Mode | Primary Remediation |
|---|---|
| Label-parameter leakage masquerading as model quality | 3.1 (label params in thesis contract) |
| Stationarity-transform selection on full dataset | 1.1, 1.2 (fold-isolated transform fitting) |
| Regime labels providing implicit future context | 2.1 (online regime labels) |
| Short backtests inflating Calmar and Sharpe | 5.1, 5.4 (trade-level Sharpe, Calmar guard) |
| Post-selection inference decorating noise | 3.3 (Deflated Sharpe Ratio) |
| Zero-fill missing funding events | Audit #1 finding; enforce `preserve_missing` default in all modes |
| IS Kelly sizing bet-size inflation | 4.1 (model-probability Kelly) |

**Additional remediation for zero-fill funding (from Audit #1).**
Change `core/pipeline.py::_resolve_backtest_funding_missing_policy()` to default to
`"preserve_missing"` in research mode instead of `"zero_fill"`. Require the caller to
explicitly opt into zero-fill by setting `backtest.funding_missing_policy: "zero_fill_debug"`.
Add `funding_zero_fill_used: bool` to the backtest summary as a top-level flag.

---

## Section 9 — Retail-Specific Risk Amplifiers

---

### 9.1 Non-determinism from `n_jobs=-1` in RandomForestClassifier (High for reproducibility)

**Fix.**
1. Change the default `n_jobs` for `RandomForestClassifier` in `build_model()` from `-1`
   to `1` in research and certification modes. Use `-1` only in explicitly configured
   high-throughput research environments.
2. Add `n_jobs` to the thesis contract (alongside `random_state`): it must be fixed before
   a study begins.
3. Add a `reproducibility_check` to the experiment manifest: run the first trial twice with
   the same seed and assert that the objective values are identical. If they differ, set
   `reproducible: False` and abort.

**Acceptance test.**
Run a RandomForest trial with `n_jobs=-1` on a 4-core machine and a 1-core setting.
Assert predictions are identical. Assert that setting `n_jobs=1` produces identical results
across runs on any hardware.

---

### 9.2 No rate-limit-aware retraining scheduler (High for operations)

**Fix.**
1. Add a `data_fetch_budget` config block to the retraining orchestrator:
   ```yaml
   data_fetch_budget:
     max_klines_per_retrain: 5000
     request_weight_per_kline: 2   # Binance klines = 2 weight per 1000 candles
     max_weight_per_minute: 1200
     retry_after_seconds: 60
   ```
2. Implement `estimate_fetch_weight(symbol, interval, start, end)` in `core/data.py`
   that computes the expected API weight for a full backfill without issuing any requests.
3. In `core/orchestration.py::run_drift_retraining_cycle()`, check the estimated weight
   before fetching. If the estimated weight exceeds `max_weight_per_minute`, split the
   fetch into batches with `time.sleep(1.0 / batch_rate)` between requests.
4. Add `estimated_api_weight` and `actual_api_weight` to the retraining run report.

**Acceptance test.**
Configure `max_klines_per_retrain=500` and attempt a backfill that requires 2000 klines.
Assert the fetch is split into 4 batches. Assert `estimated_api_weight` is computed before
any HTTP requests are made. Assert the total elapsed time includes inter-batch delays.

---

### 9.3 No `minNotional` guard in paper research (High for live parity)

**Fix.**
1. Add `minNotional` validation to `core/backtest.py::_validate_order_intent()` for paper
   research mode (it already exists for live mode). Currently the `min_notional` key is
   checked only when `symbol_filters` is non-empty.
2. In `core/pipeline.py`, always fetch symbol filters via
   `fetch_binance_symbol_filters(symbol, market)` before running a backtest, even in
   research mode. Store them in the pipeline state and pass to `run_backtest()`.
3. Add a `filter_compliance_rate` metric to the backtest summary: the fraction of intended
   orders that passed all symbol filter checks. A compliance rate below 0.95 means the
   strategy's assumed fill rate is not achievable at Binance.
4. Add a gate `min_filter_compliance_rate` (default `0.95` in certification).

**Acceptance test.**
Configure a strategy with equity = $10 and Kelly fraction = 0.5 → intended notional = $5.
Set `minNotional = $10` in symbol filters. Assert the order is rejected with
reason `"min_notional"` and `filter_compliance_rate < 1.0`. Assert the certification gate
fails.

---

## Audit #1 Open Items

The following findings from Audit #1 (2026-04-30) were not superseded by Audit #2 plans
and require separate remediations.

---

### A1.1 Lookahead guard disabled in `example_trade_ready_automl.py` (Critical)

**Fix.**
1. Remove `features.lookahead_guard.enabled = False` from
   `example_trade_ready_automl.py`. The guard must always be enabled for trade-ready runs.
2. In `core/automl.py`, change the promotion gate read from
   `lookahead_guard.get("promotion_pass", True)` to
   `lookahead_guard.get("promotion_pass", False)`. Absent evidence is treated as failure.
3. Add a hard check: if `lookahead_guard.enabled == False` in the config, set
   `promotion_pass = False` in the lookahead report and log a `CRITICAL` warning.

**Acceptance test.**
Run the trade-ready config with `lookahead_guard.enabled=False`. Assert `promotion_pass`
in the lookahead report is `False`. Assert the promotion eligibility report shows
`"lookahead_guard_disabled"` as a blocking failure.

---

### A1.2 Fail-open governance gates (High)

The five governance functions that return `promotion_pass=True` on missing or zero
evidence must be changed to return `False` unless evidence is explicitly provided.
Affected functions:
- `evaluate_feature_portability()` — pass requires `len(top_features) > 0`.
- `summarize_feature_admission_reports()` — pass requires `len(reports) > 0`.
- `summarize_regime_ablation_reports()` — covered by Issue 6.4 plan above.
- Any AutoML gate that reads `gate_report.get("promotion_pass", True)` — change default
  to `False`.

**Fix.**
Apply the change `get("promotion_pass", True)` → `get("promotion_pass", False)` to all
gate reads in `core/automl.py`. Similarly change `evaluate_feature_portability()` and
`summarize_feature_admission_reports()` to return `promotion_pass=False` when called
with empty inputs, with `reason="no_evidence"`.

**Acceptance test.**
Call each of the five functions with empty inputs. Assert each returns
`promotion_pass=False` and `reason` contains `"no_evidence"` or `"no_reports"`.

---

### A1.3 Research futures backtests default to zero-fill missing funding (High)

**Fix (summary — detailed in Section 8 table above).**
Change the `_resolve_backtest_funding_missing_policy()` default from `"zero_fill"` to
`"preserve_missing"`. Add `funding_zero_fill_used` as a top-level boolean flag in the
backtest summary. When `funding_zero_fill_used=True`, gate certification promotion.

---

### A1.4 Universe eligibility hardcoded with fabricated liquidity in examples (Medium-High)

**Fix.**
1. Remove all hardcoded `status="TRADING"` and `listing_start` fields from
   `example_utils.py::build_example_universe_config()`.
2. Replace with calls to `core/universe.py::load_historical_universe_snapshot()`. If no
   snapshot is available, the example must print a clear message explaining that the
   universe config is synthetic and results are not survivorship-safe.
3. Add a `universe_is_synthetic: bool` field to the universe report and the experiment
   manifest. Gate certification on `universe_is_synthetic == False`.

---
