# Binance Derivatives Indicator Extension Plan

## Objective

Add Binance USD-M derivatives context to the existing research examples without changing any core pipeline, training, AutoML, or shared data-abstraction code. The extension surface remains limited to indicator files and example wiring.

## Final Scope

- Keep all shared research pipeline abstractions untouched.
- Add derivatives context only through new indicator modules plus example-level configuration.
- Keep funding on derivative-based event changes, but migrate open interest to non-level change features.
- Support both the existing spot-base MTF FVG example and an added futures-base variant.

## OI Feature Migration Plan

### Objective

Replace the current open-interest first-difference feature family with three non-level, scale-normalized transforms that better capture positioning change without anchoring the model to symbol-specific OI magnitude.

### Target Feature Set

The current application will use exactly these three OI features:

- `oi_ctx_log_change_4h`: multi-step log change of OI notional over a fixed four-hour horizon.
- `oi_ctx_trend_spread`: short-vs-long EMA spread of log OI, computed on the OI event stream before alignment.
- `oi_ctx_pressure_vs_notional_volume`: aligned OI notional delta divided by same-bar traded notional volume, used as a participation-intensity proxy.

### Technical Steps

1. Extend `OpenInterestContext` parameters so the emitted feature contract is explicit rather than implicit.
	- Replace the generic `rolling_window`-based derivative surface with `change_horizon`, `trend_short_span`, and `trend_long_span`.
	- Keep `period`, `warmup`, `max_age`, and coverage logic unchanged.

2. Compute OI transforms on the raw OI event stream before asof alignment.
	- Build a positive-only `log_oi_notional = log(sumOpenInterestValue)` series.
	- Convert `change_horizon` into an integer number of OI event steps using the configured OI sampling `period`.
	- Compute event-level `log_change_4h = log_oi_notional - log_oi_notional.shift(horizon_steps)`.
	- Compute event-level EMA spread as `EMA_short(log_oi_notional) - EMA_long(log_oi_notional)`.
	- Compute event-level OI notional delta strictly as an intermediate for the normalized pressure feature; do not emit raw delta itself.

3. Align only the derived OI transforms to the base decision index.
	- Use the existing conservative backward-asof alignment path.
	- Preserve the existing `max_age` and coverage gate behavior.
	- Emit `oi_ctx_age_minutes` as before because it is an availability diagnostic, not a level feature.

4. Normalize OI pressure by same-bar traded notional.
	- After alignment, divide aligned OI notional delta by `close * volume` on the base frame.
	- Guard zero and missing traded-notional values by replacing them with `NaN` before division.
	- Keep the result signed so it distinguishes accumulation from unwind.

5. Retarget all immediate consumers to the new OI contract.
	- Update `DerivativesCombined` to consume `oi_ctx_log_change_4h`, `oi_ctx_trend_spread`, and `oi_ctx_pressure_vs_notional_volume`.
	- Update example diagnostics and regime-feature selection in `example_mtf_fvg.py` so they reference the new OI columns.
	- Update the focused OI and combined-indicator tests to assert the new output surface and semantics.

6. Validate with the narrowest executable checks first.
	- Run `pytest tests/test_derivatives_indicator_extensions.py` immediately after the indicator and consumer edits.
	- If that passes, rerun `python example_mtf_fvg.py` and `python example_mtf_fvg_futures.py`.

### Falsifiable Local Hypothesis

If `OpenInterestContext` computes and aligns the three non-level OI transforms on the event stream, and all direct consumers are updated to the new column names, the untouched pipeline will continue to ingest the features correctly and the existing focused derivatives tests will still pass after their expectations are updated.

### Cheap Disconfirming Check

The first discriminating check is `pytest tests/test_derivatives_indicator_extensions.py`. If it fails, the defect is local to the new OI feature contract or its immediate combined-indicator consumer rather than the broader pipeline.

## Confirmed Implementation Decisions

- Derivatives context source: Binance USD-M futures REST endpoints only.
- Data access model: deterministic local on-disk cache owned by the indicator layer.
- Funding feature policy: never expose a funding event before its timestamp; use the first derivative of funding as the primary modeled signal.
- Open-interest feature policy: fetch recent `5m` OI history, align backward with asof semantics, and emit only non-level OI change features.
- Example controls: boolean toggles remain in the example config dict.
- Interaction features: use a dedicated combined indicator that consumes funding-change features plus the new non-level OI change columns.
- Architecture boundary: direct example imports register the new indicators locally without changing shared exports.

## Architecture Constraints

- No edits to pipeline orchestration, feature-building logic, training flow, AutoML, or shared Binance adapters.
- No special-case branches in the core pipeline for funding or open interest features.
- All new market-context feature generation happens in indicator files.
- Example integration imports and configures the indicators directly.

## Implemented Components

### 1. Shared Derivatives Helper

Implemented a private indicator-local helper in `core/indicators/_derivatives_binance.py` for:

- UTC normalization,
- deterministic cache reads and writes,
- paginated Binance USD-M REST fetching,
- interval conversion for OI history,
- conservative asof alignment to a base frame,
- local coverage diagnostics.

### 2. Funding Indicator

Implemented `FundingRateContext` in `core/indicators/funding_rate.py`.

Current emitted outputs:

- `funding_ctx_delta`
- `funding_ctx_abs_delta`
- `funding_ctx_mean_<window>`
- `funding_ctx_z_<window>`
- `funding_ctx_age_hours`
- `funding_ctx_extreme_pos`
- `funding_ctx_extreme_neg`
- `funding_ctx_observed_event`

Final semantic change:

- Raw funding level outputs were removed from the modeled feature surface.
- The indicator now computes `diff(funding_rate)` at the event level before alignment.

### 3. Open-Interest Indicator

Implemented `OpenInterestContext` in `core/indicators/open_interest.py`.

Current emitted outputs:

- `oi_ctx_log_change_4h`
- `oi_ctx_trend_spread`
- `oi_ctx_pressure_vs_notional_volume`
- `oi_ctx_age_minutes`

Final semantic change:

- Raw OI level outputs were removed from the modeled feature surface.
- The indicator now computes a four-hour log OI change, a short-vs-long EMA spread on log OI, and OI pressure normalized by traded notional volume.

### 4. Combined Derivatives Indicator

Implemented `DerivativesCombined` in `core/indicators/derivatives_combined.py`.

Current interaction outputs:

- `deriv_combo_funding_delta_x_return`
- `deriv_combo_oi_log_change_x_vol`
- `deriv_combo_price_up_oi_trend_up`
- `deriv_combo_price_down_oi_trend_up`
- `deriv_combo_funding_delta_return_sign`
- `deriv_combo_crowding_proxy`
- `deriv_combo_funding_x_oi_pressure`

### 5. Spot Example Integration

Updated `example_mtf_fvg.py` to:

- import and configure the new indicators directly,
- switch to a recent rolling window when live OI context is enabled,
- keep shared futures-context core features disabled,
- print derivatives alignment and distribution diagnostics,
- use derivative-based funding plus non-level OI outputs in the regime feature builder.

### 6. Futures Example Integration

Added `example_mtf_fvg_futures.py` as the extra example requested.

Key details:

- Uses futures base bars instead of spot base bars.
- Reuses the same derivative-first indicator stack and custom-data feature builders.
- Attaches runtime funding history at the example layer only.
- Normalizes funding timestamps onto the fetched futures bar grid to absorb Binance funding timestamp millisecond jitter.
- Uses research-mode funding coverage policy `zero_fill_debug` for this example because CPCV training evaluates funding both on the raw futures bar index and on sparse fold-level research indices. Strict funding coverage is appropriate for capital-facing paths, but it is too brittle for this research-only example once custom-data alignment drops many bars from the modeling matrix.

## Validation Strategy And Outcome

### Focused Local Hypothesis

The indicator contract was sufficient as long as derivatives context stayed fully encapsulated inside indicator modules and examples. The only expected failure mode was timestamp alignment or coverage handling, not feature ingestion by the pipeline.

### Cheap Disconfirming Check

Focused synthetic tests were added to verify:

- funding timestamps are not visible before their event time,
- open-interest values align backward with asof semantics,
- the combined indicator emits finite outputs.

### Executed Validation

- `pytest tests/test_derivatives_indicator_extensions.py`
	- passed: `3/3`
- `python example_mtf_fvg.py`
	- completed successfully after the non-level OI migration
- `python example_mtf_fvg_futures.py`
	- completed successfully after the non-level OI migration

Latest validated spot-example outcome:

- end equity: `$9,923.85`
- net return: `-0.76%`

Latest validated futures-example outcome:

- end equity: `$10,102.46`
- net return: `1.02%`
- funding PnL: `$0.72`
- funding coverage: `status=strict  missing=0  pass=True` at final standalone backtest stage

## Execution Notes

- The spot and futures examples now use a recent rolling window when open interest is enabled. This is an example-layer accommodation for Binance's recent-history OI endpoint, not a pipeline change.
- The open-interest endpoint must be chunked by `limit x period`, not by one wide request window. For `5m` OI history the public endpoint otherwise returns only the most recent page.
- Cached derivatives timestamps can mix fractional-second and whole-second ISO strings. The cache reader now parses mixed timestamp formats explicitly.
- Binance funding timestamps can also include millisecond jitter. The futures example normalizes funding events onto the actual bar grid before attaching them to runtime context.
- Research-only futures examples that manually attach funding onto sparse custom-data pipelines should not assume strict fold-level coverage is satisfiable. The repo's existing `zero_fill_debug` research policy is the correct example-layer fit; strict coverage remains appropriate for certification and trade-ready paths.

## Progress

- [x] Requirements clarified with implementation questions.
- [x] Extension surface confirmed: indicator files plus example wiring only.
- [x] Shared derivatives helper implemented.
- [x] Funding indicator implemented.
- [x] Open-interest indicator implemented.
- [x] Combined derivatives indicator implemented.
- [x] `example_mtf_fvg.py` wired with boolean toggles and diagnostics.
- [x] Funding semantics kept on event-level derivatives.
- [x] Open-interest semantics migrated from first derivatives to non-level change features.
- [x] Extra futures-base example added in `example_mtf_fvg_futures.py`.
- [x] Runtime funding attachment normalized at the example layer without changing core pipeline code.
- [x] Lightweight validation tests added.
- [x] Focused validation executed.