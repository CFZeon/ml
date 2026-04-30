# Adversarial Audit

- Date: 2026-04-30
- Scope: data ingestion, preprocessing, feature engineering, AutoML selection, backtesting, evaluation, and deployment realism for a retail trader on consumer hardware.
- Industry baseline used for comparison: purged/embargo time-series validation for financial labels, strict point-in-time feature availability, Binance kline open-time semantics, and conservative treatment of missing futures funding events.
- Bottom line: the core split discipline is better than most retail trading stacks, but several decisive controls are still fail-open, mode-dependent, or bypassed in shipped entrypoints. That is enough to invalidate profitability claims.

## Highest-Impact Findings

- Critical: the operator-facing trade-ready example disables the pre-training lookahead replay while the promotion stack still treats lookahead as passed when the report is missing or disabled. `example_trade_ready_automl.py` sets `features.lookahead_guard.enabled = False`; `core/automl.py` then reads `lookahead_guard.get("promotion_pass", True)`. Result: a run can present as lookahead-cleared without actually running the feature-surface replay.

- Critical: custom data joins are not strict point-in-time by default. `core/data.py` sets `CustomDataset.default_allow_exact_matches = True`, and `join_custom_dataset()` merges external data onto market bars indexed by Binance kline open time. Any external feature stamped exactly at the bar timestamp is treated as available for that bar. For coarse feeds, vendor aggregates, and many event datasets, that is an availability leak.

- High: missing candles are tolerated in research mode, and the labeler does not verify forward-window continuity. `core/data.py::fetch_binance_vision(..., gap_policy="warn")` allows incomplete windows outside fail-closed modes. `core/labeling.py::triple_barrier_labels()` never checks that future `high`, `low`, and `close` windows are complete before testing PT/SL barriers. NaNs in future highs/lows silently convert barrier hits into misses, biasing labels toward benign time exits and overstating model quality.

- High: data-quality quarantine is advisory unless the run is capital-facing or the caller explicitly blocks on quarantine. `core/data_quality.py` defaults return spikes, range spikes, quote-volume inconsistencies, and trade-count anomalies to `"flag"`; `core/pipeline.py::DataQualityStep` keeps those rows in `raw_data` and downstream features/labels. The model can therefore learn from exchange glitches, wash-trade bursts, or reporting errors while the run still looks operationally clean.

- High: robustness governance is fail-open on missing evidence. `core/feature_governance.py::evaluate_feature_portability()` passes when there is no meaningful top-feature evidence; `summarize_feature_admission_reports()` returns `promotion_pass = True` on zero reports; `core/regime.py::summarize_regime_ablation_reports()` passes when no required ablations fail; `core/automl.py` then reads several gates with `get(..., True)`. Missing portability, admission, or regime-stability evidence is therefore interpreted as success rather than unknown.

- High: research futures backtests zero-fill missing funding events. `core/pipeline.py::_resolve_backtest_funding_missing_policy()` defaults research mode to `zero_fill`, and `core/backtest.py::_normalize_runtime_funding_rates()` fills missing funding with `0.0` whenever the run is not capital-facing. Binance funding history is a discrete event series with bounded endpoint coverage; converting unknown carry into zero carry mechanically inflates net returns.

- Medium-High: the shipped universe gate in the examples is synthetic, not historical. `example_utils.py::build_example_universe_config()` hardcodes `status="TRADING"`, `listing_start="2020-01-01T00:00:00Z"`, and fabricated liquidity. Any cross-symbol, cross-context, or lifecycle conclusion drawn from example configs is not survivorship-safe and understates listing/liquidity failure modes.

- Medium: the core backtest API is optimistic by default if callers bypass the pipeline helpers. `core/backtest.py::run_backtest()` falls back to executing on `close` when `execution_prices` is omitted. The example builders usually override this with open-price execution plus signal delay, but the underlying primitive is still easy to misuse in a same-bar-fill way.

## Crypto-Specific Integrity Gaps

- Binance-specific anomaly handling exists, but outside strict modes it is not binding. That matters more in crypto than equities because the market never closes, outages create genuine missing path information, and venue-specific prints can dominate bar-level features.

- Cross-venue integrity is optional in the baseline research configs. The repo has reference validation and portability controls, but the accessible research path can still learn on Binance-only price, volume, and taker-flow anomalies without forcing a cross-venue sanity check.

- The code handles UTC normalization and 24/7 timestamps correctly enough. The larger crypto failure mode is not timezone drift; it is treating incomplete weekend or outage periods as if they were harmless sparse observations.

## AutoML And OOS Interpretation

- The split mechanics are stronger than average retail AutoML: CPCV, embargo, locked holdout, and post-selection tests are present. The main invalidator is not random CV. It is that several certification gates can be disabled, advisory, or passed on missing evidence.

- Because portability, regime, and lookahead governance can fail open, the selection stack can still elevate a venue-specific, leakage-adjacent, or weakly evidenced model while emitting a summary that sounds institutionally hardened.

- For a retail user, this is the dangerous pattern: the system is statistically literate enough to inspire confidence, but not fail-closed enough to justify confidence.

## Backtest / Deployment Realism

- The repo distinguishes research-only, local-certification, and trade-ready modes, which is good. But the research path remains permissive enough that profitable surrogate results can survive materially degraded data integrity assumptions.

- Futures research results are especially fragile because missing funding is softened instead of treated as unknown cost. A strategy can look carry-neutral in research while being materially negative once real funding debits are applied.

- The trade-ready example’s disabled lookahead guard is the biggest false-robustness problem in the deployment story. It weakens the single control that is supposed to certify the causal feature surface before any model evidence is interpreted as promotion-relevant.

## Ways This Can Look Profitable But Fail Live

- A custom feature published at the bar timestamp is merged into that same bar and appears predictive; live you only know it after the decision.

- An outage removes part of the future price path; the triple-barrier labeler misses a stop-loss or profit-taking event and converts the sample into a gentler time-barrier outcome.

- Missing funding events are treated as zero funding; futures Sharpe survives research but collapses once actual carry is applied.

- A wash-trade-like volume spike is only flagged, not removed; the model learns a feature that will not survive on clean live data.

- A trade-ready run appears to have passed the lookahead gate even though the guard was disabled upstream.

- Example-level universe eligibility looks robust because the symbols were predeclared tradable with fabricated liquidity; live listing and liquidity churn were never tested.

## Final Assessment

- The architecture knows about the right institutional controls.

- The invalidating weakness is that too many of those controls are still fail-open, mode-dependent, or contradicted by shipped entrypoints.

- That combination can easily produce outputs that look statistically respectable while still overstating deployable edge for a retail trader.