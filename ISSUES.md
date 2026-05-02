# Adversarial Audit

---

## Audit #2 — 2026-05-02

- Scope: full pipeline re-audit from first principles, with fresh code inspection of `core/automl.py`, `core/features.py`, `core/labeling.py`, `core/models.py`, `core/backtest.py`, `core/drift.py`, `core/data_quality.py`, `core/monitoring.py`, `core/regime.py`, `core/signal_decay.py`, `core/stat_tests.py`, and all example entrypoints.
- Industry baseline: Advances in Financial Machine Learning (López de Prado), purge-and-embargo time-series cross-validation, strict point-in-time feature availability, calibration before threshold selection, and conservative estimation errors for Kelly sizing.
- Bottom line: the architecture is more disciplined than the median retail system, but at least a dozen code-level flaws can silently inflate performance in a way that survives all shipped reporting outputs. A retail user acting on these results without understanding every failure mode listed below faces material risk of loss.

---

### 1. Data Integrity and Preprocessing

- **Critical — ADF stationarity screening uses the full dataset before any split.** `core/features.py::check_stationarity()` is called on the entire feature series, then the decision to apply fractional differentiation (or another transform from `DEFAULT_STATIONARITY_TRANSFORM_ORDER`) is made globally. The differentiation order `d` selected this way is implicitly informed by future out-of-sample observations. A different `d` might not pass ADF on the training window alone. The correct procedure is to fit `d` on the training fold and apply it to the test fold without re-estimating.

- **Critical — `DEFAULT_STATIONARITY_TRANSFORM_ORDER` is a sequential fallback that constitutes model selection on the full sample.** The code tries `log_diff → pct_change → diff → zscore → frac_diff` until ADF passes. Because this search is done on all data, the transform chosen is the one that happens to make the full series look stationary, including its OOS portion. This is a form of hyperparameter selection on the test set disguised as preprocessing.

- **High — Volatility series used for triple-barrier label construction is computed on the full series before labeling.** In `core/labeling.py::triple_barrier_labels()`, the caller passes a pre-computed `volatility` series. Nothing prevents the volatility estimate from being a rolling window that looks backward into what is, after splitting, the test set. Label quality claims about the training fold are therefore inflated whenever the caller computes volatility globally.

- **High — `check_data_quality()` defaults four anomaly types to `"flag"` rather than `"drop"` or `"block"`.** Return spikes, range spikes, quote-volume inconsistencies, and trade-count anomalies are flagged but not removed unless `block_on_quarantine=True` or the disposition action is explicitly overridden. These anomalies are not cosmetic in crypto: a return spike from a wash-trade burst or an exchange glitch directly creates a non-reproducible feature vector, biases volatility estimates, and potentially triggers a false triple-barrier label. The model trains on artefacts that will never appear in clean live data, and the run still reports clean data integrity.

- **High — Binance Vision ZIP downloads have no intra-bar path information.** The pipeline uses OHLCV kline data, so the only intrabar information for triple-barrier is high/low, which reflects the worst-case price excursion but not the sequence. When a bar has `high >= upper` and `low <= lower` simultaneously (tie), the code picks the winner based on `barrier_tie_break` (default: `sl`). In reality the actual exit depends on which barrier was hit first intrabar, which is unknown. Assuming the stop-loss hit first is conservative, but it is still an assumption not a measurement. Strategies optimized over `barrier_tie_break` in the search space will find whichever assumption is more flattering for that particular dataset and period.

- **Medium — Missing-candle gap policy defaults to `"warn"` in research mode.** `core/data.py` validates the `gap_policy` against `_VALID_GAP_POLICIES = {"fail", "warn", "flag", "drop_windows"}`. The research pipeline path commonly uses `"warn"`, which leaves gaps in the index. These silent gaps break rolling-window calculations (returns, volatilities, moving averages) by treating discontinuous bars as adjacent. A 4-hour gap in Binance data during a volatile period means the `close[T+1] / close[T] - 1` return appears at bar-level when it actually spans hours of unseen price movement. This biases volatility estimates downward and understates drawdown risk in backtests.

- **Medium — `CustomDataset.default_allow_exact_matches = False` at class definition, but `join_custom_dataset()` overrides it via a passed config.** Any caller setting `allow_exact_matches=True` or passing a dataset with exact-match semantics merges the external feature onto the same bar as its timestamp. For coarse external signals (daily fundamentals, macro indicators, sentiment feeds, on-chain metrics with delayed publication), the feature value known at decision time is the prior period's value, not the current bar's value. This is a standard but easily overlooked point-in-time violation.

---

### 2. Feature Engineering

- **Critical — Regime labels computed before the train/test split contaminate fold-level feature engineering.** `core/regime.py::build_instrument_regime_state()` runs on the full `base_data` frame passed from the pipeline. Regime labels for each bar are a function of all data including future bars. Any feature that uses regime membership as an input — regime-conditioned z-scores, regime interaction terms, regime-specific rolling statistics — therefore contains forward-looking regime state when evaluated inside a training fold.

- **High — Fractional differentiation weights are computed without an upper truncation on window size.** `core/features.py::fractional_diff()` iterates weights until `|w| < threshold` (default `1e-5`). For small `d` values (e.g. `d=0.2`, which the search space allows), this produces very long lag windows — potentially hundreds of bars. When the series is short (typical for regime-specific sub-samples or short backtest windows), the fraction of valid output rows after the burn-in period can be below 30%. Stationarity checks downstream may then be evaluating only the tail of the series, which is unrepresentative.

- **High — Cross-asset context features use Binance-sourced pair data, creating circular venue dependency.** Features prefixed `ctx_` are derived from other Binance symbols (e.g., ETHUSDT as a context for BTCUSDT). During exchange outages, both series are simultaneously missing. This creates correlated missingness that breaks the assumption that context features are independent signals. The model learns that when the context feature disappears, nothing interesting happens — the opposite of reality.

- **Medium — Rolling statistical features do not enforce minimum-period requirements beyond `min_periods` in pandas.** A `min_periods=1` rolling mean with a 20-bar window will produce a feature value from a single observation at the start of each fold. These edge-case values are qualitatively different from the steady-state rolling values. If training folds begin at different points, the distribution of these startup artifacts varies per fold, degrading CPCV comparability.

- **Medium — Feature lags from the search space (`"1,3,6"`, `"1,2,4,8"`, etc.) are not purged at fold boundaries.** When the training set ends at bar T and the gap is G bars, any lagged feature with lag L where L > G can carry test-set information into the last G - L training bars. The embargo removes G bars after training, but lagged features in the training data reaching back into what would become OOS territory for a shorter embargo are not explicitly checked. This is a subtle but real source of leakage when lags are longer than the embargo.

- **Low — z-score features (`_rolling_zscore`) are computed using a rolling window that makes them undefined for the first `window` bars but silently forward-fills or uses pandas NaN semantics.** These NaN values propagate into the model unless explicitly dropped. The behavior differs between the start of the full series and the start of each fold window, creating inconsistent input distributions between research and live inference.

---

### 3. AutoML Process

- **Critical — The search space includes label-generation hyperparameters (`pt_mult`, `sl_mult`, `max_holding`, `volatility_window`, `barrier_tie_break`).** Searching over these means each Optuna trial creates a different label set and evaluates its OOS performance. But the OOS set boundaries are fixed across trials, so the locked holdout (or validation fold) is being used to select label parameters. This is selection on the holdout set. The "holdout" is no longer unseen once the selection is complete. The `_classify_search_space` function tags these as `thesis_space` and restricts variation in trade-ready mode, but they remain searchable in the default and research profiles.

- **Critical — `validation_fraction` is part of the search space.** Searching over validation set size (`[0.15, 0.2, 0.25, 0.3]`) means the model can inflate its apparent OOS metric by finding the validation partition where noise happens to favour the objective. Smaller validation fractions concentrate the lucky noise periods; larger ones dilute it. Optuna's TPE sampler will converge on whichever size is most permissive for the given seed and data.

- **High — Post-selection inference (`core/stat_tests.py::compute_post_selection_inference()`) selects candidates by correlation with the top-ranked trial, not by independent significance.** The p-value deflation from multiple comparisons is not fully corrected even with the BH procedure when trials share the same dataset and overlapping feature sets. Two strategies that are both noise can have low correlation with each other while both having inflated Sharpe — they would both survive the deduplication filter and both be reported as significant.

- **High — Optuna's TPE sampler exploits seed-dependent randomness.** The default `seed=42` in the search configs and `automl_config.get("seed", 42)` in the runner produces a deterministic search trajectory. A researcher who observes good results and then retrains with the same seed on the same data is not producing an independent replication — they are producing an exact replay. The manifest hashes capture this, but nothing prevents a user from re-running on the same data window and treating the result as confirmation.

- **High — `meta_n_splits` (number of meta-labeling CV splits) is in the search space.** Varying meta-label split count per trial means different trials have different meta-model training data volumes. A trial with `meta_n_splits=2` has more meta-training data per split than one with `meta_n_splits=3`. The selection is therefore partly selecting for the split count that produces the most meta-model overfitting under the appearance of validation improvement.

- **Medium — The `calibration_params.c` for logistic calibration is searched jointly with model selection.** Calibration should be tuned on a held-out calibration set after model selection, not co-optimised with the model. Joint tuning on the same data that is used for model selection confounds probability calibration quality with model selection score, meaning the reported calibration quality is likely overstated.

- **Medium — `n_trials` defaults to small values in research mode** (readable from example configs as low as 5–20). Optuna's TPE sampler requires a warm-up phase (default 25 random trials before tree-structure exploitation begins). Running fewer than 25 trials means the entire search is random sampling with a false impression of intelligent exploration. Results from these short studies are equivalent to a random hyperparameter search, not AutoML.

---

### 4. Backtesting Methodology

- **Critical — Kelly sizing in `core/backtest.py::kelly_fraction()` uses in-sample win probability and average win/loss.** The Kelly fraction computed for a given model is calculated from the backtest results of that same model on the same data used to evaluate it. This is a direct form of look-ahead in position sizing: the bet size is calibrated to the realised profit distribution of the strategy. In live trading, the true win probability is unknown and must be estimated from the model's calibrated probability output, not from the backtest equity curve. Using IS kelly estimates inflates apparent risk-adjusted returns.

- **Critical — Same-bar execution fallback remains reachable in research mode.** `core/backtest.py::_resolve_execution_price_input()` returns `close` with a warning when `execution_prices` is omitted and the mode is not capital-facing. The warning is recorded in the summary dict but does not prevent the backtest from completing. Any research entrypoint that omits `execution_prices` silently assumes fills at the same bar's close price — a bar-close execution assumption that cannot be replicated live unless the trader places IOC market orders at the exact candle close time.

- **High — Signal delay (`signal_delay_bars=2`) is applied as a static bar offset, not as wall-clock latency.** On a 1h kline dataset, 2 bars = 2 hours of assumed delay. In a live system, the actual delay from signal generation to fill confirmation includes API response time, order routing, order book position, and partial fill risk. These are not constant multiples of the bar period. On 1h bars the 2-bar assumption is a generous overestimate of latency, making the results look more realistic than they are. On 1m or 5m bars the same assumption would be catastrophic.

- **High — `_infer_periods_per_year()` uses median inter-bar interval.** For crypto data with exchange outages or candle gaps, the median interval may differ significantly from the nominal interval. A 1h series with 200 missing candles has a median interval of 1h but a true annualisation factor closer to the nominal series. When gaps cluster (as they do during outages), the Sharpe ratio annualisation is wrong in exactly the periods that matter most to a risk assessment — periods of market stress.

- **High — VectorBT is an optional import with a graceful fallback to the pandas adapter.** When VectorBT is unavailable (consumer hardware without the library installed), the pandas adapter runs silently. The pandas adapter does not implement realistic order lifecycle, partial fills, or order-book-dependent execution. Results from the two engines are not equivalent, but the output format is identical. A user who ran research on the pandas adapter and drew conclusions cannot be confident those conclusions hold under the VectorBT simulation, let alone under NautilusTrader.

- **Medium — Stress scenario fill ratios use worst-case over configured scenarios, not worst-case over market regimes.** The `worst_fill_ratio` in `evaluate_stress_realism_gate()` reflects only the scenarios explicitly configured. A 35% worst-case fill ratio might occur across three synthetic downtime/halt scenarios, all of which were drawn from 2024 market conditions. A 2022-style 60% drawdown with sustained low-liquidity conditions is not represented unless the researcher explicitly configures it. The gate passes but the stress coverage is narrow.

- **Medium — The significance floor `min_observations=32` (smoke) or `64` (certification) is tested against the total trade count, not the count of directional trades.** If the model abstains frequently (which it should, per the design), 64 total outcomes may include 40 abstains and only 24 directional trades. The effective sample for measuring directional accuracy is 24, far below the 64 threshold that appeared to be met.

---

### 5. Evaluation Metrics

- **Critical — Sharpe ratio is computed on bar-level returns, not on trade-level returns.** The annualisation factor is derived from the bar interval. For a strategy with a 48-bar holding period that trades infrequently, bar-level return series contains mostly zero-return intervals between trades. Annualising a series dominated by zeros produces a highly stable-looking ratio that vastly overstates the actual information ratio of the signal. The correct denominator is the standard deviation of trade-level returns, or alternatively, portfolio-level returns with a realistic equity curve.

- **High — `profit_factor` in the backtest summary is not adjusted for trade count.** A strategy with 5 winning trades and 2 losing trades has a profit factor greater than 1.0, but 7 trades is insufficient to distinguish signal from noise. The metric is reported without minimum-trade-count context, creating an impression of robustness from a small sample.

- **High — Maximum drawdown is computed on the equity curve of the backtest, not on a mark-to-market basis.** For positions with long holding periods (24–48 bars), the unrealised drawdown during an open position is not captured until the position is closed. The reported max drawdown is therefore the realised drawdown between trade exits, not the actual peak-to-trough capital degradation the strategy experiences intraperiod.

- **Medium — `calmar_ratio` uses annualised return over max drawdown.** With few trades and short histories (3–6 months per the examples), max drawdown is strongly path-dependent and highly sensitive to a single losing trade. Calmar ratios above 2.0 on 3-month backtests with fewer than 50 trades are almost always an artefact of insufficient sample size rather than genuine risk-adjusted performance.

- **Medium — No out-of-sample Brier score decomposition (reliability, resolution, uncertainty).** The pipeline reports Brier score as a scalar. Decomposition into reliability (calibration) and resolution (sharpness) is necessary to distinguish a well-calibrated model from one that bets near 50% on everything. A model that always outputs 0.52 probability for any direction can have a respectable aggregate Brier score on a balanced dataset while providing zero tradeable edge.

---

### 6. Out-of-Sample and Robustness

- **Critical — CPCV validation produces overlapping train sets across splits.** In `core/models.py::cpcv_split()`, all combinations of `test_block_count` out of `n_blocks` blocks are tested. The same training data appears in multiple splits. The resulting OOS performance estimates are statistically dependent, and their average overstates the information content of a single walk-forward fold. The reported aggregate OOS score is not the same as the score of an independent hold-out test on unseen data.

- **High — The locked holdout is defined relative to the full dataset, but the full dataset is also used for regime detection, stationarity transform selection, and feature schema construction.** All three of these operations use future holdout information, meaning the holdout is not cleanly unseen: its distributional properties have already influenced the feature pipeline. The locked holdout test gives a score, but the score is for a model whose inputs were built with partial foreknowledge of the holdout period.

- **High — Hyperparameter sensitivity analysis is not conducted as part of the selection pipeline.** The pipeline reports the best hyperparameters and their OOS score, but does not report whether small perturbations to those parameters produce comparable scores. Fragile models (narrow performance peaks) are indistinguishable from robust ones in the current output. A model that performs well at `learning_rate=0.037` but collapses at `0.040` may be selected over a model with a flat performance surface.

- **Medium — Regime ablation reports can pass with zero completed ablations** (as identified in the previous audit). For a regime-aware model, this means the claim that "the model performs well across regimes" is made without evidence rather than being supported by ablation results. A crypto strategy can easily appear robust across regimes in a bull-only period, then fail completely when the market regime changes.

- **Medium — Sequential bootstrap for RandomForest is only activated when `mean_uniqueness < 0.90`.** This threshold is a configuration default that is not validated against the actual label concurrency of the training set. For a 24-bar holding period on 1h data with moderate ATR-based barriers, label overlap is near-certain, but the uniqueness may still exceed 0.90 on shorter runs. The model then trains with standard random bootstrap on correlated samples, inflating apparent cross-fold consistency.

---

### 7. Deployment Realism

- **Critical — ADWIN drift detection operates on performance residuals, not on feature distributions independently.** `core/drift.py::DriftMonitor` runs PSI and KS tests on features, but `performance_detector = ADWINDetector(delta=0.002)` is the primary trigger for retraining. ADWIN with `delta=0.002` requires a statistically significant change in the mean of the input stream before triggering. For infrequent trading (a few trades per day), hundreds of bars may pass before ADWIN accumulates enough samples to detect a regime change that a human looking at the price chart would notice immediately. The drift gate is lagged relative to the onset of regime change.

- **High — The retraining cooldown and minimum sample thresholds are configured as defaults but are not validated against the symbol's actual trade frequency.** `core/drift.py::DriftMonitor.config` has `cooldown_bars=500` and `min_samples=200`. For a low-frequency strategy on 1h BTCUSDT, 200 samples = 200 hours ≈ 8 days. For a high-frequency strategy on 5m data, 200 samples = 17 hours. The same numeric threshold has completely different implications depending on the trading interval, and no normalisation is applied.

- **High — No model warm-up or burn-in period after retraining.** When drift is detected and the model is retrained, the new model begins generating live signals on the next bar. The new model's probability outputs are not yet calibrated on the live distribution — they are calibrated on the training set, which may be from a different market regime. A badly calibrated model in a Kelly sizing system will size bets incorrectly from the first bar.

- **High — The monitoring `research` profile sets all limits to `None` or `np.inf`.** `core/monitoring.py::_POLICY_PROFILES["research"]` is an empty dict, inheriting all `_DEFAULT_POLICY` infinity values. This means that in research mode, schema drift, slippage deterioration, inference latency, and signal half-life decay generate no alerts and raise no gates. A researcher who promotes a model without switching to `local_certification` or `trade_ready` monitoring will never see a monitoring failure, regardless of how degraded the live data is.

- **Medium — Inference latency is not modelled on consumer hardware.** The monitoring policy requires p95 latency ≤ 250ms in trade-ready mode. This is measured in the research environment. On a consumer machine running Python with scikit-learn GBM (which uses non-parallelised C++ tree traversal under sklearn's default), feature computation for 60–100 features on a single bar takes microseconds; the bottleneck is Binance REST API latency (~50–200ms per request from most residential connections). The 250ms gate is therefore almost certainly satisfied without any meaningful inference optimisation, creating false confidence in the latency budget.

- **Medium — Champion/challenger model comparison uses the challenger's own backtest period as evidence.** `core/registry/__init__.py::evaluate_challenger_promotion()` compares challenger vs. champion using the challenger's submitted evidence. There is no requirement that the same OOS period is evaluated for both models. A challenger trained on a recent favourable period can displace a champion trained on a longer, harder period, with the comparison appearing fair because both exceeded their respective thresholds.

- **Low — `skops` serialisation is attempted with a try/except fallback to pickle.** `core/models.py` imports `skops_dump`, `skops_get_untrusted_types`, `skops_load` with `except ImportError: skops_* = None`. When skops is unavailable, the system falls back to pickle, which is executable on load and a known deserialization vulnerability. For a research system this is acceptable, but any model artifact loaded from an untrusted source would execute arbitrary code. On consumer hardware where Binance API keys may be stored in environment variables, this is a genuine risk.

---

### 8. Structural Failure Modes — How This Looks Profitable But Fails Live

- **Label-parameter leakage masquerading as model quality.** The search space includes `pt_mult`, `sl_mult`, and `max_holding`. Optuna finds the label definition where the model fits the training data best. When these labels are evaluated on the OOS set, the OOS score reflects the best label definition that the training-set model happened to prefer, not a genuinely forward-looking prediction. Sharpe ratios of 3–8 are entirely consistent with this failure mode.

- **Stationarity-transform selection on the full dataset inflating feature predictability.** The transform order `log_diff → pct_change → diff → zscore → frac_diff` searches for stationarity using future data. The selected transform is the one that makes the full-sample feature distribution look most stationary, which in crypto often correlates with low-volatility periods. The model trained on this feature behaves as if it has forward-looking smoothing applied to its inputs.

- **Regime labels from full-data KMeans/HMM providing implicit future context.** Regime features (`build_instrument_regime_state`) computed on the full dataset assign regime labels that reflect the eventual long-run regime membership of each bar. During live trading, the regime at bar T is unknown until enough future data arrives to confirm the classification. The model trained on full-sample regimes is implicitly using a signal that does not exist in real time.

- **Short backtest windows producing inflated Calmar and Sharpe from low drawdown.** 3–6 month backtests have too few drawdown events to estimate max drawdown reliably. A strategy that has never experienced a drawdown of more than 5% in 6 months of bull-market data will produce a Calmar ratio of 5+ with 25% annualised return, appearing elite by any institutional benchmark. Live trading will eventually encounter a market regime not represented in the 6-month window.

- **Post-selection inference decorating noise as signal.** With 20–50 Optuna trials on a 2,000-bar dataset, the best trial by Sharpe has significant positive bias from maximisation over noise. The bootstrap resampling p-value for the selected model is computed against the empirical null of reshuffled returns, but the null is computed from the same short data window, which has fat tails and autocorrelated returns. The effective sample for the null distribution is far smaller than the nominal observation count.

- **Zero-fill missing futures funding events turning carry-negative strategies into apparent carry neutrals.** A strategy that is systematically short during high-funding periods looks breakeven in research (funding set to zero) and is actually negative in live trading. The sign of the carry effect is always in the direction of the majority view (longs pay shorts), and high-funding periods coincide exactly with the trend-following and momentum signals that crypto ML models tend to produce.

- **Kelly sizing from IS win probability creating bet-size inflation.** A model that wins 55% of IS trades uses that figure to compute a positive Kelly fraction. The true OOS win rate for a marginally-above-chance classifier is likely lower, possibly below 50% after costs. With half-Kelly sizing, the model bets 2.5% of equity per trade under the IS assumption. If the true win rate is 48% after fees, this produces a systematic negative-expectation bet at 2.5% stake per trade, with leverage-like compounding losses.

---

### 9. Retail-Specific Risk Amplifiers

- **Consumer hardware introduces non-determinism in parallel feature computation.** `RandomForestClassifier` uses `n_jobs=-1` by default, which varies job scheduling across different CPU core counts. Results on a developer's 16-core machine will differ from results on a 4-core machine. This means the example run results in AGENTS.md are not reproducible on arbitrary consumer hardware without explicitly setting `n_jobs=1`.

- **No rate-limit-aware retraining scheduler.** The system has no concept of Binance API rate limit weight accumulation during the backfill phase of a retrain trigger. A drift event mid-session may trigger a full data refetch for the training window, which on 1h data for 6 months means ~4,380 kline fetches. At Binance's 1,000-weight-per-minute limit, this saturates the API budget for the session. No circuit breaker or backfill batching is implemented at the drift-retraining orchestration layer.

- **No Binance-specific minimum notional guard in paper research.** `core/execution/costs.py` may compute a position size from Kelly fraction × equity, but there is no validation that the resulting order size meets Binance's symbol-level `minNotional` filter in the paper research path. An order that passes the research check but is below `minNotional` will be silently rejected by the exchange in live trading, producing a fill rate well below the assumed 100% and invalidating every fill-dependent metric.

---

## Audit #1 — 2026-04-30
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