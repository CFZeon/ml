# IMPROVEMENTS — Adversarial Audit & Industry Gap Analysis

Sorted by priority: **critical** flaws that invalidate results first, then **high** flaws that inflate performance, then **medium** gaps that limit deployability, then **low** improvements for polish.

---

## CRITICAL — Results May Be Invalid

### C1. No Multiple-Testing Correction (Deflated Sharpe Ratio / PBO) [implemented]

- **What**: The AutoML study runs N trials (default 25) via Optuna, selects the best trial by objective score, and reports its backtest Sharpe / net profit as if it were a single unbiased estimate. No correction is applied for the number of configurations tested.
- **Industry standard**: De Prado's *Deflated Sharpe Ratio* (DSR) and Bailey & López de Prado's *Probability of Backtest Overfitting* (PBO) are standard at quantitative firms. DSR adjusts the Sharpe ratio for the number of trials, skewness, and kurtosis. PBO uses combinatorial symmetric cross-validation to estimate the probability that the best in-sample strategy underperforms the median out-of-sample.
- **Why it matters**: With 25+ trials, even noise-fitting configurations will occasionally produce attractive backtests. Reporting the raw best-trial Sharpe without DSR is the primary mechanism by which this pipeline can appear profitable while being pure noise. The locked holdout partially mitigates this, but a single holdout period is one draw and does not produce a distribution of OOS performance.
- **How firms do it**: Two Sigma, AQR, and most systematic macro shops apply DSR as a minimum filter. More rigorous shops estimate PBO via CPCV. Citadel's research framework reportedly imposes Bonferroni-like trial penalties before any strategy reaches paper trading.
- **Gap in repo**: `compute_objective_value()` in `core/automl.py` returns a raw score. No DSR, no PBO, no trial-count penalty. The `locked_holdout` is a single chronological block—not CPCV.
- **File**: [core/automl.py](core/automl.py)

### C2. No Combinatorial Purged Cross-Validation (CPCV) pimplemented[]

- **What**: Walk-forward CV splits data into sequential folds. This produces exactly N test-set observations per fold, and the test sets are non-overlapping. CPCV generates all $\binom{N}{N/2}$ (or a tractable subset) combinations of train/test paths, with purging and embargo applied to each, producing a distribution of OOS paths rather than a single equity curve per fold. 
- **Industry standard**: CPCV is the validation method recommended in *Advances in Financial Machine Learning* (AFML Ch. 12). It enables computing PBO and provides a statistically meaningful distribution of strategy performance. Standard walk-forward CV with 3 folds provides 3 correlated performance estimates—insufficient for statistical inference.
- **Why it matters**: With only 3 walk-forward folds, the pipeline cannot distinguish between a strategy with genuine edge and one that was lucky on 3 particular test windows. CPCV would expose whether performance is stable across many possible train/test partitions.
- **Gap in repo**: `walk_forward_split()` in `core/models.py` implements standard sequential splits only. No CPCV implementation exists.
- **File**: [core/models.py](core/models.py)

### C3. Slippage Model Is Naive (Flat Rate, Not Volume/Liquidity-Dependent)

- **What**: `run_backtest()` applies `slippage_rate` as a flat percentage of turnover, independent of trade size, order book depth, or time of day. The default is 0.0 (zero slippage).
- **Industry standard**: Production crypto systems model slippage as a function of trade size relative to available liquidity. The standard approach is a square-root market impact model: $\text{impact} \propto \sigma \cdot \sqrt{V/\text{ADV}}$ where $V$ is trade volume and ADV is average daily volume. Alameda (pre-collapse), Jump Crypto, and Wintermute all used volume-profile-aware slippage models in their backtests.
- **Why it matters**: For any strategy that trades meaningful size or trades during low-volume hours, flat-rate slippage dramatically understates execution cost. Crypto order books are thin—a $50K market order on BTCUSDT can move price 5-20 bps during Asia hours. The pipeline's 0 bps default makes every backtest look better than reality.
- **Gap in repo**: `_run_pandas_backtest()` and `_run_vectorbt_backtest()` in `core/backtest.py` use `slippage_rate` as a scalar multiplied by turnover. No volume data is consumed. No time-of-day adjustment.
- **File**: [core/backtest.py](core/backtest.py)

### C4. No Statistical Significance Testing on Backtest Results [implemented]

- **What**: The pipeline reports metrics (Sharpe, net profit, win rate) as point estimates. No confidence intervals, no bootstrap distributions, no hypothesis tests.
- **Industry standard**: At minimum, block-bootstrap the equity curve to produce confidence intervals on Sharpe. Better: use the stationary bootstrap (Politis & Romano) which preserves serial correlation. Report p-values for Sharpe > 0 and for Sharpe > benchmark. AQR and Man Group publish bootstrapped confidence intervals as standard practice.
- **Why it matters**: A Sharpe of 1.5 computed from 3 months of hourly data has wide confidence intervals. Without them, there is no way to distinguish signal from noise. The reported Sharpe 8.48 from the initial test run (43 trades, 3 months) almost certainly has a 95% CI that includes 0.
- **Gap in repo**: `_summarize_backtest()` returns scalar metrics only. No bootstrapping, no CI, no hypothesis testing.
- **File**: [core/backtest.py](core/backtest.py)

---

## HIGH — Performance Likely Inflated

### H1. Regime Detection Leakage Through KMeans Scaling [implemented]

- **What changed**: The public regime API no longer exposes KMeans. `detect_regime()` now accepts only `method="hmm"` and `method="explicit"`, and fold-local regime frames are still rebuilt from buffered fold windows before fitting on the training slice.
- **Industry standard**: Firms that use regime conditioning typically prefer explicitly defined thresholds or sequence-aware latent-state models. The repo now defaults to Gaussian HMM with stable norm-sorted state ordering so regime IDs retain consistent semantics across walk-forward folds.
- **Why it mattered**: Arbitrary KMeans cluster IDs across folds added noisy, inconsistent conditioning even when the scaler and centroids were fit on train-only data.
- **Status in repo**: Closed. HMM is the default, explicit regimes remain supported, and unsupported legacy methods fail fast instead of silently reintroducing unstable clustering semantics.
- **Files**: [core/models.py](core/models.py), [core/pipeline.py](core/pipeline.py), [tests/test_regime_leakage_controls.py](tests/test_regime_leakage_controls.py)

### H2. Fractional Differentiation Applied Globally, Not Per-Fold

- **What**: `fractional_diff()` is applied in `build_feature_set()` during the `build_features` step, which runs once on the full dataset before walk-forward splits. The differentiation weights are computed from the global series. While the fold-local stationarity screening can re-transform features, the base `close_fracdiff` column is computed from all data including future test bars.
- **Industry standard**: Any transformation that uses a window of data (rolling stats, frac diff) must be computed purely from data available at prediction time. AFML recommends computing frac-diff weights once (they depend only on `d` and threshold) but applying them in an expanding window that never looks ahead. The weight vector is deterministic given `d`, so the only leakage risk is from the series values themselves in an expanding application.
- **Why it matters**: For `fractional_diff()` with the default threshold=1e-5 and d=0.4, the weight window is ~30 values. This means the frac-diff output at bar T uses bars T-30 to T. If bar T is in the training set and bar T+1 is in the test set, there is no leakage. However, if stationarity *screening* uses the test-period distribution to decide whether to apply frac-diff, that decision is leaked.
- **Nuance**: The current fold-local screening mitigates this partially. But `build_features` still computes the frac-diff column globally. If the screening decides "this feature doesn't need frac-diff" based on the full distribution (which includes the test period), that decision is contaminated.
- **File**: [core/features.py](core/features.py)

### H3. Feature Selection MI Scores Computed on Training Data Only—But Column Retention Propagates Across Folds [implemented]

- **What**: MI-based feature selection runs inside each fold on training data only (good). However, the `last_selected_columns` variable carries the final fold's column selection into the `SignalsStep` fallback path. If the pipeline re-generates signals outside the walk-forward loop, it uses the last fold's feature set—which was selected based on the last fold's training data.
- **Industry standard**: Feature selection should be strictly fold-local, and any inference after training should use the final fold's feature set *only* on data that comes after the final fold's training window.
- **Why it matters**: Minor in the walk-forward path (since OOS predictions are already accumulated per-fold). But the `SignalsStep` fallback path (when `training["oos_continuous_signals"]` is None) applies the last fold's model and feature selection to the *entire* aligned dataset, which includes training periods from earlier folds.
- **File**: [core/pipeline.py](core/pipeline.py#L1650)

### H4. AutoML Objective Optimizes Training Metrics, Not True OOS Performance

- **What**: `compute_objective_value()` uses `training["avg_directional_accuracy"]` and `training["avg_log_loss"]`—which are walk-forward OOS metrics within the search window. However, the search window itself is sliced from the full data, and Optuna sees these OOS metrics across all trials. The best trial is the one that achieved the highest OOS metric *within the search window*, but this is still in-sample with respect to Optuna's selection.
- **Industry standard**: The outer selection loop (Optuna) treats the inner walk-forward OOS metrics as "training" from its perspective. The only truly OOS metric is from the locked holdout. But the locked holdout is not used for trial selection—it's evaluated only on the best trial after selection. This means the trial selection itself is not penalized for multiple testing.
- **Why it matters**: In the default `accuracy_first` objective, the composite score includes log loss, Brier score, and calibration error with configurable weights. Optimizing this composite across 25 trials is equivalent to running 25 backtests and picking the best—classic backtest overfitting.
- **File**: [core/automl.py](core/automl.py)

### H5. Kelly Sizing Uses In-Sample Estimates of avg_win / avg_loss

- **What**: Per the repo memory, OOS-estimated avg_win/avg_loss were added. However, `_estimate_trade_outcome_stats()` pools all OOS trade outcomes after the walk-forward loop. The problem: these outcomes come from different folds with different models. The avg_win/avg_loss from fold 0 has no bearing on the model used in fold 2. The pipeline uses the pooled estimate for sizing in the `SignalsStep`, but during the walk-forward loop itself, each fold uses `fold_avg_win` / `fold_avg_loss` which may come from the validation set (which is part of the training window).
- **Industry standard**: Kelly sizing should use strictly out-of-sample win/loss estimates from a held-out period that the model has never seen during training or validation. Better yet, use a shrinkage estimator (half-Kelly or fractional Kelly with conservative estimates).
- **Why it matters**: Overestimating avg_win or underestimating avg_loss inflates Kelly fractions, leading to oversized positions that blow up in production. The current default `fraction=0.5` (half-Kelly) provides some protection, but the underlying estimates are still contaminated.
- **File**: [core/pipeline.py](core/pipeline.py)

### H6. No Transaction Cost Sensitivity Analysis

- **What**: The pipeline uses a single fee_rate (default 0.001 = 10 bps) and does not test whether strategy profitability survives at 2x or 3x the assumed cost.
- **Industry standard**: Robust strategy evaluation requires sweeping costs from 0 to 3x the base assumption and reporting the breakeven cost level. Jane Street and Optiver routinely evaluate strategies at multiple cost scenarios before allocation.
- **Why it matters**: Many strategies that look profitable at 10 bps round-trip become unprofitable at 15-20 bps. Without a cost sensitivity sweep, there is no way to assess the strategy's margin of safety.
- **Gap in repo**: No cost sweep functionality exists. 
- **File**: [core/backtest.py](core/backtest.py)

---

## MEDIUM — Limits Deployability and Robustness

### M1. No ADWIN or Any Drift Detection Implementation

- **What**: AGENTS.md specifies "ADWIN or an equivalent detector" for drift-triggered retraining. No drift detection is implemented.
- **Industry standard**: Evidently AI, NannyML, and Alibi Detect provide production-grade drift detection (PSI, KS-test, ADWIN, CUSUM). Two Sigma and Citadel reportedly use feature distribution monitoring with automatic alerts. River (online ML library) provides streaming ADWIN.
- **Why it matters**: Without drift detection, the system has no mechanism to trigger retraining when market conditions change. This is the gap between a research tool and a production system. In crypto, regime shifts can happen within hours (exchange delistings, regulatory announcements, liquidity crises).
- **Gap in repo**: No implementation of ADWIN, CUSUM, PSI, or any drift detector.

### M2. No Model Registry or Artifact Versioning

- **What**: AGENTS.md specifies "model registry and artifact store." Models are stored via `pickle` (`save_model` / `load_model`). No versioning, no metadata tracking, no rollback support.
- **Industry standard**: MLflow, Weights & Biases, or DVC are standard for experiment tracking and model registry. Firms maintain champion/challenger model pairs with automatic rollback if the new model underperforms.
- **Why it matters**: Without a registry, there is no way to reproduce past experiments, compare model versions, or roll back to a previous model when a retrained model degrades. This is a hard requirement for production deployment.
- **Gap in repo**: `save_model()` and `load_model()` use raw pickle. No metadata, no versioning.
- **File**: [core/models.py](core/models.py)

### M3. No Structural Break Detection (CUSUM / SADF / GSADF)

- **What**: AGENTS.md lists structural break tests as "optional complement to ADWIN." None are implemented.
- **Industry standard**: SADF and GSADF (sup ADF and generalized sup ADF) are used to detect asset price bubbles—critical in crypto where bubble/crash dynamics are common. CUSUM is used for mean-shift detection in feature distributions. De Prado (AFML Ch. 17) uses SADF to detect explosive behavior in log prices and filter false signals during bubble regimes.
- **Why it matters**: A directional model trained on trending data will confidently predict trend continuation during a bubble—right until the crash. Without structural break detection, the system has no early warning.
- **Gap in repo**: No SADF, GSADF, or CUSUM implementation.

### M4. Signal Half-Life / Alpha Decay Not Tracked

- **What**: AGENTS.md requires "Track signal half-life and alpha decay per signal type as a first-class metric." This is not implemented.
- **Industry standard**: Compute the autocorrelation of signal returns at various lags. The half-life is the lag at which autocorrelation drops to 0.5. Alternatively, regress cumulative signal return on lag and estimate the decay constant. Firms like WorldQuant and Millennium track alpha decay per signal as a core portfolio construction input.
- **Why it matters**: A signal with a 2-bar half-life needs immediate execution (market orders, co-located infrastructure). A signal with a 50-bar half-life can afford limit orders and patient execution. Without measuring this, the pipeline cannot match execution strategy to signal characteristics.
- **Gap in repo**: No half-life or decay estimation anywhere in the codebase.

### M5. No Walk-Forward Stability Analysis (Fold Variance)

- **What**: The pipeline computes `avg_accuracy`, `avg_f1_macro`, etc. across folds but does not report variance, standard deviation, min/max range, or any measure of fold-to-fold stability.
- **Industry standard**: Report both mean and standard deviation of OOS metrics across folds. Flag strategies where the coefficient of variation exceeds a threshold (e.g., >0.5). AQR's research framework computes per-decile Sharpe across rolling windows and rejects strategies with high inter-decile variance.
- **Why it matters**: A strategy with avg Sharpe 2.0 but fold Sharpes of [5.0, 1.5, -0.5] is very different from one with [2.2, 1.8, 2.0]. The current pipeline cannot distinguish these cases.
- **File**: [core/pipeline.py](core/pipeline.py)

### M6. Pandas Backtest Engine Does Not Model Discrete Trades

- **What**: The pandas fallback engine applies `position * returns` at every bar, which is a continuous rebalancing assumption. This is correct for target-percentage position sizing but does not model the discrete entry/exit events that the triple-barrier labeling implies.
- **Industry standard**: When labels define discrete trade events (enter at bar T, exit at barrier hit), the backtest should model those discrete trades. VectorBT handles this via `Portfolio.from_orders()`, but the pandas fallback applies positions as continuous weights.
- **Why it matters**: Continuous rebalancing at every bar incurs far more turnover than the discrete events the model actually predicts. The trade count and win rate metrics from `_build_trade_ledger()` are derived from sign changes in the position series, not from the model's actual predicted entry/exit events. This disconnect inflates turnover costs and understates per-trade profitability.
- **File**: [core/backtest.py](core/backtest.py)

### M7. No Liquidity Filtering or Symbol Eligibility Screen

- **What**: The pipeline accepts any symbol but does not filter based on minimum volume, spread, or market cap. Binance has hundreds of symbols, many with extremely thin liquidity.
- **Industry standard**: Filter symbols by minimum ADV (average daily volume), maximum bid-ask spread, and minimum market cap before including them in the research universe. Alameda's research pipeline reportedly filtered symbols below $1M 24h volume.
- **Why it matters**: Training a model on SHIB/USDT or a newly listed micro-cap will produce signals that cannot be executed at backtest-assumed prices. The model may appear profitable on illiquid symbols precisely because it is fitting to noise in thin order books.
- **Gap in repo**: No liquidity filter exists.

### M8. No Benchmark Comparison

- **What**: Strategy performance is reported in absolute terms (Sharpe, return, drawdown). No comparison to buy-and-hold, equal-weight, or a simple momentum benchmark.
- **Industry standard**: Always report alpha relative to a benchmark. At minimum, compare to buy-and-hold of the same asset. Better: compare to a simple trend-following model (e.g., 50/200 SMA crossover). Best: report information ratio and tracking error versus a crypto benchmark index.
- **Why it matters**: In a bull market, any long-biased strategy will show positive returns. Without a benchmark, there's no way to determine whether the model adds value over passive exposure.
- **File**: [core/backtest.py](core/backtest.py)

### M9. No Embargo Period Between Train and Test

- **What**: `walk_forward_split()` has a `gap` parameter (default 0), but this is a simple row-level gap. The `_purge_overlapping_training_rows()` function removes training rows whose label end-times overlap the test boundary. However, there is no formal embargo period that excludes a fixed number of bars after the purge boundary to account for serial correlation in returns.
- **Industry standard**: AFML Ch. 7 recommends an embargo period of at least `max_holding` bars after the purge boundary. This accounts for the fact that even after purging overlapping labels, the return process has serial correlation that can leak information.
- **Why it matters**: With triple-barrier labels using `max_holding=24`, the purge removes labels whose events overlap the test boundary, but returns at the test boundary are still correlated with the last training returns. A 24-bar embargo would eliminate this.
- **Gap in repo**: `gap` exists in `walk_forward_split()` but is not automatically set to `max_holding`. The default is 0.
- **File**: [core/models.py](core/models.py)

### M10. Sequential Bootstrapping Not Used in Ensemble Training

- **What**: AGENTS.md requires "sequential bootstrapping for any ensemble model (random forest, bagged classifiers)." The pipeline includes `sequential_bootstrap()` in `core/labeling.py` but does not wire it into the training of `RandomForestClassifier`.
- **Industry standard**: Standard random bootstrapping in RF with overlapping triple-barrier labels inflates apparent accuracy because bootstrap samples contain redundant information from concurrent labels. Sequential bootstrapping (AFML Ch. 4) draws samples with probability proportional to their uniqueness, reducing this inflation.
- **Why it matters**: When using RF (which the pipeline supports), the OOB accuracy and feature importance estimates are inflated by redundant samples. This makes the model appear more powerful than it is.
- **Gap in repo**: `sequential_bootstrap()` exists in `core/labeling.py` but is never called from `train_model()` or `TrainModelsStep`. RF is trained with standard sklearn bootstrapping.
- **File**: [core/labeling.py](core/labeling.py), [core/models.py](core/models.py)

---

## LOW — Polish and Completeness

### L1. No Position Limit / Exposure Cap Enforcement

- **What**: AGENTS.md requires "max position, max leverage, max loss, exposure caps, kill switch." The backtest clips positions to `[-leverage, leverage]` but does not enforce:
  - Maximum drawdown kill switch (halt trading after X% drawdown)
  - Per-bar exposure limits
  - Correlation-based exposure caps (if trading multiple symbols)
- **Industry standard**: Risk overlays are always applied outside the model. Bridgewater and Renaissance Technologies reportedly have multiple independent risk systems that can override model signals.
- **File**: [core/backtest.py](core/backtest.py)

### L2. No Data Freshness / Staleness Detection

- **What**: No mechanism to detect whether the data pipeline has stopped updating. In live deployment, if the data feed stalls, the model will continue generating signals based on stale features.
- **Industry standard**: Implement a data freshness check: if the latest bar is more than 2x the expected interval ago, halt signal generation and alert. This is a hard requirement for any production trading system.
- **Gap in repo**: No freshness check.

### L3. Missing Candle / Gap Detection Is Passive

- **What**: `_detect_and_report_gaps()` in `core/data.py` reports missing candles in the output metadata but does not fill gaps or adjust features. Features like rolling z-scores, lags, and returns will be silently wrong when gaps are present.
- **Industry standard**: Either forward-fill missing candles explicitly (noting this in metadata), drop the affected window, or flag features as unreliable during gaps.
- **File**: [core/data.py](core/data.py)

### L4. Indicator Library Is Minimal

- **What**: Five indicators (RSI, MACD, Bollinger Bands, ATR, FVG). No volume profile, no order flow indicators, no market microstructure indicators.
- **Industry standard**: Production crypto systems typically include: VWAP, volume-weighted price momentum, funding rate momentum, open interest change rate, liquidation cascade indicators, order book imbalance (if L2 data available), and correlation regime indicators.
- **Gap in repo**: Only classical TA indicators plus FVG. No crypto-native indicators that exploit the unique data available from Binance (funding rates, open interest, liquidation data).
- **File**: [core/indicators/](core/indicators/)

### L5. No Hyperparameter Sensitivity Analysis

- **What**: The pipeline selects a single best configuration from AutoML and reports its performance. No analysis of how sensitive that performance is to small perturbations of the selected hyperparameters.
- **Industry standard**: Plot performance as a function of each hyperparameter (partial dependence plots). Flag parameters where small changes cause large performance swings—these are signs of overfitting to noise. Optuna provides `plot_param_importances()` but it is not called.
- **Gap in repo**: No sensitivity analysis, no partial dependence plots, no Optuna visualization integration.

### L6. No Per-Regime Performance Decomposition

- **What**: Regime detection exists, but backtest performance is not broken down by regime. There is no way to see "this strategy makes money in low-volatility regimes but loses in high-volatility regimes."
- **Industry standard**: Report Sharpe, win rate, and drawdown per regime. This is critical for understanding whether the strategy has genuine edge or is simply long volatility.
- **Gap in repo**: No regime-conditioned performance reporting.

### L7. Pickle-Based Artifact Store Is a Security Risk

- **What**: `save_model()` uses `pickle.dump()` and `load_model()` uses `pickle.load()`. Pickle deserialization can execute arbitrary code.
- **Industry standard**: Use `joblib` (which has the same risk but is standard for sklearn models), or better, serialize model parameters to JSON/ONNX and reconstruct. For sklearn, `skops.io` provides safe serialization.
- **File**: [core/models.py](core/models.py)

### L8. No Unit Test Coverage for Core Statistical Functions

- **What**: Tests exist for integration scenarios but not for unit-level correctness of critical functions: `fractional_diff()`, `triple_barrier_labels()`, `sample_weights_by_uniqueness()`, `kelly_fraction()`, `walk_forward_split()`.
- **Industry standard**: Every numerical function should have unit tests with known inputs/outputs. Especially: frac-diff should be verified against a reference implementation (e.g., mlfinlab or fracdiff package). Triple-barrier labels should be verified against hand-computed examples.
- **File**: [tests/](tests/)

### L9. No Feature Stability Across Retrains

- **What**: AGENTS.md requires "Keep feature schemas stable across retrains or version them when they change." The pipeline tracks `schema_version` as a string tag but does not enforce that the feature set at inference time matches the feature set at training time.
- **Industry standard**: Store the exact feature column list with each model artifact. At inference time, validate that the incoming feature set matches. Reject or adapt if columns are missing or extra.
- **Gap in repo**: Feature schema is a config tag, not a validated contract.

### L10. No Portfolio-Level Risk Controls

- **What**: The pipeline operates one model per symbol with no cross-symbol coordination. If running multiple symbols, there is no mechanism to limit total portfolio exposure, correlation-adjusted risk, or sector concentration.
- **Industry standard**: Even with per-symbol models, a portfolio risk overlay should limit total gross exposure, net exposure, and drawdown across all positions. This is explicitly noted as an open question in AGENTS.md but has no implementation.
- **Gap in repo**: No portfolio-level aggregation or risk limits.

---

## Summary of Gaps vs. Industry Practice

| Capability | This Repo | Industry Standard | Gap Severity |
|---|---|---|---|
| Multiple testing correction | None | DSR / PBO mandatory | **Critical** |
| Cross-validation | Walk-forward (3 folds) | CPCV with purging + embargo | **Critical** |
| Slippage modeling | Flat rate (default 0) | Volume-dependent impact model | **Critical** |
| Statistical significance | Point estimates only | Bootstrap CI + hypothesis tests | **Critical** |
| Drift detection | Not implemented | ADWIN / PSI / KS monitoring | Medium |
| Model registry | Pickle + manual | MLflow / W&B with versioning | Medium |
| Structural breaks | Not implemented | SADF / GSADF / CUSUM | Medium |
| Alpha decay tracking | Not implemented | Autocorrelation decay estimation | Medium |
| Benchmark comparison | Not implemented | Buy-and-hold + trend baseline | Medium |
| Position limits / kill switch | Basic clipping only | Multi-layer risk overlay | Low |
| Indicator library | 5 classical TA | TA + crypto-native (funding, OI, liquidation) | Low |
| Feature stability | String tag only | Schema validation at inference | Low |
| Sequential bootstrapping | Implemented, not wired | Used in all RF/bagging training | Medium |

---

## Recommendations for Validation Before Trusting Any Result

1. **Immediately**: Add DSR calculation to AutoML summary. If DSR < 1.0, the strategy should not proceed.
2. **Short term**: Implement cost sensitivity sweep (0.5x to 3x base cost). Report breakeven cost level.
3. **Short term**: Add bootstrap confidence intervals on Sharpe ratio. Reject strategies where 95% CI lower bound < 0.
4. **Medium term**: Implement CPCV and compute PBO. Reject strategies where PBO > 0.5.
5. **Medium term**: Replace flat slippage with a volume-dependent impact model using `volume` data already available in the OHLCV dataset.
6. **Medium term**: Wire `sequential_bootstrap()` into RF training. Set `gap` = `max_holding` by default.
7. **Longer term**: Integrate MLflow for experiment tracking and model registry. Replace pickle with safe serialization.
