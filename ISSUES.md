# ISSUES

## Adversarial Audit Findings

Scope: repository state as of 2026-04-25.

Institutional baseline used for cross-reference:

- [Federal Reserve SR 11-7 / OCC 2011-12a](https://www.federalreserve.gov/boarddocs/srletters/2011/sr1107.htm): conceptual soundness, effective challenge, outcomes analysis, ongoing monitoring.
- [ESMA MiFID II Article 17](https://www.esma.europa.eu/publications-and-data/interactive-single-rulebook/mifid-ii/article-17-algorithmic-trading): tested and resilient algo systems, thresholds and limits, monitoring, business continuity, records.
- [NIST AI RMF 1.0](https://www.nist.gov/itl/ai-risk-management-framework): validity, reliability, data quality, governance, measurement, monitoring.
- [Binance Spot Filters](https://developers.binance.com/docs/binance-spot-api-docs/filters), [Klines](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints#klinecandlestick-data), [Funding History](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History): exchange constraints, UTC timestamp semantics, discrete funding events.

The repo is stronger than a toy trading stack: CPCV, walk-forward validation, purging/embargo logic, lookahead provocation, post-selection inference, and promotion gates all exist. The problem is that the default retail path still falls short of institutional-grade evidence because several critical controls are optional, advisory, or disabled in the common fallback path.

### Highest-Impact Flaws

- Critical | `example_trade_ready_automl.py` disables `locked_holdout_enabled`, `selection_policy`, DSR, PBO, and post-selection inference when Nautilus is unavailable. A profitable fallback run is not weaker certification; it is a different experiment with materially weaker protection against noise-fitting. Why it inflates performance: the strongest effective-challenge layer is removed exactly on the path most retail users will run. [SR 11-7, ESMA Art. 17]

- Critical | `core/backtest.py` is still a bar surrogate by default. It rejects passive and limit orders and explicitly reports `bar_surrogate_only`, `no_queue_position_model`, `no_event_driven_ack_latency`, and `no_order_book_matching_engine`. Why it inflates performance: bar volume is treated as executable liquidity and any edge requiring maker rebates, spread capture, queue priority, or intrabar matching is untested. [ESMA Art. 17, Binance]

- Critical | `core/pipeline.py` defaults `funding_missing_policy` to `zero_fill`, and aligned funding is reindexed then `fillna(0.0)`. Missing funding observations become zero carry. Why it inflates performance: futures Sharpe, Calmar, and Kelly sizing improve mechanically whenever the funding feed is incomplete. [Binance Funding History, SR 11-7]

- High | `core/monitoring.py` leaves data-lag, L2-age, slippage-drift, inference-latency, queue-backlog, and signal-decay thresholds at `None`, `False`, or `inf` unless the operator overrides them. Why it gives false comfort: the system can emit live-risk telemetry without making that telemetry economically binding for selection, promotion, or shutdown. [SR 11-7, ESMA Art. 17, NIST AI RMF]

### Data Integrity & Preprocessing

- High | `core/data.py` defaults market-data gaps to `gap_policy="warn"`. In a 24/7 crypto market, missing candles are often exchange outages, symbol halts, or data transport failures, not harmless nuisance rows. Why it invalidates conclusions: volatility, barrier timing, annualization, and slippage assumptions are biased exactly when market quality is worst. [Binance Klines, NIST AI RMF]

- High | `check_data_quality()` in `core/data_quality.py` defaults `block_on_quarantine=False`, and return spikes, range spikes, quote-volume mismatches, and trade-count anomalies default to `flag` rather than removal or nulling. Why it inflates performance: the model can still train on rows already identified as suspicious. [SR 11-7, NIST AI RMF]

- High | The anomaly surface is not crypto-complete. It checks OHLC inconsistency, duplicate and retrograde timestamps, spikes, nonpositive volume, quote-volume mismatch, and trade-count anomalies, but not wash trading, spoofing, self-trade bursts, or venue migration. Why it matters: features built from `trades`, `quote_volume`, and taker-flow can learn exchange-specific abuse patterns instead of portable signal. [ESMA Art. 17, NIST AI RMF]

- Medium | `build_example_universe_config()` in `example_utils.py` fabricates `status="TRADING"`, fixed `listing_start="2020-01-01T00:00:00Z"`, and synthetic liquidity. Why it invalidates conclusions if reused outside demos: delistings, late listings, and historical ineligibility disappear, so survivorship bias is reintroduced. [SR 11-7]

### Feature Engineering

- High | `core/context.py` defaults context missingness to `zero_fill`, uses backward as-of joins, and finalizes unknowns with `fillna(0.0)`. Why it inflates performance: unavailable, stale, and economically neutral states collapse to the same number, so the model can learn API gaps, stale context, or venue outages as alpha. [NIST AI RMF, SR 11-7]

- High | The shipped feature surface is still mostly endogenous transforms of one venue's own tape and context: lags, rolling windows, squeeze thresholds, regime counts, indicator interactions, and label settings. Why it is fragile: this is the exact feature family that can backfit a Binance-specific microstructure regime and disappear when liquidity mix or derivatives positioning changes. [SR 11-7]

- Medium | Stationarity screens are being over-credited. Fractional differencing and ADF-style checks reduce trend contamination, but a stationary feature can still be regime-fragile, path-dependent, or microstructure-specific. Why it matters: stationarity is not transportability. [NIST AI RMF]

### AutoML Process

- High | `core/automl.py` is searching economic specifications, not just hyperparameters. It varies lags, frac-diff order, rolling windows, label barriers, holding periods, regime count, validation gap, model family, and model params. Why it inflates performance: many distinct trading theses are being tested on the same symbol and timeframe history; DSR, PBO, SPA, and post-selection inference reduce naive multiple-testing error but cannot recover causal identification after broad specification search. [SR 11-7]

- High | The lookahead guard is mode-dependent. In `core/pipeline.py`, it auto-enables for custom builders, AutoML, or `trade_ready`, but a plain research pipeline using built-in features can still skip automatic prefix-vs-baseline replay. Why it matters: there is a gap between protected example paths and generic research runs, so leakage regressions are easier to miss. [SR 11-7, NIST AI RMF]

- Medium | Robustness checks are mostly on-manifold. Local perturbations in `core/automl.py` vary nearby config values, not dropped bars, timestamp jitter, fee shocks, stale L2, throttling, or revised funding prints. Why it gives false robustness: the system can look stable to tuning noise while remaining brittle to operational noise. [NIST AI RMF]

- Medium | Lenient overlap policies remain available. PBO and post-selection inference support `pairwise_overlap` and `zero_fill_debug`. Why it matters: if an operator loosens the hardened defaults, candidate return paths become easier to compare than the data actually justify. [SR 11-7]

### Backtesting Methodology

- High | `core/backtest.py` supports `execution_price_policy` and `valuation_price_policy` values `ffill` and `ffill_with_limit`. Why it inflates performance: stale prices can survive into fills or PnL instead of forcing rejection during outages or stale-mark windows. [ESMA Art. 17, Binance]

- High | Stationary-bootstrap significance is enabled by default and accepts `min_observations=8`. Why it gives incorrect confidence: for sparse strategies, the effective sample size is trade count, not bar count, so confidence intervals can look tighter than the number of independent bets warrants. [SR 11-7]

- High | Kelly sizing amplifies every upstream misspecification. If calibration, slippage, funding, or fill probability are optimistic, the error is not additive; it compounds through leverage and notional sizing. Why it matters: small research bias becomes nonlinear drawdown and ruin bias. [SR 11-7]

- Medium | `requirements.txt` ships `vectorbt` but not NautilusTrader, while the fully gated trade-ready path still expects a real Nautilus backend. Why it matters: the default retail install is not the execution environment being certified. [ESMA Art. 17]

### Evaluation Metrics

- High | DSR, PBO, SPA, White reality check, and bootstrap intervals can be mathematically correct about the wrong simulator. Why it matters: if the return path was generated by a bar surrogate, stale-price fill policy, or zero-filled funding, the inference is internally precise and externally wrong. [SR 11-7]

- Medium | Sharpe and Calmar remain fragile to crypto data breaks. When missing candles survive as warnings and suspicious rows survive as flags, volatility, drawdown depth, and annualization are biased. Why it matters: a clean metric can be a timestamp-handling artifact rather than a real edge. [NIST AI RMF]

### Out-of-Sample & Robustness

- High | True OOS evidence is not the default retail experience. The repo supports locked holdouts, CPCV, purging, embargo, and post-selection inference, but the common fallback path explicitly disables the strongest subset. Why it inflates performance: users can confuse a convenient smoke run with deployable evidence. [SR 11-7, ESMA Art. 17]

- High | Replication breadth is still thin relative to crypto regime diversity. The runnable trade-ready example keeps a small trial budget and limited alternate windows to stay feasible on consumer hardware. Why it matters: a pass under that budget mainly shows the stack did not reject the candidate, not that the edge survived multiple independent market narratives. [SR 11-7]

- Medium | Cross-venue portability is not part of the default research path. Many examples remain Binance-only for both training and context. Why it matters: venue-specific distortions can be learned as alpha and then disappear when liquidity fragments or migrates. [NIST AI RMF, ESMA Art. 17]

### Deployment Realism

- High | Drift handling exists as an orchestration hook, not as an autonomous live defense. `run_drift_retraining_cycle()` still needs an external caller and a scheduled window. Why it matters: the design looks adaptive, but operationally it can remain manual and delayed exactly when regime change is fastest. [SR 11-7, NIST AI RMF]

- High | Consumer-hardware latency is measured more than priced. Inference latency and queue backlog can be reported, but default thresholds are non-binding and the default simulator lacks event-driven acknowledgment latency. Why it matters: a retail machine can clear research gates on edges that vanish before the order reaches market. [ESMA Art. 17]

- Medium | Black-swan handling is still scenario-based rather than adversarial. Named scenarios such as downtime, stale marks, and halts are useful, but they do not cover correlated venue failure, liquidation cascades, funding spikes, or synchronized API and routing degradation. Why it matters: the system can appear stress-tested while still being blind to the failure modes that matter most in crypto. [ESMA Art. 17, NIST AI RMF]

### Failure Modes That Can Look Profitable But Fail In Production

- Missing funding prints are treated as zero carry, so futures strategies look cleaner than they are and size too aggressively.

- Bar volume is treated as executable liquidity, so aggressive entries appear fillable even when a retail trader would lose queue priority or sweep the book.

- A research-only fallback run looks almost trade-ready, but the strongest OOS and multiple-testing controls were switched off.

- A model passes DSR, PBO, SPA, or bootstrap significance, but those tests were run on a misspecified fill and cost process.

- Context outages or stale joins become zeros, and the model learns operational failure states as alpha.

- Synthetic universe snapshots make the requested symbols look historically tradable and liquid, masking survivorship and universe-selection bias.

- A signal survives local hyperparameter perturbations but breaks under one missing-bar burst, one funding API gap, or one stale L2 window.
