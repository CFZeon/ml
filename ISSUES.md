# ISSUES

## Adversarial Audit Findings

Scope: repository state as of 2026-04-28.

Method:

- Code review of `core/data.py`, `core/data_quality.py`, `core/context.py`, `core/pipeline.py`, `core/automl.py`, `core/backtest.py`, `core/monitoring.py`, `example_trade_ready_automl.py`, and `example_utils.py`.
- External baseline from [Binance klines](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints#klinecandlestick-data), [Binance filters](https://developers.binance.com/docs/binance-spot-api-docs/filters), [Binance futures funding history](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History), and [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework).
- Standard applied: break the pipeline intellectually, not improve it. The question is whether the reported edge would survive live trading by a retail user on consumer hardware.

The repo contains serious controls that most hobby trading stacks lack. The problem is that the retail-feasible path still strips out or weakens the controls that matter most, while the default execution model remains a research surrogate rather than a realistic market simulator.

### Highest-Impact Flaws

- Critical | `example_trade_ready_automl.py` reduced-power fallback explicitly disables `locked_holdout_enabled`, `selection_policy`, `deflated_sharpe`, `pbo`, and `post_selection`. Impact: the path most likely to run on consumer hardware is not a lighter certification path; it is a materially weaker experiment that removes the main defenses against search-noise selection.

- Critical | `core/backtest.py` still defaults to a surrogate execution path and explicitly reports `bar_surrogate_only`, `no_queue_position_model`, `no_event_driven_ack_latency`, and `no_order_book_matching_engine`. Impact: any edge that depends on queue priority, passive fills, spread capture, or intrabar liquidity can be entirely simulated PnL.

- Critical | `core/pipeline.py` defaults futures `funding_missing_policy` to `zero_fill` outside strict trade-ready mode, and `core/backtest.py` / `core/models.py` reindex funding with `.fillna(0.0)`. Impact: missing funding becomes free carry, which mechanically inflates futures performance and Kelly sizing.

- High | `core/monitoring.py` research defaults leave data lag at `None`, L2 freshness unset, slippage / inference / queue thresholds at `inf`, and fallback-assumption tolerance at `1.0`. Impact: the system can report degradation without invalidating the model or stopping a live workflow.

### Data Integrity & Preprocessing

- High | `core/data.py` defaults `gap_policy="warn"`. Impact: in a 24/7 market, missing candles caused by outages, halts, or ingestion gaps remain in-sample as warnings rather than hard failures, which biases volatility, barrier timing, and annualization.

- High | `core/data_quality.py` defaults `block_on_quarantine=False`; return spikes, range spikes, quote-volume inconsistencies, and trade-count anomalies are only `flag`ged by default. Impact: the pipeline can knowingly train on rows it already considers suspect.

- High | The anomaly checks are not crypto-complete. They cover OHLC inconsistency, duplicate or retrograde timestamps, spikes, nonpositive volume, quote-volume mismatch, and trade-count anomalies, but not wash trading, spoofing, self-trade bursts, or cross-venue dislocations. Impact: tape-derived features can learn exchange-specific manipulation rather than portable signal.

- Medium | `example_utils.py` fabricates universe snapshots with `status="TRADING"` and fixed `listing_start="2020-01-01T00:00:00Z"`. Impact: if demo helpers leak into research, survivorship bias and historical eligibility bias re-enter immediately.

- Medium | Binance documents that kline `startTime` and `endTime` are always interpreted in UTC even when `timeZone` is supplied. Impact: any local-time slicing or naive timestamp join can silently move data across train/test boundaries.

### Feature Engineering

- High | `core/context.py` defaults missing context to `zero_fill`, uses backward as-of joins, and finalizes many context features with `fillna(0.0)`. Impact: unavailable, stale, and economically neutral states collapse to the same value, so the model can learn data outages as alpha.

- High | The shipped feature surface is still dominated by endogenous transforms of Binance tape and internal context: lags, rolling windows, indicator interactions, regime counts, and funding derivatives. Impact: this is exactly the feature family that can fit one venue's microstructure and then disappear when participation, fee mix, or leverage conditions change.

- Medium | The lookahead guard is broader than before, but plain research can still continue after a failure when the guard is advisory rather than blocking. Impact: leakage can be detected without becoming a hard stop outside stricter modes.

- Medium | Stationarity controls are being asked to do more than they can. Fractional differencing and ADF-style screening reduce trend contamination, but they do not establish causal relevance or regime transportability. Impact: a stationary feature can still be pure correlation.

### AutoML Process

- High | `core/automl.py` is not just tuning models; it is searching over trading theses. The default space varies lags, frac-diff order, rolling window, squeeze threshold, label barriers, holding period, regime count, validation fraction, model family, and model hyperparameters. Impact: repeated trials are testing multiple economic stories on the same history, not merely selecting a classifier.

- High | Time-aware gaps and holdout planning exist, but the retail fallback path removes the strongest post-selection controls precisely when the user lacks the infrastructure for strict certification. Impact: having rigorous machinery in the repo is not enough if the accessible path bypasses it.

- Medium | `core/automl.py` local robustness checks are near-manifold only. They perturb neighboring config values, not missing-bar bursts, timestamp jitter, stale L2, fee shocks, funding revisions, or API throttling. Impact: the study can look stable to tuning noise while remaining brittle to operational noise.

- Medium | Thesis-space search is frozen only in the strict trade-ready profile. Once the workflow downgrades to reduced-power research, that protection disappears. Impact: consumer-hardware feasibility and methodological rigor are directly in conflict in the shipped workflow.

### Backtesting Methodology

- Critical | The common execution path has no event-driven acknowledgments, no queue model, and no order book matching engine. Impact: fill probability, partial fills, spread crossing, and rejection behavior are approximated at the bar level rather than generated by a market process.

- High | `core/backtest.py` enables stationary-bootstrap significance by default with `min_observations=8`. Impact: for sparse strategies the independent sample is closer to effective bet count than bar count, so confidence intervals can look precise on too few true bets.

- High | Kelly sizing amplifies every upstream error. If calibration, fill quality, slippage, or funding is optimistic, the error compounds through notional allocation instead of remaining additive. Impact: a small research bias becomes a large capital-allocation error.

- Medium | Binance filters are modeled, but passing tick-size and min-notional checks does not make the simulator execution-realistic. Impact: the system can be exchange-valid and still be microstructure-invalid.

### Evaluation Metrics

- High | Deflated Sharpe, PBO, post-selection inference, and bootstrap intervals are only as honest as the simulator that generated the return path. Impact: mathematically careful inference on a bar-surrogate path is still false precision.

- High | The pipeline still creates a strong temptation to read Sharpe as the decision variable. In crypto, Sharpe is especially fragile to missing candles, stale pricing, clustered volatility, and short samples. Impact: a strong Sharpe can be a timestamp-handling artifact rather than a durable edge.

- Medium | I do not see default evidence that performance stability is being forced across structurally different market regimes rather than only across contiguous walk-forward slices. Impact: the system can pass OOS while still being concentrated in one market narrative.

### Out-of-Sample & Robustness

- High | The repo supports CPCV, purging, embargo, locked holdouts, and post-selection inference, but the practical retail path is still the one most likely to run without the full stack. Impact: users can confuse "it ran locally" with "it survived adversarial OOS evidence."

- High | The default research surface remains effectively single-venue Binance spot/futures, while many features are venue-conditioned. Impact: OOS success can reflect venue-specific order flow and inventory behavior rather than broader crypto price formation.

- Medium | Robustness to small numerical perturbations is not the same as robustness to small operational perturbations. Impact: a strategy can survive neighboring hyperparameters and still fail from one delayed bar, one stale context join, or one funding API gap.

### Deployment Realism

- Critical | The repo is closer to a strong research harness than to a live system a retail user should trust unattended. The evidence-gated path expects a real Nautilus backend; the consumer-hardware fallback degrades to research-only surrogate execution. Impact: the operationally accessible path is not the operationally trusted path.

- High | Monitoring and retraining are present, but in research mode they are largely descriptive rather than fail-closed. Impact: model decay, data staleness, and slippage drift can be observed without automatically invalidating the strategy.

- High | Regime awareness exists, but regime labeling is not a defense against black-swan transitions. Impact: historical regime segmentation does not protect against exchange outages, forced deleveraging, or liquidity evaporation.

- Medium | Consumer hardware is itself a hidden assumption in the trading thesis. If the signal half-life is short, local inference latency, network jitter, and API pacing limits become part of the edge. Impact: the strategy may only work in the backtest environment that ignores those bottlenecks.

### Failure Modes That Can Look Profitable But Fail In Production

- Missing funding observations are zero-filled, so perpetual futures appear to carry less financing drag than they will live.

- Bar volume is treated as executable liquidity, so entries that need queue priority or intrabar book depth look fillable when they are not.

- A run reports sophisticated overfitting diagnostics, but the user actually executed the downgraded path with holdout and post-selection controls disabled.

- Context outages and stale joins collapse to zeros, so operational failure states become predictive features.

- A Sharpe-significant result survives bootstrap inference, but the underlying return path came from a surrogate fill model, so the inference is about the wrong experiment.

- Demo universe snapshots make symbols look historically tradable and liquid even when historical eligibility differed.

- Kelly sizing turns a small calibration error into an outsized drawdown because position size is learned off biased probabilities.

- Walk-forward validation passes on one venue, but the strategy fails when Binance-specific microstructure or retail routing conditions change.
